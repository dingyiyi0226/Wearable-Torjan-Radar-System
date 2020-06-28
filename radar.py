import csv
import os
from abc import ABC, abstractmethod

import numpy as np
import scipy.signal as sg

from adc import ADC, ADCConnector
from config import ADF_HIGH_PINS, ADF_LOW_PINS, DIR_PINS
from module import ADF4158
from module.A4988 import A4988
from utils import port


def loadArduinoADC() -> list:
    """ Check ADC Exists. """
    pairs = {}

    for adc in [ ADCConnector(p) for p in port() ]:
        if adc.name == "5.8":
            pairs["highFreqRadar"] = adc
        if adc.name == "915":
            pairs["lowFreqRadar"] = adc

    return pairs


class FMCWRadar:
    """ FMCW Radar model for each freqency """

    def __init__(self, adc: ADC):

        ## SIGNAL IDENTITY
        
        self.setModuleProperty(0, 0, 0, 0)

        ## MODULES

        self._adc = adc
        self._adc.setProcessor(self)
        if self._adc.name == "5.8":
            self._signalModule = ADF4158.set5800Default(pins=ADF_HIGH_PINS)
            self.setModuleProperty(5.8e9, 1e8, 8e-3, 1 * 2.24)
        if self._adc.name == "915":
            self._signalModule = ADF4158.set915Default(pins=ADF_LOW_PINS)
            self.setModuleProperty(915e6, 1.5e7, 3.3e-5, 1 * 2.24)

        ## SIGNAL PROCESSING

        self._signalProcessor = None

        self._signalLength = 0   ## length of received data. len(timeSig) = 2*len(freqSig)

        self._samplingTime = 0.  ## in mircosecond
        self._peakFreqsIdx = []  ## peak freq index in fftSig
        self._objectFreqs  = []  ## [(f1,f2), (f3,f4), ... ] two freqs cause by an object
                                 ## the tuple contain only one freq iff the object is stationary

        self.backgroundSig = None
        self.realTimeSig = {
            'timeSig':      np.zeros(1),
            'timeAxis':     np.zeros(1),
            'freqSig':      np.zeros(1),
            'freqAxis':     np.zeros(1),
            'processedSig': np.zeros(1)
        }

        ## PUSH CONTAINER

        self._container = None

    ## PUBLIC FUNCTION

    def start(self):
        self._adc.start()

    def stop(self):
        self._adc.stop()

    def close(self):
        self._adc.disconnect()

    @property
    def name(self):
        return self._adc.name

    # TODO
    def getConfig(self) -> dict:
        # return self._signalModule.getConfig()
        return {}

    def save(self, fname):
        if os.path.exists(fname):
            print("File {} exists. Overwrite it.".format(fname))
        
        with open(os.path.join(fname), 'w') as file:
            writer = csv.writer(file)
            writer.writerow(['X', 'Sig', '', 'Increment'])
            writer.writerow(['', '', '', str(self.realTimeSig['timeAxis'][1])])
            
            for ind, i in enumerate(self.realTimeSig['timeSig']):
                writer.writerow([ind, i])

    ## RADAR ADJUSTMENT

    def setModuleProperty(self, freq, BW, tm, distanceOffset):
        self._freq  = freq                              ## the operation frequency
        self._slope = np.nan if tm == 0 else BW / tm    ## the slope of transmitting signal (Hz/s)
        self._BW    = BW
        self._tm    = tm
        self._fm    = np.nan if tm == 0 else 1 / tm
        self._distanceOffset = 1 * 2.24 ** 0.5

        # TODO
        # self._signalModule.setRampAttribute()

    ## SIGNAL ACCESSING FUCNTION

    def loadData(self, signal: list, time: float):
        """
        Update signal and some variable at the end of the signal and start signal processing
        
        Parameters
        ----------
        signal : list
            the signal point in list

        time : int
            time record in unit (s)
        """

        self.realTimeSig['timeSig'] = np.array(signal)
        self._signalLength = self.realTimeSig['timeSig'].shape[0]
        self._samplingTime = time

        self.realTimeSig['timeAxis'] = np.arange(self._signalLength) * self._samplingTime / self._signalLength
        self.realTimeSig['freqAxis'] = np.arange(self._signalLength // 2) / self._samplingTime

        self._container.clear()

        info = self._signalProcessing()
        if info is not None:
            self._container.extend(info)

    def clearBgSig(self):
        self.backgroundSig = None

    def setBgSig(self, overwrite):
        if overwrite:
            self.backgroundSig = self.realTimeSig.copy()
        else:
            currentSig = self.realTimeSig.copy()
            for key, sig in currentSig.items():
                self.backgroundSig[key] = 0.75 * self.backgroundSig[key] + 0.25 * sig

    ## PRIVATE FUNCTION

    ## ATTRIBUTE

    def __str__(self):
        return ("(FMCWRadar, {})".format(self.name))

    ## SIGNAL PROCESSING FUNCTIONS

    def _signalProcessing(self):

        self._fft()
        self._rmBgSig()
        self._avgFreqSig()
        
        if not self._findPeaks(height=1e-4, prominence=1e-4):
            return None

        self._findFreqPair(peakDiff=1e-3)
        return self._calculateInfo()

    def _fft(self):
        """ Perform FFT on realtime signal """

        fftSignal = np.abs(np.fft.fft(self.realTimeSig['timeSig'])) / self._signalLength
        self.realTimeSig['freqSig'] = fftSignal[:self._signalLength//2]  ## only save the positive freqs.
        self.realTimeSig['processedSig'] = fftSignal[:self._signalLength//2].copy()

    def _rmBgSig(self):
        """ Remove background and set min value to 0 """

        if self.backgroundSig is not None:
            self.realTimeSig['processedSig'] -= self.backgroundSig['freqSig']
            self.realTimeSig['processedSig'] = self.realTimeSig['processedSig'].clip(0)

    def _avgFreqSig(self):
        """ Averaging the FFT signal """

        BW = self._fm * 2                       ## bandwidth of the window
        winLength = int(BW*self._samplingTime)  ## length = BW / df = BW*T
        # window = np.ones(winLength)           ## window for averaging the signal
        window = sg.blackman(winLength)         ## window for averaging the signal
        window = window / window.sum()

        self.realTimeSig['processedSig'] = sg.convolve(self.realTimeSig['processedSig'], window, mode='same')

    def _findPeaks(self, height, prominence) -> bool:
        """
        Find peaks in processedSig
        
        Parameters
        ----------
        height : float
            min amplitude of peak frequencies
        prominence : float
            min prominence of peak frequencies

        Return
        ------
        status : bool
            True if peak frequency is found
        """

        self._peakFreqsIdx, _ = sg.find_peaks(self.realTimeSig['processedSig'], height=height, prominence=prominence)
 
        return len(self._peakFreqsIdx) != 0

    def _findFreqPair(self, peakDiff):
        """
        Split the freqs in `_peakFreq` with same intensity into pairs

        Parameters
        ----------
        peakDiff : float
            we assume two peaks belong to same object if and only if the amplitude
            difference between two peaks < peakDiff
        """

        sortedFreqIndex = sorted(self._peakFreqsIdx, 
            key=lambda k: self.realTimeSig['processedSig'][k], reverse=True)

        freqAmplitude = 0
        tmpFreqIndex = 0
        
        for freqIndex in sortedFreqIndex:
            # print(freqIndex, self.realTimeSig['freqSig'][freqIndex])
            if freqAmplitude == 0:
                freqAmplitude = self.realTimeSig['processedSig'][freqIndex]
                tmpFreqIndex = freqIndex
                continue

            if (freqAmplitude - self.realTimeSig['processedSig'][freqIndex]) < peakDiff:
                self._objectFreqs.append((int(tmpFreqIndex/self._samplingTime), int(freqIndex/self._samplingTime)))
                freqAmplitude = 0.
            else:
                # print('ff',int(tmpFreqIndex/self._samplingTime))
                self._objectFreqs.append((int(tmpFreqIndex/self._samplingTime), ))
                freqAmplitude = self.realTimeSig['processedSig'][freqIndex]
                tmpFreqIndex = freqIndex
            # print(freqIndex)

        if freqAmplitude != 0:
            self._objectFreqs.append((int(tmpFreqIndex/self._samplingTime), ))

    def _calculateInfo(self):
        """ Calculate range and velocity of every object from `_objectFreqs` """

        objRange = 0.
        objVelo  = 0.
        infoList = []

        for tup in self._objectFreqs:
            if len(tup) == 1:
                fb = tup[0]
                objRange = self._freq2Range(fb)
                objVelo  = 0.

                objRange -= self._distanceOffset

                infoList.append((objRange, objVelo))  # have one solution only
            else:
                f1 =    (tup[0] + tup[1]) / 2
                f2 = abs(tup[0] - tup[1]) / 2

                objRange1 = self._freq2Range(f1)
                objVelo1  = self._freq2Velo(f2)

                objRange2 = self._freq2Range(f2)
                objVelo2  = self._freq2Velo(f1)

                objRange1 -= self._distanceOffset
                objRange2 -= self._distanceOffset

                # have two solutions
                # if velo > 25, omit it

                if objVelo1 < 25:
                    infoList.append((objRange1, objVelo1))
                if objVelo2 < 25:
                    infoList.append((objRange2, objVelo2))
                # print( (objRange1, objVelo1))
                # print( (objRange2, objVelo2))

        self._objectFreqs.clear()
        return infoList

    def _freq2Range(self, freq):
        return freq / self._slope * 3e8 / 2

    def _freq2Velo(self, freq):
        return freq / self._freq * 3e8 / 2

class LowFreqFMCWRadar(FMCWRadar):
    pass

    ## OVERRIDE: SIGNAL PROCESSING FUNCTIONS

    # def _signalProcessing(self):

    #     self._fft()
    #     self._rmBgSig()
    #     self._avgFreqSig()
        
    #     if not self._findPeaks(height=1e-4, prominence=1e-4):
    #         return None

    #     self._findFreqPair(peakDiff=1e-3)
    #     return self._calculateInfo()


class HighFreqFMCWRadar(FMCWRadar):
    pass

    ## OVERRIDE: SIGNAL PROCESSING FUNCTIONS

    # def _signalProcessing(self):

    #     self._fft()
    #     self._rmBgSig()
    #     self._avgFreqSig()
        
    #     if not self._findPeaks(height=3e-3, prominence=1e-4):
    #         return None

    #     self._findFreqPair(peakDiff=1e-3)
    #     return self._calculateInfo()


class Troy:
    
    def __init__(self):

        ## Data

        self.currentDir = 0
        self.lowData    = []
        self.highData   = []

        ## Modules

        self.rotateMotor = A4988(DIR_PINS)
        self.lowFreqRadar = None
        self.highFreqRadar = None

        adcs = loadArduinoADC()
        if "highFreqRadar" in adcs:
            self.highFreqRadar = HighFreqFMCWRadar(adcs["highFreqRadar"])
            self.highFreqRadar._container = self.highData
        if "lowFreqRadar" in adcs:
            self.lowFreqRadar = LowFreqFMCWRadar(adcs["lowFreqRadar"])
            self.lowFreqRadar._container = self.lowData

    ## Public Function

    ## ACTION FUNCTION

    def start(self):
        """ Start Method """
        for radar in self.availableChannels:
            print("> Load {} ADC. ".format(radar.name))
            radar.start()

    def stop(self):
        """ Pause Method """
        for radar in self.availableChannels:
            print("> Load {} ADC Paused!. ".format(radar.name))
            radar.stop()

    def close(self):
        """ Release all occupied pins and processes. """
        for radar in self.availableChannels:
            radar.close()

    # TODO
    def save(self, highFreqFname, lowFreqFname) -> bool:
        if self.highFreqRadar is not None:
            self.highFreqRadar.save(highFreqFname)

        if self.lowFreqRadar is not None:
            self.lowFreqRadar.save(lowFreqFname)

    def setDirection(self, direction: int):
        deltaDir = direction - self.currentDir
        self.rotateMotor.spin(abs(deltaDir), deltaDir > 0)
        self.currentDir = direction
        self.clearBgSignal()

    def resetDirection(self, direction=0):
        self.currentDir = direction

    def flush(self):
        for radar in self.availableChannels:
            radar._adc._serial.flush()

    ## ATTRIBUTE FUNCTION

    @property
    def availableChannels(self):
        tmp = []

        if self.highFreqRadar is not None:
            tmp.append(self.highFreqRadar)
        if self.lowFreqRadar is not None:
            tmp.append(self.lowFreqRadar)

        return tmp

    @property
    def objectInfo(self):
        return { self.currentDir: [*self.lowData, *self.highData] }

    def getInfo(self):
        print("========================================")
        print("| Current Directory: ", self.currentDir)
        print("========================================")
        print("| Current Config: ")
        print("| ", self.lowFreqRadar)
        print("| ", self.highFreqRadar)
        print("| Detected Object: ")
        print("| ", self.lowData)
        print("| ", self.highData)
        print("========================================")

    def tracking(self):
        """ Keep update the info to stream. """
        try:
            while True: pass
        except KeyboardInterrupt as e:
            print()

    ## SIGNAL PROCESSING RELATED FUCNTION

    def clearBgSignal(self):
        for radar in self.availableChannels:
            radar.clearBgSig()

    def setBgSignal(self, overwrite: bool):
        for radar in self.availableChannels:
            radar.setBgSig(overwrite)
