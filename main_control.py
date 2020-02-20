import csv
import os
import serial
import sys
import time
import threading

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks


class FMCWRadar:
    """FMCW Radar model for each freqency"""

    def __init__(self, freq, slope):

        ## SIGNAL IDENTITY

        self._freq  = freq       ## the operation frequency
        self._slope = slope      ## the slope of transmitting signal (Hz/s)

        ## DATA

        self._info = {}          ## info of each direction: { angle: [(range, velo)] }
        self._direction = 90     ## operating direction , at 90 degree by default

        ## SIGNAL PROCESSING

        self._signalLength = 0   ## length of received data , fft data
        self._resetFlag = False  ## reset signal flag
        self._signal       = []  ## received data
        self._fftSignal    = []  ## fft data
        self._timeAxis     = []  ## time axis of received data
        self._freqAxis     = []  ## freq axis of fft data
        self._samplingTime = 0.  ## in mircosecond
        self._peakFreqsIdx = []  ## peak freq index in fftSignal
        self._objectFreqs  = []  ## [(f1,f2), (f3,f4), ... ] two freqs cause by an object
                                 ## the tuple contain only one freq iff the object is stationary

        ## PLOTTING

        self._plotEvent = threading.Event()

    ## DATA FUNCTIONS

    @property
    def direction(self):
        return self._direction

    @property
    def info(self):
        return self._info

    def infoAtDirection(self, key):
        return self._info.get(key)

    def resetInfo(self):
        self._info = {}

    ## SIGNAL PROCESSING FUNCTIONS

    def resetSignal(self):
        self._resetFlag = True

    def readSignal(self, signal):
        # print(signal)

        if self._resetFlag:
            self._signal = []
            self._resetFlag = False

        self._signal.extend(signal)

    def endReadSignal(self, time):
        """update some variable at the end of the signal and start signal processing"""

        if not self._signal: return
        self._signalLength = len(self._signal)
        self._samplingTime = time * 1e-6
        self._timeAxis = [i*self._samplingTime/self._signalLength for i in range(self._signalLength)]
        self._freqAxis = [i/self._samplingTime for i in range(self._signalLength)]

        self._signalProcessing()

    def _signalProcessing(self):
        self._fft()
        if not self._findFreqPair():
            # print('no peak frequency')
            return
        self._calculateInfo()

    def _fft(self):
        """perform fft on `_signal`"""

        PEAK_HEIGHT = 1e-2  ## amplitude of peak frequency must exceed PEAK_HEIGHT
        fftSignal_o = abs(np.fft.fft(self._signal))
        self._fftSignal = [i*2/self._signalLength for i in fftSignal_o]

        self._peakFreqsIdx, _ = find_peaks(self._fftSignal[:self._signalLength//2], height=PEAK_HEIGHT)
        # print(self._peakFreqsIdx)
        self._plotEvent.set()

    def _findFreqPair(self) -> bool:
        """split the freqs in `_peakFreq` with same intensity into pairs
        
        Returns:
            return false if no peak frequency is found
        """

        PEAK_DIFF = 1e-3  ## we assume two peak belong to same object if and only if the amplitude
                          ## difference between two peaks < PEAK_DIFF
        sortedFreqIndex = sorted(self._peakFreqsIdx, key=lambda k: self._fftSignal[k], reverse=True)
        if not sortedFreqIndex: return False

        freqAmplitude = 0
        tmpFreqIndex = 0
        # print('sortedFreqIndex', sortedFreqIndex)
        for freqIndex in sortedFreqIndex:
            # print(freqIndex, self._fftSignal[freqIndex])
            if freqAmplitude == 0:
                freqAmplitude = self._fftSignal[freqIndex]
                tmpFreqIndex = freqIndex
                continue
            if (freqAmplitude - self._fftSignal[freqIndex]) < PEAK_DIFF:
                self._objectFreqs.append((int(tmpFreqIndex/self._samplingTime), int(freqIndex/self._samplingTime)))
                freqAmplitude = 0.
            else:
                self._objectFreqs.append((int(tmpFreqIndex/self._samplingTime), ))
                freqAmplitude = self._fftSignal[freqIndex]
                tmpFreqIndex = freqIndex
            # print(freqIndex)
        if freqAmplitude != 0:
            self._objectFreqs.append((int(tmpFreqIndex/self._samplingTime), ))
        return True

    def _calculateInfo(self):
        """calculate range and velocity of every object from `_objectFreqs`"""

        # print('calculateInfo', self._objectFreqs)
        objRange = 0.
        objVelo  = 0.
        infoList = []

        for tup in self._objectFreqs:
            if len(tup) == 1:
                fb = tup[0]
                objRange = fb / self._slope * 3e8 / 2
                objVelo  = 0.
            else:
                fb =    (tup[0] + tup[1]) / 2
                fd = abs(tup[0] - tup[1]) / 2
                objRange = fb / self._slope * 3e8 / 2
                objVelo  = fd / self._freq  * 3e8 / 2

            infoList += (objRange, objVelo)

            # if self._direction in self._info:
            #     self._info[self._direction] += (objRange, objVelo)
            # else:
            #     self._info[self._direction] = [(objRange, objVelo)]

        self._info[self._direction] = infoList
        self._objectFreqs = []

    ## PLOTTING FUNCTIONS

    @property
    def plotEvent(self):
        return self._plotEvent

    def plotSignal(self, DCBlock : bool=False, maxFreq=None) -> bool:
        """only plot signal with frequencies in (0, maxFreq)"""

        if not self._signal: return False
        ## convert maxFreq to corresponding index (maxIndex)
        maxIndex = self._signalLength//2 if maxFreq is None else int(maxFreq * self._samplingTime)

        if maxIndex > self._signalLength//2:
            print('maxFreq do not exceed ', int(self._signalLength//2 / self._samplingTime))
            self._plotEvent.clear()
            return False

        plt.clf()

        plt.subplot(211)
        plt.plot(self._timeAxis,self._signal)
        plt.xlabel('time (s)')
        plt.ticklabel_format(axis='x', style='sci', scilimits=(0,0), useMathText=True)

        plt.subplot(212)

        if DCBlock:
            plt.plot(self._freqAxis[1:maxIndex], self._fftSignal[1:maxIndex],'r')
        else:
            plt.plot(self._freqAxis[0:maxIndex], self._fftSignal[0:maxIndex],'r')

        plt.plot([self._freqAxis[i]  for i in self._peakFreqsIdx if i < maxIndex],
                 [self._fftSignal[i] for i in self._peakFreqsIdx if i < maxIndex], 'x')

        # plt.yscale('log')
        plt.xlabel('freq(Hz)')
        plt.ticklabel_format(axis='x', style='sci', scilimits=(0,0), useMathText=True)

        plt.subplots_adjust(hspace=0.4)
        plt.pause(0.001)
        self._plotEvent.clear()
        return True


def read(ser, radar, readEvent):
    """read signal at anytime in other thread"""

    while True:
        readEvent.wait()
        ## maybe have to reset buffer
        try:
            s = ser.readline().decode()
            if s.startswith('i'):
                radar.resetSignal()

            elif s.startswith('d'):
                # print('readSignal ',s[2:])
                try:
                    radar.readSignal(signal=[float(i) for i in s[2:].split()])
                except ValueError:
                    print('Value Error: ',s[2:])
                    continue

            elif s.startswith('e'):
                # print('endReadSignal ', s[2:])
                try:
                    radar.endReadSignal(time=float(s[2:]))
                except ValueError:
                    print('Value Error: ',s[2:])
                    continue

            else:
                print('Read: ', s)

        except UnicodeDecodeError:
            print('UnicodeDecodeError')
            continue

        time.sleep(0.001)

def readSimSignal(filename, samFreq, samTime, radar, readEvent):
    """without connecting to Arduino, read signal from data"""
    
    simSignal = []
    simSampFreq = 0

    with open('data/distance_raw_0118_early/'+filename+'.csv') as file:
        datas = csv.reader(file)
        for ind, data in enumerate(datas):
            if ind==0: continue
            elif ind==1:
                simSampFreq = 1/float(data[3])
            else:
                simSignal.append(float(data[1]))

    samSig = []
    i=1
    while True:
        readEvent.wait()
        if i % int(samTime*samFreq) != 0:
            samSig.append(simSignal[(int(i*simSampFreq/samFreq) % len(simSignal))])
            i+=1
        else:
            radar.readSignal(signal=samSig)

            radar.endReadSignal(time=samTime*1e6 )
            samSig = []
            time.sleep(0.001)

def port() -> str:
    """find the name of the port"""

    try:
        ## on mac
        if(sys.platform.startswith('darwin')):
            ports = os.listdir('/dev/')
            for i in ports:
                if i[0:-2] == 'tty.usbserial-14':
                    port = i
                    break;
            port = '/dev/' + port
        ## on rpi
        if(sys.platform.startswith('linux')):
            ports = os.listdir('/dev/')
            for i in ports:
                if i[0:-1] == 'ttyUSB':
                    port = i
                    break;
            port = '/dev/' + port

    except UnboundLocalError:
        sys.exit('Cannot open port')

    return port


def main():

    # ## Port Connecting
    # ser = serial.Serial(port())
    # print('Successfully open port: ', ser)

    ## initialize the model
    radar = FMCWRadar(freq=58e8 , slope=16e6/25e-6)  ## operating at 5.8GHz, slope = 16MHz/25us

    ## start reading in another thread but block by readEvent
    readEvent  = threading.Event()

    # ## practical version
    # readThread = threading.Thread(target=read, args=[ser, radar, readEvent], daemon=True)

    ## simulation version
    readThread = threading.Thread(target=readSimSignal, daemon=True,
        kwargs={'filename':'75', 'samFreq':1e6, 'samTime':5e-3, 'radar':radar, 'readEvent':readEvent})
    
    readThread.start()

    try:
        prompt = ''
        while True:
            s = input("commands: " + prompt).strip()

            if s == '': pass

            elif s.startswith('read'):
                if readEvent.is_set():
                    print('has been reading signal')
                else:
                    print('Reading Signal')
                    readEvent.set()

            elif s.startswith('stopread'):
                if not readEvent.is_set():
                    print('not been reading signal')
                else:
                    readEvent.clear()
                    print('Stop Reading Signal')

            elif s.startswith('draw'):
                if not readEvent.is_set():
                    print('readEvent has not set')
                else:
                    try:
                        plt.figure('Signal')
                        while True:
                            radar.plotEvent.wait()
                            # if not radar.plotSignal(DCBlock=True, maxFreq=100000):
                            if not radar.plotSignal(DCBlock=True):
                                plt.close('Signal')
                                print('no signal')
                                break
                            time.sleep(0.01)

                    except KeyboardInterrupt:
                        plt.close('Signal')
                        print('Quit drawing')

            elif s.startswith('currentdir'):
                print('current direction:', radar.direction)

            elif s.startswith('infoat'):
                try:
                    ss = input('direction: ')
                    info = radar.infoAtDirection(float(ss))
                    if info is None:
                        print('{} is not a valid direction'.format(ss))
                    else:
                        print('direction: {}, (range, velocity): {}'.format(ss, info))
                except ValueError:
                    print('{} is not a valid direction'.format(ss))

            elif s.startswith('info'):
                for ind,val in radar.info.items():
                    print('direction: {}, (range, velocity): {}'.format(ind, val))

            elif s.startswith('resetinfo'):
                radar.resetInfo()
                print('reset all direction data')

            elif s.startswith('q'):
                break
            else:
                print('Undefined Command')

    except KeyboardInterrupt:
        pass
    finally: print('Quit main')

    # ser.close()

if __name__ == '__main__':
    main()
