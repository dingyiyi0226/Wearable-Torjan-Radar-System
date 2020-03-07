import csv
import os
import random
import sys
import threading
import time
import warnings

import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as sg
import serial
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button

import ADF4158
from view import PPIView, SigView


class FMCWRadar:
    """FMCW Radar model for each freqency"""

    def __init__(self, freq, BW, tm):

        ## SIGNAL IDENTITY

        self._freq = freq        ## the operation frequency
        self._slope = BW/tm      ## the slope of transmitting signal (Hz/s)
        self._BW = BW
        self._tm = tm
        self._fm = 1/tm
        self._distanceOffset = 1*2.24**0.5

        ## SIGNAL GENERATOR MODULE

        self._module = ADF4158.ADF4158()

        ## DATA

        self.info = {}          ## info of each direction: { angle: [(range, velo)] }
        self.direction = 90     ## operating direction , at 90 degree by default

        ## SIGNAL PROCESSING

        self._signalLength = 0   ## length of received data. len(timeSig) = 2*len(freqSig)
        self._resetFlag = False  ## reset signal flag
        self._signal = []        ## received data (temp signal)

        self._samplingTime = 0.  ## in mircosecond
        self._peakFreqsIdx = []  ## peak freq index in fftSig
        self._objectFreqs  = []  ## [(f1,f2), (f3,f4), ... ] two freqs cause by an object
                                 ## the tuple contain only one freq iff the object is stationary

        self.backgroundSig = {}
        self.realTimeSig = {'timeSig':[], 'timeAxis':[],'freqSig':[],'freqAxis':[], 'avgFreqSig':[]}


    ## SIGNAL PROCESSING FUNCTIONS

    def resetSignal(self):
        self._signal = []

    def readSignal(self, signal):
        # print(signal)
        # self._signal.extend([i/1024 for i in signal])  # read from arduino
        self._signal.extend(signal)  # read from file

    def endReadSignal(self, time):
        """update some variable at the end of the signal and start signal processing"""

        if not self._signal: return
        self._signalLength = len(self._signal)
        self._samplingTime = time * 1e-6

        self.realTimeSig['timeAxis'] = [i*self._samplingTime/self._signalLength for i in range(self._signalLength)]
        self.realTimeSig['freqAxis'] = [i/self._samplingTime for i in range(self._signalLength//2)]
        self.realTimeSig['timeSig'] = self._signal[:]

        self._signalProcessing()

    def setBackgroundSig(self):
        self.backgroundSig = self.realTimeSig

    def _signalProcessing(self):
        self._fft()
        if not self._findFreqPair():
            print('no peak frequency')
            return
        self._calculateInfo()

    def _fft(self):
        """perform fft on `_signal`"""

        PEAK_HEIGHT = 5e-3      ## amplitude of peak frequency must exceed PEAK_HEIGHT
        PEAK_PROMINENCE = 1e-4  ## prominence of peak frequency must exceed PEAK_PROMINENCE
        fftSignal_o = abs(np.fft.fft(self._signal))
        self.realTimeSig['freqSig'] = [i*2/self._signalLength for i in fftSignal_o[:self._signalLength//2]]
        self._avgFFTSig()

        self._peakFreqsIdx, _ = sg.find_peaks(self.realTimeSig['avgFreqSig'], height=PEAK_HEIGHT, prominence=PEAK_PROMINENCE)
        # print(self._peakFreqsIdx)

    def _avgFFTSig(self):
        """ averaging the fft signal """

        AVGTICK = 3   ## the number of ticks on frequency axis
        minFreqdiff = 1/self._samplingTime
        winLength = int(self._fm*self._samplingTime)
        window = np.ones(winLength)       ## window for averaging the signal
        window = window/window.sum()

        self.realTimeSig['avgFreqSig'] = sg.oaconvolve(self.realTimeSig['freqSig'], window, mode='same')
        # self.realTimeSig['avgFreqSig'] = self.realTimeSig['freqSig']

    def _findFreqPair(self) -> bool:
        """split the freqs in `_peakFreq` with same intensity into pairs
        
        Returns:
            return false if no peak frequency is found
        """

        PEAK_DIFF = 1e-4  ## we assume two peak belong to same object if and only if the amplitude
                          ## difference between two peaks < PEAK_DIFF
        sortedFreqIndex = sorted(self._peakFreqsIdx, key=lambda k: self.realTimeSig['avgFreqSig'][k], reverse=True)
        if not sortedFreqIndex: return False

        freqAmplitude = 0
        tmpFreqIndex = 0
        # print('sortedFreqIndex', sortedFreqIndex)

        for freqIndex in sortedFreqIndex:
            # print(freqIndex, self.realTimeSig['freqSig'][freqIndex])
            if freqAmplitude == 0:
                freqAmplitude = self.realTimeSig['avgFreqSig'][freqIndex]
                tmpFreqIndex = freqIndex
                continue
            if (freqAmplitude - self.realTimeSig['avgFreqSig'][freqIndex]) < PEAK_DIFF:
                self._objectFreqs.append((int(tmpFreqIndex/self._samplingTime), int(freqIndex/self._samplingTime)))
                freqAmplitude = 0.
            else:
                # print('ff',int(tmpFreqIndex/self._samplingTime))
                self._objectFreqs.append((int(tmpFreqIndex/self._samplingTime), ))
                freqAmplitude = self.realTimeSig['avgFreqSig'][freqIndex]
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
                infoList.append((objRange1, objVelo1))  # have two solutions
                infoList.append((objRange2, objVelo2))
                # print( (objRange1, objVelo1))
                # print( (objRange2, objVelo2))

        # print(infoList)

        self.info[self.direction] = infoList
        self._objectFreqs = []

    def _freq2Range(self, freq):
        return freq / self._slope * 3e8 / 2

    def _freq2Velo(self, freq):
        return freq / self._freq * 3e8 / 2

    def _beatFreqLim(self):
        # TODO:

        pass

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

    with open('rawdata/0225nolinefm05/'+filename+'.csv') as file:
        datas = csv.reader(file)
        for ind, data in enumerate(datas):
            if ind==0: continue
            elif ind==1:
                simSampFreq = 1/float(data[3])
            else:
                simSignal.append(float(data[1]))

    samSig = []
    i=1
    j=random.randrange(len(simSignal))
    while True:
        readEvent.wait()
        if i % int(samTime*samFreq) != 0:
            samSig.append(simSignal[(int(j+i*simSampFreq/samFreq) % len(simSignal))])

        else:
            radar.resetSignal()
            j = random.randrange(len(simSignal))
            # print(samSig)
            radar.readSignal(signal=samSig)

            radar.endReadSignal(time=samTime*1e6 )
            samSig = []
            time.sleep(0.001)
        i+=1

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

    ## Port Connecting
    # ser = serial.Serial(port())
    # print('Successfully open port: ', ser)

    ## initialize the model
    radar = FMCWRadar(freq=58e8 , BW=99.9969e6, tm=2.048e-3)  ## operating at 5.8GHz, slope = 100MHz/1ms

    ## start reading in another thread but block by readEvent
    readEvent  = threading.Event()

    ## practical version
    # readThread = threading.Thread(target=read, args=[ser, radar, readEvent], daemon=True)

    ## simulation version
    readThread = threading.Thread(target=readSimSignal, daemon=True,
        kwargs={'filename':'3502', 'samFreq':1e4, 'samTime':2.4e-2, 'radar':radar, 'readEvent':readEvent})
    
    readThread.start()
    print('Reading Signal')
    readEvent.set()

    views = []

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

            elif s.startswith('stop'):
                if not readEvent.is_set():
                    print('not been reading signal')
                else:
                    readEvent.clear()
                    print('Stop Reading Signal')


            elif s.startswith('sig'):
                # view = SigView(maxFreq=2e3, maxTime=0.2)  # for arduino
                view = SigView(maxFreq=10e3, maxTime=1e-2)  # for simulation
                views.append(view)
                animation = FuncAnimation(view.fig, view.update,
                    init_func=view.init, interval=20, blit=False,
                    fargs=(radar.realTimeSig,))
                view.figShow()

            elif s.startswith('ppi'):
                view = PPIView(maxR=5)
                views.append(view)

                animation = FuncAnimation(view.fig, view.update,
                    frames=100, init_func=view.init, interval=20, blit=True,
                    fargs=(radar.direction, radar.info[radar.direction]))

                view.figShow()

            elif s.startswith('close'):
                for view in views:
                    plt.close(view.fig)
                views.clear()

            elif s.startswith('setfreq'):
                # TODO: ADF4158 module
                pass

            elif s.startswith('info'):
                print(radar.info)
            elif s.startswith('test'):
                print('hello world')
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
