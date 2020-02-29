import csv
import os
import random
import serial
import sys
import time
import threading

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button
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

        self._signalLength = 0   ## length of received data. len(timeSig) = 2*len(freqSig)
        self._resetFlag = False  ## reset signal flag
        self._signal       = []  ## received data (temp signal)

        self._samplingTime = 0.  ## in mircosecond
        self._peakFreqsIdx = []  ## peak freq index in fftSig
        self._objectFreqs  = []  ## [(f1,f2), (f3,f4), ... ] two freqs cause by an object
                                 ## the tuple contain only one freq iff the object is stationary

        self.realTimeSig = {'timeSig':[], 'timeAxis':[],'freqSig':[],'freqAxis':[]}

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

        self.realTimeSig['timeAxis'] = [i*self._samplingTime/self._signalLength for i in range(self._signalLength)]
        self.realTimeSig['freqAxis'] = [i/self._samplingTime for i in range(self._signalLength//2)]
        self.realTimeSig['timeSig'] = self._signal[:]

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
        self.realTimeSig['freqSig'] = [i*2/self._signalLength for i in fftSignal_o[:self._signalLength//2]]

        self._peakFreqsIdx, _ = find_peaks(self.realTimeSig['freqSig'], height=PEAK_HEIGHT)
        # print(self._peakFreqsIdx)

    def _findFreqPair(self) -> bool:
        """split the freqs in `_peakFreq` with same intensity into pairs
        
        Returns:
            return false if no peak frequency is found
        """

        PEAK_DIFF = 1e-3  ## we assume two peak belong to same object if and only if the amplitude
                          ## difference between two peaks < PEAK_DIFF
        sortedFreqIndex = sorted(self._peakFreqsIdx, key=lambda k: self.realTimeSig['freqSig'][k], reverse=True)
        if not sortedFreqIndex: return False

        freqAmplitude = 0
        tmpFreqIndex = 0
        # print('sortedFreqIndex', sortedFreqIndex)
        for freqIndex in sortedFreqIndex:
            # print(freqIndex, self.realTimeSig['freqSig'][freqIndex])
            if freqAmplitude == 0:
                freqAmplitude =  self.realTimeSig['freqSig'][freqIndex]
                tmpFreqIndex = freqIndex
                continue
            if (freqAmplitude -  self.realTimeSig['freqSig'][freqIndex]) < PEAK_DIFF:
                self._objectFreqs.append((int(tmpFreqIndex/self._samplingTime), int(freqIndex/self._samplingTime)))
                freqAmplitude = 0.
            else:
                self._objectFreqs.append((int(tmpFreqIndex/self._samplingTime), ))
                freqAmplitude =  self.realTimeSig['freqSig'][freqIndex]
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


class RadarView:


    def __init__(self, t):

        if t==1:
            self.fig, self.ax = plt.subplots(2,1, num='fig')
            # self.fig2, self.ax2 = plt.subplots(3,2, num='fig2')

            self.ax[0].set_xlim(0, 1e-2)
            self.ax[0].set_ylim(-0.12, 0.12)
            
            self.ax[1].set_xlim(0, 25e3)
            self.ax[1].set_ylim(0, 0.05)

            self.fig.subplots_adjust(left=0.3)

            self.timeLine, = self.ax[0].plot([], [])
            self.freqLine, = self.ax[1].plot([], [], 'r')

            self.buttonAx = plt.axes([0.05, 0.05, 0.15, 0.1])
            self.button = Button(self.buttonAx, 'Test', color='0.8', hovercolor='0.6')
            self.button.on_clicked(self.onClick)

        else:
            self.fig = plt.figure('fig2')
            self.ax = self.fig.add_subplot(111, polar=True)

            self.ax.set_rmax(2)
            self.ax.set_rticks(np.arange(0, 2, 0.5))

            self.fig.subplots_adjust(left=0.3)

            self.ppiData, = self.ax.plot([], [], 'ro')

            self.buttonAx = plt.axes([0.05, 0.05, 0.15, 0.1])
            self.button = Button(self.buttonAx, 'Testt', color='0.8', hovercolor='0.6')
            self.button.on_clicked(self.onClick)


    def figShow(self):
        plt.pause(1)

    def initPPI(self):
        return self.ppiData,

    def updatePPI(self, frame, ppiTheta, ppiR):

        ## dynamic set rmax ??

        self.ppiData.set_data([i+frame/100*2*np.pi for i in ppiTheta], [i+frame/100 for i in ppiR])
        # self.ppiData.set_data(ppiTheta, ppiR)

        return self.ppiData,


    def initRealTimeSig(self):
        # print('initRealTimeSig')

        return self.timeLine, self.freqLine,

    def updateRealTimeSig(self, frame, sigDict):
        # print('frame', frame)

        ## dynamic set ax.x_lim here ??

        self.timeLine.set_data(sigDict['timeAxis'], sigDict['timeSig'])
        self.freqLine.set_data(sigDict['freqAxis'], sigDict['freqSig'])

        return self.timeLine, self.freqLine,

    def onClick(self, event):
        print('click')

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

    with open('rawdata/0225/'+filename+'.csv') as file:
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

    # ## Port Connecting
    # ser = serial.Serial(port())
    # print('Successfully open port: ', ser)

    ## initialize the model
    radar = FMCWRadar(freq=58e8 , slope=100e6/1e-3)  ## operating at 5.8GHz, slope = 100MHz/1ms

    ## start reading in another thread but block by readEvent
    readEvent  = threading.Event()

    # ## practical version
    # readThread = threading.Thread(target=read, args=[ser, radar, readEvent], daemon=True)

    ## simulation version
    readThread = threading.Thread(target=readSimSignal, daemon=True,
        kwargs={'filename':'2252', 'samFreq':5e4, 'samTime':1e-2, 'radar':radar, 'readEvent':readEvent})
    
    readThread.start()
    print('Reading Signal')
    readEvent.set()

    # radarView = RadarView()

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
                radarView = RadarView(1)

                animation = FuncAnimation(radarView.fig, radarView.updateRealTimeSig,
                    frames=100, init_func=radarView.initRealTimeSig, interval=20, blit=True,
                    fargs=(radar.realTimeSig,))
                radarView.figShow()

            elif s.startswith('ppi'):
                radarView = RadarView(2)

                animation = FuncAnimation(radarView.fig, radarView.updatePPI,
                    frames=100, init_func=radarView.initPPI, interval=20, blit=True,
                    fargs=([random.random(), np.pi/3], [1,0.5]))

                radarView.figShow()

            elif s.startswith('close'):
                plt.close(radarView.fig)


            elif s.startswith('q'):
                break
            elif s.startswith('test'):
                print('hello world')
            else:
                print('Undefined Command')

    except KeyboardInterrupt:
        pass
    finally: print('Quit main')

    # ser.close()

if __name__ == '__main__':
    main()
