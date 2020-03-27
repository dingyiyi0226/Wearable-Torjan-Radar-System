import csv
import os
import random
import sys
import threading
import time
import warnings
import argparse
from collections import namedtuple
from datetime import datetime

import numpy as np
import scipy.signal as sg
import serial
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation

import ADF4158
from A4988 import A4988
from view import PPIView, SigView

# GND  = 6      # T3
W_CLK  = 12     # T4
DATA   = 16     # T5
LE     = 18     # T6
TXDATA = 13     # T16
MUXOUT = 15     # T8

STEP = 3
DIR = 5
ENA = 7

class FMCWRadar:
    """ FMCW Radar model for each freqency """

    def __init__(self, freq, BW, tm):

        ## SIGNAL IDENTITY

        self._freq = freq        ## the operation frequency
        self._slope = BW/tm      ## the slope of transmitting signal (Hz/s)
        self._BW = BW
        self._tm = tm
        self._fm = 1/tm
        self._distanceOffset = 1 * 2.24 ** 0.5

        ## SIGNAL GENERATOR MODULE

        self._signalModule = ADF4158.set5800Default(ADF4158.ADF4158(W_CLK, DATA, LE, TXDATA, MUXOUT))
        self._rotateMotor = A4988(ENA, STEP, DIR)

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
        self.realTimeSig = {'timeSig':[1], 'timeAxis':[1],'freqSig':[1],'freqAxis':[1], 'avgFreqSig':[1]}

    ## PUBLIC FUNCTION

    ## RADAR INFO

    # TODO
    def showConfig(self):
        pass

    ## RADAR ADJUSTMENT

    # TODO
    def setModuleProperty(self, tm):
        pass

    # TODO
    def setDirection(self, direction):
        pass

    # TODO
    def alignDirection(self):
        pass

    ## SIGNAL ACCESSING FUCNTION

    def resetSignal(self):
        self._signal = []

    def readSignal(self, signal):
        """ Read analog signal from Arduino (signal values in range(0, 1024)) """
        # print(signal)
        self._signal.extend([i/1024 for i in signal])  
        # self._signal.extend(signal)  # read from file

    def endReadSignal(self, time):
        """ 
        Update some variable at the end of the signal and start signal processing 
        
        Parameters
        ----------
        time : int
            time record in unit (us)
        """

        if not self._signal: return

        self._signalLength = len(self._signal)
        self._samplingTime = time * 1e-6

        self.realTimeSig['timeAxis'] = [i*self._samplingTime/self._signalLength for i in range(self._signalLength)]
        self.realTimeSig['freqAxis'] = [i/self._samplingTime for i in range(self._signalLength//2)]
        self.realTimeSig['timeSig'] = self._signal[:]

        self._signalProcessing()

    def setBackgroundSig(self):
        self.backgroundSig = self.realTimeSig

    ## PRIVATE FUNCTION

    ## SIGNAL PROCESSING FUNCTIONS

    def _signalProcessing(self):
        self._fft()
        # if not self._findFreqPair():
        #     print('no peak frequency')
        #     return
        # self._calculateInfo()

    def _fft(self):
        """ Perform FFT on self._signal """

        PEAK_HEIGHT = 5e-3      ## amplitude of peak frequency must exceed PEAK_HEIGHT
        PEAK_PROMINENCE = 1e-4  ## prominence of peak frequency must exceed PEAK_PROMINENCE
        fftSignal_o = abs(np.fft.fft(self._signal))
        self.realTimeSig['freqSig'] = [i*2/self._signalLength for i in fftSignal_o[:self._signalLength//2]]
        self._avgFFTSig()

        self._peakFreqsIdx, _ = sg.find_peaks(self.realTimeSig['avgFreqSig'], height=PEAK_HEIGHT, prominence=PEAK_PROMINENCE)
        # print(self._peakFreqsIdx)

    def _avgFFTSig(self):
        """ Averaging the FFT signal """

        AVGTICK = 3   ## the number of ticks on frequency axis
        minFreqdiff = 1/self._samplingTime
        winLength = int(self._fm*self._samplingTime)
        window = np.ones(winLength)       ## window for averaging the signal
        window = window/window.sum()

        self.realTimeSig['avgFreqSig'] = sg.oaconvolve(self.realTimeSig['freqSig'], window, mode='same')
        # self.realTimeSig['avgFreqSig'] = self.realTimeSig['freqSig']

    def _findFreqPair(self) -> bool:
        """
        Split the freqs in `_peakFreq` with same intensity into pairs
        
        Return
        ------
        status : bool
            false if no peak frequency is found.
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
        """
        Calculate range and velocity of every object from `_objectFreqs`
        """

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

def read(ser, radar: FMCWRadar, readEvent: threading.Event):
    """ Read signal at anytime in other thread """

    while True:
        readEvent.wait()
        ## maybe have to reset buffer

        ser.write(b'r ')
        # print(ser.readline().decode().strip())

        try:
            s = ser.readline().decode().strip()
    
            if s.startswith('i'):
                radar.resetSignal()

            elif s.startswith('d'):
                # print('readSignal ',s[2:])
                try:
                    radar.readSignal(signal=[float(i) for i in s[2:].split()])
                except ValueError:
                    print('Value Error: ', s[2:])
                    continue

            elif s.startswith('e'):
                # print('endReadSignal ', s[2:])
                try:
                    radar.endReadSignal(time=float(s[2:]))
                except ValueError:
                    print('Value Error: ', s[2:])
                    continue

            # BUG: Not execute by 1st condition 'if s.startswith('i')'
            elif s.startswith('init'):
                pass

            else:
                print('\nRead: ', s)

        except UnicodeDecodeError:
            print('UnicodeDecodeError')
            continue

        time.sleep(0.001)

def readSimSignal(filename, samFreq, samTime, radar: FMCWRadar, readEvent: threading.Event):
    """
    Load signal from data

    Parameters
    ----------
    filename :

    samFreq :
    
    samTime : 
    """
    
    simSignal = []
    simSampFreq = 0

    # Load csvfile
    with open(filename) as file:
        datas = csv.reader(file)

        for ind, data in enumerate(datas):
            if ind==0: continue
            elif ind==1:
                simSampFreq = 1 / float(data[-1])
            else:
                simSignal.append(float(data[1]))

    samSig = []
    i=1
    j=random.randrange(len(simSignal))

    while True:
        readEvent.wait()

        if i % int(samTime * samFreq) != 0:
            samSig.append(simSignal[(int(j+i*simSampFreq/samFreq) % len(simSignal))])

        else:
            radar.resetSignal()
            j = random.randrange(len(simSignal))
            radar.readSignal(signal=samSig)

            radar.endReadSignal(time=samTime*1e6)
            samSig.clear()
            time.sleep(0.001)

        i+=1

def port() -> str:
    """ Find the name of the port """

    try:
        ## on mac
        if sys.platform.startswith('darwin'):
            ports = os.listdir('/dev/')
            for i in ports:
                if i[0:-2] == 'tty.usbserial-14':
                    port = i
                    break;
            port = '/dev/' + port
            
        ## on rpi
        if (sys.platform.startswith('linux')):
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
    ## Main function initialization arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--simulation', type=str, help='Read signal files and simulation')
    args = parser.parse_args()

    ## Arguments checking
    if args.simulation is not None:
        if not os.path.exists(args.simulation):
            sys.exit('Argument --simulation ({}) wrong, exit'.format(args.simulation))

    ## For file writing (Command: Save)
    now = datetime.today().strftime('%Y%m%d')

    ## Initialize radar 
    radar = FMCWRadar(freq=58e8 , BW=99.9969e6, tm=2.048e-3)  ## operating at 5.8GHz, slope = 100MHz/1ms

    ## Start reading in another thread but block by readEvent
    readEvent  = threading.Event()

    if args.simulation is None:
        ## Port Connecting
        ser = serial.Serial(port(), baudrate=115200)
        print('Successfully open port: ', ser)
        readThread = threading.Thread(target=read, args=[ser, radar, readEvent], daemon=True)
    else:
        readThread = threading.Thread(target=readSimSignal, daemon=True,
            kwargs={'filename': args.simulation, 'samFreq': 1e4, 'samTime': 2.4e-2, 'radar': radar, 'readEvent': readEvent})
        
    readThread.start()
    readEvent.set()
    print('Reading Signal')

    views = []

    try:
        prompt = ''
        while True:
            s = input("commands: " + prompt).strip()

            if s == '': pass

            elif s.startswith('read'):
                if readEvent.is_set():
                    print('Has been reading signal')
                else:
                    print('Reading Signal')
                    readEvent.set()

            elif s.startswith('stop'):
                if not readEvent.is_set():
                    print('Not been reading signal')
                else:
                    readEvent.clear()
                    print('Stop Reading Signal')


            elif s.startswith('sig'):
                view = SigView(maxFreq=5e3, maxTime=0.2)  # for arduino
                # view = SigView(maxFreq=10e3, maxTime=1e-2)  # for simulation
                views.append(view)
                animation = FuncAnimation(view.fig, view.update,
                    init_func=view.init, interval=100, blit=True,
                    fargs=(radar.realTimeSig,))
                view.figShow()

            elif s.startswith('ppi'):
                view = PPIView(maxR=5)
                views.append(view)

                animation = FuncAnimation(view.fig, view.update,
                    frames=100, init_func=view.init, interval=100, blit=True,
                    fargs=(radar.direction, radar.info[radar.direction]))

                view.figShow()

            elif s.startswith('close'):
                for view in views:
                    plt.close(view.fig)
                views.clear()

            elif s.startswith('save'):
                """ Save time domain signal """
                
                distance = input('Distances: ').strip()
                comments = input('Comments: ').strip()

                path = './rawdata/arduino/{}'.format(now)
                if not os.path.exists(path):
                    os.makedirs(path)

                if os.path.exists(os.path.join(path, distance + '.csv')):
                    print("File exists. Overwrite it.")
            
                with open(os.path.join(path, distance + '.csv'), 'w') as file:
                    writer = csv.writer(file)
                    writer.writerow(['X', 'Sig', '', 'Increment'])
                    writer.writerow(['', '', '', str(radar.realTimeSig['timeAxis'][1])])
                    
                    for ind, i in enumerate(radar.realTimeSig['timeSig']):
                        writer.writerow([ind, i])

                print("File is saved at: {}".format(os.path.join(path, distance + '.csv')))

            # TODO: ADF4158 module
            # SetFreq with receive 1 argument: frequency
            elif s.startswith('setfreq'):
                pass

            # TODO: ADF4158 module
            elif s.startswith('setModulation'):
                pass

            # TODO: A4988 module
            # SetDirection with receive 1 argument: angle
            elif s.startswith('setdirection'):
                pass

            # TODO: A4988 module and FMCWRadar
            elif s.startswith('resetdirection'):
                pass

            # TODO: Temp function
            elif s.startswith('flush'):
                ser.reset_input_buffer()

            elif s.startswith('info'):
                print(radar.info)

            elif s.startswith('q'):
                break

            else:
                print('Undefined Command')

    except KeyboardInterrupt:
        pass

    finally:
        print('Quit main')

    # ser.close()

if __name__ == '__main__':
    main()
