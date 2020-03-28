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

class Troy:
    
    def __init__(self):

        ## Pins

        ADF_HIGH_PINS = {
            # 'GND': 6,        # T3
            'W_CLK': 12,       # T4
            'DATA' : 16,       # T5
            'LE'   : 18,       # T6
            'TXDATA' : 13,     # T16
            'MUXOUT' : 15,     # T8
        }

        ADF_LOW_PINS = {
            # 'GND': 60,        # T3
            'W_CLK': 120,       # T4
            'DATA' : 160,       # T5
            'LE'   : 180,       # T6
            'TXDATA' : 130,     # T16
            'MUXOUT' : 150,     # T8
        }

        DIR_PINS = {
            'STEP': 3,
            'DIR' : 5,
            'ENA' : 7,
        }

        ## Modules

        self.rotateMotor = A4988(DIR_PINS)
        self.highFreqRadar = FMCWRadar(freq=5.8e9 , BW=99.9969e6, tm=2.048e-3, pins=ADF_HIGH_PINS)
        self.lowFreqRadar  = FMCWRadar(freq=915e6 , BW=15e6, tm=614e-6, pins=ADF_LOW_PINS)

        ## Data

        self.currentDir = 90
        self.lowData  = {}      ## info of each direction: { angle: [(range, velo)] }
        self.highData = {}


    def setSignal(self, signal, time, isHigh):

        if isHigh:
            self.highData[self.currentDir] = self.highFreqRadar.setSignal(signal, time)
        else:
            self.lowData[self.currentDir] = self.lowFreqRadar.setSignal(signal, time)


    def setDirection(self, direction):

        deltaDir = direction - self.currentDir
        self.rotateMotor.spin(abs(deltaDir), deltaDir>0)
        self.currentDir = direction


class FMCWRadar:
    """ FMCW Radar model for each freqency """

    def __init__(self, freq, BW, tm, pins):

        ## SIGNAL IDENTITY

        self._freq = freq        ## the operation frequency
        self._slope = BW/tm      ## the slope of transmitting signal (Hz/s)
        self._BW = BW
        self._tm = tm
        self._fm = 1/tm
        self._distanceOffset = 1 * 2.24 ** 0.5

        ## SIGNAL GENERATOR MODULE

        self._signalModule = ADF4158.set5800Default(ADF4158.ADF4158(pins['W_CLK'], pins['DATA'], pins['LE'], pins['TXDATA'], pins['MUXOUT']))

        ## SIGNAL PROCESSING

        self._signalLength = 0   ## length of received data. len(timeSig) = 2*len(freqSig)
        # self._resetFlag = False  ## reset signal flag
        # self._signal = []        ## received data (temp signal)

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

    ## SIGNAL ACCESSING FUCNTION

    def setSignal(self, signal, time):
        """
        Update signal and some variable at the end of the signal and start signal processing
        
        Parameters
        ----------
        time : int
            time record in unit (us)
        """

        if not signal: return

        self.realTimeSig['timeSig'] = signal.copy()
        self._signalLength = len(self.realTimeSig['timeSig'])
        self._samplingTime = time * 1e-6

        self.realTimeSig['timeAxis'] = [i*self._samplingTime/self._signalLength for i in range(self._signalLength)]
        self.realTimeSig['freqAxis'] = [i/self._samplingTime for i in range(self._signalLength//2)]

        return self._signalProcessing()


    def setBackgroundSig(self):
        self.backgroundSig = self.realTimeSig

    ## PRIVATE FUNCTION

    ## SIGNAL PROCESSING FUNCTIONS

    def _signalProcessing(self):
        self._fft()
        if not self._findFreqPair():
            print('no peak frequency')
            return
        return self._calculateInfo()

    def _fft(self):
        """ Perform FFT on realtime signal """

        PEAK_HEIGHT = 5e-3      ## amplitude of peak frequency must exceed PEAK_HEIGHT
        PEAK_PROMINENCE = 1e-4  ## prominence of peak frequency must exceed PEAK_PROMINENCE
        fftSignal_o = abs(np.fft.fft(self.realTimeSig['timeSig']))
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
            False if no peak frequency is found.
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

        self._objectFreqs.clear()
        return infoList

    def _freq2Range(self, freq):
        return freq / self._slope * 3e8 / 2

    def _freq2Velo(self, freq):
        return freq / self._freq * 3e8 / 2

def read(ser, troy: Troy, readEvent: threading.Event):
    """ Read signal at anytime in other thread """

    signal = []
    samplingTime = 0

    while True:
        readEvent.wait()
        ## maybe have to reset buffer

        ser.write(b'r ')
        # print(ser.readline().decode().strip())

        try:
            s = ser.readline().decode().strip()
    
            if s.startswith('i'):
                signal.clear()

            elif s.startswith('d'):
                # print('readSignal ',s[2:])
                try:
                    signal.extend([float(i)/1024 for i in s[2:].split()])

                except ValueError:
                    print('Value Error: ', s[2:])
                    continue

            elif s.startswith('e'):
                # print('endReadSignal ', s[2:])
                try:
                    samplingTime = float(s[2:])
                    troy.setSignal(signal, samplingTime, isHigh=True)

                except ValueError:
                    print('Value Error: ', s[2:])
                    continue

            else:
                print('\nRead: ', s)

        except UnicodeDecodeError:
            print('UnicodeDecodeError')
            continue

        time.sleep(0.001)

def readSimSignal(filename, samFreq, samTime, troy: Troy, readEvent: threading.Event):
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

            troy.setSignal(samSig, samTime*1e6, isHigh=True)

            samSig.clear()
            j = random.randrange(len(simSignal))
            time.sleep(0.001)

        i+=1

def port() -> str:
    """ Find the name of the port """

    try:
        ## on mac
        if sys.platform.startswith('darwin'):
            ports = os.listdir('/dev/')
            for i in ports:
                if i.startswith('tty.usbserial-14') or i.startswith('tty.usbmodem14'):
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
    parser.add_argument('-s', '--simulation', type=str, help='Read signal files and simulation')
    args = parser.parse_args()

    ## Arguments checking
    if args.simulation is not None:
        if not os.path.exists(args.simulation):
            sys.exit('Argument --simulation ({}) wrong, exit'.format(args.simulation))

    ## For file writing (Command: Save)
    now = datetime.today().strftime('%Y%m%d')

    ## Initialize troy model

    troy = Troy()

    ## Start reading in another thread but block by readEvent
    readEvent  = threading.Event()

    if args.simulation is None:
        ## Port Connecting
        ser = serial.Serial(port(), baudrate=115200)
        print('Successfully open port: ', ser)
        readThread = threading.Thread(target=read, args=[ser, troy, readEvent], daemon=True)
    else:
        readThread = threading.Thread(target=readSimSignal, daemon=True,
            kwargs={'filename': args.simulation, 'samFreq': 1e4, 'samTime': 2.4e-2, 'troy': troy, 'readEvent': readEvent})
        
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
                    fargs=(troy.highFreqRadar.realTimeSig,))
                view.figShow()

            elif s.startswith('ppi'):
                view = PPIView(maxR=5)
                views.append(view)

                animation = FuncAnimation(view.fig, view.update,
                    frames=100, init_func=view.init, interval=100, blit=True,
                    fargs=(troy.currentDir, troy.highData[troy.currentDir]))

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
                    writer.writerow(['', '', '', str(troy.highFreqRadar.realTimeSig['timeAxis'][1])])
                    
                    for ind, i in enumerate(troy.highFreqRadar.realTimeSig['timeSig']):
                        writer.writerow([ind, i])

                print("File is saved at: {}".format(os.path.join(path, distance + '.csv')))

            # TODO: ADF4158 module
            # SetFreq with receive 1 argument: frequency
            elif s.startswith('setfreq'):
                pass

            # TODO: ADF4158 module
            elif s.startswith('setModulation'):
                pass

            elif s.startswith('setdirection'):

                direction = input('Direction: ').strip()

                try:
                    direction = float(direction)
                    troy.setDirection(direction)

                except ValueError:
                    print('invalid direction')

            # TODO: A4988 module and FMCWRadar
            elif s.startswith('resetdirection'):
                pass

            # TODO: Temp function
            elif s.startswith('flush'):
                ser.reset_input_buffer()

            elif s.startswith('info'):
                print('high freq info:', troy.highData)
                print('low freq info:', troy.lowData)

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
