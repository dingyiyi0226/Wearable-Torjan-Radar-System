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
            'STEP': 16,
            'DIR' : 18,
            'ENA' : 7,  # no use
        }

        ## Modules

        self.rotateMotor = A4988(DIR_PINS)
        self.highFreqRadar = FMCWRadar(freq=5.8e9 , BW=100e6, tm=8e-3, pins=ADF_HIGH_PINS)
        # self.lowFreqRadar  = FMCWRadar(freq=915e6 , BW=15e6, tm=614e-6, pins=ADF_LOW_PINS)

        ## Data

        self.currentDir = 90
        self.lowData  = {}      ## info of each direction: { angle: [(range, velo)] }
        self.highData = {}


    def setSignal(self, signal, time, isHigh):

        if isHigh:
            self.highData[self.currentDir] = self.highFreqRadar.setSignal(signal, time)
        # else:
        #     self.lowData[self.currentDir] = self.lowFreqRadar.setSignal(signal, time)


    def setDirection(self, direction):

        deltaDir = direction - self.currentDir
        self.rotateMotor.spin(abs(deltaDir), deltaDir>0)
        self.currentDir = direction

    def setBgSignal(self, overwrite):
        self.highFreqRadar.setBgSig(overwrite)
        # self.lowFreqRadar.setBgSig(overwrite)
    def resetDirection(self):
        self.currentDir = 90


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

        self._samplingTime = 0.  ## in mircosecond
        self._peakFreqsIdx = []  ## peak freq index in fftSig
        self._objectFreqs  = []  ## [(f1,f2), (f3,f4), ... ] two freqs cause by an object
                                 ## the tuple contain only one freq iff the object is stationary

        self.backgroundSig = None
        self.realTimeSig = {'timeSig':np.zeros(1),'timeAxis':np.zeros(1),
                            'freqSig':np.zeros(1),'freqAxis':np.zeros(1), 'processedSig':np.zeros(1)}

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
            time record in unit (s)
        """

        if not signal: return

        self.realTimeSig['timeSig'] = np.array(signal)
        self._signalLength = self.realTimeSig['timeSig'].shape[0]
        self._samplingTime = time

        self.realTimeSig['timeAxis'] = np.arange(self._signalLength) * self._samplingTime / self._signalLength
        self.realTimeSig['freqAxis'] = np.arange(self._signalLength//2) / self._samplingTime

        return self._signalProcessing()


    def setBgSig(self, overwrite):
        if overwrite:
            self.backgroundSig = self.realTimeSig.copy()
        else:
            currentSig = self.realTimeSig.copy()
            for key, sig in currentSig.items():
                self.backgroundSig[key] = 0.75 * self.backgroundSig[key] + 0.25 * sig

    ## PRIVATE FUNCTION

    ## SIGNAL PROCESSING FUNCTIONS

    def _signalProcessing(self):

        self._fft()
        self._rmBgSig()
        self._avgFreqSig()
        if not self._findPeaks(height=3e-3, prominence=1e-4):
            # print('no peak frequency')
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

        BW = self._fm * 2                 ## bandwidth of the window
        winLength = int(BW*self._samplingTime)  ## length = BW/df = BW*T
        # window = np.ones(winLength)       ## window for averaging the signal
        window = sg.blackman(winLength)   ## window for averaging the signal
        # print(winLength)
        window = window/window.sum()

        self.realTimeSig['processedSig'] = sg.convolve(self.realTimeSig['processedSig'], window, mode='same')

    def _findPeaks(self, height, prominence) -> bool:
        """ Find peaks in processedSig
        
        Parameters
        ----------
        height : float
            min amplitude of peak frequencies
        prominence : float
            min prominence of peak frequencies

        Return
        ------
        status : bool
            False if no peak frequency is found
        """

        self._peakFreqsIdx, _ = sg.find_peaks(self.realTimeSig['processedSig'], height=height, prominence=prominence)
        # print(self._peakFreqsIdx)
        return len(self._peakFreqsIdx)!=0

    def _findFreqPair(self, peakDiff):
        """
        Split the freqs in `_peakFreq` with same intensity into pairs

        Parameters
        ----------
        peakDiff : float
            we assume two peaks belong to same object if and only if the amplitude
            difference between two peaks < peakDiff
        """

        sortedFreqIndex = sorted(self._peakFreqsIdx, key=lambda k: self.realTimeSig['processedSig'][k], reverse=True)

        freqAmplitude = 0
        tmpFreqIndex = 0
        # print('sortedFreqIndex', sortedFreqIndex)

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

                # have two solutions
                # if velo > 25, omit it

                if objVelo1<25:
                    infoList.append((objRange1, objVelo1))
                if objVelo2<25:
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
    isValid = True
    samplingTime = 0

    while readEvent.is_set():
        # readEvent.wait()
        ## maybe have to reset buffer

        ser.write(b'r ')
        # print(ser.readline().decode().strip())

        try:
            s = ser.readline().decode().strip()
    
            if s.startswith('i'):
                isValid = True
                signal.clear()

            elif s.startswith('d'):
                # print('readSignal ',s[2:])
                try:
                    signal.extend([float(i)/1024 for i in s[2:].split()])

                except ValueError:
                    print('Value Error: ', s[2:])
                    isValid = False

            elif s.startswith('e'):
                # print('endReadSignal ', s[2:])
                try:
                    samplingTime = float(s[2:]) * 1e-6
                    
                except ValueError:
                    print('Value Error: ', s[2:])
                    isValid = False

                if isValid:
                    troy.setSignal(signal, samplingTime, isHigh=True)

            else:
                print('\nRead:', s)

        except UnicodeDecodeError:
            print('UnicodeDecodeError')

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
        row = next(datas)
        if len(row)==1:   # new format
            simSampFreq = 16e3
            channel = 0 # 1~8
            for data in datas:
                if len(data[channel]):   # omit the last line
                    simSignal.append(float(data[channel])/1e4)
        else:
            row = next(datas)
            simSampFreq = 1/float(row[-1])
            for data in datas:
                simSignal.append(float(data[1]))

    samSig = []
    i=1
    j=random.randrange(len(simSignal))

    while readEvent.is_set():
        # readEvent.wait()

        if i % int(samTime * samFreq) != 0:
            samSig.append(simSignal[(int(j+i*simSampFreq/samFreq) % len(simSignal))])

        else:

            troy.setSignal(samSig, samTime, isHigh=True)

            samSig.clear()
            j = random.randrange(len(simSignal))
            time.sleep(0.001)

        i+=1
    # print('exit thread')

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
                if i.startswith('ttyUSB') or i.startswith('ttyACM'):
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
        ser = serial.Serial(port(), baudrate=115200, timeout=3)
        print('Successfully open port: ', ser)
        readThread = threading.Thread(target=read, args=[ser, troy, readEvent], daemon=True)
    else:
        readThread = threading.Thread(target=readSimSignal, daemon=True,
            kwargs={'filename': args.simulation, 'samFreq': 1.6e4, 'samTime': 1, 'troy': troy, 'readEvent': readEvent})
        
    readEvent.set()
    readThread.start()
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

                    readEvent = threading.Event()

                    if args.simulation is None:
                        readThread = threading.Thread(target=read, args=[ser, troy, readEvent], daemon=True)
                    else:
                        readThread = threading.Thread(target=readSimSignal, daemon=True,
                            kwargs={'filename': args.simulation, 'samFreq': 1.6e4, 'samTime': 1, 'troy': troy, 'readEvent': readEvent})

                    readEvent.set()
                    readThread.start()

            elif s.startswith('stop'):
                if not readEvent.is_set():
                    print('Not been reading signal')
                else:
                    readEvent.clear()
                    print('Stop Reading Signal')

            elif s.startswith('setbg'):
                # if s contains one argument only, overwrite background sig
                # otherwise, take average of previous background sig

                if len(s.split())==1:
                    print('Reset Background Signal')
                    troy.setBgSignal(overwrite=True)
                else:
                    troy.setBgSignal(overwrite=False)

            elif s.startswith('sig'):
                view = SigView(maxFreq=4e3, maxTime=0.25)  # for arduino
                # view = SigView(maxFreq=5e3, maxTime=1)  # for simulation
                views.append(view)
                animation = FuncAnimation(view.fig, view.update,
                    init_func=view.init, interval=200, blit=True,
                    fargs=(troy.highFreqRadar.realTimeSig,))
                view.figShow()

            elif s.startswith('ppi'):
                view = PPIView(maxR=25)
                views.append(view)

                animation = FuncAnimation(view.fig, view.update,
                    init_func=view.init, interval=200, blit=True,
                    fargs=(troy.highData,))

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

            elif s.startswith('resetdirection'):
                troy.resetDirection()

            # TODO: Temp function
            elif s.startswith('flush'):
                ser.reset_input_buffer()

            elif s.startswith('info'):
                print('high freq info:', troy.highData)
                print('low freq info:', troy.lowData)

            elif s.startswith('bgsig'):
                if troy.highFreqRadar.backgroundSig is None:
                    print('Background signal not exist')
                    continue
                view = SigView(maxFreq=5e3, maxTime=1, figname='Bg Waveform')  # for arduino
                views.append(view)
                animation = FuncAnimation(view.fig, view.update,
                    init_func=view.init, interval=200, blit=True,
                    fargs=(troy.highFreqRadar.backgroundSig,))
                view.figShow()

            elif s.startswith('file'):

                filename = s.split()[-1]
                filename = os.path.join('./rawdata', filename)
                if not os.path.exists(filename):
                    print('file {} not exists'.format(filename))
                    continue

                print('Stop current reading thread')

                readEvent.clear()
                readEvent = threading.Event()

                readThread = threading.Thread(target=readSimSignal, daemon=True,
                    kwargs={'filename': filename, 'samFreq': 1.6e4, 'samTime': 1, 'troy': troy, 'readEvent': readEvent})
                
                readEvent.set()
                readThread.start()
                print('Reading Signal')

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
