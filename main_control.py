import numpy as np
import matplotlib.pyplot as plt
import serial
import sys, os
import time, threading
from scipy.signal import find_peaks

### FMCW Radar model for each freqency
class FMCWRadar:

	def __init__(self, freq, slope):

		## SIGNAL IDENTITY

		self._freq  = freq   ## the operation frequency
		self._slope = slope  ## the slope of transmitting signal (Hz/s)

		## DATA

		self._info = {}      ## info of each direction: { angle: [(range, velo)] }
		self._direction = 90 ## operating direction , at 90 degree by default

		## SIGNAL PROCESSING

		self._signalLength = 0   ## length of received data , fft data
		self._signal       = []  ## received data
		self._resetFlag = False  ## reset signal flag
		self._timeAxis     = []  ## time axis of received data
		self._fftSignal    = []  ## fft data
		self._freqAxis     = []  ## freq axis of fft data
		self._samplingTime = 0.  ## in mircosecond
		self._peakFreqs    = []  ## peak freq in fftdata
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
		try:
			if self._resetFlag:
				self._signal = []
				self._resetFlag = False

			self._signal.extend([float(i) for i in signal.split()])
			
		except ValueError:
			print('ValueError')

	### update some varieble at the end of the signal and start signal processing
	def endReadSignal(self, time):
		if not self._signal: return
		self._signalLength = len(self._signal)
		self._timeAxis     = [ i for i in range(self._signalLength)]
		self._samplingTime = time * 1e-6
		self._freqAxis     = [i/self._samplingTime for i in self._timeAxis]

		self._signalProcessing()

	def _signalProcessing(self):
		self._fft()
		# self._findFreqPair()
		# self._calculateInfo()

	def _fft(self):
		fftSignal_o     = abs(np.fft.fft(self._signal))
		self._fftSignal = [ i*2/self._signalLength for i in fftSignal_o ]
		self._peakFreqs, _ = find_peaks(self._fftSignal, height = 10)
		self._plotEvent.set()

	### make the freqs in peakFreq with same intensity into pairs
	def _findFreqPair(self):
		## TODO
		## Update _objectFreqs from _peakFreqs
		pass

	### calculate range and velocity of every object from _objectFreqs
	def _calculateInfo(self):
		objRange = 0.
		objVelo  = 0.
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

			if self._info.has_key(self._direction):
				self._info[self._direction] += (objRange, objVelo)
			else:
				self._info[self._direction] = [(objRange, objVelo)]

	## PLOTTING FUNCTIONS

	@property
	def plotEvent(self):
		return self._plotEvent

	### only plot signal with frequencies in (0, maxFreq) 
	def plotSignal(self, DCBlock : bool = False, maxFreq = None) -> bool:
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

		plt.subplot(212)
		plt.xlabel('freq(Hz)')

		if DCBlock:	plt.plot( self._freqAxis[1:maxIndex], self._fftSignal[1:maxIndex],'r')
		else:       plt.plot( self._freqAxis[1:maxIndex], self._fftSignal[0:maxIndex],'r')

		plt.plot([ self._freqAxis[i]  for i in self._peakFreqs if i < maxIndex ],
				 [ self._fftSignal[i] for i in self._peakFreqs if i < maxIndex ], 'x')

		# plt.yscale('log')
		plt.pause(0.001)
		self._plotEvent.clear()
		return True

### read signal at anytime in other thread
def read(ser, radar, readEvent):
	while True:
		readEvent.wait()   
		## maybe have to reset buffer
		try:
			s = ser.readline().decode()
			if s.startswith('i'):
				radar.resetSignal()

			elif s.startswith('d'):
				# print('readSignal ',s[2:])
				radar.readSignal(signal = s[2:])

			elif s.startswith('e'):
				# print('endReadSignal ', s[2:])
				try: 
					radar.endReadSignal(time = float(s[2:]))
				except ValueError:
					print('Value Error: ',s[2:])
					continue

			else: 
				print('Read: ', s)

		except UnicodeDecodeError:
			print('UnicodeDecodeError')
			continue

		time.sleep(0.001)

### find the port name
def port() -> str:
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

	### Port Connecting
	ser = serial.Serial(port())
	print('Successfully open port: ', ser)

	### initialize the model 
	radar = FMCWRadar(freq = 58e8 , slope = 1e4/5e-4)  ## operating at 5.8GHz, slope = 10kHz/0.5ms

	### start reading in another thread but block by readEvent
	readEvent  = threading.Event()
	readThread = threading.Thread(target = read, args = [ser, radar, readEvent], daemon = True)
	readThread.start()

	try:
		prompt = ''
		while True: 
			s = input("commands: " + prompt ).strip()

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
							if not radar.plotSignal(DCBlock = True, maxFreq = 300): 
							# if not radar.plotSignal(DCBlock = True):
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

			elif s.startswith('q'): break		
			else: print('Undefined Command')

	except KeyboardInterrupt: 
		pass
	finally: print('Quit main')

	ser.close()

if __name__ == '__main__':
	main()
