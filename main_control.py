import numpy as np
import matplotlib.pyplot as plt
import serial
import sys, os
import time, threading
from scipy.signal import find_peaks

### FMCW Radar model
class Signal:
	def __init__(self, plotEvent):

		## DATA

		self._info = {} ## info of each direction: { angle: [(range, velo)] }
		self._direction  = 90  ## operating direction , at 90 degree by default

		## SIGNAL PROCESSING

		self._dataLength = 0   ## length of received data , fft data
		self._datas      = []  ## received data
		self._timeAxis   = []  ## time axis of received data
		self._fftDatas   = []  ## fft data
		self._freqAxis   = []  ## freq axis of fft data
		self._samplingTime = 0.   ## in mircosecond
		self._peakFreqs  = []  ## peak freq in fftdata

		## PLOTTING 

		self._plotEvent = plotEvent


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

	def setSamplingTime(self, time):
		self._samplingTime = time * 1e-6
		self._freqAxis = [i/self._samplingTime for i in self._timeAxis]

	def readSerialData(self, data):
		# print(data)
		self._datas = [ float(i) for i in data.split() ]
		self._dataLength = len(self._datas)
		self._timeAxis = [ i for i in range(self._dataLength)]

		self._fftData()

	def _fftData(self):

		fftdata_o = abs(np.fft.fft(self._datas))
		self._fftDatas = [ i*2/self._dataLength for i in fftdata_o ]
		self._peakFreqs, _ = find_peaks(self._fftDatas, height = 10)

		self._calculateInfo()
		self._plotEvent.set()

	def _calculateInfo(self):
		pass



	## PLOTTING FUNCTIONS

	### only plot data with frequencies in (0, maxFreq) 
	def plotData(self, DCBlock : bool = False, maxFreq = None) -> bool:

		## convert maxFreq to corresponding index (maxIndex)
		maxIndex = self._dataLength//2 if maxFreq is None else int(maxFreq * self._samplingTime)

		if maxIndex > self._dataLength//2: 
			print('maxFreq do not exceed ', int(self._dataLength//2 / self._samplingTime))
			self._plotEvent.clear()
			return False
		
		plt.clf()

		plt.subplot(211)
		plt.plot(self._timeAxis,self._datas)

		plt.subplot(212)
		plt.xlabel('freq(Hz)')

		if DCBlock:	plt.plot( self._freqAxis[1:maxIndex], self._fftDatas[1:maxIndex],'r')
		else:       plt.plot( self._freqAxis[1:maxIndex], self._fftDatas[0:maxIndex],'r')

		plt.plot([ self._freqAxis[i] for i in self._peakFreqs if i < maxIndex ],
				 [ self._fftDatas[i] for i in self._peakFreqs if i < maxIndex ], 'x')

		# plt.yscale('log')
		plt.pause(0.001)

		self._plotEvent.clear()
		return True
		
def read():

	global ser
	global readEvent
	global signal

	while True:

		readEvent.wait()   
		try:
			s = ser.readline().decode()
			if s.startswith('d'):
				# print('readSerialData ',s[2:])
				signal.readSerialData(s[2:])

			elif s.startswith('t'):
				# print('setSamplingTime ', s[2:])
				try: 
					signal.setSamplingTime(float(s[2:]))
				except ValueError:
					print('Value Error: ',s[2:])
					continue

			else: 
				print('Read: ', s)

		except UnicodeDecodeError:
			print('UnicodeDecodeError')
			continue

		time.sleep(0.001)

### write time info to serial (for debugging)
def writeTime(ser):
	while True:
		s = time.asctime()
		b = ser.write(s.encode())
		# print('writing',s)
		time.sleep(2)


def main():

	## global variebles

	global ser
	global signal

	global readEvent
	global plotEvent

	## ------------- Port Connecting ------------- ##

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

		ser = serial.Serial(port)
		print('Successfully open port: ', ser)

	except (serial.SerialException, UnboundLocalError):
		print('Cannot open port')
		sys.exit()

	## ------------------------------------------ ##

	plotEvent = threading.Event()
	signal = Signal(plotEvent = plotEvent)

	readEvent = threading.Event()

	readThread = threading.Thread(target = read, daemon = True)
	readThread.start()

	# writeThread = threading.Thread(target = writeTime, args = [ser],daemon = True)
	# writeThread.start()

	try:
		prompt = ''
		while True: 
			s = input("commands: " + prompt ).strip()
			if s.startswith('q'): break

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
							plotEvent.wait()
							if not signal.plotData(DCBlock = True, maxFreq = 300): 
							# if not signal.plotData(DCBlock = True):
								plt.close('Signal')
								break
							time.sleep(0.01)

					except KeyboardInterrupt:
						plt.close('Signal')
						print('Quit drawing')

						# ## stop read signal
						# readEvent.clear()
						# print('Stop Reading Signal')

			elif s.startswith('currentdir'):
				print('current direction: ', signal.direction)

			elif s.startswith('infoat'):
				try:
					ss = input('direction: ')
					info = signal.infoAtDirection(float(ss))
					if info is None:
						print('{} is not a valid direction'.format(ss))
					else:
						print('direction: {}, (range, velocity): {}'.format(ss, info))
				except ValueError:
					print('{} is not a valid direction'.format(ss))

			elif s.startswith('info'):
				for ind,val in signal.info.items():
					print('direction: {}, (range, velocity): {}'.format(ind, val))

			elif s.startswith('resetinfo'):
				signal.resetInfo()
				print('reset all direction data')

			elif s == '':
				pass
			else:
				print('Undefined Command')

	except KeyboardInterrupt: 
		pass
	finally: print('Quit main')

	ser.close()

if __name__ == '__main__':
	main()
