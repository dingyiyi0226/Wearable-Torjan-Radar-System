import numpy as np
import matplotlib.pyplot as plt
import serial
import sys, os
import time, threading
from calphase import Phase

class singleDirectionFMCW:
	def __init__(self, bw, tm): ## tm(ms)
		self._bw = bw
		self._tm = tm
		self._slope = self._bw/self._tm*2

		self._realTimeFreq = []

		self._fb1 = 0
		self._fb2 = 0

		self._beatFreq = self._slope*0.0001  ## assume travel time = 0.1us
		self._dopplerFreq = 0

		self._range = 0
		self._velocity = 0

	def setRealTimeFreq(self,freq):
		self._realTimeFreq.append(freq)

	def RangeAndVelocity(self):
		if not self._realTimeFreq: return None
		# print(self._realTimeFreq)
		self._calculateFreq()
		self._calculateInfo()
		return self._range, self._velocity

	def reset(self):
		self._realTimeFreq = []
		self._fb1 = 0
		self._fb2 = 0
		self._dopplerFreq = 0
		self._range = 0
		self._velocity = 0

	### setting beatFreq and doppler Freq from realTimeFreq
	def _calculateFreq(self):
		diff = np.abs(np.diff(self._realTimeFreq))
		change = []
		for index, value in enumerate(diff):
			if value > 1: change.append(index)
		# if len(change) > 1: 
		# 	print("freq2info Error", diff)
		if change:
			self._fb1 = sum(self._realTimeFreq[:change[0]+1])/(change[0]+1)
			self._fb2 = sum(self._realTimeFreq[change[0]+1:])/len(self._realTimeFreq[change[0]+1:])
		else: 
			self._fb2 = self._fb1 = sum(self._realTimeFreq)/len(self._realTimeFreq)

		ftmp1 = (self._fb1 + self._fb2)/2
		ftmp2 = abs(self._fb1 - self._fb2)/2

		if abs(ftmp1 - self._beatFreq) < abs(ftmp2 - self._beatFreq):
			self._beatFreq = ftmp1
			self._dopplerFreq = ftmp2
		else:
			self._beatFreq = ftmp2
			self._dopplerFreq = ftmp1

	### calculating range and velocity from beatFreq and doppler Freq
	def _calculateInfo(self):
		self._range = self._beatFreq * 3e8 / self._slope / 2
		self._velocity = self._dopplerFreq * 3e8 / 5.8e9 / 2

class Signal:
	def __init__(self, N, drawEvent):
		self._N = N
		self._datacnt   = 0
		self._datas     = [0. for i in range(self._N)]
		self._x         = [i  for i in range(self._N)]
		self._fftDatas  = []
		self._peakFreq  = 0
		self._drawEvent = drawEvent

	def readData(self,data):
		# print('readData', data)
		self._datas[self._datacnt] = data
		if self._datacnt < self._N-1:
			self._datacnt+=1
		else:
			## receive N datas
			self._datacnt=0
			# print('readend')
			self.fftData()
			# print('fftend')

	def fftData(self):
		global directionInfo
		global direction
		fftdata_o = abs(np.fft.fft(self._datas))
		self._fftDatas = [ i*2/self._N for i in fftdata_o ]
		self._peakFreq = max(self._fftDatas[1:])  ## block DC

		# print('setRealTimeFreq: ', direction)
		directionInfo[direction].setRealTimeFreq(self._peakFreq)

		self._drawEvent.set()
		# print(self._fftDatas)
	def drawData(self, DCBlock = False, maxFreq = None):
		if maxFreq is None: maxFreq = self._N//2
		if maxFreq > self._N//2: 
			print('maxFreq do not exceed N/2')
			self._drawEvent.clear()
			return False
		
		plt.clf()

		plt.subplot(211)
		plt.plot(self._x,self._datas)

		plt.subplot(212)
		plt.xlabel('freq(Hz)')
		if DCBlock:	plt.plot(self._x[1:maxFreq],self._fftDatas[1:maxFreq],'r')
		else:       plt.plot(self._x[:maxFreq] ,self._fftDatas[:maxFreq],'r')

		if maxFreq < 100:
			plt.xticks(self._x[:maxFreq:maxFreq//15])
		else: 
			plt.xticks(self._x[:maxFreq:maxFreq//100*10])
		# plt.ylim(0,20)
		plt.pause(0.001)

		self._drawEvent.clear()
		return True

	@property
	def peakFreq(self):
		return self._peakFreq
	
"""
## adding a variable for threading.event
class ReadEvent(threading.Event):
	def __init__(self):
		super().__init__()
		self._didSet = False

	def set(self): 
		with self._cond:
			self._flag = True
			self._didSet = True
			self._cond.notify_all()

	def wait(self, timeout=None):
		with self._cond:
			signaled = self._flag
			if not signaled:
				signaled = self._cond.wait(timeout)
			if self._didSet:
				self._didSet = False
				return signaled, True
			else: 
				return signaled, False
"""
def read(ser,readEvent):

	global signal
	global direction
	global setphaseIntervalFlag

	while True:

		readEvent.wait()
		# _, didset = readEvent.wait()
		# if didset:
		# 	ser.reset_input_buffer()

		try:
			s = ser.readline().decode().strip()
			# print('reading', s)

			if s.startswith('d'):
				if not setphaseIntervalFlag: 
					# print('interval')
					continue
				try:
					signal.readData(float(s[2:]))


				except ValueError:
					print('Value Error: ',s[2:])
					continue
			elif s.startswith('p'):
				print('did set phase')
				setphaseIntervalFlag = True

			else: 
				print('Read: ', s)
				pass

		except UnicodeDecodeError:
			print('UnicodeDecodeError')
			continue
		time.sleep(0.001)

def writeTime(ser):
	while True:
		s = time.asctime()
		b = ser.write(s.encode())
		# print('writing',s)
		time.sleep(2)

def setphase(angle):
	global ser
	global phase

	global setphaseIntervalFlag

	global direction
	global actdirection


	if not phase.isValidDirection(angle): return False

	direction = angle
	actdirection = phase.getActualDirection(angle)
	thetas = phase.getEachPhase2Pin(angle)

	data = 'p '+str(thetas[0])+' '+str(thetas[1])+' '+str(thetas[2])+' '+str(thetas[3])
	setphaseIntervalFlag = False
	ser.write(data.encode())
	print('setting:', angle)

	return True


def main():

	## global variebles

	global ser
	global signal
	global phase
	global readEvent
	global drawEvent

	global directionInfo

	global setphaseIntervalFlag
	global direction
	global actdirection

	## ------------- Port Connecting ------------- ##

	try:
		## on mac
		if(sys.platform.startswith('darwin')):
			port = '/dev/tty.usbserial-1420'
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

	except serial.SerialException:
		print('Cannot open port: ', port)
		sys.exit()

	## ------------------------------------------ ##

	## the Number of sampling data 
	N = 500

	drawEvent = threading.Event()
	signal = Signal(N = N,drawEvent = drawEvent)

	directionInfo = { i: singleDirectionFMCW(100000,500) for i in range(10,171,5)}

	phase = Phase()
	setphaseIntervalFlag = True

	readEvent = threading.Event()
	# readEvent = ReadEvent()

	readThread = threading.Thread(target = read, args = [ser,readEvent], daemon = True)
	readThread.start()

	# writeThread = threading.Thread(target = writeTime, args = [ser],daemon = True)
	# writeThread.start()

	setphase(90)
	setphaseIntervalFlag = True

	try:
		prompt = ''
		while True: 
			s = input("commands: "+prompt)
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
							drawEvent.wait()
							if not signal.drawData(DCBlock = True, maxFreq = 20): 
								break
							time.sleep(0.01)

					except KeyboardInterrupt:
						plt.close('Signal')
						print('Quit drawing')

						# ## stop read signal
						# readEvent.clear()
						# print('Stop Reading Signal')

			elif s.startswith('set'):
				if not readEvent.is_set(): 
					print('readEvent has not set')
					continue
				try: 
					deg = int(input('degrees: '))
					if not setphase(deg):
						print('{} is not a valid direction angle'.format(deg))

				except ValueError:
					print('invalid value')

			elif s.startswith('sweep'):
				if not readEvent.is_set(): 
					print('readEvent has not set')
					continue
				for i in range(10,171,20):
					setphase(i)
					time.sleep(3)

				## stop read
				readEvent.clear()
				print('Stop Reading Signal')


			elif s.startswith('currentdir'):
				print(direction)
			elif s.startswith('info'):
				for ind,val in directionInfo.items():
					print('degrees: {}, (range, velocity): {}'.format(ind, val.RangeAndVelocity()))

			elif s.startswith('resetinfo'):
				for ind,val in directionInfo.items():
					val.reset()
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
