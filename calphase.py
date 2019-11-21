import csv
import numpy as np
import time

## --------------------- Phase data --------------------- ##

class Phase:
	def __init__(self):

		self._phasePins = [5,11,22,45,90,180]
		self._phase2pin = {}
		self._pin2phase = {}
		self._phaseList = []
		self._phaseRawData = {}

		tmp = 0
		for i in range(64):
			if i&1: tmp += self._phasePins[0]
			if i&2: tmp += self._phasePins[1] 
			if i&4: tmp += self._phasePins[2] 
			if i&8: tmp += self._phasePins[3] 
			if i&16: tmp += self._phasePins[4] 
			if i&32: tmp += self._phasePins[5] 
			self._phaseList.append(tmp)
			self._pin2phase[i] = tmp
			self._phase2pin[tmp] = i
			# print(i, tmp)
			tmp = 0
		self._phaseList.append(360)

		with open('phasedata/phase.csv', newline = '') as file:
			di = csv.DictReader(file)
			for row in di:

				self._phaseRawData[ float(row['assume beam dir']) ] = \
						(float(row['actual beam dir']),float(row['element1 phase']), float(row['element2 phase']),
						 float(row['element3 phase']), float(row['element4 phase']))
				# print(float(row['assume beam dir']), self._phaseRawData.get(float(row['assume beam dir'])) )

	@property
	def phasePins(self):
		return self._phasePins

	@property
	def phaseList(self):
		return self._phaseList

	def phase2pin(self,phase):
		if phase not in self._phases: 
			print('phase not available')
			return None
		return self._phase2pin.get(phase)

	def pin2phase(self, pin):
		if pin > 63 or pin < 0:
			print('pin not available')
			return None
		return self._pin2phase.get(pin)

	def isValidDirection(self, theta):
		return theta >= 10 and theta <= 170

	def getEachPhase(self, theta):
		if not self.isValidDirection(theta):
			print('{} is not a valid direction'.format(theta))
			return None
		return self._phaseRawData.get(theta)[1:]

	def getEachPhase2Pin(self, theta):
		if not self.isValidDirection(theta):
			print('{} is not a valid direction'.format(theta))
			return None
		tmp = self._phaseRawData.get(theta)
		el1 = self._phase2pin.get(tmp[1])
		el2 = self._phase2pin.get(tmp[2])
		el3 = self._phase2pin.get(tmp[3])
		el4 = self._phase2pin.get(tmp[4])
		return (el1, el2, el3, el4)

	def getActualDirection(self, theta):
		if not self.isValidDirection(theta):
			print('{} is not a valid direction'.format(theta))
			return None
		return self._phaseRawData.get(theta)[0]

## ------------------------------------------------------ ##


## -------------- For Generating csv File --------------- ##

## find the nearest phases
def findPhase(phases, phase):
	dis = 10000
	ans = 0
	for p in phases:
		if dis>abs(phase-p):
			dis = abs(phase-p)
			ans = p
		else: break
	return ans

def calRealDir(p1,p2,p3,p4):
	numer = {}  ##{ dirction angle (0-180): electric field}

	for thetad in np.arange(0,180.01,0.1):
		q2 = (p2-p1) - 0.5*np.cos(thetad/180.*np.pi)*360
		q3 = (p3-p1) - 0.5*np.cos(thetad/180.*np.pi)*360*2
		q4 = (p4-p1) - 0.5*np.cos(thetad/180.*np.pi)*360*3
		tmpE = 1 + np.cos(q2/180.*np.pi) + np.cos(q3/180.*np.pi) + np.cos(q3/180.*np.pi)
		numer[thetad] = tmpE
		# print(thetad, tmpE)

	
	direction = max(numer,key=numer.get)
	field = numer.get(direction)
	return direction, field

def main():

	# phase = Phase()

	# print(findPhase(phase.phaseList,222))

	# print(phase.getEachPhase2Pin(33))



	# print(phase.calRealDir(0,180,353,180))
	
	
	### Generate direction to each element phase 
	phasepins = [5,11,22,45,90,180]

	phases = []
	tmp = 0
	for i in range(64):
		if i&1: tmp += phasepins[0]
		if i&2: tmp += phasepins[1] 
		if i&4: tmp += phasepins[2] 
		if i&8: tmp += phasepins[3] 
		if i&16: tmp += phasepins[4] 
		if i&32: tmp += phasepins[5] 
		phases.append(tmp)
		# print(i, tmp)
		tmp = 0
	phases.append(360)
	print(phases)
	
	tmprow = []
	el1=el2=el3=el4 = 0.

	

	with open('phasedata/phase_o.csv', 'w', newline='') as file:
		writer = csv.writer(file)
		writer.writerow(['distance (# lamda)','beam direction','element1 phase','element2 phase','element3 phase','element4 phase'])
		for i in range(0,181,1):
			for j in np.arange(0,180.1,0.5):
				tmprow.append(0.5)
				tmprow.append(i)

				el1 = j
				el2 = j+0.5*np.cos(i*np.pi/180)*2*180
				el3 = j+0.5*np.cos(i*np.pi/180)*2*180*2
				el4 = j+0.5*np.cos(i*np.pi/180)*2*180*3

				if el2 < 0: el2+=360
				while el3 < 0: el3+=360
				while el3 > 360: el3-=360
				while el4 < 0: el4+=360
				while el4 > 360: el4-=360

				tmprow += [el1,el2,el3,el4]
				
				writer.writerow(tmprow)
				tmprow = []
	

	time.sleep(1)
	print('--------------------------')
	### Find each element's accruate phase 
	
	tmprow = []
	el1=el2=el3=el4 = 0.
	caldir = {}

	with open('phasedata/phase_o.csv', newline = '') as file:
		# rows = csv.reader(file)
		di = csv.DictReader(file)
		# for r in rows:
		# 	print(r)
		
		# for i in di:
		# 	print(i['beam direction'], i['element2 phase'])

		with open('phasedata/phase_raw.csv', 'w', newline='') as file:
			writer = csv.writer(file)
			writer.writerow(['distance (# lamda)','assume beam dir','actual beam dir','element1 phase','element2 phase','element3 phase','element4 phase'])
			for i in di:
				# print(i['beam direction'], i['element2 phase'])
				el1 = float(i['element1 phase']) 
				el2 = float(i['element2 phase']) 
				el3 = float(i['element3 phase'])
				el4 = float(i['element4 phase'])

				if el1 < 0: el1+=360
				if el2 < 0: el2+=360
				if el3 < 0: el3+=360
				if el4 < 0: el4+=360

				el1 = findPhase(phases,el1)
				el2 = findPhase(phases,el2)
				el3 = findPhase(phases,el3)
				el4 = findPhase(phases,el4)

				if caldir.get( (el1,el2,el3,el4) ) is None:
					direction, field = calRealDir(el1,el2,el3,el4)
					caldir[(el1,el2,el3,el4)] = (direction,field)
				else:
					direction, field = caldir.get( (el1,el2,el3,el4))

				writer.writerow([0.5, i['beam direction'],direction,el1,el2,el3,el4 ])
	time.sleep(1)
	print('--------------------------')
	"""
	###  choose the best phase of element 1
	"""
	tmprow = []
	el1=el2=el3=el4 = 0.

	with open('phasedata/phase_raw.csv', newline = '') as file:

		di = csv.DictReader(file)

		with open('phasedata/phase.csv', 'w', newline='') as file:
			writer = csv.writer(file)
			writer.writerow(['distance (# lamda)','assume beam dir','actual beam dir','element1 phase','element2 phase','element3 phase','element4 phase'])
		
			assume_ang = 0.
			min_diff = 200.
			assume_dir = 0.
			actual_dir = 0.

			for i in di:

				if assume_ang != float(i['assume beam dir']):
					# print(i['assume beam dir'])
					writer.writerow([0.5, assume_dir, actual_dir, el1, el2, el3, el4])
					assume_ang = float(i['assume beam dir'])
					min_diff = abs(assume_ang-float(i['actual beam dir']))
					assume_dir = float(i['assume beam dir'])
					actual_dir = float(i['actual beam dir'])
					el1 = float(i['element1 phase'])
					el2 = float(i['element2 phase'])
					el3 = float(i['element3 phase'])
					el4 = float(i['element4 phase'])

				else: 
					if min_diff > abs(assume_ang-float(i['actual beam dir'])):
						min_diff = abs(assume_ang-float(i['actual beam dir']))
						assume_dir = float(i['assume beam dir'])
						actual_dir = float(i['actual beam dir'])
						el1 = float(i['element1 phase'])
						el2 = float(i['element2 phase'])
						el3 = float(i['element3 phase'])
						el4 = float(i['element4 phase'])
			writer.writerow([0.5, assume_dir, actual_dir, el1, el2, el3, el4])
	
if __name__ == '__main__':
	main()

