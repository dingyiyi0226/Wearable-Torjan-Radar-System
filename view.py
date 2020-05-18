import matplotlib
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.widgets import Button


class SigView:
    """ Signal Interface """

    def __init__(self, maxFreq, maxTime, figname='Waveform'):

        self.fig, self.ax = plt.subplots(3,1, num=figname)

        ## Axis 0: Signal in Time Domain
        self.ax[0].set_xlim(0, maxTime)
        self.ax[0].set_ylim(-10, 10)
        self.ax[0].ticklabel_format(axis='x', style='sci', scilimits=(0,0), useMathText=True)
        self.ax[0].set_xlabel('Time (s)')
        
        ## Axis 1: Signal in Frequency Domain
        self.ax[1].set_xlim(0, maxFreq)
        self.ax[1].set_ylim(-0.002, 0.1)
        self.ax[1].set_xlabel('Frequency (Hz)')

        ## Axis 2: Signal in Average Frequency Domain
        self.ax[2].set_xlim(0, maxFreq)
        self.ax[2].set_ylim(-0.0002, 0.01)

        self.fig.subplots_adjust(hspace=0.3)

        self.timeLine, = self.ax[0].plot([], [])
        self.freqLine, = self.ax[1].plot([], [], 'r')
        self.avgFreqLine, = self.ax[2].plot([], [], 'r')

    def figShow(self):
        plt.pause(1)

    def init(self):
        """ Return elements to matplotlib.Animation.FuncAnimation """
        return self.timeLine, self.freqLine, self.avgFreqLine,

    def update(self, frame, sigDict):
        """ Return elements to matplotlib.Animation.FuncAnimation """
        # print('frame', frame)

        ## if you needs to set ax.x_lim dynamically, blit has to be False

        # self.ax[0].set_xlim(0, sigDict['timeAxis'][-1])
        # self.ax[1].set_xlim(0, sigDict['freqAxis'][-1])
        # self.ax[2].set_xlim(0, sigDict['freqAxis'][-1])

        self.timeLine.set_data(sigDict['timeAxis'], sigDict['timeSig'])
        self.freqLine.set_data(sigDict['freqAxis'], sigDict['freqSig'])
        self.avgFreqLine.set_data(sigDict['freqAxis'], sigDict['processedSig'])

        return self.timeLine, self.freqLine, self.avgFreqLine,

class PPIView:
    """ Radar Interface """

    def __init__(self, maxR, figname='PPI'):
        self.fig = plt.figure(figname)
        self.ax = self.fig.add_subplot(111, polar=True)

        self.ax.set_rmax(maxR)
        self.ax.set_rticks(np.arange(0, maxR, maxR//5))

        self.ppiData, = self.ax.plot([], [], '.r')

    def figShow(self):
        plt.pause(1)

    def init(self):
        """ Return elements to matplotlib.Animation.FuncAnimation """
        return self.ppiData,

    def update(self, frame, infoDict):
        """ Return elements to matplotlib.Animation.FuncAnimation """

        directionData = []
        rangeData = []
        veloData = []

        for direction, info in infoDict.items():
            # info type: [(range, velo),]
            if info is not None:
                directionData.extend([direction/180*np.pi for i in info])
                rangeData.extend([i[0] for i in info])
                veloData.extend([i[1] for i in info])

        self.ppiData.set_data(directionData, rangeData)

        # self.ppiData.set_markersize(frame%5 + 12)
        # self.ppiData.set_markerfacecolor((1,0,0,0.2*(frame%5)))

        return self.ppiData,
