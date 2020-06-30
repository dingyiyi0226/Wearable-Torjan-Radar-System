import matplotlib
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D

def figKwargs(fig, ax, **kwargs):
    if 'title' in kwargs:
        fig.suptitle = kwargs['title']
        
class SigView:
    """ Signal Interface """

    def __init__(self, timeYMax, freqYMax, avgFreqYMax, maxFreq, maxTime, linecolor, figname='Waveform', **kwargs):
    
        self.fig, self.ax = plt.subplots(3, 1, num=figname, figsize=(5,5))
        figKwargs(self.fig, self.ax, **kwargs)

        ## Axis 0: Signal in Time Domain
        self.ax[0].set_xlim(0, maxTime)
        self.ax[0].set_ylim(0, timeYMax)
        self.ax[0].set_xlabel('Time (s)')
        # self.ax[0].ticklabel_format(axis='x', style='sci', scilimits=(0,0), useMathText=True)
        
        ## Axis 1: Signal in Frequency Domain
        self.ax[1].set_xlim(0, maxFreq)
        self.ax[1].set_ylim(0, freqYMax)
        self.ax[1].set_xlabel('Frequency (Hz)')

        ## Axis 2: Signal in Average Frequency Domain
        self.ax[2].set_xlim(0, maxFreq)
        self.ax[2].set_ylim(0, avgFreqYMax)
        self.ax[1].set_xlabel('Frequency (Hz)')
        self.fig.subplots_adjust(hspace=0.3)

        self.timeLine, = self.ax[0].plot([], [], linecolor)
        self.freqLine, = self.ax[1].plot([], [], linecolor)
        self.avgFreqLine, = self.ax[2].plot([], [], linecolor)

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

class ObjView:
    """ Radar Interface """

    def __init__(self, maxR, maxV, figname='Objects'):

        self.fig, self.ax = plt.subplots(1, 1, num=figname, figsize=(5,3))

        self.ax.set_xlim(0, maxR)
        self.ax.set_ylim(-2, maxV)

        self.ax.set_xlabel('Distance (m)')
        self.ax.set_ylabel('Speed (m/s)')

        # self.cmap = plt.get_cmap('magma')
        self.objData = self.ax.scatter([], [], marker='o',s=30, c='m')

        legend_elements = [ Line2D([0], [0], marker='o', color='w', label='915 MHz Frequency Radar',
                            markerfacecolor='r', markersize=8),
                            Line2D([0], [0], marker='o', color='w', label='5.8 GHz Frequency Radar',
                            markerfacecolor='g', markersize=8),]

        self.ax.legend(handles=legend_elements, loc='upper right')


    def figShow(self):
        plt.pause(1)

    def init(self):
        """ Return elements to matplotlib.Animation.FuncAnimation """
        return self.objData,

    def update(self, frame, lowInfo, highInfo):
        """ Return elements to matplotlib.Animation.FuncAnimation """

        rangeData = []
        veloData = []
        colorData = []

        if lowInfo is not None:

            rangeData.extend([i[0] for i in lowInfo])
            veloData.extend([i[1] for i in lowInfo])
            colorData.extend(['r' for i in lowInfo])

        if highInfo is not None:

            rangeData.extend([i[0] for i in highInfo])
            veloData.extend([i[1] for i in highInfo])
            colorData.extend(['g' for i in highInfo])

        self.objData.set_offsets(np.c_[rangeData, veloData])
        self.objData.set_color(colorData)

        return self.objData,

class PPIView:
    """ Radar Interface """

    def __init__(self, maxR, figname='PPI'):
        self.fig = plt.figure(figname)
        self.ax  = self.fig.add_subplot(111, polar=True)
        self.cax = self.fig.add_axes([0.85, 0.1, 0.03, 0.8])

        self.ax.set_theta_zero_location("N")
        self.ax.set_rmax(maxR)
        self.ax.set_rticks(np.arange(0, maxR, maxR//5))

        self.maxV = 25

        self.cmap = plt.get_cmap('magma')
        self.ppiData = self.ax.scatter([], [], marker='o',s=20, c=[], cmap=self.cmap)

        self.cbar = plt.colorbar(self.ppiData, cax=self.cax)
        self.cbar.set_ticks(np.linspace(0, self.maxV, 6)/self.maxV)
        self.cbar.set_ticklabels(np.linspace(0, self.maxV, 6).astype(int))
        self.cbar.set_label('Velocity (m/s)')

    def figShow(self):
        plt.pause(1)

    def init(self):
        """ Return elements to matplotlib.Animation.FuncAnimation """
        return self.ppiData,

    def update(self, frame, infoDict: dict):
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

        self.ppiData.set_offsets(np.c_[directionData, rangeData])
        self.ppiData.set_color([self.cmap(i/self.maxV) for i in veloData])

        return self.ppiData,
