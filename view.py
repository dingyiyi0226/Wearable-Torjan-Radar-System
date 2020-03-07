import matplotlib
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button
import numpy as np

class SigView:

    def __init__(self, maxFreq, maxTime):

        self.fig, self.ax = plt.subplots(3,1, num='fig', figsize=(8,7))

        self.ax[0].set_xlim(0, maxTime)
        # self.ax[0].set_ylim(-0.3, 1)  # for arduino
        self.ax[0].set_ylim(-0.3, 0.3)
        self.ax[0].ticklabel_format(axis='x', style='sci', scilimits=(0,0), useMathText=True)
        
        self.ax[1].set_xlim(0, maxFreq)
        self.ax[1].set_ylim(0, 0.15)

        self.ax[2].set_xlim(0, maxFreq)
        self.ax[2].set_ylim(0, 0.15)

        self.fig.subplots_adjust(left=0.3, hspace=0.3)

        self.timeLine, = self.ax[0].plot([], [])
        self.freqLine, = self.ax[1].plot([], [], 'r')
        self.avgFreqLine, = self.ax[2].plot([], [], 'r')

        self.buttonAx = plt.axes([0.05, 0.05, 0.15, 0.1])
        self.button = Button(self.buttonAx, 'Test', color='0.8', hovercolor='0.6')
        self.button.on_clicked(self.onClick)

    def figShow(self):
        plt.pause(1)

    def init(self):
        # print('initRealTimeSig')
        return self.timeLine, self.freqLine, self.avgFreqLine,

    def update(self, frame, sigDict):
        # print('frame', frame)

        ## if you needs to set ax.x_lim dynamically, blit has to be False

        # self.ax[0].set_xlim(0, sigDict['timeAxis'][-1])
        # self.ax[1].set_xlim(0, sigDict['freqAxis'][-1])
        # self.ax[2].set_xlim(0, sigDict['freqAxis'][-1])

        self.timeLine.set_data(sigDict['timeAxis'], sigDict['timeSig'])
        self.freqLine.set_data(sigDict['freqAxis'], sigDict['freqSig'])
        self.avgFreqLine.set_data(sigDict['freqAxis'], sigDict['avgFreqSig'])

        return self.timeLine, self.freqLine, self.avgFreqLine,

    def onClick(self, event):
        print('click')

class PPIView:

    def __init__(self, maxR):
        self.fig = plt.figure('fig2')
        self.ax = self.fig.add_subplot(111, polar=True)

        self.ax.set_rmax(maxR)
        self.ax.set_rticks(np.arange(0, maxR))

        self.fig.subplots_adjust(left=0.3)

        self.ppiData, = self.ax.plot([], [], 'ro')

        self.buttonAx = plt.axes([0.05, 0.05, 0.15, 0.1])
        self.button = Button(self.buttonAx, 'Testt', color='0.8', hovercolor='0.6')
        self.button.on_clicked(self.onClick)

    def figShow(self):
        plt.pause(1)

    def init(self):
        return self.ppiData,

    def update(self, frame, direction, info):
        direction = direction/180*np.pi
        self.ppiData.set_data([direction for i in info], [i[0] for i in info])
        # self.ppiData.set_markersize(frame%5 + 12)
        # self.ppiData.set_markerfacecolor((1,0,0,0.2*(frame%5)))

        return self.ppiData,

    def onClick(self, event):
        print('click')
