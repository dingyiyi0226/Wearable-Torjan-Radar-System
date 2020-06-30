import argparse
import csv
import math
import os
from types import SimpleNamespace

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import random as rd
import scipy.signal as sg

from matplotlib.image import NonUniformImage
# from simulation.process import averageSignalFrequency

def roundto(num, tick):
    if (num/tick - num//tick) < 0.5:
        return math.floor(num/tick) * tick
    else:
        return math.ceil(num/tick) * tick

def angV2V(rpsList, radius):
    return [i*2*np.pi*radius*0.01 for i in rpsList]

def parseCSV(filename, today, start=0, end=1):
    """ Return signal list and simulation data frequency """

    signal = []

    with open('./rawdata/{}/{}'.format(today, filename)) as file:
        datas = csv.reader(file)
        row = next(datas)

        if len(row) == 1:      # new format
            
            simFreq = 16e3
            channel = 1 # 0~7
            for data in datas:
                if len(data[channel]):   # omit the last line
                    signal.append(float(data[channel])/1e6)

        else:
            row = next(datas)
            simFreq = 1/float(row[-1])
            for data in datas:
                signal.append(float(data[1]))
    # signal = signal[5000:]
    N = len(signal)
    sig_len = N//end

    N_sta = int(N*start)
    N_end = N_sta+sig_len
    if N_end > N:
        N_end = N
        N_sta = N_end - sig_len

    return signal[N_sta:N_end], simFreq

def loadBatchCSV(filenames, today, tm, removeBG: bool, normalizeFreq: bool, avgFreq: bool):
    """
    Parameters
    ----------
    filenames : list
        names of the file

    today : str
        name of the folder

    Return
    ------
    freqDataNp : np.array

    tm : float

    maxFreq :

    maxFreqIndex :

    minFreqDiff :

    Raise
    -----
    ValueError
    """

    freqData = []
    freqDataFs = []
    freqDataN = []

    if len(filenames)==1:
        filename = filenames[0]
        SPLIT_NUM = 1
        for ind in range(SPLIT_NUM):
            y, fs = parseCSV(filename, today, start=ind/SPLIT_NUM, end=SPLIT_NUM)

            N = len(y)                          ## number of simulation data points
            minFreqDiff = fs / N                ## spacing between two freqencies on axis

            print('-----------------------------')
            print('read {} file'.format(filename))
            print('N =', N)
            print('fs =', fs)
            print('minFreqDiff =', minFreqDiff)

            # t_axis = [i/fs for i in range(N)]
            # f_axis = [i*minFreqDiff for i in range(N)]

            yf = abs(np.fft.fft(y))
            # yfs = np.fft.fftshift(yf)         ## shift 0 frequency to middle
                                                ## [0,1,2,3,4,-4,-3,-2,-1] -> [-4,-3,-2,-1,0,1,2,3,4]
                                                ## (-fs/2, fs/2)
                                                ## just plot the positive frequency, so dont need to shift

            yfn = [i/N for i in yf]             ## normalization
                                                ## let the amplitude of output signal equals to inputs

            maxFreq = (N // 2) * minFreqDiff
            maxFreqIndex = int(maxFreq / minFreqDiff)

            # Truncated signal
            freqData.append(yfn[:maxFreqIndex])

            # Recording each fs and N
            freqDataFs.append(fs)
            freqDataN.append(N)

    else:
        for pltind, filename in enumerate(filenames):

            y, fs = parseCSV(filename, today)

            N = len(y)                          ## number of simulation data points
            minFreqDiff = fs / N                ## spacing between two freqencies on axis

            if pltind==0:
                maxFreq = (N // 2) * minFreqDiff
                maxFreqIndex = int(maxFreq / minFreqDiff)

            print('-----------------------------')
            print('read {} file'.format(filename))
            print('N =', N)
            print('fs =', fs)
            print('minFreqDiff =', minFreqDiff)

            # t_axis = [i/fs for i in range(N)]
            # f_axis = [i*minFreqDiff for i in range(N)]

            yf = abs(np.fft.fft(y))
            # yfs = np.fft.fftshift(yf)         ## shift 0 frequency to middle
                                                ## [0,1,2,3,4,-4,-3,-2,-1] -> [-4,-3,-2,-1,0,1,2,3,4]
                                                ## (-fs/2, fs/2)
                                                ## just plot the positive frequency, so dont need to shift

            yfn = [i/N for i in yf]             ## normalization
                                                ## let the amplitude of output signal equals to inputs


            # Truncated signal
            freqData.append(yfn[:maxFreqIndex])
            # print(np.shape(freqData))
            # print(minFreqDiff)

            # Recording each fs and N
            freqDataFs.append(fs)
            freqDataN.append(N)

    freqDataNp = np.array(freqData).transpose()

    ## Functionaln operation

    if removeBG:
        for i in range(1, freqDataNp.shape[1]):
            freqDataNp[:, i] = freqDataNp[:, i] - freqDataNp[:, 0]
            
        freqDataNp = np.clip(freqDataNp, a_min=0, a_max=None)
        # freqDataNp[:, 0] = 0

    if normalizeFreq:
        for i in range(1, freqDataNp.shape[1]):
            freqDataNp[:, i] = freqDataNp[:, i] * (i * 0.25) ** 0.3

    if avgFreq:

        fm = 1 / tm
        BW = fm * 2
        avgLength = max(1, int(BW/minFreqDiff))
        # window = np.ones(avgLength)
        window=sg.blackman(avgLength)
        for i in range(1, freqDataNp.shape[1]):
            freqDataNp[:, i] = sg.oaconvolve(freqDataNp[:, i], window/window.sum(), mode='same')
    # freqDataNp = np.clip(freqDataNp ,0, 5e-7)

    return freqDataNp, 1 / fs, maxFreq, maxFreqIndex, minFreqDiff

def plotTheoretical(varibleList, setting, roundup, doPlot=True):
    """ plot threoretical frequency

        settings:
       
           ↑
         _ |         /\                 /\
         ↑ |      /\/  \             /\/  \
         | |     / /\   \           / /\   \
         B |    / /  \   \         / /  \
         W |   / /    \   \       / /    \
         | |  / /      \   \_____/_/      \
         ↓ | /          \_______/          \
         ¯ + -------------------------->
             |<-- tm -->|<-tm2->|
             |<----   simTime   ---->|

        delayTmRatio = tm2 / tm
    """

    assert(setting['varible'] == 'd' or setting['varible'] == 'v')

    fm = 1/setting['tm']
    slope = setting['BW']/setting['tm']*2
    freqRes = 1/setting['simTime']

    print('===================================')
    print('fm:', fm)
    print('freqRes:', freqRes)
    print('===================================')

    f1List = []
    f2List = []

    if setting['varible']=='d':
        for distance in varibleList:

            distance*=2

            timeDelay = (distance+setting['distanceOffset'])/3e8
            beatFreq = timeDelay*slope
            doppFreq = setting['velo']*2/3e8*setting['freq']

            f1 = beatFreq+doppFreq
            f2 = abs(beatFreq-doppFreq)

            f1RoundUp = roundto(roundto(f1, fm/(setting['delayTmRatio']+1)), freqRes)
            f2RoundUp = roundto(roundto(f2, fm/(setting['delayTmRatio']+1)), freqRes)
            if not roundup:
                # print('f1', f1)
                # print('f2', f2)
                f1List.append(f1)
                f2List.append(f2)
            else:
                # print('f1RoundUp', f1RoundUp)
                # print('f2RoundUp', f2RoundUp)
                f1List.append(f1RoundUp)
                f2List.append(f2RoundUp)

    elif setting['varible']=='v':
        for velocity in varibleList:

            timeDelay = (2*setting['distance']+setting['distanceOffset'])/3e8
            beatFreq = timeDelay*slope
            doppFreq = velocity*2/3e8*setting['freq']

            # print('beatFreq:', beatFreq)
            # print('doppFreq:', doppFreq)

            f1 = beatFreq+doppFreq
            f2 = abs(beatFreq-doppFreq)

            f1RoundUp = roundto(roundto(f1, fm), freqRes)
            f2RoundUp = roundto(roundto(f2, fm), freqRes)

            if not roundup:
                # print('f1', f1)
                # print('f2', f2)
                f1List.append(f1)
                f2List.append(f2)
            else:
                # print('f1RoundUp', f1RoundUp)
                # print('f2RoundUp', f2RoundUp)
                f1List.append(f1RoundUp)
                f2List.append(f2RoundUp)
                
    if doPlot:

        plt.figure('Figure')

        if setting['varible']=='d':
            plt.suptitle('velo: {} m/s'.format(setting['velo']))
        elif setting['varible']=='v':
            plt.suptitle('distance: {} m'.format(setting['distance']))

        plt.plot(varibleList, f1List, '.:m')
        plt.plot(varibleList, f2List, '.:r')

        if setting['varible']=='d':
            plt.xlabel('Distance (m)')
        elif setting['varible']=='v':
            plt.xlabel('Velocity (m/s)')

        plt.ylabel('Frequency (Hz)')
        plt.xticks(varibleList)
        plt.ticklabel_format(axis='y', style='sci', scilimits=(0,0), useMathText=True)
        plt.grid(True)
        plt.show()

    return f1List, f2List

def plotSingleFile(today, filename, maxFreq=None, removeDC=False):
    """ Plot time domain signal and spectrum """

    peakHeight = 0.02
    peakProminence = 0.005
    avgPeakHeight = 0.02
    avgPeakProminence = 0.005

    ## Load signle CSV file.

    y, fs = parseCSV(filename, today)

    N = len(y)                          ## number of simulation data points
    minFreqDiff = fs/N                  ## spacing between two freqencies on axis
    
    print('-----------------------------')
    print('read {} file'.format(filename))
    print('N =', N)
    print('fs =', fs)
    print('minFreqDiff =',minFreqDiff)

    t_axis = [i/fs for i in range(N)]
    f_axis = [i*minFreqDiff for i in range(N)]

    yf = abs(np.fft.fft(y))
    # yfs = np.fft.fftshift(yf)         ## shift 0 frequency to middle
                                        ## [0,1,2,3,4,-4,-3,-2,-1] -> [-4,-3,-2,-1,0,1,2,3,4]
                                        ## (-fs/2, fs/2)
                                        ## just plot the positive frequency, so dont need to shift

    yfn = [i*2/N for i in yf]           ## normalization
                                        ## let the amplitude of output signal equals to inputs

    if removeDC:
        yfn[0] = 0

    ## Figure 1: x(t)

    plt.figure('Figure')
    plt.suptitle(today)

    plt.subplot(311)
    plt.plot(t_axis, y)
    plt.title('Signal of '+filename[:-4]+' cm')
    plt.xlabel('time (s)')
    plt.ticklabel_format(axis='x', style='sci', scilimits=(0,0), useMathText=True)

    ## Figure 2: X[f]

    plt.subplot(312)

    if maxFreq is None:
        maxFreq = (len(f_axis)//2)*minFreqDiff
    maxFreqIndex = int(maxFreq/minFreqDiff)

    plt.plot(f_axis[:maxFreqIndex],yfn[:maxFreqIndex], 'r')
    peaks, _ = sg.find_peaks(yfn[:maxFreqIndex], height=peakHeight, prominence=peakProminence)

    plt.plot(peaks*minFreqDiff,[ yfn[i] for i in peaks], 'x')
    peakList = []

    for ind, i in enumerate(peaks):
        plt.annotate(s=int(peaks[ind]*minFreqDiff), xy=(peaks[ind]*minFreqDiff,yfn[i]))
        print('peaks at: {} Hz, amplitude = {}'.format(int(peaks[ind]*minFreqDiff), yfn[i]))
        peakList.append( (int(peaks[ind]*minFreqDiff), yfn[i]) )

    plt.title('FFT')
    plt.xlabel('freq (Hz)')
    # plt.yscale('log')
    plt.ticklabel_format(axis='x', style='sci', scilimits=(0,0), useMathText=True)

    ## Figure 3: Average X[f]

    plt.subplot(313)

    plt.title('average')
    plt.xlabel('freq (Hz)')

    AVGTICK = 3
    assumeFm = 250
    avgLength = max(1, int(assumeFm/minFreqDiff*AVGTICK))
    window = np.ones(avgLength)


    # window = sg.gaussian(avgLength, std=int(assumeFm/minFreqDiff))
    avgyfn = sg.oaconvolve(yfn, window/window.sum(), mode='same')

    plt.plot(f_axis[:maxFreqIndex],avgyfn[:maxFreqIndex], 'r')

    # ## mark the max value
    # maxIndex = avgyfn[:maxFreqIndex//2].argmax()
    # plt.plot(f_axis[maxIndex], avgyfn[maxIndex], 'x')
    # plt.annotate(s=int(maxIndex*minFreqDiff), xy=(maxIndex*minFreqDiff,avgyfn[maxIndex]))

    ## mark the peak values
    avgPeaks, _ = sg.find_peaks(avgyfn[:maxFreqIndex], height=avgPeakHeight, prominence=avgPeakProminence)

    plt.plot(avgPeaks*minFreqDiff,[ avgyfn[i] for i in avgPeaks], 'x')
    avgPeakList = []
    for ind, i in enumerate(avgPeaks):
        plt.annotate(s=int(avgPeaks[ind]*minFreqDiff), xy=(avgPeaks[ind]*minFreqDiff,avgyfn[i]))
        print('avgPeaks at: {} Hz, amplitude = {}'.format(int(avgPeaks[ind]*minFreqDiff), avgyfn[i]))
        avgPeakList.append( (int(avgPeaks[ind]*minFreqDiff), avgyfn[i]) )
    # print()

    plt.subplots_adjust(hspace=0.9)
    plt.show()

def plotMultipleFile(freqDataNp, increment, maxFreq, minFreqDiff, today, filenames, oneColumn, removeBG, normalizeFreq, avgFreq):
    """ plot fft signal for each file """
    
    ## Const

    maxFreqIndex = int(maxFreq / minFreqDiff)

    ## Graph



    # if freqDataNp.shape[1] <= 3:
    #     fig, ax =  plt.subplots(max(2, freqDataNp.shape[1]), 1, sharex=False, figsize=(8,7), num='Figure Name')  ## num is **kw_fig
    # else:
        # fig, ax =  plt.subplots(math.ceil(freqDataNp.shape[1]/3), 3, sharex=False, figsize=(8,7), num='Figure Name')  ## num is **kw_fig
    if oneColumn:
        fig, ax =  plt.subplots(freqDataNp.shape[1], 1, sharex=False, figsize=(5,7), num='Figure Name')  ## num is **kw_fig
    else:
        fig, ax =  plt.subplots(math.ceil(freqDataNp.shape[1]/2), 2, sharex=False, figsize=(5,7), num='Figure Name')  ## num is **kw_fig
  

    N = freqDataNp.shape[0]
    t_axis = [i*increment for i in range(N)]
    f_axis = [i*minFreqDiff for i in range(N)]

    title = today

    if len(filenames)==1:
        title += ' {}'.format(filenames[0])
        for pltind in range(freqDataNp.shape[1]):

            currentAxes = ax[pltind]

            # currentAxes = ax[pltind] if  else ax[pltind//3, pltind%3]
            # currentAxes.set_yscale('log')

            currentAxes.plot(f_axis[:maxFreqIndex], freqDataNp[:maxFreqIndex, pltind], color='red')
            currentAxes.set_title('{}/{}'.format(pltind, freqDataNp.shape[1]))
            currentAxes.ticklabel_format(axis='y', style='sci', scilimits=(0,0), useMathText=True)

            maxIndex = freqDataNp[:maxFreqIndex, pltind].argmax()
            currentAxes.plot(f_axis[maxIndex], freqDataNp[maxIndex, pltind], 'x')
            currentAxes.annotate(s=int(maxIndex*minFreqDiff), xy=(maxIndex*minFreqDiff,freqDataNp[maxIndex, pltind]))
    else:
        for pltind, filename in enumerate(filenames):


            if oneColumn:
                currentAxes = ax[pltind]
            else:
                if pltind >= math.ceil(freqDataNp.shape[1]/2):
                    currentAxes = ax[pltind%math.ceil(freqDataNp.shape[1]/2),1]
                else:
                    currentAxes = ax[pltind,0]


            currentAxes.plot(f_axis[:maxFreqIndex], freqDataNp[:maxFreqIndex, pltind], color='red')
            # currentAxes.set_ylim((-5e-5, 1e-2))

            # currentAxes.set_title(filename[:-4]+' cm')
            # currentAxes.set_yscale('log')
            # currentAxes.ticklabel_format(axis='y', style='sci', scilimits=(0,0), useMathText=True)


    if removeBG:
        title += ' remove background'
    if normalizeFreq:
        title += ' normalizeFreq'
    if avgFreq:
        title += ' avgFreq'

    fig.suptitle(title)

    fig.subplots_adjust(hspace=0.8)
    plt.show()

def plotExpAndTheo(freqDataNp, increment, maxFreq, minFreqDiff, today, variableList, setting, roundup, removeBG, avgFreq):
    """ Plot experimental peak value and theoretical value for distance resolution verification """

    freqList = []
    refyfn=[]

    # Truncated unwant line (DC and bugs line)
    tmp = freqDataNp.shape[0] // 2
    freqDataNp[tmp:] = 0
    freqDataNp[0] = 0
    
    N = freqDataNp.shape[0]
    t_axis = [i*increment for i in range(N)]
    f_axis = [i*minFreqDiff for i in range(N)]
    
    for pltind in range(freqDataNp.shape[1]):
        peakIndex = np.argmax(freqDataNp[:, pltind])
        freqList.append(f_axis[peakIndex])

    theoFreqList, _ = plotTheoretical(variableList, setting, roundup, doPlot=False)

    plt.figure('Figure',figsize=(6,8))

    title = today
    if removeBG:
        title+=' remove background'
    if avgFreq:
        title+=' avgFreq'
    plt.suptitle(title)

    plt.plot(variableList[1:], freqList[1:], '.-', label='exp')
    plt.plot(variableList[1:], theoFreqList[1:], '.-', label='theo')

    if setting['varible'] == 'd':
        plt.xlabel('Distance (m)')
    elif setting['varible'] == 'v':
        plt.xlabel('Velocity (m/s)')

    plt.ylabel('Frequency (Hz)')
    plt.legend()
    plt.ticklabel_format(axis='y', style='sci', scilimits=(0,0), useMathText=True)

    plt.grid(True)
    plt.show()

def plotMap(freqDataNp, increment, maxFreq, minFreqDiff, today, variableList, setting, roundup, removeBG, normalizeFreq, avgFreq):
    """ Plot FFT signal at each distance in heatmap (using NonUniformImage) """

    ## Const

    maxFreqIndex = int(maxFreq / minFreqDiff)

    ## map config

    YTICKCNT = 11
    CMAP = 'Greys'
    INTERP = 'nearest'

    ## Generate Figure

    fig, ax = plt.subplots(1,2, num='Figure', figsize=(9,4), constrained_layout=True)

    title = today
    if removeBG:
        title += ' remove background'
    if normalizeFreq:
        title += ' normalizeFreq'
    if avgFreq:
        title += ' avgFreq'

    fig.suptitle(title)

    if removeBG:
        freqDataNp[:, 0]=0

    ## Figure 1: Experiment

    im = NonUniformImage(ax[0], extent=(0,0,0,0), cmap=CMAP, interpolation=INTERP)
    im.set_data(variableList[1:], np.arange(maxFreqIndex), freqDataNp[:maxFreqIndex, 1:])
    ax[0].images.append(im)

    if setting['varible'] == 'd':
        ax[0].set_title('Distance Detection')
    else:
        ax[0].set_title('Speed Detection')
    # ax[0].set_xticks(variableList[::1])
    ax[0].set_xlim(0, variableList[-1])

    ax[0].set_ylabel('Frequency (Hz)')
    ax[0].set_yticks(np.linspace(0, maxFreqIndex, YTICKCNT))
    ax[0].set_yticklabels(np.linspace(0, maxFreq, YTICKCNT, dtype=int))
    ax[0].set_ylim(0, maxFreqIndex)

    # ax[0].tick_params(right=True, left=False, labelleft=False)

    if setting['varible'] == 'd':
        ax[0].set_xlabel('Distance (m)')
    elif setting['varible'] == 'v':
        ax[0].set_xlabel('Angular Velocity (rps)')

    ## Figure 2: Experiment with Theoritical Line

    im = NonUniformImage(ax[1], extent=(0,0,0,0), cmap=CMAP, interpolation=INTERP)
    im.set_data(variableList[1:], np.arange(maxFreqIndex), freqDataNp[:maxFreqIndex, 1:])
    ax[1].images.append(im)

    if setting['varible'] == 'd':
        ax[1].set_title('5.8 GHz Radar')
    else:
        ax[1].set_title('5.8 GHz Radar')

    ax[1].set_xticks(range(0, 20, 2))
    ax[1].set_xlim(0, variableList[-1])

    ax[1].set_ylabel('Frequency (Hz)')
    ax[1].set_yticks(np.linspace(0,maxFreqIndex,YTICKCNT))
    ax[1].set_yticklabels(np.linspace(0, maxFreq, YTICKCNT, dtype=int))
    ax[1].set_ylim(0, maxFreqIndex)

    if setting['varible'] == 'd':
        ax[1].set_xlabel('Distance (m)')

        theoF1List, theoF2List = plotTheoretical(variableList, setting, roundup, doPlot=False)
        ax[1].plot(variableList, [i // minFreqDiff for i in theoF1List], ':m')
        ax[1].plot(variableList, [i // minFreqDiff for i in theoF2List], ':r', label='theoretical line')
        ax[1].legend()

    elif setting['varible'] == 'v':
        ax[1].set_xlabel('Velocity (m/s)')

        # veloList = angV2V(variableList, radius=7)
        # theoF1List, theoF2List = plotTheoretical(veloList, setting, roundup, doPlot=False)
        # ax[1].plot(variableList, [i // minFreqDiff for i in theoF1List], '.:r', alpha=1, label='r=7')
        # ax[1].plot(variableList, [i // minFreqDiff for i in theoF2List], '.:r', alpha=0.5)

        # veloList = angV2V(variableList, radius=14)
        veloList = variableList
        theoF1List, theoF2List = plotTheoretical(veloList, setting, roundup, doPlot=False)
        ax[1].plot(variableList, [i // minFreqDiff for i in theoF1List], ':r', alpha=1, label='theoretical line')
        ax[1].plot(variableList, [i // minFreqDiff for i in theoF2List], ':r', alpha=1)

        # veloList = angV2V(variableList, radius=10)
        # theoF1List, theoF2List = plotTheoretical(veloList, setting, roundup, doPlot=False)
        # ax[1].plot(variableList, [i // minFreqDiff for i in theoF1List], '.:g', alpha=1, label='r=10')
        # ax[1].plot(variableList, [i // minFreqDiff for i in theoF2List], '.:g', alpha=0.5)

        ax[1].legend()

    ## Figure legend

    cbar = plt.colorbar(im, ax=ax.ravel().tolist())
    cbar.set_label(' Normalized Amplitude (dB)')
    plt.show()

    return

def plot3DMap(freqDataNp, increment, maxFreq, minFreqDiff, today, variableList, setting, roundup, removeBG, normalizeFreq, avgFreq):
    """ Plot FFT signal at each distance in heatmap (using NonUniformImage) """

    ## Const

    maxFreqIndex = int(maxFreq / minFreqDiff)

    ## map config

    YTICKCNT = 16
    CMAP = 'gray'
    INTERP = 'nearest'

    ## Generate Figure

    # fig, ax = plt.subplots(1,2, num='Figure', figsize=(4,2), constrained_layout=True)
    fig = plt.figure()
    ax = Axes3D(fig)


    title = today
    if removeBG:
        title += ' remove background'
    if normalizeFreq:
        title += ' normalizeFreq'
    if avgFreq:
        title += ' avgFreq'

    fig.suptitle(title)

    if removeBG:
        freqDataNp[:, 0]=0

    ## Figure 1: Experiment

    # im = NonUniformImage(ax[0], extent=(0,0,0,0), cmap=CMAP, interpolation=INTERP)
    print(np.shape(variableList))
    print(np.arange(maxFreqIndex).shape)
    print(freqDataNp[:maxFreqIndex].shape)

    X, Y = np.meshgrid(variableList, np.arange(maxFreqIndex))

    ax.plot_surface(X, Y, freqDataNp[:maxFreqIndex], cmap=plt.get_cmap('rainbow'))
    ax.set_zlim(-1e-7, 2e-5)
    # ax.images.append(im)

    ax.set_title('Experiment')


    ax.set_ylabel('Frequency (Hz)')


    if setting['varible'] == 'd':
        ax.set_xlabel('Distance (m)')
    elif setting['varible'] == 'v':
        ax.set_xlabel('Angular Velocity (rps)')

    plt.show()

    return

def rcsPlot(variableList, freqDataNp):

    peaks = freqDataNp.max(axis=0)
    angles = np.array(variableList) / 360. * 2 * np.pi

    print(freqDataNp.shape)
    print(peaks.shape)

    ax = plt.subplot(111, projection='polar')

    ax.set_title('Reflected Power (dB) @ 5.8 GHz')
    ax.plot(angles, peaks, 'g')

    ax.set_theta_zero_location("N")

    ax.set_rticks(np.arange(-40, 0, 10))
    ax.set_rlabel_position(-77)

    ax.grid(True)
    plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--removeBG', action='store_true')
    parser.add_argument('-a', '--averageFreq', action='store_true')
    parser.add_argument('-n', '--normalizeFreq', action='store_true')
    args = parser.parse_args()

    ## Settings definitions at plotTheoretical() documentation

    DELAYLINE = 10 * (2.24 ** 0.5)
    SETUPLINE = 0.6 * (2.24 ** 0.5)

    ## Tide up setting

    todaySetting = {
        'BW': 100e6,
        'tm': 4000e-6 * 2,
        'delayTmRatio': 0,
        'distanceOffset': SETUPLINE,
        'freq': 5800e6,
        'varible': 'v',
        'distance': 3.4,
        'velo': 0,
    }

    ## Load files

    today = '0628_rcs_2_tmp'
    # filenames = [i for i in  os.listdir('./rawdata/{}/'.format(today)) if i.endswith('1.csv')]
    filenames = [i for i in  os.listdir('./rawdata/{}/'.format(today)) if i.endswith('1.csv') and i.startswith('high')]
    filenames.sort()
    filenames = filenames[:-1]
    # filenames = [filenames[0], ] + filenames[:20]
    filenames = filenames[:-1] + [filenames[1], ]
    print('filenames:', filenames)

    if todaySetting['varible'] == 'd':
        variableList = [float(i[-9:-5]) / 100 for i in filenames]
    else:
        variableList = [int(i[-7:-5]) for i in filenames]

    variableList = [int(i[-8:-5]) for i in filenames]
    print('variableList:', variableList)

    freqDataNp, increment, maxFreq, maxFreqIndex, minFreqDiff = loadBatchCSV(
        filenames, today, todaySetting['tm'], args.removeBG, args.normalizeFreq, args.averageFreq)

    # bgfreqDataNp, _, _, _, _ = loadBatchCSV(
    #     ['200001.csv',], today, todaySetting['tm'], args.removeBG, args.normalizeFreq, args.averageFreq)

    ## Modify setting

    todaySetting['simTime'] = freqDataNp.shape[0] * 2 * increment

    ## Show Config

    print()
    print('=====================Config=====================')

    for key, value in todaySetting.items():
        print("{:16}{:>32}".format(key, value))

    for key, value in vars(args).items():
        print("{:16}{:>32}".format(key, value))

    print('=====================Config=====================')
    print()

    ## freqDataNp adjustment by HAND

    print('freqdata shape: ', freqDataNp.shape)
    freqDataNp[0] = 0   # removeDC
    # print(np.max(freqDataNp))

    # freqDataNp /= np.max(freqDataNp, axis=0)
    freqDataNp /= np.max(freqDataNp)
    freqDataNp = 20 * np.log10(freqDataNp + 1e-2)
    # freqDataNp[freqDataNp>0.005] = 0
    # freqDataNp = freqDataNp.clip(0, 0.001)
    maxFreq = 2000
    # minFreqDiff = 4.4


    ## Plot Files

    # plotSingleFile(today, 'low-200201.csv', maxFreq=5000, removeDC=False)

    rcsPlot(variableList[1:], freqDataNp[:, 1:])

    # plotMultipleFile(freqDataNp, increment, maxFreq, minFreqDiff, today, filenames, oneColumn=False,
    #     removeBG=args.removeBG, normalizeFreq=args.normalizeFreq, avgFreq=args.averageFreq)

    # veloList = angV2V(variableList, radius=14)
    # plotMap(freqDataNp, increment, maxFreq, minFreqDiff, today, veloList, setting=todaySetting,
    #     roundup=False, removeBG=args.removeBG, normalizeFreq=args.normalizeFreq, avgFreq=args.averageFreq)

    # plotExpAndTheo(freqDataNp, increment, maxFreq, minFreqDiff, today, filenames, variableList, setting=todaySetting,
    #     roundup=True, removeBG=args.removeBG, avgFreq=args.averageFreq)

    # plotTheoretical([i for i in np.arange(1, 6, 0.25)], todaySetting, roundup=False, doPlot=True)

    return

if __name__ == '__main__':
    main()
