import argparse
import csv
import math
import os
from types import SimpleNamespace

import matplotlib.pyplot as plt
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

def parseCSV(filename, today):
    """ Return signal list and simulation data frequency """

    signal = []

    with open('./rawdata/{}/{}'.format(today, filename)) as file:
        datas = csv.reader(file)
        simFreq = 0
        for ind, data in enumerate(datas):
            if ind==0: continue
            elif ind==1:
                simFreq = 1/float(data[-1])
            else:
                signal.append(float(data[1]))

    return signal, simFreq

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

    for pltind, filename in enumerate(filenames):

        y, fs = parseCSV(filename, today)

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

        yfn = [i*2/N for i in yf]           ## normalization
                                            ## let the amplitude of output signal equals to inputs

        maxFreq = (N // 2) * minFreqDiff
        maxFreqIndex = int(maxFreq / minFreqDiff)

        # Truncated signal
        freqData.append(yfn[:maxFreqIndex])

        # Recording each fs and N
        freqDataFs.append(fs)
        freqDataN.append(N)

    freqDataNp = np.array(freqData).transpose()

    ## Functionaln operation

    if removeBG:
        for i in range(1, freqDataNp.shape[1]):
            freqDataNp[:, i] = freqDataNp[:, i] - freqDataNp[:, 0]
            
        freqDataNp = np.clip(freqDataNp, a_min=0, a_max=None)
        freqDataNp[:, 0] = 0

    if normalizeFreq:
        for i in range(1, freqDataNp.shape[1]):
            freqDataNp[:, i] = freqDataNp[:, i] * (i * 0.25) ** 0.3

    if avgFreq:
        AVGTICK = 3
        fm = 1 / tm
        avgLength = int(fm/minFreqDiff*AVGTICK)
        window = np.ones(avgLength)
        for i in range(1, freqDataNp.shape[1]):
            freqDataNp[:, i] = sg.oaconvolve(freqDataNp[:, i], window/window.sum(), mode='same')

    return freqDataNp, 1 / fs, maxFreq, maxFreqIndex, minFreqDiff

def _plotTheoretical(varibleList, setting, roundup, doPlot=True):
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

            timeDelay = (setting['distance']+setting['distanceOffset'])/3e8
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

def plotSingleFile(today, filename):
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

    max_freq = (len(f_axis)//2)*minFreqDiff
    # max_freq = 5e5
    max_freq_index = int(max_freq/minFreqDiff)

    plt.plot(f_axis[:max_freq_index],yfn[:max_freq_index], 'r')
    peaks, _ = sg.find_peaks(yfn[:max_freq_index], height=peakHeight, prominence=peakProminence)

    plt.plot(peaks*minFreqDiff,[ yfn[i] for i in peaks], 'x')
    peakList = []

    for ind, i in enumerate(peaks):
        plt.annotate(s=int(peaks[ind]*minFreqDiff), xy=(peaks[ind]*minFreqDiff,yfn[i]))
        print('peaks at: {} Hz, amplitude = {}'.format(int(peaks[ind]*minFreqDiff), yfn[i]))
        peakList.append( (int(peaks[ind]*minFreqDiff), yfn[i]) )
    print()

    plt.title('FFT')
    plt.xlabel('freq (Hz)')
    plt.yscale('log')
    plt.ticklabel_format(axis='x', style='sci', scilimits=(0,0), useMathText=True)

    ## Figure 3: Average X[f]

    plt.subplot(313)

    plt.title('average')
    plt.xlabel('freq (Hz)')

    AVGTICK = 3
    assumeFm = 250
    avgLength = int(assumeFm/minFreqDiff*AVGTICK)
    window = np.ones(avgLength)


    # window = sg.gaussian(avgLength, std=int(assumeFm/minFreqDiff))
    avgyfn = sg.oaconvolve(yfn, window/window.sum(), mode='same')

    plt.plot(f_axis[:max_freq_index],avgyfn[:max_freq_index], 'r')

    # ## mark the max value
    # maxIndex = avgyfn[:max_freq_index//2].argmax()
    # plt.plot(f_axis[maxIndex], avgyfn[maxIndex], 'x')
    # plt.annotate(s=int(maxIndex*minFreqDiff), xy=(maxIndex*minFreqDiff,avgyfn[maxIndex]))

    ## mark the peak values
    avgPeaks, _ = sg.find_peaks(avgyfn[:max_freq_index], height=avgPeakHeight, prominence=avgPeakProminence)

    plt.plot(avgPeaks*minFreqDiff,[ avgyfn[i] for i in avgPeaks], 'x')
    avgPeakList = []
    for ind, i in enumerate(avgPeaks):
        plt.annotate(s=int(avgPeaks[ind]*minFreqDiff), xy=(avgPeaks[ind]*minFreqDiff,avgyfn[i]))
        print('avgPeaks at: {} Hz, amplitude = {}'.format(int(avgPeaks[ind]*minFreqDiff), avgyfn[i]))
        avgPeakList.append( (int(avgPeaks[ind]*minFreqDiff), avgyfn[i]) )
    # print()

    plt.subplots_adjust(hspace=0.9)
    plt.show()

def plotMultipleFile(freqDataNp, increment, maxFreq, minFreqDiff, today, filenames, removeBG, normalizeFreq, avgFreq):
    """ plot fft signal for each file """
    
    ## Const

    maxFreqIndex = maxFreq // minFreqDiff

    ## Graph

    fig, ax =  plt.subplots(math.ceil(freqDataNp.shape[1]/3), 3, sharex=False, figsize=(8,7), num='Figure Name')  ## num is **kw_fig

    N = freqDataNp.shape[0]
    t_axis = [i*increment for i in range(N)]
    f_axis = [i*minFreqDiff for i in range(N)]

    for pltind, filename in enumerate(filenames):
        ax[pltind//3, pltind%3].plot(f_axis[:maxFreqIndex], freqDataNp[:, pltind], color='red')
        ax[pltind//3, pltind%3].set_title(filename[:-4]+' cm')
        ax[pltind//3, pltind%3].ticklabel_format(axis='y', style='sci', scilimits=(0,0), useMathText=True)

    title = today
    if removeBG:
        title += ' remove background'
    if normalizeFreq:
        title += ' normalizeFreq'
    if avgFreq:
        title += ' avgFreq'

    fig.suptitle(title)

    fig.subplots_adjust(hspace=0.8)
    plt.show()

def plotExpAndTheo(freqDataNp, increment, maxFreq, minFreqDiff, today, distanceList, setting, roundup, removeBG, avgFreq):
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

    theoFreqList, _ = _plotTheoretical(distanceList, setting, roundup, doPlot=False)

    plt.figure('Figure')

    title = today
    if removeBG:
        title+=' remove background'
    if avgFreq:
        title+=' avgFreq'
    plt.suptitle(title)

    plt.plot(distanceList[1:], freqList[1:], '.-', label='exp')
    plt.plot(distanceList[1:], theoFreqList[1:], '.-', label='theo')
    plt.xlabel('Distance (m)')
    plt.ylabel('Frequency (Hz)')
    plt.legend()
    plt.ticklabel_format(axis='y', style='sci', scilimits=(0,0), useMathText=True)

    plt.grid(True)
    plt.show()

def plotMap(freqDataNp, increment, maxFreq, minFreqDiff, today, distanceList, setting, roundup, removeBG, normalizeFreq, avgFreq):
    """ Plot FFT signal at each distance in heatmap (using NonUniformImage) """

    ## Const

    maxFreqIndex = maxFreq // minFreqDiff

    ## map config

    YTICKCNT = 16
    CMAP = 'gray'
    INTERP = 'nearest'

    ## Generate Figure

    fig, ax = plt.subplots(1,2, num='Figure', figsize=(8,4), constrained_layout=True)

    title = today
    if removeBG:
        title += ' remove background'
    if normalizeFreq:
        title += ' normalizeFreq'
    if avgFreq:
        title += ' avgFreq'

    fig.suptitle(title)

    ## Figure 1: Experiment

    im = NonUniformImage(ax[0], extent=(0,0,0,0), cmap=CMAP, interpolation=INTERP)
    im.set_data(distanceList, np.arange(maxFreqIndex), freqDataNp)
    ax[0].images.append(im)

    ax[0].set_title('Experiment')
    # ax[0].set_xticks(distanceList[::1])
    ax[0].set_xlim(0, distanceList[-1])

    ax[0].set_ylabel('Frequency (Hz)')
    ax[0].set_yticks(np.linspace(0, maxFreqIndex, YTICKCNT))
    # ax[0].set_yticklabels(np.linspace(0, max_freq, YTICKCNT, dtype=int))
    ax[0].set_ylim(0, maxFreqIndex)

    ax[0].tick_params(right=True, left=False, labelleft=False)

    if setting['varible'] == 'd':
        ax[0].set_xlabel('Distance (m)')
    elif setting['varible'] == 'v':
        ax[0].set_xlabel('Velocity (m/s)')

    ## Figure 2: Experiment with Theoritical Line

    im = NonUniformImage(ax[1], extent=(0,0,0,0), cmap=CMAP, interpolation=INTERP)
    im.set_data(distanceList, np.arange(maxFreqIndex), freqDataNp)
    ax[1].images.append(im)

    ax[1].set_title('Theoretical')

    # ax[1].set_xticks(distanceList[::1])
    ax[1].set_xlim(0, distanceList[-1])

    ax[1].set_yticks(np.linspace(0,maxFreqIndex,YTICKCNT))
    ax[1].set_yticklabels(np.linspace(0, maxFreq, YTICKCNT, dtype=int))
    ax[1].set_ylim(0, maxFreqIndex)

    if setting['varible'] == 'd':
        ax[1].set_xlabel('Distance (m)')
    elif setting['varible'] == 'v':
        ax[1].set_xlabel('Velocity (m/s)')

    theoF1List, theoF2List = _plotTheoretical(distanceList, setting, roundup, doPlot=False)
    ax[1].plot(distanceList, [i // minFreqDiff for i in theoF1List], '.:m')
    ax[1].plot(distanceList, [i // minFreqDiff for i in theoF2List], '.:r')

    ## Figure legned

    plt.colorbar(im, ax=ax.ravel().tolist())
    plt.show()

    return

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--removeBG', action='store_true')
    parser.add_argument('-a', '--averageFreq', action='store_true')
    parser.add_argument('-n', '--normalizeFreq', action='store_true')
    args = parser.parse_args()

    ## Settings definitions at plotTheoretical() documentation

    DELAYLINE = 10 * (2 ** 0.5)
    SETUPLINE = 1 * (2.24 ** 0.5)

    ## Tide up setting

    todaySetting = {
        'BW': 100e6, 
        'tm': 2000e-6, 
        'delayTmRatio': 0, 
        'distanceOffset': SETUPLINE,
        'freq': 5.8e9, 
        'varible': 'd', 
        'distance': 1, 
        'velo': 0,
    }

    ## Load files

    today = '0225hornfm05'
    filenames = [i for i in  os.listdir('./rawdata/{}/'.format(today)) if i.endswith('1.csv')]
    filenames.sort()
    variableList = [float(i[:-5]) / 100 for i in filenames]

    freqDataNp, increment, maxFreq, maxFreqIndex, minFreqDiff = loadBatchCSV(
        filenames, today, todaySetting['tm'], args.removeBG, args.normalizeFreq, args.averageFreq)    

    ## Modify setting

    todaySetting['simTime'] = freqDataNp.shape[0] * 2 * increment

    ## Show Config

    print()
    print('=====================Config======================')

    for key, value in todaySetting.items():
        print("{:16} {:>32}".format(key, value))

    for key, value in vars(args).items():
        print("{:16} {:>32}".format(key, value))

    print('=====================Config======================')
    print()

    ## Plot Files

    # plotSingleFile(today, '100202.csv')
    # plotMultipleFile(freqDataNp, increment, maxFreq, maxFreqIndex, minFreqDiff, today, filenames, 
    #     removeBG=args.removeBG, normalizeFreq=args.normalizeFreq, avgFreq=args.averageFreq)

    plotExpAndTheo(freqDataNp, increment, maxFreq, maxFreqIndex, minFreqDiff, today, filenames, variableList, setting=todaySetting,
        roundup=True, removeBG=args.removeBG, avgFreq=args.averageFreq)

    # plotMap(freqDataNp, increment, maxFreq, maxFreqIndex, minFreqDiff, today, variableList, setting=todaySetting,
    #     roundup=True, removeBG=args.removeBG, normalizeFreq=args.normalizeFreq, avgFreq=args.averageFreq)

    return 

if __name__ == '__main__':
    main()
