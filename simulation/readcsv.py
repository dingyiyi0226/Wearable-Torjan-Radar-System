import csv
import math
import os

import matplotlib.pyplot as plt
import numpy as np
import random as rd
import scipy.signal as sg

def roundto(num, tick):
    if (num/tick - num//tick) < 0.5:
        return math.floor(num/tick) * tick
    else:
        return math.ceil(num/tick) * tick

def readcsv(filename, today):
    """return signal list and simulation data frequency"""

    signal = []
    # today = '0225fm05'
    with open('./rawdata/{}/{}'.format(today, filename)) as file:
        datas = csv.reader(file)
        simFreq = 0
        for ind, data in enumerate(datas):
            if ind==0: continue
            elif ind==1:
                simFreq = 1/float(data[-1])
                # simFreq = 1/(float(data[3])*3)
            # elif ind%3!=2: continue
            else:
                signal.append(float(data[1]))
    # print(len(signal))
    return signal, simFreq

def plotSingleFile(today, filename):
    """ plot time domain signal and fft signal """

    peakHeight = 0.02
    peakProminence = 0.005
    avgPeakHeight = 0.02
    avgPeakProminence = 0.005

    y, fs = readcsv(filename, today)
    print('-----------------------------')
    print('read {} file'.format(filename))
    N = len(y)                          ## number of simulation data points
    minFreqDiff = fs/N                ## spacing between two freqencies on axis
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

    plt.figure('Figure')
    plt.suptitle(today)

    plt.subplot(311)
    plt.plot(t_axis, y)
    plt.title('Signal of '+filename[:-4]+' cm')
    plt.xlabel('time (s)')
    plt.ticklabel_format(axis='x', style='sci', scilimits=(0,0), useMathText=True)

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
    plt.ticklabel_format(axis='x', style='sci', scilimits=(0,0), useMathText=True)

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

    """

    plt.figure('Figure2')
    plt.suptitle('after filter')

    plt.subplot(211)

    yfnFilter = []
    for ind, i in enumerate(yfn):
        if ind<len(yfn)//4 or ind>len(yfn)//4*3:
            yfnFilter.append(i)
        else:
            yfnFilter.append(0)

    plt.plot(f_axis[:max_freq_index],yfnFilter[:max_freq_index])
    plt.xlabel('freq (Hz)')
    plt.ticklabel_format(axis='x', style='sci', scilimits=(0,0), useMathText=True)

    sigFilter = np.fft.ifft(yfnFilter)

    plt.subplot(212)
    plt.plot(t_axis, sigFilter)
    plt.xlabel('time (s)')
    plt.ticklabel_format(axis='x', style='sci', scilimits=(0,0), useMathText=True)

    plt.subplots_adjust(hspace=0.5)
    plt.show()
    """

def plotMultipleFile(today, filenames, removeBG, normalizeFreq, avgFreq):
    """ plot fft signal for each file """

    fig, ax =  plt.subplots(math.ceil(len(filenames)/3), 3, sharex=False, figsize=(8,7), num='Figure Name')  ## num is **kw_fig

    peakHeight = 5e-2
    peakProminence = 1e-3

    refyfn=[]

    for pltind, filename in enumerate(filenames):

        y, fs = readcsv(filename, today)
        print('-----------------------------')
        print('read {} file'.format(filename))
        N = len(y)                          ## number of simulation data points
        minFreqDiff = fs/N                ## spacing between two freqencies on axis
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

        max_freq = (len(f_axis)//2)*minFreqDiff
        max_freq = 10e3
        max_freq_index = int(max_freq/minFreqDiff)

        ## remove background signal

        offsetyfn = yfn

        if removeBG:
            if pltind==0:
                refyfn=yfn
            else:
                # offsetyfn = [yfn[i]-refyfn[i] for i in range(max_freq_index)]
                offsetyfn = [max(0, yfn[i]-refyfn[i]) for i in range(max_freq_index)]

        ## normalize siganl amplitude for different distance

        normalizeyfn = offsetyfn

        if normalizeFreq:
            normalizeyfn = [i*(pltind*0.25)**1 for i in offsetyfn]

        ## take average of frequencies

        avgyfn = normalizeyfn

        if avgFreq:
            AVGTICK = 3
            assumeFm = 250
            avgLength = int(assumeFm/minFreqDiff*AVGTICK)
            window = np.ones(avgLength)

            # window = np.ones(int(minFreqDiff*10))  # for velocity cases
            # window = sg.gaussian(avgLength, std=int(assumeFm/minFreqDiff))
            avgyfn = sg.oaconvolve(normalizeyfn, window/window.sum(), mode='same')

        ax[pltind//3, pltind%3].plot(f_axis[:max_freq_index],avgyfn[:max_freq_index], color='red')

        if avgFreq:
            maxIndex = avgyfn[:max_freq_index//2].argmax()
            # maxIndex = avgyfn[:max_freq_index].argmax()
            ax[pltind//3, pltind%3].plot(f_axis[maxIndex], avgyfn[maxIndex], 'x')
            ax[pltind//3, pltind%3].annotate(s=int(maxIndex*minFreqDiff), xy=(maxIndex*minFreqDiff,avgyfn[maxIndex]))

        else:
            peaks, _ = sg.find_peaks(avgyfn[:max_freq_index], height=peakHeight, prominence=peakProminence)

            ax[pltind//3, pltind%3].plot(peaks*minFreqDiff,[ normalizeyfn[i] for i in peaks], 'x')
            peakList = []
            for ind, i in enumerate(peaks):
                ax[pltind//3, pltind%3].annotate(s=int(peaks[ind]*minFreqDiff), xy=(peaks[ind]*minFreqDiff,normalizeyfn[i]))
                print('peaks at: {} Hz, amplitude = {}'.format(int(peaks[ind]*minFreqDiff), normalizeyfn[i]))
                peakList.append( (int(peaks[ind]*minFreqDiff), normalizeyfn[i]) )

        ax[pltind//3, pltind%3].set_title(filename[:-4]+' cm')
        # ax[pltind//3, pltind%3].tick_params(bottom=False, labelbottom=False)
        ax[pltind//3, pltind%3].ticklabel_format(axis='y', style='sci', scilimits=(0,0), useMathText=True)

        # if removeBG:
        #     ax[pltind//3, pltind%3].set_ylim((0, 0.05))
        # else:
        #     ax[pltind//3, pltind%3].set_ylim((0, 0.1))

    title = today
    if removeBG:
        title+=' remove background'
    if normalizeFreq:
        title+=' normalizeFreq'
    if avgFreq:
        title+=' avgFreq'

    fig.suptitle(title)

    fig.subplots_adjust(hspace=0.8)
    plt.show()

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

    fm = 1/setting['tm']
    slope = setting['BW']/setting['tm']*2
    freqRes = 1/setting['simTime']

    print('fm', fm)
    print('freqRes', freqRes)

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

    else:
        print('ughhhhhhh')

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

def plotExpAndTheo(today, filenames, distanceList, setting, roundup, removeBG, avgFreq):
    """ plot experimental peak value and theoretical value """

    freqList = []
    refyfn=[]

    for pltind, filename in enumerate(filenames):

        y, fs = readcsv(filename, today)
        print('-----------------------------')
        print('read {} file'.format(filename))
        N = len(y)                          ## number of simulation data points
        minFreqDiff = fs/N                  ## spacing between two freqencies on axis
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

        ## remove aliasing of image frequency and DC

        for i in range(len(yfn)//4,len(yfn)):
            # print(i)
            yfn[i]=0
        yfn[0]=0

        max_freq = (len(f_axis)//2)*minFreqDiff
        # max_freq = 5e5
        max_freq_index = int(max_freq/minFreqDiff)

        ## remove background signal

        offsetyfn = yfn

        if removeBG:
            if pltind==0:
                refyfn=yfn
            else:
                # offsetyfn = [yfn[i]-refyfn[i] for i in range(max_freq_index)]
                offsetyfn = [max(0, yfn[i]-refyfn[i]) for i in range(max_freq_index)]

        if avgFreq:
            AVGTICK = 3
            fm = 1/setting['tm']
            avgLength = int(fm/minFreqDiff*AVGTICK)
            window = np.ones(avgLength)
            # window = sg.gaussian(avgLength, std=int(fm/minFreqDiff))
            avgyfn = sg.oaconvolve(offsetyfn, window/window.sum(), mode='same')
            freqList.append(f_axis[avgyfn[:max_freq_index].argmax()])
        else:
            freqList.append(f_axis[offsetyfn.index(max(offsetyfn[:max_freq_index]))])


    print('freqList:', freqList)

    theoFreqList, _ = plotTheoretical(distanceList, setting, roundup, doPlot=False)

    plt.figure('Figure')

    title = today
    if removeBG:
        title+=' remove background'
    if avgFreq:
        title+=' avgFreq'
    plt.suptitle(title)

    plt.plot(distanceList[1:], freqList[1:], '.-', label='exp')
    plt.plot(distanceList[1:], theoFreqList[1:], '.-', label='theo')
    # plt.scatter(distanceList, freqList, label='exp')
    # plt.scatter(distanceList, theoFreqList, label='theo')
    plt.xlabel('Distance (m)')
    plt.ylabel('Frequency (Hz)')
    plt.xticks(distanceList[::3])
    plt.legend()
    plt.ticklabel_format(axis='y', style='sci', scilimits=(0,0), useMathText=True)

    plt.grid(True)
    plt.show()

def plotHeatmap(today, filenames, distanceList, setting, roundup, removeBG, normalizeFreq, avgFreq):
    """ plot fft signal at each distance in heatmap """

    freqData = []
    refyfn=[]

    # heatmapXWidth=600//len(distanceList)  ## pixels of x axis

    for pltind, filename in enumerate(filenames):

        y, fs = readcsv(filename, today)
        print('-----------------------------')
        print('read {} file'.format(filename))
        N = len(y)                          ## number of simulation data points
        minFreqDiff = fs/N                  ## spacing between two freqencies on axis
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


        max_freq = (len(f_axis)//2)*minFreqDiff
        max_freq = 15000
        max_freq_index = int(max_freq/minFreqDiff)
        heatmapXWidth=max_freq_index//len(distanceList)  ## pixels of x axis

        ## remove background signal

        offsetyfn = yfn
        if removeBG:
            if pltind==0:
                refyfn=yfn
                offsetyfn = [0 for i in offsetyfn]
            else:
                # offsetyfn = [yfn[i]-refyfn[i] for i in range(max_freq_index)]
                offsetyfn = [max(0, yfn[i]-refyfn[i]) for i in range(max_freq_index)]

        ## normalize siganl amplitude for different distance

        normalizeyfn = offsetyfn

        if normalizeFreq:
            normalizeyfn = [i*(pltind*0.25)**0.3 for i in offsetyfn]

        ## take average of frequencies

        avgyfn = normalizeyfn

        if avgFreq:
            AVGTICK = 3
            fm = 1/setting['tm']
            avgLength = int(fm/minFreqDiff*AVGTICK)
            window = np.ones(avgLength)
            # window = sg.gaussian(avgLength, std=int(fm/minFreqDiff*0.5))
            avgyfn = sg.oaconvolve(normalizeyfn, window/window.sum(), mode='same')


        for i in range(heatmapXWidth):
            freqData.append(np.flip(avgyfn[:max_freq_index]))

    # print(freqList)
    freqDataNp = np.array(freqData).transpose()

    xtickPos = [i for i in range(heatmapXWidth*len(distanceList)) if i%heatmapXWidth==heatmapXWidth//2]

    XTICKSAMPLE = 1
    YTICKCNT = 16

    fig, ax = plt.subplots(1,2, num='Figure', figsize=(10,5))

    title = today
    if removeBG:
        title+=' remove background'
    if normalizeFreq:
        title+=' normalizeFreq'
    if avgFreq:
        title+=' avgFreq'

    fig.suptitle(title)

    ax[0].set_title('Experiment')
    ax[0].imshow(freqDataNp, cmap='gray')

    ax[0].set_xticks(xtickPos[::XTICKSAMPLE])
    ax[0].set_xticklabels(distanceList[::XTICKSAMPLE])

    ax[0].set_ylabel('Frequency (Hz)')
    ax[0].set_yticks(np.linspace(0,max_freq_index,YTICKCNT))
    ax[0].set_yticklabels(np.flip(np.linspace(0, max_freq, YTICKCNT, dtype=int)))
    ax[0].set_ylim((max_freq_index, 0))

    ax[0].tick_params(right=True, left=False, labelleft=False)


    im = ax[1].imshow(freqDataNp, cmap='gray')

    ax[1].set_title('Theoretical')


    if setting['varible']=='d':
        ax[0].set_xlabel('Distance (m)')
        ax[1].set_xlabel('Distance (m)')
    elif setting['varible']=='v':
        ax[0].set_xlabel('Velocity (m/s)')
        ax[1].set_xlabel('Velocity (m/s)')

    ax[1].set_xticks(xtickPos[::XTICKSAMPLE])
    ax[1].set_xticklabels(distanceList[::XTICKSAMPLE])
    ax[1].set_yticks(np.linspace(0,max_freq_index,YTICKCNT))
    ax[1].set_yticklabels(np.flip(np.linspace(0, max_freq, YTICKCNT, dtype=int)))

    theoF1List, theoF2List = plotTheoretical(distanceList, setting, roundup, doPlot=False)
    ax[1].plot(xtickPos, [(max_freq-i)//minFreqDiff for i in theoF1List], '.:m')
    ax[1].plot(xtickPos, [(max_freq-i)//minFreqDiff for i in theoF2List], '.:r')
    ax[1].set_ylim((max_freq_index, 0))


    plt.subplots_adjust(wspace=0.25)
    plt.colorbar(im, ax=ax.ravel().tolist(), shrink=0.7)
    plt.show()


def main():

    ## Settings definitions at plotTheoretical() documentation

    DELAYLINE = 10*2**0.5
    SETUPLINE = 1*2.24**0.5

    today = '0312h'
    todaySetting = {'BW':15e6, 'tm':607e-6, 'delayTmRatio':1, 'simTime':24e-3, 'distanceOffset':SETUPLINE,
                    'freq':915e6, 'varible':'d', 'distance':1, 'velo':0}


    filenames = [i for i in  os.listdir('./rawdata/{}/'.format(today)) if i.endswith('3.csv')]
    filenames.sort()

    # filenames = filenames[:12]

    variableList = [float(i[:-5])/100 for i in filenames]
    # variableList = [0,10,12,14, 16, 18, 20, 22]
    # variableList = [0,8,11,14,16]

    # variableList = [i-2 for i in variableList]
    # variableList[0] = 0
    # variableList = np.arange(0, 5, 0.25)


    # plotSingleFile(today, '2001.csv')
    # plotMultipleFile(today, filenames, removeBG=False, normalizeFreq=False, avgFreq=False)

    # plotTheoretical(variableList, setting=todaySetting, roundup=True)

    # plotExpAndTheo(today, filenames, variableList, setting=todaySetting,
    #                roundup=True, removeBG=False, avgFreq=False)

    plotHeatmap(today, filenames, variableList, setting=todaySetting,
                roundup=True, removeBG=False, normalizeFreq=False, avgFreq=False)


if __name__ == '__main__':
    main()
