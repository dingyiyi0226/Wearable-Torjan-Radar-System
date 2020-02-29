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
                simFreq = 1/float(data[3])
                # simFreq = 1/(float(data[3])*3)
            # elif ind%3!=2: continue
            else:
                signal.append(float(data[1]))
    # print(len(signal))
    return signal, simFreq

def plotSingleFile(today, filename):
    """ plot time domain signal and fft signal """

    y, fs = readcsv(filename, today)
    print('-----------------------------')
    print('read {} file'.format(filename))
    N = len(y)                          ## number of simulation data points
    min_freq_diff = fs/N                ## spacing between two freqencies on axis
    print('N =', N)
    print('fs =', fs)
    print('min_freq_diff =',min_freq_diff)

    t_axis = [i/fs for i in range(N)]
    f_axis = [i*min_freq_diff for i in range(N)]

    yf = abs(np.fft.fft(y))
    # yfs = np.fft.fftshift(yf)         ## shift 0 frequency to middle
                                        ## [0,1,2,3,4,-4,-3,-2,-1] -> [-4,-3,-2,-1,0,1,2,3,4]
                                        ## (-fs/2, fs/2)
                                        ## just plot the positive frequency, so dont need to shift

    yfn = [i*2/N for i in yf]           ## normalization
                                        ## let the amplitude of output signal equals to inputs

    plt.figure('Figure')
    plt.suptitle(today)

    plt.subplot(211)
    plt.plot(t_axis, y)
    plt.title('Signal of '+filename+' cm')
    plt.xlabel('time (s)')
    plt.ticklabel_format(axis='x', style='sci', scilimits=(0,0), useMathText=True)

    plt.subplot(212)

    max_freq = (len(f_axis)//2)*min_freq_diff
    # max_freq = 5e5
    max_freq_index = int(max_freq/min_freq_diff)

    plt.plot(f_axis[:max_freq_index],yfn[:max_freq_index], 'r')
    peaks, _ = sg.find_peaks(yfn[:max_freq_index], height = 0.01)

    plt.plot(peaks*min_freq_diff,[ yfn[i] for i in peaks], 'x')
    peakList = []
    for ind, i in enumerate(peaks):
        plt.annotate(s=int(peaks[ind]*min_freq_diff), xy=(peaks[ind]*min_freq_diff,yfn[i]))
        print('peaks at: {} Hz, amplitude = {}'.format(int(peaks[ind]*min_freq_diff), yfn[i]))
        peakList.append( (int(peaks[ind]*min_freq_diff), yfn[i]) )

    plt.title('FFT')
    plt.xlabel('freq (Hz)')
    plt.ticklabel_format(axis='x', style='sci', scilimits=(0,0), useMathText=True)

    plt.subplots_adjust(hspace=0.5)
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

def plotMultipleFile(today, filenames):
    """ plot fft signal for each file """

    fig, ax =  plt.subplots(math.ceil(len(filenames)/3), 3, sharex=False, num='Figure Name')  ## num is **kw_fig

    refyfn=[]

    # remove background
    removeBG = False

    for pltind, filename in enumerate(filenames):

        y, fs = readcsv(filename, today)
        print('-----------------------------')
        print('read {} file'.format(filename))
        N = len(y)                          ## number of simulation data points
        min_freq_diff = fs/N                ## spacing between two freqencies on axis
        print('N =', N)
        print('fs =', fs)
        print('min_freq_diff =',min_freq_diff)

        t_axis = [i/fs for i in range(N)]
        f_axis = [i*min_freq_diff for i in range(N)]

        yf = abs(np.fft.fft(y))
        # yfs = np.fft.fftshift(yf)         ## shift 0 frequency to middle
                                            ## [0,1,2,3,4,-4,-3,-2,-1] -> [-4,-3,-2,-1,0,1,2,3,4]
                                            ## (-fs/2, fs/2)
                                            ## just plot the positive frequency, so dont need to shift

        yfn = [i*2/N for i in yf]           ## normalization
                                            ## let the amplitude of output signal equals to inputs

        max_freq = (len(f_axis)//2)*min_freq_diff
        # max_freq = 5e5
        max_freq_index = int(max_freq/min_freq_diff)

        offsetyfn = yfn

        if removeBG:
            if pltind==0:
                refyfn=yfn
            else:
                offsetyfn = [yfn[i]-refyfn[i] for i in range(max_freq_index)]

        # print(ax[pltind])

        ax[pltind//3, pltind%3].plot(f_axis[:max_freq_index],offsetyfn[:max_freq_index], color='red')
        peaks, _ = sg.find_peaks(offsetyfn[:max_freq_index], height = 0.02)

        # ax[pltind//3, pltind%3].plot(peaks*min_freq_diff,[ offsetyfn[i] for i in peaks], marker='x')
        peakList = []
        for ind, i in enumerate(peaks):
            ax[pltind//3, pltind%3].annotate(s=int(peaks[ind]*min_freq_diff), xy=(peaks[ind]*min_freq_diff,offsetyfn[i]))
            print('peaks at: {} Hz, amplitude = {}'.format(int(peaks[ind]*min_freq_diff), offsetyfn[i]))
            peakList.append( (int(peaks[ind]*min_freq_diff), offsetyfn[i]) )

        ax[pltind//3, pltind%3].set_title(filename+' cm')
        # ax[pltind//3, pltind%3].ticklabel_format(axis='x', style='sci', scilimits=(0,0), useMathText=True)

        if removeBG:
            ax[pltind//3, pltind%3].set_ylim((-0.03, 0.1))
        else:
            ax[pltind//3, pltind%3].set_ylim((0, 0.1))

    if removeBG:
        fig.suptitle(today+' remove background')
    else:
        fig.suptitle(today)
    fig.subplots_adjust(hspace=0.6)
    plt.show()

def plotTheoretical(distanceList, distanceOffset, BW, tm, simTime, roundup, doPlot=True):
    """ plot threoretical frequency
       
         _ ↑
         ↑ |      /\/\        /\/\
         | |     / /\ \      / /\ \
         B |    / /  \ \    / /  \ \
         W |   / /    \ \  / /
         | |  / /      \ \/ /
         ↓ | / /        \/\/
         ¯ + -------------------------->
             |<-- tm -->|
             |<----  simTime  ---->|

    """

    fm = 1/tm
    slope = BW/tm*2
    freqRes = 1/simTime

    print('fm', fm)
    print('freqRes', freqRes)

    freqList = []

    for distance in distanceList:

        distance*=2

        timeDelay = (distance+distanceOffset)/3e8
        beatFreq = timeDelay*slope
        fbRoundUp = roundto(roundto(beatFreq, fm), freqRes)
        if not roundup:
            # print('beatFreq', beatFreq)
            freqList.append(beatFreq)
        else:
            # print('fbRoundUp', fbRoundUp)
            freqList.append(fbRoundUp)

    if doPlot:

        plt.figure('Figure')

        plt.plot(distanceList, freqList)
        # plt.scatter(distanceList, freqList)
        plt.xlabel('Distance (m)')
        plt.ylabel('Frequency (Hz)')
        plt.xticks(distanceList)
        plt.ticklabel_format(axis='y', style='sci', scilimits=(0,0), useMathText=True)

        plt.show()

    return freqList

def plotExpAndTheo(today, filenames, distanceList, distanceOffset, BW, tm, simTime, roundup):
    """ plot experimental value and theoretical value """

    freqList = []
    refyfn=[]

    # remove background
    removeBG = True

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

        ## remove aliasing of image frequency and DC

        for i in range(len(yfn)//4,len(yfn)):
            # print(i)
            yfn[i]=0
        yfn[0]=0

        max_freq = (len(f_axis)//2)*minFreqDiff
        # max_freq = 5e5
        max_freq_index = int(max_freq/minFreqDiff)

        offsetyfn = yfn

        if removeBG:
            if pltind==0:
                refyfn=yfn
            else:
                offsetyfn = [yfn[i]-refyfn[i] for i in range(max_freq_index)]


        freqList.append(f_axis[offsetyfn.index(max(offsetyfn[:max_freq_index]))])

    print('freqList:', freqList)

    theoFreqList = plotTheoretical(distanceList, distanceOffset, BW, tm, simTime, roundup, doPlot=False)

    plt.figure('Figure')
    plt.title(today)
    plt.plot(distanceList, freqList, label='exp')
    plt.plot(distanceList, theoFreqList, label='theo')
    # plt.scatter(distanceList, freqList, label='exp')
    # plt.scatter(distanceList, theoFreqList, label='theo')
    plt.xlabel('Distance (m)')
    plt.ylabel('Frequency (Hz)')
    plt.xticks(distanceList)
    plt.legend()
    plt.ticklabel_format(axis='y', style='sci', scilimits=(0,0), useMathText=True)

    plt.show()

def plotHeatmap(today, filenames, distanceList, distanceOffset, BW, tm, simTime, roundup):
    """ plot fft signal at each distance in heatmap """

    freqData = []
    refyfn=[]

    heatmapXWidth=600//len(distanceList)  ## pixels of x axis

    # remove background
    removeBG = False

    # normalize frequency amplitude
    normalizeFreq = True

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
        # max_freq = 5e5
        max_freq_index = int(max_freq/minFreqDiff)

        offsetyfn = yfn

        if removeBG:
            if pltind==0:
                refyfn=yfn
            else:
                offsetyfn = [yfn[i]-refyfn[i] for i in range(max_freq_index)]

        normalizeyfn = offsetyfn

        if normalizeFreq:
            normalizeyfn = [i*(pltind*0.25)**1 for i in offsetyfn]

        for i in range(heatmapXWidth):
            freqData.append(np.flip(normalizeyfn[:max_freq_index]))

    # print(freqList)
    freqDataNp = np.array(freqData).transpose()

    theoFreqList = plotTheoretical(distanceList, distanceOffset, BW, tm, simTime, roundup, doPlot=False)

    xtickPos = [i for i in range(heatmapXWidth*len(distanceList)) if i%heatmapXWidth==heatmapXWidth//2]
    xtickSample = 3
    ytickcnt = 16

    fig, ax=  plt.subplots(1,2, num='Figure', figsize=(10,5))

    if removeBG:
        fig.suptitle(today+' remove background')
    else:
        fig.suptitle(today)

    ax[0].set_title('Experiment')
    ax[0].imshow(freqDataNp, cmap='gray')

    ax[0].set_xlabel('Distance (m)')
    ax[0].set_xticks(xtickPos[::xtickSample])
    ax[0].set_xticklabels(distanceList[::xtickSample])

    ax[0].set_ylabel('Frequency (Hz)')
    ax[0].set_yticks(np.linspace(0,max_freq_index,ytickcnt))
    ax[0].set_yticklabels(np.flip(np.linspace(0, max_freq, ytickcnt, dtype=int)))

    ax[0].tick_params(right=True, left=False, labelleft=False)


    im = ax[1].imshow(freqDataNp, cmap='gray')

    ax[1].set_title('Theoretical')
    ax[1].set_xlabel('Distance (m)')
    ax[1].set_xticks(xtickPos[::xtickSample])
    ax[1].set_xticklabels(distanceList[::xtickSample])
    ax[1].set_yticks(np.linspace(0,max_freq_index,ytickcnt))
    ax[1].set_yticklabels(np.flip(np.linspace(0, max_freq, ytickcnt, dtype=int)))

    theoFreqList = plotTheoretical(distanceList, distanceOffset, BW, tm, simTime, roundup, doPlot=False)
    ax[1].plot(xtickPos, [(max_freq-i)//minFreqDiff for i in theoFreqList], 'r')


    plt.subplots_adjust(wspace=0.25)
    plt.colorbar(im, ax=ax.ravel().tolist(), shrink=0.7)
    plt.show()


def main():

    today = '0225fm05'

    filenames = [i for i in  os.listdir('./rawdata/{}/'.format(today)) if i.endswith('2.csv')]
    filenames.sort()

    distanceList = [float(i[:-5])/100 for i in filenames]

    # plotSingleFile(today, '4252')
    plotMultipleFile(today, filenames)

    # plotTheoretical(distanceList, distanceOffset=10*2.24**0.5,
    #                 BW=99.9969e6, tm=4096e-6, simTime=24e-3, roundup=True)

    # plotExpAndTheo(today, filenames, distanceList, distanceOffset=10*2.24**0.5,
    #                BW=99.9969e6, tm=4096e-6, simTime=24e-3, roundup=True)

    # plotHeatmap(today, filenames, distanceList, distanceOffset=10*2.24**0.5,
    #             BW=99.9969e6, tm=4096e-6, simTime=24e-3, roundup=True)


if __name__ == '__main__':
    main()
