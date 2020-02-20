import csv
import matplotlib.pyplot as plt
import numpy as np
import random as rd
import scipy.signal as sg

def test():

    N = 2000
    T = 10

    print('fs =', N/T)

    t = np.linspace(0, T, N)
    w = sg.chirp(t, f0=6, f1=1, t1=T, method='linear', phi=0)

    # sig = w
    # actualN = N

    sig = np.concatenate( (w, np.flip(w)) )
    sig = np.tile(sig, 5)
    actualN = 10*N
    actualT = 10*T
    t = np.linspace(0, actualT, actualN)

    plt.figure()
    plt.subplot(211)
    plt.plot(t, sig)
    plt.title("Linear Chirp")
    plt.xlabel('t (sec)')

    wf = abs(np.fft.fft(sig))
    wf *= 2/actualN
    f_axis = np.arange(actualN, dtype=float) * 1./actualT

    plt.subplot(212)
    plt.plot(f_axis[:actualN//2], wf[:actualN//2])

    plt.show()

def genTxChirp():

    simFreq = 5e6
    simTime = 25e-6
    N = int(simFreq * simTime)
    print('fs:',simFreq)
    print('T:', simTime)
    print('N:', N)
    t = np.linspace(0, simTime, N)
    txSignal = sg.chirp(t, f0=5792e6, t1=simTime, f1=5808e6, method='linear')

    plt.figure()
    plt.subplot(211)
    plt.plot(t, txSignal)
    plt.ticklabel_format(axis='x', style='sci', scilimits=(0,0), useMathText=True)
    
    fftSig = abs(np.fft.fft(txSignal))
    fftSig *= 2/N
    f_axis = np.arange(N, dtype=float) * 1./simTime

    plt.subplot(212)
    plt.plot(f_axis[:N//2], fftSig[:N//2])
    plt.xlabel('fs = '+str(simFreq))

    plt.subplots_adjust(hspace=0.5)
    plt.show()

def genTxSig():

    # simFreq = 5e6
    simFreq = 200e6
    # chirpTime = 25e-6
    chirpTime = 25e-6
    chirpN = int(simFreq * chirpTime)

    simTime = chirpTime * 10
    simN = int(simFreq * simTime)

    print('fs:',simFreq)
    print('T:', simTime)
    print('N:', simN)

    
    t_chirp_axis = np.linspace(0, chirpTime, chirpN, endpoint=False)
    t_sim_axis = np.linspace(0, simTime, simN, endpoint=False)

    # txChirpSignal = sg.chirp(t_chirp_axis, f0=5792e6, t1=chirpTime, f1=5808e6, method='linear')
    txChirpSignal = sg.chirp(t_chirp_axis, f0=46e6, t1=chirpTime, f1=54e6, method='linear')
    txSignal = np.tile(np.concatenate( (txChirpSignal, np.flip(txChirpSignal)) ), 5)

    plt.figure()
    plt.subplot(211)
    plt.plot(t_sim_axis, txSignal)
    plt.ticklabel_format(axis='x', style='sci', scilimits=(0,0), useMathText=True)
    
    fftSig = abs(np.fft.fft(txSignal))
    fftSig *= 2/simN
    f_axis = np.arange(simN, dtype=float) * 1./simTime

    maxFreqIdx = simN//2
    maxFreq = 5e5
    # maxFreqIdx = int(maxFreq * simTime)

    plt.subplot(212)
    plt.plot(f_axis[:maxFreqIdx], fftSig[:maxFreqIdx])
    plt.xlabel('fs = '+str(simFreq))
    plt.ticklabel_format(axis='x', style='sci', scilimits=(0,0), useMathText=True)

    plt.subplots_adjust(hspace=0.5)
    plt.show()

def radar():
    # simFreq = 300e6
    simFreq = 24000e6

    # freqLow = 46e6
    # freqHigh = 54e6
    freqLow = 5.792e9
    freqHigh = 5.808e9
    BW = freqHigh-freqLow

    chirpTime = 25e-6
    chirpN = int(simFreq * chirpTime)

    triangleNum = 20

    simTime = chirpTime * 2 * triangleNum
    simN = int(simFreq * simTime)
    modFreq = 1/(chirpTime*2)

    print('fs:',simFreq)
    print('T:', simTime)
    print('N:', simN)
    print('freqResolution:', 1/simTime)
    print('BW:', BW)
    print('fm:', modFreq)

    distance = 40
    delta_t = distance/3e8
    delayIdx = int(delta_t*simFreq)
    theoreticalFreq = delta_t*BW/chirpTime

    print('distance:', distance)
    print('delta_t:', delta_t)
    print('delayIdx:', delayIdx)
    print('theoreticalFreq:', theoreticalFreq)
    
    t_chirp_axis = np.linspace(0, chirpTime, chirpN, endpoint=False)
    txChirpSignal = sg.chirp(t_chirp_axis, f0=freqLow, t1=chirpTime, f1=freqHigh, method='linear')

    txSignal = np.tile(np.concatenate( [txChirpSignal, np.flip(txChirpSignal)] ), triangleNum)

    # freqDop = 20000
    # rxDopSigChirp = sg.chirp(t_chirp_axis, f0=freqLow+freqDop, t1=chirpTime, f1=freqHigh+freqDop, method='linear')
    # rxDopSig = np.tile(np.concatenate( (rxDopSigChirp, np.flip(rxDopSigChirp)) ), triangleNum)
    # rxSignal = np.roll(rxDopSig, delayIdx)


    # ## add phase difference

    # txChirpPhaseSig = []
    # for phi in np.linspace(0, 300, triangleNum):
    #     # tmpUpSig = sg.chirp(t_chirp_axis, f0=freqLow, t1=chirpTime, f1=freqHigh, phi=phi, method='linear')
    #     # tmpDnSig = sg.chirp(t_chirp_axis, f0=freqHigh, t1=chirpTime, f1=freqLow, phi=2*phi, method='linear')
    #     tmpUpSig = sg.chirp(t_chirp_axis, f0=freqLow, t1=chirpTime, f1=freqHigh, phi=rd.random()*360, method='linear')
    #     tmpDnSig = sg.chirp(t_chirp_axis, f0=freqHigh, t1=chirpTime, f1=freqLow, phi=rd.random()*360, method='linear')
        
    #     txChirpPhaseSig.append(tmpUpSig)
    #     txChirpPhaseSig.append(np.flip(tmpUpSig))
    #     # txChirpPhaseSig.append(tmpDnSig)
    # txSignal = np.concatenate(txChirpPhaseSig)

    rxSignal = np.roll(txSignal, delayIdx)


    ## lowpass filter

    ifSignalPre = txSignal * rxSignal
    fc = freqLow
    # fc = 300e6
    nfc = fc / simFreq * 2
    b, a = sg.butter(5, nfc, 'low')
    ifSignalPost = sg.filtfilt(b,a,ifSignalPre)

    # ifSignalPost = txSignal * rxSignal  ## w/o lowpass filter

    ## sampling signal

    samFreq = simFreq
    samFreq = 10e6
    samN = int(samFreq * simTime)

    print('')
    print('samFreq:', samFreq)
    print('samN:', samN)

    condition = np.mod(np.arange(simN), int(simFreq/samFreq))==0
    ifSignal = np.extract(condition, ifSignalPost)
    t_axis = np.linspace(0, simTime, samN, endpoint=False)

    plt.figure()

    plt.subplot(211)
    plt.plot(t_axis, ifSignal)
    plt.title('IF Signal of {} m'.format(distance))
    plt.xlabel('time (s)')
    plt.ticklabel_format(axis='x', style='sci', scilimits=(0,0), useMathText=True)


    fftIfSig = abs(np.fft.fft(ifSignal))
    fftIfSig *= 2/samN

    f_axis = np.arange(simN, dtype=float) * 1./simTime
    maxFreqIdx = samN//2
    # maxFreq = 5e5
    # maxFreqIdx = int(maxFreq * simTime)

    plt.subplot(212)
    plt.plot(f_axis[:maxFreqIdx], fftIfSig[:maxFreqIdx])
    plt.title('FFT')
    plt.xlabel('freq (Hz)')
    plt.ticklabel_format(axis='x', style='sci', scilimits=(0,0), useMathText=True)

    plt.subplots_adjust(hspace=0.5)
    plt.show()

def genMixer():

    simFreq = 300e6

    delta_t_1 = 0.005e-6
    delta_t_2 = 0.005e-6
    delta_N_1 = int(simFreq * delta_t_1)
    delta_N_2 = int(simFreq * delta_t_2)

    maintain_t = (50e-6 - delta_t_1 - delta_t_2)/2
    maintain_N = int(simFreq * maintain_t)

    modTime = maintain_t * 2 + delta_t_1 + delta_t_2
    modFreq = 1/modTime

    triangleNum = 10

    simTime = modTime * triangleNum
    simN = (delta_N_1+delta_N_2+maintain_N)*2*triangleNum ##

    print('simFreq:',simFreq)
    print('simT:', simTime)
    print('simN:', simN, (delta_N_1+delta_N_2+maintain_N)*2*triangleNum)

    print('freqResolution:', 1/simTime)
    print('fm:', modFreq)

    f1 = 1e6
    f2 = 25e4

    print('f1',f1)
    print('f2',f2)
    
    t_chirp_axis_1 = np.linspace(0, delta_t_1, delta_N_1, endpoint=False)
    t_chirp_axis_2 = np.linspace(0, delta_t_2, delta_N_2, endpoint=False)
    t_maintain_axis = np.linspace(0, maintain_t, maintain_N, endpoint=False)
    t_axis = np.linspace(0, simTime, simN, endpoint=False)

    ifChirpSignal_1 = sg.chirp(t_chirp_axis_1, f0=0, t1=delta_t_1, f1=f1, method='linear')
    ifChirpSignal_2 = sg.chirp(t_chirp_axis_1, f0=0, t1=delta_t_2, f1=f2, method='linear')
    maintainSignal_1 = sg.chirp(t_maintain_axis, f0=f1, t1=maintain_t, f1=f1, method='linear')
    maintainSignal_2 = sg.chirp(t_maintain_axis, f0=f2, t1=maintain_t, f1=f2, method='linear')

    ifSignal = np.tile(np.concatenate( (maintainSignal_1, np.flip(ifChirpSignal_1), ifChirpSignal_2, maintainSignal_2, np.flip(ifChirpSignal_2), ifChirpSignal_1) ), triangleNum)

    ## add phase difference

    ifSignal_phase = []
    for phi in np.linspace(0, 650, triangleNum):
        # tmpSig_1 = sg.chirp(t_maintain_axis, f0=f1, t1=maintain_t, f1=f1, phi=phi, method='linear')
        # tmpSig_2 = sg.chirp(t_maintain_axis, f0=f2, t1=maintain_t, f1=f2, phi=phi, method='linear')

        tmpSig_1 = sg.chirp(t_maintain_axis, f0=f1, t1=maintain_t, f1=f1, phi=rd.random()*360, method='linear')
        tmpSig_2 = sg.chirp(t_maintain_axis, f0=f2, t1=maintain_t, f1=f2, phi=rd.random()*360, method='linear')
        
        ifSignal_phase.extend([ tmpSig_1, np.flip(ifChirpSignal_1), ifChirpSignal_2, tmpSig_2, np.flip(ifChirpSignal_2), ifChirpSignal_1 ])

    ifSignal = np.concatenate(ifSignal_phase)

    plt.figure()

    plt.subplot(211)
    plt.plot(t_axis, ifSignal)
    plt.title('IF Signal')
    plt.xlabel('time (s)')
    plt.ticklabel_format(axis='x', style='sci', scilimits=(0,0), useMathText=True)


    fftIfSig = abs(np.fft.fft(ifSignal))
    fftIfSig *= 2/simN

    f_axis = np.arange(simN, dtype=float) * 1./simTime
    # maxFreqIdx = simN//2
    maxFreq = max(f1,f2)*2
    maxFreqIdx = int(maxFreq * simTime)

    plt.subplot(212)
    plt.plot(f_axis[:maxFreqIdx], fftIfSig[:maxFreqIdx])
    plt.title('FFT')
    plt.xlabel('freq (Hz)')
    plt.ticklabel_format(axis='x', style='sci', scilimits=(0,0), useMathText=True)

    plt.subplots_adjust(hspace=0.5)

    plt.show()


def readcsv(filename):
    """return signal list and simulation data frequency"""

    signal = []
    with open('../distance_raw_0118_early/'+filename+'.csv') as file:
        datas = csv.reader(file)
        simFreq = 0
        for ind, data in enumerate(datas):
            if ind==0: continue
            elif ind==1:
                simFreq = 1/float(data[3])
            else:
                signal.append(float(data[1]))
    return signal, simFreq
def mmain():
    # genMixer()
    radar()
    # genTxSig()
    # test()

def main():

    filename = '75'

    y, fs = readcsv(filename)
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


    plt.subplot(211)
    plt.plot(t_axis, y, 'b')
    plt.title('Signal of '+filename+' cm')
    plt.xlabel('time (s)')
    plt.ticklabel_format(axis='x', style='sci', scilimits=(0,0), useMathText=True)

    plt.subplot(212)

    # max_freq = (len(f_axis)//2)*min_freq_diff
    max_freq = 5e5
    max_freq_index = int(max_freq/min_freq_diff)


    # ## block modulation frequency
    # for i in range(len(yfn)):
    #     if i % 5 ==0: yfn[i]=0

    plt.plot(f_axis[:max_freq_index],yfn[:max_freq_index], 'r')
    peaks, _ = sg.find_peaks(yfn[:max_freq_index], height = 0.01)

    plt.plot(peaks*min_freq_diff,[ yfn[i] for i in peaks], 'x')
    for ind, i in enumerate(peaks):
        plt.annotate(s=int(peaks[ind]*min_freq_diff), xy=(peaks[ind]*min_freq_diff,yfn[i]))
    plt.title('FFT')
    plt.xlabel('freq (Hz)')
    plt.ticklabel_format(axis='x', style='sci', scilimits=(0,0), useMathText=True)

    plt.subplots_adjust(hspace=0.5)
    plt.show()
if __name__ == '__main__':
    main()
