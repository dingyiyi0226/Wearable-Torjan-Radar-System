import argparse
import numpy as np
from matplotlib import pyplot as plt

HIGH_FREQUENCY    = 102
LOW_FREQUENCY     = 100
SAMPLE_FREQUENCY  = 3e3          # Arduino Sampling Frequency (Hz)
SIMULATION_TIME   = 1e0          # Time to simulate (s)
SIMULATION_POINTS = 1e7          # Simulation
MODULATION_DURATION = 1e-2       # delta_t in FM Modulation (s)
CENTER_FREQUENCY  = 5.8e9        # Radar (Hz)
LIGHT_SPEED       = 3e8          # (m/s)

DELTA_T           = SIMULATION_TIME / SIMULATION_POINTS
SQUARE_WAVE_WIDTH = int(MODULATION_DURATION / 2 / DELTA_T)

def getSpec():
    print("Delta t: ", SIMULATION_TIME / SIMULATION_POINTS)
    print("Modulation: ", MODULATION_DURATION)

    return

def setSpec():
    assert(SIMULATION_TIME / SIMULATION_POINTS < MODULATION_DURATION)

    timeaxis = np.linspace(0, SIMULATION_TIME, SIMULATION_POINTS)

    coshigh  = np.cos(np.linspace(0, 2 * np.pi * SIMULATION_TIME * HIGH_FREQUENCY, SIMULATION_POINTS, dtype=np.float32))
    coslow   = np.cos(np.linspace(0, 2 * np.pi * SIMULATION_TIME * LOW_FREQUENCY, SIMULATION_POINTS, dtype=np.float32))

    mask = np.tile(np.array([1, 0], dtype=np.uint8).repeat(SQUARE_WAVE_WIDTH), reps=int(SIMULATION_TIME / MODULATION_DURATION))
    mask2 = np.tile(np.array([0, 1], dtype=np.uint8).repeat(SQUARE_WAVE_WIDTH), reps=int(SIMULATION_TIME / MODULATION_DURATION))
    
    signal = coslow * mask + coshigh * mask2

    signal =  signal[::int(SIMULATION_POINTS / (SIMULATION_TIME * SAMPLE_FREQUENCY))]
    print('sampling point', (SIMULATION_TIME * SAMPLE_FREQUENCY))

    return timeaxis[::int(SIMULATION_POINTS / (SIMULATION_TIME * SAMPLE_FREQUENCY))], signal

def main():
    getSpec()

    timeaxis, receivedSignal = setSpec()

    plt.subplot(211)
    plt.scatter(timeaxis, receivedSignal)
    plt.xlim(0, SIMULATION_TIME)

    # print(receivedSignal.shape)
    fftsignal = np.fft.fft(receivedSignal)

    # print('fftsignal length ',fftsignal.shape)
    fftDatas = [ i*2/len(fftsignal) for i in fftsignal ]

    plt.subplot(212)

    fax = [(i/SIMULATION_TIME) for i in range(int(len(fftDatas)/2))]

    # plt.plot(fax, fftDatas[:int(len(fftDatas)/2)])
    plt.plot(fax[:200], fftDatas[:200])
    
    plt.show()

    return

if __name__ == "__main__":
    main()
