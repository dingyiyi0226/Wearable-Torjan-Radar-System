"""
  Filename      [ AD7770.py ]
  PackageName   [ Radar ]
  Synopsis      [ AD7770 Signal Reader and processing module ]
"""

import argparse
import os
import numpy as np
import pandas as pd
from functools import reduce
from matplotlib import cm
from matplotlib import pyplot as plt
from matplotlib.image import NonUniformImage
from scipy.signal import correlate, stft
from scipy.constants import c, pi
from scipy.fftpack import fft, fft2, fftshift

def floor(x, step):
    return int(np.floor(x / step) * step)

def listdir(dirname):
    return sorted([os.path.join(dirname, fname) for fname in os.listdir(dirname)])

def readFile(fname, ratio=1e6, removeDC=True):
    df = pd.read_csv(fname, header=None, low_memory=False).loc[:, 1:8]
    df = df[df.notnull().all(axis=1)].to_numpy().transpose() / ratio

    if removeDC:
        df = df - np.mean(df, axis=1)[:, None]

    return df

def dopplerMap(m, bw, fc, tm):
    distance    = c / (2 * bw) * np.arange(-m.shape[1] / 2, m.shape[1] / 2)
    velocity    = c / (2 * fc) / (m.shape[0] * tm) * np.arange(-m.shape[0] / 2, m.shape[0] / 2)
    dopplerMap  = np.abs(fftshift(fft2(m), axes=None)) / m.size
    
    return dopplerMap, distance, velocity

def singleFreqInference(signal, centerFreq, sampleFreq, threshold, segments, stride):
    SEGMENT_POINTS = int(segments * sampleFreq)
    STRIDE_POINTS  = int(stride * sampleFreq)

    f, t, spectrogram = stft(signal, fs=sampleFreq, nperseg=SEGMENT_POINTS, noverlap=STRIDE_POINTS)
    freqIndex = np.argmax(spectrogram, axis=0)
    freqDiff = f[freqIndex]
    speed = freqDiff * c / (2 * centerFreq)

    return speed

def visualize(fname, method, fs, t_window, t_stride, drop_last=True, *args, **kwargs):
    """
    Parameters
    ----------
    fname : str

    method : function
        
    fs : float
        sampling frequency

    t_window : float
        observed time

    t_stride : float
        sliding length of time
    """

    # CONSTANTS
    WINDOW_POINT = int(t_window * fs)
    STRIDE_POINT = int(t_stride * fs)

    data = readFile(fname)[0]

    fig, axs = plt.subplots(nrows=3, ncols=1, figsize=(12.8, 7.2))
    for start in range(0, data.size - WINDOW_POINT, STRIDE_POINT):
        # Canvas and drawing method
        for ax in axs: ax.clear()
        method(data[start: start + WINDOW_POINT], fs, axs, **kwargs)

        # Decoration
        fig.suptitle("{}: {} - {}".format(fname, start / fs, (start + WINDOW_POINT) / fs))
        
        # Replay setting
        plt.pause(0.1)

    plt.close()

    return

def single(data, fs, axs, *args, **kwargs):
    """
    Parameters
    ----------
    data : np.array

    fs : float
        sampling frequency

    t_window : float
        observed time

    t_stride : float
        sliding length of time
    """
    # Constant
    CENTER_FREQ = kwargs['fc']

    spectrum = np.abs(fftshift(fft(data))) / data.size
    spectrum = spectrum[int(data.size / 2):]
    freqAxis = np.linspace(0, fs / 2, spectrum.shape[0], endpoint=False)
    timeAxis = np.linspace(0, data.shape[0] / fs, data.shape[0], endpoint=False)    
    maxIndex = np.argmax(spectrum)

    # Time Axis
    ax = axs[1]
    ax.plot(timeAxis, data)
    ax.set_title('Waveform')

    # Frequency Axis
    ax = axs[2]
    ax.plot(freqAxis, spectrum)
    ax.annotate("{:.2f}".format(freqAxis[maxIndex] * c / (2 * CENTER_FREQ)), (freqAxis[maxIndex], np.max(spectrum)))
    ax.set_title('Frequency')

    return

def triangular(data, fs, axs, *args, **kwargs):
    """
    Parameters
    ----------
    data : np.array

    fs : float
        sampling frequency

    t_window : float
        observed time

    t_stride : float
        sliding length of time
    """

    # Constant
    MODULE_TIME   = kwargs['tm']
    BANDWIDTH     = kwargs['bw']
    CENTER_FREQ   = kwargs['fc']

    MODULE_POINTS = int(MODULE_TIME * fs)

    # Data Processing
    data = data[:floor(data.size, MODULE_POINTS)]
    spectrum = np.abs(fftshift(fft(data)))
    spectrum = spectrum[int(data.size / 2):]
    doppler, distance, velocity = dopplerMap(data.reshape(-1, MODULE_POINTS), BANDWIDTH, CENTER_FREQ, MODULE_TIME)

    timeAxis = np.linspace(0, data.shape[0] / fs, data.shape[0], endpoint=False)
    freqAxis = np.linspace(0, fs / 2, spectrum.shape[0], endpoint=False)
    maxIndex = np.argmax(spectrum)

    # Doppler Map
    ax = axs[0]
    img = NonUniformImage(ax, cmap=cm.Purples)
    img.set_data(distance, velocity, doppler)
    ax.images.append(img)
    ax.set_xlim(np.min(distance), np.max(distance))
    ax.set_ylim(np.min(velocity), np.max(velocity))
    ax.set_title('Doppler Map')

    # Time Axis
    ax = axs[1]
    ax.plot(timeAxis, data)
    ax.set_title('Waveform')

    # Frequency Axis
    ax = axs[2]
    ax.plot(freqAxis, spectrum)
    ax.annotate("{:.2f}".format(freqAxis[maxIndex]), (freqAxis[maxIndex], np.max(spectrum)))
    ax.set_title('Frequency')

    return

def sawtooth(data, fs, axs, *args, **kwargs):
    """
    Parameters
    ----------
    data : np.array

    fs : float
        sampling frequency

    t_window : float
        observed time

    t_stride : float
        sliding length of time
    """
    # Constant
    MODULE_TIME   = kwargs['tm']
    BANDWIDTH     = kwargs['bw']
    CENTER_FREQ   = kwargs['fc']

    MODULE_POINTS = int(MODULE_TIME * fs)

    # Data Processing
    data = data[:floor(data.size, MODULE_POINTS)]
    spectrum = np.abs(fftshift(fft(data)))
    spectrum = spectrum[int(data.size / 2):]
    doppler, distance, velocity = dopplerMap(data.reshape(-1, MODULE_POINTS), BANDWIDTH, CENTER_FREQ, MODULE_TIME)

    timeAxis = np.linspace(0, data.shape[0] / fs, data.shape[0], endpoint=False)
    freqAxis = np.linspace(0, fs / 2, spectrum.shape[0], endpoint=False)
    maxIndex = np.argmax(spectrum)
 
    # Doppler Map
    ax = axs[0]
    img = NonUniformImage(ax, cmap=cm.Purples)
    img.set_data(distance, velocity, doppler)
    ax.images.append(img)
    ax.set_xlim(np.min(distance), np.max(distance))
    ax.set_ylim(np.min(velocity), np.max(velocity))
    ax.set_title('Doppler Map')

    # Time Axis
    ax = axs[1]
    ax.plot(timeAxis, data)
    ax.set_title('Waveform')

    # Frequency Axis
    ax = axs[2]
    ax.plot(freqAxis, spectrum)
    ax.annotate("{:.2f}".format(freqAxis[maxIndex]), (freqAxis[maxIndex], np.max(spectrum)))
    ax.set_title('Frequency')

    return

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dir', help="Directory of the records.")
    args = parser.parse_args()

    # CONSTANT
    SAMPLE_FREQ = 1.6e4
    SAMPLE_TIME = 1 / SAMPLE_FREQ
    BANDWIDTH   = 1e8
    CENTER_FREQ = 5.8e9
    MODULE_TIME = 4e-3
    
    SEGMENT_TIME = 0.5
    STRIDE_TIME  = 0.125

    # for fname in listdir('./rawdata/0503_single'):
    #     visualize(fname, single, SAMPLE_FREQ, SEGMENT_TIME, STRIDE_TIME, fc=CENTER_FREQ)

    # for fname in listdir('./rawdata/0503_triangular'):
    #     visualize(fname, triangular, SAMPLE_FREQ, SEGMENT_TIME, STRIDE_TIME, fc=CENTER_FREQ, bw=BANDWIDTH, tm=2 * MODULE_TIME)

    # for fname in listdir('./rawdata/0503_sawtooth'):
    #     visualize(fname, sawtooth, SAMPLE_FREQ, SEGMENT_TIME, STRIDE_TIME, fc=CENTER_FREQ, bw=BANDWIDTH, tm=MODULE_TIME)

    # for fname in listdir('./rawdata/0505_single'):
    #     visualize(fname, single, SAMPLE_FREQ, SEGMENT_TIME, STRIDE_TIME, fc=CENTER_FREQ)

    # for fname in listdir('/rawdata/0505_triangular'):
    #     visualize(fname, triangular, SAMPLE_FREQ, SEGMENT_TIME, STRIDE_TIME)
    
    return

if __name__ == "__main__":
    main()
