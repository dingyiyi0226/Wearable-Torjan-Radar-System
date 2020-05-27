#!/usr/bin/python
# -*- coding:utf-8 -*-

import argparse
import os
import time

import numpy as np
from matplotlib import pyplot as plt
from scipy.fft import fft, fftshift

import ADS1256
import DAC8532
import RPi.GPIO as GPIO

# os.nice(20)

parser = argparse.ArgumentParser("High Precision Anolog / Digital Converter")
parser.add_argument("-n", "--number", default=4000, type=int, help="Number of data points.")
parser.add_argument("--accept", default=1.6e4, type=float, help="Threshold to accept the sampling")
parser.add_argument("-c", "--channel", default=0, type=int, help="Serial port number.")
args = parser.parse_args()

def main():
    buf = np.zeros(args.number)
    fig, ax = plt.subplots(nrows=2, ncols=1)
        
    try:
        ADC = ADS1256.ADS1256()

        ADC.init()
        ADC.SetChannal(args.channel)
        ADC.WriteCmd(ADS1256.CMD['CMD_SYNC'])
        ADC.WriteCmd(ADS1256.CMD['CMD_WAKEUP'])
        ADC.Start_Read_ADC_Data_Continuous()

        while True:
            # Load data points
            timestamp = time.time()
            for i in range(args.number):
                buf[i] = ADC.Read_ADC_Data_Continuous()
            timedelta = time.time() - timestamp
            fs = args.number / timedelta
            print("\r{}".format(fs), end="")

            # Reset plot
            if (fs > args.accept):
                buf /= (10 ** 6)

                # Time Axis
                ax[0].clear()
                ax[0].plot(
                    np.linspace(0, timedelta, args.number), 
                    buf
                )

                # Frequency Axis
                ax[1].clear()
                ax[1].plot(
                    np.linspace(-fs / 2, fs / 2, args.number, endpoint=False), 
                    np.abs(fftshift(fft(buf))) / args.number
                )

                # Show figure
                plt.pause(0.01)

    except Exception as e:
        print(e)
        GPIO.cleanup()
        exit()

   
if __name__ == "__main__":
    main()
