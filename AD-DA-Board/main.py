#!/usr/bin/python
# -*- coding:utf-8 -*-

from matplotlib import pyplot as plt
import numpy as np
import time
import ADS1256
import DAC8532
import RPi.GPIO as GPIO

def main():
    try:
        ADC = ADS1256.ADS1256()
        ADC.ADS1256_init()
        timestamp = time.time()
        buf = np.zeros(1200)
        ADC.ADS1256_SetChannal(0)
        ADC.ADS1256_WriteCmd(ADS1256.CMD['CMD_SYNC'])
        ADC.ADS1256_WriteCmd(ADS1256.CMD['CMD_WAKEUP'])
        ADC.ADS1256_Start_Read_ADC_Data_Continuous()
        
        fig, ax = plt.subplots()

        while True:
            timestamp = time.time()
            plt.cla()

            for i in range(1200):
                buf[i] = ADC.ADS1256_Read_ADC_Data_Continuous()
            
            ax.plot(np.linspace(0, time.time() - timestamp, 1200, endpoint=True), buf)
            plt.pause(0.25)

    except Exception as e:
        print(e)
        GPIO.cleanup()
        exit()

   
if __name__ == "__main__":
    main()

