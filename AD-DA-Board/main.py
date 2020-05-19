#!/usr/bin/python
# -*- coding:utf-8 -*-

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

        while True:
            buf = np.zeros(1200)

            for i in range(1200):
                buf[i] = ADC.ADS1256_GetChannalValue(0)
            
            # print("0 ADC = %lf" % (ADC_Value * 5.0 / 0x7fffff), end='\r')
            print(time.time() - timestamp)
            timestamp = time.time()

    except Exception as e:
        print(e)
        GPIO.cleanup()
        exit()

   
if __name__ == "__main__":
    main()

