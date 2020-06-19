#!/usr/local/bin/python
# Run an eBay AD9850 on the RPi GPIO
#
# translated from nr8o's Arduino sketch
# at http://nr8o.dhlpilotcentral.com/?p=83
#
# m0xpd
# shack.nasties 'at Gee Male dot com'
#
# modified by team Troy
#
import RPi.GPIO as GPIO
# setup GPIO options...
GPIO.setmode(GPIO.BOARD)
GPIO.setwarnings(False)

W_CLK = 12                          # Define GPIO pins
FQ_UD = 16
DATA = 18
PARA_DATA = [40,38,36,37,35,33,31,29]
RESET = 22
f_basis = 1000
f_shift = 10000
f_step = 11
f_high = f_basis+f_shift*f_step
f_list = []

def pulseHigh(pin):                     # Function to send a pulse
    GPIO.output(pin, True)              # do it a few times to increase pulse width
    GPIO.output(pin, True)              # (easier than writing a delay loop!)
    GPIO.output(pin, True)
    GPIO.output(pin, False)             # end of the pulse
    return


def old_tfr_byte(data):                 # Function to send a byte by serial "bit-banging"
    for i in range (0,8):
        GPIO.output(DATA, data & 0x01)  # Mask out LSB and put on GPIO pin "DATA"
        pulseHigh(W_CLK)                # pulse the clock line
        data=data>>1                    # Rotate right to get next bit
    return

def old_sendFrequency(freq):            # Function to send frequency (assumes 125MHz xtal)
    for b in range (0,4):
        old_tfr_byte(freq & 0xFF)
        freq=freq>>8
    old_tfr_byte(0x01)
    pulseHigh(FQ_UD)
    return


""" --------------- serial input --------------- """

def tfr_byte(data):                 # Function to send a byte by serial "bit-banging"
    for i in range (8):
        GPIO.output(DATA, data[i])  # Mask out LSB and put on GPIO pin "DATA"
        pulseHigh(W_CLK)            # pulse the clock line
    return

def tfr_int(data):                  # Function to send a byte by serial "bit-banging"
    for i in range (32):
        GPIO.output(DATA, data[i])  # Mask out LSB and put on GPIO pin "DATA"
        pulseHigh(W_CLK)            # pulse the clock line
    return

def sendFrequency(freq):            # Function to send frequency (assumes 125MHz xtal)
    tfr_int(freq)
    tfr_byte( [1,0,0,0,0,0,0,0] )
    pulseHigh(FQ_UD)
    return


""" -------------------------------------------- """


""" -------------- parallel input -------------- """


def tfr_word(word):
    for i in range (8):
        GPIO.output(PARA_DATA[i], word[i])  # Mask out LSB and put on GPIO pin "DATA"
    pulseHigh(W_CLK)                        # pulse the clock line
    return

def sendFreqParallel(freq):
    tfr_word([1,0,0,0,0,0,0,0])
    tfr_word(freq[24:32])
    tfr_word(freq[16:24])
    tfr_word(freq[8:16])
    tfr_word(freq[0:8])
    pulseHigh(FQ_UD)

""" -------------------------------------------- """

def triangularList(f_step, f_shift, f_basis):
    for i in range (0,f_step):
        frequency = f_basis+i*f_shift                # choose frequency and
        freq=int(frequency*4294967296/180000000)
        f_list.append(freq)
    for i in range (0,f_step):
        frequency = f_high-i*f_shift                 # choose frequency and
        freq=int(frequency*4294967296/180000000)
        f_list.append(freq)
    print (f_list)
    return


def sawtoothList(f_step, f_shift, f_basis):
    for i in range (0,f_step):
        frequency = f_basis+i*f_shift                # choose frequency and
        freq=int(frequency*4294967296/180000000)
        f_list.append(freq)
    print (f_list)
    return

def flist2Binary(flist):
    blist = []
    bfreq = []
    for freq in flist:
        for i in range(32):
            bfreq.append(freq & 0x01)
            freq = freq >> 1
        blist.append(bfreq)
        bfreq = []
    print(blist)
    return blist



GPIO.setup(W_CLK, GPIO.OUT)             # setup IO bits...
GPIO.setup(FQ_UD, GPIO.OUT)             #
GPIO.setup(RESET, GPIO.OUT)             #
# GPIO.setup(DATA, GPIO.OUT)              #
for i in range(8):
    GPIO.setup(PARA_DATA[i], GPIO.OUT)


GPIO.output(W_CLK, False)               # initialize everything to zero...
GPIO.output(FQ_UD, False)
GPIO.output(RESET, False)
# GPIO.output(DATA, False)
for i in range(8):
    GPIO.output(PARA_DATA[i], False)

pulseHigh(RESET)                        # start-up sequence...
pulseHigh(W_CLK)
pulseHigh(FQ_UD)

sawtoothList(f_step, f_shift, f_basis)

b_list = flist2Binary(f_list)
while True:
    for freq in b_list:
        sendFrequency(freq)
        # sendFreqParallel(freq)
    # old_sendFrequency(28360)
#    GPIO.output(DATA, True)
#    GPIO.output(DATA, False)
