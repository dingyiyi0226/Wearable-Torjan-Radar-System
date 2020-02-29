"""
  Filename      [ controller.py ]
  PackageName   [ Radar ]
  Synopsis      [ ADF4158 Signal Generator Control Module ]

  R0: FRAC/INT REGISTER
  R1: LSB FRAC REGISTER
  R2: R-DIVIDER REGISTER
  R3: FUNCTION REGISTER
  R4: TEST REGISTER
  R5: DEVIATION REGISTER
  R6: STEP REGISTER
  R7: DELAY REGISTER
"""

from functools import wraps
from enum import Enum, IntEnum, unique
import RPi.GPIO as GPIO

GPIO.setmode(GPIO.BOARD)
GPIO.setwarnings(False)

# Define GPIO pins
# Pin numbering scheme: 
#   - Defined by GPIO.setmode(GPIO.BOARD)
#   - Use physical pin number

# GND  = 6      # T3
W_CLK  = 12     # T4
DATA   = 16     # T5
LE     = 18     # T6
TXDATA = 13     # T16
MUXOUT = 15     # T8

@unique
class Clock(IntEnum):
    FALLING_EDGE = 0
    RISING_EDGE  = 1

@unique
class RampMode(IntEnum):
    """ DB11 - DB10 at Register 3: FUNCTION REGISTER """
    CONT_SAWTOOTH   = 0
    CONT_TRIANGULAR = 1
    SING_SAWTOOTH   = 2
    SING_BURST      = 3

@unique
class Muxout(IntEnum):
    """ DB30 - DB27 at Register 0: FRAC/INT REGISTER """
    THREE_STATE  = 0
    D_VDD        = 1
    D_GND        = 2
    R_DIVIDER    = 3
    N_DIVIDER    = 4
    # RESERVED   = 5
    DIGITAL_LOCK = 6
    SERIAL_DATA  = 7
    # RESERVED   = 8
    # RESERVED   = 9
    CLK_DIVIDER  = 10
    # RESERVED   = 11
    FAST_LOCK    = 12
    R_DIVIDER_2  = 13
    N_DIVIDER_2  = 14
    READBACK     = 15

def initGPIOPins():
    """ GPIO Pins initialization for ADF4158 config. """
    for pin in (W_CLK, LE, DATA):
        GPIO.setup(pin, GPIO.OUT)
    
    for pin in (TXDATA, MUXOUT, ):
        GPIO.setup(pin, GPIO.IN)

def verbose(func):
    """ Decorator for sendWord(). Show words to send in binary format. """
    @wraps(func)
    def wrapper(*args):
        print("{}".format(bin(args[0]).zfill(32)))
        return func(*args)

    return wrapper

def pulseHigh(pin):
    """ 
    Send a pulse 
    
    :param pin: number of pin to send a pulse
    """
    GPIO.output(pin, True)
    GPIO.output(pin, False)

def setReady():
    GPIO.output(W_CLK, False)
    GPIO.output(DATA, False)
    GPIO.output(LE, False)

# @verbose
def sendWord(word, clk=Clock.RISING_EDGE):
    """
    :param word: 32-bits information

    :param clk: { Clock.RISING_EDGE, Clock.FALLING_EDGE } optional
    """
    
    # Raise Clock after setup DATA 
    for i in range(31, 0, -1):
        GPIO.output(DATA, bool((word >> i) % 2))
        pulseHigh(W_CLK)

    # Last bit
    GPIO.output(DATA, bool(word % 2))
    GPIO.output(W_CLK, True)
    GPIO.output(LE, True)
    GPIO.output(W_CLK, False)
    GPIO.output(LE, False)

    # Raise LE
    # pulseHigh(LE)

    # Reset Data as 0
    GPIO.output(DATA, False)

    return True

# TODO
def readWord():
    """ Readback words """
    word = 0

    GPIO.output(LE, True)
    pulseHigh(TXDATA)

    for _ in range(36, -1, -1):
        GPIO.output(W_CLK, True)
        word = word << 1 + GPIO.input(MUXOUT)
        GPIO.output(W_CLK, False)

    GPIO.output(LE, False)

    return word

# TODO
def setModulationInterval(tm=None, fm=None):
    """
    :param tm:

    :param fm:
    """
    return

# TODO
def setDeviationRegister():
    return

# TODO
def setStepRegister(step1, step2):
    return

# TODO
def setDelayRegister():
    return 

# TODO
def setPLL():
    return

# TODO
def initADF4851():
    """ 
    Initial ADF4851 Signal Generator

    1. Delay register (R7)
    2. Step register (R6)
       load the step register (R6) twice, first with STEP SEL = 0 and then with STEP SEL = 1
    3. Deviation register (R5)
       load the deviation register (R5) twice, first with DEV SEL = 0 and then with DEV SEL = 1
    4. Test register (R4)
    5. Function register (R3)
    6. R-divider register (R2)
    7. LSB FRAC register (R1)
    8. FRAC/INT register (R0)

    .. References:
        (Datasheet p.25)
    """
    
    initGPIOPins()
    setReady()

    return True

def test_singleFreq():
    """ Unittest: send single freq. control words """
    sendWord(0x00000007)
    sendWord(0x00000006)
    sendWord(0x00800006)
    sendWord(0x00000005)
    sendWord(0x00800005)
    sendWord(0x00180104)
    sendWord(0x00000043)
    sendWord(0x0040800A)
    sendWord(0x00000001)
    sendWord(0x01220000)

def test_triangle():
    """ Unittest: send ramp freq. control words """
    sendWord(0x00000007)
    sendWord(0x0000A006)
    sendWord(0x00800006)
    sendWord(0x000BFFFD)
    sendWord(0x00800005)
    sendWord(0x00180104)
    sendWord(0x00000443)
    sendWord(0x0740800A)
    sendWord(0x00000001)
    sendWord(0x811F8000)

def main():
    initADF4851()

    while True:
        # test_singleFreq()
        test_triangle()

    return

if __name__ == "__main__":
    main()
