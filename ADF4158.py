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

import functools.wraps
from enum import Enum, IntEnum, unique
import RPi.GPIO as GPIO

GPIO.setmode(GPIO.BOARD)
GPIO.setwarnings(False)

# Define GPIO pins
W_CLK  = 12
LE     = 14
DATA   = 18
TXDATA = 15
MUXOUT = 17
RESET  = 22

@unique
class RampMode(IntEnum):
    """ Write to Register 3 """
    CONT_SAWTOOTH   = 0
    CONT_TRIANGULAR = 1
    SING_SAWTOOTH   = 2
    SING_BURST      = 3

@unique
class Clock(IntEnum):
    FALLING_EDGE = 0
    RISING_EDGE  = 1

def initGPIO():
    for pin in (W_CLK, LE, DATA, TXDATA, RESET):
        GPIO.setup(pin, GPIO.OUT)
    
    for pin in (MUXOUT, ):
        GPIO.setup(pin, GPIO.IN)

# TODO
def parseWord(word):
    return

# TODO
def verbose(func):
    @functools.wraps(func)
    def wrapper(*args):
        parseWord(args[0])
        return func(*args)

    return wrapper


def pulseHigh(pin):
    """ Function to send a pulse """
    GPIO.output(pin, True)
    GPIO.output(pin, False)
    
    return

# TODO
@verbose
def sendWord(word, clk=Clock.RISING_EDGE):
    """
    :param word: 32-bits information

    :param clk: { Clock.RISING_EDGE, Clock.FALLING_EDGE } optional
    """
    
    GPIO.output(LE, True)
    
    # Start sending bits
    pulseHigh(TXDATA)

    for i in range(31, -1, -1):
        GPIO.output(DATA, word[i])
        pulseHigh(W_CLK)

    GPIO.output(LE, False)

    return True

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
    sendWord(0x0000A006)
    sendWord(0x00800006)
    return

# TODO
def setDelayRegister():
    sendWord(0x7)
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
    
    initGPIO()
    return True

def main():
    initADF4851()

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

    return

if __name__ == "__main__":
    main()