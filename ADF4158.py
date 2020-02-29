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

from collections import OrderedDict
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

# ------------------------------------------------------ #
# Enumerate                                              #
# ------------------------------------------------------ #

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

# ------------------------------------------------------ #
# Helper function                                        #
# ------------------------------------------------------ #

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

def bitMask(pos):
    """ 0...010...0 Mask. """
    return (1 << pos)

def mask(start, end=0):
    """ 0...011...1 Mask. """
    return (1 << start) - (1 << end)

def overwrite(value, start, end, newValue):
    return (value & mask(start, end)) | (newValue << end)

def setReadyToWrite():
    GPIO.output(W_CLK, False)
    GPIO.output(DATA, False)
    GPIO.output(LE, False)

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

# ------------------------------------------------------ #
# Public Function                                        #
# ------------------------------------------------------ #

@verbose
def sendWord(word, clk=Clock.RISING_EDGE):
    """
    :param word: 32-bits information

    :param clk: { Clock.RISING_EDGE, Clock.FALLING_EDGE } optional
    """
    
    # Raise Clock after setup DATA 
    for i in range(31, 0, -1):
        GPIO.output(DATA, bool((word >> i) % 2))
        pulseHigh(W_CLK)

    # Hold LE for last clock
    GPIO.output(DATA, bool(word % 2))
    GPIO.output(W_CLK, True)
    GPIO.output(LE, True)
    GPIO.output(W_CLK, False)
    GPIO.output(LE, False)

    # Reset Data as 0
    GPIO.output(DATA, False)

    return True

def initBitPatterns() -> OrderedDict:
    """ Initialize bit patterns """
    patterns = OrderedDict()

    patterns['PIN7']  = 0x00000007
    patterns['PIN6A'] = 0x00000006
    patterns['PIN6B'] = 0x00800006
    patterns['PIN5A'] = 0x00000005
    patterns['PIN5B'] = 0x00800005
    patterns['PIN4']  = 0x00180104
    patterns['PIN3']  = 0x00000043
    patterns['PIN2']  = 0x0040800A
    patterns['PIN1']  = 0x00000001
    patterns['PIN0']  = 0x01220000

    return patterns

def initGPIOPins() -> None:
    """ GPIO Pins initialization for ADF4158 config. """
    for pin in (W_CLK, LE, DATA):
        GPIO.setup(pin, GPIO.OUT)
    
    for pin in (TXDATA, MUXOUT, ):
        GPIO.setup(pin, GPIO.IN)

def initADF4851() -> OrderedDict:
    """ 
    Initial ADF4851 Signal Generator

    :return patterns: the initial patterns wrote in ADF4158

    .. References:
        (Datasheet p.25)
    """
    
    initGPIOPins()
    setReadyToWrite()

    patterns = initBitPatterns()
    for value in patterns.values():
        sendWord(value)

    return patterns

def setRamp(patterns, status: bool):
    patterns['PIN0'] = overwrite(patterns['PIN0'], 31, 31, int(status))
    return patterns

def setRampMode(patterns, mode: RampMode):
    patterns['PIN3'] = overwrite(patterns['PIN3'], 11, 9, int(mode))
    return patterns

def setMuxout(patterns, mode: Muxout):
    patterns['PIN0'] = overwrite(patterns['PIN0'], 30, 27, int(mode))
    return patterns

# TODO
def setRampAttr(patterns, clk=None, dev=None, devOffset=None, steps=None):
    """
    :param clk: CLK_2 devider value at range [0, 4095]

    :param dev: Deviation words at range [-32768, 32767]

    :param devOffset: Deviation offset at range [0, 9]

    :param steps: Step words at range [0, 1048575]

    :return patterns
    """

    if clk is not None:
        assert(clk >= 0 and clk <= 4095)
        patterns['PIN4']  = overwrite(patterns['PIN4'], 18, 7, clk)

    if dev is not None:
        assert(dev >= -32768 and dev <= 32767)
        patterns['PIN5A'] = overwrite(patterns['PIN5A'], 18, 3, dev)

    if devOffset is not None:
        assert(devOffset >= 0 and devOffset <= 9)
        patterns['PIN5A'] = overwrite(patterns['PIN5A'], 22, 19, devOffset)

    if steps is not None:
        assert(steps >= 0 and steps <= 1048575)
        patterns['PIN6A'] = overwrite(patterns['PIN6A'], 22, 3, steps)

    return patterns

# TODO
def setPumpSetting(patterns, current):
    return patterns

# TODO
def setModulationInterval(tm=None, fm=None):
    """
    :param tm:

    :param fm:
    """
    return

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
        test_triangle()

    return

if __name__ == "__main__":
    main()
