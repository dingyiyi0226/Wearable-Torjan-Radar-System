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
from math import log2, floor, ceil
import RPi.GPIO as GPIO

GPIO.setmode(GPIO.BOARD)
GPIO.setwarnings(False)

# ------------------------------------------------------ #
# Define GPIO Pins                                       #
# ------------------------------------------------------ #

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
# Define ADF4158.Constant                                #
# ------------------------------------------------------ #

DEV_MAX  = (1 << 15)
REF_IN   = 10 ** 7      # 10 MHz
REF_DOUB = 0            # in [0,  1]
REF_COUN = 1            # in [1, 32]
REF_DIVD = 0            # in [0,  1]
CLK1     = 1

""" 
Equation (3)
$f_{pfd} = \text{REF_{IN}} * \frac{(1 + D)}{ R \times (1 + T) }$ 
"""
FREQ_PFD = REF_IN * (1 + REF_DOUB) / (REF_COUN * (1 + REF_DIVD))


# ------------------------------------------------------ #
# Enumerate                                              #
# ------------------------------------------------------ #

@unique
class Clock(IntEnum):
    FALLING_EDGE = 0
    RISING_EDGE  = 1

@unique
class ClkDivMode(IntEnum):
    """ DB20 - DB19 at Register 4: TEST REGISTER """
    CLOCK_DIVIDER_OFF = 0
    FAST_LOCK_DIVIDER = 1
    # RESERVED        = 2
    RAMP_DIVIDER      = 3

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
    return (1 << (start + 1)) - (1 << (end))

def overwrite(value, start, end, newValue):
    return (value & (~mask(start, end))) | (newValue << end)

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
    patterns['PIN3'] = overwrite(patterns['PIN3'], 11, 10, int(mode))
    return patterns

def setMuxout(patterns, mode: Muxout):
    patterns['PIN0'] = overwrite(patterns['PIN0'], 30, 27, int(mode))
    return patterns

def setRampAttribute(patterns, clk2=None, dev=None, devOffset=None, steps=None):
    """
    :param clk2: CLK_2 divider value at range [0, 4095]

    :param dev: Deviation words at range [-32768, 32767]

    :param devOffset: Deviation offset at range [0, 9]

    :param steps: Step words at range [0, 1048575]

    :return patterns
    """

    if clk2 is not None:
        assert(clk2 >= 0 and clk2 <= 4095)
        patterns['PIN4']  = overwrite(patterns['PIN4'], 18, 7, clk2)

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

def setPumpSetting(patterns, current):
    """
    :param current: must be the times of 0.3125 and at range [0.3125, 16 x 0.3125 = 5.0]
    """
    assert((current / 0.3125) == (current // 0.3125))

    current = int(current // 0.3125) - 1
    
    assert(current >= 0 and current <= 15)

    patterns['PIN2'] = overwrite(patterns['PIN2'], 27, 24, current)
    return patterns

def setCenterFrequency(patterns, freq, ref=10):
    """
    $$
    RF_{out} = f_{PFD} \times (\text{INT} + ( \frac{ \text{FRAC} }{ 2 ^ {25} } ))
    $$

    where
    $$
    f_{PFD} = \text{REF_{IN}} \times ( \frac{(1 + D)} { R \times (1 + T) })
    $$

    :param freq: Center frequency (MHz)

    :param span: Reference clock
    """
    frac = int((freq % ref) / ref * (1 << 25))
    
    patterns['PIN0'] = overwrite(patterns['PIN0'], 26, 15, freq // ref)
    patterns['PIN0'] = overwrite(patterns['PIN0'], 14,  3, (frac >> 13))
    patterns['PIN1'] = overwrite(patterns['PIN0'], 27, 15, (frac % (1 << 13)))
    
    return patterns

# TODO
def parsePatterns(patterns):
    return

# TODO
def setModulationInterval(patterns, centerFreq=None, bandwidth=None, tm=None, fm=None):
    """
    To determined the word of **DEV**, **DEV_OFFSET**

    :param centerFreq:

    :param bandwidth:

    :param tm:

    :param fm:
    """

    # freq_res = FREQ_PFD / (1 << 25)
    # devOffset = ceil(log2(frqe_dev / (freq_res * DEV_MAX)))
    # dev = round(freq_dev / (freq_res * (1 << devOffset)))
    
    # patterns = setCenterFrequency(centerFreq)
    # patterns = setRampAttribute(patterns, dev=dev, devOffset=devOffset)

    return patterns

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
