"""
  Filename      [ controller.py ]
  PackageName   [ Radar ]
  Synopsis      [ ADF4158 Signal Generator Control Module ]
"""

from collections import OrderedDict
from enum import Enum, IntEnum, unique
from functools import wraps
from math import ceil, floor, log2

import RPi.GPIO as GPIO

GPIO.setmode(GPIO.BOARD)
GPIO.setwarnings(False)

""" 
Equation (3)
$f_{pfd} = \text{REF_{IN}} * \frac{(1 + D)}{ R \times (1 + T) }$ 
"""

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

@unique
class Prescaler(IntEnum):
    """ DB22 at Register 2: R-DIVIDEr REGISTER """
    PRESCALER45 = 0
    PRESCALER89 = 1

# ------------------------------------------------------ #
# Helper function                                        #
# ------------------------------------------------------ #

def bitMask(pos):
    """ 0...010...0 Mask. """
    return (1 << pos)

def mask(start, end=0):
    """ 0...011...1 Mask. """
    return (1 << (start + 1)) - (1 << (end))

def overwrite(value, start, end, newValue):
    """ Rewrite the bits at given a 32-bit length bit sequences """
    assert (newValue >> (start - end + 1) == 0)
    return (value & (~mask(start, end))) | (newValue << end)

def parseBits(value, start, end):
    """ Get the sub-bitSequences """
    return (value & (mask(start, end))) >> end

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


class ADF4158:
    DEV_MAX  = (1 << 15)
    REF_IN   = 1e7          # 10 MHz

    register_pins = set()   # Static sets for maintaining used pins

    def __init__(self, CLK, DATA, LE, TXDATA, MUXOUT):
        # ------------------------------------------------------ #
        # Define GPIO Pins                                       #
        # ------------------------------------------------------ #

        # self.GND  = 6      # T3
        self.W_CLK  = CLK    # T4
        self.DATA   = DATA   # T5
        self.LE     = LE     # T6
        self.TXDATA = TXDATA # T16
        self.MUXOUT = MUXOUT # T8

        # PINs Registration
        for pin in (self.W_CLK, self.DATA, self.LE, self.TXDATA, self.MUXOUT):
            if pin in self.register_pins:
                raise ValueError("Repeat PINs in ADF4158")

            self.register_pins.add(pin)

        # ------------------------------------------------------ #
        # Define ADF4158.Constant                                #
        # ------------------------------------------------------ #

        self.REF_DOUB = 0    # in [0,  1]
        self.REF_COUN = 1    # in [1, 32]
        self.REF_DIVD = 0    # in [0,  1]
        self.CLK1     = 1
        self.CLK2     = 2

        self.FREQ_PFD = int(self.REF_IN * (1 + self.REF_DOUB) / (self.REF_COUN * (1 + self.REF_DIVD)))

        self.reset()

    # ------------------------------------------------------ #
    # Private Function                                       #
    # ------------------------------------------------------ #


    def setReadyToWrite(self):
        GPIO.output(self.W_CLK, False)
        GPIO.output(self.DATA, False)
        GPIO.output(self.LE, False)

    # TODO
    def readWord(self):
        """ Readback words from MUXOUT """
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

    # @verbose
    def sendWord(self, word, clk=Clock.RISING_EDGE):
        """
        :param word: 32-bits information

        :param clk: { Clock.RISING_EDGE, Clock.FALLING_EDGE } optional
        """
        
        # Raise Clock after setup DATA 
        for i in range(31, 0, -1):
            GPIO.output(self.DATA, bool((word >> i) % 2))
            pulseHigh(self.W_CLK)

        # Hold LE for last clock
        GPIO.output(self.DATA, bool(word % 2))
        GPIO.output(self.W_CLK, True)
        GPIO.output(self.LE, True)
        GPIO.output(self.W_CLK, False)
        GPIO.output(self.LE, False)

        # Reset Data as 0
        GPIO.output(self.DATA, False)

        return True

    def initBitPatterns(self):
        """ Initialize bit patterns """
        self.patterns = OrderedDict()

        self.patterns['PIN7']  = 0x00000007
        self.patterns['PIN6A'] = 0x00000006
        self.patterns['PIN6B'] = 0x00800006
        self.patterns['PIN5A'] = 0x00000005
        self.patterns['PIN5B'] = 0x00800005
        self.patterns['PIN4']  = 0x00180104
        self.patterns['PIN3']  = 0x00000043
        self.patterns['PIN2']  = 0x0040800A
        self.patterns['PIN1']  = 0x00000001
        self.patterns['PIN0']  = 0x01220000

    def initGPIOPins(self):
        """ GPIO Pins initialization for ADF4158 config. """
        for pin in (self.W_CLK, self.LE, self.DATA):
            GPIO.setup(pin, GPIO.OUT)
        
        for pin in (self.TXDATA, self.MUXOUT, ):
            GPIO.setup(pin, GPIO.IN)

    def reset(self):
        """ 
        Initial ADF4851 Signal Generator

        :return patterns: the initial patterns wrote in ADF4158

        .. References:
            (Datasheet p.25)
        """
        
        self.initGPIOPins()
        self.setReadyToWrite()
        self.initBitPatterns()

        for value in self.patterns.values():
            self.sendWord(value)

    def setRamp(self, status: bool):
        self.patterns['PIN0'] = overwrite(self.patterns['PIN0'], 31, 31, int(status))
        
    def setRampMode(self, mode: RampMode):
        self.patterns['PIN3'] = overwrite(self.patterns['PIN3'], 11, 10, int(mode))
        
    def setMuxout(self, mode: Muxout):
        self.patterns['PIN0'] = overwrite(self.patterns['PIN0'], 30, 27, int(mode))
        
    def setRampAttribute(self, clk2=None, dev=None, devOffset=None, steps=None):
        """
        :param clk2: CLK_2 divider value at range [0, 4095]

        :param dev: Deviation words at range [-32768, 32767]

        :param devOffset: Deviation offset at range [0, 9]

        :param steps: Step words at range [0, 1048575]

        :return patterns
        """

        if clk2 is not None:
            assert(clk2 >= 0 and clk2 <= 4095)
            self.patterns['PIN4']  = overwrite(self.patterns['PIN4'], 18, 7, clk2)

        if dev is not None:
            assert(dev >= -32768 and dev <= 32767)
            self.patterns['PIN5A'] = overwrite(self.patterns['PIN5A'], 18, 3, dev)

        if devOffset is not None:
            assert(devOffset >= 0 and devOffset <= 9)
            self.patterns['PIN5A'] = overwrite(self.patterns['PIN5A'], 22, 19, devOffset)

        if steps is not None:
            assert(steps >= 0 and steps <= 1048575)
            self.patterns['PIN6A'] = overwrite(self.patterns['PIN6A'], 22, 3, steps)

    def setPumpSetting(self, current):
        """
        :param current: must be the times of 0.3125 and at range [0.3125, 16 x 0.3125 = 5.0]
        """
        assert((current / 0.3125) == (current // 0.3125))

        current = int(current // 0.3125) - 1
        
        assert(current >= 0 and current <= 15)

        self.patterns['PIN2'] = overwrite(self.patterns['PIN2'], 27, 24, current)
        
    def setCenterFrequency(self, freq, ref=1e7):
        """
        $$
        RF_{out} = f_{PFD} \times (\text{INT} + ( \frac{ \text{FRAC} }{ 2 ^ {25} } ))
        $$

        where
        $$
        f_{PFD} = \text{REF_{IN}} \times ( \frac{(1 + D)} { R \times (1 + T) })
        $$

        :param freq: Center frequency

        :param ref: Reference clock frequency
        """
        frac = int((freq % ref) / ref * (1 << 25))
        frac_MSB = (frac >> 13)
        frac_LSB = (frac % (1 << 13))

        self.patterns['PIN0'] = overwrite(self.patterns['PIN0'], 26, 15, int(freq // ref))
        self.patterns['PIN0'] = overwrite(self.patterns['PIN0'], 14,  3, frac_MSB)
        self.patterns['PIN1'] = overwrite(self.patterns['PIN1'], 27, 15, frac_LSB)
        
        prescaler = Prescaler.PRESCALER89 if freq > 3e9 else Prescaler.PRESCALER45
        self.patterns['PIN2'] = overwrite(self.patterns['PIN2'], 22, 22, prescaler)

    def setModulationInterval(self, centerFreq, bandwidth, tm):
        """
        To determined the word of **DEV**, **DEV_OFFSET**.

        Optimize
        --------
        - Steps: As much as possible to form a approx. linear wave
        - DevOffset: As low as possible

        :param centerFreq:

        :param bandwidth:

        :param tm:
        """

        f_res = self.FREQ_PFD / (1 << 25)
        steps = int(tm * self.FREQ_PFD / (self.CLK1 * self.CLK2))
        f_dev = bandwidth / steps

        dev = round(f_dev / f_res)
        devOffset = 0

        while dev > 32767:
            dev = dev >> 1
            devOffset += 1

        self.setCenterFrequency(int(centerFreq))
        self.setRampAttribute(dev=dev, devOffset=devOffset, steps=steps)

def set5800Default(module):
    module.initBitPatterns()

    module.setRamp(True)
    module.setRampMode(RampMode.CONT_TRIANGULAR)

    module.setPumpSetting(current=0.3125)
    module.setModulationInterval(centerFreq=5.75e9, bandwidth=1e8, tm=1.024e-3)
    module.setMuxout(Muxout.THREE_STATE)

    return module

def set915Default():
    module.initBitPatterns(module)

    module.setRamp(True)
    module.setRampMode(RampMode.CONT_TRIANGULAR)

    module.setPumpSetting(current=0.3125)
    module.setModulationInterval(centerFreq=9.15e8, bandwidth=1e8, tm=1.024e-3)
    module.setMuxout(Muxout.THREE_STATE)
    
    return module

def main():
    module = set5800Default()
    
if __name__ == "__main__":
    main()
