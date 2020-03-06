import unittest
from ADF4158 import *

class TestBitModifyMethod(unittest.TestCase):
    def test_bitMask(self):
        self.assertEqual(bitMask(0),  0x00000001)
        self.assertEqual(bitMask(31), 0x80000000)
        self.assertEqual(bitMask(16), 0x00010000)

    def test_mask(self):
        self.assertEqual(mask( 0, 0), 0x00000001)
        self.assertEqual(mask( 1, 0), 0x00000003)
        self.assertEqual(mask( 3, 2), 0x0000000C)
        self.assertEqual(mask(11, 3), 0x00000FF8)
    
    def test_overwrite(self):
        self.assertEqual(overwrite(0x12345678, 31, 24, 0), 0x00345678)
        self.assertEqual(overwrite(0x12345678, 31, 12, 0), 0x00000678)
        self.assertEqual(overwrite(0x12345678, 11,  0, 0), 0x12345000)

    def test_praseBits(self):
        self.assertEqual(parseBits(0x12345678, 31, 24), 0x12)

    def test_ramp5800(self):
        patterns = initBitPatterns()

        patterns = setRamp(patterns, True)
        patterns = setRampMode(patterns, RampMode.CONT_TRIANGULAR)
        
        patterns = setPumpSetting(patterns, current=2.5)
        patterns = setModulationInterval(patterns, centerFreq=5.75e9, bandwidth=1e8, tm=1.024e-3)
        patterns = setMuxout(patterns, Muxout.THREE_STATE)

        self.assertEqual(patterns['PIN7'],  0x00000007)
        self.assertEqual(patterns['PIN6A'], 0x0000A006)
        self.assertEqual(patterns['PIN6B'], 0x00800006)
        self.assertEqual(patterns['PIN5A'], 0x00120005)
        self.assertEqual(patterns['PIN5B'], 0x00800005)
        self.assertEqual(patterns['PIN4'],  0x00180104)
        self.assertEqual(patterns['PIN3'],  0x00000443)
        self.assertEqual(patterns['PIN2'],  0x0740800A)
        self.assertEqual(patterns['PIN1'],  0x00000001)
        self.assertEqual(patterns['PIN0'],  0x811F8000)

    def test_ramp915(self):
        patterns = initBitPatterns()

        patterns = setRamp(patterns, True)
        patterns = setRampMode(patterns, RampMode.CONT_TRIANGULAR)
        
        patterns = setPumpSetting(patterns, current=2.5)
        patterns = setModulationInterval(patterns, centerFreq=9.15e8, bandwidth=1e8, tm=1.024e-3)
        patterns = setMuxout(patterns, Muxout.THREE_STATE)

        self.assertEqual(patterns['PIN7'],  0x00000007)
        self.assertEqual(patterns['PIN6A'], 0x0000A006)
        self.assertEqual(patterns['PIN6B'], 0x00800006)
        self.assertEqual(patterns['PIN5A'], 0x00120005)
        self.assertEqual(patterns['PIN5B'], 0x00800005)
        self.assertEqual(patterns['PIN4'],  0x00180104)
        self.assertEqual(patterns['PIN3'],  0x00000443)
        self.assertEqual(patterns['PIN2'],  0x0700800A)
        self.assertEqual(patterns['PIN1'],  0x00000001)
        self.assertEqual(patterns['PIN0'],  0x802DC000)


if __name__ == "__main__":
    unittest.main()
