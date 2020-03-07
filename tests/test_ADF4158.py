import unittest

import ADF4158
from ADF4158 import Muxout, RampMode, bitMask, mask, overwrite, parseBits

module = ADF4158.ADF4158(12, 16, 18, 13, 15)


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
        module.setRamp(True)
        module.setRampMode(RampMode.CONT_TRIANGULAR)
        module.setPumpSetting(current=0.3125)
        module.setModulationInterval(centerFreq=5.75e9, bandwidth=1e8, tm=1.024e-3)
        module.setMuxout(Muxout.THREE_STATE)

        self.assertEqual(module.patterns['PIN7'],  0x00000007)
        self.assertEqual(module.patterns['PIN6A'], 0x0000A006)
        self.assertEqual(module.patterns['PIN6B'], 0x00800006)
        self.assertEqual(module.patterns['PIN5A'], 0x00120005)
        self.assertEqual(module.patterns['PIN5B'], 0x00800005)
        self.assertEqual(module.patterns['PIN4'],  0x00180104)
        self.assertEqual(module.patterns['PIN3'],  0x00000443)
        self.assertEqual(module.patterns['PIN2'],  0x0040800A)
        self.assertEqual(module.patterns['PIN1'],  0x00000001)
        self.assertEqual(module.patterns['PIN0'],  0x811F8000)

    def test_ramp915(self):
        module.setRamp(True)
        module.setRampMode(RampMode.CONT_TRIANGULAR)

        module.setPumpSetting(current=0.3125)
        module.setModulationInterval(centerFreq=9.15e8, bandwidth=1e8, tm=1.024e-3)
        module.setMuxout(Muxout.THREE_STATE)

        self.assertEqual(module.patterns['PIN7'],  0x00000007)
        self.assertEqual(module.patterns['PIN6A'], 0x0000A006)
        self.assertEqual(module.patterns['PIN6B'], 0x00800006)
        self.assertEqual(module.patterns['PIN5A'], 0x00120005)
        self.assertEqual(module.patterns['PIN5B'], 0x00800005)
        self.assertEqual(module.patterns['PIN4'],  0x00180104)
        self.assertEqual(module.patterns['PIN3'],  0x00000443)
        self.assertEqual(module.patterns['PIN2'],  0x0000800A)
        self.assertEqual(module.patterns['PIN1'],  0x00000001)
        self.assertEqual(module.patterns['PIN0'],  0x802DC000)


if __name__ == "__main__":
    unittest.main()
