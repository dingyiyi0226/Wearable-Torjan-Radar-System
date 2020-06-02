import enum
import config
import RPi.GPIO as GPIO
import time
import numpy as np

ScanMode = 0

# gain channel
GAIN_E = {
    'GAIN_1' : 0,  # GAIN   1
    'GAIN_2' : 1,  # GAIN   2
    'GAIN_4' : 2,  # GAIN   4
    'GAIN_8' : 3,  # GAIN   8
    'GAIN_16' : 4, # GAIN  16
    'GAIN_32' : 5, # GAIN  32
    'GAIN_64' : 6, # GAIN  64
}

# data rate
DRATE_E = {
    '30000SPS' : 0xF0, # reset the default values
    '15000SPS' : 0xE0,
    '7500SPS' : 0xD0,
    '3750SPS' : 0xC0,
    '2000SPS' : 0xB0,
    '1000SPS' : 0xA1,
    '500SPS' : 0x92,
    '100SPS' : 0x82,
    '60SPS' : 0x72,
    '50SPS' : 0x63,
    '30SPS' : 0x53,
    '25SPS' : 0x43,
    '15SPS' : 0x33,
    '10SPS' : 0x20,
    '5SPS' : 0x13,
    '2d5SPS' : 0x03
}

# registration definition
REG_E = {
    'REG_STATUS' : 0,  # x1H
    'REG_MUX' : 1,     # 01H
    'REG_ADCON' : 2,   # 20H
    'REG_DRATE' : 3,   # F0H
    'REG_IO' : 4,      # E0H
    'REG_OFC0' : 5,    # xxH
    'REG_OFC1' : 6,    # xxH
    'REG_OFC2' : 7,    # xxH
    'REG_FSC0' : 8,    # xxH
    'REG_FSC1' : 9,    # xxH
    'REG_FSC2' : 10,   # xxH
}

# command definition
CMD = {
    'CMD_WAKEUP' : 0x00,     # Completes SYNC and Exits Standby Mode 0000  0000 (00h)
    'CMD_RDATA' : 0x01,      # Read Data 0000  0001 (01h)
    'CMD_RDATAC' : 0x03,     # Read Data Continuously 0000   0011 (03h)
    'CMD_SDATAC' : 0x0F,     # Stop Read Data Continuously 0000   1111 (0Fh)
    'CMD_RREG' : 0x10,       # Read from REG rrr 0001 rrrr (1xh)
    'CMD_WREG' : 0x50,       # Write to REG rrr 0101 rrrr (5xh)
    'CMD_SELFCAL' : 0xF0,    # Offset and Gain Self-Calibration 1111    0000 (F0h)
    'CMD_SELFOCAL' : 0xF1,   # Offset Self-Calibration 1111    0001 (F1h)
    'CMD_SELFGCAL' : 0xF2,   # Gain Self-Calibration 1111    0010 (F2h)
    'CMD_SYSOCAL' : 0xF3,    # System Offset Calibration 1111   0011 (F3h)
    'CMD_SYSGCAL' : 0xF4,    # System Gain Calibration 1111    0100 (F4h)
    'CMD_SYNC' : 0xFC,       # Synchronize the A/D Conversion 1111   1100 (FCh)
    'CMD_STANDBY' : 0xFD,    # Begin Standby Mode 1111   1101 (FDh)
    'CMD_RESET' : 0xFE,      # Reset to Power-Up Values 1111   1110 (FEh)
}

class ADS1256:
    def __init__(self):
        self.rst_pin    = config.RST_PIN
        self.cs_pin     = config.CS_PIN
        self.drdy_pin   = config.DRDY_PIN
        self.channel    = None
        self.mode       = None

    # ---------------------------------------------------------- # 
    # Private Function of ADS1256                                #
    # ---------------------------------------------------------- #
    
    def _writeCmd(self, reg):
        config.digital_write(self.cs_pin, GPIO.LOW)
        config.spi_writebyte([reg])
        config.digital_write(self.cs_pin, GPIO.HIGH)
    
    def _writeReg(self, reg, data):
        config.digital_write(self.cs_pin, GPIO.LOW)
        config.spi_writebyte([CMD['CMD_WREG'] | reg, 0x00, data])
        config.digital_write(self.cs_pin, GPIO.HIGH)

    def _readReg(self, reg):
        config.digital_write(self.cs_pin, GPIO.LOW)
        config.spi_writebyte([CMD['CMD_RREG'] | reg, 0x00])
        data = config.spi_readbytes(1)
        config.digital_write(self.cs_pin, GPIO.HIGH)

        return data
        
    def WaitDRDY(self, timeout=400000):
        """ 
        Return True if ADS1256 data is ready. (DRDY) 

        Argument
        --------
        timeout: int
            Waiting time in (ms)
        
        Return
        ------
        status: bool
            Return True if ready or False if timeout.
        """
        for i in range(timeout):
            if (config.digital_read(self.drdy_pin) == 0):
                return True

        return False
        
    def ReadChipID(self):
        self.WaitDRDY()
        id = self._readReg(REG_E['REG_STATUS'])
        id = id[0] >> 4
        
        return id

    # ---------------------------------------------------------- # 
    # Public Function of ADS1256                                 #
    #   - Configuration                                          # 
    # ---------------------------------------------------------- #
        
    # Hardware reset
    def reset(self):
        config.digital_write(self.rst_pin, GPIO.HIGH)
        config.delay_ms(200)
        config.digital_write(self.rst_pin, GPIO.LOW)
        config.delay_ms(200)
        config.digital_write(self.rst_pin, GPIO.HIGH)

    def ConfigADC(self, gain, drate):
        """ The configuration parameters of ADC, gain and data rate """
        self.WaitDRDY()
        buf = [0, 0, 0, 0, 0, 0, 0, 0]
        buf[0] = (0<<3) | (1<<2) | (0<<1)
        buf[1] = 0x08
        buf[2] = (0<<5) | (0<<3) | (gain<<0)
        buf[3] = drate
        
        config.digital_write(self.cs_pin, GPIO.LOW)
        config.spi_writebyte([CMD['CMD_WREG'] | 0, 0x03])
        config.spi_writebyte(buf)
        
        config.digital_write(self.cs_pin, GPIO.HIGH)
        config.delay_ms(1) 

    def SetChannal(self, Channal):
        """ Set the desired input channel number. """
        assert(Channal >= 0 and Channal < 8 and isinstance(Channal, int))
        self._writeReg(REG_E['REG_MUX'], (Channal << 4) | (1 << 3))

    def SetDiffChannal(self, Channal):
        """ 
        Set the desired input channel number for differential mode 
        
        Argument
        --------
        Channel : int
            Channel in option {0, 1, 2, 3}
        """
        # DiffChannal AIN0-AIN1, AIN2-AIN3, AIN4-AIN5 or AIN6-AIN7
        self._writeReg(REG_E['REG_MUX'], ((2 * Channal) << 4) | (2 * Channal + 1)) 	

    def SetMode(self, Mode):
        ScanMode = Mode

    def init(self):
        if (config.module_init() != 0):
            return -1

        self.reset()
        id = self.ReadChipID()
        if id == 3 :
            print("ID Read success  ")
        else:
            print("ID Read failed   ")
            return -1

        self.ConfigADC(GAIN_E['GAIN_1'], DRATE_E['30000SPS'])
        return 0
        
    # ---------------------------------------------------------- # 
    # Public Function of ADS1256                                 #
    #   - RDATA                                                  # 
    # ---------------------------------------------------------- #

    def Read_ADC_Data(self):
        self.WaitDRDY()
        config.digital_write(self.cs_pin, GPIO.LOW)#cs  0
        config.spi_writebyte([CMD['CMD_RDATA']])
        # config.delay_ms(10)

        buf = config.spi_readbytes(3)
        config.digital_write(self.cs_pin, GPIO.HIGH)#cs 1
        read = (buf[0]<<16) & 0xff0000
        read |= (buf[1]<<8) & 0xff00
        read |= (buf[2]) & 0xff

        if (read & 0x800000):
            read &= 0xF000000

        return read

    def GetChannalValue(self, Channel=None):
        """
        Arguments
        ---------
        Channel : {None, int} optional
            Switch the MUX if needed.
        """
        if (ScanMode == 0):# 0  Single-ended input  8 channel1 Differential input  4 channe 
            if (Channel is not None):
                if (Channel >= 8):
                    return 0
                self.SetChannal(Channel)
                self._writeCmd(CMD['CMD_SYNC'])
                self._writeCmd(CMD['CMD_WAKEUP'])

            Value = self.Read_ADC_Data()
        else:
            if (Channel is not None):
                if(Channel >= 4):
                    return 0
                self.SetDiffChannal(Channel)
                self._writeCmd(CMD['CMD_SYNC'])
                # config.delay_ms(10) 
                self._writeCmd(CMD['CMD_WAKEUP'])
                # config.delay_ms(10) 

            Value = self.Read_ADC_Data()

        return Value

    def GetAllChannalValue(self):
        ADC_Value = [0, 0, 0, 0, 0, 0, 0, 0]
        for i in range(8):
            ADC_Value[i] = self.GetChannalValue(i)

        return ADC_Value

    # ---------------------------------------------------------- # 
    # Public Function of ADS1256                                 #
    #   - RDATAC                                                 # 
    # ---------------------------------------------------------- #

    def Start_Read_ADC_Data_Continuous(self):
        """ Change to RDATAC mode """
        self.WaitDRDY()
        config.digital_write(self.cs_pin, GPIO.LOW)
        config.spi_writebyte([CMD['CMD_RDATAC']])
        config.digital_write(self.cs_pin, GPIO.HIGH)

    def Read_ADC_Data_WithoutCommand(self):
        """ Running in RDATAC mode """
        self.WaitDRDY()
        config.digital_write(self.cs_pin, GPIO.LOW)
        buf = config.spi_readbytes(3)
        config.digital_write(self.cs_pin, GPIO.HIGH)
        read  = (buf[0] << 16) & 0xff0000
        read |= (buf[1] << 8)  & 0xff00
        read |= (buf[2])       & 0xff

        if (read & 0x800000):
            read &= 0xF000000  

        return read

    def Read_ADC_Data_Continuous(self, npoints):
        buf = np.empty(npoints)
        timestamp = time.time()
        for i in range(npoints):
            buf[i] = self.Read_ADC_Data_WithoutCommand()
        timedelta = time.time() - timestamp
        fs = npoints / timedelta

        return buf, fs

    def Stop_Read_ADC_Data_Continuous(self):
        """ Stop RDATAC mode """
        self.ADS1256_WaitDRDY()
        config.digital_write(self.cs_pin, GPIO.LOW)
        config.spi_writebyte([CMD['CMD_SDATAC']])
        config.digital_write(self.cs_pin, GPIO.HIGH)

### END OF FILE ###
