"""
  FileName    [ adc.py ]
  PackageName [ Radar ] 
  Synopsis    [ ADC Protocal. ]
"""

import threading
import time
from abc import ABC, abstractmethod

import serial


class ADC(ABC):
    @abstractmethod
    def __init__(self):
        super().__init__()

class ADCConnector(ADC):
    """ ADC implemented by Arduino and pySerial. """
    def __init__(self, portName, baudrate=115200, timeout=3):
        super().__init__()

        ## PHYSICAL MODULE

        self._serial = serial.Serial(portName, baudrate=baudrate, timeout=timeout)

        ## ATTRIBUTE

        self.name = ""
        self._processor = None
        self._getname()

        ## THREADING

        self._event  = threading.Event()
        self._event.clear()
        self._thread = threading.Thread(target=self._read, daemon=True)
        self._thread.start()    

    ## Public Function

    def setProcessor(self, processor):
        assert(self._processor is None)
        self._processor = processor

    def start(self):
        self._serial.flush()
        self._event.set()
    
    def stop(self):
        self._event.clear()

    def disconnect(self):
        self._serial.close()

    ## Private Function

    def _getname(self):
        """ Get name by sending the NameCommand to Arduino """
        s = ""
        while not s.startswith('n'):
            self._serial.write(b'n ')
            s = self._serial.readline().decode().strip()
        self.name = s[1:]

    def _read(self):
        """ Read signal at anytime in other thread """

        signal = []
        isValid = True
        samplingTime = 0

        while True:
            self._event.wait()
            self._serial.write(b'r ')

            try:
                s = self._serial.readline().decode().strip()
        
                if s.startswith('i'):
                    isValid = True
                    signal.clear()

                elif s.startswith('d'):
                    try:
                        signal.extend([float(i) / 1024 for i in s[2:].split()])

                    except ValueError:
                        isValid = False

                elif s.startswith('e'):
                    try:
                        samplingTime = float(s[2:]) * 1e-6
                        
                    except ValueError:
                        isValid = False

                    # Push the data to self._manager
                    if isValid and self._processor is not None:
                        self._processor.loadData(signal, samplingTime)

                # FIXME: Unknown bugs
                elif s.startswith("Unknown Command: rr"):
                    pass

                else:
                    print('\nRead:', s)

            except UnicodeDecodeError:
                print('ADC._read: UnicodeDecodeError')

            time.sleep(0.001)
