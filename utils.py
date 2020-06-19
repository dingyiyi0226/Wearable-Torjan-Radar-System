import glob
import os
import sys

from serial.tools import list_ports


def port() -> list:
    """ Find the name of the port """
    ports = None

    ## on windows
    if sys.platform.startswith('win32'):
        ports = [ p.device for p in list_ports.comports() ]

    ## on mac
    elif sys.platform.startswith('darwin'):
        ports = glob.glob('/dev/tty.usberial-14*') + glob.glob('/dev/tty.usbmodem14*')
         
    ## on rpi
    elif (sys.platform.startswith('linux')):
        ports = glob.glob('/dev/ttyACM*') + glob.glob('/dev/ttyUSB*')
      
    return ports
 
if __name__ == "__main__":
    print(sys.platform)
    print(port())
