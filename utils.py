import os
import glob
import sys

def port() -> list:
    """ Find the name of the port """
    ports = None

    ## on mac
    if sys.platform.startswith('darwin'):
        ports = glob.glob('/dev/tty.usberial-14*') + glob.glob('/dev/tty.usbmodem14')
         
    ## on rpi
    elif (sys.platform.startswith('linux')):
        ports = glob.glob('/dev/ttyACM*') + glob.glob('/dev/ttyUSB*')
      
    return ports
 
if __name__ == "__main__":
    print(port())
