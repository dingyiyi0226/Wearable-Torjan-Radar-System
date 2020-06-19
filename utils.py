import os
import sys

def port() -> str:
    """ Find the name of the port """

    try:
        ## on mac
        if sys.platform.startswith('darwin'):
            ports = os.listdir('/dev/')
            
            for i in ports:
                if i.startswith('tty.usbserial-14') or i.startswith('tty.usbmodem14'):
                    port = i
                    break
            
            port = '/dev/' + port
            
        ## on rpi
        if (sys.platform.startswith('linux')):
            ports = os.listdir('/dev/')
            
            for i in ports:
                if i.startswith('ttyUSB') or i.startswith('ttyACM'):
                    port = i
                    break

            port = '/dev/' + port

    except UnboundLocalError:
        sys.exit('Cannot open port')

    return port
