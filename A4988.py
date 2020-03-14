import RPi.GPIO as GPIO
import time


# STEP = 3
# DIR  = 5
# ENA  = 7     # working at enable == False


GPIO.setmode(GPIO.BOARD)
GPIO.setwarnings(False)

def pulse(pin, pauseTime):

    GPIO.output(pin, True)
    time.sleep(pauseTime)
    GPIO.output(pin, False)

class A4988:
    def __init__(self, ENA, STEP, DIR):
        self.ENA = ENA
        self.STEP = STEP
        self.DIR = DIR

        self.initGPIOPins()

    def initGPIOPins(self):
        for pin in (self.ENA, self.STEP, self.DIR):
            GPIO.setup(pin, GPIO.OUT)

    def stop(self):
        GPIO.output(self.ENA, True)

    def start(self):
        GPIO.output(self.ENA, False)

    def spin(self, deg, dir):
        # 3200 step -> 360 degrees

        self.start()

        if dir:
            GPIO.output(self.DIR, False)

        else:
            GPIO.output(self.DIR, True)

        for i in range(int(deg*3200/360)):
            pulse(self.STEP, 5e-3)

        self.stop()

def main():

    try:
        module = A4988(7, 3, 5)

        while(1):
            s = input('deg: ')
            module.spin(int(s), True)

    except KeyboardInterrupt:
        GPIO.cleanup()


if __name__=='__main__':
    main()
