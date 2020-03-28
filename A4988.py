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

    def __init__(self, pins):
        self.ENA = pins['ENA']
        self.STEP = pins['STEP']
        self.DIR = pins['DIR']

        self.initGPIOPins()

    def initGPIOPins(self):
        for pin in (self.ENA, self.STEP, self.DIR):
            GPIO.setup(pin, GPIO.OUT)

    def _stop(self):
        GPIO.output(self.ENA, True)

    def _start(self):
        GPIO.output(self.ENA, False)

    def spin(self, deg, clkwise):
        # 3200 step -> 360 degrees

        self._start()
        GPIO.output(self.DIR, clkwise)

        for i in range(int(deg*3200/360)):
            pulse(self.STEP, 5e-3)

        self._stop()

    def spinBnF(self, iter, deg):
        """ spin back and forth """

        for i in range(iter):
            self.spin(deg, True)
            time.sleep(0.5)
            self.spin(deg, False)
            time.sleep(0.5)

    def spinSteps(self, deg, step):
        """ spin `deg` degrees for `step` steps """

        for i in range(step):
            self.spin(deg, True)
            time.sleep(0.5)


def main():

    DIR_PINS = {
        'STEP': 3,
        'DIR' : 5,
        'ENA' : 7,
    }

    try:
        module = A4988(DIR_PINS)

        # module.spinBnF(iter=20, deg=180)
        module.spinSteps(deg=10, step=36*5)

        # while(1):

        #     s = input('deg: ')
        #     t = input('dir(0, 1): ')
        #     module.spin(int(s), int(t))

    except KeyboardInterrupt:
        GPIO.cleanup()


if __name__=='__main__':
    main()
