import sys
from math import sin, cos, radians, pi

sys.path.insert(1, '../')
import main_control

from kivy.app import App
from kivy.cache import Cache
from kivy.clock import Clock
from kivy.config import Config
from kivy.core.window import Window
from kivy.graphics import Ellipse, Color
from kivy.properties import *
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.relativelayout import RelativeLayout
from kivy.uix.widget import Widget
from kivy.vector import Vector

class ControlPanel(RelativeLayout):
    pass

class SignalOscilloscope(RelativeLayout):
    pass

class TargetInterface(FloatLayout):
    """ Radar target interface """
    def update(self, polarCoors):
        d = 0.02 * self.width
        with self.canvas:
            # clear()
            for radius, angle, alpha in polarCoors:
                Color(0.8, 0.8, 0, alpha, mode='rgba')
                x = self.center_x + radius * cos(radians(angle)) - d / 2
                y = self.center_y + radius * sin(radians(angle)) - d / 2
                Ellipse(pos=(x, y), size=(d, d))
        

class RadarInterface(FloatLayout):
    """ Radar background interface """
    targets = ObjectProperty(None)

    def update(self, polarCoors):
        """ Clean all and add new targets to radar. """
        self.targets.update(polarCoors)

class UserInterface(FloatLayout):
    """ Root widget """
    radar = ObjectProperty(None)

    def update(self, dt: float):
        self.radar.update([ (0, 0, 1), (150, 45, 1), (150, 135, 1) ])

class RadarApp(App):
    """ ... """
    def __init__(self, frame: float):
        super(RadarApp, self).__init__()
        self.frameRate = frame

    def build(self):
        # Window.maximize()
        # Clock.max_iteration = 20

        # Launch the root widget and set clock
        interface = UserInterface()
        Clock.schedule_interval(interface.update, 1.0 / self.frameRate)

        return interface

if __name__ == "__main__":
    RadarApp(frame=1).run()
