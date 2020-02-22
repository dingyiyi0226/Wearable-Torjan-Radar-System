import sys
from math import sin, cos

sys.path.insert(1, '../')
import main_control

from kivy.app import App
from kivy.cache import Cache
from kivy.clock import Clock
from kivy.config import Config
from kivy.core.window import Window
from kivy.properties import *
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.relativelayout import RelativeLayout
from kivy.uix.widget import Widget
from kivy.vector import Vector

class Target(Widget):
    """
    ...
    """
    def __init__(self, distance=0, angle=0, v=0):
        super(Target, self).__init__()

        self.pos = (distance * cos(angle), distance * sin(angle))
        self.vx  = NumericProperty(v)
        self.vy  = NumericProperty(v)
        self.v   = ReferenceListProperty(self.v, self.v)

    def move(self):
        pass

class ControlPanel(RelativeLayout):
    pass

class SignalOscilloscope(RelativeLayout):
    pass

class RadarInterface(RelativeLayout):
    """ Circular radar Interface """
    def update(self):
        pass

    def on_touch_down(self, touch):
        print(self.size, min(self.size))

    # def doIt(self, newTargets):
    #     self.clear_widgets()
    #     for target in newTargets:
    #         self.add_widget(target)

class UserInterface(FloatLayout):
    """
    Attributes of each widget should be add as <WidgetName> in .kv file.

    Notes: Root of widgets. 
    """
    # radar = ObjectProperty(RadarInterface())

    def __init__(self):
        super(UserInterface, self).__init__()

        radar = RadarInterface()
        radar.height, radar.width = min(self.size), min(self.size)
        
        self.add_widget(radar)

    def update(self, dt: float):
        self.radar.update()

class RadarApp(App):
    """ 
    Notes: Read radar.kv automatically.
    """
    def __init__(self, frame: float):
        super(RadarApp, self).__init__()
        self.frameRate = frame

    def build(self):
        # Launch the children widgets
        interface = UserInterface()
        Clock.schedule_interval(interface.update, 1.0 / self.frameRate)
        return interface

if __name__ == "__main__":
    Window.maximize()
    RadarApp(frame=60.0).run()
