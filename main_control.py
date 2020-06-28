import os
from datetime import datetime

from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation

from radar import FMCWRadar, Troy
from RPi import GPIO
from view import PPIView, SigView, ObjView


def main():

    ## For File Writing (Command: Save)
    now = datetime.today().strftime('%Y%m%d')

    ## For Matplotlib
    views = {}

    ## Initialize troy model

    troy = Troy()
    troy.getInfo()

    try:
        prompt = ''

        while True:
            s = input("commands: " + prompt).strip()

            if s == '':
                pass

            elif s.startswith('read'):
                troy.start()

            elif s.startswith('stop'):
                troy.stop()

            elif s.startswith('setbg'):
                # If s contains one argument only, overwrite background signal.
                # Otherwise, take average of previous background signal

                if len(s.split()) == 1:
                    print('Reset Background Signal')
                    troy.setBgSignal(overwrite=True)
                else:
                    print('Take Average on Background Signal')
                    troy.setBgSignal(overwrite=False)

            elif s.startswith('clearbg'):
                # If s contains one argument only, overwrite background signal.
                # Otherwise, take average of previous background signal

                print('Clear Background Signal')
                troy.resetBgSignal()
                
            elif s.startswith('sig'):
                # Open SigView (Oscillscope)

                if ("Oscilloscope-5.8" not in views) and (isinstance(troy.highFreqRadar, FMCWRadar)):
                    view = SigView(timeYMax=1, freqYMax=0.05, avgFreqYMax=5e-4,
                        maxFreq=4e3, maxTime=0.24, figname='Waveform: 5.8GHz')
                    animation = FuncAnimation(view.fig, view.update, init_func=view.init, interval=200, blit=True,
                        fargs=(troy.highFreqRadar.realTimeSig, ))
                    view.figShow()

                    views["Oscilloscope-5.8"] = (view, animation)

                if ("Oscilloscope-915" not in views) and (isinstance(troy.lowFreqRadar, FMCWRadar)):
                    view = SigView(timeYMax=1, freqYMax=0.1, avgFreqYMax=1e-4,
                        maxFreq=4e3, maxTime=0.24, figname='Waveform: 915MHz')
                    animation = FuncAnimation(view.fig, view.update, init_func=view.init, interval=200, blit=True,
                        fargs=(troy.lowFreqRadar.realTimeSig, ))
                    view.figShow()

                    views["Oscilloscope-915"] = (view, animation)

            elif s.startswith('obj'):
                # Open Objview

                if 'OBJ' in views:
                    continue

                view = ObjView(maxR=100, maxV=30)
                animation = FuncAnimation(view.fig, view.update,
                    init_func=view.init, interval=200, blit=True,
                    fargs=(troy.objectInfo, ))
                view.figShow()

                # Record down the view
                views['OBJ'] = (view, animation)

            elif s.startswith('ppi'):
                # Open PPIView (Object Inferencing)

                if 'PPI' in views:
                    continue

                view = PPIView(maxR=25)
                animation = FuncAnimation(view.fig, view.update,
                    init_func=view.init, interval=200, blit=True,
                    fargs=(troy.objectInfo, ))
                view.figShow()

                # Record down the view
                views['PPI'] = (view, animation)

            elif s.startswith('close'):
                for view, _ in views.values(): plt.close(view.fig)
                views.clear()

            elif s.startswith('save'):
                ## Save time domain signal
                
                distance = input('Distances: ').strip() if len(s.split()) == 1 else s.split()[1]

                path = './rawdata/arduino/{}'.format(now)
                if not os.path.exists(path): os.makedirs(path)
                troy.save(
                    os.path.join(path, 'high-' + distance + '.csv'),
                    os.path.join(path, 'low-' + distance + '.csv')
                )
                print(" > File is saved! Check at: {}".format(path))

            elif s.startswith('setdirection'):
                direction = input('Direction: ').strip() if len(s.split()) == 1 else s.split()[1]

                try:
                    direction = float(direction)
                    troy.setDirection(direction)

                except ValueError:
                    print('invalid direction')

            elif s.startswith('resetdirection'):
                troy.resetDirection()

            elif s.startswith('flush'):
                troy.flush()

            elif s.startswith('info'):
                troy.getInfo()

            elif s.startswith('track'):
                troy.tracking()

            elif s.startswith('bgsig'):
                # Open Background SigView (Oscillscope)

                for channel in troy.availableChannels:
                    # Reject repeated views
                    if str(channel) + '-bg' in views:
                        continue

                    view = SigView(maxAmplitude=1, maxFreq=4e3, maxTime=0.25, figname='Background: {}'.format(str(channel)))
                    animation = FuncAnimation(view.fig, view.update,
                        init_func=view.init, interval=200, blit=True,
                        fargs=(channel.backgroundSig, ))
                    view.figShow()

                    # Record down the view
                    views[str(channel) + '-bg'] = (view, animation)

            elif s.startswith('q'):
                break

            else:
                print('Undefined Command')

    except KeyboardInterrupt:
        pass

    except Exception as e:
        print(e)

    finally:
        for view, _ in views.values():
            plt.close(view.fig)
        views.clear()
        troy.close()
        GPIO.cleanup()

        print('Quit main')

if __name__ == '__main__':
    main()
