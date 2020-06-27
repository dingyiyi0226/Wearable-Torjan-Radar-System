# Instruction

> Link to [Repository](https://github.com/dingyiyi0226/Wearable-Trojan-Radar-System)

> Link to [Report](./wearable-trojan-radar-system.docx)

> Link to [README.md](./README.md)

## Abstract (0.5 pages)

Wearable-Trojan-Radar-System (Troy) is a dual-band radar system that runs on 5.8 GHz and 915 MHz. It has the following features: 

1. ~~Dual-band~~
2. Opeating at ISM Band and avoid using frequently used band.
3. Real-time detecting
4. Portable and DIY friendly
5. Interactive and enable to review
6. Design for indoor demonstration.

Troy is made for education purpose, demonstrate radar operation property to non-specialized in convenient way. Once who want to understand how radar operate can rebuild, modify or extend Troy easily.

## Setup (2.5 - 3 pages)

- [ ] System Overview (0.5 pages)
  - [ ] Hardware
    - [ ] How to achieve wearable and demo the radar property at small room?
    - [ ] Why using Speed Gen and Distance Geneartor
    - [ ] ![System-Architecture](./figure-architecture.png)
  - [ ] Software
    - [ ] Real time detection based on DFT
      - [ ] ADC Sampling Rate and Duration, Resolution
      - [ ] Setting background, remove DC 
      - [ ] Range and Velocity Estimation Principle
    - [ ] Why designing peripheral like this?
      - [ ] Demo 4 properties of FMCW Radar
- [ ] Material List (Suggest ref. to report) 
  - [ ] Radar
    - [ ] ADF4158 x 2
    - [ ] Gain Controler with Limiter (Fit the operation voltage)
    - [ ] Antenna
    - [ ] Mixer, PA, LNA (number of LNA)
  - [ ] Coaxial cable x N
  - [ ] Design BluePrint
  - [ ] RPi & Arduino
    - [ ] PinOut
  - [ ] Peripheral
    - [ ] Bluetooth Module
- [ ] Manufactoring (0.25 pages)
  - [ ] PCB
- [ ] Soldering & Connection (1 page)
- [ ] System (Rpi & Arduino) (0.25 - 0.5 pages)
  - [ ] Network Setup & Login (For Visualization, Recommend: SSH by MobaXTerm / VNC)
  - [ ] [Repository](https://github.com/dingyiyi0226/Wearable-Trojan-Radar-System.git) (Git download)
  - [ ] Installation: 
    - [ ] Python (3.7 upper) 
    - [ ] Pip install dependancy (Related library)
    - [ ] Write arduino program
- [ ] Peripheral (Person, Reflector) (0.5 pages)

### System Overview (0.5 pages)

![](./figure/system-architecture.png)

### Setup Software (0.25 pages)

Our source code is available at [Repository](https://github.com/dingyiyi0226/Wearable-Trojan-Radar-System.git). It's implemented by Python 3, suggest to use Python 3.7 or higher version to execute it. Please move to the folder and type `python3 -m pip install -r requirements.txt` at `cmd` to check the library dependancy.

2 ADCs are implemented by Arduino Mega250. Use any device with Arduino IDE to compile, update code to ADC. We named the code with two names, `ADC-915MHz` and `ADC-5.8GHz` for each channel. Please upload the corresponding code refer to which IF signal ADC attached.

Raspberry PI Routing Table: 

|   OUTPUT   |  PIN  |  PIN  |  OUTPUT   |
| :--------: | :---: | :---: | :-------: |
|            |   1   |   2   |           |
|            |   3   |   4   |           |
|  LOW-T16   |   5   |   6   |           |
|   LOW-T8   |   7   |   8   |  HIGH-T4  |
|   LOW-T3   |   9   |  10   |  HIGH-T5  |
|   LOW-T4   |  11   |  12   |  HIGH-T6  |
|   LOW-T5   |  13   |  14   |  HIGH-T3  |
|   LOW-T6   |  15   |  16   | HIGH-T16  |
|            |  17   |  18   |  HIGH-T8  |
|            |  19   |  20   |           |
|            |  21   |  22   |           |
|            |  23   |  24   |           |
|            |  25   |  26   |           |
|            |  27   |  28   |           |
|            |  29   |  30   | MOTOR-GND |
| MOTOR-STEP |  31   |  32   |           |
| MOTOR-DIR  |  33   |  34   |           |
|            |  35   |  36   |           |
|            |  37   |  38   |           |
|            |  39   |  40   |           |

ADC (Arduino Mega 250) x 2 Routing Table 

|  OUTPUT   |  PIN  |  PIN  |    OUTPUT    |
| :-------: | :---: | :---: | :----------: |
|           |       |  USB  | Raspberry-PI |
|  IF-GND   |  GND  |       |              |
| IF-SIGNAL |  A1   |       |              |

Speed Generator (Arduino UNO) Routing Table 

| OUTPUT |  PIN  |  PIN  |    OUTPUT    |
| :----: | :---: | :---: | :----------: |
|        |       |  USB  | Raspberry-PI |
|        |       |  GND  |   LED-NEG    |
|        |       |  13   |   LED-POS    |
|        |       |   9   |   ESC-POS    |
|        |       |       |   ESC-NEG    |

Our speed generator is controlled by Arduino also. We suggest to use `Arduino Bluetooth Control` (Google Play) to control it.

## Getting Start (2 page)

...

### Property Demonstration 1: Distance Measurement {#}

In first part, we will walk through the experiment of distance measurement. Assumed that the system is setup correctly, run the source code with Python3: 

```
python3 main_control.py
```

The system should print out the configuration and make sure all components connect correctly. It provides a command line interface (CLI). Type `read` to capture the signal, type `sig` to open the oscilloscope and type `ppi` to open the plan position indicator.

![Distance Measurement](./figure/distance-measurment.png)

There are 3 plots in the oscilloscope: 1. IF Signal Waveform; 2. IF Signal Spectrum and 3. IF Signal Spectrum with Window Averaging. We might see a periodic waveform because part of radar signal reflected by wall or coupled between transmitter and receiver. We named all this unwanted signal as **background signal**. Record it down might help radar detecting more accurate. Type the following command

```
>>> Command: setbg
```

The third plot should be suppressed to nearly 0 now. Then we place a reflector in front of the antenna, we might see the waveform changes.

Due to the area limitation of a small class room, we can add a (DGen) between the Rx antenna and the mixer. It's actually a delay line, lag the arrival time of received signal to act the effect of long distance.

Finally, if want to restart or quit program, just type `quit` to close the program. Remove the DGen and ready to go on the next experiment.

### Property Demonstration 2: Radar Cross Section (RCS) {#}

Now we prepare to demonstrate the RCS property. The material of the target affects how much of radar signal can be reflected. We place a wood plate as far as the aluminum plate from the radar system. Switching the radar direction use the following command. 

```
>>> Command: dir <dir>
```

A plate made of wood reflected less EM Wave than the plate made of Aluminum while they have the same area size. We might see a different signal strength reflected by aluminum or wood.

### Property Demonstration 3: Speed Measurement {#}

We have tried to measure the static object in the previous experiments, but a FMCW radar should be able to detect the distance and speed simutaneously. Now replace the object as speed generator, switch on the power and connected it using any cell phone with Bluetooth.

Open the Serial Monitor of the speed generator. Type `o` to initialize, type `s` and send, the Serial Monitor will ask an integer for input. Type the desired rotating speed (unit: rps) and sent it, the motor will speed up until attained the target speed, and the LED light will switch on.

The speed detection is implemented by triangular modulation. The illustration is show as below. Due to the doppler effect, it split the main frequency as $f_d \pm f_b$. We tried to retrived $f_d$ by catching $f_{high}$ and $f_{low}$

![Triangular Modulation Principle](./figure/triangular-modulation.png)

![Speed Measurement](./figure/speed-measurment.png)

### Property Demonstration 4: Multiple Object Resolution {#}

Finally we would like to demonstrate the resolution property. Troy is available to detect multiple objects, if the objects is spearated with enough distance.

![Multiple Resolution](./figure/multiple-resolution.png)

## Remarks
