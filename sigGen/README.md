# SigGen Controller

## Written Language

C++

## Device

Respberry Pi 3 Model B (OS: Respbian)

## Setup

### Open SPI

1. Make SPI Driver available. Reboot if need

```
>>> sudo /boot/config.txt

...
dtparam=spi=on <--- should not be commented out
...
```

2. Check device `/dev/spidev0.0` is available.

```
>>> ls -al /dev/

...
spidev0.0 <--- this option will appears
...
```

### Install Broadcom 2835 Libaray

```
```

## Reference

1. [RaspberryPi SPI Documentation](raspberrypi.org/documentation/hardware/raspberrypi/spi/README.md)
2. [RaspberryPi GPIO Pads Control](raspberrypi.org/documentation/hardward/raspberrypi/gpio/gpio_pads_control.md)
3. [RaspberryPi Broadcom 2835 Library](https://www.airspayce.com/mikem/bcm2835/)
4. [Analyzing SPI Driver performance on the Raspberry Pi](jumpnowtek.com/rpi/Analyzing-raspberry-pi-spi-performance.html)
