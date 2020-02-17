# SigGen Controller

## Written Languages

C++

## Device

Respberry Pi 3 Model B

## Setup

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

## Reference

1. [RPi Documentation](raspberrypi.org/documentation/hardware/raspberrypi/spi/README.md)
