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

1. [RespberryPi SPI Documentation](raspberrypi.org/documentation/hardware/raspberrypi/spi/README.md)
2. [RespberryPi Broadcom 2835 Library](https://www.airspayce.com/mikem/bcm2835/)
