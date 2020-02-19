/*
 * SPI testing utility (using spidev driver)
 *
 * Copyright (c) 2007  MontaVista Software, Inc.
 * Copyright (c) 2007  Anton Vorontsov <avorontsov@ru.mvista.com>
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License.
 *
 * Cross-compile with cross-gcc -I/path/to/cross-kernel/include
 */

#include <stdint.h>
#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <getopt.h>
#include <fcntl.h>
#include <sys/ioctl.h>
#include <linux/types.h>
#include <linux/spi/spidev.h>

#include <bcm2835.h>
#include "gpio.h"
#include "sysfs.h"

#define POUT 23 /* P1-16 */

#define ARRAY_SIZE(a) (sizeof(a) / sizeof((a)[0]))

static const char *device = "/dev/spidev0.0";
static uint8_t mode;
static uint8_t bits = 8;
static uint32_t speed = 1e6;
static uint16_t delay = 0;

static uint8_t tx[5]  = { 0xF0, 0xF0, 0xF0, 0xF0, 0xF0 };
static uint8_t tx2[5] = { 0xCC, 0xCC, 0xCC, 0xCC, 0xCC };
static char tx3[5]    = { 0xF0, 0xF0, 0xF0, 0xF0, 0xF0 };
static char tx4[5]    = { 0xCC, 0xCC, 0xCC, 0xCC, 0xCC };
static uint8_t rx[ARRAY_SIZE(tx)]   = {0, };

static void
Dec2Hex(uint8_t *value, int dec)
{
    // TODO 
}

static void
getControlWord(int freq, int phase)
{
    // TODO
}

int
main(int argc, char **argv)
{
    // Setup SPI pins
    if (!bcm2835_init() || !bcm2835_spi_begin())
    {
        printf("BCM2835 SPI Begin failure, please try with sudo\n");
        return EXIT_SUCCESS;
    }

    // Set SPI MSB First
    bcm2835_spi_setBitOrder(BCM2835_SPI_BIT_ORDER_MSBFIRST);

    // Set CHIP (CS0 (19, 21, 23) or CS1)
    bcm2835_spi_chipSelect(BCM2835_SPI_CS0);

    // Set CS pins polarity to low
    bcm2835_spi_setChipSelectPolarity(BCM2835_SPI_CS0, 0);
    bcm2835_spi_setChipSelectPolarity(BCM2835_SPI_CS1, 0);
    
    // Set SPI clock speed 
    // Notes: In RPi2, CDIV4 and CDIV2 mode are unstable.
    bcm2835_spi_setClockDivider(BCM2835_SPI_CLOCK_DIVIDER_8);

    // Set SPI data mode
    bcm2835_spi_setDataMode(BCM2835_SPI_MODE0);

    while (1)
    {
        bcm2835_spi_writenb(tx3, ARRAY_SIZE(tx3));
        bcm2835_spi_writenb(tx4, ARRAY_SIZE(tx4));
    }

    bcm2835_spi_end();
    bcm2835_close();

    return EXIT_SUCCESS;
}
