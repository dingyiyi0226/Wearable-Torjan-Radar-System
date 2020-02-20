#include <stdint.h>
#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <getopt.h>
#include <fcntl.h>
#include <sys/ioctl.h>
#include <linux/types.h>
#include <linux/spi/spidev.h>

static void pabort(const char*);
static int transfer(int, struct spi_ioc_transfer*);
static void print_usage(const char* );
static void parse_opts(int , char* );
int sysfsMain(int, char**);
