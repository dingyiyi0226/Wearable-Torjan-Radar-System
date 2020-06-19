#ifndef COUNTER_H
#define COUNTER_H
#include <Arduino.h>

class Counter
{
public:
    Counter();
    unsigned long getCount();
    void count(unsigned long /* count */);
    void count();
    void start();
    void reset();
 
private:
    unsigned long _timer;
    unsigned long _count;
};

#endif
