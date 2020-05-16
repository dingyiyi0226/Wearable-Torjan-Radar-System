#include "Counter.h"

Counter::Counter(): _timer(0), _count(0) {}

unsigned long 
Counter::getCount() 
{ 
    return _count; 
}

void 
Counter::count() 
{ 
    ++_count; 
}

void 
Counter::start() 
{ 
    _timer = millis(); 
}

void 
Counter::reset() 
{ 
    _count = 0; 
    _timer = millis(); 
}
