#include <Servo.h>
#include <AutoPID.h>

#define counterPin 2
#define LEDPin 12
#define motorPin 9

class PID
{
public:
    PID(double *input, double *setpoint, double *output, double Kp, double Ki, double Kd): 
        _input(input), _setpoint(setpoint), _output(output), _Kp(Kp), _Ki(Ki), _Kd(Kd), 
        _cumError(0), _diffError(0) {}
    bool atSetPoint(double precision) {}
    void setGain(double Kp, double Ki, double Kd) 
    {
        _Kp = Kp; _Ki = Ki; _Kd = Kd;
    }
    void reset() {}
    void run() {}
    void setBangBang() {}
    
private:
    double *_setpoint;
    double *_input;
    double *_output;
    double _cumError;
    double _diffError;
    double _Kp;
    double _Ki;
    double _Kd;
};

class Counter
{
public:
    Counter(): _timer(0), _count(0) {}
    unsigned long getCount() { return _count; }
    void count() { ++_count; }
    void start() { _timer = millis(); }
    void reset() { _count = 0; _timer = millis(); }
 
private:
    unsigned long _timer;
    unsigned long _count;
};

unsigned long count = 0;
unsigned long time2;
double currentRPS;
double targetRPS;
double measureTime = 1000;
double rpsUpLimit = 40;
double rpsLowLimit = 0;
double rpsPrecision = 0.5;
double grid_num = 8;
double Kp = 4;
double Ki = 0.375;
double Kd = 1;
int motorPulse = 30;
double motorInput = 30;
String ans;
boolean initializeState = 0;
boolean constSpeed = 0;

AutoPID pid(&currentRPS, &targetRPS, &motorInput, 20, 360, Kp, Ki, Kd);
Servo Brushless1;

void 
counter() 
{
    ++count;
}

void 
initialize_motor() 
{
    Serial.print("Okay, Starting to initailize...\n");
    Serial.print("Setting high speed! and wait 2 sec! ");
    Serial.println("(hearing beep-beep)");
    Brushless1.write(360);

    delay(2000);
    Serial.print("Setting back to low speed! and wait 4 sec! ");
    Serial.println("(hearing beep-beep-beep)");
    Brushless1.write(20);
    delay(4000);
    Serial.print("MOTOR IS READY! ");
    Serial.println("(hearing regular beep---beep---beep--- )");
}

void 
testConstantRPS() 
{
    int rps = 0;

    // Wait until user input targetRPS value
    Serial.println("Enter the targetRPS ...(5~40)");
    while (Serial.available() == 0) {}
    
    rps = Serial.parseInt();
    if ((rps > 5) && (rps < 40)) 
        { Serial.print("setting Motor to targetRPS = "); targetRPS = rps; pid.reset(); } 
    else
        { Serial.print("Wrong RPS command, using previous targetRPS == "); }
}


void 
stopmotor() 
{
    Serial.println("STOP!");
    Brushless1.write(5);
}

void 
sensorcontrol()
{
    // 計算 rps 時，停止計時
    float rps;

    detachInterrupt(digitalPinToInterrupt(counterPin));
    
    Serial.print("millis = ");
    Serial.println((millis() - time2));
    rps = (1.0 * 1000 * count) / (grid_num * (float)(millis() - time2));
    time2 = millis();    
    count = 0;
    currentRPS = rps;
    Serial.print("RPS sensor is good, ");
    Serial.print("currentRPS = ");
    Serial.print(currentRPS);
    Serial.print(", targetRPS = ");
    Serial.println(targetRPS);
        
    motorFeedback();
    
    // Restart the interrupt processing
    attachInterrupt(digitalPinToInterrupt(counterPin), counter, RISING);
    Serial.print("Start to initailize? [y/n] or testing ?[s] or stoping? [o]\n");
}

void 
motorFeedback() 
{
    // PID Method
    Serial.print("currentPulse = ");
    Serial.print(motorPulse);
    motorPulse = static_cast<int>(motorInput);
    Serial.print(", newPulse = ");
    Serial.println(motorPulse);

    pid.run();
    Brushless1.write(motorPulse);

    // Light up if meet the setpoint.
    if (!pid.atSetPoint(rpsPrecision))
        { digitalWrite(LEDPin, LOW); }
    else
        { digitalWrite(LEDPin, HIGH); }

    // If-else Method
    /*if (currentRPS > targetRPS + rpsPrecision)
    {
        if (currentRPS-targetRPS>5*rpsPrecision)
        {
            Serial.print("Motor is far too fast, current Pulse = ");
            Serial.print(motorPulse);
            motorPulse = motorPulse-2;
            Brushless1.write(motorPulse);
        }
        else
        {
            Serial.print("Motor is slightly too fast, current Pulse = ");
            Serial.print(motorPulse);
            motorPulse = motorPulse - 1;
            Brushless1.write(motorPulse);
        }
        digitalWrite(LEDPin, LOW);
        Serial.print(", new Pulse =");
        Serial.println(motorPulse);
    }
    else if(currentRPS < targetRPS-rpsPrecision)
    {
        if (targetRPS-currentRPS>5*rpsPrecision)
        {
            Serial.print("Motor is far too slow, current Pulse = ");
            Serial.print(motorPulse);
            motorPulse = motorPulse+2;
            Brushless1.write(motorPulse);
        }
        else
        {
            Serial.print("Motor is slightly too slow, current Pulse = ");
            Serial.print(motorPulse);
            motorPulse = motorPulse+1;
            Brushless1.write(motorPulse);
        }
        Serial.print(" new Pulse =");
        Serial.println(motorPulse);
        digitalWrite(LEDPin, LOW);
    }
    else
    {
        Serial.println("Motor is in control, feedback is not invoked");
        digitalWrite(LEDPin, HIGH);
    }*/
}

void 
setup() 
{
    Serial.begin(9600);
    Brushless1.attach(motorPin);
    Brushless1.write(0);
    pinMode(LEDPin, OUTPUT);
    pinMode(counterPin, INPUT);
    
    attachInterrupt(digitalPinToInterrupt(counterPin), counter, RISING);
    Serial.print("Arming the motor! ");
    Serial.println("(hearing regular beep---beep---beep--- )");
    count = 0;
    time2 = 0;

    pid.setTimeStep(measureTime);
}

void 
loop() 
{
    // include speed printing and feedback control    
    if (millis() - time2 > measureTime)
    {
        if (constSpeed == 1)
            { sensorcontrol(); }
    }

    if (Serial.available()) 
    { 
        ans = Serial.readString(); 
    
        if (ans == "y") 
        {
            initialize_motor();
            initializeState = 1;
            constSpeed = 0;
        } 
        else if (ans == "n") 
            { Serial.print("Okay, waiting...\n"); constSpeed = 0; } 
        else if ((ans == "o") || (ans == "s")) 
        {
            if (initializeState == 0) 
                { Serial.print("Sorry, you need to initailize motor...\n"); } 
            else 
            {
                if (ans == "s") 
                    { testConstantRPS(); constSpeed = 1; pid.reset(); } 
                else if (ans == "o") 
                    { stopmotor(); constSpeed = 0; }
            }
        }
        else 
            { Serial.print("Wrong ans!\n"); constSpeed = 0; }
    }

    ans = "";
}
