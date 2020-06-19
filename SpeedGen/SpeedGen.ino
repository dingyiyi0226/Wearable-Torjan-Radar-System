#include <Servo.h>
#include <AutoPID.h>
#include "Counter.h"

#define counterPin 2
#define LEDPin 12
#define motorPin 9

unsigned long count = 0;
unsigned long time2 = 0;
double currentRPS = 0;
double targetRPS = 0;
double measureTime = 1000;
double rpsUpLimit = 40;
double rpsLowLimit = 0;
double rpsPrecision = 0.5;
double grid_num = 8;
double Kp = 4;
double Ki = 0.375;
double Kd = 1;
double motorInput = 20;
String ans;
bool operatingState = false;

AutoPID pid(&currentRPS, &targetRPS, &motorInput, 20, 80, Kp, Ki, Kd);
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
    while (!Serial.available()) {}
    
    rps = Serial.parseInt();
    if ((rps >= 0) && (rps < 40)) 
        { Serial.print("setting Motor to targetRPS = "); targetRPS = rps; } 
    else
        { Serial.print("Wrong RPS command, using previous targetRPS == "); }
}

void 
stopMotor() 
{
    Serial.println("STOP!");
    
    operatingState = false;
    targetRPS = 0;
    pid.run();
    Brushless1.write(static_cast<int>(motorInput));
}

void 
sensorcontrol()
{
    float rps;

    // Pause the counting process.
    detachInterrupt(digitalPinToInterrupt(counterPin));
    
    Serial.print("millis = ");
    Serial.println((millis() - time2));
    rps = (1.0 * 1000 * count) / (grid_num * (float)(millis() - time2));
    time2 = millis();    
    count = 0;
    currentRPS = rps;
    Serial.print("currentRPS = ");
    Serial.print(currentRPS);
    Serial.print(", targetRPS = ");
    Serial.println(targetRPS);
        
    motorFeedback();
    
    // Restart the interrupt processing
    attachInterrupt(digitalPinToInterrupt(counterPin), counter, RISING);
    Serial.print("Testing or stoping? [s/o]\n");
}

void 
motorFeedback() 
{
    // PID Method
    Serial.print("currentPulse = ");
    Serial.print(static_cast<int>(motorInput));
    Serial.print(", newPulse = ");
    Serial.println(static_cast<int>(motorInput));

    pid.run();
    Brushless1.write(motorInput);

    // Light up if meet the setpoint.
    if (!pid.atSetPoint(rpsPrecision))
        { digitalWrite(LEDPin, LOW); }
    else
        { digitalWrite(LEDPin, HIGH); }
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

    // Wait until motor initialize command
    Serial.println("Arming the motor!");
    Serial.println("(hearing regular beep---beep---beep--- )");
    Serial.println("Initialize? Press [y] to initailize");

    ans = "";
    while (ans != "y")
        { if (Serial.available()) ans = Serial.readString(); }

    initialize_motor(); 
    pid.setTimeStep(measureTime);
    operatingState = false;
    targetRPS = 0; 
    time2 = millis();
    count = 0;

    Serial.print("Testing or stoping? [s/o]\n");
}

void 
loop() 
{
    // include speed printing and feedback control    
    if ((millis() - time2 > measureTime) && operatingState)
        { sensorcontrol(); }

    // If accept a command
    if (Serial.available()) 
    { 
        ans = Serial.readString(); 
    
        if ((ans == "o") || (ans == "s")) 
        {
            if (ans == "s") 
            { 
                // Check whether is speed init...
                if (!operatingState) { pid.reset(); operatingState = true; }
                
                // Accept New RPS.
                testConstantRPS(); 
            } 
            else if (ans == "o") 
            { 
                // Switch off everythings
                stopMotor(); 
            }
        }
        else 
        { 
            Serial.print("Wrong ans!\n"); 
        }

        Serial.print("Testing or stoping? [s/o]\n");
    }
}
