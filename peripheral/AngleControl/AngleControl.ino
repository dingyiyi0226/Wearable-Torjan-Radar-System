#define dirPin 2
#define stepPin 3
#define enablePin 1

char cmd[3];
float currentAngle = 0;
float newAngle = 0;
float stepPerRound = 3200;
unsigned int delayMicro = 2000;
int stepDiff = 0;

void 
setup() 
{
    // Declare pins as output:
    Serial.begin(9600);
    pinMode(stepPin, OUTPUT);
    pinMode(dirPin, OUTPUT);
    pinMode(enablePin, OUTPUT);
    digitalWrite(enablePin, LOW);
    digitalWrite(stepPin, LOW);
    digitalWrite(dirPin, LOW);
    Serial.print("StepPerRound: ");
    Serial.println(stepPerRound);
}

void 
loop() 
{
    if (Serial.available())
    {
        Serial.readBytes(cmd, 3);
        newAngle = atof(cmd);

        Serial.print("Current Angle is");
        Serial.println(currentAngle);
        Serial.print("Enter new Angle: ");
        Serial.println(newAngle);

        stepDiff = stepPerRound * ((newAngle-currentAngle) / 360);
        Serial.print("stepdiff = ");
        Serial.println(stepDiff);

        goStep(stepDiff);
        stepDiff = 0;
        currentAngle = newAngle;
    }
}

void 
goStep(int stepDiff)
{
    if (stepDiff < 0) {
        digitalWrite(dirPin, LOW);
        Serial.println("Counter-clockwise");
        stepDiff = -stepDiff;
    }
    else if(stepDiff > 0){
        digitalWrite(dirPin, HIGH);
        Serial.println("Clockwise");
    }
    for (int i = 0; i < stepDiff; i++) {
        // These four lines result in 1 step:
        digitalWrite(stepPin, HIGH);
        delayMicroseconds(2000);
        digitalWrite(stepPin, LOW);
        delayMicroseconds(2000);
    }

}
