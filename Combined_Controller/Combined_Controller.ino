#include <Servo.h>
unsigned long count=0;
int counterPin = 2;
int LEDPin = 13;
unsigned long time2;
float currentRPS;
float targetRPS;
float rpsUpLimit = 40;
float rpsLowLimit =0;
float rpsPrecision = 0.5;
int motorPulse = 30;     
float grid_num = 8;
Servo Brushless1;
String ans;
boolean initializeState = 0;
boolean jumpToEnd =0;
boolean constSpeed = 0;
void counter() {
   count = count+1;
}

void setup() {
   Serial.begin(9600);
   Brushless1.attach(9);
   Brushless1.write(0);
   pinMode(LEDPin, OUTPUT);
   pinMode(counterPin, INPUT);
   attachInterrupt(digitalPinToInterrupt(counterPin), counter, RISING);
   Serial.print("Arming the motor! ");
   Serial.println("(hearing regular beep---beep---beep--- )");
   count = 0;
   time2 = 0;
}
void initialize_motor() {
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
void testConstantRPS() {
  int rps = 0;
  Serial.println("Enter the targetRPS ...(5~40)");
  while (Serial.available() == 0)  {
  }
  rps = Serial.parseInt();
  if ((rps > 5) && (rps < 40)) {
    Serial.print("setting Motor to targetRPS = ");
    targetRPS = rps;
    Serial.println(targetRPS);
  } else
    Serial.print("Wrong RPS command, using previous targetRPS == ");
    Serial.println(targetRPS);
}


void stopmotor() {
  Serial.println("STOP!");
  Brushless1.write(5);
}
void sensorcontrol(){
     // 計算 rps 時，停止計時
     float rps;
     detachInterrupt(digitalPinToInterrupt(counterPin));
     Serial.print("millis = ");
     Serial.println((millis() - time2));   
     // 偵測的格數count * (1000 / 一圈網格數20）/ 時間差) 
     rps = (1.0*1000* count)/ (grid_num*(float)(millis() - time2));
     time2 = millis();
     count = 0;
     currentRPS = rps;
     Serial.print("RPS sensor is good, ");
     Serial.print("currentRPS = ");
     Serial.println(currentRPS);
     motorFeedback();
     //Restart the interrupt processing
     attachInterrupt(digitalPinToInterrupt(counterPin), counter, RISING);
     Serial.print("Start to initailize? [y/n] or testing ?[s] or stoping? [o]\n");
}
void motorFeedback(){
  if (currentRPS>targetRPS+rpsPrecision){
    if (currentRPS-targetRPS>5*rpsPrecision){
      Serial.print("Motor is far too fast, current Pulse = ");
      Serial.print(motorPulse);
      motorPulse = motorPulse-2;
      Brushless1.write(motorPulse);
    }
    else{
      Serial.print("Motor is slightly too fast, current Pulse = ");
      Serial.print(motorPulse);
      motorPulse = motorPulse-1;
      Brushless1.write(motorPulse);
    }
    digitalWrite(LEDPin, LOW);
    Serial.print(" new Pulse =");
    Serial.println(motorPulse);
  }
  else if(currentRPS<targetRPS-rpsPrecision){
    if (targetRPS-currentRPS>5*rpsPrecision){
      Serial.print("Motor is far too slow, current Pulse = ");
      Serial.print(motorPulse);
      motorPulse = motorPulse+2;
      Brushless1.write(motorPulse);
    }
    else{
      Serial.print("Motor is slightly too slow, current Pulse = ");
      Serial.print(motorPulse);
      motorPulse = motorPulse+1;
      Brushless1.write(motorPulse);
    }
    Serial.print(" new Pulse =");
    Serial.println(motorPulse);
    digitalWrite(LEDPin, LOW);
  }
  else{
    Serial.println("Motor is in control, feedback is not invoked");
    digitalWrite(LEDPin, HIGH);
  }
}

void loop() {

 if (millis() - time2 > 2000){
    if (constSpeed == 1){
      // include speed printing and feedback control
      sensorcontrol();
    }
  }

  if (Serial.available() == 0)  {
    jumpToEnd =1;
  }
  else{
    ans = Serial.readString(); 
  }
  if (jumpToEnd == 0){
    if (ans == "y") {
      initialize_motor();
      initializeState = 1;
      constSpeed =0;
    } 
    else if (ans == "n") {
      Serial.print("Okay, waiting...\n");
      constSpeed =0;
    } 
    else if ((ans == "o") || (ans == "s")) {
      if (initializeState == 0) {
        Serial.print("Sorry, you need to initailize motor...\n");
      } 
      else {
        if (ans == "s") {
          testConstantRPS();
          constSpeed =1;
        } 
        else if (ans == "o") {
          stopmotor();
          constSpeed =0;
        }
      }
    }
    else {
      Serial.print("Wrong ans!\n");
      constSpeed =0;
    }
   }
   jumpToEnd = 0;
   ans = "";
}
