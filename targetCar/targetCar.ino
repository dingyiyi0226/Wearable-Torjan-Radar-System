

// set motor pin

int MotorDir[4][2] = {
    {1, 2},
    {3, 4},
    {5, 6},
    {7, 8}
};

int MotorSpeed[4] = {A1, A2, A3, A4};

// set motor speed and direction
// force should in range [0, 255]

void setMotor(int index, int speed=0){
    if(speed>0){
        digitalWrite(MotorDir[index][0], HIGH);
        digitalWrite(MotorDir[index][1], LOW);
        AnalogWrite(MotorSpeed[index], speed);
    }
    else{
        digitalWrite(MotorDir[index][0], LOW);
        digitalWrite(MotorDir[index][1], HIGH);
        AnalogWrite(MotorSpeed[index], -speed);
    }
}

void stopMotor(int index){
    digitalWrite(MotorDir[index][0], LOW);
    digitalWrite(MotorDir[index][1], LOW);
}





void setup() {
    for(int i=0;i<4;i++){
        for(int j=0;j<2;j++){
            pinMode(MotorDir[i][j], OUTPUT);
        }
    }

}

void loop() {


}
