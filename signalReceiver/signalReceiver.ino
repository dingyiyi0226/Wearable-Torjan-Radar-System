#define DATA_NUM 2940
#define SEGMENTS 4
#define DATA_SPLIT (DATA_NUM / SEGMENTS)

// Define input pin
int IN[4] = {A0,A1,A2,A3};

void setup() {

    Serial.begin(9600);
    Serial.setTimeout(100);        // for serial.readstring
    // analogReference(INTERNAL2V56); // options: DEFAULT, INTERNAL1V1, INTERNAL2V56, EXTERNAL
}

unsigned long sampling_time, timetmp;
int signalDatas[DATA_NUM];
String tmpString = "";

void loop() {

    // --------- Sampling datas --------- //

    timetmp = micros();
    for(int i=0; i < DATA_NUM; i++){
        signalDatas[i] = analogRead(IN[0]);
    }
    sampling_time = micros() - timetmp;

    // ---------------------------------- //


    // ------- Transmitting datas ------- //

    Serial.print("i\n");                              // start transmitting

    for (int i = 0; i < SEGMENTS; ++i) {
        tmpString = "";
        
        for (int i = 0; i < DATA_SPLIT; i++) {
            tmpString += String(signalDatas[i])+' ';
        }

        // Serial.print("d "+tmpString+'\n');  // on Mega (fail on Uno WTF)
    
        Serial.print("d ");                    // on Uno
        Serial.println(tmpString);

    }
    
    Serial.println(sampling_time);  // end transmitting

    // ---------------------------------- //
}
