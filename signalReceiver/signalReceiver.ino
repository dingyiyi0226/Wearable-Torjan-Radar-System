#define DATA_NUM 2000
#define SEGMENTS 4
#define DATA_SPLIT (DATA_NUM / SEGMENTS)

// Define input pin
int IN[4] = {A0,A1,A2,A3};

// Global Variables
unsigned long sampling_time, timetmp;
int signalDatas[DATA_NUM];
String tmpString = "", rs;

// unsigned long testtime1, testtime2;

void setup() {

    Serial.begin(115200);
    Serial.setTimeout(100);                 // for serial.readstring
    // analogReference(INTERNAL2V56);       // options: DEFAULT, INTERNAL1V1, INTERNAL2V56, EXTERNAL

    Serial.println("init"); // I don't know why i need this line but i have to.
}


void loop() {

    // Serial.println("ddd");
    
    if(Serial.available()){
        rs = Serial.readStringUntil(' ');
        // Serial.println(rs);

        if(rs=="r"){

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
                
                // testtime1 = micros();
                for (int j = 0; j < DATA_SPLIT; j++) {
                    tmpString += String(signalDatas[i*SEGMENTS+j])+' ';
                }
                // testtime2 = micros();
                Serial.print("d ");                           // on Uno
                Serial.println(tmpString);
                // Serial.println(micros()-testtime2);
                // Serial.println(testtime2-testtime1);

            }

            Serial.print("e ");
            Serial.println(sampling_time);                    // end transmitting
            // ---------------------------------- //

        }
        else{
            // Serial.print("nooo ");
            // Serial.println(rs);
        }

    }
}
