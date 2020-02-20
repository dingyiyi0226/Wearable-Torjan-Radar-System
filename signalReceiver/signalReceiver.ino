
// Define input pin
int IN[4] = {A4,A5,A6,A7};

void setup() {

    Serial.begin(9600);
    Serial.setTimeout(100);        // for serial.readstring
    analogReference(INTERNAL2V56); // options: DEFAULT, INTERNAL1V1, INTERNAL2V56, EXTERNAL
}

unsigned long sampling_time, timetmp;
const int DATA_NUM = 1000;
const int DATA_SPLIT = DATA_NUM/3;
int signalDatas[DATA_NUM] = {0};
String tmpString = "";

void loop() {

    // --------- Sampling datas --------- //

    timetmp = micros();
    for(int i=0;i<DATA_NUM;i++){
        signalDatas[i] = analogRead(IN[0]);
    }
    sampling_time = micros() - timetmp;

    // ---------------------------------- //


    // ------- Transmitting datas ------- //

    Serial.print("i\n");                              // start transmitting

    tmpString = "";
    for(int i=0;i<DATA_SPLIT;i++){
        tmpString += String(signalDatas[i])+' ';
    }
    Serial.print("d "+tmpString+'\n');

    tmpString = "";
    for(int i=DATA_SPLIT;i<2*DATA_SPLIT;i++){
        tmpString += String(signalDatas[i])+' ';
    }
    Serial.print("d "+tmpString+'\n');

    tmpString = "";
    for(int i=2*DATA_SPLIT;i<DATA_NUM;i++){
        tmpString += String(signalDatas[i])+' ';
    }
    Serial.print("d "+tmpString+'\n');

    Serial.print("e "+ String(sampling_time)+ '\n');  // end transmitting

    // ---------------------------------- //
}