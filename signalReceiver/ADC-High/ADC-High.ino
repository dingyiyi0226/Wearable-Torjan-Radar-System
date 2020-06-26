#define DATA_NUM 2000
#define SEGMENTS 4
#define DATA_SPLIT (DATA_NUM / SEGMENTS)

// Define input pin
int IN = A1;

// Global Variables
unsigned long sampling_time, timetmp;
int signalDatas[DATA_NUM];
String tmpString = "", rs;

// unsigned long testtime1, testtime2;

void
setup()
{
    Serial.begin(115200);
    Serial.setTimeout(100);                 // for serial.readstring

    Serial.println("init");                 // I don't know why i need this line but i have to.
}


void
loop()
{
    if (Serial.available())
    {
        rs = Serial.readStringUntil(' ');
        
        if (rs == "r")
        {

            // --------- Sampling datas --------- //

            timetmp = micros();
            for(int i=0; i < DATA_NUM; i++)
                { signalDatas[i] = analogRead(IN); }
            sampling_time = micros() - timetmp;

            // ---------------------------------- //


            // ------- Transmitting datas ------- //

            Serial.print("i\n");                              // start transmitting

            for (int i = 0; i < SEGMENTS; ++i)
            {
                tmpString = "";
                
                for (int j = 0; j < DATA_SPLIT; j++)
                    { tmpString += String(signalDatas[i*DATA_SPLIT+j]) + ' '; }

                Serial.print("d ");                           // on Uno
                Serial.println(tmpString);
            }

            Serial.print("e ");
            Serial.println(sampling_time);                    // end transmitting
            // ---------------------------------- //

        }
        else if (rs == "n")
            { Serial.println("n5.8"); }
        else
            { Serial.print("Unknown Command: "); Serial.println(rs); }

    }
}
