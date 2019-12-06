#include "arduinoFFT.h"

// Define input pin
int IN[4] = {A4,A5,A6,A7};

void setup() {

	Serial.begin(9600);
	Serial.setTimeout(100);        // for serial.readstring
	analogReference(INTERNAL2V56); // options: DEFAULT, INTERNAL1V1, INTERNAL2V56, EXTERNAL
}

String s = "";
int pos=0;
int tmp,a;
unsigned long sampling_time, timetmp;

String anaRead = "";

void loop() {

	// tmp=0;
	// for(int i=0;i<4;i++){
	// 	tmp+= analogRead(IN[i]);
	// }
	// tmp/=4;

	timetmp = micros();

 	for(int i=0;i<500;i++){
 		// timetmp = millis();
 		anaRead += (String(analogRead(IN[0]))+ ' ');

 	}
 	sampling_time = micros() - timetmp;

 	Serial.print("d "+ anaRead+ '\n');
 	Serial.print("t "+ String(sampling_time)+ '\n');
 	anaRead = "";

 	// Serial.print("d "+ String(analogRead(IN[0]))+'\n');

 	/*
	if(Serial.available()){
		s = Serial.readString();
		if (s.startsWith("p")){
		
			s = s.substring(2);
			pos = s.indexOf(' ');
			el1 = s.substring(0,pos).toInt();

			s = s.substring(pos+1);
			pos = s.indexOf(' ');
			el2 = s.substring(0,pos).toInt();

			s = s.substring(pos+1);
			pos = s.indexOf(' ');
			el3 = s.substring(0,pos).toInt();

			el4 = s.substring(pos+1).toInt();

			setphase(0, el1);
			setphase(1, el2);
			setphase(2, el3);
			setphase(3, el4);

			// Serial.println("p "+ String(el1)+' '+ String(el2)+' '+ String(el3)+' '+ String(el4));
			Serial.print("p \n");

		} 
	}
	*/
	
}