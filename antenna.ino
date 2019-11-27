#include "arduinoFFT.h"

// Define input pin
int IN[4] = {A4,A5,A6,A7};

// Define antenna phase shifter

// int ANTENNA[4][6] = {
// 	{30,32,34,36,38,40},    // Antenna #0  {5,11,22,45,90,180} degree
// 	{31,33,35,37,39,41},    // Antenna #1
// 	{42,44,46,48,50,52},    // Antenna #2
// 	{43,45,47,49,51,53}     // Antenna #3
// };

int ANTENNA[4][6] = {
	{40,38,36,34,32,30},    // Antenna #0  {5,11,22,45,90,180} degree
	{41,39,37,35,33,31},    // Antenna #1
	{52,50,48,46,44,42},    // Antenna #2
	{53,51,49,47,45,43}     // Antenna #3
};


void setphase(int ant, int pin){
	if(ant>=4) {return;}
	int tmp[6] = {1,2,4,8,16,32};

	for(int i=0;i<6;i++){
		digitalWrite(ANTENNA[ant][i], pin&tmp[i] ? HIGH : LOW);
	}
}

void setup() {

	Serial.begin(9600);
	Serial.setTimeout(100);  // for serial.readstring

	for(int i=0;i<4;i++){
		for(int j=0;j<6;j++){
			pinMode(ANTENNA[i][j],OUTPUT);
		}
	}
}


String s = "";
int pos=0;
int el1,el2,el3,el4;
int phasePin[6];
int tmp,a;

void loop() {

	// tmp=0;
	// for(int i=0;i<4;i++){
	// 	tmp+= analogRead(IN[i]);	}
	// tmp/=4;

 // 	Serial.print("d "+String(tmp)+'\n');

 	// Serial.print("d ");
 	// Serial.print(analogRead(IN[0]));
 	// String a = "ffff";
 	Serial.print("d "+ String(analogRead(IN[0]))+'\n');
 	// Serial.write("ff");

 	// Serial.print('\n')
 	// tmp+=1;
 	// if (tmp >400){
 	// 	Serial.println("");
 	// 	Serial.println("yea");tmp=0;
 	// }
 	// Serial.println(millis());
 	
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
	
}