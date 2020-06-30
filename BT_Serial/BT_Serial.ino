#include <BluetoothSerial.h>
BluetoothSerial BT;

void setup() {
  Serial.begin(9600);
  BT.begin("SGen");//請改名
}

void loop() {
  //檢查序列內是否有資料
  while (Serial.available()) {
    //讀取序列資料
    String Sdata = Serial.readString();
    //傳輸給藍芽
    BT.print(Sdata);
  }
  //檢查藍芽內是否有資料
  while (BT.available()) {
    //讀取藍芽資料
    String BTdata = BT.readString();
    //顯示在序列視窗
    Serial.print(BTdata);
  }
  delay(1);
}
