/*
  i2c_scanner.ino
  ----------------
  Scans the I2C bus and prints every address that responds.
  For your dual MPU6050 setup, you should see two addresses:
    0x68 (Sensor A, default)
    0x69 (Sensor B, AD0 tied to 3.3V)

  If you only see one, or neither, that confirms a wiring issue before
  we even touch the DMP firmware.
*/

#include <Wire.h>

void setup() {
  Serial.begin(115200);
  Wire.begin(21, 22);
  delay(1000);
  Serial.println("I2C Scanner starting...");
}

void loop() {
  byte error, address;
  int devicesFound = 0;

  Serial.println("Scanning...");

  for (address = 1; address < 127; address++) {
    Wire.beginTransmission(address);
    error = Wire.endTransmission();

    if (error == 0) {
      Serial.print("Device found at address 0x");
      if (address < 16) Serial.print("0");
      Serial.println(address, HEX);
      devicesFound++;
    }
  }

  if (devicesFound == 0) {
    Serial.println("No I2C devices found. Check wiring.");
  } else {
    Serial.print(devicesFound);
    Serial.println(" device(s) found.");
  }

  Serial.println();
  delay(3000);
}
