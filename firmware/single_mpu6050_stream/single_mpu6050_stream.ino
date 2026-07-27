/*
  dmp_stream.ino
  ---------------
  Streams stable orientation (quaternion) from the MPU6050's onboard
  Digital Motion Processor (DMP) over Serial, in a simple CSV line format:

      qw,qx,qy,qz

  This is meant to be read by Unity (see unity/Scripts/SerialIMUReceiver.cs)
  or any other serial-reading tool for live visualization.

  Wiring (ESP32):
    MPU6050 VCC -> 3.3V
    MPU6050 GND -> GND
    MPU6050 SDA -> GPIO21
    MPU6050 SCL -> GPIO22

  Board: ESP32 Dev Module
*/

#include <Wire.h>
#include "MPU6050_6Axis_MotionApps20.h"

MPU6050 mpu;

// DMP state
bool dmpReady = false;
uint8_t devStatus;
uint16_t packetSize;
uint8_t fifoBuffer[64];

// Orientation containers
Quaternion q;

void setup() {
  Serial.begin(115200);
  Wire.begin(21, 22);
  Wire.setClock(400000); // 400kHz I2C, standard for MPU6050

  Serial.println("Initializing MPU6050...");
  mpu.initialize();

  if (!mpu.testConnection()) {
    Serial.println("MPU6050 connection failed! Check wiring.");
    while (1) delay(10);
  }

  Serial.println("Initializing DMP...");
  devStatus = mpu.dmpInitialize();

  // IMPORTANT: these offsets are specific to YOUR chip.
  // Run the calibration sketch (see docs/) or use the offsets you found
  // earlier with your accel/gyro bias calibration and adapt them here
  // if you want tighter accuracy. Defaults below are a safe starting point.
  mpu.setXGyroOffset(0);
  mpu.setYGyroOffset(0);
  mpu.setZGyroOffset(0);
  mpu.setZAccelOffset(0);

  if (devStatus == 0) {
    mpu.setDMPEnabled(true);
    packetSize = mpu.dmpGetFIFOPacketSize();
    dmpReady = true;
    Serial.println("DMP ready. Streaming quaternion data...");
  } else {
    Serial.print("DMP Initialization failed (code ");
    Serial.print(devStatus);
    Serial.println(")");
    while (1) delay(10);
  }
}

void loop() {
  if (!dmpReady) return;

  // Blocks until a full FIFO packet is available, then reads it.
  if (mpu.dmpGetCurrentFIFOPacket(fifoBuffer)) {
    mpu.dmpGetQuaternion(&q, fifoBuffer);

    // CSV line: qw,qx,qy,qz
    Serial.print(q.w, 4); Serial.print(",");
    Serial.print(q.x, 4); Serial.print(",");
    Serial.print(q.y, 4); Serial.print(",");
    Serial.println(q.z, 4);
  }
}
