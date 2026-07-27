#include <Wire.h>
#include <esp_now.h>
#include <WiFi.h>
#include "MPU6050_6Axis_MotionApps20.h"
//this is for the main hand esp32+ first mpu6050 (sensor a)
MPU6050 mpuA(0x68);

uint8_t fifoBufferA[64];
Quaternion qA;
bool dmpReadyA = false;

String currentLabel = "idle";

typedef struct {
  float w;
  float x;
  float y;
  float z;
} QuaternionPacket;

// Latest data received wirelessly from Sensor B (hand ESP32)
QuaternionPacket latestB = {1.0, 0.0, 0.0, 0.0};
volatile bool newBDataAvailable = false;

void onDataReceived(const esp_now_recv_info *info, const uint8_t *incomingData, int len) {
  if (len == sizeof(QuaternionPacket)) {
    memcpy(&latestB, incomingData, sizeof(QuaternionPacket));
    newBDataAvailable = true;
  }
}

void calibrateSensor(MPU6050 &mpu, const char* label) {
  const int SAMPLES = 500;
  long ax_sum = 0, ay_sum = 0, az_sum = 0, gx_sum = 0, gy_sum = 0, gz_sum = 0;
  int16_t ax, ay, az, gx, gy, gz;

  Serial.print("Calibrating sensor ");
  Serial.print(label);
  Serial.println(" (keep still)...");

  for (int i = 0; i < SAMPLES; i++) {
    mpu.getMotion6(&ax, &ay, &az, &gx, &gy, &gz);
    ax_sum += ax; ay_sum += ay; az_sum += az;
    gx_sum += gx; gy_sum += gy; gz_sum += gz;
    delay(2);
  }

  mpu.setXAccelOffset(ax_sum / SAMPLES);
  mpu.setYAccelOffset(ay_sum / SAMPLES);
  mpu.setZAccelOffset((az_sum / SAMPLES) - 16384);
  mpu.setXGyroOffset(gx_sum / SAMPLES);
  mpu.setYGyroOffset(gy_sum / SAMPLES);
  mpu.setZGyroOffset(gz_sum / SAMPLES);
}

void handleSerialCommands() {
  if (Serial.available()) {
    String cmd = Serial.readStringUntil('\n');
    cmd.trim();
    if (cmd.startsWith("START:")) {
      currentLabel = cmd.substring(6);
      currentLabel.trim();
    } else if (cmd == "STOP") {
      currentLabel = "idle";
    }
  }
}

void setup() {
  Serial.begin(115200);
  Wire.begin(21, 22);
  Wire.setClock(400000);

  mpuA.initialize();
  if (!mpuA.testConnection()) {
    Serial.println("Sensor A connection failed!");
    while (1) delay(10);
  }

  calibrateSensor(mpuA, "A");

  uint8_t devStatus = mpuA.dmpInitialize();
  if (devStatus == 0) {
    mpuA.setDMPEnabled(true);
    dmpReadyA = true;
    Serial.println("Sensor A DMP ready.");
  } else {
    Serial.println("DMP init failed!");
    while (1) delay(10);
  }

  WiFi.mode(WIFI_STA);

  if (esp_now_init() != ESP_OK) {
    Serial.println("ESP-NOW init failed!");
    while (1) delay(10);
  }

  esp_now_register_recv_cb(onDataReceived);

  Serial.println("Ready. Waiting for wireless data from hand ESP32...");
  Serial.println("timestamp_ms,label,aw,ax,ay,az,bw,bx,by,bz");
}

void loop() {
  handleSerialCommands();

  if (!dmpReadyA) return;

  bool gotA = mpuA.dmpGetCurrentFIFOPacket(fifoBufferA);
  if (gotA) mpuA.dmpGetQuaternion(&qA, fifoBufferA);

  if (gotA) {
    Serial.print(millis()); Serial.print(",");
    Serial.print(currentLabel); Serial.print(",");
    Serial.print(qA.w, 4); Serial.print(",");
    Serial.print(qA.x, 4); Serial.print(",");
    Serial.print(qA.y, 4); Serial.print(",");
    Serial.print(qA.z, 4); Serial.print(",");
    Serial.print(latestB.w, 4); Serial.print(",");
    Serial.print(latestB.x, 4); Serial.print(",");
    Serial.print(latestB.y, 4); Serial.print(",");
    Serial.println(latestB.z, 4);
  }
}