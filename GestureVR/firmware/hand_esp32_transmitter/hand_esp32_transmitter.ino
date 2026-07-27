#include <Wire.h>
#include <esp_now.h>
#include <WiFi.h>
#include "MPU6050_6Axis_MotionApps20.h"
//this is for the second esp32 + second mpu6050 (sensor b)
// Main ESP32's MAC address (from get_mac_address.ino)
uint8_t receiverMac[] = {0x3C, 0x8A, 0x1F, 0x09, 0xAA, 0x08};

MPU6050 mpuB(0x68); // default address, alone on this bus

uint8_t fifoBuffer[64];
Quaternion q;
bool dmpReady = false;

typedef struct {
  float w;
  float x;
  float y;
  float z;
} QuaternionPacket;

QuaternionPacket packet;

void onDataSent(const wifi_tx_info_t *tx_info, esp_now_send_status_t status) {
  // Optional: could log send success/failure here if debugging.
}

void setup() {
  Serial.begin(115200);
  Wire.begin(21, 22);
  Wire.setClock(400000);

  mpuB.initialize();
  if (!mpuB.testConnection()) {
    Serial.println("Sensor B connection failed!");
    while (1) delay(10);
  }

  const int SAMPLES = 500;
  long ax_sum = 0, ay_sum = 0, az_sum = 0, gx_sum = 0, gy_sum = 0, gz_sum = 0;
  int16_t ax, ay, az, gx, gy, gz;
  Serial.println("Calibrating Sensor B (keep still)...");
  for (int i = 0; i < SAMPLES; i++) {
    mpuB.getMotion6(&ax, &ay, &az, &gx, &gy, &gz);
    ax_sum += ax; ay_sum += ay; az_sum += az;
    gx_sum += gx; gy_sum += gy; gz_sum += gz;
    delay(2);
  }
  mpuB.setXAccelOffset(ax_sum / SAMPLES);
  mpuB.setYAccelOffset(ay_sum / SAMPLES);
  mpuB.setZAccelOffset((az_sum / SAMPLES) - 16384);
  mpuB.setXGyroOffset(gx_sum / SAMPLES);
  mpuB.setYGyroOffset(gy_sum / SAMPLES);
  mpuB.setZGyroOffset(gz_sum / SAMPLES);

  uint8_t devStatus = mpuB.dmpInitialize();
  if (devStatus == 0) {
    mpuB.setDMPEnabled(true);
    dmpReady = true;
    Serial.println("Sensor B DMP ready.");
  } else {
    Serial.println("DMP init failed!");
    while (1) delay(10);
  }

  WiFi.mode(WIFI_STA);

  if (esp_now_init() != ESP_OK) {
    Serial.println("ESP-NOW init failed!");
    while (1) delay(10);
  }

  esp_now_register_send_cb(onDataSent);

  esp_now_peer_info_t peerInfo = {};
  memcpy(peerInfo.peer_addr, receiverMac, 6);
  peerInfo.channel = 0;
  peerInfo.encrypt = false;

  if (esp_now_add_peer(&peerInfo) != ESP_OK) {
    Serial.println("Failed to add ESP-NOW peer!");
    while (1) delay(10);
  }

  Serial.println("ESP-NOW ready. Transmitting Sensor B quaternion...");
}

void loop() {
  if (!dmpReady) return;

  if (mpuB.dmpGetCurrentFIFOPacket(fifoBuffer)) {
    mpuB.dmpGetQuaternion(&q, fifoBuffer);

    packet.w = q.w;
    packet.x = q.x;
    packet.y = q.y;
    packet.z = q.z;

    esp_now_send(receiverMac, (uint8_t *)&packet, sizeof(packet));
  }
}