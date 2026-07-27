/*
  dual_mpu6050_stream_calibrated.ino
  -----------------------------------
  Improvements over the basic dual-sensor version:
    1. Each sensor is calibrated at startup (accel + gyro bias offsets),
       and those offsets are fed into the DMP before enabling it. This
       significantly improves accuracy vs. hardcoded zero offsets.
    2. Output format switched to CSV (not JSON) with a timestamp and a
       label column, so this can be piped straight into a CSV file for
       ML dataset building (see ml/collect_data.py).
    3. Serial commands control labeling for data collection:
         START:<label>   -> tags every following line with that label
         STOP             -> returns to "idle" label

  Output line format:
    timestamp_ms,label,aw,ax,ay,az,bw,bx,by,bz

  Wiring: same dual-sensor setup you already have working
    (both on I2C bus, Sensor A = 0x68, Sensor B = 0x69 via AD0->3.3V)
*/

#include <Wire.h>
#include "MPU6050_6Axis_MotionApps20.h"

MPU6050 mpuA(0x68);
MPU6050 mpuB(0x69);

uint8_t fifoBufferA[64];
uint8_t fifoBufferB[64];
Quaternion qA, qB;

bool dmpReadyA = false;
bool dmpReadyB = false;

String currentLabel = "idle";

// ---- Calibration ----
// Averages raw accel/gyro samples while the sensor is held still, then
// applies the resulting bias as DMP offsets. Run once per sensor at boot.
void calibrateSensor(MPU6050 &mpu, const char* label) {
  const int SAMPLES = 500;
  long ax_sum = 0, ay_sum = 0, az_sum = 0;
  long gx_sum = 0, gy_sum = 0, gz_sum = 0;
  int16_t ax, ay, az, gx, gy, gz;

  Serial.print("Calibrating sensor ");
  Serial.print(label);
  Serial.println(" (keep this sensor still and flat)...");

  for (int i = 0; i < SAMPLES; i++) {
    mpu.getMotion6(&ax, &ay, &az, &gx, &gy, &gz);
    ax_sum += ax; ay_sum += ay; az_sum += az;
    gx_sum += gx; gy_sum += gy; gz_sum += gz;
    delay(2);
  }

  int16_t ax_off = ax_sum / SAMPLES;
  int16_t ay_off = ay_sum / SAMPLES;
  int16_t az_off = (az_sum / SAMPLES) - 16384; // expect ~1g on Z at rest
  int16_t gx_off = gx_sum / SAMPLES;
  int16_t gy_off = gy_sum / SAMPLES;
  int16_t gz_off = gz_sum / SAMPLES;

  mpu.setXAccelOffset(ax_off);
  mpu.setYAccelOffset(ay_off);
  mpu.setZAccelOffset(az_off);
  mpu.setXGyroOffset(gx_off);
  mpu.setYGyroOffset(gy_off);
  mpu.setZGyroOffset(gz_off);

  Serial.print("Sensor ");
  Serial.print(label);
  Serial.println(" calibration applied.");
}

bool initSensor(MPU6050 &mpu, const char* label) {
  mpu.initialize();
  if (!mpu.testConnection()) {
    Serial.print("Sensor ");
    Serial.print(label);
    Serial.println(" connection failed!");
    return false;
  }

  // Calibrate BEFORE dmpInitialize, so the DMP picks up correct offsets.
  calibrateSensor(mpu, label);

  uint8_t devStatus = mpu.dmpInitialize();
  if (devStatus == 0) {
    mpu.setDMPEnabled(true);
    Serial.print("Sensor ");
    Serial.print(label);
    Serial.println(" DMP ready.");
    return true;
  } else {
    Serial.print("Sensor ");
    Serial.print(label);
    Serial.print(" DMP init failed (code ");
    Serial.print(devStatus);
    Serial.println(")");
    return false;
  }
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

  Serial.println("Hold BOTH sensors still and flat during calibration.");
  delay(500);

  dmpReadyA = initSensor(mpuA, "A");
  dmpReadyB = initSensor(mpuB, "B");

  if (!dmpReadyA || !dmpReadyB) {
    Serial.println("One or both sensors failed to initialize. Check wiring/addresses.");
  } else {
    Serial.println("Both sensors ready. Streaming...");
    Serial.println("Send START:<label> or STOP to tag recorded data.");
    Serial.println("timestamp_ms,label,aw,ax,ay,az,bw,bx,by,bz");
  }
}

void loop() {
  handleSerialCommands();

  if (!dmpReadyA || !dmpReadyB) return;

  bool gotA = mpuA.dmpGetCurrentFIFOPacket(fifoBufferA);
  bool gotB = mpuB.dmpGetCurrentFIFOPacket(fifoBufferB);

  if (gotA) mpuA.dmpGetQuaternion(&qA, fifoBufferA);
  if (gotB) mpuB.dmpGetQuaternion(&qB, fifoBufferB);

  if (gotA || gotB) {
    Serial.print(millis()); Serial.print(",");
    Serial.print(currentLabel); Serial.print(",");
    Serial.print(qA.w, 4); Serial.print(",");
    Serial.print(qA.x, 4); Serial.print(",");
    Serial.print(qA.y, 4); Serial.print(",");
    Serial.print(qA.z, 4); Serial.print(",");
    Serial.print(qB.w, 4); Serial.print(",");
    Serial.print(qB.x, 4); Serial.print(",");
    Serial.print(qB.y, 4); Serial.print(",");
    Serial.println(qB.z, 4);
  }
}
