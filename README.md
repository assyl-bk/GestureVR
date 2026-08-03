# GestureVR — Dual-IMU Joint Tracking for VR Hand Gesture Recognition

Tracks a forearm/wrist joint using two MPU6050 IMUs read by an ESP32,
streams orientation data for live Unity visualization, and builds a
labeled gesture dataset for training a 1D-CNN gesture classifier.

## Project status / roadmap

- [x] Single MPU6050 raw streaming
- [x] Accel/gyro calibration
- [x] DMP-based quaternion output (drift-corrected orientation)
- [x] Dual-sensor setup (two IMUs, two I2C addresses, one joint)
- [x] Per-sensor calibration + CSV data logging
- [ ] Unity live visualization (forearm/hand rig)
- [ ] Gesture dataset collection
- [ ] 1D-CNN training
- [ ] Real-time inference -> Unity gesture events

## Project structure

```
./
├── firmware/
│   ├── i2c_scanner/              Verifies both sensors are detected
│   │                             at the correct I2C addresses (0x68, 0x69)
│   │                             before running anything more complex.
│   │
│   ├── single_mpu6050_stream/    Simple single-sensor DMP quaternion
│   │                             streaming. Good reference / starting
│   │                             point, or for single-joint tests.
│   │
│   ├── get_mac_adress/           Utility sketch to read the ESP32 MAC
│   │                             address for device setup.
│   │
│   ├── hand_esp32_transmitter/   Wearable-side ESP-NOW transmitter.
│   │                             Sends the hand sensor stream.
│   │
│   ├── main_esp32_receiver/      Base ESP32 receiver sketch. This is
│   │                             the matching side of the wireless
│   │                             pipeline and the next focus area.
│   │
│   └── dual_mpu6050_calibrated/  Main firmware. Reads both sensors,
│                                 calibrates each at boot, streams
│                                 CSV: timestamp_ms,label,aw,ax,ay,az,bw,bx,by,bz
│                                 Accepts START:<label> / STOP over
│                                 Serial to tag data while recording.
│
├── ml/
│   ├── data/raw/                 Labeled gesture CSVs land here
│   ├── collect_data.py           Talks to the firmware, records
│   │                             labeled gesture sessions to CSV
│   ├── dataset.py                Loads CSVs, builds windowed
│   │                             (window_size, features) tensors
│   ├── model.py                  1D-CNN architecture
│   ├── train.py                  Trains the model on collected data
│   ├── infer_realtime.py         Live serial -> window -> predict ->
│   │                             gesture label (Step 8)
│   └── requirements.txt
│
├── unity/
│   └── Scripts/
│       └── DualIMUReceiver.cs    Reads the dual-sensor stream, applies
│                                 quaternions to a Forearm/Hand rig
│
├── docs/
│   └── wiring.md                 Full wiring diagram + troubleshooting
│                                 notes from actually building this
│
└── README.md
```

The repository is intentionally flat at the root now, so the paths above
match the folders you see in the workspace and on GitHub.

The firmware work is centered on the receiver/transmitter pair, with the
`hand_esp32_transmitter` and `main_esp32_receiver` sketches carrying the
wireless live-stream path used by the Unity side.

## Hardware

- 1x ESP32 Dev Module
- 2x MPU6050 (GY-521)
- Breadboard + jumper wires
- See `docs/wiring.md` for the full diagram and sensor placement
  (Sensor A = forearm, Sensor B = hand/wrist)

## Workflow

### 1. Verify hardware
Flash `firmware/i2c_scanner/i2c_scanner.ino`, confirm both `0x68` and
`0x69` appear in Serial Monitor.

### 2. Run the main firmware
Flash `firmware/dual_mpu6050_calibrated/dual_mpu6050_calibrated.ino`.
Hold both sensors still during the boot calibration window. Confirm
clean, stable CSV output in Serial Monitor, then move each sensor
independently to confirm both streams update correctly.

### 3. Visualize live in Unity (once built)
Close the Arduino Serial Monitor (only one program can hold the port).
Open the Unity project, attach `DualIMUReceiver.cs` to a manager
object, wire up the Forearm/Hand transforms, set the correct COM port,
press Play.

### 4. Collect gesture data
```
cd ml
pip install -r requirements.txt
python collect_data.py
```
Record each gesture class multiple times, plus an "idle" (no motion)
class. Files land in `ml/data/raw/`.

### 5. Train the model
```
python train.py
```
Produces `gesture_model.h5` and `label_classes.npy`.

### 6. Real-time inference
```
python infer_realtime.py
```
Streams live predictions to the console (and, eventually, onward to
Unity — see the TODO in that file).

## Notes on accuracy
Every physical MPU6050 chip has its own manufacturing bias in both the
accelerometer and gyroscope. `dual_mpu6050_calibrated.ino` averages
~500 samples per sensor at boot (while held still) and feeds the
resulting offsets into the DMP before enabling it, which measurably
improves stability over using hardcoded zero offsets.
