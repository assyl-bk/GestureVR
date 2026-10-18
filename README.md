# GestureVR — Meta Quest Hand-Tracking Gesture Recognition

The current project focus and final ML deliverable is the `ml_quest/`
pipeline. It trains and evaluates gesture classifiers from Meta Quest
hand-tracking recordings. The ESP32 dual-IMU and Unity scripts remain
supporting/experimental work from earlier project stages.

## Project status / roadmap

- [x] Single MPU6050 raw streaming
- [x] Accel/gyro calibration
- [x] DMP-based quaternion output (drift-corrected orientation)
- [x] Dual-sensor setup, wired (two IMUs, two I2C addresses, one joint)
- [x] Per-sensor calibration + CSV data logging
- [x] Wireless dual-ESP32 setup (ESP-NOW) — hand sensor fully decoupled
      from the forearm/receiver board
- [x] Quest hand-tracking dataset collection
- [x] Quest 1D-CNN training with session-based train/validation/test splits
- [x] Multi-seed evaluation and saved aggregate results
- [x] Classical frame-level baselines (Random Forest and SVM)
- [ ] Unity live visualization (forearm/hand rig)
- [ ] Real-time Quest inference -> Unity gesture events
- [ ] Hand-side board moved to battery power (currently USB-powered
      for development)

## Hardware architecture

```
[Hand ESP32] --- I2C ---> [Sensor B, wrist/hand]
      |
      |  ESP-NOW (wireless, ~1-2ms latency)
      v
[Main ESP32] --- I2C ---> [Sensor A, forearm]
      |
      |  USB Serial
      v
   Your PC (supporting firmware tools / Unity)
```

- **Sensor A** (`0x68`) — forearm — reference segment
- **Sensor B** (`0x68` on its own bus, alone — no address conflict) —
  hand/wrist — tracked segment
- Only the Main ESP32 needs a MAC address hardcoded anywhere (the hand
  ESP32 needs to know where to *send*; the receiver just listens for
  whoever shows up)
- See `docs/wiring.md` for full pin-by-pin wiring and the "gotchas"
  encountered while building this (breadboard rail gaps, floating AD0,
  loose GND causing flicker, etc.)

## Project structure

```
GestureVR/
├── firmware/
│   ├── i2c_scanner/                 Verifies both sensors are detected
│   │                                 at 0x68 / 0x69 before anything else
│   │
│   ├── single_mpu6050_stream/       Simple single-sensor DMP reference
│   │
│   ├── dual_mpu6050_calibrated/     WIRED dual-sensor version (kept as
│   │                                 a fallback / debugging reference)
│   │
│   ├── get_mac_adress/              Run once on the Main ESP32 to get
│   │                                 the MAC address needed by the
│   │                                 transmitter firmware
│   │
│   ├── hand_esp32_transmitter/      Runs on the SECOND ESP32 (Sensor B).
│   │                                 Reads Sensor B's DMP quaternion,
│   │                                 sends it wirelessly via ESP-NOW.
│   │
│   └── main_esp32_receiver/         Runs on the MAIN ESP32 (Sensor A).
│                                     Reads Sensor A locally, receives
│                                     Sensor B wirelessly, streams the
│                                     combined CSV over USB:
│                                     timestamp_ms,label,aw,ax,ay,az,bw,bx,by,bz
│
├── ml_quest/
│   ├── data/raw/                    Quest sessions:
│   │                                 <label>_<session>/hand_path.csv
│   ├── quest_dataset.py             Pivots Quest's long CSV format,
│   │                                 encodes cyclic angles, and builds
│   │                                 fixed-length windows
│   ├── model.py                     Quest 1D-CNN architecture
│   ├── quest_train.py               Multi-seed CNN training and held-out
│   │                                 test-set evaluation
│   ├── quest_baseline.py            Random Forest and SVM frame-level
│   │                                 baselines with grid search
│   ├── splits.py                    Session-based splitting utilities
│   ├── quest_gesture_model.keras    Trained Quest CNN (Keras format)
│   ├── quest_gesture_model.h5       Trained Quest CNN (HDF5 format)
│   ├── quest_label_classes.npy      Class-label mapping
│   ├── multi_seed_results.npy       Saved multi-seed metrics
│   └── requirements.txt
│
├── unity/
│   └── Scripts/
│       └── DualIMUReceiver.cs       Reads the dual-sensor CSV stream,
│                                     applies quaternions to a
│                                     Forearm/Hand rig (not yet wired
│                                     into a built Unity scene)
│
├── docs/
│   ├── wiring.md                    Full wiring diagram + troubleshooting
│   └── data_dictionary.md           Column-by-column reference for the
│                                     recorded CSV files
│
└── README.md
```

## Quest dataset and features

Each recording is stored as
`ml_quest/data/raw/<gesture_label>_<session_number>/hand_path.csv`.
The CSV contains one row per tracked object and timestep:

```text
t,object,x,y,z,pitch,yaw,roll
```

The loader combines `LeftHandAnchor`, `RightHandAnchor`, and `Head` into
one row per timestep. Each object contributes position (`x`, `y`, `z`)
and sine/cosine encodings of the three Euler angles, for 27 features per
timestep. Lost-tracking frames are detected and windows with excessive
dropout are discarded. The available labels are determined from the
session folder names in `ml_quest/data/raw/`.

## Workflow

### 1. Prepare the Quest dataset

Place each Quest recording at:

```text
ml_quest/data/raw/<gesture_label>_<session_number>/hand_path.csv
```

Use at least three separate sessions per class so the train,
validation, and test splits can all contain held-out recordings. Keep
the label and session number in the folder name, for example
`wave_1/hand_path.csv`.

### 2. Install ML dependencies

```powershell
cd ml_quest
pip install -r requirements.txt
```

### 3. Inspect the dataset

```powershell
python quest_dataset.py
```

This reports the loaded sessions, labels, and generated windows.

### 4. Train and evaluate the Quest CNN

```powershell
python quest_train.py
```

`quest_train.py` runs the configured multi-seed experiment, reports
accuracy and weighted/macro F1 on the held-out test set, and writes:

- `quest_gesture_model.h5`
- `quest_label_classes.npy`
- `multi_seed_results.npy`

The existing `quest_gesture_model.keras` is an additional Keras-format
model artifact.

### 5. Run classical baselines

From `ml_quest/`, run either:

```powershell
python quest_baseline.py
python quest_baseline.py --rotation-only
python quest_baseline.py --no-velocity
```

The baseline compares Random Forest and SVM classifiers using
session-level splits. It reports accuracy, balanced accuracy, weighted
F1, classification reports, and confusion matrices.

### 6. Optional ESP32/Unity work

The following hardware workflow is retained for the supporting
dual-IMU prototype; it is not required to train or evaluate the
`ml_quest` deliverable.

#### Verify hardware
Flash `firmware/i2c_scanner/i2c_scanner.ino` on a single ESP32 with
both sensors wired to it, confirm both `0x68` and `0x69` appear in
Serial Monitor.

#### Get the Main ESP32's MAC address
Flash `firmware/get_mac_adress/get_mac_adress.ino` to the ESP32 that
will stay wired to Sensor A and tethered to your PC. Copy the printed
MAC address.

#### Flash the wireless firmware
- Hardcode the Main ESP32's MAC address into
  `firmware/hand_esp32_transmitter/hand_esp32_transmitter.ino`
  (`receiverMac[]`), then flash it to the SECOND ESP32 (wired to
  Sensor B only).
- Flash `firmware/main_esp32_receiver/main_esp32_receiver.ino` to the
  Main ESP32 (wired to Sensor A).
- Power both boards (Main ESP32 via USB to PC; hand ESP32 via any USB
  power source for now — battery swap is a later step).
- Confirm the Main ESP32's Serial Monitor shows live CSV rows, and that
  moving each sensor independently only changes that sensor's 4
  columns.

#### Unity integration (not yet built)
Close the Arduino Serial Monitor (only one program can hold the port).
Open the Unity project, attach `DualIMUReceiver.cs` to a manager
object, wire up Forearm/Hand transforms, set the correct COM port,
press Play. Connecting the Quest model's predicted label to Unity is
the next planned integration step.

## Methodology notes

**Session-based train/validation/test split.** `quest_dataset.py` builds
overlapping windows via a sliding stride, so adjacent windows from the
same recording are highly similar. `quest_train.py` and
`quest_baseline.py` therefore split by whole recording **session**
(never by individual window or frame). An entire session goes fully to
train, validation, or test. A random window-level split would let a
model see near-duplicates of evaluation data during training, producing
an inflated, untrustworthy score. Use at least three sessions per class
for a meaningful three-way split.

**Windows never cross a session boundary.** `quest_dataset.py` groups
rows by `session_id` before slicing windows, so separate recordings of
the same gesture are never concatenated and sliced across their
boundary as if they were one continuous recording.

**IMU calibration (supporting prototype).** Every physical MPU6050 chip has its own manufacturing
bias in both the accelerometer and gyroscope. The firmware averages
~500 samples per sensor at boot (while held still) and feeds the
resulting offsets into the DMP before enabling it, which measurably
improves stability over hardcoded zero offsets.

## Related work / references

- Abbas, K.A., Rashid, M.T., Fortuna, L. (2025). *The mechanomyographic
  dataset of hand gestures harvested using an accelerometer and
  gyroscope.* Data in Brief. DOI: 10.1016/j.dib.2025.111558 — single
  wrist-worn IMU, 11 gestures including wrist flexion/extension;
  notably uses the same ESP32-to-ESP32 wireless relay architecture
  (transmitter + USB-linked receiver) as this project, though with a
  single sensor rather than a dual-sensor joint-tracking setup.
- Wang, Y. et al. (2022). *Multi-position wearable human activity
  recognition dataset (MPJA-HAD).* IEEE DataPort. DOI:
  10.21227/z8bc-pt92 — computes joint angles from adjacent sensor
  pairs (e.g. hand + lower-arm for the wrist joint), the same
  dual-sensor relative-rotation concept used in this project, across a
  full-body 15-sensor setup. Requires an IEEE DataPort subscription to
  access the raw files.()
