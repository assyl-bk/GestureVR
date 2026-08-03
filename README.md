# GestureVR — Wireless Dual-IMU Wrist Joint Tracking & Gesture Recognition

Tracks the wrist joint using two MPU6050 IMUs (one on the forearm, one
on the hand), streams orientation wirelessly between two ESP32 boards,
and trains a 1D-CNN gesture classifier on the resulting joint-rotation
data.

## Project status / roadmap

- [x] Single MPU6050 raw streaming
- [x] Accel/gyro calibration
- [x] DMP-based quaternion output (drift-corrected orientation)
- [x] Dual-sensor setup, wired (two IMUs, two I2C addresses, one joint)
- [x] Per-sensor calibration + CSV data logging
- [x] Wireless dual-ESP32 setup (ESP-NOW) — hand sensor fully decoupled
      from the forearm/receiver board
- [x] Gesture dataset collection (7 classes)
- [x] 1D-CNN training with proper session-based validation split
- [x] Real-time inference (console predictions)
- [ ] Unity live visualization (forearm/hand rig)
- [ ] Real-time inference -> Unity gesture events
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
   Your PC (collect_data.py / infer_realtime.py / Unity)
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
│   ├── get_mac_address/             Run once on the Main ESP32 to get
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
├── ml/
│   ├── data/raw/                    Labeled gesture CSVs land here
│   ├── collect_data.py              Talks to the Main ESP32, records
│   │                                 labeled gesture sessions to CSV
│   ├── dataset.py                   Loads CSVs, builds windows —
│   │                                 grouped by (session, label) so
│   │                                 windows never cross a session
│   │                                 boundary
│   ├── model.py                     1D-CNN architecture
│   ├── train.py                     Trains the model with a
│   │                                 SESSION-based train/val split
│   │                                 (not a random window split — see
│   │                                 "Methodology notes" below)
│   ├── infer_realtime.py            Live serial -> window -> predict
│   │                                 -> gesture label (console output;
│   │                                 Unity hookup still TODO)
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

## Gesture set (current)

7 classes, chosen to be physically distinguishable via forearm/hand
relative rotation alone:

1. `idle` — no motion (baseline class)
2. `wrist_flex_up`
3. `wrist_flex_down`
4. `wrist_rotate`
5. `wave`
6. `arm_up`
7. `arm_down`

Recognizable with this setup: any wrist flexion/extension, radial/ulnar
deviation, forearm rotation, and compound dynamic gestures built from
those. **Not** recognizable: individual finger gestures (no
finger-mounted sensors) or absolute hand position in a room (IMUs give
orientation, not position — see `docs/wiring.md` / project notes for
why).

## Workflow

### 1. Verify hardware
Flash `firmware/i2c_scanner/i2c_scanner.ino` on a single ESP32 with
both sensors wired to it, confirm both `0x68` and `0x69` appear in
Serial Monitor.

### 2. Get the Main ESP32's MAC address
Flash `firmware/get_mac_address/get_mac_address.ino` to the ESP32 that
will stay wired to Sensor A and tethered to your PC. Copy the printed
MAC address.

### 3. Flash the wireless firmware
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

### 4. Collect gesture data
```
cd ml
pip install -r requirements.txt
python collect_data.py
```
For each gesture: type the label, Enter, perform the gesture
**continuously and repeatedly** (not a single static hold — the model
needs to learn the motion pattern, not a fixed end pose), Enter to
stop. Record **at least 2 separate sessions per gesture** (ideally
varying arm position/strap tightness between sessions) — this is
required for a trustworthy validation score, not just a nice-to-have
(see "Methodology notes" below). Files land in `ml/data/raw/`.

### 5. Sanity-check the dataset
```
python dataset.py
```
Reports how many windows were built per label and across how many
sessions. Look for every class having 2+ sessions and roughly balanced
window counts.

### 6. Train the model
```
python train.py
```
Produces `gesture_model.h5` and `label_classes.npy`. Watch for the "WARNING: '<label>' has only 1 recording session" messages — any class showing this needs another recording session before its validation score can be trusted.

### 7. Real-time inference
```
python infer_realtime.py
```
Streams live gesture predictions to the console.

### 8. Unity integration (not yet built)
Close the Arduino Serial Monitor (only one program can hold the port).
Open the Unity project, attach `DualIMUReceiver.cs` to a manager
object, wire up Forearm/Hand transforms, set the correct COM port,
press Play. Wiring `infer_realtime.py`'s predicted label onward into
Unity (not just raw orientation) is the next planned step.

## Methodology notes

**Session-based train/validation split.** `dataset.py` builds
overlapping windows via a sliding stride, so adjacent windows from the
same recording are highly similar. `train.py` therefore splits by
whole recording **session** (never by individual window) — an entire
session goes fully to train or fully to validation. A random
window-level split would let the model "see" near-duplicates of
validation data during training, producing an inflated, untrustworthy
accuracy score. A class with only one recorded session cannot be
validated yet; record a second session for any such class.

**Windows never cross a session boundary.** `dataset.py` groups rows
by `(session_id, label)` before slicing windows, so multiple separate
`idle` recordings (for example) are never concatenated and sliced
across their boundary as if they were one continuous recording.

**Calibration.** Every physical MPU6050 chip has its own manufacturing
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
  access the raw files.
