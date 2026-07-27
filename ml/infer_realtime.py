"""
infer_realtime.py
-------------------
Reads the live CSV stream from dual_mpu6050_calibrated.ino, maintains
a rolling window of recent readings, and runs the trained model on
that window continuously to predict a gesture label in real time.

This is STEP 8 in the project plan -- only usable after train.py has
produced gesture_model.h5 and label_classes.npy.

Run: python infer_realtime.py
"""

import collections
import numpy as np
import serial
from tensorflow import keras

from dataset import WINDOW_SIZE, FEATURE_COLUMNS

SERIAL_PORT = "COM3"
BAUD_RATE = 115200
MODEL_PATH = "gesture_model.h5"
LABELS_PATH = "label_classes.npy"

CONFIDENCE_THRESHOLD = 0.7  # ignore predictions below this confidence


def main():
    model = keras.models.load_model(MODEL_PATH)
    labels = np.load(LABELS_PATH, allow_pickle=True)

    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)

    window = collections.deque(maxlen=WINDOW_SIZE)
    last_prediction = None

    print("Listening for gestures... Ctrl+C to quit.")

    try:
        while True:
            line = ser.readline().decode(errors="ignore").strip()
            if not line or "," not in line or line.startswith("timestamp_ms"):
                continue

            parts = line.split(",")
            if len(parts) != 10:  # timestamp,label,aw,ax,ay,az,bw,bx,by,bz
                continue

            # Skip timestamp (0) and label (1) columns; keep the 8 feature values
            try:
                features = [float(v) for v in parts[2:]]
            except ValueError:
                continue

            window.append(features)

            if len(window) == WINDOW_SIZE:
                x = np.array(window)[np.newaxis, :, :]  # shape (1, window, features)
                probs = model.predict(x, verbose=0)[0]
                best_idx = np.argmax(probs)
                confidence = probs[best_idx]

                if confidence >= CONFIDENCE_THRESHOLD:
                    predicted_label = labels[best_idx]
                    if predicted_label != last_prediction:
                        print(f"Gesture: {predicted_label}  (confidence: {confidence:.2f})")
                        last_prediction = predicted_label
                        # TODO: send this label onward to Unity here
                        # (e.g. over a second serial/socket connection,
                        # or a shared file/pipe Unity polls)

    except KeyboardInterrupt:
        print("\nStopping.")
    finally:
        ser.close()


if __name__ == "__main__":
    main()
