"""
train.py
---------
Trains the 1D-CNN gesture classifier on data collected via
collect_data.py, using dataset.py for loading/windowing and
model.py for the architecture.

Run: python train.py
"""

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from dataset import load_all_csvs, make_windows, WINDOW_SIZE, FEATURE_COLUMNS
from model import build_1dcnn

MODEL_OUTPUT_PATH = "gesture_model.h5"


def main():
    print("Loading data...")
    df = load_all_csvs()
    X, y_raw = make_windows(df)

    print(f"Total windows: {len(X)}  |  Window shape: {X.shape[1:]}")

    # Encode string labels ("wave", "grab", "idle", ...) into integers
    encoder = LabelEncoder()
    y = encoder.fit_transform(y_raw)
    print(f"Classes: {list(encoder.classes_)}")

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = build_1dcnn(
        window_size=WINDOW_SIZE,
        num_features=len(FEATURE_COLUMNS),
        num_classes=len(encoder.classes_),
    )
    model.summary()

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=30,
        batch_size=32,
    )

    model.save(MODEL_OUTPUT_PATH)
    print(f"Model saved to {MODEL_OUTPUT_PATH}")

    # Save the label encoder classes so infer_realtime.py can map
    # predictions back to gesture names.
    np.save("label_classes.npy", encoder.classes_)


if __name__ == "__main__":
    main()
