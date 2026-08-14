"""
train.py
---------
Trains the 1D-CNN gesture classifier (both sensors, 8 features) on
data collected via collect_data.py, using dataset.py for loading/
windowing, splits.py for the session-based split, and model.py for
the architecture.

Run: python train.py
"""

import numpy as np
from sklearn.preprocessing import LabelEncoder

from dataset import load_all_csvs, make_windows, WINDOW_SIZE, FEATURE_COLUMNS
from splits import session_based_split
from model import build_1dcnn

MODEL_OUTPUT_PATH = "gesture_model.h5"


def main():
    print("Loading data...")
    df = load_all_csvs()
    X, y_raw, groups = make_windows(df)
    X = np.asarray(X)
    y_raw = np.asarray(y_raw)
    groups = np.asarray(groups)

    print(f"Total windows: {len(X)}  |  Window shape: {X.shape[1:]}")

    encoder = LabelEncoder()
    y = np.asarray(encoder.fit_transform(y_raw), dtype=int)
    print(f"Classes: {list(encoder.classes_)}")

    print("\nSplitting by recording session (not by window):")
    train_idx, val_idx = session_based_split(y, groups)

    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]

    print(f"\nTrain windows: {len(X_train)}  |  Val windows: {len(X_val)}")

    if len(X_val) == 0:
        print("\nNo validation windows at all -- every class currently has "
              "only 1 session. Record at least one more session per "
              "gesture, then re-run this script for a trustworthy result.")
        return

    model = build_1dcnn(
        window_size=WINDOW_SIZE,
        num_features=len(FEATURE_COLUMNS),
        num_classes=len(encoder.classes_),
    )
    model.summary()

    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=30,
        batch_size=32,
    )

    model.save(MODEL_OUTPUT_PATH)
    print(f"Model saved to {MODEL_OUTPUT_PATH}")

    np.save("label_classes.npy", encoder.classes_)


if __name__ == "__main__":
    main()