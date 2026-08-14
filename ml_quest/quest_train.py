"""
quest_train.py
---------------
Trains a 1D-CNN on the Quest hand-tracking dataset (see
quest_dataset.py), using a session-based train/val split.

Run: python quest_train.py
"""

import numpy as np
from sklearn.preprocessing import LabelEncoder

from quest_dataset import load_all_sessions, make_windows, WINDOW_SIZE, get_feature_columns
from splits import session_based_split
from model import build_1dcnn

MODEL_OUTPUT_PATH = "quest_gesture_model.h5"
#added now
import numpy as np
import numpy.typing as npt

def as_idx(arr) -> npt.NDArray[np.intp]:
    return np.asarray(arr, dtype=np.intp)


def main():
    print("Loading Quest sessions...")
    df = load_all_sessions()
    X, y_raw, groups = make_windows(df)

    print(f"Total windows: {len(X)}  |  Window shape: {X.shape[1:]}")

    encoder = LabelEncoder()
    y = encoder.fit_transform(y_raw)
    print(f"Classes: {list(encoder.classes_)}")

    print("\nSplitting by recording session (not by window):")  
    train_idx, val_idx = session_based_split(y, groups)
    train_idx = np.asarray(train_idx).astype(np.int64)
    val_idx = np.asarray(val_idx).astype(np.int64)

    X_train = np.take(X, train_idx, axis=0)
    y_train = np.take(y, train_idx, axis=0)
    X_val = np.take(X, val_idx, axis=0)
    y_val = np.take(y, val_idx, axis=0)

    print(f"\nTrain windows: {len(X_train)}  |  Val windows: {len(X_val)}")

    if len(X_val) == 0:
        print("\nNo validation windows -- every class has only 1 session. "
              "Record at least one more session per gesture first.")
        return

    model = build_1dcnn(
        window_size=WINDOW_SIZE,
        num_features=len(get_feature_columns()),
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
    np.save("quest_label_classes.npy", encoder.classes_)


if __name__ == "__main__":
    main()