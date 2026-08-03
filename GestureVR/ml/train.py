"""
train.py
---------
Trains the 1D-CNN gesture classifier on data collected via
collect_data.py, using dataset.py for loading/windowing and
model.py for the architecture.

IMPORTANT: validation is split by RECORDING SESSION (whole CSV files),
not by individual window. Windows from the same session overlap
heavily (sliding stride), so a random window-level split would let
the model "see" near-duplicates of validation data during training,
producing an artificially inflated, untrustworthy accuracy score.

Caveat: if a gesture only has ONE recorded session, that session
cannot be split (it has to go entirely to train OR validation), so
that class won't have a meaningful validation score yet. Record at
least 2 separate sessions per gesture for a trustworthy result.

Run: python train.py
"""

import numpy as np
from numpy.typing import NDArray
from sklearn.preprocessing import LabelEncoder

from dataset import load_all_csvs, make_windows, WINDOW_SIZE, FEATURE_COLUMNS
from model import build_1dcnn

MODEL_OUTPUT_PATH = "gesture_model.h5"
VAL_FRACTION = 0.2
RANDOM_SEED = 42


def session_based_split(
    y: np.ndarray,
    groups: np.ndarray,
    val_fraction: float = VAL_FRACTION,
    seed: int = RANDOM_SEED,
) -> tuple[np.ndarray, np.ndarray]:
    """
    For each label, split its unique recording SESSIONS (not windows)
    into train/val, so an entire session lands fully on one side.
    Labels with only 1 session go entirely to train (flagged below).
    """
    rng = np.random.default_rng(seed)
    train_idx, val_idx = [], []

    for label in np.unique(y):
        label_mask = (y == label)
        label_sessions = np.unique(groups[label_mask])
        rng.shuffle(label_sessions)

        n_val_sessions = max(1, int(len(label_sessions) * val_fraction)) \
            if len(label_sessions) > 1 else 0

        val_sessions = set(label_sessions[:n_val_sessions])
        train_sessions = set(label_sessions[n_val_sessions:])

        if len(label_sessions) == 1:
            print(f"  WARNING: '{label}' has only 1 recording session -- "
                  f"all its windows go to TRAIN. Record a 2nd session "
                  f"of this gesture for a real validation score on it.")

        for i in np.where(label_mask)[0]:
            if groups[i] in val_sessions:
                val_idx.append(i)
            else:
                train_idx.append(i)

    return np.asarray(train_idx, dtype=int), np.asarray(val_idx, dtype=int)


def main():
    print("Loading data...")
    df = load_all_csvs()
    X, y_raw, groups = make_windows(df)
    X: NDArray[np.float_] = np.asarray(X, dtype=np.float32)
    y_raw: NDArray[np.str_] = np.asarray(y_raw)
    groups: NDArray[np.str_] = np.asarray(groups)

    print(f"Total windows: {len(X)}  |  Window shape: {X.shape[1:]}")

    encoder = LabelEncoder()
    y: NDArray[np.int_] = np.asarray(encoder.fit_transform(y_raw), dtype=np.int_)
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