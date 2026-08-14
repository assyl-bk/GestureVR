"""
ablation.py
------------
Direct ablation: trains two otherwise-identical 1D-CNN models on the
SAME recordings, differing only in which sensor channels are visible
to the model:

    Single-IMU condition: only Sensor A (forearm), 4 features
                           (aw, ax, ay, az)
    Dual-IMU condition:   both sensors, 8 features
                           (aw, ax, ay, az, bw, bx, by, bz)

Because both conditions are evaluated on identical recordings with
the identical session-based split, any accuracy difference is
attributable to the presence/absence of the second sensor, not to a
different dataset, different subjects, or a different data collection
protocol -- this is the core evidence for the project's hypothesis
that relative forearm-hand rotation improves discrimination for
gestures where wrist motion and whole-arm/hand motion are correlated.

Run: python ablation.py
"""

import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix

from dataset import load_all_csvs, make_windows, WINDOW_SIZE
from splits import session_based_split
from model import build_1dcnn

SINGLE_IMU_COLUMNS = ["aw", "ax", "ay", "az"]        # Sensor A only
DUAL_IMU_COLUMNS = ["aw", "ax", "ay", "az",
                     "bw", "bx", "by", "bz"]           # both sensors

EPOCHS = 30
BATCH_SIZE = 32


def run_condition(name, X, y, groups, encoder, train_idx, val_idx):
    print(f"\n{'=' * 60}")
    print(f"Condition: {name}  ({X.shape[-1]} features)")
    print("=" * 60)

    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]

    model = build_1dcnn(
        window_size=WINDOW_SIZE,
        num_features=X.shape[-1],
        num_classes=len(encoder.classes_),
    )

    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
    )

    y_pred_probs = model.predict(X_val)
    y_pred = np.argmax(y_pred_probs, axis=1)

    val_accuracy = (y_pred == y_val).mean()
    print(f"\nFinal validation accuracy: {val_accuracy:.4f}")
    print(classification_report(
        y_val,
        y_pred,
        labels=np.arange(len(encoder.classes_)),
        target_names=encoder.classes_,
        zero_division=0,
    ))
    print("Confusion matrix:")
    print(confusion_matrix(y_val, y_pred, labels=np.arange(len(encoder.classes_))))

    return val_accuracy


def main():
    print("Loading data...")
    df = load_all_csvs()

    # Build windows once using ALL 8 columns, then slice per-condition
    # below -- this guarantees both conditions use the exact same
    # underlying time windows (same rows, same session boundaries),
    # not just the same raw data reloaded separately.
    X_full, y_raw, groups = make_windows(df)
    print(f"Total windows: {len(X_full)}  |  Window shape: {X_full.shape[1:]}")

    encoder = LabelEncoder()
    y = encoder.fit_transform(y_raw)
    print(f"Classes: {list(encoder.classes_)}")

    print("\nSplitting by recording session (shared across both conditions):")
    train_idx, val_idx = session_based_split(y, groups)

    if len(val_idx) == 0:
        print("\nNo validation windows -- every class has only 1 session. "
              "Record a 2nd session per gesture before running this.")
        return

    print(f"\nTrain windows: {len(train_idx)}  |  Val windows: {len(val_idx)}")

    all_columns = ["aw", "ax", "ay", "az", "bw", "bx", "by", "bz"]
    single_idx = [all_columns.index(c) for c in SINGLE_IMU_COLUMNS]
    dual_idx = [all_columns.index(c) for c in DUAL_IMU_COLUMNS]

    X_single = X_full[:, :, single_idx]
    X_dual = X_full[:, :, dual_idx]

    acc_single = run_condition(
        "Single IMU (forearm only)", X_single, y, groups, encoder, train_idx, val_idx
    )
    acc_dual = run_condition(
        "Dual IMU (forearm + hand)", X_dual, y, groups, encoder, train_idx, val_idx
    )

    print(f"\n{'=' * 60}")
    print("ABLATION SUMMARY")
    print("=" * 60)
    print(f"Single-IMU (forearm only) validation accuracy: {acc_single:.4f}")
    print(f"Dual-IMU (forearm + hand) validation accuracy:  {acc_dual:.4f}")
    print(f"Difference (dual - single):                     {acc_dual - acc_single:+.4f}")


if __name__ == "__main__":
    main()