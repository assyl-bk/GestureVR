"""
baseline.py
------------
Classical ML baseline (SVM, Random Forest) trained on hand-crafted
summary statistics of each window, evaluated with the same
session-based split as train.py.

Purpose: a sanity-check floor for the 1D-CNN. If a simple classical
model gets close to the CNN's accuracy, the CNN isn't earning its
extra complexity yet; if the CNN clearly outperforms it, that's
evidence the deep learning approach is worthwhile for this task.

Features per window, per channel (8 channels x 6 stats = 48 features):
    mean, std, min, max, range, and root-mean-square

Run: python baseline.py
"""

import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

from dataset import load_all_csvs, make_windows, FEATURE_COLUMNS
from splits import session_based_split


def extract_window_features(X):
    """
    X: (num_windows, window_size, num_channels)
    Returns: (num_windows, num_channels * 6) hand-crafted features
    """
    mean = X.mean(axis=1)
    std = X.std(axis=1)
    minimum = X.min(axis=1)
    maximum = X.max(axis=1)
    rng = maximum - minimum
    rms = np.sqrt((X ** 2).mean(axis=1))

    return np.concatenate([mean, std, minimum, maximum, rng, rms], axis=1)


def main():
    print("Loading data...")
    df = load_all_csvs()
    X, y_raw, groups = make_windows(df)
    X = np.asarray(X)
    y_raw = np.asarray(y_raw)
    groups = np.asarray(groups)
    print(f"Total windows: {len(X)}  |  Window shape: {X.shape[1:]}")

    encoder = LabelEncoder()
    y = encoder.fit_transform(y_raw)
    print(f"Classes: {list(encoder.classes_)}")

    print("\nSplitting by recording session (not by window):")
    train_idx, val_idx = session_based_split(y, groups)

    if len(val_idx) == 0:
        print("\nNo validation windows -- every class has only 1 session. "
              "Record a 2nd session per gesture before running this.")
        return

    print("\nExtracting hand-crafted features (mean/std/min/max/range/rms "
          "per channel)...")
    X_feat = np.asarray(extract_window_features(X))
    print(f"Feature vector size: {X_feat.shape[1]} "
          f"({len(FEATURE_COLUMNS)} channels x 6 stats)")

    X_feat = np.asarray(X_feat)
    y = np.asarray(y)

    X_train, y_train = X_feat[train_idx], y[train_idx]
    X_val, y_val = X_feat[val_idx], y[val_idx]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    print(f"\nTrain windows: {len(X_train)}  |  Val windows: {len(X_val)}")

    for name, clf in [
        ("SVM (RBF kernel)", SVC(kernel="rbf", C=1.0)),
        ("Random Forest", RandomForestClassifier(n_estimators=200, random_state=42)),
    ]:
        print(f"\n{'=' * 60}")
        print(f"{name}")
        print("=" * 60)

        clf.fit(X_train_scaled, y_train)
        y_pred = clf.predict(X_val_scaled)

        print(classification_report(
            y_val,
            y_pred,
            labels=np.arange(len(encoder.classes_)),
            target_names=encoder.classes_,
            zero_division=0,
        ))
        print("Confusion matrix:")
        print(confusion_matrix(y_val, y_pred, labels=np.arange(len(encoder.classes_))))


if __name__ == "__main__":
    main()