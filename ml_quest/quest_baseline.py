"""
quest_baseline.py
-------------------
Frame-level classical ML baseline (Random Forest + SVM), following
the methodology in:

  Janstova, Tomes, Mares (2025). "Augmented Reality-Based Gesture
  Classification: A Data-Driven Analysis Using Meta Quest 3 Hand
  Tracking." ACDSA 2025.

Key methodological choices matched to that paper:
  - Classification is FRAME-LEVEL (each single timestep is one
    sample), not window-level -- their Section II.C.
  - Splits are done at the recording SESSION level (their Section
    II.C, single-subject case), matching our splits.py.
  - Hyperparameters are tuned via grid search (their Section II.F).
  - Reports accuracy, balanced accuracy, F1, and confusion matrix.
  - Supports a rotation-only vs. rotation+position ablation.

NEW: optional velocity features (frame-to-frame deltas), computed per
session. This gives frame-level classifiers local temporal context --
distinguishing a genuinely still 'idle' frame (near-zero velocity)
from a frame that happens to pass through a similar position/
orientation mid-gesture (non-zero velocity) -- without abandoning the
paper's frame-based methodology.

Run:
    python quest_baseline.py                        # rotation + position
    python quest_baseline.py --rotation-only          # rotation only
    python quest_baseline.py --no-velocity            # disable velocity features
"""

import argparse
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, PredefinedSplit
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, balanced_accuracy_score, f1_score,
)

from quest_dataset import load_all_sessions, get_feature_columns
from splits import session_based_split

RF_PARAM_GRID = {
    "n_estimators": [50, 100, 300],
    "max_depth": [None, 10, 30],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 3, 5],
    "max_features": ["sqrt", "log2"],
    "class_weight": [None, "balanced"],
}

SVM_PARAM_GRID = {
    "kernel": ["rbf", "poly"],
    "C": [0.1, 1, 10, 100],
    "gamma": ["scale", "auto", 0.01, 0.1],
    "degree": [2, 3],
    "class_weight": [None, "balanced"],
}


def build_frame_level_dataset(df, rotation_only=False, include_velocity=True):
    """
    Unlike make_windows() (used by the CNN), this treats every single
    timestep as one training sample -- matching the reference paper's
    frame-based classification approach.
    """
    feature_columns = get_feature_columns(rotation_only=rotation_only)
    dropout_columns = [f"{obj}_dropout" for obj in
                        ["LeftHandAnchor", "RightHandAnchor", "Head"]]

    any_dropout = df[dropout_columns].any(axis=1)
    clean_df = df[~any_dropout].copy()

    if include_velocity:
        clean_df = clean_df.sort_values(["session_id", "t_rounded"])
        velocity_frames = []
        for session_id, group in clean_df.groupby("session_id"):
            deltas = group[feature_columns].diff().fillna(0.0)
            deltas.columns = [f"{c}_vel" for c in feature_columns]
            velocity_frames.append(deltas)
        velocity_df = pd.concat(velocity_frames)
        clean_df = pd.concat([clean_df, velocity_df], axis=1)
        feature_columns = feature_columns + [f"{c}_vel" for c in feature_columns]

    X = clean_df[feature_columns].to_numpy()
    y = clean_df["label"].to_numpy()
    groups = clean_df["session_id"].to_numpy()

    return X, y, groups, feature_columns


def run_grid_search(name, estimator, param_grid, X_train, y_train,
                     X_val, y_val):
    X_combined = np.concatenate([X_train, X_val])
    y_combined = np.concatenate([y_train, y_val])
    test_fold = np.concatenate([
        np.full(len(X_train), -1),
        np.full(len(X_val), 0),
    ])
    ps = PredefinedSplit(test_fold)

    print(f"\nRunning grid search for {name}...")
    search = GridSearchCV(estimator, param_grid, cv=ps,
                           scoring="balanced_accuracy", n_jobs=-1)
    search.fit(X_combined, y_combined)

    print(f"Best params for {name}: {search.best_params_}")
    return search.best_estimator_


def evaluate(name, model, X_test, y_test, encoder):
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    bal_acc = balanced_accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted")

    print(f"\n{'=' * 60}\n{name} -- TEST SET RESULTS\n{'=' * 60}")
    print(f"Accuracy:          {acc:.4f}")
    print(f"Balanced accuracy: {bal_acc:.4f}")
    print(f"Weighted F1:       {f1:.4f}")
    print(classification_report(
        y_test, y_pred, target_names=encoder.classes_, zero_division=0
    ))
    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_pred))

    return acc, bal_acc, f1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rotation-only", action="store_true",
                         help="Use only rotational features (no x/y/z position).")
    parser.add_argument("--no-velocity", action="store_true",
                         help="Disable velocity (frame-to-frame delta) features.")
    args = parser.parse_args()

    print("Loading Quest sessions...")
    df = load_all_sessions()

    include_velocity = not args.no_velocity
    print(f"\nBuilding frame-level dataset "
          f"({'rotation-only' if args.rotation_only else 'rotation+position'}, "
          f"{'with' if include_velocity else 'without'} velocity)...")
    X, y_raw, groups, feature_columns = build_frame_level_dataset(
        df, rotation_only=args.rotation_only, include_velocity=include_velocity
    )
    print(f"Total frames: {len(X)}  |  Feature dimension: {X.shape[1]}")

    encoder = LabelEncoder()
    y = encoder.fit_transform(y_raw)
    print(f"Classes: {list(encoder.classes_)}")

    print("\nSplitting by recording session (train/val/test):")
    trainval_idx, test_idx = session_based_split(y, groups, val_fraction=0.2)
    trainval_idx = np.asarray(trainval_idx, dtype=np.int64)
    test_idx = np.asarray(test_idx, dtype=np.int64)

    y_trainval = np.take(y, trainval_idx, axis=0)
    groups_trainval = np.take(groups, trainval_idx, axis=0)

    train_idx_rel, val_idx_rel = session_based_split(
        y_trainval, groups_trainval, val_fraction=0.2
    )
    train_idx_rel = np.asarray(train_idx_rel, dtype=np.int64)
    val_idx_rel = np.asarray(val_idx_rel, dtype=np.int64)

    train_idx = np.take(trainval_idx, train_idx_rel, axis=0)
    val_idx = np.take(trainval_idx, val_idx_rel, axis=0)

    print(f"\nTrain frames: {len(train_idx)}  |  Val frames: {len(val_idx)}  "
          f"|  Test frames: {len(test_idx)}")

    if len(val_idx) == 0 or len(test_idx) == 0:
        print("\nNot enough sessions per class yet for a proper "
              "train/val/test split (need 3+ sessions per class). "
              "Record more sessions first.")
        return

    scaler = StandardScaler()
    X_train = scaler.fit_transform(np.take(X, train_idx, axis=0))
    X_val = scaler.transform(np.take(X, val_idx, axis=0))
    X_test = scaler.transform(np.take(X, test_idx, axis=0))

    y_train = np.take(y, train_idx, axis=0)
    y_val = np.take(y, val_idx, axis=0)
    y_test = np.take(y, test_idx, axis=0)

    results = {}

    rf_best = run_grid_search(
        "Random Forest", RandomForestClassifier(random_state=42),
        RF_PARAM_GRID, X_train, y_train, X_val, y_val
    )
    results["Random Forest"] = evaluate("Random Forest", rf_best, X_test, y_test, encoder)

    svm_best = run_grid_search(
        "SVM", SVC(), SVM_PARAM_GRID, X_train, y_train, X_val, y_val
    )
    results["SVM"] = evaluate("SVM", svm_best, X_test, y_test, encoder)

    print(f"\n{'=' * 60}\nSUMMARY\n{'=' * 60}")
    print(f"{'Model':<20}{'Accuracy':<12}{'Balanced Acc':<15}{'F1':<10}")
    for name, (acc, bal_acc, f1) in results.items():
        print(f"{name:<20}{acc:<12.4f}{bal_acc:<15.4f}{f1:<10.4f}")

    if hasattr(rf_best, "feature_importances_"):
        importances = rf_best.feature_importances_
        top_idx = np.argsort(importances)[::-1][:10]
        print("\nTop 10 most important features (Random Forest):")
        for i in top_idx:
            print(f"  {feature_columns[i]}: {importances[i]:.4f}")


if __name__ == "__main__":
    main()