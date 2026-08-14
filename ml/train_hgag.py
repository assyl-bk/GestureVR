"""
train_hgag.py
--------------
Train/evaluate the gesture classifier on the external HGAG dataset.

This script uses subject-level grouping so windows from one subject do not
appear across train/validation/test simultaneously.

Run:
    python train_hgag.py
Optional:
    python train_hgag.py --epochs 20 --batch-size 64
"""

# pyright: reportMissingImports=false

import argparse
import os
import numpy as np
from numpy.typing import NDArray
import keras
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

from dataset import (
    HGAG_DATASET_DIR,
    HGAG_FEATURE_COLUMNS,
    WINDOW_SIZE,
    make_hgag_windows,
)
from model import build_hgag_cnn

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_OUTPUT_PATH = os.path.join(BASE_DIR, "gesture_model_hgag.h5")
LABEL_OUTPUT_PATH = os.path.join(BASE_DIR, "label_classes_hgag.npy")
REPORT_OUTPUT_PATH = os.path.join(BASE_DIR, "hgag_test_report.txt")
CONFUSION_OUTPUT_PATH = os.path.join(BASE_DIR, "hgag_confusion_matrix.npy")
RANDOM_SEED = 42


def split_subjects_per_label(
    y: NDArray[np.int_],
    groups: NDArray[np.str_],
    train_fraction: float = 0.7,
    val_fraction: float = 0.15,
    seed: int = RANDOM_SEED,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split by subject-session within each label to avoid leakage."""
    rng = np.random.default_rng(seed)
    train_idx, val_idx, test_idx = [], [], []

    for label in np.unique(y):
        label_mask = y == label
        label_groups = np.unique(groups[label_mask])
        rng.shuffle(label_groups)
        n_groups = len(label_groups)

        if n_groups == 1:
            train_groups = set(label_groups)
            val_groups = set()
            test_groups = set()
        elif n_groups == 2:
            train_groups = {label_groups[0]}
            val_groups = set()
            test_groups = {label_groups[1]}
        else:
            n_train = max(1, int(n_groups * train_fraction))
            n_val = max(1, int(n_groups * val_fraction))

            if n_train + n_val >= n_groups:
                n_val = max(1, n_groups - n_train - 1)
                if n_train + n_val >= n_groups:
                    n_train = max(1, n_groups - 2)

            train_groups = set(label_groups[:n_train])
            val_groups = set(label_groups[n_train:n_train + n_val])
            test_groups = set(label_groups[n_train + n_val:])

            if not test_groups:
                moved = next(iter(val_groups))
                val_groups.remove(moved)
                test_groups.add(moved)

        for idx in np.where(label_mask)[0]:
            group = groups[idx]
            if group in train_groups:
                train_idx.append(idx)
            elif group in val_groups:
                val_idx.append(idx)
            else:
                test_idx.append(idx)

    return (
        np.asarray(train_idx, dtype=int),
        np.asarray(val_idx, dtype=int),
        np.asarray(test_idx, dtype=int),
    )


def standardize_from_train(
    x_train: np.ndarray,
    x_val: np.ndarray,
    x_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Normalize channels using train-only statistics."""
    mean = x_train.mean(axis=(0, 1), keepdims=True)
    std = x_train.std(axis=(0, 1), keepdims=True) + 1e-6

    x_train = (x_train - mean) / std
    x_val = (x_val - mean) / std if len(x_val) else x_val
    x_test = (x_test - mean) / std if len(x_test) else x_test
    return x_train, x_val, x_test


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train/evaluate on HGAG-DATA")
    parser.add_argument("--dataset-root", default=HGAG_DATASET_DIR)
    parser.add_argument("--variant", default="HGAG-DATA1")
    parser.add_argument("--window-size", type=int, default=WINDOW_SIZE)
    parser.add_argument("--stride", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print("Loading HGAG dataset...")
    x, y_raw, groups = make_hgag_windows(
        dataset_root=args.dataset_root,
        variant=args.variant,
        window_size=args.window_size,
        stride=args.stride,
    )

    x: NDArray[np.float32] = np.asarray(x, dtype=np.float32)
    y_raw: NDArray[np.str_] = np.asarray(y_raw)
    groups: NDArray[np.str_] = np.asarray(groups)

    print(f"Total windows: {len(x)} | Window shape: {x.shape[1:]}")
    print(f"Total subject sessions: {len(np.unique(groups))}")

    encoder = LabelEncoder()
    y: NDArray[np.int_] = np.asarray(encoder.fit_transform(y_raw), dtype=np.int_)
    print(f"Classes ({len(encoder.classes_)}): {list(encoder.classes_)}")

    train_idx, val_idx, test_idx = split_subjects_per_label(y, groups)
    x_train, y_train = x[train_idx], y[train_idx]
    x_val, y_val = x[val_idx], y[val_idx]
    x_test, y_test = x[test_idx], y[test_idx]

    print(
        f"Split sizes -> train: {len(x_train)}, val: {len(x_val)}, test: {len(x_test)}"
    )

    if len(x_train) == 0 or len(x_test) == 0:
        raise RuntimeError("Insufficient train/test windows after subject split.")

    x_train, x_val, x_test = standardize_from_train(x_train, x_val, x_test)

    model = build_hgag_cnn(
        window_size=args.window_size,
        num_features=len(HGAG_FEATURE_COLUMNS),
        num_classes=len(encoder.classes_),
    )

    callbacks = []
    if len(x_val) > 0:
        callbacks.extend([
            keras.callbacks.EarlyStopping(
                monitor="val_accuracy",
                mode="max",
                patience=5,
                restore_best_weights=True,
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.5,
                patience=2,
                min_lr=1e-5,
            ),
        ])

    fit_kwargs = {
        "x": x_train,
        "y": y_train,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "callbacks": callbacks,
        "verbose": 1,
    }
    if len(x_val) > 0:
        fit_kwargs["validation_data"] = (x_val, y_val)

    model.fit(**fit_kwargs)

    test_probs = model.predict(x_test)
    y_pred = np.argmax(test_probs, axis=1)

    test_acc = accuracy_score(y_test, y_pred)
    report_text = classification_report(
        y_test,
        y_pred,
        target_names=encoder.classes_,
        output_dict=False,
    )
    cm = confusion_matrix(y_test, y_pred)

    print(f"\nHGAG test accuracy: {test_acc:.4f}")
    print("\nClassification report:")
    print(report_text)
    print("Confusion matrix:")
    print(cm)

    model.save(MODEL_OUTPUT_PATH)
    np.save(LABEL_OUTPUT_PATH, encoder.classes_)
    np.save(CONFUSION_OUTPUT_PATH, cm)

    with open(REPORT_OUTPUT_PATH, "w", encoding="utf-8") as f:
        f.write(f"HGAG test accuracy: {test_acc:.4f}\n\n")
        f.write("Classification report:\n")
        f.write(str(report_text))
        f.write("\nConfusion matrix:\n")
        f.write(np.array2string(cm))
        f.write("\n")

    print(f"\nSaved model: {MODEL_OUTPUT_PATH}")
    print(f"Saved label classes: {LABEL_OUTPUT_PATH}")
    print(f"Saved test report: {REPORT_OUTPUT_PATH}")
    print(f"Saved confusion matrix: {CONFUSION_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
