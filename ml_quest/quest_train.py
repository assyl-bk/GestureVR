"""
quest_train.py
---------------
Trains a 1D-CNN on the Quest hand-tracking dataset (see
quest_dataset.py), using a session-based train/val/test split.

Fixes applied:
  1. Global random seeding (Python/NumPy/TensorFlow) -- without this,
     weight initialization and batch shuffling are non-deterministic,
     producing different accuracy on every run of identical code and
     data (observed previously: 0.93 vs 0.92 vs 0.78 across runs).
  2. Final metrics are now computed on the TEST set, not the
     validation set. The validation set is used only for early
     stopping / best-epoch selection; reporting validation metrics as
     "final results" is methodologically invalid, since the
     validation set influenced model selection.

Run: python quest_train.py
"""

import os
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow import keras

# --- Reproducibility: must happen before any model/layer is built ---
#SEED = 42
#keras.utils.set_random_seed(SEED)
SEEDS = [42, 0, 1, 7, 123]
os.environ["TF_DETERMINISTIC_OPS"] = "1"  # extra safety for CPU determinism

from quest_dataset import load_all_sessions, make_windows, WINDOW_SIZE, get_feature_columns
from splits import session_based_split
from model import build_1dcnn

MODEL_OUTPUT_PATH = "quest_gesture_model.h5"


def main(seed):
    keras.utils.set_random_seed(seed)

    print(f"\n{'=' * 60}")
    print(f"Running experiment with SEED = {seed}")
    print(f"{'=' * 60}")
    df = load_all_sessions()
    X, y_raw, groups = make_windows(df)

    print(f"Total windows: {len(X)}  |  Window shape: {X.shape[1:]}")

    encoder = LabelEncoder()
    y = encoder.fit_transform(y_raw)
    print(f"Classes: {list(encoder.classes_)}")

    print("\nSplitting by recording session (train/val/test):")

    trainval_idx, test_idx = session_based_split(y, groups, val_fraction=0.2)
    trainval_idx = np.asarray(trainval_idx).astype(np.int64)
    test_idx = np.asarray(test_idx).astype(np.int64)

    y_trainval = np.take(y, trainval_idx, axis=0)
    groups_trainval = np.take(groups, trainval_idx, axis=0)

    train_idx_rel, val_idx_rel = session_based_split(
        y_trainval, groups_trainval, val_fraction=0.2
    )
    train_idx_rel = np.asarray(train_idx_rel).astype(np.int64)
    val_idx_rel = np.asarray(val_idx_rel).astype(np.int64)

    train_idx = np.take(trainval_idx, train_idx_rel, axis=0)
    val_idx = np.take(trainval_idx, val_idx_rel, axis=0)

    X_train = np.take(X, train_idx, axis=0)
    y_train = np.take(y, train_idx, axis=0)
    X_val = np.take(X, val_idx, axis=0)
    y_val = np.take(y, val_idx, axis=0)
    X_test = np.take(X, test_idx, axis=0)
    y_test = np.take(y, test_idx, axis=0)

    print(f"\nTrain windows: {len(X_train)}  |  Val windows: {len(X_val)}  "
          f"|  Test windows: {len(X_test)}")

    if len(X_val) == 0 or len(X_test) == 0:
        print("\nNot enough sessions per class for a proper "
              "train/val/test split (need 3+ sessions per class). "
              "Record more sessions first.")
        return

    model = build_1dcnn(
        window_size=WINDOW_SIZE,
        num_features=len(get_feature_columns()),
        num_classes=len(encoder.classes_),
    )
    model.summary()

    # Validation set is used ONLY for early stopping / best-epoch
    # selection -- never for final reported metrics.
    early_stop = EarlyStopping(
        monitor="val_loss", patience=6, restore_best_weights=True
    )

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=50,
        batch_size=32,
        callbacks=[early_stop],
        verbose=2,
    )

    best_epoch = np.argmin(history.history["val_loss"]) + 1
    print(f"\nBest epoch (lowest val_loss): {best_epoch} (weights restored)")

    # --- FINAL evaluation on the held-out TEST set (never seen during
    #     training or model selection) ---
    y_pred_probs = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)

    accuracy = np.mean(y_pred == y_test)
    f1_weighted = f1_score(y_test, y_pred, average="weighted")
    f1_macro = f1_score(y_test, y_pred, average="macro")

    print("\n" + "=" * 60)
    print("1D-CNN -- FINAL TEST RESULTS")
    print("=" * 60)
    print(f"Accuracy:            {accuracy:.4f}")
    print(f"Weighted F1-score:   {f1_weighted:.4f}")
    print(f"Macro F1-score:      {f1_macro:.4f}")
    print("\nClassification Report:")
    print(classification_report(
        y_test, y_pred, target_names=encoder.classes_, zero_division=0
    ))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    model.save(MODEL_OUTPUT_PATH)
    print(f"\nModel saved to {MODEL_OUTPUT_PATH}")
    np.save("quest_label_classes.npy", encoder.classes_)
    return {
    "seed": seed,
    "accuracy": accuracy,
    "f1_weighted": f1_weighted,
    "f1_macro": f1_macro,
}


if __name__ == "__main__":

    results = []

    for seed in SEEDS:
        result = main(seed)
        results.append(result)

    # Convert results to arrays
    accuracies = np.array([r["accuracy"] for r in results])
    f1_weighted_scores = np.array([r["f1_weighted"] for r in results])
    f1_macro_scores = np.array([r["f1_macro"] for r in results])

    print("\n" + "=" * 60)
    print("MULTI-SEED FINAL RESULTS")
    print("=" * 60)

    print("\nIndividual results:")

    for r in results:
        print(
            f"Seed {r['seed']}: "
            f"Accuracy={r['accuracy']:.4f} | "
            f"Weighted F1={r['f1_weighted']:.4f} | "
            f"Macro F1={r['f1_macro']:.4f}"
        )

    print("\n" + "-" * 60)
    print("MEAN ± STANDARD DEVIATION")
    print("-" * 60)

    print(
        f"Accuracy:        "
        f"{accuracies.mean():.4f} ± {accuracies.std(ddof=1):.4f}"
    )

    print(
        f"Weighted F1:     "
        f"{f1_weighted_scores.mean():.4f} ± "
        f"{f1_weighted_scores.std(ddof=1):.4f}"
    )

    print(
        f"Macro F1:        "
        f"{f1_macro_scores.mean():.4f} ± "
        f"{f1_macro_scores.std(ddof=1):.4f}"
    )