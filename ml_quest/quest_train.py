"""
quest_train.py
---------------
Trains a 1D-CNN on the Quest hand-tracking dataset (see
quest_dataset.py), using a session-based train/val/test split
(matching quest_baseline.py's splitting for a fair, apples-to-apples
comparison in the paper).

Run: python quest_train.py
"""

import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.callbacks import EarlyStopping

from quest_dataset import load_all_sessions, make_windows, WINDOW_SIZE, get_feature_columns
from splits import session_based_split
from model import build_1dcnn

MODEL_OUTPUT_PATH = "quest_gesture_model.h5"

#added now
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

    print("\nSplitting by recording session (train/val/test):")

    # first split point: carve off the test sessions
    trainval_idx, test_idx = session_based_split(y, groups, val_fraction=0.2)
    trainval_idx = np.asarray(trainval_idx).astype(np.int64)
    test_idx = np.asarray(test_idx).astype(np.int64)

    y_trainval = np.take(y, trainval_idx, axis=0)
    groups_trainval = np.take(groups, trainval_idx, axis=0)

    # second split point: carve the remainder into train/val
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

    early_stop = EarlyStopping(
        monitor="val_loss", patience=6, restore_best_weights=True
    )

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=50,
        batch_size=32,
        callbacks=[early_stop],
    )

    best_epoch = int(np.argmin(history.history["val_loss"])) + 1
    print(f"\nBest epoch (lowest val_loss): {best_epoch} (weights restored)")

    y_pred_probs = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)

    print("\n" + "=" * 60)
    print("1D-CNN -- TEST SET RESULTS (best epoch restored)")
    print("=" * 60)
    print(classification_report(
        y_test, y_pred, target_names=encoder.classes_, zero_division=0
    ))
    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_pred))

    model.save(MODEL_OUTPUT_PATH)
    print(f"\nModel saved to {MODEL_OUTPUT_PATH}")
    np.save("quest_label_classes.npy", encoder.classes_)


if __name__ == "__main__":
    main()