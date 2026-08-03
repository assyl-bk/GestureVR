"""
dataset.py
-----------
Loads the labeled CSV files produced by collect_data.py (in data/raw/)
and turns them into fixed-length windows suitable for training a
1D-CNN gesture classifier.

Expected input CSV columns (from dual_mpu6050_calibrated.ino /
main_esp32_receiver.ino):
    timestamp_ms,label,aw,ax,ay,az,bw,bx,by,bz
"""

import glob
import os
import numpy as np
import pandas as pd

FEATURE_COLUMNS = ["aw", "ax", "ay", "az", "bw", "bx", "by", "bz"]
WINDOW_SIZE = 50        # number of timesteps per window (tune this)
WINDOW_STRIDE = 10      # step size between windows (overlapping windows)
DATA_DIR = "data/raw"


def load_all_csvs(data_dir=DATA_DIR):
    """
    Load every CSV in data_dir into one combined DataFrame, tagging
    each row with which file it came from (session_id). This matters
    because windows must never be built across a boundary between two
    separate recording sessions, and later splits must keep whole
    sessions together (not just individual windows) to get an honest
    validation accuracy.
    """
    files = glob.glob(f"{data_dir}/*.csv")
    if not files:
        raise FileNotFoundError(
            f"No CSV files found in {data_dir}. Record some gestures "
            f"with collect_data.py first."
        )

    frames = []
    for f in files:
        df = pd.read_csv(f)
        df["session_id"] = os.path.basename(f)
        frames.append(df)

    return pd.concat(frames, ignore_index=True)


def make_windows(df, window_size=WINDOW_SIZE, stride=WINDOW_STRIDE):
    """
    Slice each (session_id, label) group's contiguous readings into
    fixed-length windows -- never sliding across a boundary between
    two different recording sessions.

    Returns:
        X: np.array of shape (num_windows, window_size, num_features)
        y: np.array of shape (num_windows,) -- string labels
        groups: np.array of shape (num_windows,) -- session_id each
                window came from, used for session-based train/val
                splitting in train.py (never split one session's
                windows across both train and validation).
    """
    X, y, groups = [], [], []

    # Group by (session_id, label) so windows never mix rows from two
    # different files, and never mix two different gestures together.
    for (session_id, label), group in df.groupby(["session_id", "label"]):
        values = group[FEATURE_COLUMNS].to_numpy()

        for start in range(0, len(values) - window_size + 1, stride):
            window = values[start:start + window_size]
            X.append(window)
            y.append(label)
            groups.append(session_id)

    return np.array(X), np.array(y), np.array(groups)


if __name__ == "__main__":
    df = load_all_csvs()
    print(f"Loaded {len(df)} raw rows across labels: {df['label'].unique()}")
    print(f"From {df['session_id'].nunique()} separate recording sessions")

    X, y, groups = make_windows(df)
    print(f"Built {len(X)} windows of shape {X.shape[1:]}")
    print("Label distribution:")
    unique, counts = np.unique(y, return_counts=True)
    for label, count in zip(unique, counts):
        sessions_for_label = df[df["label"] == label]["session_id"].nunique()
        print(f"  {label}: {count} windows across {sessions_for_label} session(s)")