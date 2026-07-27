"""
dataset.py
-----------
Loads the labeled CSV files produced by collect_data.py (in data/raw/)
and turns them into fixed-length windows suitable for training a
1D-CNN gesture classifier.

This is a STARTING SKELETON -- fill in / adjust once you have real
recorded data and have decided on final window size / feature set.

Expected input CSV columns (from dual_mpu6050_calibrated.ino):
    timestamp_ms,label,aw,ax,ay,az,bw,bx,by,bz
"""

import glob
import numpy as np
import pandas as pd

FEATURE_COLUMNS = ["aw", "ax", "ay", "az", "bw", "bx", "by", "bz"]
WINDOW_SIZE = 50        # number of timesteps per window (tune this)
WINDOW_STRIDE = 10      # step size between windows (overlapping windows)
DATA_DIR = "data/raw"


def load_all_csvs(data_dir=DATA_DIR):
    """Load every CSV in data_dir into one combined DataFrame."""
    files = glob.glob(f"{data_dir}/*.csv")
    if not files:
        raise FileNotFoundError(
            f"No CSV files found in {data_dir}. Record some gestures "
            f"with collect_data.py first."
        )

    frames = [pd.read_csv(f) for f in files]
    return pd.concat(frames, ignore_index=True)


def make_windows(df, window_size=WINDOW_SIZE, stride=WINDOW_STRIDE):
    """
    Slice each label's contiguous readings into fixed-length windows.
    Returns:
        X: np.array of shape (num_windows, window_size, num_features)
        y: np.array of shape (num_windows,) -- string labels
    """
    X, y = [], []

    # Process each label's rows separately so windows never mix
    # two different gestures together.
    for label, group in df.groupby("label"):
        values = group[FEATURE_COLUMNS].to_numpy()

        for start in range(0, len(values) - window_size + 1, stride):
            window = values[start:start + window_size]
            X.append(window)
            y.append(label)

    return np.array(X), np.array(y)


if __name__ == "__main__":
    df = load_all_csvs()
    print(f"Loaded {len(df)} raw rows across labels: {df['label'].unique()}")

    X, y = make_windows(df)
    print(f"Built {len(X)} windows of shape {X.shape[1:]}")
    print("Label distribution:")
    unique, counts = np.unique(y, return_counts=True)
    for label, count in zip(unique, counts):
        print(f"  {label}: {count}")
