"""
dataset.py
-----------
Loads the labeled CSV files produced by collect_data.py (in data/raw/)
and turns them into fixed-length windows suitable for training a
1D-CNN gesture classifier. Also includes loaders for the HGAG-DATA
external dataset, used separately for the baseline comparison (see
baseline.py / the report's related-work section) -- NOT merged into
the same feature space as your own 8-feature quaternion data.

IMPORTANT -- quaternion reference frame normalization:
The MPU6050 DMP has no fixed "zero rotation" reference; it's whatever
orientation the sensor happened to be in the instant the DMP
initialized. This means every power-cycle/reset gives each sensor a
brand-new, arbitrary rotational reference frame -- so RAW quaternions
from two different recording sessions are not directly comparable,
even for the exact same physical gesture. Training on raw absolute
quaternions and validating across sessions can produce near-0%
accuracy despite high training accuracy, purely because of this
reference-frame mismatch, NOT because the sensor data itself is
uninformative or the dataset is too small.

Fix: each session's quaternions are normalized relative to that
session's OWN starting orientation (the first few readings, averaged
and re-normalized to a unit quaternion), separately for Sensor A and
Sensor B. This expresses every reading as "rotation since this
session started" rather than an absolute, session-arbitrary value.

Expected input CSV columns (from dual_mpu6050_calibrated.ino /
main_esp32_receiver.ino):
    timestamp_ms,label,aw,ax,ay,az,bw,bx,by,bz
"""

import glob
import os
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

FEATURE_COLUMNS = ["aw", "ax", "ay", "az", "bw", "bx", "by", "bz"]
WINDOW_SIZE = 50        # number of timesteps per window (tune this)
WINDOW_STRIDE = 10      # step size between windows (overlapping windows)
DATA_DIR = os.path.join(BASE_DIR, "data", "raw")
BASELINE_SAMPLES = 5    # how many initial rows to average for the baseline

HGAG_DATASET_DIR = os.path.join(
    BASE_DIR,
    "dataset",
    "Hand Gesture Accelerometer and Gyroscope Dataset (HGAG-DATA)",
)

HGAG_FEATURE_COLUMNS = [
    "accel_x",
    "accel_y",
    "accel_z",
    "gyro_x",
    "gyro_y",
    "gyro_z",
]


# ---------------------------------------------------------------------
# Quaternion reference-frame normalization (own dataset only)
# ---------------------------------------------------------------------

def quat_conjugate(q):
    """q: (..., 4) array in (w, x, y, z) order. Unit quaternion inverse."""
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    return np.stack([w, -x, -y, -z], axis=-1)


def quat_multiply(q1, q2):
    """Hamilton product q1 * q2, both (..., 4) in (w, x, y, z) order."""
    w1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    w2, x2, y2, z2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]

    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

    return np.stack([w, x, y, z], axis=-1)


def normalize_quaternion_group(values, baseline_samples=BASELINE_SAMPLES):
    """
    values: (num_rows, 4) quaternion array for ONE sensor, ONE session,
            in chronological order.
    Returns: (num_rows, 4) values expressed relative to this session's
             own starting orientation.

    Handles the quaternion "double cover" property: q and -q represent
    the exact same physical rotation, and the DMP can output either
    sign somewhat arbitrarily between samples. Averaging raw samples
    without correcting for this can produce a near-zero vector (if a
    sign flip occurs within the baseline window), which then causes a
    divide-by-near-zero -> NaN. Fix: flip any baseline sample whose
    sign disagrees with the first sample before averaging.
    """
    n = min(baseline_samples, len(values))
    baseline_samples_arr = values[:n].copy()

    reference = baseline_samples_arr[0]
    for i in range(1, n):
        if np.dot(baseline_samples_arr[i], reference) < 0:
            baseline_samples_arr[i] = -baseline_samples_arr[i]

    baseline = baseline_samples_arr.mean(axis=0)
    norm = np.linalg.norm(baseline)

    if norm < 1e-8:
        # Extremely unlikely edge case even after sign correction --
        # fall back to the identity quaternion (no normalization) for
        # this session rather than crashing.
        baseline = np.array([1.0, 0.0, 0.0, 0.0])
    else:
        baseline = baseline / norm

    baseline_inv = quat_conjugate(baseline[np.newaxis, :])  # (1, 4)
    baseline_inv = np.repeat(baseline_inv, len(values), axis=0)  # (num_rows, 4)

    return quat_multiply(baseline_inv, values)


# ---------------------------------------------------------------------
# Own dataset (ESP32 dual-IMU) loading
# ---------------------------------------------------------------------

def load_all_csvs(data_dir=DATA_DIR):
    """
    Load every CSV in data_dir into one combined DataFrame, tagging
    each row with which file it came from (session_id), and
    normalizing each session's quaternions relative to that session's
    own starting orientation (see module docstring).
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

        a_vals = df[["aw", "ax", "ay", "az"]].to_numpy()
        b_vals = df[["bw", "bx", "by", "bz"]].to_numpy()

        df[["aw", "ax", "ay", "az"]] = normalize_quaternion_group(a_vals)
        df[["bw", "bx", "by", "bz"]] = normalize_quaternion_group(b_vals)

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

    for (session_id, label), group in df.groupby(["session_id", "label"]):
        values = group[FEATURE_COLUMNS].to_numpy()

        for start in range(0, len(values) - window_size + 1, stride):
            window = values[start:start + window_size]
            X.append(window)
            y.append(label)
            groups.append(session_id)

    return np.array(X), np.array(y), np.array(groups)


# ---------------------------------------------------------------------
# HGAG-DATA loading (external benchmark, kept separate -- see
# baseline comparison discussion, NOT merged into your own pipeline)
# ---------------------------------------------------------------------

def _resolve_hgag_variant_root(dataset_root: str, variant: str) -> str:
    """Resolve HGAG's nested folder layout to the requested variant root."""
    candidates = [
        os.path.join(dataset_root, variant),
        os.path.join(dataset_root, "HGAG-DATA", variant),
        os.path.join(
            dataset_root,
            "Hand Gesture Accelerometer and Gyroscope Dataset (HGAG-DATA)",
            "HGAG-DATA",
            variant,
        ),
    ]

    for path in candidates:
        if os.path.isdir(path):
            return path

    raise FileNotFoundError(
        f"Could not find HGAG variant '{variant}' under {dataset_root}."
    )


def _read_hgag_axis_csv(csv_path: str) -> np.ndarray:
    """Read HGAG axis CSVs that are typically single long comma-separated lines."""
    with open(csv_path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read().strip()

    if not text:
        return np.array([], dtype=np.float32)

    normalized = text.replace("\n", ",").replace("\r", ",")
    values = np.fromstring(normalized, sep=",", dtype=np.float32)
    return values[~np.isnan(values)]


def make_hgag_windows(
    dataset_root: str = HGAG_DATASET_DIR,
    variant: str = "HGAG-DATA1",
    window_size: int = WINDOW_SIZE,
    stride: int = WINDOW_STRIDE,
):
    """
    Build HGAG windows shaped (num_windows, window_size, 6) where 6 channels are
    accel_x/y/z and gyro_x/y/z. Session groups are subject-level paths so
    subject-wise splitting can prevent leakage between train/val/test.
    """
    variant_root = _resolve_hgag_variant_root(dataset_root, variant)

    axis_files = [
        "accel_x_data.csv",
        "accel_y_data.csv",
        "accel_z_data.csv",
        "gyro_x_data.csv",
        "gyro_y_data.csv",
        "gyro_z_data.csv",
    ]

    X, y, groups = [], [], []

    for gesture in sorted(os.listdir(variant_root)):
        gesture_path = os.path.join(variant_root, gesture)
        if not os.path.isdir(gesture_path):
            continue

        for subject in sorted(os.listdir(gesture_path)):
            subject_path = os.path.join(gesture_path, subject)
            if not os.path.isdir(subject_path):
                continue

            csv_dir = os.path.join(subject_path, ".csv")
            if not os.path.isdir(csv_dir):
                continue

            axis_arrays = []
            missing_axis = False
            for axis_file in axis_files:
                axis_path = os.path.join(csv_dir, axis_file)
                if not os.path.isfile(axis_path):
                    missing_axis = True
                    break
                axis_arrays.append(_read_hgag_axis_csv(axis_path))

            if missing_axis:
                continue

            min_len = min((len(arr) for arr in axis_arrays), default=0)
            if min_len < window_size:
                continue

            values = np.column_stack([arr[:min_len] for arr in axis_arrays]).astype(np.float32)
            session_id = f"{gesture}/{subject}"

            for start in range(0, min_len - window_size + 1, stride):
                X.append(values[start:start + window_size])
                y.append(gesture)
                groups.append(session_id)

    if not X:
        raise RuntimeError(
            "No HGAG windows were built. Check dataset path/layout and window params."
        )

    return np.asarray(X), np.asarray(y), np.asarray(groups)


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