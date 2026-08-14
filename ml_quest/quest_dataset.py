"""
quest_dataset.py
------------------
Loads Quest hand-tracking sessions (hand_path.csv files from
HandTracker.cs) and builds fixed-length windows for a 1D-CNN.

Expected folder structure:
    data/raw/<gesture_label>_<session_number>/hand_path.csv
    e.g. data/raw/wave_1/hand_path.csv, data/raw/wave_2/hand_path.csv

Each hand_path.csv is in LONG format (one row per tracked object per
timestep):
    t,object,x,y,z,pitch,yaw,roll

This script PIVOTS that into WIDE format (one row per timestep, all
3 tracked objects' features side by side), since a 1D-CNN needs a
fixed feature vector per timestep, not a variable number of rows.

IMPORTANT -- cyclic angle handling:
Raw Euler angles (pitch/yaw/roll) wrap discontinuously at 360/0 deg.
A real rotation from 359 to 1 degrees would otherwise look like a
huge, fake jump to the model. Each angle is converted to (sin, cos)
pairs, which have no discontinuity, before windowing.

Per-object feature count: 3 position (x,y,z) + 3 angles x 2 (sin,cos)
                         = 3 + 6 = 9 features
Total feature count: 3 objects (LeftHandAnchor, RightHandAnchor, Head)
                    x 9 = 27 features per timestep
"""

import glob
import os
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data", "raw")

TRACKED_OBJECTS = ["LeftHandAnchor", "RightHandAnchor", "Head"]
WINDOW_SIZE = 40      # ~2 seconds at 20 samples/sec (sampleInterval=0.05)
WINDOW_STRIDE = 8

# Quest reports a lost-tracking frame (hand out of camera view,
# occluded, etc.) as an exact all-zero row (x=y=z=0, pitch=yaw=roll=0).
# Real motion essentially never produces exactly zero on all six
# values simultaneously, so this is a reliable dropout signal, not a
# real reading. Windows with more than this fraction of dropout rows
# for any tracked object are discarded rather than fed to the model.
MAX_DROPOUT_FRACTION = 0.10

# Folders whose meaning is currently ambiguous are skipped rather than
# guessed at. Numbers 1-10 are now known to be gesture classes
# (number hand-motions), so they are no longer excluded here.
EXCLUDE_FOLDERS = set()


def _angle_to_sincos(df, col):
    radians = np.deg2rad(df[col])
    return np.sin(radians), np.cos(radians)


def load_session(csv_path, label, session_id):
    """
    Load one hand_path.csv, pivot to wide format, return a DataFrame
    with one row per timestep and all objects' features as columns.
    """
    long_df = pd.read_csv(csv_path)

    long_df["t_rounded"] = long_df["t"].round(2)

    wide_frames = []
    for obj in TRACKED_OBJECTS:
        obj_df = long_df[long_df["object"] == obj].copy()
        if obj_df.empty:
            continue

        is_dropout = (
            (obj_df["x"] == 0) & (obj_df["y"] == 0) & (obj_df["z"] == 0) &
            (obj_df["pitch"] == 0) & (obj_df["yaw"] == 0) & (obj_df["roll"] == 0)
        )

        sin_pitch, cos_pitch = _angle_to_sincos(obj_df, "pitch")
        sin_yaw, cos_yaw = _angle_to_sincos(obj_df, "yaw")
        sin_roll, cos_roll = _angle_to_sincos(obj_df, "roll")

        features = pd.DataFrame({
            "t_rounded": obj_df["t_rounded"].values,
            f"{obj}_x": obj_df["x"].values,
            f"{obj}_y": obj_df["y"].values,
            f"{obj}_z": obj_df["z"].values,
            f"{obj}_sin_pitch": sin_pitch.values,
            f"{obj}_cos_pitch": cos_pitch.values,
            f"{obj}_sin_yaw": sin_yaw.values,
            f"{obj}_cos_yaw": cos_yaw.values,
            f"{obj}_sin_roll": sin_roll.values,
            f"{obj}_cos_roll": cos_roll.values,
            f"{obj}_dropout": is_dropout.values,
        })
        wide_frames.append(features.set_index("t_rounded"))

    if not wide_frames:
        return None

    wide_df = pd.concat(wide_frames, axis=1, join="inner").reset_index()
    wide_df["label"] = label
    wide_df["session_id"] = session_id
    return wide_df


def load_all_sessions(data_dir=DATA_DIR):
    """
    Scans data_dir for <label>_<n>/hand_path.csv folders, loads and
    pivots each one, returns one combined wide-format DataFrame.
    """
    session_folders = sorted(glob.glob(os.path.join(data_dir, "*")))
    if not session_folders:
        raise FileNotFoundError(f"No session folders found in {data_dir}")

    frames = []
    skipped = []
    for folder in session_folders:
        if not os.path.isdir(folder):
            continue

        folder_name = os.path.basename(folder)

        if folder_name in EXCLUDE_FOLDERS:
            skipped.append(folder_name)
            continue

        csv_path = os.path.join(folder, "hand_path.csv")
        if not os.path.isfile(csv_path):
            continue

        parts = folder_name.rsplit("_", 1)
        if len(parts) == 2 and parts[1].isdigit():
            label = parts[0]
        else:
            label = folder_name

        session_df = load_session(csv_path, label, folder_name)
        if session_df is not None:
            frames.append(session_df)

    if skipped:
        print(f"Skipped {len(skipped)} unlabeled/ambiguous folder(s): "
              f"{sorted(skipped)}")
        print("Rename these to <gesture_name>_<n> and remove them from "
              "EXCLUDE_FOLDERS in quest_dataset.py to include them.")

    if not frames:
        raise RuntimeError(f"No valid sessions loaded from {data_dir}")

    return pd.concat(frames, ignore_index=True)


def get_feature_columns(rotation_only=False):
    """
    rotation_only=True excludes x/y/z position, keeping only the
    sin/cos-encoded orientation features -- matches the reference
    paper's finding that rotational data alone is more robust than
    position (high inter-subject variability in position), and lets
    you run the same rotation-only vs. rotation+position ablation
    they report in Section III.A.
    """
    cols = []
    for obj in TRACKED_OBJECTS:
        if not rotation_only:
            cols += [f"{obj}_x", f"{obj}_y", f"{obj}_z"]
        cols += [
            f"{obj}_sin_pitch", f"{obj}_cos_pitch",
            f"{obj}_sin_yaw", f"{obj}_cos_yaw",
            f"{obj}_sin_roll", f"{obj}_cos_roll",
        ]
    return cols


def make_windows(df, window_size=WINDOW_SIZE, stride=WINDOW_STRIDE,
                  max_dropout_fraction=MAX_DROPOUT_FRACTION,
                  rotation_only=False):
    """
    Slice each session's contiguous readings into fixed-length
    windows -- never sliding across a session boundary. Windows with
    more than max_dropout_fraction of lost-tracking rows (for any
    tracked object) are skipped rather than fed to the model.
    """
    feature_columns = get_feature_columns(rotation_only=rotation_only)
    dropout_columns = [f"{obj}_dropout" for obj in TRACKED_OBJECTS]

    X, y, groups = [], [], []
    skipped_windows = 0

    for session_id, group in df.groupby("session_id"):
        group = group.sort_values("t_rounded")
        values = group[feature_columns].to_numpy()
        dropout_flags = group[dropout_columns].to_numpy()
        label = group["label"].iloc[0]

        for start in range(0, len(values) - window_size + 1, stride):
            window = values[start:start + window_size]
            window_dropout = dropout_flags[start:start + window_size]

            bad_rows = window_dropout.any(axis=1)
            dropout_fraction = bad_rows.mean()

            if dropout_fraction > max_dropout_fraction:
                skipped_windows += 1
                continue

            X.append(window)
            y.append(label)
            groups.append(session_id)

    if skipped_windows:
        print(f"Skipped {skipped_windows} window(s) with >"
              f"{max_dropout_fraction:.0%} tracking dropout.")

    return np.array(X), np.array(y), np.array(groups)


if __name__ == "__main__":
    df = load_all_sessions()
    print(f"Loaded {len(df)} timesteps across sessions: "
          f"{df['session_id'].unique()}")
    print(f"Labels found: {df['label'].unique()}")

    X, y, groups = make_windows(df)
    print(f"\nBuilt {len(X)} windows of shape {X.shape[1:]}")
    print("Label distribution:")
    unique, counts = np.unique(y, return_counts=True)
    for label, count in zip(unique, counts):
        sessions_for_label = df[df["label"] == label]["session_id"].nunique()
        print(f"  {label}: {count} windows across {sessions_for_label} session(s)")