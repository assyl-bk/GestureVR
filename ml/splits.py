"""
splits.py
----------
Shared session-based train/validation splitting logic, used by
train.py, baseline.py, and ablation.py so every experiment in the
project uses the exact same, honest splitting method -- this matters
for methodological consistency when comparing results across scripts
in the report/paper.

Splits by whole recording SESSION (not by individual window), since
windows from the same session overlap heavily (sliding stride) and a
random window-level split would let a model "see" near-duplicates of
validation data during training.
"""

import numpy as np

VAL_FRACTION = 0.2
RANDOM_SEED = 42


def session_based_split(y, groups, val_fraction=VAL_FRACTION, seed=RANDOM_SEED):
    """
    For each label, split its unique recording SESSIONS (not windows)
    into train/val, so an entire session lands fully on one side.
    Labels with only 1 session go entirely to train (flagged below).

    Returns (train_idx, val_idx) -- integer index arrays into the
    original X/y/groups arrays.
    """
    rng = np.random.default_rng(seed)
    train_idx, val_idx = [], []

    for label in np.unique(y):
        label_mask = (y == label)
        label_sessions = np.unique(groups[label_mask])
        rng.shuffle(label_sessions)

        n_val_sessions = max(1, int(len(label_sessions) * val_fraction)) \
            if len(label_sessions) > 1 else 0

        val_sessions = set(label_sessions[:n_val_sessions])

        if len(label_sessions) == 1:
            print(f"  WARNING: '{label}' has only 1 recording session -- "
                  f"all its windows go to TRAIN. Record a 2nd session "
                  f"of this gesture for a real validation score on it.")

        for i in np.where(label_mask)[0]:
            if groups[i] in val_sessions:
                val_idx.append(i)
            else:
                train_idx.append(i)

    return np.array(train_idx), np.array(val_idx)