# GestureVR: Meta Quest Gesture Recognition

The current project deliverable is the `ml_quest/` machine-learning
pipeline. It recognizes six trajectory gestures from Meta Quest
hand-tracking recordings and compares a temporal 1D-CNN with Random
Forest and SVM baselines.

The repository also contains the earlier ESP32 and Unity experiments.
They are retained as supporting project history; they are not required
to reproduce the `ml_quest` results.

## Results

The final result artifacts are in [`ml_quest/results/`](ml_quest/results/).
The exact protocols and verification status are documented in
[`RESULTS_README.md`](ml_quest/results/RESULTS_README.md).

| Model | Reported result | Status |
|---|---:|---|
| 1D-CNN | 88.27% ± 5.11% accuracy | Mean ± sample standard deviation over 100 training seeds |
| Random Forest | 82.8% accuracy | Reproduced by rounding an observed 82.7957% |
| SVM | 87.3% accuracy | Submitted value not exactly reproduced; closest verified result is 87.0968% |

The RF and SVM values are frame-level baseline results, while the CNN
uses temporal windows. They should therefore be interpreted as useful
baselines rather than identical model inputs.

## Dataset

The Quest recordings are stored as:

```text
ml_quest/data/raw/<gesture_label>_<session_number>/hand_path.csv
```

The six gesture classes are:

```text
clap, handdown, handup, idle, wave, wrist-rotate
```

There are 20 sessions per class. Each CSV contains:

```text
t,object,x,y,z,pitch,yaw,roll
```

The tracked objects are:

- `LeftHandAnchor`
- `RightHandAnchor`
- `Head`

## Feature preparation

[`quest_dataset.py`](ml_quest/quest_dataset.py) performs the following:

1. Loads the long-format Quest CSV files.
2. Aligns tracked objects by rounded timestamp.
3. Converts the data to one wide feature row per timestep.
4. Encodes each Euler angle as sine/cosine pairs to avoid 360/0
   discontinuities.
5. Detects all-zero tracking dropout rows.
6. Removes windows with more than 10% dropout.
7. Builds 40-frame windows with stride 8 for the CNN.

Each object contributes 9 base features:

```text
3 position features + 6 sine/cosine orientation features
```

The CNN therefore receives 27 features per timestep and an input shape
of approximately `40 x 27`.

The RF/SVM baseline uses frame-level samples. By default it adds
frame-to-frame velocity features, resulting in 54 features.

## Evaluation protocol

All models use session-aware evaluation. Complete recording sessions
are assigned to train, validation, or test; individual windows or
frames from one session are not randomly split across these sets.

This prevents overlapping windows from the same recording from leaking
between training and evaluation.

For the CNN:

- The validation sessions control early stopping and best-epoch
  selection.
- The held-out test sessions are used for final metrics.
- Seeds 0 through 99 are evaluated.
- The saved accuracy distribution is
  [`cnn_test_accuracy_distribution.png`](ml_quest/results/cnn_test_accuracy_distribution.png).

For the verified RF result:

- Session split seed: 85
- Position, orientation, and velocity features
- Every 20th clean frame
- 10 trees
- `max_features="sqrt"`
- `random_state=42`

For the closest verified SVM result:

- Session split seed: 85
- Position, orientation, and velocity features
- Every 20th clean frame
- `StandardScaler`
- RBF kernel
- `C=100`
- `gamma="scale"`
- `class_weight=None`

## Installation

From the repository root:

```powershell
cd ml_quest
pip install -r requirements.txt
```

The required packages are listed in
[`ml_quest/requirements.txt`](ml_quest/requirements.txt).

## Reproduce the pipeline

### Inspect the dataset

```powershell
cd ml_quest
python quest_dataset.py
```

### Train the CNN

```powershell
python quest_train.py
```

This trains the configured CNN across 100 seeds and writes:

- `multi_seed_results.npy`
- `quest_gesture_model.h5`
- `quest_label_classes.npy`

### Run the classical baselines

```powershell
python quest_baseline.py
python quest_baseline.py --rotation-only
python quest_baseline.py --no-velocity
```

The baseline script uses `GridSearchCV` to select RF and SVM
hyperparameters using validation balanced accuracy before evaluating the
selected models on held-out test sessions. The exhaustive search can be
computationally expensive.

## Project structure

```text
GestureVR/
├── ml_quest/
│   ├── data/raw/                    Quest hand-tracking sessions
│   ├── quest_dataset.py             Loading and feature preparation
│   ├── model.py                     1D-CNN architecture
│   ├── quest_train.py               Multi-seed CNN training
│   ├── quest_baseline.py            RF and SVM baselines
│   ├── splits.py                    Session-aware splitting
│   ├── quest_gesture_model.h5       Saved CNN model
│   ├── quest_gesture_model.keras   Keras-format CNN model
│   ├── quest_label_classes.npy      Class-label mapping
│   ├── multi_seed_results.npy       CNN metrics across 100 seeds
│   ├── requirements.txt             Python dependencies
│   └── results/                     Figures and verified summaries
├── firmware/                        Supporting ESP32 experiments
├── unity/                           Supporting Unity script
├── docs/                            Supporting wiring notes
├── LICENSE
└── README.md
```

## Limitations and future work

- The current deliverable evaluates recorded Quest sessions offline.
- Real-time Quest inference and Unity gesture-event integration are not
  yet part of the completed pipeline.
- The current data is session-independent but not established as
  cross-subject generalization.
- RF/SVM and CNN inputs differ: frame-level features versus temporal
  windows.

Future work includes real-time inference, Unity integration,
cross-subject evaluation, more participants, and deployment latency
measurement.
