# Final Results and Configurations

This folder contains the figures and result summaries used to verify the
submitted presentation values.

## 1D-CNN

- Source results: `../multi_seed_results.npy`
- Number of runs: 100
- Random seeds: 0 through 99
- Input: 40-frame temporal windows with 27 features per frame
- Split: session-based held-out test split
- Validation: validation sessions used for early stopping and best-epoch
  selection
- Final metrics: evaluated on the held-out test sessions
- Accuracy: 88.27% ± 5.11 percentage points
- Weighted F1: 88.86% ± 5.34 percentage points
- Macro F1: 85.93% ± 6.03 percentage points

The CNN accuracy figure is:

- `cnn_test_accuracy_distribution.png`

The reported spread is the sample standard deviation across the 100
training seeds. These are not 100 different test datasets.

## Random Forest

The submitted accuracy of 82.8% was reproduced by rounding the observed
accuracy:

- Observed accuracy: 82.7957%
- Reported accuracy: 82.8%
- Session split seed: 85

Configuration:

- Features: position + orientation + velocity
- Data reduction: every 20th clean frame
- Estimator: `RandomForestClassifier`
- Number of trees: 10
- `max_features`: `sqrt`
- `random_state`: 42
- Evaluation: held-out test sessions

Additional metrics:

- Balanced accuracy: 82.65%
- Weighted F1: 82.91%

Figure:

- `random_forest_verified_result.png`

## SVM

The submitted accuracy of 87.3% was not exactly reproduced. The closest
tested result was:

- Observed accuracy: 87.0968%
- Rounded to two decimals: 87.10%
- Session split seed: 85

Configuration:

- Features: position + orientation + velocity
- Data reduction: every 20th clean frame
- Preprocessing: `StandardScaler`
- Estimator: `SVC`
- Kernel: `rbf`
- `C`: 100
- `gamma`: `scale`
- `class_weight`: `None`
- Evaluation: held-out test sessions

Additional metrics:

- Balanced accuracy: 88.04%
- Weighted F1: 87.11%

Figure:

- `svm_closest_verified_result.png`

## Verification status

| Model | Submitted value | Status |
|---|---:|---|
| 1D-CNN | 88.27% ± 5.11% | Verified across 100 seeds |
| Random Forest | 82.8% | Reproduced by rounding 82.7957% |
| SVM | 87.3% | Exact value not reproduced; closest result 87.0968% |

The machine-readable details are also stored in:

- `final_results_summary.csv`
- `final_results_verification.json`

