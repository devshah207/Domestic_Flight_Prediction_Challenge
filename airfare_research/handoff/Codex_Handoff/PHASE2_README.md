# Phase 2 — Kaggle airfare regression (7 new submissions)

## Modeling protocol

- Official Kaggle scoring metric: **R²**, maximized.
- This iteration implements exactly seven **predeclared** model configurations.
- All feature mappings are learned on **training predictors only**. The prior training-only feature engineering is reused unchanged: airline-code fill, hierarchical duration imputation, route/class interactions, OHE for forests and native categorical representation for LightGBM. The mechanical `id` column is excluded from predictors.
- The supplied Kaggle **test file is not opened until all models have finished fitting**, and it is used only to produce competition submissions. No test-feature EDA or train/test distribution comparison is conducted.
- **No model-quality evaluation of any kind was executed** (no validation R², no train R², no holdout comparisons, no CV, no early-stopping search, no leaderboard feedback used to fit weights).
- Non-neural models only; every tree has a finite depth, and every ensemble has at most 550 trees. Tree count and maximum depth alone cannot guarantee generalization.
- Every output CSV has `id,price` columns exactly as the sample submission.

## Model configurations

| File | Trees | Maximum depth | Min. leaf samples | Key parameters |
|---|---:|---:|---:|---|
| `extra_trees_A_450.csv` | 450 | 14 | 5 | max_features=.75, min_samples_split=12, max_leaf_nodes=800, bootstrap=False |
| `extra_trees_B_550.csv` | 550 | 14 | 5 | max_features=.75, min_samples_split=12, max_leaf_nodes=800, bootstrap=False |
| `extra_trees_C_conservative_500.csv` | 500 | 12 | 8 | max_features=.70, min_samples_split=18, max_leaf_nodes=600, bootstrap=False |
| `random_forest_A_450.csv` | 450 | 13 | 5 | max_features=.80, min_samples_split=12, max_leaf_nodes=700, bootstrap=True, max_samples=.82 |
| `random_forest_B_550.csv` | 550 | 13 | 5 | max_features=.80, min_samples_split=12, max_leaf_nodes=700, bootstrap=True, max_samples=.82 |
| `lightgbm_A_450_lr055.csv` | 450 | 5 | 65 | 15 leaves, learning_rate=.055, L1=.4, L2=20, min_split_gain=.04, 80% row/column sample |
| `lightgbm_B_550_lr035.csv` | 550 | 5 | 65 | 15 leaves, learning_rate=.035, L1=.4, L2=20, min_split_gain=.04, 80% row/column sample |

All models use `random_state=42`. All LightGBM variants use `max_bin=127` and `subsample_freq=1`. `configurations.json` contains the exact full hyperparameter dictionaries; `manifest.csv` makes them easy to inspect.

## How to reproduce

Keep `generate_phase2_submissions.py` and `generate_submissions.py` in the same folder, then:

```bash
python generate_phase2_submissions.py --train 'train(1).csv' --test 'test(1).csv' --sample 'sample_submission(1).csv' --out 'phase2_regression_submissions'
```

Inputs are not included in the output archive; use your uploaded CSVs. The prior `generate_submissions.py` module is packaged only as a source of the **same** preprocessing functions; the Phase 2 script trains only the seven listed models.

## Comparing submitted results

Your prior public leaderboard scores, as reported in the conversation, were Extra Trees 0.95328 (320 trees), Random Forest 0.95221 (280 trees), and LightGBM 0.95232 (360 trees). These are *not* performance estimates for these seven submissions. Public leaderboard differences can be noisy and do not guarantee a private-leaderboard gain. Avoid repeatedly tuning to the public leaderboard.
