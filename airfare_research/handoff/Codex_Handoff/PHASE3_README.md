# Phase 3 — Controlled gradient boosting candidates for airline ticket-price regression

## Why this phase exists

The user's latest reported public leaderboard results were LightGBM 450 trees, 0.055 learning rate **0.95354**; LightGBM 550 trees, 0.035 learning rate **0.95253**; Extra Trees 450/500-conservative/550 trees **0.95298/0.94965/0.95287**; Random Forest 450/550 trees **0.95190/0.95194**. Earlier Extra Trees 320-tree model scored **0.95328**. These are user-reported scores, *not local evaluations*.

Boosting was prioritized based on the improving LightGBM results and reports that leading teams used boosting. This does not prove a boosted model will generalize best or that 800 trees are optimal.

## New submission files and design hypotheses

| File | Family | Trees | Depth | Learning rate | Hypothesis |
|---|---|---:|---:|---:|---|
| `lgb_650_d5_lr045.csv` | LightGBM | 650 | 5 | 0.045 | Keep original shallow leaf structure and extend boosting with smaller steps |
| `lgb_800_d5_lr035.csv` | LightGBM | 800 | 5 | 0.035 | More boosting rounds with small updates and unchanged leaf structure |
| `lgb_750_d6_leaves23.csv` | LightGBM | 750 | 6 | 0.035 | Try slightly richer interactions while strengthening L1/L2 and minimum leaf size |
| `xgb_750_d5_lr045.csv` | XGBoost | 750 | 5 | 0.045 | Moderate-depth alternative with row/column sampling and L2 regularization |
| `xgb_850_d7_regularized.csv` | XGBoost | 850 | 7 | 0.030 | Test more expressive interactions using stronger regularization and smaller learning rate |
| `cat_780_d6_lr040.csv` | CatBoost | 780 | 6 | 0.040 | Native categorical treatment of many flight IDs, regularized ordered encoding |
| `xgb_800_d6_log1p.csv` | XGBoost | 800 | 6 | 0.040 | Isolate a log1p(target) hypothesis mentioned by another team; inverse transform at inference |

All `max_depth` / CatBoost `depth` parameters are positive finite integers. LightGBM has `num_leaves <= 23`, `min_child_samples >= 65`, subsampling and strong L2. XGBoost has `min_child_weight >= 6`, row/column subsampling, finite max depth and max leaves, gamma and L1/L2. CatBoost has finite depth, L2, randomized categorical splits and Bernoulli subsampling. All models use `random_state=42` (CatBoost `random_seed=42`). Each model has at most **850** boosting rounds.

The full exact hyperparameters appear in `configs.json`, `manifest.csv` and `generate_phase3_boosting.py`.

## Preprocessing

This phase reuses the existing **training-only**, target-independent feature builder in `generate_submissions.py` *without modification*. It learns observed airline prefix mappings, hierarchical duration medians and flight frequency levels exclusively from training features. Flight numbers, routes, route/class/stops and other categorical interactions are maintained. LightGBM uses categorical data, CatBoost uses full flight IDs and native categoricals, XGBoost uses the prior one-hot representation. The mechanical `id` is excluded from modeling and used solely for submission alignment.

Training data price is divided by 10,000 for original-price models (only to stabilize magnitude-sensitive penalty settings). The log-target experiment instead uses `log1p(price)` directly, and `expm1` maps its predictions back to monetary units. **Because the contest scores R² on original-scale prices, a log-target model can systematically underpredict expensive fares; its inclusion is exploratory, not a recommendation to replace other models.**

## Why not early stopping here?

The other team's stated approach of training up to 1,800 rounds with `early_stopping_rounds=70` uses 1,800 **as a ceiling**, not necessarily as final tree count. With correctly isolated validation data, early stopping can identify promising tree counts. For k-fold CV one can collect each fold's best iteration and fit a final model on all training rows using a principled summary (such as the median). Their messages disagree on `min(100, median(...))` versus `max(100, median(...))` — this must be clarified. Their 5-fold OOF R²=0.81648, RMSE=95.0741 also appear on an unclear scale, not directly comparable to the public R² near 0.96.

**The user explicitly prohibited us from testing the models ourselves.** Therefore, no fold selection, early stopping, holdout score, OOF score, fitted-model training score, or other quality evaluation was performed. The training rounds are predeclared fixed hypotheses and are NOT optimal values inferred from validation. The Kaggle public leaderboard is a 60% subset per the user's screenshot, so differences can be noisy and susceptible to leaderboard overfitting. Retain the original LightGBM 450 tree submission and strong Extra Trees 320 submission as baselines.

## File validation

- Exactly 7 new CSV files, each with the sample submission's `id,price` schema and 3,898 rows.
- Predictions finite and nonnegative, IDs aligned with sample submission.
- Test input first opened **after all seven models completed fitting**, strictly for inference and validity checks, never for test-feature EDA, selection, tuning or comparison.
- No blended submissions were added because their weights would be unvalidated.

## Reproduction

Keep the scripts `generate_phase3_boosting.py` and `generate_submissions.py` in the same folder and install `numpy pandas scipy scikit-learn lightgbm xgboost catboost`.

```bash
python generate_phase3_boosting.py --train 'train(1).csv' --test 'test(1).csv' --sample 'sample_submission(1).csv' --out phase3_boosting_submissions
```

Keep `configs.json` / `manifest.csv` for an exact record, and avoid choosing a model based on tiny public-leaderboard differences. Local cross-validation, when permitted by the user, is the better approach to estimate generalization and select boosting rounds.
