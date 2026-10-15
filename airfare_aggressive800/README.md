# One exploratory 800-tree Business candidate

**[Download the single new CSV](https://github.com/devshah207/Domestic_Flight_Prediction_Challenge/raw/refs/heads/main/submissions/submission_aggressive_business_xgb800.csv)**

This is the one higher-capacity candidate requested for an exploratory portfolio slot. No other new submission files were created. Prior submissions remain unchanged. The earlier Business-augmented model’s public R² of 0.95881 is user-reported; this new candidate has no known Kaggle score.

| Reused split seed | Control R² | Aggressive R² | Control mean gap | Aggressive mean gap | Paired fold wins |
| --- | ---: | ---: | ---: | ---: | ---: |
| primary | 0.926853 | 0.927473 | 0.038239 | 0.056928 | 2/5 |
| sensitivity | 0.926129 | 0.927483 | 0.037247 | 0.055091 | 5/5 |

Pooled validation R² improved on both prescribed split seeds, while the training–validation gaps increased substantially. The primary gain is concentrated in two folds; all five secondary folds improved. Higher capacity captured additional validation signal but also increased fitting flexibility. This does not establish a top-three public rank, private-score improvement or absence of overfitting.

## Exact prediction model

The new Business specialist has **800 trees**, max_depth=7, max_leaves=48, learning_rate=0.045, min_child_weight=8, reg_lambda=3, reg_alpha=0.05, gamma=0.01, subsample=0.95 and colsample_bytree=0.90. Hist/depthwise/native categorical processing, max_bin=192, max_cat_threshold=32, max_cat_to_onehot=4, random_state=42 and four threads remain fixed. These settings were registered together before any new scores, so differences cannot be attributed to tree count alone.

The global XGBoost remains the original 454-tree model. Economy uses that global prediction unchanged. Business uses exactly 50% global plus 50% new specialist. Thus the CSV represents one composite predictor with an 800-tree Business branch. Targets train as price/10000; each component is inverted and clipped below at zero before mixing.

One fixed profile, zero Optuna trials, zero stopping-count searches, 21 research fits: ten reconstructed global outer models, ten new Business outer models and one full Business model trained on 6,380 rows. Final global weights were reused without retraining.

## Validation evidence

Read [analysis/FINDINGS.md](analysis/FINDINGS.md) for original-price R²/RMSE/MAE, measured composite training gaps, paired folds, high-price errors and conditional group-bootstrap intervals. Subgroup/worst-error CSVs and plots are under analysis/. Both primary and secondary splits reuse observations and folds from development; neither is a pristine holdout. Bootstrap intervals omit retraining and accumulated selection uncertainty.

Global controls are reconstructed from partition-local settings and verified against saved OOF predictions. Stored raw/target identities and group separation are checked before fitting. New feature mappings learn only from each outer Business-training subset. Outer labels never choose this fixed profile, tree count or weights. No competition test features or predictions informed development.

## Files and reproduction

model_configurations.json contains both exact fitted model dictionaries, resolved booster settings, versions, hashes, feature lists and the fixed prediction policy. PROTOCOL.md and REGISTRATION.json record the prospective recipe and budget. experiment_ledger.json records actual completed results and inference audit. ARTIFACT_MANIFEST.json covers all candidate files except itself.

Clone the complete GitHub repository to run the code: source dependencies and frozen controls live in sibling airfare_business/, airfare_round2/ and airfare_fresh/ folders. Install airfare_fresh/requirements.txt with Python 3.12 and supply the original CSVs separately. The candidate archive packages its source, model, diagnostics and one submission; it requires those sibling dependencies.

```bash
export OMP_NUM_THREADS=4 OMP_WAIT_POLICY=PASSIVE OPENBLAS_NUM_THREADS=1
export MPLCONFIGDIR=/tmp/airfare-mpl
python airfare_aggressive800/check_inference.py
python airfare_aggressive800/experiment.py --stage all --train /path/to/train.csv
```

Completed fitting scopes resume from verified existing artifacts. For a fresh rerun work in a copy, archive the generated primary/, sensitivity/, full_training/, analysis/ and submissions/ folders there, and preserve the original results. The source/input hash guards require unchanged inputs and source contents.

Final inference uses inference.py --train PATH --out NEW_DIRECTORY --freeze-only, then the same arguments plus --test PATH --sample PATH. Do not reuse the supplied submissions/ directory: its exclusive inference marker prevents repeated access or overwrites. Models were frozen before this phase’s one test-content read; no model changes followed.

The single CSV passed exact sample-ID ordering, id/price schema, 3,898 rows, unique IDs and finite nonnegative-value checks. No Kaggle submission was uploaded. Keep the earlier candidate files; the competition’s actual final-selection allowance remains unverified.
