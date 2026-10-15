# Follow-up airfare experiments — completed

Two separate training-only experiments were executed: direct CatBoost64 tuning and a Business-only XGBoost specialist. The previous best is preserved. Four new comparison CSVs are in submissions/. These R² values are local training-only validation results, not Kaggle scores.

| Candidate | Primary outer R² | CSV |
| --- | ---: | --- |
| Frozen previous best blend | 0.928594 | [submission_equal_blend_onehot64.csv](https://github.com/devshah207/Domestic_Flight_Prediction_Challenge/raw/refs/heads/main/submissions/submission_equal_blend_onehot64.csv) |
| Directly tuned CatBoost64 | 0.923316 | [submission_catboost64_direct_tuned.csv](submissions/submission_catboost64_direct_tuned.csv) |
| Blend with directly tuned CatBoost64 | 0.927932 | [submission_equal_blend_direct64.csv](submissions/submission_equal_blend_direct64.csv) |
| Business-augmented XGBoost | 0.926853 | [submission_business_augmented_xgb.csv](submissions/submission_business_augmented_xgb.csv) |
| Blend with Business-augmented XGBoost | 0.928633 | [submission_business_equal_blend64.csv](submissions/submission_business_equal_blend64.csv) |



Direct tuning did not improve the frozen CatBoost64 or its blend; its registered additional-seed compute gate was false. Those two CSVs are comparison candidates, not an evidence-based replacement for the previous best.

The Business-augmented XGBoost primary R² is 0.926853, versus 0.925416 for global XGBoost. Its fixed three-family blend is 0.928633, versus 0.928594 for the frozen blend. Inspect the paired folds: a pooled improvement does not mean every fold improved.

The prescribed second split seed reran global tuning within each partition. Global XGBoost R²: 0.925634; Business-augmented XGBoost: 0.926129. This is standalone robustness evidence on reused observations, not a fresh holdout or a secondary blend result.

The primary specialist trade-off is material: highest-priced-1% RMSE is 63,698.32, versus 62,818.57 for global XGBoost. Mean training–validation R² gaps are 0.038239 and 0.042506, respectively. A smaller average gap and better pooled R² do not imply improvement on the expensive tail.

## Exact fitting behavior

Original global XGBoost: 454 trees; original LightGBM: 764; original CatBoost64: 189; new directly tuned CatBoost64: 1003; new Business specialist: 461.

Business-augmented XGBoost predicts Economy exactly as the original global XGBoost. On Business rows only it averages global XGBoost and the specialist 50/50. Its three-model blend averages that augmented prediction with original LightGBM and original CatBoost64 in fixed thirds. The directly tuned CatBoost64 is used only in the separate direct-CatBoost blend. No weights were optimized.

The specialist trains only on 6,380 labeled Business rows, with the existing context features and a smaller bounded search. Its inner stopping metric evaluates the fixed halfblend, not outer labels. Pre-fit synthetic API checks resolved a documented XGBoost/scikit-learn estimator-tag compatibility issue without changing installed libraries.

Full configuration dictionaries, resolved defaults, feature lists, hashes and weights are in NEXT_MODEL_CONFIGURATIONS.json. Per-fold selected settings and every trial are retained under each experiment directory.

## Evidence and limitations

[Direct CatBoost64 findings](airfare_round2/analysis/FINDINGS.md) and [Business-specialist findings](airfare_business/analysis/FINDINGS.md) report all folds, measured train–validation gaps, cabin/price-tail/rare-flight errors, stopping records and conditional group-bootstrap intervals. Blend training R² is not fabricated by averaging component scores.

These folds and observations have informed prior development. The additional prescribed seed is a sensitivity check, not a pristine test. Bootstrap intervals omit retraining and accumulated selection uncertainty. Small differences can be noise; no absence of overfitting, Kaggle score improvement or rank is guaranteed. User-reported public results are provenance only and were not used as a tuning target.

All five required full-data model bundles and both fixed prediction rules were frozen together before one final competition-test read. Four CSVs passed hashes, exact IDs/order, 3,898 rows, id/price schema and finite nonnegative-value checks. No models changed after test access and no Kaggle uploads were performed.

## Run or inspect the code

Use Python3.12 and install airfare_fresh/requirements.txt. Supply the original CSVs separately. The scripts use project-relative output folders and enforce source/input identities. Existing completed outputs resume or skip completed fits, rather than constituting new independent evidence. For a fresh rerun, work in a copy and archive its generated phase output directories first; preserve the original completed results and frozen controls.

```bash
export OMP_NUM_THREADS=4 OMP_WAIT_POLICY=PASSIVE OPENBLAS_NUM_THREADS=1
export MPLCONFIGDIR=/tmp/airfare-mpl
python tools/check_new_inference.py
python airfare_round2/direct_cat64.py --stage all --train /path/to/train.csv
python airfare_round2/analyze.py --train /path/to/train.csv
python airfare_business/specialist.py --stage all --train /path/to/train.csv
python airfare_business/analyze.py --train /path/to/train.csv
```

Final inference uses airfare_round2/inference.py with --include-business, --train, a new --out directory and --freeze-only first; then the same arguments plus --test and --sample. The supplied submissions directory already has an exclusive inference marker and refuses repeated access or overwrites. Do not remove the marker. Never tune using test predictions.

The shared frozen source and original models remain in airfare_fresh/. Read each experiment PROTOCOL.md and REGISTRATION.json for prospective budgets, seeds and compute rules. No Business specialist was included in the earlier direct-tuning study’s budget; it is a separately registered experiment.
