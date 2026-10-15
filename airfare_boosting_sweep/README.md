# Broad high-round Optuna boosting study

Eight global boosting procedures and two predeclared equal-weight blends, executed in response to the request for multiple aggressive candidates. Every successful candidate is supplied, including weaker comparisons. These are measured training-only results; no new public/private Kaggle R² or rank is claimed.

| Candidate | Actual trees / rounds | Outer CV R² | Mean train–validation gap | CSV |
| --- | ---: | ---: | ---: | --- |
| equal_dart3 | 1400 + 1400 + 1400 | 0.927333 | 0.041439 | [submission_highround_optuna_equal_dart3.csv](submissions/submission_highround_optuna_equal_dart3.csv) |
| equal_main3 | 1400 + 1400 + 1400 | 0.927105 | 0.042712 | [submission_highround_optuna_equal_main3.csv](submissions/submission_highround_optuna_equal_main3.csv) |
| xgb | 1400 | 0.924665 | 0.047242 | [submission_highround_optuna_xgb.csv](submissions/submission_highround_optuna_xgb.csv) |
| lgb_gbdt | 1400 | 0.923440 | 0.044385 | [submission_highround_optuna_lgb_gbdt.csv](submissions/submission_highround_optuna_lgb_gbdt.csv) |
| hist_gradient | 800 | 0.923166 | 0.043152 | [submission_highround_optuna_hist_gradient.csv](submissions/submission_highround_optuna_hist_gradient.csv) |
| cat_symmetric | 1400 | 0.922904 | 0.039935 | [submission_highround_optuna_cat_symmetric.csv](submissions/submission_highround_optuna_cat_symmetric.csv) |
| classic_gradient | 1100 | 0.922722 | 0.035480 | [submission_highround_optuna_classic_gradient.csv](submissions/submission_highround_optuna_classic_gradient.csv) |
| cat_leafwise | 1400 | 0.922688 | 0.032297 | [submission_highround_optuna_cat_leafwise.csv](submissions/submission_highround_optuna_cat_leafwise.csv) |
| lgb_dart | 1400 | 0.922404 | 0.040262 | [submission_highround_optuna_lgb_dart.csv](submissions/submission_highround_optuna_lgb_dart.csv) |
| adaboost | 1100 | 0.914292 | 0.029444 | [submission_highround_optuna_adaboost.csv](submissions/submission_highround_optuna_adaboost.csv) |

Same-fold historical controls: control_global_xgb: R²=0.925416, control_business_augmented: R²=0.926853, control_previous_equal64: R²=0.928594.

[The saved-tree audit](TREE_COMPLEXITY_AUDIT.json) checks actual fitted counts and leaf bounds. Its structural inspection performs no model fitting or test-data access.

Every Optuna trial requested 800, 1100 or 1400 rounds. Actual fitted counts are shown above; AdaBoost or a library without further useful splits can terminate before its requested budget. Symmetric CatBoost’s leaf cap is implicit in its finite depth; other tree depths/leaves/support/regularization are explicitly bounded where supported.

## Evidence and exact configurations

- [Full measured findings](analysis/FINDINGS.md)
- [Exact final model dictionaries, resolved parameters and blend weights](MODEL_CONFIGURATIONS.json)
- [All 336 trial records](analysis/all_trials.csv)
- [Paired outer fold metrics](analysis/fold_metrics.csv)
- [Tail errors](analysis/tail_metrics.csv)
- [Cabin, prefix and rare-flight diagnostics](analysis/subgroup_metrics.csv)
- [Grouped-bootstrap uncertainty](analysis/bootstrap_intervals.json)
- [Partition/source identity audit](VALIDATION_AUDIT.json)
- [Fit accounting](EXPERIMENT_LEDGER.json)
- [Frozen inference manifest](submissions/frozen_manifest.json)
- [Submission schema/ID/checksum audit](submissions/inference_audit.json)

A parallel-scheduling collision required one identical-configuration LightGBM trial recovery. Actual accounting is 336 distinct Optuna trials, 384 canonical research fits and one duplicate fit attempt (385 total), plus 15 API smoke fits. See SCHEDULING_RECOVERY.json and EXPERIMENT_LEDGER.json. No search-space expansion or outer/test-based choice was introduced. Saved parameters preserve the exact asynchronous TPE history for the affected scope.

## Validation meaning

Five outer grouped/stratified folds evaluate each whole tuning procedure. Inside each outer fitting partition, a separate inner 80/20 group split ranks the fixed-budget Optuna trials. Outer validation labels never choose parameters or counts. Full-data settings come from a fresh same-budget inner tuning run, then a refit on all 15589 labeled rows. All feature mappings fit only on the corresponding fitting partition. The unchanged prior global and Business-augmented controls use identical outer rows/folds.

These folds and training observations have already informed earlier development. There is no pristine new holdout. Single-inner-split tuning is an efficiency compromise, and comparing many families adds selection uncertainty. Confidence intervals condition on saved predictions. A high count or high training R² does not ensure a public gain. The previous user-reported best 0.95881 remains available; this study neither uses public scores to tune settings nor verifies final-selection rules.

## Reproduce

From a fresh checkout/copy without this phase’s generated outputs, create a Python3.12 environment and install the pinned dependencies in ../airfare_fresh/requirements.txt. Set OMP_NUM_THREADS=4, OMP_WAIT_POLICY=PASSIVE, OPENBLAS_NUM_THREADS=1 and MPLCONFIGDIR to a writable directory. All commands below run from the repository root. Supply your own original competition data paths.

```bash
python airfare_boosting_sweep/study.py --stage smoke --train /path/train.csv
python airfare_boosting_sweep/study.py --stage register --train /path/train.csv
python airfare_boosting_sweep/study.py --stage run --train /path/train.csv
python airfare_boosting_sweep/analyze.py --train /path/train.csv
python airfare_boosting_sweep/check_inference.py
python airfare_boosting_sweep/inference.py --freeze-only --train /path/train.csv
python airfare_boosting_sweep/inference.py --train /path/train.csv --test /path/test.csv --sample /path/sample_submission.csv
```

The supplied generated experiment is frozen and inference has already completed. Do not delete its markers or overwrite artifacts to rerun it. Reproduction needs a fresh output copy without generated registration/results/analysis/audits/submissions; keep all source files and sibling dependencies. The original training/test/sample SHA256 checks protect input identity. Runtime parallel_*.py and recover_scheduling.py helpers preserve this machine’s scheduling history and process IDs; use the documented sequential study.py workflow on another machine. Serialized joblib files should be loaded only from a trusted source.

## Downloads

The top-level submissions folder contains byte-identical copies. downloads/Airfare_HighRound_Submissions_and_Configs.zip contains the ten CSVs, exact configurations, code and diagnostics. downloads/Airfare_HighRound_Complete.zip additionally contains the eight fitted model bundles and trial databases. Sibling airfare_fresh source/dependencies are included in both archives. Raw competition datasets and the virtual environment are excluded.
