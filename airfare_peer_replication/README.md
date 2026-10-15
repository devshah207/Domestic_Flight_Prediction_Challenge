# Replication of the disclosed RF + XGBoost + CatBoost ensemble

Two equal-weight blends reproduce the disclosed model settings with explicit preprocessing assumptions. The `smooth="auto"` clue is interpreted as scikit-learn TargetEncoder; its exact use by the peer is unconfirmed. This does not claim an exact full-pipeline replication or reproduction of their reported 0.96080.

| Model / file | Trees | Max depth | Learning rate | Training-only outer R² | Mean train–validation gap |
| --- | ---: | ---: | ---: | ---: | ---: |
| [rf_ohe](submissions/submission_peer_rf_ohe.csv) | 200 | 2 | — | 0.790188 | 0.001138 |
| [xgb_ohe](submissions/submission_peer_xgb_ohe.csv) | 964 | 9 | 0.054 | 0.919984 | 0.072884 |
| [rf_smooth_auto](submissions/submission_peer_rf_smooth_auto.csv) | 200 | 2 | — | 0.855224 | 0.000308 |
| [xgb_smooth_auto](submissions/submission_peer_xgb_smooth_auto.csv) | 964 | 9 | 0.054 | 0.922700 | 0.077133 |
| [cat](submissions/submission_peer_cat.csv) | 1000 | 6 | 0.139 | 0.901714 | 0.063348 |
| [equal_ohe](submissions/submission_peer_equal_ohe.csv) | 200 + 964 + 1000 | 2 / 9 / 6 | — | 0.907913 | 0.047881 |
| [equal_smooth_auto](submissions/submission_peer_equal_smooth_auto.csv) | 200 + 964 + 1000 | 2 / 9 / 6 | — | 0.917575 | 0.052886 |

RF uses all fitting rows in every tree (`bootstrap=False`), 70% feature sampling per split and seed42. XGB uses 964 trees, depth9, lr0.054 and seed42. CatBoost uses 1000 trees, depth6, lr0.139 and seed42. Both blends are fixed equal arithmetic thirds. Unknown peer parameters are documented explicitly in [the protocol](PROTOCOL.md); every constructor, resolved library setting, actual structure and feature schema is in [MODEL_CONFIGURATIONS.json](MODEL_CONFIGURATIONS.json).

`equal_smooth_auto` uses smoothed categorical target encodings for RF and XGB and native categorical CatBoost. Each encoding training row is produced by a mapping that excludes its entire itinerary group. A separate map fit only on the corresponding full fitting partition encodes held-out validation/test rows; unseen categories fall back to the fitting target mean. `equal_ohe` changes RF and XGB to training-fitted sparse one-hot encoding and uses the identical CatBoost. Both use the existing compact context feature set as an assumption. No ID or validation prices enter the features.

Five training-only grouped/stratified folds evaluated fixed configurations. There was no Optuna search, early stopping, weight optimization or selection of a winning encoding from these scores. Thirty regressor fits completed (25 outer +5 full), plus72 research TargetEncoder mapping fits. Twelve additional encoder fits were synthetic leakage checks; no smoke regressors were fitted. These folds and observations have informed earlier development and are not a pristine holdout. Outer R² is not a public/private Kaggle score.

The shallow RF may materially weaken an equal blend; all components and both registered blends are delivered to make this fixed replication reviewable. Their gaps, tail errors and cabin errors are measured rather than presented as guarantees. The earlier user-reported public best remains 0.95885 from classic gradient boosting; no new public score is known for this phase.

## Evidence

- [Training-only findings and historical controls](analysis/FINDINGS.md)
- [Fold metrics](analysis/fold_metrics.csv)
- [Tail errors](analysis/tail_metrics.csv)
- [Cabin errors](analysis/cabin_metrics.csv)
- [Residual correlations](analysis/residual_correlations.csv)
- [Exact configurations](MODEL_CONFIGURATIONS.json)
- [Fit ledger](EXPERIMENT_LEDGER.json)
- [Synthetic leakage and inference checks](synthetic_checks.json)
- [Frozen source/model/policy manifest](submissions/frozen_manifest.json)
- [Completed inference checks and CSV hashes](submissions/inference_audit.json)

## Reproduce

Use Python3.12 and the pins in ../airfare_fresh/requirements.txt. Start from a fresh output copy without generated registration/results/analysis/submissions/audits; preserve all source files and sibling source dependencies. The delivered experiment is already frozen: do not remove its one-use inference marker or overwrite it. Provide your own original competition CSVs. From the repository root:
```bash
export OMP_NUM_THREADS=4 OMP_WAIT_POLICY=PASSIVE OPENBLAS_NUM_THREADS=1
export MPLCONFIGDIR=/tmp/airfare-mpl
python airfare_peer_replication/check_inference.py
python airfare_peer_replication/run.py --stage register --train /path/train.csv
python airfare_peer_replication/run.py --stage train --train /path/train.csv
python airfare_peer_replication/inference.py --freeze-only --train /path/train.csv
python airfare_peer_replication/inference.py --train /path/train.csv --test /path/test.csv --sample /path/sample_submission.csv
```

Both encodings, all five full models, source/version hashes, schemas, price scaling and blend weights were frozen together before this phase read test content once. Seven files passed 3898 rows, exact sample ID order, id/price columns and finite nonnegative prices. No model fitting or revision followed test access. The top-level submissions folder contains byte-identical copies. No Kaggle upload was performed. Only load serialized models from a trusted source.

downloads/Airfare_Peer_Replication_Submissions.zip includes seven CSVs, code, exact configurations and diagnostics. downloads/Airfare_Peer_Replication_Complete.zip also includes the fitted model bundles and cross-fitted training prediction arrays. Archives exclude raw competition datasets and the virtual environment.
