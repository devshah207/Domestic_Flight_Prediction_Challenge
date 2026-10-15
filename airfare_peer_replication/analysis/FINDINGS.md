# Disclosed-peer settings: training-only results

This is an approximate replication. smooth=auto is interpreted as sklearn TargetEncoder, with group cross-fitting to protect labels. One-hot controls are also supplied. Feature lists, other settings and target scale are explicit assumptions; no claim to reproduce the peer score of 0.96080.

| Model | Outer OOF R² | Mean training R² | Mean gap |
| --- | ---: | ---: | ---: |
| control_previous_equal64 | 0.928594 | — | — |
| control_business_augmented | 0.926853 | — | — |
| xgb_smooth_auto | 0.922700 | 0.999877 | 0.077133 |
| xgb_ohe | 0.919984 | 0.992904 | 0.072884 |
| equal_smooth_auto | 0.917575 | 0.970490 | 0.052886 |
| equal_ohe | 0.907913 | 0.955816 | 0.047881 |
| cat | 0.901714 | 0.965079 | 0.063348 |
| rf_smooth_auto | 0.855224 | 0.855569 | 0.000308 |
| rf_ohe | 0.790188 | 0.791354 | 0.001138 |

No hyperparameters, rounds, encodings or mixture weights were selected from these scores. All five component models and both predeclared equal blends are supplied under the explicit replication request. Reused folds and past development limit confidence; these are not Kaggle scores. No extra holdout has been manufactured retrospectively.

RF: 200 trees, depth2, no bootstrap, .7 features per split. XGB: 964 trees, depth9, lr .054. CatBoost: 1000 trees, depth6, lr .139. All seed42. See ../MODEL_CONFIGURATIONS.json and ../PROTOCOL.md for every setting and assumption.

For target-encoded models, reported training scores use the cross-fitted training matrix actually passed to the regressor. Validation/inference use the separate full-training categorical mapping. No validation price is used in either mapping. Numeric medians and feature-only mappings fit on outer training.

Tail/cabin errors and residual correlations are retained in adjacent CSVs. Previous controls have verified identical IDs, labels, folds and groups. The earlier public best 0.95885 for classic gradient remains a user-reported public result, not a comparison on these validation scores.
