# Business specialist results

A bounded Business-only XGBoost is mixed 50/50 with the existing global XGBoost on Business rows only. Economy is exactly unchanged. One fixed thirds replacement blend is evaluated; no weights are fitted.

## Primary evaluation

| model | r2 | rmse | mae | worst_fold_r2 | mean_train_r2 | mean_gap |
| --- | --- | --- | --- | --- | --- | --- |
| global_xgb | 0.925416 | 10,685.503579 | 3,929.034092 | 0.916717 | 0.967967 | 0.042506 |
| business_augmented_xgb | 0.926853 | 10,582.096041 | 3,885.951530 | 0.916187 | 0.965137 | 0.038239 |
| frozen_equal_blend64 | 0.928594 | 10,455.402371 | 3,890.235098 | 0.919705 | — | — |
| business_equal_blend64 | 0.928633 | 10,452.510247 | 3,890.623292 | 0.917939 | — | — |

Paired candidate minus control changes:

| candidate | control | fold | delta_r2 | delta_rmse |
| --- | --- | --- | --- | --- |
| business_augmented_xgb | global_xgb | 0 | 0.002582 | -190.282690 |
| business_augmented_xgb | global_xgb | 1 | 0.003763 | -259.105555 |
| business_augmented_xgb | global_xgb | 2 | 0.001587 | -121.352753 |
| business_augmented_xgb | global_xgb | 3 | 0.001016 | -75.206641 |
| business_augmented_xgb | global_xgb | 4 | -0.001765 | 120.279463 |
| business_augmented_xgb | global_xgb | pooled | 0.001437 | -103.407538 |
| business_equal_blend64 | frozen_equal_blend64 | 0 | 0.000612 | -45.879686 |
| business_equal_blend64 | frozen_equal_blend64 | 1 | 0.000671 | -47.965833 |
| business_equal_blend64 | frozen_equal_blend64 | 2 | 0.000644 | -49.324981 |
| business_equal_blend64 | frozen_equal_blend64 | 3 | 0.000048 | -3.623734 |
| business_equal_blend64 | frozen_equal_blend64 | 4 | -0.001766 | 121.682160 |
| business_equal_blend64 | frozen_equal_blend64 | pooled | 0.000039 | -2.892125 |

High-price tail diagnostics:

| model | business_sse_share | top1pct_target_sse_share | top1pct_target_rmse | top1pct_target_bias |
| --- | --- | --- | --- | --- |
| global_xgb | 0.96748 | 0.34585 | 62,818.57206 | -48,041.49131 |
| business_augmented_xgb | 0.96684 | 0.36259 | 63,698.31755 | -50,267.06739 |
| frozen_equal_blend64 | 0.97349 | 0.36423 | 63,077.25579 | -49,500.50147 |
| business_equal_blend64 | 0.97347 | 0.36934 | 63,500.71329 | -50,242.36016 |

## Sensitivity evaluation

| model | r2 | rmse | mae | worst_fold_r2 | mean_train_r2 | mean_gap |
| --- | --- | --- | --- | --- | --- | --- |
| global_xgb | 0.925634 | 10,669.855301 | 3,990.249244 | 0.914094 | 0.962796 | 0.037118 |
| business_augmented_xgb | 0.926129 | 10,634.311616 | 3,930.319420 | 0.913435 | 0.963440 | 0.037247 |

Paired candidate minus control changes:

| candidate | control | fold | delta_r2 | delta_rmse |
| --- | --- | --- | --- | --- |
| business_augmented_xgb | global_xgb | 0 | 0.000397 | -28.320119 |
| business_augmented_xgb | global_xgb | 1 | -0.000584 | 41.423826 |
| business_augmented_xgb | global_xgb | 2 | 0.004165 | -301.326630 |
| business_augmented_xgb | global_xgb | 3 | -0.000744 | 58.420416 |
| business_augmented_xgb | global_xgb | 4 | -0.000659 | 44.564174 |
| business_augmented_xgb | global_xgb | pooled | 0.000495 | -35.543685 |

High-price tail diagnostics:

| model | business_sse_share | top1pct_target_sse_share | top1pct_target_rmse | top1pct_target_bias |
| --- | --- | --- | --- | --- |
| global_xgb | 0.95758 | 0.36140 | 64,121.18928 | -51,328.24524 |
| business_augmented_xgb | 0.95729 | 0.36842 | 64,524.69804 | -51,760.62358 |

## Validation limits

Global and specialist parameters and stopping counts are selected inside outer training partitions. The primary global fits exactly reconstruct the frozen XGBoost OOF predictions. The custom stopping metric evaluates the registered halfblend on Business inner validation rows; those scores remain tuning observations.

Training scores for augmented XGBoost are measured from actual composite predictions, not averaged component R². Training scores for the three-family blends are not measured and remain null. Inspect all folds, cabin support, rare flights, price tails and stopping dispersion; smaller gaps alone do not establish generalization.

These observations and primary folds were used in earlier development. The conditional second seed, when run, is a prescribed sensitivity analysis of global versus augmented XGBoost; no secondary three-family blend is fabricated. It is not an untouched holdout. Group-bootstrap intervals condition on current OOF predictions and omit retraining and accumulated selection uncertainty.

Completed trials: 192; research fits: 427. One synthetic API smoke fit is separate. No public scores or test features chose parameters, weights or postprocessing. No Kaggle upload or hidden-test improvement is claimed.
