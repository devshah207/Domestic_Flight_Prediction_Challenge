# Direct CatBoost64 tuning results

This phase retunes the entire bounded CatBoost search with threshold 64. No new features, weight search, loss change or competition test data informed development. All comparisons are developmental on previously used training observations. The additional split seed, when run, is a registered robustness check, not a pristine confirmation holdout.

## Primary evaluation

| model | r2 | rmse | mae | worst_fold_r2 | mean_train_r2 | mean_gap |
| --- | --- | --- | --- | --- | --- | --- |
| cat64_direct | 0.923316 | 10,834.931491 | 4,302.278206 | 0.915922 | 0.951511 | 0.028162 |
| xgb | 0.925416 | 10,685.503579 | 3,929.034092 | 0.916717 | 0.967967 | 0.042506 |
| lgb | 0.925142 | 10,705.114322 | 3,920.029139 | 0.913261 | 0.965731 | 0.040547 |
| cat_onehot64 | 0.925056 | 10,711.244563 | 4,249.047923 | 0.913653 | 0.951453 | 0.026362 |
| frozen_equal_blend64 | 0.928594 | 10,455.402371 | 3,890.235098 | 0.919705 | — | — |
| direct_equal_blend64 | 0.927932 | 10,503.701587 | 3,909.855339 | 0.920088 | — | — |

Paired changes use candidate minus control. Negative RMSE change is better.

| seed | candidate | control | fold | delta_r2 | delta_rmse |
| --- | --- | --- | --- | --- | --- |
| primary | cat64_direct | cat_onehot64 | 0 | 0.000000 | 0.000000 |
| primary | cat64_direct | cat_onehot64 | 1 | -0.006721 | 464.769331 |
| primary | cat64_direct | cat_onehot64 | 2 | -0.002905 | 212.413250 |
| primary | cat64_direct | cat_onehot64 | 3 | -0.002234 | 164.230469 |
| primary | cat64_direct | cat_onehot64 | 4 | 0.003147 | -212.154524 |
| primary | cat64_direct | cat_onehot64 | pooled | -0.001741 | 123.686928 |
| primary | direct_equal_blend64 | frozen_equal_blend64 | 0 | 0.000000 | 0.000000 |
| primary | direct_equal_blend64 | frozen_equal_blend64 | 1 | -0.002926 | 206.679254 |
| primary | direct_equal_blend64 | frozen_equal_blend64 | 2 | -0.000464 | 35.394202 |
| primary | direct_equal_blend64 | frozen_equal_blend64 | 3 | -0.000284 | 21.352143 |
| primary | direct_equal_blend64 | frozen_equal_blend64 | 4 | 0.000383 | -26.531611 |
| primary | direct_equal_blend64 | frozen_equal_blend64 | pooled | -0.000661 | 48.299216 |

Tail diagnostics use original-price errors; negative bias means underprediction.

| seed | model | business_sse_share | top1pct_target_sse_share | top1pct_target_bias | top1pct_target_rmse | worst1pct_error_sse_share |
| --- | --- | --- | --- | --- | --- | --- |
| primary | cat64_direct | 0.97224 | 0.34758 | -50,716.38055 | 63,855.65379 | 0.58726 |
| primary | xgb | 0.96748 | 0.34585 | -48,041.49131 | 62,818.57206 | 0.61218 |
| primary | lgb | 0.96721 | 0.36684 | -50,377.28613 | 64,814.95850 | 0.61497 |
| primary | cat_onehot64 | 0.97272 | 0.35312 | -50,082.72697 | 63,628.03792 | 0.58431 |
| primary | frozen_equal_blend64 | 0.97349 | 0.36423 | -49,500.50147 | 63,077.25579 | 0.60963 |
| primary | direct_equal_blend64 | 0.97307 | 0.36241 | -49,711.71933 | 63,210.21985 | 0.61252 |

## Interpretation and limitations

The primary fixed replacement blend uses exactly one third each of the frozen original XGBoost, frozen original LightGBM and the newly tuned CatBoost64. No weights were optimized. The sensitivity evaluation reruns original CatBoost selection, its threshold-64 hybrid and direct threshold-64 tuning on a different prescribed set of partitions. It does not evaluate a secondary three-family blend.

Outer labels never choose per-fold parameters or stopping rounds. However, the research direction follows prior training diagnostics and reported public outcomes. Reusing folds and selecting a preferred candidate across studies introduces selection optimism. Bootstrap intervals condition on current OOF predictions, not retraining, search, folds or accumulated model-selection uncertainty.

A small training gap does not prove generalization. Business/Economy errors, high-price tails, rare-flight support, stopping dispersion and worst-fold behavior must be considered together. Full-data configurations are not independently tested by the outer scores. Do not compare these local R² values numerically with Kaggle public R².

Completed Optuna trials: 72. Research fits: 150. One synthetic API smoke fit is separate. See PROTOCOL.md for the prospective budget and conditional split-seed compute rule.

The frozen earlier best is retained regardless of this phase’s result. No Kaggle uploads are performed by this research.
