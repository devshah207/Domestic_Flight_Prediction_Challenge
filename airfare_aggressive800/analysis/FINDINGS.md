# Fixed 800-tree exploratory Business candidate

This is one predeclared aggressive profile, not an optimized tree-count search. The user explicitly requested an exploratory portfolio slot with weaker validation acceptable. It changes several capacity/regularization parameters together, so tree-count causality cannot be isolated.

## Primary reused-fold evaluation

| model | r2 | rmse | mae | mean_train_r2 | mean_gap | worst_fold_r2 |
| --- | --- | --- | --- | --- | --- | --- |
| current_business_augmented | 0.926853 | 10,582.096041 | 3,885.951530 | 0.965137 | 0.038239 | 0.916187 |
| aggressive800 | 0.927473 | 10,537.112583 | 3,753.347827 | 0.984444 | 0.056928 | 0.918401 |

| model | fold | r2 | train_r2 | gap |
| --- | --- | --- | --- | --- |
| current_business_augmented | 0 | 0.930757 | 0.962959 | 0.032202 |
| current_business_augmented | 1 | 0.920481 | 0.975764 | 0.055283 |
| current_business_augmented | 2 | 0.935958 | 0.967761 | 0.031802 |
| current_business_augmented | 3 | 0.931105 | 0.958864 | 0.027760 |
| current_business_augmented | 4 | 0.916187 | 0.960337 | 0.044150 |
| aggressive800 | 0 | 0.928149 | 0.986621 | 0.058472 |
| aggressive800 | 1 | 0.918401 | 0.987223 | 0.068822 |
| aggressive800 | 2 | 0.937537 | 0.982523 | 0.044986 |
| aggressive800 | 3 | 0.930797 | 0.984087 | 0.053289 |
| aggressive800 | 4 | 0.922696 | 0.981768 | 0.059072 |

| model | top1pct_target_sse_share | top1pct_target_rmse | top1pct_target_bias | business_sse_share |
| --- | --- | --- | --- | --- |
| current_business_augmented | 0.36259 | 63,698.31755 | -50,267.06739 | 0.96684 |
| aggressive800 | 0.35089 | 62,395.81904 | -46,875.33552 | 0.96655 |

## Sensitivity reused-fold evaluation

| model | r2 | rmse | mae | mean_train_r2 | mean_gap | worst_fold_r2 |
| --- | --- | --- | --- | --- | --- | --- |
| current_business_augmented | 0.926129 | 10,634.311616 | 3,930.319420 | 0.963440 | 0.037247 | 0.913435 |
| aggressive800 | 0.927483 | 10,536.389251 | 3,805.477729 | 0.982641 | 0.055091 | 0.914501 |

| model | fold | r2 | train_r2 | gap |
| --- | --- | --- | --- | --- |
| current_business_augmented | 0 | 0.925447 | 0.962087 | 0.036640 |
| current_business_augmented | 1 | 0.924782 | 0.959702 | 0.034920 |
| current_business_augmented | 2 | 0.930354 | 0.971020 | 0.040666 |
| current_business_augmented | 3 | 0.936948 | 0.965281 | 0.028333 |
| current_business_augmented | 4 | 0.913435 | 0.959109 | 0.045674 |
| aggressive800 | 0 | 0.927656 | 0.984023 | 0.056367 |
| aggressive800 | 1 | 0.926779 | 0.980181 | 0.053402 |
| aggressive800 | 2 | 0.931002 | 0.985174 | 0.054172 |
| aggressive800 | 3 | 0.937814 | 0.982979 | 0.045165 |
| aggressive800 | 4 | 0.914501 | 0.980849 | 0.066348 |

| model | top1pct_target_sse_share | top1pct_target_rmse | top1pct_target_bias | business_sse_share |
| --- | --- | --- | --- | --- |
| current_business_augmented | 0.36842 | 64,524.69804 | -51,760.62358 | 0.95729 |
| aggressive800 | 0.35798 | 63,018.56279 | -49,558.41124 | 0.95650 |

## Interpretation

Original-price outer validation never chooses this profile, its fixed800 rounds or the mixture. All prior model-selection history and these folds/observations have already informed development; no fresh holdout is claimed. Conditional group-bootstrap intervals omit retraining and accumulated selection uncertainty. Smaller gaps and additional trees do not prove generalization.

This candidate can be delivered despite weaker validation under the user’s explicit exploratory authorization. It is not represented as a stronger conservative finalist, and no public/private score or top-three rank is promised. Preserve earlier submissions. The competition’s final-selection limit remains unverified.

The global branch exactly reconstructs previous per-partition controls. Economy predictions remain unchanged. Every feature mapping is fitted only on the corresponding global or Business fitting rows; no test features/predictions inform model development. Budget21fits: ten reconstructed global outer fits, ten specialist outer fits, one full Business fit. No Optuna trials or early-stopping fits.

Final file uses the unchanged454-tree full global XGBoost and the new800-tree Business specialist in fixed halves on Business; Economy uses global only. Exact configurations, sources, versions and inference audit are delivered separately.
