# Phase 4 – leakage-aware training-only validation and submissions

Training rows: **15,589**. Duplicate feature rows: **314** (excluded from validation overlap). Official competition metric: **R²** on original price scale.

## Dataset EDA (training only)
- Ticket price: median 10,183, mean 29,817, 99th percentile 185,837.
- Airline missing 3,089 rows; duration missing 3,095.
- Airline price class difference: economy median 7,587; business median 38,546.
- Observed duration vs ticket price Pearson correlation: 0.699; booking lead days vs ticket price correlation: -0.023.
- Learned duration imputations, rare-flight thresholds, categorical encodings and any feature frequency mappings are refitted within each CV training fold.

## Validation
Five folds: stratified by deciles of target and grouped by exact original predictor rows; each group appears in only one fold. No test features read before selection.
Early stopping on the validation fold: patience 75 boosting rounds, upper ceiling 1450. For final full-data refit, median best fold iterations (bounded 100 to 1450).

## Measured results (not Kaggle public scores)

| model               |   oof_r2 |   mean_fold_r2 |   sd_fold_r2 |   pooled_oof_rmse |   mean_train_r2 |   mean_train_minus_val_r2 |   median_best_rounds |   rounds_min |   rounds_max |
|:--------------------|---------:|---------------:|-------------:|------------------:|----------------:|--------------------------:|---------------------:|-------------:|-------------:|
| xgb_stronger_reg_d7 |  0.92917 |        0.92924 |      0.01271 |           10413   |         0.95879 |                   0.02955 |                  955 |          310 |         1399 |
| xgb_reference_d7    |  0.92916 |        0.92927 |      0.01257 |           10413.9 |         0.95637 |                   0.0271  |                  640 |          170 |          913 |
| xgb_shallower_d6    |  0.92868 |        0.92876 |      0.01293 |           10449.3 |         0.95587 |                   0.02712 |                  909 |          184 |         1418 |
| xgb_engineered_d7   |  0.9272  |        0.92734 |      0.01217 |           10556.9 |         0.9586  |                   0.03126 |                  634 |          177 |          818 |
| xgb_engineered_d6   |  0.92609 |        0.92621 |      0.01231 |           10637.3 |         0.95486 |                   0.02865 |                  704 |          177 |          917 |
| lgb_engineered_d5   |  0.92455 |        0.92473 |      0.01469 |           10747.1 |         0.95783 |                   0.0331  |                  522 |          313 |          846 |

## Frozen submissions

| model               |   trees |   depth | features   |   cv_oof_r2 | source                                                             |
|:--------------------|--------:|--------:|:-----------|------------:|:-------------------------------------------------------------------|
| xgb_stronger_reg_d7 |     955 |       7 | original   |    0.929171 | full-training refit with median early-stopping rounds from 5 folds |
| xgb_reference_d7    |     640 |       7 | original   |    0.929159 | full-training refit with median early-stopping rounds from 5 folds |

Optional predefined 50/50 blend achieved OOF R² 0.929440 and exceeded the leading individual model by at least 0.0002. No fitted blend weights.

## Additional validation cautions
- Each scored fold also provided the early-stopping signal for its model. Consequently, these out-of-fold scores are **not unbiased outer-holdout estimates**; selecting stopping rounds on the same fold can make them mildly optimistic. A nested train-only CV is needed for a less biased estimate.
- Stronger-regularized model mean train R² 0.95879 vs fold mean validation R² 0.92924 (gap approximately 0.02955): appreciable evidence of an in-sample/generalization gap, not proof of any private-leaderboard result.
- On the highest-price 1% of training observations, OOF predictions from the stronger-regularized model undershoot actual prices by about 51,273 on average; this tail is a significant remaining generalization weakness.
- The two leading OOF R² scores differ by only 0.000012, far smaller than fold-to-fold score differences. Prefer the simpler reference if private stability is paramount.

## Interpretation and limitations
- Five-fold validation informs overfitting risk but cannot guarantee hidden-test performance. Selection among multiple models still has some optimism.
- The folds block *identical rows*, not all shared flight IDs; unseen-flight generalization could be harder. Public/private differences remain possible.
- Engineered features are deterministic domain interactions and transformations; no target encoding or test-feature distributions are used.
- Training-vs-validation R² gaps and OOF errors should be interpreted jointly; leakage-safe transforms are refitted separately for each fold.
- Fixed seed, exact parameters, fold assignments, OOF predictions and training-only EDA are included for audit and reproducibility.

Runtime: 317.6 seconds.
