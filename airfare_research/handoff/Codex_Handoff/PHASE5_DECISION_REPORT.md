# Phase 5 — Independent validation audit

**Decision: Do not replace the existing 850-tree XGBoost benchmark. No new Kaggle submission justified.**

## Scope and control

This analysis uses *only* `train(1).csv` to evaluate and select models. The Kaggle test dataset was **not opened** because no new candidate passed the predeclared validation gate. The original XGBoost benchmark is the exact Phase 3 850-tree depth-7 model (learning rate 0.030, min child weight 12, max leaves 48, row subsample .80, column subsample .85, L2 15, L1 .60, gamma .08, max bin 192, tree method hist, fixed seed 42). Its existing public leaderboard score, **0.95797**, was reported by the user; it is not a locally measured test score.

We isolated the hypothesis that business and economy prices have different data-generating relationships and should use separate regressors, motivated by earlier observed training-only Phase 4 prediction errors.

All models used the same preprocessor learned within their respective training folds. Exact duplicate feature rows were grouped so they could not appear in both the training and evaluation portion of a split.

## Protocol

1. Create one 20% approximately independent holdout using grouped, stratified splitting; do not use the holdout for feature engineering or model choice.
2. On the other ~80% of training data, compare three **predeclared** procedures under three grouped-stratified folds:
   - **Global 850**: exact original single XGBoost architecture, 850 rounds.
   - **Split same 850**: separate economy/business regressors using the identical global hyperparameters, 850 rounds each; preprocessing shared from the train portion.
   - **Split tailored + early stop**: business `max_depth=7`, `min_child_weight=7`, `reg_lambda=12`, maximum 1,050 rounds; economy `max_depth=5`, `min_child_weight=8`, `reg_lambda=16`, maximum 650 rounds. Both use 0.03 learning rate and other original settings. Early stopping (patience=70) is determined with a **further inner fold** drawn exclusively from the corresponding training partition, then a model with that number of trees is fit on all of the training partition. The CV evaluation fold does **not** control stopping.
3. Before seeing the independent holdout, advance only an improvement of at least 0.003 development OOF R², positive on >=2 of 3 folds, with positive lower 95% grouped paired bootstrap confidence bound.
4. Compare the frozen development winner exactly once on the untouched holdout against the unchanged global baseline; accept only if it improves R² by at least 0.002 and has a positive lower 95% grouped paired bootstrap confidence bound. This is a conservative safeguard against spurious gains and overfitting validation.

## Development 3-fold results

| Model | OOF R² | RMSE (price units) | MAE | Notes |
|---|---:|---:|---:|---|
| Global XGBoost 850 | 0.917619 | 11,256.96 | 4,101.34 | Benchmark |
| Separate class, same 850 settings | 0.917297 | 11,278.96 | 3,975.97 | No R² gain |
| Separate class, tailored and inner early-stop | **0.920639** | **11,048.73** | 4,010.92 | Provisional winner |

For the tailored model, differences from the global benchmark across the three development folds were **+0.004382**, **+0.004463**, and **+0.000219** R². The predeclared development selection gate passed. The paired grouped bootstrap 95% interval for the pooled R² improvement was **[+0.000860, +0.005359]**. These are not Kaggle scores.

Early-stop counts for business were **218, 391, 307**; for economy **580, 650, 649** across development folds. This variability indicates a universal number of trees is not necessarily optimal, but these stopping decisions did not guarantee generalization.

## Independent ~20% holdout — never used for fitting/tuning these submissions

| Metric | Global 850 benchmark | Tailored separate-class candidate |
|---|---:|---:|
| R² | **0.937351** | 0.934767 |
| RMSE | **9,699.00** | 9,897.02 |
| MAE | **3,726.21** | 3,814.48 |
| Business-class RMSE | **14,838.37** | 15,230.95 |
| Economy-class RMSE | 2,428.80 | **2,058.85** |
| Top 1% of prices, average prediction error | **−45,112.87** | −52,535.26 |

The holdout R² change was **−0.002584**, in the wrong direction. The paired grouped bootstrap 95% interval for this change was **[−0.008839, +0.003874]**, containing zero. **The model is not independently supported as better.** We rejected it before loading Kaggle test data.

## What we learned

- The very high contribution of business-class observations to total squared error is real in training-only diagnostics, but splitting the model by class does not automatically improve the difficult business price tail.
- The tailored split model improved economy performance, yet its deterioration in business performance outweighed that benefit on the independent holdout, where high business fares dominate squared errors.
- An impressive cross-validation gain can disappear or reverse on a separate sample; this is *exactly* the type of generalization risk the user asked to guard against.
- Additional hyperparameter searches guided by the same holdout would invalidate its status as untouched; we will not do that.
- We cannot conclude that Kaggle test data are distribution-shifted merely because grouped CV scores differ from the user's public leaderboard R².

## Decision and honest next step

Preserve `xgb_850_d7_regularized.csv` as the best **publicly reported** submission. No newly generated submission is claimed to beat it. A future, separate study could examine leakage-safe, heavily smoothed route/airline/flight target encoding or explicit high-fare modeling using *fresh outer evaluations* (e.g. nested cross-validation), without repeatedly consulting this holdout or the Kaggle leaderboard for parameter tuning. This report makes **no promise** that those changes will improve the hidden test.

All comparisons, group assignments, early-stopping rounds, and predictions on training-only folds are reproducible from `run_phase5.py`.
