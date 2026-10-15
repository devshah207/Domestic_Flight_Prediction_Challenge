# Controlled exploratory CatBoost representation follow-up

Registered before any scores for this variant are computed. This is an
**exploratory hybrid procedure** motivated by diagnostics from the completed
original training-only study. Its outer folds have already informed the choice
to investigate CatBoost representation, so they are reused sensitivity evidence,
not a new untouched test or a new independent confirmation experiment.

## Hypothesis and the one permitted change

Training EDA identifies strong low-cardinality price regimes, whereas the
original CatBoost fits appear relatively weak even on their fitting rows.
The actual original full-training configuration records
`one_hot_max_size=2`, `ctr_target_border_count=1`,
`max_ctr_complexity=1`, `ctr_leaf_count_limit=512`, and `boosting_type=Plain`.
Its simple CTR descriptions contain `Borders` with `TargetBorderCount=1` and
`Counter`. A single target threshold in a Borders CTR can compress category
price information; it is not a continuous category mean. This is a plausible
representation limitation, rather than a proven explanation for model errors.

Change **only `one_hot_max_size` from 2 to 64** in the CatBoost constructor.
All categorical fields with at most 64 levels in the corresponding fitting
partition become eligible for native one-hot splits. The competition's
high-cardinality flight identities remain CTR-based. Do not manually one-hot
encode, add features, change targets, alter CTR settings, change tree structure,
increase trial counts, modify regularization, or change any other selected
hyperparameter. Tree counts are independently reselected under the unchanged
stopping policy below, as required when changing the representation.

## Fixed experiment, fits, and validation

- Variant name: `cat_onehot64`; base family: `cat`.
- Reuse each of the five original CatBoost outer-fold 12-trial selected
  configurations and feature modes; do not run a new Optuna search.
- Outer split seed remains `20261020`. For outer fold `i`, use the same first
  two of five inner grouped/stratified 80/20 splits and seed `20262220 + i`.
- Use exactly the original finite 1,200-round ceiling, 70-round patience, and
  `max(100, min(1200, int(median(two best iteration counts))))` policy.
  Stop only on these inner splits, refit the complete outer training partition
  at the selected count, then score its original outer validation fold.
- Refit all feature mappings independently inside every fitting partition.
  Grouping, stratification, target scale 10,000, original-price metrics,
  nonnegative clipping, seeds, thread count, depth and leaf bounds stay fixed.
- Full-data analogue: reuse the original full-training 12-trial-selected
  configuration and feature mode, with inner split seed `20263220`; reselect
  rounds on the same two full-training inner splits, then refit all training
  rows. Do not derive final rounds from the outer folds.
- Fit budget is exactly **18 CatBoost fits**: two stopping fits and one refit
  for each of five outer folds and the full-training analogue. No grid,
  additional trials, repeated seed search, or weight fitting is allowed.
- Require all six original selected-configuration JSON files, original final
  CatBoost configuration, and original source hashes to be available and
  consistent before starting any competition fit. Fail clearly if missing.

## Interpretation and outputs

Report pooled and per-fold original-price R², RMSE, MAE, bias, train R²,
train–validation gaps, stopping counts, timing, and the full original versus
new stopping records. Compare paired original CatBoost and new predictions
without declaring the reused outer folds pristine. Retain failures and do not
promise leaderboard improvement or freedom from overfitting.

Save results under `results/cat_onehot64/`, including OOF predictions,
`summary.json`, per-fold metrics and configuration, serialized final model,
exact constructor and resolved CatBoost parameters, feature lists, original
selected-configuration hashes, original study/features source hashes, and the
new module's source hash. The serialized feature transformer remains
`family='cat'`; variant configuration metadata explicitly records
`family='cat_onehot64'` and `base_family='cat'`.

Pre-register one optional alternative ensemble:
**1/3 original final XGBoost + 1/3 original final LightGBM + 1/3 CatBoost64**,
with individual nonnegative clipping before the arithmetic mean. No weight
search, target calibration, model reselection, or public-score feedback is
permitted. This addendum does not authorize opening competition test data.
Final inference, if requested, must freeze the complete variant and fixed
blend together before test access.

Implementation and synthetic API checks alone may be performed now. The root
agent will run the competition-data experiment after reviewing registration
and confirming the original artifacts are complete.
