# Training-only protected model diagnostics

This report evaluates the three originally predeclared nested tuning procedures, not a single common hyperparameter dictionary. Each outer fold may choose different settings using only its inner training/tuning data. Original families share the same five itinerary-grouped outer folds. An optional registered CatBoost one-hot representation ablation and its replacement blend use those same folds and are explicitly exploratory after original fit diagnostics. No model was fitted by this analysis and no competition test, sample, or historical submission predictions were opened.

| model | r2 | rmse | mae | mean_fold_r2 | std_fold_r2 | worst_fold_r2 | mean_train_r2 | mean_train_validation_gap |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| xgb | 0.92542 | 10,685.50358 | 3,929.03409 | 0.92546 | 0.00776 | 0.91672 | 0.96797 | 0.04251 |
| lgb | 0.92514 | 10,705.11432 | 3,920.02914 | 0.92518 | 0.00832 | 0.91326 | 0.96573 | 0.04055 |
| cat | 0.91087 | 11,680.86810 | 4,553.58406 | 0.91090 | 0.00641 | 0.90147 | 0.94715 | 0.03624 |
| equal_blend | 0.92674 | 10,590.50002 | 3,964.83856 | 0.92677 | 0.00626 | 0.91735 | — | — |
| cat_onehot64 | 0.92506 | 10,711.24456 | 4,249.04792 | 0.92509 | 0.00718 | 0.91365 | 0.95145 | 0.02636 |
| equal_blend_onehot64 | 0.92859 | 10,455.40237 | 3,890.23510 | 0.92863 | 0.00646 | 0.91970 | — | — |

R², RMSE, MAE and bias use original ticket-price units. Pooled OOF R² differs from the mean of fold R². Training gap is mean training R² minus mean outer-fold R². Neither equal blend has a measured training R² or gap: averaging component scores would be mathematically invalid.

The fixed equal blend, when present, is exactly 1/3 XGBoost + 1/3 LightGBM + 1/3 CatBoost on the price scale. Its weights were not optimized. It uses the same held-out component predictions, and inherits their protected evaluation.

The optional `cat_onehot64` changes only CatBoost one_hot_max_size from 2 to 64, reuses each originally selected feature mode/configuration, and reselects rounds on the exact original two inner splits before outer refitting. It introduces zero new Optuna trials, with 18 additional fits planned across five outer scopes and the final scope. Its motivation followed the original CatBoost fit/representation diagnostics, so its evaluation is exploratory on reused outer observations, not pristine confirmation. The original CatBoost and original blend are retained regardless of which scores higher. The additional fixed `equal_blend_onehot64` is exactly 1/3 XGBoost + 1/3 LightGBM + 1/3 cat_onehot64; neither weights nor component membership are searched.

## Tail and cabin diagnostics

| model | top_1pct_target_n | top_1pct_target_sse_share | top_1pct_target_rmse | top_1pct_target_bias | top_1pct_squared_error_sse_share | business_sse_share |
| --- | --- | --- | --- | --- | --- | --- |
| xgb | 156 | 0.3459 | 62,818.5721 | -48,041.4913 | 0.6122 | 0.9675 |
| lgb | 156 | 0.3668 | 64,814.9585 | -50,377.2861 | 0.6150 | 0.9672 |
| cat | 156 | 0.4013 | 73,966.6728 | -64,667.8377 | 0.5523 | 0.9708 |
| equal_blend | 156 | 0.3898 | 66,093.3058 | -54,362.2050 | 0.5960 | 0.9728 |
| cat_onehot64 | 156 | 0.3531 | 63,628.0379 | -50,082.7270 | 0.5843 | 0.9727 |
| equal_blend_onehot64 | 156 | 0.3642 | 63,077.2558 | -49,500.5015 | 0.6096 | 0.9735 |

Top-1% target share means the fraction of total squared error from the highest-priced training observations. Top-1% squared-error share instead sorts rows by their model errors; these are different sets. Negative bias means underprediction. Detailed cabin/prefix/route/missingness/target-band metrics and the 40 worst errors per procedure are saved as CSVs. Small subgroups can have unstable or negative R²; their RMSE, bias and support should be considered together.

## Tuning and stopping audit

| family | scope | trials_completed | feature_mode | rounds | inner_best_rounds | inner_best_at_ceiling | leaves_requested | leaves_effective |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| xgb | full_training | 12 | context | 454 | [258, 650] | 0 | 32 | 32 |
| xgb | outer0 | 12 | context | 592 | [174, 1011] | 0 | 32 | 32 |
| xgb | outer1 | 12 | minimal | 812 | [439, 1185] | 0 | 32 | 32 |
| xgb | outer2 | 12 | context | 344 | [303, 386] | 0 | 32 | 32 |
| xgb | outer3 | 12 | context | 418 | [551, 286] | 0 | 32 | 32 |
| xgb | outer4 | 12 | minimal | 738 | [590, 886] | 0 | 24 | 24 |
| lgb | full_training | 12 | context | 764 | [975, 553] | 0 | 32 | 32 |
| lgb | outer0 | 12 | context | 581 | [800, 362] | 0 | 32 | 32 |
| lgb | outer1 | 12 | context | 690 | [992, 389] | 0 | 24 | 24 |
| lgb | outer2 | 12 | context | 826 | [928, 725] | 0 | 16 | 16 |
| lgb | outer3 | 12 | context | 324 | [560, 89] | 0 | 32 | 32 |
| lgb | outer4 | 12 | context | 287 | [224, 350] | 0 | 48 | 32 |
| cat | full_training | 12 | context | 344 | [509, 180] | 0 | 32 | 32 |
| cat | outer0 | 12 | context | 1048 | [1064, 1032] | 0 | 32 | 32 |
| cat | outer1 | 12 | context | 1016 | [1140, 893] | 0 | 32 | 32 |
| cat | outer2 | 12 | context | 557 | [737, 377] | 0 | 48 | 48 |
| cat | outer3 | 12 | context | 699 | [996, 403] | 0 | 32 | 32 |
| cat | outer4 | 12 | context | 774 | [974, 575] | 0 | 24 | 24 |
| cat_onehot64 | full_training | 0 | context | 189 | [162, 217] | 0 | 32 | 32 |
| cat_onehot64 | outer0 | 0 | context | 717 | [558, 877] | 0 | 32 | 32 |
| cat_onehot64 | outer1 | 0 | context | 916 | [1199, 633] | 0 | 32 | 32 |
| cat_onehot64 | outer2 | 0 | context | 378 | [455, 302] | 0 | 48 | 48 |
| cat_onehot64 | outer3 | 0 | context | 326 | [391, 261] | 0 | 32 | 32 |
| cat_onehot64 | outer4 | 0 | context | 256 | [284, 228] | 0 | 24 | 24 |

Across 24 completed tuning scopes, 216 archived trials were inspected. 13 selected scopes have inner best-round ranges of at least 300 trees. This is a warning about round-transfer uncertainty, not an automatic rejection or permission to change the predeclared rule.

`all_trials.csv` records every archived trial, its raw Optuna parameters, effective clipped leaf count, feature option, inner tuning objective and stop dispersion. `selected_configurations.csv` contains per-outer and available full-training choices. `all_trial_inner_stopping.csv` gives each inner split separately. `inner_best_at_ceiling` counts recorded best iterations exactly at the finite ceiling. The additional `best_plus_patience_at_ceiling` flag identifies counts close enough to the cap that the stopping budget may constrain the fit; actual trained iteration histories were not saved, so this is not presented as a measured training-cap hit count.

Final full-training dictionaries, when available, are copied to `final_configurations_snapshot.json`. Their in-sample fit scores are not new validation evidence. Original full-training tuning reruns the registered procedure on all labeled data; the optional representation ablation reuses original settings with fresh inner round selection. Outer scores describe these procedures at the outer-training sample size and do not independently validate each final dictionary. The optional ablation and its replacement blend remain exploratory even if their bootstrap intervals favor them.

## Comparisons and uncertainty

| model | estimate | ci_low | ci_high |
| --- | --- | --- | --- |
| xgb | 0.92542 | 0.91760 | 0.93300 |
| lgb | 0.92514 | 0.91745 | 0.93285 |
| cat | 0.91087 | 0.90300 | 0.91889 |
| equal_blend | 0.92674 | 0.91907 | 0.93396 |
| cat_onehot64 | 0.92506 | 0.91595 | 0.93218 |
| equal_blend_onehot64 | 0.92859 | 0.92063 | 0.93564 |

| model_a | model_b | delta_r2_a_minus_b | ci_low | ci_high | bootstrap_fraction_positive |
| --- | --- | --- | --- | --- | --- |
| xgb | lgb | 0.00027 | -0.00284 | 0.00321 | 0.57600 |
| xgb | cat | 0.01454 | 0.00861 | 0.01999 | 1.00000 |
| xgb | equal_blend | -0.00132 | -0.00383 | 0.00092 | 0.11800 |
| xgb | cat_onehot64 | 0.00036 | -0.00359 | 0.00400 | 0.58000 |
| xgb | equal_blend_onehot64 | -0.00318 | -0.00543 | -0.00148 | 0.00000 |
| lgb | cat | 0.01427 | 0.00820 | 0.01935 | 1.00000 |
| lgb | equal_blend | -0.00159 | -0.00428 | 0.00063 | 0.09600 |
| lgb | cat_onehot64 | 0.00009 | -0.00403 | 0.00411 | 0.51600 |
| lgb | equal_blend_onehot64 | -0.00345 | -0.00570 | -0.00141 | 0.00000 |
| cat | equal_blend | -0.01586 | -0.01968 | -0.01222 | 0.00000 |
| cat | cat_onehot64 | -0.01418 | -0.01857 | -0.00963 | 0.00000 |
| cat | equal_blend_onehot64 | -0.01772 | -0.02240 | -0.01286 | 0.00000 |
| equal_blend | cat_onehot64 | 0.00168 | -0.00104 | 0.00453 | 0.87200 |
| equal_blend | equal_blend_onehot64 | -0.00186 | -0.00320 | -0.00048 | 0.00200 |
| cat_onehot64 | equal_blend_onehot64 | -0.00354 | -0.00586 | -0.00123 | 0.00200 |

Intervals use 500 deterministic bootstrap resamples of whole itinerary groups (seed 20261021), recomputing target variance in every resample. They describe the current OOF predictions and do not include retraining, fold-generation or repeated-selection uncertainty. They are not historical-baseline promotion gates. Pairwise differences are descriptive; comparing families also creates selection optimism.

Validation-supported candidates can be delivered even when a difference between two families is uncertain. None of these scores is a Kaggle score, and none guarantees absence of overfitting, a future score improvement, or a rank. The prior user-reported public R² of 0.95797 comes from different hidden observations and must not be directly compared with these local values.

The experiment remains bounded by its registered families/search spaces. A train–validation gap is an overfitting diagnostic, not a proof that a model is good or bad. Weak tail behavior, fold variability and unstable stopping counts remain explicit risks. Report every registered procedure instead of describing the best observed family as independently proven superior.

## Completion state

```json
{
  "xgb": {
    "status": "complete",
    "outer_folds_completed": 5,
    "exploratory_after_original_diagnostics": false
  },
  "lgb": {
    "status": "complete",
    "outer_folds_completed": 5,
    "exploratory_after_original_diagnostics": false
  },
  "cat": {
    "status": "complete",
    "outer_folds_completed": 5,
    "exploratory_after_original_diagnostics": false
  },
  "cat_onehot64": {
    "status": "complete",
    "outer_folds_completed": 5,
    "exploratory_after_original_diagnostics": true
  },
  "equal_blend": {
    "status": "complete",
    "weights": {
      "xgb": 0.3333333333333333,
      "lgb": 0.3333333333333333,
      "cat": 0.3333333333333333
    },
    "weight_search": false
  },
  "equal_blend_onehot64": {
    "status": "complete",
    "weights": {
      "xgb": 0.3333333333333333,
      "lgb": 0.3333333333333333,
      "cat_onehot64": 0.3333333333333333
    },
    "weight_search": false,
    "exploratory_after_original_diagnostics": true
  }
}
```

This analysis is idempotent and can be rerun as pending scopes/families finish. Partial reports do not silently invent missing models or redefine the blend. See `analysis_metadata.json` for the training hash and completion state.
