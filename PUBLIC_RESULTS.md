# User-reported Kaggle submission feedback

These public R² scores were supplied by the user and have not been independently verified. They are feedback records, separate from training-only validation results. No private leaderboard scores are known.

| Submission | Reported public R² |
| --- | ---: |
| `submission_highround_optuna_classic_gradient.csv` | **0.95885** |
| `submission_business_augmented_xgb.csv` | 0.95881 |
| `submission_equal_blend_onehot64.csv` | 0.95803 |
| `submission_xgboost.csv` | 0.95797 |
| `submission_aggressive_business_xgb800.csv` | 0.95736 |
| `submission_catboost_onehot64.csv` | 0.95638 |
| `submission_equal_blend.csv` | 0.95504 |
| `submission_lightgbm.csv` | 0.95167 |
| `submission_catboost.csv` | 0.94199 |

The latest user feedback identifies `submission_highround_optuna_classic_gradient.csv` as the only candidate in the ten-file high-round batch that improved the reported best, reaching **0.95885**. This is **0.00004** above the preceding best of **0.95881**. Exact public scores for the other nine files have not been supplied and are not inferred here.

The preceding aggressive 800-tree Business candidate scored 0.95736, below its predecessor by 0.00145 despite local validation gains. The new classic gradient procedure had outer validation R² 0.922722, below the older equal-blend64 control's 0.928594 on the same folds. These observations show that local rankings have not reliably predicted public rankings. They do not identify the cause or establish private leaderboard performance. See [the feedback review](MODEL_FEEDBACK_REVIEW.md) for measured diagnostics and limitations.

No models were trained, parameters changed, predictions regenerated, or competition test data read to record this feedback. Frozen experiment folders, configurations, manifests, CSVs and archives remain unchanged; their original statements describe what was known at the time of each experiment. Public feedback should not be used to search repeatedly for model settings or blend weights.
