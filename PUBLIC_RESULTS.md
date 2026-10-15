# User-reported Kaggle submission feedback

These public R² scores were supplied by the user and have not been independently verified. They are feedback records, separate from training-only validation results. No private leaderboard scores are known.

| Submission | Reported public R² |
| --- | ---: |
| `submission_business_augmented_xgb.csv` | **0.95881** |
| `submission_equal_blend_onehot64.csv` | 0.95803 |
| `submission_xgboost.csv` | 0.95797 |
| `submission_aggressive_business_xgb800.csv` | 0.95736 |
| `submission_catboost_onehot64.csv` | 0.95638 |
| `submission_equal_blend.csv` | 0.95504 |
| `submission_lightgbm.csv` | 0.95167 |
| `submission_catboost.csv` | 0.94199 |

The latest feedback, recorded on 2026-10-15 UTC, is interpreted as referring to the single most recently delivered candidate, `submission_aggressive_business_xgb800.csv`. Its public score is 0.00145 below the current best. Other follow-up submissions have no reported public scores here.

The aggressive candidate improved pooled training-only validation on both reused split seeds, but increased its training–validation gap. Its public result shows that this local gain did not transfer to the public evaluation subset. One public result does not identify the cause, prove overfitting, establish private leaderboard performance, or show that 800 trees alone caused the decline. The previous 0.95881 submission remains the best reported public result.

No models were trained, parameters changed, predictions regenerated, or competition test data read to record this feedback. Frozen experiment folders, configurations, manifests, CSVs and archives remain unchanged; their original statements describe what was known at the time of each experiment. Public feedback should not be used to search repeatedly for model settings or blend weights.
