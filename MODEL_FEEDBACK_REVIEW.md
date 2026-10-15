# Review of the reported classic gradient public result

The user reports public R² 0.95885 for submission_highround_optuna_classic_gradient.csv, a gain of 0.00004 over 0.95881. The other nine batch scores are unknown individually. This report reads existing training-only artifacts; no model fitting, test reads, prediction changes, or archive changes are performed.

## What the evidence supports

The classic gradient tuning procedure scored outer OOF R² 0.922722; the older equal64 blend scored 0.928594. The lower locally ranked procedure achieved the new reported public best. The local evaluation is therefore an imperfect guide to the public ranking. This does not by itself prove the folds are wrong or identify the competition sampling scheme.

Business rows account for 97.56% of this procedure's held-out squared error, while the most expensive 1% of fares account for 37.39%. The top-1% mean prediction error is -52,882 in price units. Expensive fares are a measured error concentration; differing public sample composition is an unverified possible explanation, not a demonstrated cause.

## Search depth and selection instability

The headline 336 trials were spread across eight families and six tuning scopes per family. Classic gradient received only six trials per scope, including two queued starting profiles. That is a shallow search in a multidimensional space, not 336 trials focused on this model. Each scope ranked trials using one inner split. Spending this budget broadly satisfied model variety but provided limited tuning depth and selection stability.

| Scope | Rounds | Max depth | Learning rate | Max leaves | Min samples/leaf | Trials |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| outer0 | 1400 | 8 | 0.012208 | 64 | 39 | 6 |
| outer1 | 1100 | 8 | 0.012473 | 32 | 7 | 6 |
| outer2 | 1400 | 6 | 0.012590 | 16 | 30 | 6 |
| outer3 | 1400 | 7 | 0.018000 | 64 | 25 | 6 |
| outer4 | 1100 | 8 | 0.014454 | 64 | 7 | 6 |
| full_training | 1100 | 8 | 0.047476 | 64 | 6 | 6 |

Outer folds evaluate the tuning procedure, with independently selected configurations. The submitted 1100-tree configuration was selected by a fresh full-data inner split. Its full-data training R² is 0.983122, while the mean outer fitted-model training R² is 0.958221. These are different configurations fitted on different sample sizes; subtracting the submitted model’s training R² from the procedure’s OOF R² would not be a valid same-model overfitting gap. The selection table shows materially different choices across partitions.

## Implication for further work

A focused next study should improve selection stability with multiple training-only inner folds, measure learning curves and paired expensive-fare errors, and compare a limited family of classic-gradient and existing strong baselines under equal budgets. More trees alone are not an identified fix. A public result can motivate a new hypothesis, but repeated search for settings or mixture weights against public feedback would adapt to that leaderboard. New configurations and blend weights must still be selected from labeled training validation.

All labeled training observations and previous folds have already informed development. Additional resampling can assess sensitivity; it cannot create a retrospectively untouched holdout. Neither the new public score nor this review establishes a top-five rank or private-test improvement. Existing frozen reports and archives retain their original, time-specific statements.
