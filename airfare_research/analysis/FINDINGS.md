# Training-only experiment diagnostics

Completed OOF procedures: 11. Each row and label was checked against the supplied labeled training CSV. No Kaggle test/sample/preserved predictions were read.

## Measured pooled performance

| Procedure | Pooled R² | RMSE | MAE | Mean train−validation R² | Top-price 1% SSE share |
|---|---:|---:|---:|---:|---:|
| flight_20261015_baseline | 0.874981 | 13,834.4 | 5,179.9 | 0.08740 | 36.5% |
| iid_20261015_baseline | 0.930814 | 10,291.5 | 3,806.4 | 0.03286 | 38.6% |
| raw_42_baseline | 0.928923 | 10,431.2 | 3,852.3 | 0.03485 | 38.3% |
| raw_20261015_blend_equal850 | 0.931243 | 10,259.6 | 3,632.5 | Unavailable | 39.1% |
| raw_20261015_duration_schedule | 0.929847 | 10,363.2 | 3,845.8 | 0.03403 | 37.5% |
| raw_20261015_baseline | 0.929811 | 10,365.9 | 3,841.5 | 0.03415 | 38.3% |
| raw_20261015_native_nestedstop | 0.929336 | 10,400.9 | 3,693.5 | 0.03554 | 39.1% |
| raw_20261015_baseline_nestedstop | 0.928699 | 10,447.7 | 3,919.4 | 0.02626 | 38.5% |
| raw_20261015_native | 0.927988 | 10,499.7 | 3,615.1 | 0.05306 | 38.4% |
| raw_20261015_prefix_duration | 0.926565 | 10,602.9 | 3,916.5 | 0.03703 | 38.7% |
| semantic_20261015_baseline | 0.929515 | 10,387.7 | 3,854.7 | 0.03426 | 36.7% |

Results from different fold schemes are sensitivity analyses; their differences are not candidate improvement estimates.

## Paired comparisons on identical folds

| Candidate | Δ R² | Positive folds | Raw-group 95% interval | Raw-group Bonferroni 98.333% interval | Flight-cluster 95% interval | Numeric promotion gate |
|---|---:|---:|---|---|---|---|
| raw_20261015_baseline_nestedstop | -0.001112 | 2/5 | [-0.002547, +0.000359] | [-0.002869, +0.000737] | [-0.002785, +0.000462] | Fail |
| raw_20261015_blend_equal850 | +0.001432 | 3/5 | [-0.000002, +0.002894] | [-0.000330, +0.003166] | [-0.000348, +0.003382] | Fail |
| raw_20261015_duration_schedule | +0.000037 | 1/5 | [-0.000765, +0.000850] | [-0.001016, +0.001048] | [-0.000761, +0.000816] | Fail |
| raw_20261015_native | -0.001823 | 2/5 | [-0.004810, +0.001177] | [-0.005491, +0.001784] | [-0.005470, +0.001888] | Fail |
| raw_20261015_native_nestedstop | -0.000475 | 3/5 | [-0.002741, +0.001936] | [-0.003162, +0.002327] | [-0.003477, +0.002187] | Fail |
| raw_20261015_prefix_duration | -0.003246 | 0/5 | [-0.005075, -0.001592] | [-0.005561, -0.001225] | [-0.005581, -0.001161] | Fail |

The numeric gate requires Δ R² ≥0.001, at least four positive folds, and a positive raw-duplicate-group 95% lower bound. A passing primary comparison still requires the predeclared unchanged repeated-CV confirmation and subgroup review; this script does not select or fit a model. Family-three adjusted-gate flags describe only that registered primary sensitivity; expanded stopping/blending comparisons additionally carry a family-six strict sensitivity flag.

## Concrete findings from completed procedures

The primary fixed850 control has pooled R² 0.929811. Business rows contribute 97.4% of its squared error; the highest-price 1% contribute 38.3%. The largest-error 1% (a different set of rows) contribute 60.4%. Tail bias is -50,389 price units.

- `raw_20261015_baseline_nestedstop`: Δ R² -0.001112, 2/5 positive folds. Fails material-effect threshold, four-positive-fold threshold, positive raw-group confidence bound.
- `raw_20261015_blend_equal850`: Δ R² +0.001432, 3/5 positive folds. Fails four-positive-fold threshold, positive raw-group confidence bound.
- `raw_20261015_duration_schedule`: Δ R² +0.000037, 1/5 positive folds. Fails material-effect threshold, four-positive-fold threshold, positive raw-group confidence bound.
- `raw_20261015_native`: Δ R² -0.001823, 2/5 positive folds. Fails material-effect threshold, four-positive-fold threshold, positive raw-group confidence bound.
- `raw_20261015_native_nestedstop`: Δ R² -0.000475, 3/5 positive folds. Fails material-effect threshold, four-positive-fold threshold, positive raw-group confidence bound.
- `raw_20261015_prefix_duration`: Δ R² -0.003246, 0/5 positive folds. Fails material-effect threshold, four-positive-fold threshold, positive raw-group confidence bound.

## Residual complementarity

No blend weights or blend predictions are computed by this section. Native850/control residual covariance and correlations are saved to `residual_complementarity.csv` for all rows, cabins, and expensive-target rows. Covariance below both constituent residual variances suggests a fixed average may reduce variance; it does not establish a total squared-error gain in the presence of bias or justify a submission without the predeclared confirmation.

## Error and validation interpretation

- `subgroup_metrics.csv` includes class, prefix, route, stops, missingness, booking horizon, target percentiles/deciles, and flight frequency measured within each corresponding training fold. Unseen and rare flights are separate.
- `top1_target_sse_share` selects the highest actual-price 1%; `top1_error_sse_share` selects the largest squared-error 1%. They answer different questions. Their exact selected row counts are saved.
- Fold composition includes target-decile, rare expensive-prefix, Business, and extreme-target counts. A percentile-specific R² can be negative or unstable because target variance within that narrow range is small; use RMSE/bias for those diagnostics.
- Native representation may reduce MAE while worsening R²; the official original-price squared-error objective decides promotion. Error concentration and train–validation gaps help diagnose the tradeoff.
- Each cluster bootstrap resamples whole groups, recomputes resampled n/sum(y)/sum(y²), and uses the resulting denominator for the paired R² difference. Raw-duplicate and flight-cluster intervals assess different dependence assumptions.
- These are conditional intervals on existing OOF predictions. Folds/repeats overlap in training data; neither bootstrap nor repeated CV captures full retraining uncertainty or erases prior selection on this labeled dataset. No reused Phase5 holdout is called independent proof.
- Bonferroni intervals use family size three (98.333% coverage), as registered, even if fewer candidates have finished. Sensitivity intervals for family sizes four (98.75%), five (99%), and six (99.1667%) also appear in the comparison CSV/JSON when stopping/blending procedures expand the exploratory family. They supplement the predeclared 95% promotion criterion. With 2,000 draws, extreme adjusted tails use relatively few draws; treat those quantiles as approximate.
- For early-stopped procedures use actual per-fold round counts in `fold_metrics.csv`; an older summary parameters dictionary can retain the default 850 even when the fitting policy differs.
- Historical public Kaggle 0.95797 is user-reported and concerns another sample. No hidden-test score or rank improvement is measured here.

All artifacts can be regenerated by rerunning this analyzer as additional completed OOF results arrive.
