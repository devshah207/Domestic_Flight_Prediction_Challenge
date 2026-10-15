# Independent training-only EDA

Scope: read only the supplied labeled training CSV (15,589 rows). No model fitted, no Kaggle test/sample or historical prediction files opened. This is fresh descriptive work, not a rerun of the historical feature builder. Reproduce with `python /workspace/airfare_fresh/independent_eda.py`.

## What the raw observations say

- There are 1,070 flight identifiers, 19 prefixes, 30 directional routes and two cabins. Prices range from 3,801 to 323,177; 99th percentile is 185,837. Airline is missing in 19.82% and duration in 19.85%.
- Every prefix with an observed airline maps to one airline label. All missing-airline prefixes have an observed training mapping. AI and IX share the same airline label, so prefix carries finer information. A compact raw pipeline can use prefix directly instead of guessing missing airlines. Any learned label mapping must remain fold-local.
- Business median prices differ sharply by prefix: AI 37,518, 6E 30,055, UL 71,781, EK 130,960, EY 147,049. QP has only 30 business observations (versus 830 economy); fine business interactions there are sparse.
- Duration–price association changes sign by prefix within Business: Spearman AI +0.632 and 6E +0.773, but EK −0.241 and EY −0.382. A global linear duration effect or a simple duration-based fare correction is unjustified. A joint tree model can represent this heterogeneity.
- Booking horizon is not globally monotonic. Business days–price Spearman is UL −0.371 but TG +0.132; Economy AI and 6E are approximately zero, IX +0.117, QP +0.109. Close/far median ratios are descriptive and mix routes, so they are not causal booking-discount estimates. Preserve continuous days_left; do not impose global monotonicity based on business intuition.
- Missingness varies modestly across the large cabin/prefix groups: duration 12.9%–21.6%, airline 18.9%–24.0%. This is compatible with broadly scattered missingness; it does not establish missing-at-random. Missing flags are reasonable low-cost options, not proven signals.

## Duration structure and ambiguity

| grouping | groups_n_ge_3 | median_std_hours | p90_std_hours | groups_range_about_24hr | max_range_hours |
| --- | --- | --- | --- | --- | --- |
| flight | 820 | 0.041 | 2.306 | 0 | 35.500 |
| flight+route | 1076 | 0.000 | 0.228 | 2 | 30.000 |
| flight+route+schedule | 1089 | 0.000 | 0.049 | 3 | 24.000 |
| prefix+route+schedule | 873 | 0.000 | 0.384 | 3 | 24.000 |

Flight numbers are reused on routes or schedules. Route/schedule conditioning sharply reduces within-group duration variation. Even fully specified schedules sometimes differ by 24 hours, and one one-stop row reports less than three hours. Do not automatically delete these as errors: layovers and day rollover can be hidden. Native missing duration is a clean baseline; any hierarchical fill should retain missingness and avoid giving tiny/noisy groups unjustified confidence. Duration diagnostics above use observed values only, no target.

## Duplicate and sparse-group risks

There are 314 extra rows with exactly the same raw features excluding id, versus 279 extra fully identical labeled rows. 35 feature groups contain different prices, covering 71 rows; the largest observed price range within identical features is 18,540. Their in-sample within-group squared variation is 0.001889% of total target SST. These collisions show hidden variables or label ambiguity, not grounds to remove rows or labels. Keep exact raw feature groups together during validation.

Flight×route×class has 2,173 groups, with 2,825 rows in groups of at most five observations. More detailed flight/schedule/class interactions would further fragment rare expensive business cases. Use compact route and prefix×class features before adding an interaction bundle. Numeric flight suffix is a code whose ordering has no guaranteed economic interpretation; preserve categorical flight and treat numeric suffix inclusion as an optional ablation.

No high-price observations were removed or relabeled. Rare expensive tickets are strongly structured by prefix/class; extremeness alone is insufficient evidence of corruption. Any residual/tail diagnosis must come from held-out predictions, not the in-sample summaries here.

## Limited features to test

1. Raw minimal: categorical flight/prefix/source/destination/departure/arrival/stops/class; numeric raw duration and days_left; native missing duration.
2. Compact interactions: add route and prefix_class to the raw minimal set. These are deterministic row-wise transforms, with no fitted cross-row information.
3. Only as a separate ablation: fold-local duration fill with explicit support/uncertainty and missing flag; retain original raw features. Avoid simultaneously changing target, feature bundle and categorical representation.

All feature recommendations remain hypotheses. This report has no CV/model score and does not select a winner. Tune bounded models with inner-only stopping, score on untouched outer folds, and report original-price pooled R2 and train–validation gaps. Do not infer that a low gap proves no overfitting.

## Artifacts

- `data_summary.json`: source hash, shape, target/missingness and duplicate facts.
- `conditional_dependence.csv`, `booking_price_curves.csv`, `price_by_class_prefix.csv`: raw associations and support.
- `interaction_sparsity.csv`: target-free support for proposed groups.
- `duration_group_consistency.csv` and detailed duration tables: schedule reliability.
- `exact_duplicate_conflicts.csv`: conflicting duplicate groups, retained unchanged.
- `feature_recommendations.json`: machine-readable, small feature variants.
- Three PNG figures show fare regimes, booking curves, duration reliability and missingness.
