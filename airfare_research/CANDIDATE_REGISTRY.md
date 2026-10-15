# Registered structural hypotheses

Written before candidate predictions/scores. Primary folds: raw duplicate blocked,
seed20261015. Fixed850 retains exact historical parameters and training scale.

H1 native: replace sparse one-hot category indicators with XGBoost native ordered
categorical partitions. Same semantic columns and rare-flight threshold12; native
unseen categories become missing rather than OHE rare pooling. Finite partition
search max_cat_threshold16 and max_cat_to_onehot1. Depth7/48 leaves, child12,
L1.6/L2 15, gamma.08, row.8/col.85 unchanged. Hypothesis: partitioning carrier,
route and class subsets costs fewer splits than separate indicators and avoids
column-budget fragmentation. Risk: sparse categories/partition overfit. It is a
whole representation change, not an isolated one-bit encoding comparison.

H2 duration_schedule: feature-only duration hierarchy, accepted for price-model
comparison only if train-only random masking reconstruction improves; add flight,
route, stops, departure/arrival matching ahead of original fallback, without target
statistics. Keep all model parameters and features otherwise unchanged. Imputation
rule must be fixed from reconstruction diagnostic before price-model scoring.

H3 prefix_duration: append prefix-specific Business-duration gated numeric terms,
using training prefixes only, retaining original features. It differs from failed
Phase4's global class-duration bundle. Observed training Business correlations
vary by prefix (+.71 for6E versus negative forEY), suggesting heterogeneous shapes;
no monotonic constraint is imposed. Bound to at most19 terms, one per observed
training prefix; unknown prefix gets zero terms. Risk: added terms alter column
subsampling and can overfit sparse carriers. No target encoding or price priors.

No combinatorial bundles or extra hyperparameter grid. Use the promotion gate in
VALIDATION_PROTOCOL.md. Rejected candidates remain documented. Advance at most one
unchanged candidate to repeated folds seed20261016. Fixed850 is deliberately held
constant to isolate structure and preserve full-data round policy; no outer fold
is used for early stopping. Separate nested stopping may be reported as an
explicitly new procedure only with a predeclared finite policy.
