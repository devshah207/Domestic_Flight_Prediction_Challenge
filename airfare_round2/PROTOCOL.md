# Direct CatBoost64 tuning: prospective experiment registration

Registered before this phase computes any new fitted scores. User authorization:
directly tune CatBoost64 while limiting overfitting, following the independent
study. Earlier code, models, predictions and freeze manifests remain unchanged.

## Hypothesis and scope

The existing CatBoost64 reused hyperparameters chosen with threshold 2 and only
reselected rounds at threshold 64. Here the entire existing bounded 12-trial
Optuna procedure runs with `one_hot_max_size=64`. This changes the representation
used during search; it does not introduce a new feature family, loss, tree-growth
policy, larger search space or neural model. The existing minimal/context feature
options remain available for inner-only selection. No warm-start configurations
learned from full-data tuning or outer validation are inserted into outer trials.
The two generic initial configurations and finite search space are exactly those
in the frozen `airfare_fresh/study.py`.

Read labeled training data only during development. Do not open competition test,
sample, historical submissions or previously generated test predictions for EDA,
tuning, selection, calibration or ensemble design. Prior public scores are
user-reported provenance, not an optimization objective or selection criterion.
Earlier phases accessed test data; this is a new development/inference boundary,
not a claim that the test has never been opened.

## Primary procedure and comparisons

- Five outer itinerary-grouped/stratified folds, seed **20261020**, exactly matching
  the independent study; all learned preprocessing refitted in each partition.
- Each outer-training scope: first two folds of an inner five-fold split, seed
  **20262220 + outer_index**. Twelve TPE trials, five startup trials including the
  two queued generic minimal/context configurations. Model seed 42; four threads.
- Parameter space unchanged: depth 4–7, effective leaves in 16/24/32/48 capped at
  2^depth, learning rate .025–.07 log scale, row sample .72–.95, column sample
  .70–.95, L2 3–35 log scale, minimum leaf support 10–60, random strength .2–2 log
  scale. Lossguide/Bernoulli, model-size regularization 1, CTR complexity 1,
  CTR leaf limit 512; one-hot threshold fixed at 64 in every fit.
- Inner validation selects parameters and stopping only. Cap 1200, patience 70,
  selected refit rounds `max(100, min(1200, int(median(two inner stops))))`.
  Refit all outer-training rows without outer early stopping. No correction for
  changing fitting sample size is invented after viewing outer scores.
- Target `price / 10000`; invert and clip each component below at zero. Original
  price MSE is the tuning objective and original-price R² is the primary metric.
- Compare direct CatBoost64 with the frozen original CatBoost64 OOF predictions.
  Also compare the fixed thirds blend of frozen original XGBoost, frozen original
  LightGBM and new direct CatBoost64 with the existing frozen thirds blend.
  No weight/subset search. Outer assignments and row identities must align.
- Main budget: five outer scopes plus one full-training scope, **72 trials,
  144 inner fits + five outer refits + one full refit = 150 model fits**.
  Full-data inner seed **20263220**, same tuner and median-stopping rule; do not
  copy outer round counts. No failed trial is silently replaced or budget expanded.

## Conditional split-seed robustness check

Register now, before primary results. Run this check only if primary direct
CatBoost64 pooled R² exceeds the frozen CatBoost64 by at least .0005, OR the fixed
replacement blend improves pooled R² by at least .0002 and improves at least three
of five paired fold R² scores. This is a compute-allocation rule, not a CSV
eligibility test or a statistical proof of superiority.

Use second outer seed **20261027**, all five folds; inner seeds **20272220 + i**.
For each partition rerun both:

1. Original threshold-2 CatBoost 12-trial selection followed by threshold-64
   round reselection and outer refit, exactly reconstructing the previous hybrid
   control on these new partitions. Both the initial threshold-2 refit and the
   threshold-64 refit are scored and retained.
2. Direct threshold-64 12-trial selection and outer refit, on the same inner splits.

Budget **120 additional trials and 265 fits**: 240 trial fits, ten original/direct
outer refits, ten hybrid-control stopping fits and five hybrid-control refits.
Maximum phase budget **192 trials and 415 research fits**. No full-data tuning is
repeated for the sensitivity check. XGBoost and LightGBM are not rerun on the
secondary folds; do not manufacture a secondary blend from their primary OOFs.
This checks the standalone CatBoost increment, not the complete replacement blend.

The second seed is a prespecified robustness check on already used observations,
not a pristine holdout. Primary folds and historical analyses have informed this
research direction; all comparisons are developmental. No additional seed search
or parameter changes follow secondary results within this registration.

## Reporting and final inference

Report all trials, selected parameters, stopping counts/dispersion, train versus
outer R², original-unit errors, paired fold changes, Business/Economy and target
band diagnostics, rare-flight support and the largest-error observations.
Include 500 whole-itinerary-group paired bootstrap resamples, explicitly
conditional on the fitted OOF predictions, not retraining uncertainty.

Do not claim absence of overfitting. A numerical CV gain or conditional interval
does not guarantee hidden-test improvement. If fits, validation boundaries,
complexity bounds or predictions fail, report the failure and withhold affected
CSVs. Valid bounded candidates may be delivered for the user's manual evaluation
even if they do not beat the frozen control; label the measured evidence and
retain the previous best. Do not alter model choices in response to test outputs.

Freeze the new full-data model, the two existing XGBoost/LightGBM full-data models,
source/configuration hashes, exact feature mappings, package versions and fixed
thirds policy together before this phase's final inference. Then read test once
and output only the standalone direct CatBoost64 and fixed replacement blend.
Preserve IDs exactly, 3898 rows, schema id/price, finite nonnegative values.
Use an exclusive inference marker and do not overwrite earlier CSVs.

Business-specialist modeling and seed-averaged training are separate future
experiments; they are not silently added to this direct-tuning study.
