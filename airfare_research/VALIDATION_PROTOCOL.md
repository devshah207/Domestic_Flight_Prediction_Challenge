# Protocol registered before new candidate scores

The fixed historical control uses original feature builder, sparse OHE, price/10000,
850 trees and seed42; no stopping or altered regularization. Historical package
versions were not supplied. Current versions and resolved booster config are saved.

Baseline audit: historical raw-duplicate SGKF five folds seed42; new raw-duplicate
SGKF five folds seed20261015; ordinary stratified five-fold seed20261015; semantic
record grouping (drop airline/duration from group key) five-fold seed20261015;
flight-identity grouping five-fold seed20261015 is a stress test, not the selection
metric. No date exists; days_left is booking horizon, not chronological time.

Primary candidate comparison: identical raw-duplicate-blocked outer five folds,
seed20261015. Predeclare no more than three structural hypotheses after train-only
EDA, before candidate scoring. Fixed-round ablations retain 850 exactly; early-stop
procedures may use cap1400/patience75 and two inner fold-pure stops with median
rounds and minimum100, then refit full outer training. Outer labels never choose
rounds. New seed20261016 repeats the baseline and the one preselected winner
without further adjustments. This is repeated-CV confirmation, not new independent
observations: the full labeled dataset has informed prior research.

Promotion gate: pooled original-price delta R2 >= .001 on primary comparison;
positive delta on at least4/5 folds; paired raw-record-group bootstrap lower95%
limit >0; positive repeated-CV pooled delta and at least4/5 positive repeat folds.
Intervals are conditional paired OOF uncertainty, not population or retraining
confidence guarantees. Report flight-cluster uncertainty too. If selection among
multiple candidates is close prefer simplest; never claim tiny gains or reused
Phase5 holdout is independent proof. A conservative Bonferroni paired interval
across tested primary hypotheses will also be reported for a winner.

All EDA and feature fitting uses training only. No Kaggle test features are read
until a frozen manifest records all pipelines, rounds, transformations, weights,
postprocessing, seeds and hashes. Final inference reads test once, validates IDs,
shape and finite nonnegative predictions, and generates only a control refit and
any justified winner. The old best submission is copied byte-for-byte. No
leaderboard-derived choices, NNs, target encoding, pseudo-labeling, target weights,
or repeated failed class-split/bundle experiments. No submission upload is part of
this research turn.
