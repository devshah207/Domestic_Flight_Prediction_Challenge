# Independent airfare study: frozen research protocol

This study treats the handoff as prior experience, not a restriction to its
models or a demand to recover a particular public leaderboard score. It starts
from training-only EDA and compares three bounded, non-neural boosting
procedures: native-categorical XGBoost, LightGBM, and CatBoost. The user's
clarified request authorizes independent modeling, Optuna tuning, and multiple
validation-supported submission files. It supersedes the earlier requirement
that a candidate must prove a statistically significant improvement over the
historical baseline before receiving a CSV.

No public score or rank is promised. A model cannot be guaranteed free from
overfitting. Validation estimates, complexity restrictions, and diagnostics
provide evidence and expose risks; they do not establish hidden-test results.

## Information boundary

- Development reads only the supplied labeled training CSV and historical
  documents. `id` is excluded from predictors. Competition test features,
  historical test predictions, and historical submission values do not inform
  feature choices, tuning, round counts, weights, or postprocessing.
- The preceding research phase already performed historical control inference.
  This new authorized phase does not treat test data as never previously
  accessed. It establishes a new boundary: no test access in this study until
  its final models and inference behavior have been frozen from training data.
- Final inference may load the test CSV only after saving the complete freeze
  manifest, fitted full-training models, exact configurations, feature schemas,
  round counts, any ensemble weights, and postprocessing. Thereafter it performs
  inference and submission schema/shape/finite-value checks, without new model
  development based on test rows or predictions.
- No neural networks, pseudo-labeling, test-derived frequency maps, external
  labels, test-assisted imputation, or target encoding fitted outside the
  appropriate training partition.

## Features, targets, and bounded search

`features.py` supplies two small feature options. `minimal` keeps the original
predictors, flight prefix, and missingness indicators. `context` adds route,
prefix-by-class, numeric stops, and fold-local duration imputation while
retaining the raw duration. Airline-prefix maps, categorical vocabularies,
duration medians, and all other learned preprocessing are fitted separately on
each corresponding training partition. CatBoost receives native categorical
strings; XGBoost and LightGBM receive categorical columns with training-fitted
vocabularies. Validation/inference rows never update these mappings.

The exact finite parameter spaces and estimator settings are defined in
`study.py` and saved with the results. The fixed stopping ceiling is 1,200
trees with patience 70; selected final counts have a floor of 100. Training
targets are `price / 10000`, all three families use squared-error/RMSE losses,
and predictions are multiplied by 10,000 and clipped below at zero. The model
seed is 42. Depth, leaf complexity,
minimum support, regularization, row/column sampling, categorical complexity,
learning rate, and tree counts must all have explicit defensible bounds where
the estimator exposes them. A stopping ceiling is not a final tree count.
Parameters ineffective for a chosen growth or bootstrap policy must not be
advertised as active regularization. No search-space expansion is permitted in
response to outer validation results within this registered study.

Target training scales or losses must be declared in the implementation.
Optuna ranks trials by original-price squared error, and all reported R²,
RMSE, MAE, residuals, and final predictions use original ticket-price units.
Any inverse transformation or clipping follows the predeclared implementation;
it is not selected using outer-fold or competition test results.

## Protected outer evaluation

1. Construct five deterministic outer folds from the training data using outer
   seed **20261020** and identical assignments for every family. The primary
   grouping deliberately excludes `airline` and `duration`: rows sharing
   `flight`, `source_city`, `departure_time`, `stops`, `arrival_time`,
   `destination_city`, `flight_class`, and `days_left` form one group. This
   conservative itinerary grouping blocks exact duplicates and additional
   repeats whose airline missingness or reported duration differs. It can make
   validation harder than ordinary random splitting; it is not an estimate of
   performance on entirely unseen flight identifiers, routes, or dates.
   Stratification starts from flight prefix × cabin class; cells with fewer
   than 100 rows are pooled into `OTHER` within class. Form five within-cell
   price-rank quintiles, breaking price ties by fixed input order, and use the
   resulting strata in shuffled `StratifiedGroupKFold`. Derive these strata
   from the partition being split. Persist outer assignments and seeds.
2. For each outer training partition, create a deterministic five-way inner
   grouped/stratified split using that partition alone. Use the **first two
   validation folds** as two separate inner 80/20 tuning splits. These are
   two fold-pure splits, **not** a complete two-fold CV or complete inner OOF
   prediction set. Each trial fit uses approximately 80% of the outer training
   rows, rather than half. Group constraints can cause small size differences.
   Inner seeds are `20262020 + 100 * family_index + outer_fold`, with family
   indices XGBoost=0, LightGBM=1, CatBoost=2 and outer folds numbered 0–4.
3. Run exactly 12 predeclared Optuna trials per family per outer fold. Two
   queued initial configurations differ only in minimal versus context
   features. A seeded TPE sampler uses five startup trials, including the
   queued trials, before its remaining adaptive suggestions. Each
   trial fits fresh preprocessing and a model independently for both inner
   splits. It may use its inner validation labels both for early stopping and
   trial ranking. These are reused **tuning** observations and their scores are
   not reported as unbiased performance evidence.
4. Rank the 12 trials using the declared aggregation of original-price inner
   squared errors. Select the winning trial's parameters and the declared
   integer aggregate of its two best iteration counts: truncate the median to
   an integer, apply the 100-round floor and the 1,200-round ceiling.
5. Fit fresh preprocessing on the **whole outer training partition**, refit
   the winning estimator at that fixed round count without outer-fold stopping,
   and then predict the untouched outer validation rows. Outer labels never
   choose trial parameters, stopping rounds, feature options, postprocessing,
   or the bounds for this fit.

The nominal budget is 360 inner trial fits plus 15 outer refits for all three
families. Failed fits are recorded with their diagnostics; they are not silently
replaced with extra opportunistic trials. Any necessary implementation repair
must be documented and must not introduce a model choice based on outer scores.

Outer scores estimate the **entire procedure**, including preprocessing,
Optuna selection, and the fixed rule for transferring inner stopping rounds to
an outer-training refit. They do not directly estimate one eventual fixed
hyperparameter dictionary. Historical analyses used training data before this
study; new folds are not claimed to erase that accumulated development history.

## Full-training models and sample-size effects

For each family, rerun the same 12-trial tuner on the complete labeled training
data with two inner 80/20 splits taken from a new deterministic five-way
grouped/stratified partition. Its seed is `20263020 + 100 * family_index`,
with the same grouping and within-partition stratification rules. Choose the
winning settings and median rounds
from **these full-training tuning splits**, rather than borrowing the median
round count from outer folds. Then fit fresh preprocessing and the final model
on all labeled training rows at the fixed selected count.

Using stopping rounds from 80% fits for a 100% refit is an explicit fitting
rule, evaluated at the smaller outer-training scale by the protected outer
procedure. It is not a claim that these are optimal rounds for the larger fit.
Record the two selected counts, dispersion, selected aggregate, and stopping
ceiling hits for every outer and final selection. Discuss unstable counts and
ceiling hits as evidence of uncertainty or possible insufficient training;
do not invent a sample-size correction after observing outer performance.

The final full-training tuning adds 72 trial fits and three final refits.
Persist both the per-outer selected dictionaries and the final full-training
dictionaries, including all defaults material to reproduction, package
versions, feature modes, losses, scaling, seeds, and iteration counts.

## Predeclared ensemble

If all three component procedures complete and remain reasonable validation
candidates, also evaluate and deliver their fixed equal-weight arithmetic mean
on the original price scale: **1/3 XGBoost + 1/3 LightGBM + 1/3 CatBoost**.
Compute its outer predictions from the three protected component predictions.
No weight search, leaderboard-derived weights, or choice of an apparently
best subset using outer scores is permitted. Final ensemble inference uses the
same fixed weights. If a component fails or has a material validation failure,
report that issue and omit this predeclared ensemble; do not silently redefine
its membership or weights.

## Reporting and delivery

Report pooled outer OOF R², original-unit RMSE/MAE, per-fold metrics, mean and
spread, worst fold, training fit scores and train–validation gaps, fit times,
selected rounds, and exact configurations. Include errors by cabin class and
target percentile, top-1% squared-error contribution, worst-error examples,
and observations about rare or conflicting training records. A small gap alone
does not prove generalization, and a high training score alone does not prove
harmful overfitting.

Use paired folds and appropriately qualified uncertainty summaries when
comparing procedures. Bootstrap intervals on OOF predictions are descriptive
and do not capture all variation from retraining and model selection. Do not
equate local R² with the historical user-reported Kaggle R² of 0.95797.

The goal is to deliver distinct, bounded model submissions supported by honest
validation, with their strengths and limitations. There is no requirement that
an interval exclude zero relative to the historical model. Models with failed
validation, invalid predictions, severe unexplained fold failures, or broken
leakage boundaries are not represented as successful candidates. A lower
ranked but defensible model may still be supplied for the user's own Kaggle
comparison, with its measured evidence clearly stated. Selecting a preferred
family after comparing its outer scores introduces some selection optimism;
report every registered family rather than presenting the winner alone as
independently proven superior.

Freeze all deliverable models together before final test inference. Deliver
runnable code, trial records, fold assignments, OOF predictions, diagnostics,
exact final configurations, serialized preprocessing/models where practical,
and Kaggle-ready CSVs. Do not upload to Kaggle without the user's explicit
submission instruction. No public leaderboard score, rank improvement, or
future private-score improvement is invented or guaranteed.
