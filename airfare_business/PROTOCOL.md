# Business-class specialist: separate prospective registration

Pre-fit synthetic API diagnosis: XGBoost3.1.1's callable-metric path requires
`_estimator_type`, which sklearn1.8 removed from its regressor mixin. Set the
standard value `regressor` on new specialist instances only. This restores
metric dispatch without changing a model parameter or installed library. The
custom stopping metric is verified on synthetic data before competition fits.

This is the next experiment proposed before the user authorized continuing.
Registered after direct CatBoost64 tuning completed its primary outer scores,
before any specialist is fitted. The direct-tuning result was negative; its
search is not expanded. All earlier source files and fitted artifacts remain
unchanged. This study uses no competition test data or public scores to tune.

## Hypothesis and fixed structure

Training-only diagnostics attribute about 97% of the existing blend's squared
error to Business fares. A global model may compromise between fare regimes.
Test one bounded, separately trained Business-only XGBoost specialist, with
explicit shrinkage toward the existing global XGBoost. No rows are selected by
their target price, removed or relabeled. Cabin class is an observed predictor.

Candidate prediction: for Economy, retain the global XGBoost prediction; for
Business, use **0.5 global XGBoost + 0.5 Business specialist**, with each component
clipped below at zero first. The fraction is fixed before fitting, never tuned.
Also evaluate one fixed thirds blend of this augmented XGBoost, the existing
LightGBM and existing CatBoost64. Do not combine it with newly tuned CatBoost64,
optimize weights or select another subset. It changes only the XGBoost branch of
the frozen best blend. Report every registered result, favorable or unfavorable.

## Validation and parameter search

- Same primary outer seed 20261020, same itinerary groups and five folds as the
  frozen control. Group key excludes airline/duration, includes cabin class.
- For each outer partition, use the old **partition-local** XGBoost-selected
  configuration, feature option and stopping records. Never initialize an outer
  fit with the full-data dictionary. Reconstruct its outer fit and require its
  validation predictions to match saved original XGBoost OOF predictions.
- Inner splits are the first two of five grouped splits, seed 20262020 + i,
  matching that local global-XGBoost selection. Reconstruct the two inner global
  fits at their originally selected per-inner stopping counts. Refit feature
  mappings independently on each inner fitting partition. Reuse these cached
  global inner validation predictions for all specialist trials in this scope.
  The global parameters have used these inner tuning labels; the resulting
  scores remain tuning scores. Outer labels are not used to choose either model.
- Specialist features: existing **context** pipeline, fixed; each mapping fitted
  on Business fitting rows alone. Original-price target, scaled by 10000 for
  fitting, same squared-error objective, model seed 42, four threads.
- Twelve TPE trials, five startup trials including two generic queued settings.
  Bounds: depth 4–6; leaves 16/24/32 capped at 2^depth; learning rate .025–.065
  log scale; row sample .75–.95; column sample .70–.95; L2 5–40 log scale;
  L1 .05–1.5 log scale; split gain .01–.20 log scale; min child weight 12–60.
  Existing hist/depthwise/native-category settings remain unchanged.
- Generic initial A: depth6/leaves32/lr.045/row.85/col.85/L2=12/L1=.2/gain.04/
  child20. Initial B: depth5/leaves24/lr.035/row.85/col.85/L2=20/L1=.2/gain.04/
  child30. No settings learned using all labels are queued.
- Both trial ranking and early stopping evaluate the fixed half-global,
  half-specialist prediction on **Business inner validation rows**. Custom RMSE
  stopping uses the cached global prediction, component clipping and inverse
  scale explicitly. No outer prediction enters this custom metric. Restricting
  the objective to Business is equivalent to ranking full inner errors because
  Economy predictions are unchanged across specialist trials.
- Stopping cap 1200, patience70; full outer-specialist refit count is integer
  median of two inner best counts, floor100, ceiling1200. No outer stopping.
  Record best counts and actual trained counts/true cap hits for inner fits.
- Specialist sampler seed 20274020+i; full-data sampler seed20275020. Full-data
  inner splits use original XGBoost full-data inner seed20263020; reconstruct its
  two local inner fits from the original full-training selected settings/stops.
  Run a fresh 12-trial specialist search on these inner splits, then refit all
  Business training rows. Do not transfer outer counts to full-data training.

Primary/full budget: **72 specialist trials and 167 fits**: 144 specialist trial
fits, five outer specialist refits, one final specialist refit, ten reconstructed
outer-scope global inner fits, five global outer refits and two full-scope global
inner fits. The existing final global model is reused without retraining.

## Conditional second split seed

Before any specialist scores, register a conditional sensitivity check. Run it
only if primary augmented-XGBoost pooled R² improves over global XGBoost by at
least .0005, OR its fixed thirds blend improves over the frozen best by .0002
and wins at least three of five paired folds. This only allocates computation;
it is not a significance or CSV eligibility test.

Secondary outer seed20261027, all five folds; global inner/tuner seed20276020+i.
Rerun the original 12-trial XGBoost procedure inside each new outer partition,
then reconstruct its two inner global fits and run the same 12-trial specialist
procedure with sampler seed20277020+i. Refit both models on that partition and
evaluate global versus augmented XGBoost. Do not reuse primary global settings
across these different partitions. Do not create a secondary three-family blend
without new matching LightGBM/CatBoost predictions.

Additional budget: **120 trials, 260 fits**: 120 global trial fits + five global
outer refits +120 specialist trial fits +five specialist outer refits +ten global
inner reconstructions. Maximum full study: **192 trials and 427 fits**. No extra
seed search or parameter-space changes follow the results.

## Interpretation and delivery

All observations have been used in earlier development. These reused primary
folds and the additional registered seed provide developmental robustness
evidence, not a pristine holdout or unbiased proof after accumulated selection.
Report pooled/per-fold R², RMSE/MAE/bias, paired deltas, measured component and
augmented-model training gaps, cabin/price-tail/rare-flight diagnostics, stopping
dispersion, all trial records and exact final parameters. Group-bootstrap
intervals condition on fitted OOF predictions and omit retraining uncertainty.

No neural models, pseudo-labeling, external labels, test-assisted mappings,
public-score optimization or target-derived inference gates. Economy predictions
must be exactly unchanged; validate this property on training OOFs and synthetic
inference. A specialist may perform worse; keep the frozen previous best.

Freeze the new specialist and the three existing final components, source hashes,
package versions, feature mappings and fixed mixture rules before final test
inference. Output two separately named CSVs: augmented XGBoost and its fixed
thirds blend. If implementation or validation fails, report it without presenting
the affected outputs as successful. Otherwise bounded, validation-supported
comparison candidates may be supplied even without improvement, with actual
evidence explicit. No model is guaranteed free from overfitting or promised a
Kaggle rank. No Kaggle upload is performed.
