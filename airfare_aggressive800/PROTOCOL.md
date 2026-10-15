# One exploratory 800-tree Business candidate

Prospective registration before any new fit or validation score in this phase.
The user explicitly authorizes one more aggressive portfolio candidate even if
validation is weaker than the preferred submissions. This overrides earlier
CSV promotion requirements for this single exploratory candidate. It does not
authorize neural networks, unbounded complexity, test-assisted development,
invalid outputs or claims of a guaranteed public/private score or rank.

## Fixed hypothesis and recipe

Business fares account for approximately 97% of the earlier blend's training-OOF
squared error. The user reports public R² .95881 for Business-augmented XGBoost.
That reported outcome and the user's portfolio preference motivate the family
being explored; it is not an untouched model-selection history. No new public
score is available for this candidate and no parameter/weight search against
public scores is performed.

Keep the existing global XGBoost procedure and its 50/50 Business combination.
Replace only the Business specialist with one fixed, more expressive profile:

- **800 trees**, depth **7**, at most **48 leaves** (also <=2^depth).
- Learning rate **.045**, minimum child weight **8**, L2 **3**, L1 **.05**,
  split gain/gamma **.01**, row sample **.95**, column sample **.90**.
- Existing hist/depthwise/native-category settings: max_bin192,
  max_cat_threshold32, max_cat_to_onehot4, squared-error objective, seed42,
  four CPU threads. Existing context features, fitted on Business fitting rows.
- Target price/10000; invert and clip each component below at zero.
- Economy: original global prediction. Business: **.5 global + .5 new specialist**.
  Mixture weights remain fixed. No other three-family blend or standalone output
  is generated in this phase.

The profile deliberately relaxes regularization and increases Business tree
capacity relative to the current specialist. It changes several settings as a
single aggressive profile; do not attribute differences causally to tree count.
800 is a predeclared final count, identically used in every validation/full fit.
No early stopping, Optuna, trial search, alternative counts or additional seeds
are used. Fixed rounds avoid outer stopping and sample-size round-transfer
choices. The purpose is to expose higher capacity, not to guarantee overfitting
or engineer a particular public leaderboard score.

## Training-only evaluation

Evaluate all five folds under each existing grouped/stratified outer seed,
20261020 and20261027. Use exactly matching previous Business-study global and
augmented OOF controls; those partitions/observations have already informed
development and are not pristine holdouts. Do not select a seed after results.

For each outer partition, reconstruct only its original partition-local global
XGBoost using saved `global_selected.json`; verify raw/target input hashes and
reproduce saved global OOF predictions. Never use the full-training dictionary
inside an outer fit. Fit fresh global preprocessing on the outer-training rows
and fresh specialist preprocessing on its Business subset. No outer labels
choose hyperparameters, rounds, mixtures or preprocessing statistics. The new
specialist has no fitted cross-partition settings: its entire dictionary is
fixed above before scores.

Record actual composite training R², original-price R²/RMSE/MAE, paired folds,
training–validation gaps, Business/Economy and highest-price-tail errors, rare
flight support and worst errors. Economy must be exactly unchanged. Report500
whole-itinerary-group bootstrap resamples as conditional OOF summaries, not
retraining/selection uncertainty. Report weaker validation plainly; no absence
of overfitting is promised.

Budget **21 model fits**: ten reconstructed global outer fits, ten new specialist
outer fits, one final specialist fit on all6380 Business training rows. Reuse
the unchanged final global454-tree bundle; no full global retraining. No extra
candidate, hyperparameter changes or post-hoc calibration after evaluation.

## Freeze and one-file delivery

Save both full-data bundles' configurations/hashes, feature schemas, versions,
source hashes, exact profile, mixture and clipping policy before test access.
Previous phases have accessed the test, so this establishes a new boundary,
not a claim the test has never been opened. This phase opens test content only
once for final inference, after all fits and the complete freeze. No test EDA,
prediction-distribution inspection, test-derived mappings or model changes
follow. Use an exclusive inference-start marker and exclusive CSV creation.

Deliver exactly one new file, `submission_aggressive_business_xgb800.csv`, with
3898 rows, exact sample IDs/order, schema id/price, finite nonnegative prices.
Label it exploratory with actual validation evidence and preserve earlier
submissions byte-for-byte. The user will upload it manually. No Kaggle upload
or assumption about the competition's unverified final-selection limit is made.
