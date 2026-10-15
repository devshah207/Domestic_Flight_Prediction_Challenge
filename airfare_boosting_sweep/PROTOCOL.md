# Broad high-round boosting study: registered before research fits

The user's latest instruction explicitly requests multiple aggressive, high-tree
boosting candidates tuned by Optuna, including exploratory models with poorer
validation. This supersedes the earlier restriction to one risky submission and
the handoff's requirement to withhold every weaker candidate. It does not
authorize leakage, neural networks, fabricated results or a promised public score.

## Information boundary and hypothesis

Use labeled training data only throughout development. Do not read competition
test data, old submission values or test-prediction summaries. Preserve existing
experiments. The broad hypothesis is that different boosting growth policies,
categorical treatments and smaller learning rates may exploit a high-round
budget differently from the previous fixed Business-only specialist. These are
global models, with the same compact context features supported by the existing
training-only EDA. No target encodings or extra supervised features are added.
Original-price squared error is the tuning objective; targets are price/10000.
Predictions are multiplied by 10000 and clipped at zero before any fixed blend.

## Registered procedures and finite budget

Eight procedures: XGBoost histogram/native categorical; LightGBM GBDT;
LightGBM DART; CatBoost SymmetricTree; CatBoost Lossguide; sklearn
HistGradientBoosting; sklearn GradientBoosting; sklearn AdaBoost.R2.
This covers eight common procedures, not literally every boosting algorithm.

Every trial proposes exactly 800, 1100 or 1400 rounds jointly with learning
rate and bounded depth/leaf/support/regularization settings. The explicit user
request motivates the high-count floor; it is not evidence that this floor is
optimal. No early stopping selects fewer rounds. Libraries can nevertheless
terminate intrinsically (AdaBoost failure/perfect-fit rules or LightGBM no
useful splits); record actual counts and never call the requested cap the count.
All exposed structural complexity bounds are finite. Exact search spaces,
constructor parameters and two queued starting profiles are in models.py.

Optuna uses seeded TPE, four startup trials for eight-trial procedures and three
for six-trial procedures. There are eight trials per scope for XGBoost, LGBM
GBDT and both CatBoost procedures; six for DART, histogram/classic gradient
boosting and AdaBoost. No expansions based on outer or public results. There
are 56 trials per set of scopes, six scopes each: **336 Optuna trials** and
**48 outer/full refits**, totaling **384 research model fits**, plus separately
reported API smoke fits. No hidden restarts or opportunistic additional seeds.

## Validation and fitting procedure

Reuse the five training-only grouped/stratified outer folds (seed20261020)
implemented in airfare_fresh/study.py. Keep duplicate itinerary groups intact,
including rows with different missingness/duration variants. The rows and folds
have informed previous research; they are not a newly untouched holdout.

Inside each outer training partition, generate five grouped/stratified folds
from that partition alone, seed20264520+outer_index. Use the first inner 80/20
split for all trial ranking. The single inner split is an efficiency compromise
and may make parameter selection noisy. All families share these inner indices.
All airline maps, duration imputers, category vocabularies, rare-category maps,
numeric medians and one-hot encoders fit only on the corresponding fitting rows.
Cache transformed matrices only within this partition. Outer labels never rank
trials, select counts or change parameters. Fit the selected configuration on
the whole outer training partition, with fresh preprocessing, before evaluating
the outer validation fold. Save inner/outer row positions and group-disjointness.

Rerun exactly the same tuner on full labeled training data using first inner
80/20 split with seed20265520, then refit all training rows at the selected
count. Do not transfer the most flattering outer dictionary to final training.
No sample-size/tree-count correction is introduced after observing scores.

All procedures use the existing context feature set. XGBoost and LightGBM
receive fold-fitted native categories. CatBoost receives strings with tuned
one_hot_max_size in {2,64}, max_ctr_complexity1, bounded CTR tables. Symmetric
trees have finite depth (their implicit leaf cap is 2**depth); min_data_in_leaf
is not advertised as active for this growth policy. Lossguide has both finite
depth and explicit leaf/support bounds. Classic GB/AdaBoost use fold-fitted
sparse one-hot encoding and numeric median imputation. Histogram GB uses
fold-fitted top127 category levels plus one OTHER category (max128 levels),
native categorical splits and numeric median imputation. It never interprets
arbitrary flight/category codes as ordered numerical measurements.

## Reporting, controls, and delivery

Report all eight procedures, including failed/weak ones: pooled OOF original
R²/RMSE/MAE, paired folds, training gaps, actual counts, time, cabin/tail/rare
flight errors, and descriptive grouped-bootstrap uncertainty. Compare to the
unchanged earlier global XGB and Business-augmented OOF controls on exactly the
same folds/rows. Earlier baseline850 reproduction remains preserved in
airfare_research; its different split scores are not a same-fold comparator.

Two fixed blends are registered before fitting: equal thirds XGBoost + GBDT +
symmetric CatBoost, and equal thirds XGBoost + DART + symmetric CatBoost.
No weight search or outer/public-score-dependent subset choice. Also report
their actual composite training scores. Deliver all successfully completed
individual candidates and these fixed blends even if they are weaker than
controls, clearly labelled as exploratory under the user's request.

Freeze every final bundle, feature schema, exact configuration, source and
version hash, prediction policy and blend weight together before final test
inference. Then read test once, produce all CSVs in exact sample ID order and
check schema/count/IDs/finite nonnegative predictions. Never alter models or
weights based on test rows or predictions. No automated Kaggle uploads.

Public/private R² and ranking are unknown. Reused outer folds, many families
and one inner tuning split limit confidence. A wider training gap can flag
risk; neither high training accuracy nor high tree count promises a public
gain. Keep the previous 0.95881 submission available.
