# Disclosed peer ensemble: incomplete-pipeline replication

The user supplied RF 200 trees/depth 2/70% feature sampling/all rows per tree;
XGBoost 964 trees/depth 9/learning rate .054; CatBoost 1000 iterations/depth 6/
learning rate .139; seed 42 for all; equal arithmetic weights. The peer reports
public R² .96080. This is context for a fixed replication, not a tuning objective.

The user clarified that smooth was used for encoded categorical features and
may have been smooth=auto. This strongly suggests sklearn TargetEncoder, but
its library and exact use remain unconfirmed. Register TWO fixed pipelines:
RF/XGB one-hot controls and RF/XGB sklearn TargetEncoder(smooth='auto',
target_type='continuous') interpretation. Share the same native CatBoost.
The complete feature list, remaining model parameters and target scale remain
unknown. Use existing compact context features as an explicit assumption.

TargetEncoder training inputs use five INNER GROUP cross-fitting folds, seed42,
inside each fitting partition (same conservative itinerary grouping as outer).
Fit mappings only on inner fitting labels and transform the inner held-out rows.
Fit a separate full-fitting-partition map solely for outer validation/inference.
Never apply same-row target maps to model training. Save all inner indices and
verify group separation. Numeric medians/feature maps fit on outer training
only; they are unsupervised. Price/10000 is the target encoding/model scale.
Do not claim an exact peer-pipeline replication or that its .96080 is reproduced.

RF bootstrap=False gives every tree all fitting rows; max_features=.7 samples
features per split (the peer's sampling granularity is unknown). Depth 2 implies
4 leaves. XGB uses CPU hist, depthwise, max_leaves512 (redundant with depth9),
common defaults explicitly set: child1/L2=1/L1=0/gamma0/row+column samples1,
max_bin256. CatBoost uses explicit common CPU settings: symmetric depth6,
L2=3/random_strength1/Bayesian temperature1/rsm1/border254/onehot2. Its CTR
complexity1 and table limit512 are bounded safety/runtime assumptions, not
known peer values. All resolved constructors and CatBoost runtime parameters
will be saved. No neural networks; price/10000 squared error, inverse10000,
clip individual predictions at zero then fixed equal arithmetic blend.

No hyperparameter or weight search. Five existing training-only grouped,
prefix/cabin/target-stratified folds (seed20261020) evaluate each fixed model
and blend, with fresh feature fitting within each fold. Duplicate itinerary
groups excluding duration/airline stay intact. Report pooled/fold R², RMSE,
MAE, actual composite training gaps, cabin/tail errors and same-row earlier
controls. These rows/folds have informed prior development and are not pristine.
Fixed configurations require no inner selection; no early stopping or count
selection. Exactly 25 outer fits +5 full fits =30 research model fits.
TargetEncoder mapping fits are counted separately: 6 per encoded model fit
(5 cross-fitted +1 full map), totaling 72 encoder fits, not extra regressors. Delivery is an
explicitly requested replication, including underperforming comparisons.

Freeze sources, versions, schemas, final bundles, exact constructors, counts,
weights and price policy together before the new phase reads Kaggle test once.
Use test exclusively for final predictions and schema/ID/count/finite checks.
Emit 5 component CSVs +2 equal blends, each RF+XGB+shared CatBoost.
No search between encodings or weights based on outer/public scores; supply both. No model revisions after test access,
no automatic Kaggle uploads, no ranking or .96080 guarantee. Keep all old
frozen experiments unchanged.
