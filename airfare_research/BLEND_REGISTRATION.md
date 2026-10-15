# One fixed blend, registered before computing blend predictions or scores

Exploratory follow-up within structural representation research. Native850 alone
underperformed but baseline/native OOF residual correlation is.93415, and covariance
101676553 is below both residual variances107457581/110248520. Opposite-sign errors
occur18.26%. Thus variance reduction is a plausible reason to consider an ensemble.
This uses exploratory OOF diagnostics; it is not independent confirmation.

Freeze exactly50% original sparse OHE XGB850 +50% native categorical XGB850.
No weight fitting/grid, no inclusion of duration/prefix candidates, no target
transforms/calibration. Both constituent training seeds42; all individual
parameters, feature rules and nonnegative price postprocessing are unchanged.
This is two finite depth7/48leaf850-tree models (1700 total trees); justified
only if it passes the same >=.001 pooled R² / >=4 positive folds / pairedCI gate,
then unchanged repeat seed20261016. Report additional multiplicity sensitivity
for this exploratory family and all stopping additions. No leaderboard fitting.

If it fails any promotion gate it is not a new Kaggle submission. If accepted,
full-data models both keep850 exactly, avoiding a sample-size-dependent round
selection rule. No further blends, weights, or mixtures will be tried in this run.
