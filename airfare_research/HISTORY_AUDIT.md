# Historical evidence audit

Scope: documentary audit of the supplied handoff. Historical values below were read from its reports/configuration artifacts; they have not been independently reproduced in this audit. Kaggle test features and preserved test predictions were not opened.

## What the strongest evidence establishes

| Historical procedure | Evidence | Interpretation |
|---|---|---|
| Original global XGBoost 850 | User-reported public R² 0.95797; Phase6 duplicate-blocked OOF 0.9295428902 | Exact unchanged control and fallback; public score and OOF assess different samples/protocols. |
| Phase4 reference + stronger regularization | Optimistically stopped/scored on the same five validation folds: 0.9291593 vs 0.9291712 | Difference 0.0000119 is negligible beside fold SD about 0.0126; no credible improvement. |
| Phase4 engineered bundle | OOF 0.927201 / 0.926087 | Combined feature expansion failed; does not identify which individual feature helped or hurt. |
| Phase5 tailored class split | Development +0.003020; separate holdout −0.002584, CI [−0.008839,+0.003874] | Development gain reversed. Economy improved, but Business error dominated total squared error. Holdout has been consulted and is not pristine for another selection cycle. |
| Phase6 coarse / fine target history | OOF −0.006650 / −0.008684; every fold worse | Strong negative evidence for these exact encodings; larger training scores and wider gaps support category overfitting/representation problems. |
| Phase6 tail-up / tail-down weights | OOF −0.001385 / −0.000367 | Optimizing expensive-ticket RMSE in isolation did not optimize overall R². No submission justified. |

## Why subsequent experiments did not establish improvement

1. The Phase4 reference was refitted at 640 rounds, replacing the original fixed 850-round procedure. Its score cannot be presented as an unchanged baseline reproduction. Early stopping used the same validation labels later scored; the median stopping count ranged over highly unstable folds (170–913 for the reference). A median from models trained on 80% of observations has no automatic optimality for a full-data refit.
2. Multiple settings changed together in most early comparisons. The public difference between XGB750 and XGB850 cannot isolate tree count, and the LightGBM 450/550 comparison changes learning rate (.055/.035). The older claim that log-target experimentation “isolated” a transform is too strong: its depth, child weight, learning rate and penalties also changed.
3. Cabin splitting discards shared estimation and reduces the effective sample size of already sparse, expensive Business groups. The reports show the resulting Business deterioration outweighed Economy gains; this is a supported diagnosis of the observed errors, not proof of the precise mechanism.
4. Tail-price observations dominate squared error: Phase6 top 1% accounted for 37.35% of total error; they were all Business, strongly concentrated in a few prefixes. Merely increasing their training weights is not a new source of predictive information, and its reported gain in top-tail RMSE came with deterioration elsewhere.
5. Cross-fitted target encoding can be label-leakage safe yet still add noisy sparse-group estimates. Differences between inner-fold training encodings and full-outer-training validation encodings may add representation shift. These remain hypotheses to inspect in code, rather than proven explanations for Phase6.
6. Early phases generated many submissions without local validation. Choosing the best public result introduces public selection bias. Historical public results remain descriptive references; they cannot supply blending weights, parameter choices or evidence about private rank.

## Validation recommendations before candidate research

- Keep the exact fixed850 feature pipeline and parameters on every comparison. Resolve implementation defaults and package versions from scripts rather than from incomplete JSON dictionaries.
- Predeclare a new primary five-fold target-stratified split blocking exact raw predictor duplicates. Verify group disjointness, fold sizes, class/prefix composition and extreme-fare coverage; decile stratification does not guarantee balance among rare high-fare prefixes. Large fold variability warrants inspection of assignments, including implementation behavior of shuffled StratifiedGroupKFold, before interpreting scores.
- Compare the same control under ordinary stratified splitting, semantic-duplicate blocking after deterministic feature canonicalization, and unseen-flight stress splitting. Semantic grouping addresses duplicate rows obscured by missing airline/duration; unseen-flight splitting answers a harder extrapolation question and should not silently replace the primary random-sample estimand. No calendar/date feature is documented, so booking lead days alone do not establish a valid temporal ordering.
- Candidate stopping decisions belong in a further inner split. Outer labels score the frozen fitting procedure, never stopping rounds. Fixed850 needs no early-stopping selection and remains a fully valid control.
- Predeclare a small hypothesis family and a material-effect gate. Use paired OOF errors, per-fold deltas, tail/class diagnostics and group bootstrap intervals. OOF bootstrap intervals condition on fitted models and do not capture all retraining/selection uncertainty. Repeated outer splits or nested selection strengthen follow-up evidence but do not turn historically inspected observations into a brand-new independent dataset.
- Prefer credible, stable pooled original-price R² improvement. Report uncertainty and adverse subgroup changes. Neither CV nor the dated public cutoff guarantees top five.

## Genuinely new hypotheses worth testing if training-only diagnostics support them

1. **Duration reliability:** compare masked-observed-duration reconstruction using the existing hierarchy against stronger feature-only flight/route/departure-arrival matching, smoothing/fallbacks, and an imputation confidence/count feature. Evaluate prediction changes separately for missing and observed duration. Change one component at a time.
2. **Representation rather than target memorization:** inspect the exact OHE/sparse/numeric representation for redundant airline/prefix/interaction indicators and any ambiguity between observed zero and missing values. Test one minimal representation ablation with the unchanged850 model only if the implementation audit finds a concrete concern.
3. **Regularized native categoricals:** one bounded native-category model could address sparse indicator fragmentation if residual diagnostics show unseen/rare identity failures. Earlier CatBoost underperformance is negative evidence requiring a specific structural reason; it does not justify repeating its old configuration.
4. **Stable signal across models/seeds:** only consider a predeclared small blend when independent OOF errors show complementarity; evaluate its fixed or inner-fitted weights on untouched outer folds. Do not repeat unvalidated public-score blends.

## Reconciliation and evidence limits

- Main master prompt supersedes the provenance prompt's neural-network suggestions, older 500-tree statements, and old prohibition on local quality testing. Current rules prohibit all neural models, require finite complexity bounds, and explicitly authorize rigorous training-only validation.
- The historical 314 duplicate rows, 310 repeated-feature groups and 624 total members are arithmetically compatible (624−310=314). Matching-target duplicate count 279 and conflicting-group count 35 are different definitions; do not subtract them and call the result a verified count of conflicting groups without recomputation.
- The history reports 0.96129 as a dated rank-five cutoff and later self-reported rank about12. The current user screenshot shows rank13 S-Rank Hunters, rank5 0.96181, and rank4 0.96325. These are different snapshots; none should guide model tuning.
- The handoff contains neither exact Phase4 public scores nor independently verifiable Kaggle/private scores. Do not fabricate missing values.
- Competitor metrics (OOF R² .81648, RMSE95.0741) have unclear scale/protocol. Their partial configuration omits learning rate and other fields. The min100 versus max100 median-round snippets contradict each other. None is a reproducible control.

## Fully read source files

All text of the following files was read, including all 676 lines of the current master and all 1,126 lines of the original provenance prompt:

- CODEX_MASTER_PROMPT.md
- README_START_HERE.md
- original_master_prompt.txt
- EDA_EXPERIMENT_HISTORY.md
- kaggle_train_only_EDA.md
- TARGETED_EDA.md
- PHASE2_README.md
- PHASE3_README.md
- PHASE4_REPORT.md
- PHASE5_DECISION_REPORT.md
- PHASE6_DECISION_REPORT.md
- TRAIN_ONLY_EDA.json
- phase1_exact_configs.json
- phase2_exact_configs.json
- phase3_exact_configs.json
- cv_results.json
- weight_results.json
- PHASE4_CV_RESULTS.csv
- PHASE6_CV_RESULTS.csv

The independent script audit is delegated separately. No modeling experiment was run for this documentary audit, and no handoff source was modified.
