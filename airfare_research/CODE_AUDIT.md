# Historical code audit — training-only, before new experiments

Scope: fully read `CODEX_MASTER_PROMPT.md` (676 lines) and all seven packaged Python files. No Kaggle CSV, sample CSV, or preserved prediction CSV was loaded by this audit. Historical code is unchanged. This is a source audit, not a measured experiment.

## Exact control procedure

The public-scoring procedure is `generate_phase3_boosting.CONFIGS['xgb_850_d7_regularized']`, constructed by `build_model`, with `TrainingOnlyFeatureBuilder` and `make_ohe_features` from `generate_submissions.py`. Keep precisely 850 rounds and seed 42. Its explicit parameters are:

```json
{
  "objective": "reg:squarederror", "tree_method": "hist", "n_estimators": 850,
  "max_depth": 7, "max_leaves": 48, "learning_rate": 0.03,
  "min_child_weight": 12, "subsample": 0.8, "colsample_bytree": 0.85,
  "reg_lambda": 15.0, "reg_alpha": 0.6, "gamma": 0.08, "max_bin": 192,
  "random_state": 42, "n_jobs": 6, "verbosity": 0
}
```

Operationally use at most four threads to match this machine's quota; record that adaptation. Target is `price / 10000`; predictions are multiplied by 10000 and clipped at zero. IDs are excluded. Clipping at zero must also be applied consistently for outer-validation metrics if it is part of the final procedure.

Feature builder:

- Parse flight prefix by whitespace split, trailing numeric flight number via regex. Fill missing airline by the observed training-prefix modal airline; retain missing flag.
- Duration median hierarchy, minimum observed count in parentheses: flight/source/destination/stops (2), flight/source/destination (2), flight/stops (2), flight (2), source/destination/stops/class (5), source/destination/stops (5), source/destination (5), stops (5), global observed median. Retain missing flag.
- Categorical features: airline_fixed, flight_prefix, source_city, destination_city, departure_time, arrival_time, stops, flight_class, route, route_stops, route_class, prefix_class, dep_arr. Ordinary OHE `handle_unknown='ignore'`.
- Separate flight OHE uses raw flight with `min_frequency=12`, `handle_unknown='infrequent_if_exist'`; this is the actual XGB rare-flight implementation. The feature builder's `flight_rare` column is used by LightGBM, not by original XGB.
- Numeric features in their exact historical order: duration_filled, duration_missing, days_left, flight_number, airline_missing, stops_numeric. CSR matrix cast to float32.

Every feature builder and encoder must be fitted anew inside each model training partition, including the inner stopping partition. Original main functions fit them globally because they originally performed no CV; that is valid for full-data refit but cannot be reused as-is for validation.

## Reproducibility and parameter-default limits

The archive contains exact explicit dictionaries but no package-version pins, package snapshots, serialized baseline model, or resolved booster configuration. Therefore matching explicit settings and training data is possible; identical historical predictions cannot yet be guaranteed. Keep the preserved best CSV unchanged as the historical fallback.

Current measured installation metadata: Python 3.12.14 (parent inspection), xgboost-cpu 3.1.1, lightgbm 4.6.0, catboost 1.2.8, sklearn 1.8.0, pandas 2.2.3, numpy 2.3.5. `xgboost-cpu` supplies the `xgboost` import; `importlib.metadata.version('xgboost')` alone misleadingly reports no distribution.

XGB defaults such as `base_score`, `boost_from_average`, `grow_policy`, device and sampling must be captured using the fitted booster's `save_config()`. In particular, newer XGB versions estimate intercepts automatically; very old defaults used 0.5. Do not invent the unknown historical version or silently force an alleged original intercept. Record current resolved behavior and call the result a faithful explicit-procedure reproduction, with this limitation.

`max_leaves=48` and `max_depth=7` are both finite. The CPU `hist` grower supports leaf caps; do not replace default depthwise growth by lossguide on the assumption that leaf caps require it. Verify actual trained leaf/depth maxima from JSON trees. Changing growth policy would be an additional model variation. Sparse OHE uses omitted values as missing in XGB; this is part of the historical representation. Densifying it changes numeric zero semantics and must not be called exact reproduction.

## Why the historical comparisons were unreliable or failed

1. **Phase4 uses outer labels for stopping.** `run_phase4.py:127–143` fits against and scores the same fold; best iteration is a selected minimum of a noisy validation curve. Its result is an optimistic estimate of that selection procedure, and the same fold should not provide an unbiased score. Independently select rounds inside outer training, refit outer training, and score the outer fold once.
2. **Phase4 changed the control.** Lines 253–259 replace 850 with the median stopping count (reported 640). Its final `xgb_reference_d7` submission is not the original 850-tree benchmark. The code does use `np.clip(best_i,100,1450)` correctly; the contradictory competitor `min`/`max` anecdote does not apply to this script.
3. **Phase4 bundled fourteen features.** Five categorical and nine numerical interactions changed simultaneously. A failed bundle cannot show which component hurt. Larger feature count also changes random column subsampling allocation. Any new structural change needs one focused ablation.
4. **Phase4 alternate/blend selection is optimistic.** Champion, alternate, and whether the 50/50 blend passes are selected from the same OOF results. Fixed weights do not remove selection bias. Its small reported blend improvement is not independent confirmation.
5. **Phase5 class splitting reduces support.** Separate cabin models lose shared learning and have smaller samples; the tailored variant changes depth, minimum child weight, L2, round policy and fitting population. Its development gain reversed on the already-consulted holdout. Never claim the holdout is now pristine or reuse split models without a new structural reason.
6. **Phase5 inner feature preprocessing is not fully isolated.** `run_phase5.py:24` fits builder/OHE on all outer training rows, then `:34` subsets that precomputed matrix into inner train/stopping rows. This uses inner stopping *features* in imputation/category fitting. Outer labels/features remain protected, so it does not invalidate outer-test independence, but it violates the requested strict fold-pure inner fitting and can affect stopping choices. Refit feature transformations within the inner fitting split.
7. **Phase6 supervised encodings are outer-pure and cross-fitted.** Class/global priors, counts and means are fitted within each encoder training subset; identical original-feature groups are blocked inside inner folds. No direct outer-target leakage found in the source. Cross-fitting creates a training/inference reliability difference (75% encoder fitting samples for training rows, 100% for outer validation), and sparse hierarchical priors can add noise; those are plausible risks, not proved causal explanations. All historical coarse/fine encodings degraded all folds, so do not simply repeat them.
8. **Tail weights change the objective.** `weight_loss_study.py` recomputes thresholds within outer training and normalizes weights correctly. It trades total SSE against tail SSE; better tail RMSE alone does not improve original-price R². Both variants failed their own gate.
9. **Bootstrap uncertainty is conditional.** Historic paired bootstrap resamples aggregate duplicate-group SSE differences with a fixed original denominator; it does not refit models and does not capture training-sample, fold-selection or experiment-selection variability. Treat its intervals as approximate paired uncertainty only. Add repeated training-only folds and/or flight-cluster resampling sensitivity where feasible.
10. **Temporal claims are unsupported.** `days_left` is booking lead time, not an actual calendar timestamp. A days-left block is a lead-time extrapolation stress test, not proof of future-date forecasting performance. Exact-duplicate grouping does not block flight IDs; held-flight validation asks a different, usually harder question. Compare these as sensitivity analyses, not directly comparable performance claims.
11. **No direct Kaggle-test leakage found in packaged scripts' executed structure.** Their guarded main functions load test only after their own selection/refits. Avoid executing historical mains during development: they can open test and write many files. Merely importing Phase5/Phase6 also creates a results directory, so prefer extracting/importing original feature definitions without those side effects.
12. **Archive relocation breaks standalone main paths.** Phase4/5/6 infer original historical `parents[1]` and `phase4` layout, while the ZIP flattens files. Those main entry points also expect `train(1).csv`. Reusable research code should accept explicit train/test/sample paths and never mutate historical files.

## Small preregistered hypothesis family

H1, **duration imputation fidelity**, conditional on training-only masking evidence. First mask observed duration within inner training with predeclared 20% masks; learn maps without masked values; evaluate MAE and bias by route/prefix/stops. Consider only one supported alternative (e.g. restrict flight medians to compatible route/departure/stops and require adequate group support, or preserve unknown for XGB rather than noisy broad medians). Keep all original columns and the 850-tree estimator. This separates imputation from arbitrary extra interactions. Masking score improvement is a prerequisite, not proof of price-prediction improvement.

H2, **native XGBoost categorical partitions**, a genuinely untested representation. Keep the original feature builder, six numeric columns, 850 trees, d7/leaves48 and original loss/regularization. Replace sparse OHE with training-fitted pandas categorical levels, grouping flight at count 12 to match historical information. Bound `max_cat_to_onehot=1`, `max_cat_threshold=16`; map unseen levels to missing with an explicit train-only mapping. Partition splits can combine airline/prefix/route subsets in one split, potentially improving rare expensive-prefix behavior without spending depth on individual indicators. Historical CatBoost/LightGBM trials changed both representation and learner; they do not answer this ablation. Native categorical gain remains a hypothesis until measured on identical folds.

H3, **round-selection procedure**, at most one nested variant of the strongest supported structural representation, while the fixed original850 remains control. Use a predeclared cap1200 or1450 and patience75, two or three duplicate-blocked inner stopping partitions, aggregate selected rounds, then refit complete outer training. Record selected count dispersion and cap hits. Compare fixed850 and selected-round procedure on untouched outer labels. More rounds or earlier stopping is not inherently better. Full-data round choice needs a declared sample-size stability check rather than blindly copying outer-fold medians.

Do not add target history, tail weights, generic class splitting, broad transformed-target searches, dozens of random parameter draws, neural networks, or public-score-based blend weights. If candidate errors complement the control, a predeclared 50/50 blend can be evaluated on a fresh repeated split or nested inner OOF; choosing the blend after seeing outer errors remains exploratory.

## Validation and compute recommendation

Before new candidates, compare unchanged control on (a) random target-stratified folds, (b) exact-feature-duplicate-grouped, target-stratified five folds, and (c) held-flight five folds as a stress test. Exact-duplicate grouped CV is the main estimate for repeated-market observations; held-flight tests unseen identities, and their scores have different meanings. If costs warrant, use one predeclared held-flight holdout instead of all five but label it clearly. A fixed distinct second seed provides confirmation without treating Phase5 holdout as unused data. No true external untouched holdout remains.

At 15,589 rows, 850 rounds, d7/leaves48, four threads and 32GB RAM, sparse XGB is modest. Historical Phase4 records report 36.5sec for five early-stopped reference folds and 52.5sec for stronger regularization on another machine; that gives only an order-of-magnitude guide. Benchmark the first unchanged fold before estimating runtime. Expect roughly minutes per five-fold family rather than hours, subject to current CPU. Native categorical fits may cost more per split. Start with at most two new structural candidates (10 outer fits), plus baseline split diagnostics; one nested stopping candidate needs ~15–20 fits depending inner count. Run no more than four total CPU threads across concurrent model jobs, reuse valid frozen split features, and save progress after each fold. No costly GPU or very large ensemble is justified.

## Files read in full

- CODEX_MASTER_PROMPT.md
- generate_submissions.py (340 lines)
- generate_phase2_submissions.py (181 lines)
- generate_phase3_boosting.py (217 lines)
- run_phase4.py (334 lines)
- run_phase5.py (104 lines)
- run_phase6.py (114 lines)
- weight_loss_study.py (66 lines)

Additional targeted searches: version/default mentions across Markdown/JSON/text source reports; historical runtime/result metadata from Phase4 report and PHASE4_CV_RESULTS.csv; installed XGBoost parameter docstrings. No fitted models or new scores were produced by this audit.
