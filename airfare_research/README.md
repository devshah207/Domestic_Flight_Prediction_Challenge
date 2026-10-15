# Airfare research — S-Rank Hunters

Executed 15 October 2026. **No new model demonstrated sufficiently stable improvement. Keep the historical best submission.** The original XGB850 pipeline was reproduced, audited, evaluated under five split settings, and compared with six focused procedures. All comparisons use original-price R² and training-only predictions. No neural models were used.

## Results on identical primary outer folds

Five raw-predictor-duplicate-blocked, target-decile-stratified folds, split seed `20261015`; model seed `42`. All 15,589 training rows have one outer prediction per procedure.

| Procedure | Pooled outer OOF R² | Δ versus fixed850 | Positive folds | Decision |
|---|---:|---:|---:|---|
| Original sparse-OHE XGB, fixed850 | 0.92981094 | — | — | Control retained |
| Departure/arrival duration hierarchy, fixed850 | 0.92984750 | +0.00003656 | 1/5 | Negligible, uncertain |
| Native categorical XGB, fixed850 | 0.92798788 | −0.00182306 | 2/5 | Worse R², larger training gap |
| Carrier-specific Business-duration terms, fixed850 | 0.92656488 | −0.00324606 | 0/5 | Rejected |
| Native categories, inner stopping | 0.92933567 | −0.00047527 | 3/5 | Does not beat control |
| Original representation, inner stopping | 0.92869856 | −0.00111239 | 2/5 | Does not beat fixed850 |
| Fixed50/50 original/native850 blend | 0.93124315 | +0.00143221 | 3/5 | Fails stability and uncertainty gate |

The blend's raw-duplicate-cluster paired95% interval is approximately `[−0.000002, +0.002894]`; flight-cluster interval `[−0.000348, +0.003382]`. Both include zero. Its numerical gain does not justify a new submission. We did not change its weights, relax the gate, or generate its Kaggle CSV. The scheduled repeat seed was conditional on passing the primary gate; none passed, so no candidate proceeded to repeat confirmation.

The three structural hypotheses were registered before their scores. The nested stopping follow-up and one fixed blend were explicitly registered exploratory additions after earlier diagnostics. These outer scores are useful conditional evidence, **not a pristine external test or an unbiased estimate after all selection**. All labeled data had already informed historical research. No new independent holdout is claimed. Adjusted intervals for family sizes3/4/5/6 are included; 2,000 bootstrap draws make extreme adjusted quantiles approximate.

## Delivered submissions

- `submissions/PRESERVED_BEST_XGB850_0.95797.csv`: the historical best, preserved byte-for-byte. **0.95797 is the user's reported public Kaggle score.** No new private score is known.
- `final/xgb850_control_refit.csv`: current-version faithful procedure refit, validated `id,price`, exactly3,898 rows in sample ID order, finite nonnegative predictions. This is a reproduction/control artifact; its Kaggle score is unmeasured and it is not recommended as an improvement.

The handoff omitted historical package versions and a fitted model. The explicit parameters and six extracted feature definitions match the historical source; exact historical prediction equality cannot be promised. This run uses XGBoost CPU3.1.1 and sklearn1.8.0, with four threads rather than historical six to match the cloud quota. The old best remains the safer known fallback. No submission was uploaded to Kaggle during this research.

All modeling choices were frozen before Kaggle test features were parsed. Only the fixed850 control was selected for full-data inference. `final/frozen_final_manifest.json` records code/data hashes, full resolved booster configuration, transformations, feature names, postprocessing and serialized model hashes. `final/final_inference_audit.json` records the single final inference read and schema/ID/validity checks. Test payload download/checksum inspection occurred for transport integrity before modeling; no test feature values or distributions informed research. After final inference, no model or prediction adjustments were made.

## What the audit found

See `CODE_AUDIT.md`, `HISTORY_AUDIT.md`, `eda/TRAINING_ONLY_EDA.md`, and `analysis/FINDINGS.md` for full evidence.

- Phase4 used the scored fold to choose stopping rounds, and replaced the original850-round control with640. Those were not clean comparisons of the same fitted procedure.
- Phase5 cabin splitting improved development CV but worsened its already-used holdout. Phase6 target-history features and tail weights also failed overall original-price R². This run did not repeat those exact experiments.
- The fixed850 control scored0.928923 on historical-style grouped seed42,0.929811 on new grouped folds,0.930814 on ordinary stratified folds,0.929515 under conservative itinerary grouping, and0.874981 when whole flight IDs were held out. Different splitting problems do not measure model improvement. `days_left` is a booking horizon, not a calendar date.
- Business rows contribute97.40% of control squared error; the highest-priced1% contribute38.29%; the largest-error1% contribute60.37%. EY is the largest carrier error contributor. We preserved those observations and did not inflate predictions or reweight them.
- Exact-feature target conflicts account for only0.00189% of total training target variation in the observed repeated rows, so they do not explain the full local error. This is an observed ambiguity statistic, not a population noise bound.
- Adding departure/arrival keys improved masked duration RMSE1.65891→1.47210 hours and MAE0.34852→0.27384, with five positive reconstruction folds. Price R² barely changed. Improved feature reconstruction alone did not establish better generalization.
- Native categories reduced MAE but increased training fit and worsened R². Separate inner stopping reduced that gap without beating fixed850. A lower train–validation gap is not itself the objective.

## Run the code

Linux CPU, Python3.12. Use the existing isolated environment/checkout; do not create a Git worktree unless explicitly requested. Repository files were left unchanged. No background service is needed.

In the cloud, the prepared interpreter is `/workspace/airfare_research/.venv/bin/python`. On another machine, unpack this bundle, create a virtual environment and install `requirements.txt`. Supply the original three competition CSVs separately; they are not embedded in this bundle.

```bash
python -m venv --system-site-packages .venv
.venv/bin/python -m pip install -r requirements.txt
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MPLCONFIGDIR=/tmp/airfare-mpl

# Descriptive training-only EDA and feature-only masked duration reconstruction
.venv/bin/python training_eda.py --train train.csv --out eda

# Exact fixed850 control; the code never opens Kaggle test
.venv/bin/python research.py --train train.csv --mode baseline --scheme raw --seed 20261015 --out results/raw_20261015_baseline

# Structural ablations: use the same command/folds with each mode and output
.venv/bin/python research.py --train train.csv --mode native --scheme raw --seed 20261015 --out results/raw_20261015_native
.venv/bin/python research.py --train train.csv --mode duration_schedule --scheme raw --seed 20261015 --out results/raw_20261015_duration_schedule
.venv/bin/python research.py --train train.csv --mode prefix_duration --scheme raw --seed 20261015 --out results/raw_20261015_prefix_duration

# Nested stopping: two fold-pure inner stopping fits per outer training fold
.venv/bin/python research.py --train train.csv --mode native --early --scheme raw --seed 20261015 --out results/raw_20261015_native_nestedstop
.venv/bin/python research.py --train train.csv --mode baseline --early --scheme raw --seed 20261015 --out results/raw_20261015_baseline_nestedstop

# Recompute diagnostics and only the previously registered equal-weight OOF blend
.venv/bin/python analyze_results.py --train train.csv --results results --out analysis --bootstrap 2000 --form-registered-blend
```

The remaining control sensitivities use `--scheme iid`, `--scheme semantic`, and `--scheme flight` with seed20261015, plus `--scheme raw --seed 42`, each with its own `--out` path. Run training fits sequentially within the four-CPU quota; concurrent OpenMP training caused contention during this run, so elapsed fit times are not a controlled speed benchmark. A paused/resumed baseline nested fit also includes waiting time. This affects timing, not held-out score definitions.

For a new, isolated replication of only the unchanged control:

```bash
.venv/bin/python finalize.py --train train.csv --test test.csv --sample sample_submission.csv --selected baseline --out fresh_control --freeze-only
# Only after inspecting the frozen procedure; same command reuses the fitted files
.venv/bin/python finalize.py --train train.csv --test test.csv --sample sample_submission.csv --selected baseline --out fresh_control
```

The second command reads test once and refuses subsequent access/refitting in that output directory. The delivered `final/` directory is already complete; its paths/hashes belong to this cloud instance, so keep it intact as evidence. The script's optional blend mode is for a future separately justified, frozen validation decision; **the blend in this report was rejected and must not be presented as promoted**. No new modeling should be justified from this run's already-consumed test features or predictions.

## Artifacts and verification

- `experiment_ledger.json`: exact configurations, features, split/model seeds, selected round counts, registrations, decisions, package versions and OOF hashes.
- `results/*/oof.csv`: row positions, IDs, labels, groups, outer fold assignments and predictions; `fold_metrics.csv`, `summary.json`, resolved configs and tree-bound audits accompany each fitted procedure. Blend training metrics are explicitly unavailable, rather than inferred from component R².
- `analysis/`: pooled/fold metrics, paired confidence intervals and fold deltas, class/prefix/route/stops/missingness/lead-time/rare-flight/tail diagnostics, fold composition, worst errors and diagnostic PNGs.
- `eda/`: training-only tables, missingness and duplicate audits, masked-duration predictions/fold metrics, four EDA plots.
- `historical_features.py`: AST-identical extracted historical feature builder/OHE definitions. Original handoff files remain in `handoff/` for provenance.
- `final/`: fitted control, frozen manifest, original booster configuration, inference audit and valid control CSV.

Executed50 outer model fits,20 inner stopping fits and one full-data control fit; the fixed blend needed no additional fitting. Every outer procedure completed all five folds. Independent review verified baseline feature-source equality, fold-pure preprocessing, monetary metric scaling, actual tree bounds, paired-bootstrap formulas and null/sign checks. Setup installation/import checks also passed on repeat execution. No Kaggle test-based tuning, neural models, external training data, target encodings, target weighting or prediction calibration were used.

Cloud setup `install_script` and `start_skill` were saved to the environment draft. Saving those fields did not publish the environment. Review/save the environment settings and publish if you want to reuse its prepared snapshot in future tasks.
