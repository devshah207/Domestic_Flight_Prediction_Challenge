# Domestic Flight Price Prediction — S-Rank Hunters

Executed airfare regression research, Python implementations, exact model configurations, validation diagnostics and Kaggle submission files. No neural networks were used. The competition datasets and Python virtual environment are not included.

**Best user-reported public Kaggle R²: 0.95885**, from `submission_highround_optuna_classic_gradient.csv`. See [public submission feedback](PUBLIC_RESULTS.md) for all reported scores and their limitations.

## Replication of the disclosed RF + XGBoost + CatBoost settings

The disclosed settings were evaluated with two fixed encodings: one-hot controls and scikit-learn `TargetEncoder(smooth="auto")` for RF/XGBoost. CatBoost uses native categories. All disclosed counts, depths, learning rates, seeds and equal-third weights are preserved. The exact peer encoder implementation, feature list and other parameters are unknown, so this is an approximate pipeline replication.

- **[Download seven replication CSVs, code, configurations and diagnostics](https://github.com/devshah207/Domestic_Flight_Prediction_Challenge/raw/refs/heads/main/downloads/Airfare_Peer_Replication_Submissions.zip)**
- [Smoothed-encoding equal blend CSV](https://github.com/devshah207/Domestic_Flight_Prediction_Challenge/raw/refs/heads/main/submissions/submission_peer_equal_smooth_auto.csv)
- [One-hot equal blend CSV](https://github.com/devshah207/Domestic_Flight_Prediction_Challenge/raw/refs/heads/main/submissions/submission_peer_equal_ohe.csv)
- [Exact settings and all individual CSV links](airfare_peer_replication/README.md)
- [Complete replication including fitted models](https://github.com/devshah207/Domestic_Flight_Prediction_Challenge/raw/refs/heads/main/downloads/Airfare_Peer_Replication_Complete.zip)

RF uses 200 trees/depth 2/no bootstrap/70% feature sampling; XGBoost uses 964 trees/depth 9/lr .054; CatBoost uses 1000 iterations/depth 6/lr .139; every seed is 42. Target encoding excludes each training row's entire itinerary group. No hyperparameters or mixture weights were optimized. On training-only outer validation, the smooth-auto equal blend scored **0.917575** and its one-hot control **0.907913**; smooth-auto XGBoost individually scored **0.922700** with a substantial training gap. These are replication candidates, not demonstrated improvements over the earlier validation controls. No new public score or replication of the peer's 0.96080 is claimed.

## Previous high-round Optuna candidates

Eight boosting procedures completed 336 Optuna trials with requested budgets of 800, 1,100 or 1,400 rounds, followed by nested training-only validation and full-data refits. Two fixed equal-weight blends bring the delivery to ten CSVs. All requested candidates are supplied, including weaker comparisons. The user subsequently reported **0.95885** for classic gradient boosting, the only improvement from this batch. The other nine exact public scores remain unreported.

- **[Download all ten new CSVs, code, configurations and diagnostics](https://github.com/devshah207/Domestic_Flight_Prediction_Challenge/raw/refs/heads/main/downloads/Airfare_HighRound_Submissions_and_Configs.zip)**
- [Measured scores, actual tree counts and individual CSV links](airfare_boosting_sweep/README.md)
- [Exact tuned configurations for every model](airfare_boosting_sweep/MODEL_CONFIGURATIONS.json)
- [Validation results and error diagnostics](airfare_boosting_sweep/analysis/FINDINGS.md)
- [Complete experiment including fitted models and trial databases](https://github.com/devshah207/Domestic_Flight_Prediction_Challenge/raw/refs/heads/main/downloads/Airfare_HighRound_Complete.zip)

The new study includes XGBoost, LightGBM GBDT and DART, CatBoost symmetric and Lossguide trees, histogram and classic gradient boosting, and AdaBoost.R2. Its five outer folds evaluate tuning performed solely within each training partition. These folds have informed earlier research, so their scores are comparative evidence rather than a fresh untouched test. High tree counts do not establish a leaderboard gain. The latest user-reported public result and its limitations are recorded above.

The strongest new candidate locally is `equal_dart3` (outer R² **0.927333**), followed by `equal_main3` (**0.927105**). The previous equal-blend64 control scored **0.928594** on the same rows and folds. The new DART blend's gain over the Business control is small and its paired uncertainty interval includes zero. These local results did not predict the subsequently reported classic gradient public win. They do not establish private-test generalization. See [the feedback review](MODEL_FEEDBACK_REVIEW.md). Every new CSV passed the 3,898-row, exact sample-ID order, `id,price` schema and finite nonnegative price checks.

## Previous exploratory 800-tree candidate

The preceding experiment supplies exactly one more aggressive Business-augmented XGBoost candidate. It keeps the original global branch and fixed 50/50 Business mixture, while replacing the specialist with a fixed 800-tree profile. Validation improved on both reused split seeds, with larger training–validation gaps. The user subsequently reported public Kaggle R² **0.95736**, below the previous best **0.95881** by **0.00145**. The local gain did not transfer to this public split; this candidate does not replace the best public submission. Frozen experiment reports and archives retain the information available when they were created.

- **[Download the one new CSV](https://github.com/devshah207/Domestic_Flight_Prediction_Challenge/raw/refs/heads/main/submissions/submission_aggressive_business_xgb800.csv)**
- [Read validation results and exact settings](airfare_aggressive800/README.md)
- [Download candidate, configuration, model and diagnostics](https://github.com/devshah207/Domestic_Flight_Prediction_Challenge/raw/refs/heads/main/downloads/Airfare_Aggressive800_Candidate.zip)

## Follow-up experiments

The follow-up studies directly tune CatBoost64 and evaluate a bounded Business-class specialist, with fixed prediction mixtures and a conditionally registered second split-seed check. The earlier six submissions remain available below.

- **[Download four new comparison submissions and exact configurations](https://github.com/devshah207/Domestic_Flight_Prediction_Challenge/raw/refs/heads/main/downloads/Airfare_New_Submissions_and_Configs.zip)**
- [Read the measured results, trade-offs and reproduction instructions](NEXT_EXPERIMENTS.md)
- [Exact follow-up model parameters](NEXT_MODEL_CONFIGURATIONS.json)
- [Direct CatBoost64 validation findings](airfare_round2/analysis/FINDINGS.md)
- [Business-specialist validation findings](airfare_business/analysis/FINDINGS.md)
- [Download the complete follow-up code, models and results](https://github.com/devshah207/Domestic_Flight_Prediction_Challenge/raw/refs/heads/main/downloads/Airfare_Next_Experiments_Results.zip)

These are training-only validation results, not new Kaggle scores. Read the paired folds and expensive-fare diagnostics before treating a small pooled gain as evidence of improvement.

## Download the submissions

**[Download all six CSVs and exact configurations as one ZIP](https://github.com/devshah207/Domestic_Flight_Prediction_Challenge/raw/refs/heads/main/downloads/Airfare_Submissions_and_Configs.zip)**

For individual CSVs, use these direct download links. If GitHub opens a preview, select **Download raw file**. The ordinary file pages are available in [`submissions/`](submissions/).

| Submission | Training-only outer CV R² | Download |
| --- | ---: | --- |
| Fixed XGBoost + LightGBM + CatBoost64 blend | 0.928594 | [CSV](https://github.com/devshah207/Domestic_Flight_Prediction_Challenge/raw/refs/heads/main/submissions/submission_equal_blend_onehot64.csv) |
| Original fixed three-model blend | 0.926736 | [CSV](https://github.com/devshah207/Domestic_Flight_Prediction_Challenge/raw/refs/heads/main/submissions/submission_equal_blend.csv) |
| XGBoost | 0.925416 | [CSV](https://github.com/devshah207/Domestic_Flight_Prediction_Challenge/raw/refs/heads/main/submissions/submission_xgboost.csv) |
| LightGBM | 0.925142 | [CSV](https://github.com/devshah207/Domestic_Flight_Prediction_Challenge/raw/refs/heads/main/submissions/submission_lightgbm.csv) |
| CatBoost64 | 0.925056 | [CSV](https://github.com/devshah207/Domestic_Flight_Prediction_Challenge/raw/refs/heads/main/submissions/submission_catboost_onehot64.csv) |
| Original CatBoost comparison | 0.910874 | [CSV](https://github.com/devshah207/Domestic_Flight_Prediction_Challenge/raw/refs/heads/main/submissions/submission_catboost.csv) |

Every CSV has 3,898 rows, the `id,price` schema, exact sample-submission ID order and finite nonnegative prices. The top-level CSVs are byte-identical copies of the frozen files in `airfare_fresh/submissions/`.

The first blend has the highest measured validation score. CatBoost64 and its blend are exploratory follow-ups evaluated on reused outer folds. The original CatBoost is a lower-performing comparison candidate. These CV scores are **not Kaggle scores** and cannot establish improvement over the historical user-reported public R² of 0.95797 or guarantee a leaderboard position. Training gaps, expensive-fare errors and stopping uncertainty are reported explicitly.

## Code and evidence

- [`airfare_fresh/README.md`](airfare_fresh/README.md): independent study findings, validation design and runnable reproduction commands.
- [`airfare_fresh/model_configurations.json`](airfare_fresh/model_configurations.json): exact tuned configurations, actual tree counts, feature lists, versions and fixed ensemble weights.
- [`airfare_fresh/analysis/FINDINGS.md`](airfare_fresh/analysis/FINDINGS.md): fold scores, overfitting diagnostics, tail errors and qualified uncertainty estimates.
- [`airfare_fresh/experiment_ledger.json`](airfare_fresh/experiment_ledger.json): all 216 completed Optuna trials and research fit accounting.
- [`airfare_fresh/results/`](airfare_fresh/results/): trial databases, OOF predictions, selected configurations and four fitted preprocessing/model bundles.
- [`airfare_fresh/eda/TRAIN_ONLY_EDA.md`](airfare_fresh/eda/TRAIN_ONLY_EDA.md): independent training-only EDA.
- [`airfare_fresh/submissions/inference_audit.json`](airfare_fresh/submissions/inference_audit.json): completed submission validation and SHA-256 checksums.
- [`airfare_research/`](airfare_research/): earlier baseline reproduction/audit, unsuccessful comparisons and historical handoff. Its earlier promotion decision concerns that earlier phase; use `airfare_fresh/` for the latest independent study.
- [`tools/finalize_airfare.py`](tools/finalize_airfare.py): original cloud-path reporting/verification helper, retained for provenance. Reproduction entry points are documented in the study README.

The independent study completed 468 research fits plus three API smoke fits. Four final models were frozen together before the study's final test-content read; no subsequent model changes were made. No Kaggle uploads were performed. Serialized joblib models should only be loaded from a trusted source.

## Complete archives

- [Original independent study: complete code, fitted models and research results](https://github.com/devshah207/Domestic_Flight_Prediction_Challenge/raw/refs/heads/main/downloads/Airfare_Fresh_Optuna_Results.zip)
- [Six submissions and configurations](https://github.com/devshah207/Domestic_Flight_Prediction_Challenge/raw/refs/heads/main/downloads/Airfare_Submissions_and_Configs.zip)
- [Earlier baseline audit and historical research](https://github.com/devshah207/Domestic_Flight_Prediction_Challenge/raw/refs/heads/main/downloads/Airfare_Research_Results.zip)

Original archives and source files are preserved without modification so their frozen hashes remain valid. Cloud-specific absolute paths in historical records describe the original run; use explicit input paths in the reproduction commands. Supply your original competition CSVs separately to rerun training or inference.
