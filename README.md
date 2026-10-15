# Domestic Flight Price Prediction — S-Rank Hunters

Executed airfare regression research, Python implementations, exact model configurations, validation diagnostics and Kaggle submission files. No neural networks were used. The competition datasets and Python virtual environment are not included.

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

- [Latest complete code, fitted models and research results](https://github.com/devshah207/Domestic_Flight_Prediction_Challenge/raw/refs/heads/main/downloads/Airfare_Fresh_Optuna_Results.zip)
- [Six submissions and configurations](https://github.com/devshah207/Domestic_Flight_Prediction_Challenge/raw/refs/heads/main/downloads/Airfare_Submissions_and_Configs.zip)
- [Earlier baseline audit and historical research](https://github.com/devshah207/Domestic_Flight_Prediction_Challenge/raw/refs/heads/main/downloads/Airfare_Research_Results.zip)

Original archives and source files are preserved without modification so their frozen hashes remain valid. Cloud-specific absolute paths in historical records describe the original run; use explicit input paths in the reproduction commands. Supply your original competition CSVs separately to rerun training or inference.
