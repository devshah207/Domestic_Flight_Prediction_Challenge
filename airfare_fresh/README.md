# Independent airfare regression results — S-Rank Hunters

Completed: fresh training-only EDA, 216 successful Optuna trials, 468 research model fits, protected five-fold evaluation, four fitted full-training models, and six Kaggle-ready submissions. Three API smoke fits are separate. No neural networks were used. No submissions were uploaded to Kaggle.

## Submission candidates

Each CSV in `submissions/` has exactly 3,898 rows and the `id,price` schema in sample-submission ID order. Prices are finite and nonnegative. These scores are pooled outer out-of-fold R² on **training observations**, not Kaggle scores.

| CSV | Procedure | Outer R² | Mean train–validation R² gap |
| --- | --- | ---: | ---: |
| submission_equal_blend_onehot64.csv | Fixed thirds: XGBoost + LightGBM + CatBoost64 | 0.928594 | Not measured for blend |
| submission_equal_blend.csv | Fixed thirds: XGBoost + LightGBM + original CatBoost | 0.926736 | Not measured for blend |
| submission_xgboost.csv | Native-categorical XGBoost | 0.925416 | 0.042506 |
| submission_lightgbm.csv | Native-categorical LightGBM | 0.925142 | 0.040547 |
| submission_catboost_onehot64.csv | CatBoost with small-category one-hot threshold 64 | 0.925056 | 0.026362 |
| submission_catboost.csv | Original CatBoost procedure, retained comparison | 0.910874 | 0.036243 |

The replacement blend has the highest measured validation R². CatBoost64 and this blend are **exploratory**: the representation hypothesis arose after original model diagnostics, and the same outer folds were reused. Original CatBoost is the weakest comparison candidate; its smaller gap does not make its lower validation accuracy preferable. All six files are supplied for the user's requested comparison, with that distinction explicit.

The historical public R² of 0.95797 is user-reported and comes from different hidden observations. These CV values cannot establish whether a CSV will improve it or reach the top five. The measured gaps show remaining overfitting risk; none of the models is certified free of overfitting.

## Findings

Training data show strong cabin-by-airline-prefix price regimes. Duration and booking-horizon associations differ by regime, arguing against a single global correction or monotonicity rule. Identical input features sometimes have conflicting prices; rows were retained and grouped to prevent repeated itineraries crossing folds. See `eda/TRAIN_ONLY_EDA.md` for missingness, sparse groups, duration ambiguities and plots.

Original CatBoost fitted even its training partitions relatively weakly. Its resolved configuration used `one_hot_max_size=2` and a one-threshold target CTR. This suggested a categorical-representation limitation. A registered, single-change follow-up raised the one-hot threshold to 64 and independently reselected stopping rounds while retaining the original selected parameters. Pooled R² rose from 0.910874 to 0.925056, improving on all five paired folds. This supports the representation hypothesis within this study; it does not prove the CTR setting was the only cause or establish hidden-test gains.

Expensive business-class fares remain the main error source. For the highest-scoring blend, business-class rows account for about 97.35% of squared error and the highest-priced 1% account for about 36.42%. Those expensive rows are systematically underpredicted. No outliers were deleted, labels changed, or post-hoc prediction calibration applied. Some inner stopping counts vary substantially, exposing sample-size and stopping uncertainty. Full tables and uncertainty qualifications are in `analysis/FINDINGS.md`.

## Validation and information boundaries

Five identical outer folds use stratified itinerary groups. Airline and duration are excluded from the group key so missingness cannot hide repeated itineraries. This is more conservative than row-random splitting, but does not hold out whole routes or flight IDs. All 15,589 training rows have one outer prediction per procedure, with zero group overlap.

Within each outer training partition, 12 Optuna trials per original family use two inner 80/20 grouped tuning splits. Learned imputers and categorical mappings are fitted within their fitting partitions. Inner labels select parameters and early stopping; outer labels do neither. Models are refitted on all outer training rows at the median of the two selected stopping counts. Bounds include depth 4–7, finite leaves at most 48 and at most 2^depth, and a 1,200-tree ceiling with patience 70 and a 100-tree final floor. Exact family-specific spaces are in `study.py`.

Full-data models rerun the same inner tuning procedure on all labeled data. The representation follow-up reuses original CatBoost-selected settings and reselects rounds. Final tree counts are XGBoost **454**, LightGBM **764**, original CatBoost **344**, and CatBoost64 **189**. All final models use context features; outer selections can differ. Targets are trained as `price / 10000`; predictions are multiplied by 10,000 and individually clipped below at zero before fixed blending. No ensemble weights were fitted.

The complete four-model freeze preceded this study's final test-content read. Inference read test content once, without subsequent model changes. An earlier, separate historical-control phase had already accessed the test data; this study does not claim it was never previously opened. No test-derived statistics, mappings or prediction distributions informed this independent study. See `PROTOCOL.md`, `CAT_REPRESENTATION_ADDENDUM.md` and the freeze/audit JSONs.

Outer scores evaluate complete selection/refit procedures, not an independently tested common full-data dictionary. Historical use of these training data and choosing the highest-scoring procedure introduce selection uncertainty. The 500 group-bootstrap intervals condition on fitted OOF predictions and omit retraining uncertainty.

## Configurations and evidence

- `model_configurations.json`: exact selected configurations, constructor/resolved parameters, actual tree counts, feature lists, seeds, versions, model hashes and fixed blend weights.
- `experiment_ledger.json`: all archived Optuna trials, outer metrics, source hashes, fit accounting and inference audit.
- `results/`: trial JSONs and SQLite studies; selected settings/stopping records; four OOF files; final configuration JSONs and serialized preprocessing/model bundles.
- `analysis/`: aligned OOF predictions, fold/subgroup metrics, worst errors, tail diagnostics, trial/stopping tables, bootstrap comparisons and plots.
- `submissions/`: six CSVs, freeze manifest, inference-start marker and completed inference audit.
- `ARTIFACT_MANIFEST.json`: packaged file byte counts and SHA-256 hashes, excluding itself.

Research fit accounting: 216 trials × two inner fits + 15 outer refits + three full refits = 450. The follow-up adds 12 inner fits, five outer refits and one full refit = 18, with zero additional Optuna trials. Total: 468.

## Reproduce

Use Python 3.12 and the original CSVs, supplied separately. Training SHA-256: `6b0c9793817dc3e1b229ecea9ebbef14eef34452abf05a6f708d197cf6d8cc07`. Retain source contents because stored studies and inference enforce source/configuration hashes. Work in the extracted `airfare_fresh` directory.

```bash
python3.12 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
export OMP_NUM_THREADS=4 OMP_WAIT_POLICY=PASSIVE OPENBLAS_NUM_THREADS=1
export MPLCONFIGDIR=/tmp/airfare-mpl
export TRAIN=/absolute/path/to/train.csv
export TEST=/absolute/path/to/test.csv
export SAMPLE=/absolute/path/to/sample_submission.csv
```

For a full research rerun use a **new** results directory. Existing studies may skip or resume completed work; they are not new independent evaluations. These commands require substantial computation:

```bash
python independent_eda.py --train "$TRAIN" --out rerun_eda
python study.py --stage outer --family all --train "$TRAIN" --out rerun_results
python study.py --stage final --family all --train "$TRAIN" --out rerun_results
python cat_representation.py --stage all --train "$TRAIN" --results rerun_results
python diagnostics.py --train "$TRAIN" --results rerun_results --out rerun_analysis
```

For inference from supplied fitted models, use `--results results`; for a rerun, use `--results rerun_results`. Freeze all four models before providing test paths:

```bash
python inference.py --results results --train "$TRAIN" --out new_submissions --include-cat-onehot64 --freeze-only
python inference.py --results results --train "$TRAIN" --test "$TEST" --sample "$SAMPLE" --out new_submissions --include-cat-onehot64
```

Use a new output directory. The included `submissions/` has an exclusive inference-start marker: repeated inference there and CSV overwrites are refused. Do not delete that marker to bypass the guard. Fitted bundles use joblib and should only be loaded from trusted sources. Platform/thread differences may affect numerical reproducibility despite pinned versions and seeds.

## Cloud setup

The current instance successfully ran training, diagnostics and final inference. Its interpreter is `/workspace/airfare_research/.venv/bin/python`; `install.sh` is a cloud-specific helper. The empty selected Git repository was left unchanged; this research lives outside it. Reusable `install_script` and `start_skill` instructions were saved to the environment draft. Saving does not publish the draft or verify a future instance; publication remains available in environment settings. Cloud tasks already have isolated environments; no additional Git worktree is needed.
