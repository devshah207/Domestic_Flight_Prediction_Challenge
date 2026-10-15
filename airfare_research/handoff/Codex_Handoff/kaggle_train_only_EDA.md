# EDA and model design — Airline ticket price regression

**Important scope:** all statistics below were computed only from `train(1).csv`. There is **no test-feature EDA** and **no trained-model quality measurement**. The official Kaggle metric was specified as **R²**, so the proposed regressors use squared-error training losses. No neural networks are included.

## 1. Dataset and target

- **Training rows:** 15,589; **columns:** 12, comprising `id`, 10 predictors and `price` target.
- **Unique flight IDs:** 1,070; 589 flight IDs have fewer than 12 observed occurrences (unreliable for naive per-flight target averages).
- **Target range:** 3,801 to 323,177; **median:** 10,183; **mean:** 29,816.7; **std:** 39,127.9; **skewness:** 2.704.
- **Target percentiles:** 1th=4,293, 5th=6,150, 25th=7,210, 50th=10,183, 75th=35,968, 90th=71,716, 95th=128,713, 99th=185,837.
- **Target recommendation:** keep target in *original monetary scale* (scaled by 10,000 numerically for the estimators), not log space. R² rewards reducing original-scale squared errors; an uncalibrated log target can systematically downweight expensive tickets.

## 2. Data integrity and missingness

|                  |   Missing rows |   Missing % |
|:-----------------|---------------:|------------:|
| airline          |           3089 |       19.82 |
| duration         |           3095 |       19.85 |
| flight           |              0 |        0    |
| source_city      |              0 |        0    |
| departure_time   |              0 |        0    |
| stops            |              0 |        0    |
| arrival_time     |              0 |        0    |
| destination_city |              0 |        0    |
| flight_class     |              0 |        0    |
| days_left        |              0 |        0    |
| id               |              0 |        0    |

- 3,089 airline names are missing (19.8%). Observed training rows show each flight prefix maps to exactly **one** airline label (**19 prefixes**, with max observed distinct airlines per prefix=1). Thus missing airline names can be imputed using the training-only prefix mapping; no price information needed.
- 3,095 durations are missing (19.9%). **2,887 / 3,095 (93.3%)** have at least two observed rows for the exact same `(flight, source_city, destination_city, stops)` combination. Within that grouping, the median of observed durations is an appropriate first-stage feature-only imputation. Remaining rows use increasingly broad groups, finally the training median, plus a missingness flag.
- Duplicate predictor rows (ignoring `id` and `price`): **314**. Duplicates with matching observed target (ignoring `id`): **279**. Preserve original rows to avoid distorting real frequencies; duplicated feature combinations that sometimes have different prices illustrate irreducible ambiguity.
- `id` is a unique sequential identifier; do not train on it. The sample submission can use its IDs only for output ordering.
- The location names appear fictional or anonymized; do not bring in external distance or flight schedules that cannot be aligned to these labels.

## 3. What matters in prices: training-only descriptive associations

### Class groupings

| flight_class   |   count |   median |    mean |
|:---------------|--------:|---------:|--------:|
| Business       |    6380 |    38546 | 60126.6 |
| Economy        |    9209 |     7587 |  8818   |

Business has roughly **5.1×** economy's median. This makes `flight_class` and interactions with flight prefix, route and stops central.

### Class × stops

|                             |   count |   median |
|:----------------------------|--------:|---------:|
| ('Business', 'one')         |    3495 |    61789 |
| ('Business', 'two_or_more') |      64 |   125611 |
| ('Business', 'zero')        |    2821 |    33712 |
| ('Economy', 'one')          |    1063 |     7697 |
| ('Economy', 'two_or_more')  |       2 |    37996 |
| ('Economy', 'zero')         |    8144 |     7555 |

### Airline prefix × class (all observed training groups)

|                    |   count |   median |
|:-------------------|--------:|---------:|
| ('6E', 'Business') |    1681 |  30055   |
| ('6E', 'Economy')  |    4701 |   7435   |
| ('9I', 'Economy')  |       2 |   5084   |
| ('AI', 'Business') |    3091 |  37518   |
| ('AI', 'Economy')  |    2846 |   8026   |
| ('EK', 'Business') |     404 | 130960   |
| ('EK', 'Economy')  |      39 |  38080   |
| ('EY', 'Business') |     362 | 147049   |
| ('EY', 'Economy')  |      45 |  32495   |
| ('FZ', 'Business') |      11 | 111529   |
| ('GF', 'Business') |      31 | 180949   |
| ('IX', 'Business') |      11 |  32689   |
| ('IX', 'Economy')  |     503 |   6712   |
| ('KU', 'Business') |       5 | 121309   |
| ('KU', 'Economy')  |       2 |  26018   |
| ('MH', 'Business') |      94 | 132072   |
| ('MH', 'Economy')  |       1 |  54314   |
| ('QP', 'Business') |      30 |  79855   |
| ('QP', 'Economy')  |     830 |   7549   |
| ('QR', 'Business') |      60 | 169353   |
| ('RA', 'Business') |       3 |  65005   |
| ('S5', 'Economy')  |       3 |   7099   |
| ('SG', 'Economy')  |     147 |   7333   |
| ('SQ', 'Business') |       8 | 205644   |
| ('TG', 'Business') |     213 | 102364   |
| ('TG', 'Economy')  |      15 |  33319   |
| ('UL', 'Business') |     352 |  71781   |
| ('UL', 'Economy')  |      70 |  27244.5 |
| ('WY', 'Business') |      24 |  97078.5 |
| ('WY', 'Economy')  |       5 |  24701   |

- `duration` vs `price` Pearson correlation among observed durations: **0.699**, Spearman: **0.598**. Correlation is not necessarily causal: it is partially confounded with long-haul routes, classes, and flight types.
- `days_left` vs price global Pearson: **-0.023**. Near-zero global correlation does not exclude nonlinear relationships within class or airline, so do not discard this feature.
- Routes: **30 observed source/destination combinations**. They are unevenly sampled (smallest=205, largest=3566). Include route and route/class interactions, with regularization to avoid fitting rare routes.
- `stops='two_or_more'` appears in only 66 training rows; do not build an intricate bespoke model for this small group.

## 4. Chosen feature engineering

**Learn only from training features, before inference**:

1. Parse `flight_prefix` and numerical `flight_number` from flight names.
2. Recover `airline_fixed` from flight prefix when `airline` is missing; include `airline_missing` indicator.
3. Impute `duration_filled` using hierarchical flight/route/stop medians; include `duration_missing`. The grouping statistics use no target labels.
4. Add `route`, `route_stops`, `route_class`, `prefix_class` and `dep_arr` as simple, interpretable categorical interactions.
5. Retain original `days_left`, stops encoding and flight class. For LightGBM and one-hot models, group flight identities seen fewer than 12 times in the training data to avoid memorizing rare codes (CatBoost retains the full flight ID with its regularized categorical statistics).
6. Fit OHE categories/scalers on train only; the test set is never used in preprocessing fitting.
7. Do **not** add price-derived group averages or target encodings without cross-fitting; this request intentionally has no local CV, so omit them.

## 5. Intended model families (proposals, **not** a measured ranking)

- CatBoost depths 4/5/6: native categorical handling, regularized ordered categorical statistics. Categorical parsing follows [CatBoost documentation](https://catboost.ai/docs/en/features/categorical-features).
- XGBoost depths 3/4: histogram boosting with finite depths, row/column subsampling, substantial L2 regularization and minimum child weight.
- LightGBM max depths 4/5 with 9/15 leaves respectively; enforce leaf counts and minimum records per leaf, consistent with [LightGBM's tuning guidance](https://lightgbm.readthedocs.io/en/latest/Parameters-Tuning.html).
- 280-tree Random Forest and 320-tree Extra Trees with bounded tree depths and 5+ records per leaf; diversity, but not guaranteed better R².
- Regularized Ridge on OHE, primarily as a low-variance, additive baseline.
- Two **fixed a priori**, **not performance-selected** blends of the above models.

Tree counts are **280–440** for boosted/bagged models, **always below 500**. Their depth is explicitly finite, and additional restrictions control leaf complexity. These are priors for plausible generalization, **not proof of no overfitting**.

## 6. Remaining uncertainty and evaluation policy

- This is an EDA-backed *proposal*, not a verified top model. **No training R², CV R², OOF predictions, local holdout tests, or hyperparameter searches were calculated**, exactly as requested.
- Without independent validation, we cannot confirm which proposal generalizes best. Finite depth and low tree counts reduce risk but cannot guarantee good unseen R².
- Public Kaggle R² may provide some information, but repeated leaderboard selection can still overfit the public split. Preserve multiple diverse candidates for private-leaderboard robustness.
- Multiple tested submissions may produce different rankings; record scores outside this report and do not retroactively claim that this EDA predicted the winning model.

## 7. Descriptive visualization

See `train_only_eda.png` (built exclusively from training rows).
