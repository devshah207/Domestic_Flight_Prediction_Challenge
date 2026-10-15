# Domestic Flight Price Prediction Challenge — complete EDA and experiment history

**Information cutoff:** 2026-10-15. **Evidence labels:** [EDA] = descriptive statistics from `train(1).csv`; [KAGGLE] = user-reported public leaderboard score; [LOCAL] = our previous training-only evaluation; [TEAM] = information relayed by the user about competitors; [INFERENCE] = hypothesis, not established fact. Do not conflate these.

## 1. Competition, evaluation and known leaderboard context

- Competition: **Domestic Flight Price Prediction Challenge** (Kaggle path `/competitions/domestic-flight-price-prediction-challenge`). Target is `price`; submission must have `id,price` in the sample submission's order; 3,898 rows.
- Official leaderboard metric: **R², higher is better, on original-price predictions**, not RMSE, MAE or R² in log-price space. Optimize held-out R². If the target is divided by 10,000 for training, invert scaling before reporting metrics and creating submissions.
- User's best confirmed public score: **0.95797**, `xgb_850_d7_regularized.csv` from Phase 3. User later reported ~**12th** rank. This rank is self-reported and may change.
- Screenshot dated 2026-10-14: public leaderboard uses ~**60%** of test labels; remaining ~**40%** decides private/final ranking. Scores observed: rank 1 0.96429, rank 2 0.96341, rank 3 0.96325, rank 4 0.96154, rank 5 0.96129, rank 6 0.96080, rank 7 0.96054. These are a dated snapshot, not a current leaderboard claim. Differences vs best: rank-5 cutoff +0.00332; rank-3 cutoff +0.00528.
- No private-leaderboard score exists yet in supplied evidence. Never claim that the top teams will overfit or that private rank will improve.

## 2. Dataset overview [EDA] (training data ONLY)

- 15,589 train rows; 12 columns total = 10 predictors plus `id` and target `price`. Predictor columns: `airline`, `flight`, `source_city`, `departure_time`, `stops`, `arrival_time`, `destination_city`, `flight_class`, `duration`, `days_left`.
- Price range 3,801–323,177; mean 29,816.7; median 10,183; standard deviation 39,127.9; skewness 2.704. Price quantiles: 1%=4,293; 5%=6,150; 25%=7,210; 50%=10,183; 75%=35,968; 90%=71,716; 95%=128,713; 99%=185,837.
- Class: Business 6,380 rows (40.9%), median 38,546, mean 60,126.6; Economy 9,209 rows (59.1%), median 7,587, mean 8,818. Median Business/Economy ratio about 5.1.
- Class × stops medians: Business nonstop 33,712 (2,821 rows), one stop 61,789 (3,495 rows), two+ stops 125,611 (64 rows); Economy nonstop 7,555 (8,144), one stop 7,697 (1,063), two+ stops 37,996 (only 2). Avoid confident modeling of the tiny two+ stop groups.
- Distinct `flight` values 1,070. 589 flight IDs occur fewer than 12 times; sparse category memorization is a concern.
- Distinct directional `source_city → destination_city` combinations: 30, uneven frequencies (205–3,566 observed). City names appear fictional/anonymized (e.g., Metropolis, Gotham, Night City), so avoid matching external geographic databases without proof of identity.
- Missing `airline`: 3,089 rows (19.82%); missing `duration`: 3,095 rows (19.85%). Among observed cases, 19 flight prefixes each identify a unique airline label. Prefix-derived airline imputation uses **feature information only**.
- For 2,887 of the 3,095 missing-duration rows (93.3%), at least two nonmissing training rows share `(flight, source_city, destination_city, stops)`; hierarchical group-median imputation is plausible, with broader fallback medians and explicit missingness flag.
- 314 duplicate rows of predictor values when ignoring `id` and `price`, corresponding to 310 duplicate groups totaling 624 rows in Phase 4 analysis; 279 repeated rows also agree on target. About 35 duplicated-feature groups have conflicting prices. Do not automatically remove these: repeated features can correspond to differing prices and may represent genuine irreducible noise. Block identical feature vectors across validation folds when appropriate.
- Correlations on observed train rows: duration vs price Pearson **0.699**, Spearman **0.598**, partially confounded with class, airline and route; `days_left` vs price global Pearson **−0.023** (not evidence of no nonlinear within-class effect).
- Flight-prefix × class medians show dramatic heterogeneity: `6E` Business 30,055 / Economy 7,435; `AI` Business 37,518 / Economy 8,026; `EK` Business 130,960 / Economy 38,080; `EY` Business 147,049 / Economy 32,495; `QR` Business 169,353 (60 rows); `SQ` Business 205,644 (8 rows); `GF` Business 180,949 (31 rows). Many of these groups are sparse, so smoothing is essential for target statistics.
- Of the 156 largest-price training observations (top 1%), all 156 are Business; flight prefixes: `EY` 102, `EK` 18, `QR` 18, `SQ` 7, `GF` 6, other 5.

## 3. Baseline feature engineering and preprocessing [EDA/design]

Implemented in `generate_submissions.py` and reused through Phase 3:
1. Exclude mechanical `id` from predictor matrix. Parse flight prefix and numeric flight number from `flight`.
2. Use **training-feature-only** mapping from `flight_prefix` to `airline` to fill absent airline. Retain `airline_missing` flag.
3. Use training-feature-only hierarchical medians for `duration`: prioritize `(flight, source_city, destination_city, stops)`, then broader flight/route groups, then median fallback; add `duration_missing` flag.
4. Build categorical interactions: `route`, `route_stops`, `route_class`, `prefix_class`, and `dep_arr` (departure × arrival). Preserve original class, stops, days-left and flight identifiers.
5. Group flight IDs occurring fewer than 12 times as a rare `flight_rare` category for some OneHot/LightGBM use; CatBoost keeps full flight identities with native categorical handling.
6. Numeric features include `duration_filled`, missing flags, `days_left`, parsed `flight_number`, `stops_numeric`.
7. Different model families use model-appropriate OHE/standardization or native categorical treatment. Every learned imputation, mapping, frequency or encoder must be refitted on each training fold and carried unchanged into its validation fold. Feature-only transformations must not look at test data during model selection.
8. Original-price models in the historic scripts trained on `price/10000` and rescaled predictions by 10000; experimental log-target model used `log1p(price)` and `expm1` on output. Price scaling is not a different leaderboard metric.

Phase 4 additionally tried nine numerical engineered features (`business_duration`, `business_days_left`, `economy_duration`, `economy_days_left`, `duration_per_leg`, `days_left_log1p`, `days_left_sq`, `duration_sq`, `duration_days_interaction`) and five categorical combinations (`class_stops`, `route_airline`, `flight_route_rare`, `class_depart`, `route_depart`). These did **not** improve 5-fold OOF R² as a combined set. That result does NOT mean all possible feature engineering is futile.

## 4. Complete model / Kaggle score ledger

Scores below are **user-reported public Kaggle R²**, not local CV, unless the cell explicitly states otherwise. `Not reported` means do not invent a score. Full original configuration dictionaries for Phases 1–3 are in `phase1_exact_configs.json`, `phase2_exact_configs.json`, and `phase3_exact_configs.json`. Scripts and reports contain all additional details.

### Phase 1 — 12 submissions, no local validation at user's original request

| Model / file stem | Trees / depth | Key settings | Public R² |
|---|---|---|---:|
| `catboost_depth5` | 360 / 5 | lr .055, L2 10, random_strength 1, rsm .9, Bayesian bagging temperature .55 | Not reported |
| `catboost_depth4` | 440 / 4 | lr .048, L2 12, random_strength 1.3, rsm .9, Bayesian bagging temperature .7 | Not reported |
| `catboost_depth6` | 380 / 6 | lr .047, L2 15, random_strength 1.6, rsm .85, Bernoulli row sample .82 | Not reported |
| `xgboost_depth3` | 340 / 3 | lr .065, child_weight 10, row .83, col .80, L2 15, L1 .3, gamma .025 | Not reported |
| `xgboost_depth4` | 410 / 4 | lr .047, child_weight 18, row .80, col .85, L2 22, L1 .6, gamma .05 | Not reported |
| `lightgbm_leaves15` | 360 / 5 | lr .055, 15 leaves, child samples 65, min split gain .04, L2 20, L1 .4, row/col .8 | **0.95232** |
| `lightgbm_leaves9` | 400 / 4 | lr .050, 9 leaves, child samples 90, min gain .035, L2 18, L1 .4 | Not reported |
| `random_forest` | 280 / 13 | min leaf 5, min split 12, max features .8, bootstrap true, max samples .82, max nodes 700 | **0.95221** |
| `extra_trees` | 320 / 14 | min leaf 5, min split 12, max features .75, bootstrap false, max nodes 800 | **0.95328** |
| `ridge_ohe` | — | alpha 35, solver lsqr, max_iter 10000, tol 1e-5 | Not reported |
| `blend_boosting_diverse` | 360+360+340 | 45% CatBoost depth5 + 35% LightGBM 15 leaves + 20% XGB depth3 | **0.95089** |
| `blend_shallow_regularized` | 440+360+400 | 50% CatBoost depth4 + 25% CatBoost depth5 + 25% LightGBM 9 leaves | Not reported |

### Phase 2 — 7 tree-count/regularization submissions, without local validation

| File stem | Trees/depth | Distinguishing parameters | Public R² |
|---|---|---|---:|
| `extra_trees_A_450` | 450 / 14 | leaf 5, split 12, max features .75, max nodes 800, no bootstrap | **0.95298** |
| `extra_trees_B_550` | 550 / 14 | same as ExtraTrees A except tree count | **0.95287** |
| `extra_trees_C_conservative_500` | 500 / 12 | leaf 8, split 18, max features .7, max nodes 600 | **0.94965** |
| `random_forest_A_450` | 450 / 13 | leaf 5, split 12, max features .8, max samples .82, max nodes 700 | **0.95190** |
| `random_forest_B_550` | 550 / 13 | same as RandomForest A except tree count | **0.95194** |
| `lightgbm_A_450_lr055` | 450 / 5 | lr **.055**, 15 leaves, leaf sample 65, L2 20, L1 .4, min gain .04 | **0.95354** |
| `lightgbm_B_550_lr035` | 550 / 5 | lr **.035**, same bounded tree structure | **0.95253** |

**Important correction:** the user once referred to LR values as 0.55 and 0.35; actual training code uses **0.055 and 0.035**. Increasing trees did not consistently raise the public score for RF/ET. Do not attribute the difference between LGB450 and LGB550 to tree count alone because learning rates changed.

### Phase 3 — 7 boosting submissions, user-reported scores

| File stem | Trees / depth | Key settings | Public R² |
|---|---|---|---:|
| `lgb_650_d5_lr045` | 650 / 5 | lr .045, 15 leaves, min samples 65, L2 20, L1 .4, row/col .8 | **0.95431** |
| `lgb_800_d5_lr035` | 800 / 5 | lr .035, 15 leaves, min samples 65, L2 20, L1 .4, row/col .8 | **0.95504** |
| `lgb_750_d6_leaves23` | 750 / 6 | lr .035, 23 leaves, min samples 75, L2 25, L1 .6, min gain .06 | **0.95521** |
| `xgb_750_d5_lr045` | 750 / 5 | lr .045, max leaves 32, child weight 8, row .8, col .85, L2 8, L1 .35, gamma .04 | **0.95740** |
| **`xgb_850_d7_regularized`** | **850 / 7** | **lr .030, max leaves 48, min child weight 12, row .80, col .85, L2 15, L1 .60, gamma .08, max_bin 192, tree_method=hist, seed 42, squared error objective** | **0.95797 (BEST)** |
| `cat_780_d6_lr040` | 780 / 6 | lr .040, L2 leaf 15, random strength 1.3, rsm .85, Bernoulli subsample .82 | **0.94914** |
| `xgb_800_d6_log1p` | 800 / 6 | lr .040, max leaves 48, child weight 6, row .8, col .85, L2 8, L1 .08, gamma .015; target log1p then expm1 | **0.95492** |

Conclusions: boosting (especially XGBoost) substantially improved the public score over shallow/bagged initial models; 850 and 750 XGB configurations differ in many settings, so the effect of number of trees alone is unidentifiable. CatBoost 780 was inferior here, but one trial does not prove CatBoost universally inferior. The log target did not improve original-price R².

### Phase 4 — training-only 5-fold grouped + decile-stratified CV and early stopping

Five folds were grouped by identical original predictors. Fixed max 1,450 boosting rounds, early-stopping patience 75; **the same fold supplied early stopping and was used to score that model**, causing some selection optimism. Final full-training refits used median best-fold iterations. These final submissions were all reported by the user to perform **worse than 0.95797**, but the user did not supply exact Phase 4 Kaggle scores.

| Candidate | OOF R² [LOCAL] | Median stopping round | Public result |
|---|---:|---:|---|
| XGB stronger regularization, depth 7 | **0.929171** | 955 | User said worse, exact unreported |
| XGB reference, depth 7 | 0.929159 | 640 | User said worse, exact unreported |
| XGB shallower, depth 6 | 0.928676 | 909 | Not submitted / no public score |
| XGB engineered, depth 7 | 0.927201 | 634 | Not submitted / no public score |
| XGB engineered, depth 6 | 0.926087 | 704 | Not submitted / no public score |
| LGB engineered, depth 5 | 0.924554 | 522 | Not submitted / no public score |
| 50/50 blend, stronger+reference | **0.929440** | two refit models: 955 and 640 | User said worse, exact unreported |

Specific changes: stronger-reg XGB depth7 used `min_child_weight=18`, `reg_lambda=22`, `gamma=.12` versus baseline weight12/L2 15/gamma .08. Depth6 used max_leaves40. For reference original depth7 in Phase4 was **refitted to only 640 rounds**, so it was NOT identical to the original winning 850-tree submission. Average train R² for stronger-reg 0.95879 versus mean fold validation R² 0.92924, gap 0.02955. Fold best rounds varied widely (reference 170–913; stronger-reg 310–1399), demonstrating unstable early-stopping counts. A naive median number from 80% folds need not be optimal for full-data fit.

Phase4 OOF high-price diagnostics: top 1% of rows contributed ~39.2% of squared error in one reported analysis; business-class rows made ~97.4% of squared error despite only 40.9% of rows. Stronger-reg model underpredicted most expensive 1% by ~51,273 on average. These are **fold-specific error diagnostics**, not claims about test labels.

### Phase 5 — genuinely held-out evaluation of class-specialized models [LOCAL]

Training-only design: 20% grouped/stratified holdout untouched during model development; on remaining 80%, three grouped-stratified development folds. Three predeclared procedures: (1) exact global XGB 850; (2) split Economy/Business models, same 850 rounds each; (3) tailored split models with inner-fold early stopping patience 70, cap Business 1,050 depth7 weight7 L2 12; Economy 650 depth5 weight8 L2 16; both lr .03 with other XGB original parameters.

| Model | Development 3-fold OOF R² | Independent 20% holdout R² (if measured) |
|---|---:|---:|
| Original global XGB 850 | 0.917619 | **0.937351** |
| Split 850/850 by cabin | 0.917297 | Not evaluated on holdout |
| Tailored separate cabin + inner early stop | **0.920639** | **0.934767 — WORSE** |

Holdout RMSE global 9,699 vs tailored 9,897; Business RMSE global 14,838 vs tailored 15,231; Economy RMSE global 2,429 vs tailored 2,059; tail top-1% underprediction global −45,113 vs tailored −52,535. Local CV gain **+0.003020** reversed to holdout **−0.002584**. Paired grouped-bootstrap 95% interval on holdout delta [-0.008839, +0.003874], includes zero. Model rejected; holdout may not be reused as a fresh independent selection set. No new Kaggle candidate from this phase.

### Phase 6 — cross-fitted target/history features and tail weighting [LOCAL]

Five training-only grouped/stratified outer folds; inner four-fold cross-fitting for price-history features within each training part; no change to baseline XGB850 structure. **This is not a fresh untouched test** after many prior modeling experiments.

| Candidate | OOF R² | Δ vs same-fold baseline | Positive folds |
|---|---:|---:|---:|
| Original global XGB850 | **0.929543** | — | — |
| Coarse smoothed airline/route price-history features | 0.922893 | −0.006650 | 0/5 |
| Fine smoothed flight/route target-history features | 0.920859 | −0.008684 | 0/5 |
| Tail-up weighted loss | 0.928158 | −0.001385 | 1/5 |
| Tail-down weighted loss | 0.929176 | −0.000367 | 2/5 |

OOF baseline Business RMSE ~16,024; top 1% RMSE ~63,445; top 1% accounts for ~37.3% squared error in this run. Tail-up weighting reduced top 1% RMSE from 63,445 to 60,804 but **decreased overall R²**. Tail-down weight also did not improve overall R². Coarse/fine target-history features degraded all folds and increased train–validation gap. Reject all four proposals. Do not reintroduce exact failed combinations without new evidence.

## 5. Intelligence provided by user from leading teams [TEAM]

- User talked to top-3/4 participants and reports that they used **boosting**, rather than primarily Random Forest, including a first-place team using about **800 trees**. These are unverified accounts of their implementations, useful as hypotheses only. Cannot infer boosting always wins, or 800 trees is optimal.
- User relayed this selected XGBoost-style configuration from one team: `{'target':'log1p_price','max_depth':7,'min_child_weight':5,'subsample':0.8,'colsample_bytree':0.85,'reg_lambda':8.0}`. **Other params (learning_rate, reg_alpha, gamma, exact optimal rounds, etc.) were NOT specified. Do not invent them.**
- Relayed best **five-fold OOF**: RMSE **95.0741**, MAE **21.3030**, R² **0.81648**. Scale/protocol unclear and not directly comparable to public Kaggle original-price R² 0.95–0.96. Do not assume the other team scored 0.81648 on Kaggle.
- User relayed code `best_n_estimators = min(100, int(np.median(best_iterations)))`, but subsequent explanation says they intended **`max(100, int(median(...)))`**. These are contradictory; flag and clarify rather than silently choosing one. `min(100, median)` caps at 100; `max(100, median)` imposes floor 100.
- Their proposed protocol was **up to 1,800 boosting rounds with `early_stopping_rounds=70` for each fold**, record fold best iterations, and select a median stopping count (with 100-round floor if `max` was intended). An 1,800-round ceiling does **not** mean the final model actually uses 1,800 trees. Correctly isolate early stopping from final evaluation; re-assess final refit round count.
- Public screenshot rank 1..7 scores are given above. User's rank-12 claim is later than the screenshot and self-reported.

## 6. Lessons / failed hypotheses / unresolved questions

1. **Never confuse public R² with local OOF R²:** public 0.95797 vs grouped-CV near 0.929 and one holdout 0.937. The distribution may differ, the splits may vary in difficulty, and high-priced sparse categories disproportionately affect R². A discrepancy does **not** prove test distribution shift.
2. **No model is guaranteed free from overfitting:** finite tree depth, >700 trees with small lr and regularization may still be fine; 850 count itself is not evidence for or against overfitting. More trees in bagged forests mainly reduces ensemble variance while boosting can overfit at excessive rounds.
3. **Leaderboard overfitting** is a serious risk, especially after numerous submissions and with a public 60% / private 40% split. User's goal is top-3/top-5 but the priority is strong unseen/private performance, not speculative public score exploitation.
4. **Training-only CV is currently authorized** by the user as of Phase 4 (supersedes earlier instruction against assistant-performed model-quality tests). Use proper nested / independent assessments; do not fit on the validation fold, and avoid tuning stopping rounds and scoring on the same fold.
5. **The Phase5 20% holdout has been consulted once**, so it is no longer a pristine untouched test for a new selection cycle. A fresh outer/nested protocol, not a reused holdout, is needed for credible follow-up experiments.
6. **No neural networks** of any kind; all relevant complexity parameters must be bounded (not `max_depth=None` or LightGBM unlimited `max_depth=-1`). Do not impose an arbitrary universal 500-tree limit; 750/850 performed better publicly. Tree count must be reasoned from learning rate, complexity, stable train-only validation and runtime.
7. Avoid accidental label leakage in target encodings, business-route statistics or any selected features. Never inspect Kaggle test feature distributions, statistics, missingness, or predictions for tuning; load it only after freezing the model to emit predictions.
8. No automatic `log1p` target; under the R² original-money metric, log target can systematically underpredict costly tickets. Investigate original-scale loss and inverse-transform effects using training folds.
9. Improve truly difficult business/tail cases **without** sacrificing overall original-price R²; the Phase6 tail-up experiment improved high-tail RMSE but lowered overall R².
10. Preserve `xgb_850_d7_regularized.csv` untouched as a reproducibility benchmark and fallback submission. Investigate genuinely different, justified hypotheses before making more Kaggle submissions.

## 7. Source artifacts available in this handoff

- Scripts: `generate_submissions.py`, `generate_phase2_submissions.py`, `generate_phase3_boosting.py`, `run_phase4.py`, `run_phase5.py`, `run_phase6.py`, `weight_loss_study.py` (if packaged).
- Detailed reports: `kaggle_train_only_EDA.md`, `PHASE2_README.md`, `PHASE3_README.md`, `PHASE4_REPORT.md`, `TARGETED_EDA.md`, `PHASE5_DECISION_REPORT.md`, `PHASE6_DECISION_REPORT.md`.
- Exact Phase1/2/3 dictionaries: `phase1_exact_configs.json`, `phase2_exact_configs.json`, `phase3_exact_configs.json`.
- Historical public scores originate from user messages, not model-evaluation experiments we independently verified. Phase4 public scores were described only as 'all worse', no numbers supplied.
