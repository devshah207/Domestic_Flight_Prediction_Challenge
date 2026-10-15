# Phase 6 — Smoothed price-history test

**Decision:** No submission: neither new candidate met the predeclared evidence gate.

## Experimental design
- All EDA, preprocessing, feature generation, and CV used the supplied training data only.
- Exact original XGBoost: 850 trees, depth 7, learning rate .03; no model hyperparameter changes.
- 5 outer folds grouped by identical feature rows and stratified by target deciles (seed 2026).
- Four inner folds produce cross-fitted target statistics for the training portion of each outer fold. Outer validation uses mappings fitted only on outer training labels.
- Strong smoothing toward class means for rare flight/route categories.
- Five-fold CV is training-only comparative evidence, **not a fresh untouched external holdout**. Prior work used much of this same training dataset.

## Results

| Candidate | OOF R² | Δ vs baseline | Positive folds | Business RMSE | Top 1% RMSE |
|---|---:|---:|---:|---:|---:|
| baseline | 0.929543 | +0.000000 | — | 16,024 | 63,445 |
| coarse_history | 0.922893 | -0.006650 | 0 | 16,794 | 73,101 |
| fine_history | 0.920859 | -0.008684 | 0 | 17,018 | 73,925 |

## Advance criterion
Improvement >=0.0015 pooled OOF R², improvement in >=4/5 folds, positive approximate grouped bootstrap 95% lower confidence bound, top-1% RMSE <=1.015x baseline and business RMSE <=1.01x baseline.

## Limitations
A grouped bootstrap on pooled OOF errors is an approximate uncertainty assessment; its resampling does not capture the full uncertainty in retraining. Kaggle public/private rankings cannot be inferred from training-only CV.

## Phase 6B — Original-price loss weighting, exploratory follow-up

After the encoding variants failed, we separately explored two modest, training-only weighting policies with the unchanged 850-tree XGBoost architecture. The thresholds for the top 5% and top 1% are recomputed within each outer training fold; weights are normalized to mean one. The same CV folds and original feature builder are used. This is not fresh independent evaluation.

| Objective | OOF R² | Δ vs baseline | Positive folds | Business RMSE | Top-1% RMSE |
|---|---:|---:|---:|---:|---:|
| tail_up | 0.928158 | -0.001385 | 1/5 | 16,179 | 60,804 |
| tail_down | 0.929176 | -0.000367 | 2/5 | 16,082 | 67,406 |

**Decision:** Neither objective weighting rule met the evidence gate. Original XGBoost 850 remains the benchmark.
