# Independent study completed

Completed: fresh training-only EDA; 216 successful Optuna trials across 18 original scopes; 20 outer folds for four model procedures; four final full-training models; two fixed blends; diagnostics, exact configurations and six validated Kaggle CSVs.

The exploratory CatBoost representation follow-up added six scopes and 18 fits with no new Optuna trials. Total research fits: 468, plus three API smoke fits. Best measured outer R²: 0.9285937076 for the exploratory fixed XGBoost/LightGBM/CatBoost64 blend. This is not a Kaggle score.

All models and inference behavior were frozen before the study's one final test-content read. Inference completed on 2026-10-15. Schema, ID ordering, 3,898 rows, finite nonnegative prices and frozen hashes passed. No training or model changes followed test access. Do not rerun inference into the existing submissions directory or remove its marker.

See README.md, model_configurations.json, experiment_ledger.json, analysis/FINDINGS.md and submissions/inference_audit.json. No Kaggle uploads were performed. Cloud dependencies were exercised and reusable setup/startup instructions saved as an unpublished environment draft.
