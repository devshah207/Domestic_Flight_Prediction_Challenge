# Codex Handoff Package — Domestic Flight Price Challenge

**Start here:** `CODEX_MASTER_PROMPT.md` is a complete, standalone prompt. Paste this file's contents into Codex along with the user's `train(1).csv`, `test(1).csv`, and `sample_submission(1).csv`, OR attach this entire ZIP as evidence with those CSV inputs.

`EDA_EXPERIMENT_HISTORY.md` is an independently readable summary of EDA decisions, original feature engineering, ALL historical public Kaggle scores and local validation results across Phase 1–6, and the top-team parameter information provided by the user.

`phase{1,2,3}_exact_configs.json` contain original exact hyperparameter dictionaries. Historical scripts and reports are provided for detailed auditing and faithful baseline replication. `PRESERVED_BEST_XGB850_0.95797.csv` is the old public-best submission, NOT a new prediction run, and is included only for reproducibility/fallback.

`original_master_prompt.txt` is the user's earlier prompt for provenance; its optional neural-network section is OVERRIDDEN by the strict NO-NN policy in `CODEX_MASTER_PROMPT.md`.

No user-provided train/test CSV is packaged here: supply them separately. The test file must not be opened for EDA, modeling decisions or validation. It is used only for final frozen-model inference.

All [KAGGLE] scores are user-reported; no private leaderboard scores are known. Phase 4 public scores were not given precisely, only described as worse than 0.95797.
