#!/usr/bin/env python3
"""Phase 2 of Kaggle airfare regression: seven PREDECLARED finite-depth models.

This script deliberately NEVER runs cross-validation, holdouts, training R2,
residual analysis, feature importance, or any test-data exploratory analysis.
All feature mappings are learned from training FEATURES only, and the test file
is read only after ALL seven fits have completed, solely to generate predictions.

Dependencies: pandas, numpy, scipy, scikit-learn, lightgbm; also xgboost/catboost
because the original feature-builder module imports them (though this script
never trains those two models).

Example:
 python generate_phase2_submissions.py \
  --train '/mnt/data/train(1).csv' --test '/mnt/data/test(1).csv' \
  --sample '/mnt/data/sample_submission(1).csv' \
  --out '/mnt/data/phase2_regression_submissions'
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from lightgbm import LGBMRegressor

from generate_submissions import (
    TrainingOnlyFeatureBuilder, make_ohe_features,
    make_lightgbm_features, SEED, TARGET, SCALE, LGB_CAT_COLS,
    CAT_COLS, NUM_COLS,
)

# Exact seven candidates agreed BEFORE examining the Kaggle test features.
# A/B differ from their previous model only in n_estimators (except LightGBM B,
# which also lowers learning_rate). The more-regularized Extra Trees C is a
# separate controlled comparison and not tuned by test leaderboard feedback.
CONFIGS = {
    "extra_trees_A_450": dict(family="ExtraTrees", n_estimators=450,
        max_depth=14, min_samples_leaf=5, min_samples_split=12,
        max_features=0.75, max_leaf_nodes=800, bootstrap=False),
    "extra_trees_B_550": dict(family="ExtraTrees", n_estimators=550,
        max_depth=14, min_samples_leaf=5, min_samples_split=12,
        max_features=0.75, max_leaf_nodes=800, bootstrap=False),
    "extra_trees_C_conservative_500": dict(family="ExtraTrees", n_estimators=500,
        max_depth=12, min_samples_leaf=8, min_samples_split=18,
        max_features=0.70, max_leaf_nodes=600, bootstrap=False),
    "random_forest_A_450": dict(family="RandomForest", n_estimators=450,
        max_depth=13, min_samples_leaf=5, min_samples_split=12,
        max_features=0.80, max_samples=0.82, max_leaf_nodes=700,
        bootstrap=True),
    "random_forest_B_550": dict(family="RandomForest", n_estimators=550,
        max_depth=13, min_samples_leaf=5, min_samples_split=12,
        max_features=0.80, max_samples=0.82, max_leaf_nodes=700,
        bootstrap=True),
    "lightgbm_A_450_lr055": dict(family="LightGBM", n_estimators=450,
        max_depth=5, num_leaves=15, learning_rate=0.055,
        min_child_samples=65, min_split_gain=0.04, reg_lambda=20.0,
        reg_alpha=0.4, subsample=0.80, subsample_freq=1,
        colsample_bytree=0.80, max_bin=127),
    "lightgbm_B_550_lr035": dict(family="LightGBM", n_estimators=550,
        max_depth=5, num_leaves=15, learning_rate=0.035,
        min_child_samples=65, min_split_gain=0.04, reg_lambda=20.0,
        reg_alpha=0.4, subsample=0.80, subsample_freq=1,
        colsample_bytree=0.80, max_bin=127),
}


def make_model(cfg):
    p = {k: v for k, v in cfg.items() if k != "family"}
    if cfg["family"] == "ExtraTrees":
        return ExtraTreesRegressor(**p, random_state=SEED, n_jobs=6)
    if cfg["family"] == "RandomForest":
        return RandomForestRegressor(**p, random_state=SEED, n_jobs=6)
    if cfg["family"] == "LightGBM":
        return LGBMRegressor(**p, objective="regression", random_state=SEED,
                             n_jobs=6, verbosity=-1)
    raise ValueError(f"Unexpected family: {cfg['family']}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True)
    parser.add_argument("--test", required=True)
    parser.add_argument("--sample", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    assert len(CONFIGS) == 7
    for name, p in CONFIGS.items():
        assert isinstance(p["max_depth"], int) and 1 <= p["max_depth"] <= 15
        assert 1 <= p["n_estimators"] <= 600
        assert p["min_samples_leaf"] >= 5 if p["family"] != "LightGBM" else p["min_child_samples"] >= 65

    print("Frozen 7 hyperparameter configs; opening TRAIN only.", flush=True)
    train = pd.read_csv(args.train)
    if TARGET not in train or "id" not in train:
        raise ValueError("Training data needs 'id' and 'price' columns.")
    y_train = train[TARGET].to_numpy(dtype=np.float64) / SCALE
    if not np.isfinite(y_train).all():
        raise ValueError("Non-finite target in training data.")
    builder = TrainingOnlyFeatureBuilder()
    feature_train = builder.fit_transform(train.drop(columns=["id", TARGET]))
    del train

    onehot_encoder, x_ohe = make_ohe_features(feature_train)
    x_lgbm, train_categories = make_lightgbm_features(feature_train)
    del feature_train

    trained = {}
    for name, p in CONFIGS.items():
        print(f"Training {name}: {p['n_estimators']} trees; max_depth={p['max_depth']}", flush=True)
        model = make_model(p)
        if p["family"] == "LightGBM":
            model.fit(x_lgbm, y_train, categorical_feature=LGB_CAT_COLS)
        else:
            model.fit(x_ohe, y_train)
        trained[name] = model
        print(f"Completed: {name}; no scores measured.", flush=True)
    # No test file has been opened before here.
    print("ALL 7 MODELS FIT. Opening TEST only for final inference.", flush=True)
    test = pd.read_csv(args.test)
    sample = pd.read_csv(args.sample)
    required_cols = ["id", "airline", "flight", "source_city", "departure_time",
                     "stops", "arrival_time", "destination_city", "flight_class",
                     "duration", "days_left"]
    if not all(c in test.columns for c in required_cols):
        raise ValueError("Test data schema does not match training input requirements.")
    if not {"id", TARGET}.issubset(sample.columns):
        raise ValueError("Sample submission needs id and price columns.")

    test_features = builder.transform(test.drop(columns=["id", TARGET], errors="ignore"))
    x_ohe_test = onehot_encoder.transform(test_features).tocsr().astype(np.float32)
    x_lgbm_test, _ = make_lightgbm_features(test_features, train_categories)
    del test_features

    # Only use IDs for ordering, not as modeling features.
    if (not test["id"].is_unique or not sample["id"].is_unique or
        len(test) != len(sample)):
        raise ValueError("Unexpected test/sample length or nonunique IDs.")
    positions = pd.Series(np.arange(len(test)), index=test["id"]).reindex(sample["id"])
    if positions.isna().any():
        raise ValueError("Unmatched test/sample IDs.")
    order = positions.to_numpy(dtype=int)

    manifest = []
    for name, p in CONFIGS.items():
        if p["family"] == "LightGBM":
            pred = trained[name].predict(x_lgbm_test)
        else:
            pred = trained[name].predict(x_ohe_test)
        pred = np.asarray(pred, dtype=np.float64).ravel() * SCALE
        if len(pred) != len(test) or not np.isfinite(pred).all():
            raise ValueError("Invalid prediction array: " + name)
        pred = np.maximum(pred, 0.0)  # domain-only constraint; not performance tuned
        result = sample[["id", TARGET]].copy()
        result[TARGET] = pred[order]
        filepath = out_dir / (name + ".csv")
        result.to_csv(filepath, index=False)
        manifest.append({"name": name, "family": p["family"],
                         "n_estimators": p["n_estimators"],
                         "max_depth": p["max_depth"],
                         "learning_rate": p.get("learning_rate", "N/A"),
                         "file": filepath.name,
                         "hyperparameters": json.dumps(p, sort_keys=True)})
        print(f"Wrote {filepath.name}, {len(result)} predictions.", flush=True)

    pd.DataFrame(manifest).to_csv(out_dir / "manifest.csv", index=False)
    (out_dir / "configurations.json").write_text(json.dumps({
        "seed": SEED, "target_scale": SCALE, "metric": "Kaggle R2",
        "train_only_features": CAT_COLS + NUM_COLS,
        "configs": CONFIGS, "quality_evaluation_performed": False,
        "notes": "All 7 models refit on full supplied training data. No train/validation/test R2, CV, or residual analysis. Test opened only for final inference. The full training-feature mappings were reused from generate_submissions.py."
    }, indent=2))
    print("Finished. Zero performance evaluations performed.", flush=True)


if __name__ == "__main__":
    main()
