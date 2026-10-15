#!/usr/bin/env python3
"""Airfare Kaggle phase 3: fixed, training-EDA-informed boosting candidate submissions.

NO model evaluation, NO CV, NO leaderboard-based tuning, NO test EDA.
The test dataset is first read after ALL model fits, solely for inference.

Run:
python generate_phase3_boosting.py --train 'train(1).csv' --test 'test(1).csv' \
 --sample 'sample_submission(1).csv' --out phase3_boosting_submissions
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor

from generate_submissions import (
    TrainingOnlyFeatureBuilder, make_catboost_features, make_lightgbm_features,
    make_ohe_features, CAT_COLS, LGB_CAT_COLS, TARGET, SCALE, SEED,
)

# Different hypotheses, not alleged winners; every count and depth is finite.
# Model complexity controlled additionally through leaves, min-child counts,
# sampling, L1/L2 regularization, and minimum gains to split.
CONFIGS = {
    "lgb_650_d5_lr045": {
        "family": "LightGBM", "n_estimators": 650, "max_depth": 5,
        "num_leaves": 15, "learning_rate": 0.045,
        "min_child_samples": 65, "min_split_gain": 0.04,
        "reg_lambda": 20.0, "reg_alpha": 0.4,
        "subsample": 0.80, "subsample_freq": 1,
        "colsample_bytree": 0.80, "max_bin": 127,
    },
    "lgb_800_d5_lr035": {
        "family": "LightGBM", "n_estimators": 800, "max_depth": 5,
        "num_leaves": 15, "learning_rate": 0.035,
        "min_child_samples": 65, "min_split_gain": 0.04,
        "reg_lambda": 20.0, "reg_alpha": 0.4,
        "subsample": 0.80, "subsample_freq": 1,
        "colsample_bytree": 0.80, "max_bin": 127,
    },
    "lgb_750_d6_leaves23": {
        "family": "LightGBM", "n_estimators": 750, "max_depth": 6,
        "num_leaves": 23, "learning_rate": 0.035,
        "min_child_samples": 75, "min_split_gain": 0.06,
        "reg_lambda": 25.0, "reg_alpha": 0.6,
        "subsample": 0.80, "subsample_freq": 1,
        "colsample_bytree": 0.80, "max_bin": 127,
    },
    "xgb_750_d5_lr045": {
        "family": "XGBoost", "n_estimators": 750, "max_depth": 5,
        "max_leaves": 32, "learning_rate": 0.045,
        "min_child_weight": 8, "subsample": 0.80,
        "colsample_bytree": 0.85, "reg_lambda": 8.0,
        "reg_alpha": 0.35, "gamma": 0.04, "max_bin": 192,
    },
    "xgb_850_d7_regularized": {
        "family": "XGBoost", "n_estimators": 850, "max_depth": 7,
        "max_leaves": 48, "learning_rate": 0.030,
        "min_child_weight": 12, "subsample": 0.80,
        "colsample_bytree": 0.85, "reg_lambda": 15.0,
        "reg_alpha": 0.60, "gamma": 0.08, "max_bin": 192,
    },
    "cat_780_d6_lr040": {
        "family": "CatBoost", "iterations": 780, "depth": 6,
        "learning_rate": 0.040, "l2_leaf_reg": 15.0,
        "random_strength": 1.3, "rsm": 0.85,
        "bootstrap_type": "Bernoulli", "subsample": 0.82,
    },
    "xgb_800_d6_log1p": {
        "family": "XGBoost", "n_estimators": 800, "max_depth": 6,
        "max_leaves": 48, "learning_rate": 0.040,
        "min_child_weight": 6, "subsample": 0.80,
        "colsample_bytree": 0.85, "reg_lambda": 8.0,
        "reg_alpha": 0.08, "gamma": 0.015, "max_bin": 192,
        "target_transform": "log1p_price",
    },
}


def build_model(c):
    family = c["family"]
    p = {k: v for k, v in c.items() if k not in ("family", "target_transform")}
    if family == "LightGBM":
        return LGBMRegressor(**p, objective="regression", random_state=SEED, n_jobs=6, verbosity=-1)
    if family == "XGBoost":
        return XGBRegressor(**p, objective="reg:squarederror", tree_method="hist",
                            random_state=SEED, n_jobs=6, verbosity=0)
    if family == "CatBoost":
        return CatBoostRegressor(**p, loss_function="RMSE", random_seed=SEED,
                                 thread_count=6, verbose=False, allow_writing_files=False)
    raise ValueError(family)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--sample", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)

    # Enforce finite tree caps; no model scores or other estimations.
    for name, c in CONFIGS.items():
        n = c.get("n_estimators", c.get("iterations"))
        depth = c.get("max_depth", c.get("depth"))
        assert 1 <= n <= 900 and 1 <= depth <= 7, name
        assert c["learning_rate"] <= 0.05, name
        if c["family"] == "LightGBM":
            assert c["num_leaves"] <= 24 and c["min_child_samples"] >= 60
        if c["family"] == "XGBoost":
            assert c["min_child_weight"] >= 5 and c["max_leaves"] <= 48

    print("Frozen 7 fixed model configs. Opening TRAIN features ONLY.", flush=True)
    train = pd.read_csv(args.train)
    assert {"id", TARGET}.issubset(train.columns)
    target_raw = train[TARGET].to_numpy(dtype=np.float64)
    if not np.isfinite(target_raw).all() or np.any(target_raw < 0):
        raise ValueError("Expected finite nonnegative training ticket prices")
    builder = TrainingOnlyFeatureBuilder()
    f_train = builder.fit_transform(train.drop(columns=["id", TARGET]))
    del train
    # All categorical encoders / frequencies / duration mapping fitted on training ONLY.
    X_lgb, category_levels = make_lightgbm_features(f_train)
    X_cat = make_catboost_features(f_train)
    ohe_prep, X_ohe = make_ohe_features(f_train)
    del f_train
    y_scaled = target_raw / SCALE
    y_log = np.log1p(target_raw)
    trained = {}
    durations = {}
    for name, c in CONFIGS.items():
        start = time.perf_counter()
        n = c.get("n_estimators", c.get("iterations"))
        depth = c.get("max_depth", c.get("depth"))
        print(f"Training {name}: {c['family']}, {n} trees, depth {depth}", flush=True)
        reg = build_model(c)
        yy = y_log if c.get("target_transform") == "log1p_price" else y_scaled
        if c["family"] == "LightGBM":
            reg.fit(X_lgb, yy, categorical_feature=LGB_CAT_COLS)
        elif c["family"] == "CatBoost":
            reg.fit(X_cat, yy, cat_features=CAT_COLS)
        else:
            reg.fit(X_ohe, yy)
        trained[name] = reg
        durations[name] = round(time.perf_counter() - start, 3)
        print(f"  Done {name} (training only; no performance score)", flush=True)

    print("ALL MODELS TRAINED. Opening TEST for INFERENCE ONLY.", flush=True)
    test = pd.read_csv(args.test)
    sample = pd.read_csv(args.sample)
    necessary = set(["id", "airline", "flight", "source_city", "departure_time",
                     "stops", "arrival_time", "destination_city", "flight_class",
                     "duration", "days_left"])
    if not necessary.issubset(test.columns) or not {"id", "price"}.issubset(sample.columns):
        raise ValueError("Incorrect submission or test schema")
    f_test = builder.transform(test.drop(columns=["id", "price"], errors="ignore"))
    X_lgb_test, _ = make_lightgbm_features(f_test, category_levels)
    X_cat_test = make_catboost_features(f_test)
    X_ohe_test = ohe_prep.transform(f_test).tocsr().astype(np.float32)
    del f_test
    if (not test["id"].is_unique or not sample["id"].is_unique or len(test) != len(sample)):
        raise ValueError("Submission IDs must be unique and cover all test rows")
    indexer = pd.Series(np.arange(len(test)), index=test["id"]).reindex(sample["id"])
    if indexer.isna().any():
        raise ValueError("Test and sample IDs do not align")
    positions = indexer.to_numpy(dtype=int)

    manifest = []
    for name, c in CONFIGS.items():
        if c["family"] == "LightGBM":
            p = trained[name].predict(X_lgb_test)
        elif c["family"] == "CatBoost":
            p = trained[name].predict(X_cat_test)
        else:
            p = trained[name].predict(X_ohe_test)
        p = np.asarray(p, dtype=np.float64).ravel()
        if c.get("target_transform") == "log1p_price":
            p = np.expm1(p)
        else:
            p *= SCALE
        if len(p) != len(test) or not np.isfinite(p).all():
            raise ValueError("Nonfinite predictions or wrong size: " + name)
        # Domain-only constraint, NOT a tuned cutoff or calibration.
        p = np.maximum(p, 0.0)
        submission = sample[["id", "price"]].copy()
        submission["price"] = p[positions]
        path = output / f"{name}.csv"
        submission.to_csv(path, index=False)
        manifest.append({"filename": path.name, "family": c["family"],
                         "trees": c.get("n_estimators", c.get("iterations")),
                         "max_depth": c.get("max_depth", c.get("depth")),
                         "learning_rate": c["learning_rate"],
                         "target_transform": c.get("target_transform", "raw_price_div_10000"),
                         "train_seconds": durations[name],
                         "config": json.dumps(c, sort_keys=True)})
        print(f"Wrote {path.name}: {len(p)} predictions", flush=True)

    pd.DataFrame(manifest).to_csv(output / "manifest.csv", index=False)
    (output / "configs.json").write_text(json.dumps({"random_seed": SEED,
        "metric": "R2 (Kaggle); no local model scores computed",
        "trained_without_validation": True, "tested_models": False,
        "test_data_used": "inference only, after every fit finished",
        "preprocessing": "Existing EDA-driven TrainingOnlyFeatureBuilder from generate_submissions.py",
        "candidate_configurations": CONFIGS}, indent=2))
    print("DONE. No model-performance evaluations or test-feature EDA.", flush=True)


if __name__ == "__main__":
    main()
