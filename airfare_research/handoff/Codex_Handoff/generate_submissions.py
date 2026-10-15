#!/usr/bin/env python3
"""Training-only EDA-driven airfare regressors, with no model evaluation.

IMPORTANT: all model choices and preprocessing are frozen before test.csv is opened.
No validation metrics, CV, scoring, parameter searches, leaderboard fitting, or test EDA.

Usage:
 python generate_submissions.py --train train.csv --test test.csv \
   --sample sample_submission.csv --out outputs
"""
import argparse
import json
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from scipy import sparse
from xgboost import XGBRegressor

SEED = 42
TARGET = "price"
SCALE = 10_000.0  # stabilizes magnitude-sensitive L1/gamma regularization for tree methods
CAT_COLS = [
    "airline_fixed", "flight", "flight_prefix", "source_city", "destination_city",
    "departure_time", "arrival_time", "stops", "flight_class",
    "route", "route_stops", "route_class", "prefix_class", "dep_arr"
]
NUM_COLS = ["duration_filled", "duration_missing", "days_left", "flight_number", "airline_missing", "stops_numeric"]
OHE_CAT_COLS = [c for c in CAT_COLS if c != "flight"]
LGB_CAT_COLS = [c for c in CAT_COLS if c != "flight"] + ["flight_rare"]

# Frozen configurations. All boosting/bagging trees <500 and have finite depth.
MODEL_CONFIGS = {
    "catboost_depth5": {"family": "CatBoost", "iterations": 360, "depth": 5, "learning_rate": .055,
        "l2_leaf_reg": 10.0, "random_strength": 1.0, "rsm": .9,
        "bootstrap_type": "Bayesian", "bagging_temperature": .55},
    "catboost_depth4": {"family": "CatBoost", "iterations": 440, "depth": 4, "learning_rate": .048,
        "l2_leaf_reg": 12.0, "random_strength": 1.3, "rsm": .9,
        "bootstrap_type": "Bayesian", "bagging_temperature": .7},
    "catboost_depth6": {"family": "CatBoost", "iterations": 380, "depth": 6, "learning_rate": .047,
        "l2_leaf_reg": 15.0, "random_strength": 1.6, "rsm": .85,
        "bootstrap_type": "Bernoulli", "subsample": .82},
    "xgboost_depth3": {"family": "XGBoost", "n_estimators": 340, "max_depth": 3,
        "learning_rate": .065, "min_child_weight": 10, "subsample": .83,
        "colsample_bytree": .80, "reg_lambda": 15.0, "reg_alpha": .3,
        "gamma": .025, "max_bin": 192},
    "xgboost_depth4": {"family": "XGBoost", "n_estimators": 410, "max_depth": 4,
        "learning_rate": .047, "min_child_weight": 18, "subsample": .80,
        "colsample_bytree": .85, "reg_lambda": 22.0, "reg_alpha": .6,
        "gamma": .05, "max_bin": 192},
    "lightgbm_leaves15": {"family": "LightGBM", "n_estimators": 360, "max_depth": 5,
        "num_leaves": 15, "learning_rate": .055, "min_child_samples": 65,
        "min_split_gain": .04, "reg_lambda": 20.0, "reg_alpha": .4,
        "subsample": .80, "subsample_freq": 1, "colsample_bytree": .80,
        "max_bin": 127},
    "lightgbm_leaves9": {"family": "LightGBM", "n_estimators": 400, "max_depth": 4,
        "num_leaves": 9, "learning_rate": .050, "min_child_samples": 90,
        "min_split_gain": .035, "reg_lambda": 18.0, "reg_alpha": .4,
        "subsample": .85, "subsample_freq": 1, "colsample_bytree": .85,
        "max_bin": 127},
    "random_forest": {"family": "RandomForest", "n_estimators": 280, "max_depth": 13,
        "min_samples_leaf": 5, "min_samples_split": 12, "max_features": .80,
        "bootstrap": True, "max_samples": .82, "max_leaf_nodes": 700},
    "extra_trees": {"family": "ExtraTrees", "n_estimators": 320, "max_depth": 14,
        "min_samples_leaf": 5, "min_samples_split": 12, "max_features": .75,
        "bootstrap": False, "max_leaf_nodes": 800},
    "ridge_ohe": {"family": "Ridge", "alpha": 35.0, "max_iter": 10000,
        "solver": "lsqr", "tol": 1e-5},
}
# Chosen in advance, NOT tuned on test scores, OOF predictions, or a leaderboard.
BLENDS = {
    "blend_boosting_diverse": {"catboost_depth5": .45, "lightgbm_leaves15": .35, "xgboost_depth3": .20},
    "blend_shallow_regularized": {"catboost_depth4": .50, "catboost_depth5": .25, "lightgbm_leaves9": .25},
}

class TrainingOnlyFeatureBuilder:
    """Learns only unsupervised mappings from the labeled training-set FEATURES."""
    def fit(self, x):
        x = x.copy()
        x["flight_prefix"] = x["flight"].astype(str).str.split().str[0]
        observed_airline = x.dropna(subset=["airline"])
        self.prefix_to_airline = observed_airline.groupby("flight_prefix")["airline"].agg(
            lambda s: s.mode().iloc[0]
        ).to_dict()
        observed_duration = x.dropna(subset=["duration"])
        self.duration_global_median = float(observed_duration["duration"].median())
        self.duration_group_tables = []
        # Each hierarchy is chosen from training-only feature consistency, not target.
        keys_with_required_count = [
            (["flight", "source_city", "destination_city", "stops"], 2),
            (["flight", "source_city", "destination_city"], 2),
            (["flight", "stops"], 2),
            (["flight"], 2),
            (["source_city", "destination_city", "stops", "flight_class"], 5),
            (["source_city", "destination_city", "stops"], 5),
            (["source_city", "destination_city"], 5),
            (["stops"], 5),
        ]
        for keys, min_count in keys_with_required_count:
            grp = observed_duration.groupby(keys, dropna=False)["duration"].agg(["median", "count"])
            medians = grp.loc[grp["count"] >= min_count, "median"]
            self.duration_group_tables.append((keys, medians))
        # Group infrequent flight IDs without looking at target prices.
        flight_counts = x["flight"].value_counts(dropna=False)
        self.frequent_flights = set(flight_counts[flight_counts >= 12].index.astype(str))
        return self

    def transform(self, x):
        # Take only feature columns; 'id' is intentionally excluded as mechanical.
        x = x.copy().reset_index(drop=True)
        out = pd.DataFrame(index=x.index)
        for col in ["flight", "source_city", "destination_city", "departure_time",
                    "arrival_time", "stops", "flight_class"]:
            out[col] = x[col].fillna("Unknown").astype(str)
        out["flight_prefix"] = out["flight"].str.split().str[0].fillna("Unknown")
        out["flight_number"] = pd.to_numeric(
            out["flight"].str.extract(r"(\d+)\s*$", expand=False), errors="coerce"
        ).fillna(-1.0).astype(float)
        out["airline_missing"] = x["airline"].isna().astype(int)
        out["airline_fixed"] = x["airline"].fillna(out["flight_prefix"].map(self.prefix_to_airline)).fillna("Unknown").astype(str)
        out["days_left"] = pd.to_numeric(x["days_left"], errors="coerce").fillna(-1.0).astype(float)
        out["stops_numeric"] = out["stops"].map({"zero": 0, "one": 1, "two_or_more": 2}).fillna(-1).astype(float)
        out["duration_missing"] = x["duration"].isna().astype(int)
        current = pd.to_numeric(x["duration"], errors="coerce").astype(float).copy()
        for keys, table in self.duration_group_tables:
            need = current.isna()
            if not need.any():
                break
            if len(keys) == 1:
                new_values = out.loc[need, keys[0]].map(table)
            else:
                idx = pd.MultiIndex.from_frame(out.loc[need, keys])
                new_values = pd.Series(table.reindex(idx).to_numpy(), index=out.index[need])
            current.loc[need] = current.loc[need].fillna(new_values)
        out["duration_filled"] = current.fillna(self.duration_global_median).astype(float)
        out["route"] = out["source_city"] + " -> " + out["destination_city"]
        out["route_stops"] = out["route"] + " / " + out["stops"]
        out["route_class"] = out["route"] + " / " + out["flight_class"]
        out["prefix_class"] = out["flight_prefix"] + " / " + out["flight_class"]
        out["dep_arr"] = out["departure_time"] + " -> " + out["arrival_time"]
        out["flight_rare"] = out["flight"].where(out["flight"].isin(self.frequent_flights), "__RARE_OR_NEW__")
        return out

    def fit_transform(self, x):
        return self.fit(x).transform(x)


def make_ohe_features(train_features):
    # Note: all sklearn estimators below are trained on these precomputed TRAIN-only features.
    # Test is transformed later using frozen OneHotEncoders (no fitting on test).
    prep = ColumnTransformer(
        transformers=[
            ("regular_categories", OneHotEncoder(handle_unknown="ignore", sparse_output=True), OHE_CAT_COLS),
            ("flight_rare", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=12,
                                          sparse_output=True), ["flight"]),
            ("numerical", "passthrough", NUM_COLS),
        ],
        sparse_threshold=1.0,
    )
    return prep, prep.fit_transform(train_features).tocsr().astype(np.float32)


def make_lightgbm_features(feat, categories=None):
    z = feat[LGB_CAT_COLS + NUM_COLS].copy()
    if categories is None:
        categories = {c: pd.Index(z[c].astype(str).unique()) for c in LGB_CAT_COLS}
    for c in LGB_CAT_COLS:
        z[c] = pd.Categorical(z[c], categories=categories[c])
    return z, categories


def make_catboost_features(feat):
    z = feat[CAT_COLS + NUM_COLS].copy()
    for c in CAT_COLS:
        z[c] = z[c].fillna("Unknown").astype(str)
    return z


def assert_limits():
    for name, p in MODEL_CONFIGS.items():
        f = p["family"]
        if f == "CatBoost":
            assert 1 <= p["iterations"] < 500 and 1 <= p["depth"] <= 8
        elif f in ("LightGBM", "XGBoost", "RandomForest", "ExtraTrees"):
            assert 1 <= p["n_estimators"] < 500
            assert isinstance(p["max_depth"], int) and 1 <= p["max_depth"] <= 15
        if f == "LightGBM":
            assert p["num_leaves"] <= 2 ** p["max_depth"]


def build_model(p):
    family = p["family"]
    q = {k: v for k, v in p.items() if k != "family"}
    if family == "CatBoost":
        return CatBoostRegressor(**q, loss_function="RMSE", eval_metric="R2",
                                 random_seed=SEED, thread_count=6, verbose=False,
                                 allow_writing_files=False)
    if family == "XGBoost":
        return XGBRegressor(**q, objective="reg:squarederror", tree_method="hist",
                            random_state=SEED, n_jobs=6)
    if family == "LightGBM":
        return LGBMRegressor(**q, objective="regression", random_state=SEED,
                             n_jobs=6, verbosity=-1)
    if family == "RandomForest":
        return RandomForestRegressor(**q, random_state=SEED, n_jobs=6)
    if family == "ExtraTrees":
        return ExtraTreesRegressor(**q, random_state=SEED, n_jobs=6)
    if family == "Ridge":
        return Ridge(**q)
    raise ValueError(family)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default="train.csv")
    parser.add_argument("--test", default="test.csv")
    parser.add_argument("--sample", default="sample_submission.csv")
    parser.add_argument("--out", default="kaggle_regression_outputs")
    args = parser.parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    assert_limits()
    print("All hyperparameters, feature rules and blend weights frozen. Reading training features only.", flush=True)
    train = pd.read_csv(args.train)
    assert TARGET in train.columns and "id" in train.columns
    y = train[TARGET].to_numpy(dtype=float) / SCALE
    x_train = train.drop(columns=[TARGET, "id"])
    builder = TrainingOnlyFeatureBuilder()
    f_train = builder.fit_transform(x_train)
    del train, x_train
    cat_train = make_catboost_features(f_train)
    lgb_train, lgb_categories = make_lightgbm_features(f_train)
    ohe_prep, ohe_train = make_ohe_features(f_train)
    # Ridge uses standardized numerical columns, with the scale fitted on training only.
    # For all other models, numeric features are left in raw units for tree splits.
    ridge_scale = StandardScaler()
    ohe_train_scaled = sparse.hstack([
        ohe_train[:, :-len(NUM_COLS)],
        sparse.csr_matrix(ridge_scale.fit_transform(f_train[NUM_COLS]).astype(np.float32))
    ], format="csr")
    models = {}
    for name, p in MODEL_CONFIGS.items():
        print(f"Training {name}: {p['family']}...", flush=True)
        model = build_model(p)
        family = p["family"]
        if family == "CatBoost":
            model.fit(cat_train, y, cat_features=CAT_COLS)
        elif family == "LightGBM":
            model.fit(lgb_train, y, categorical_feature=LGB_CAT_COLS)
        elif family == "Ridge":
            model.fit(ohe_train_scaled, y)
        else:
            model.fit(ohe_train, y)
        models[name] = model
        print(f"  Done: {name} (no scores computed)", flush=True)

    print("ALL TRAINING FINISHED. First test.csv read occurs now, for inference only.", flush=True)
    test = pd.read_csv(args.test)
    sample = pd.read_csv(args.sample)
    assert "id" in test.columns and TARGET in sample.columns and "id" in sample.columns
    # Schema-validity checks only: do not run test statistics or EDA.
    assert all(c in test.columns for c in ["id", "flight", "airline", "source_city",
            "destination_city", "departure_time", "arrival_time", "stops", "flight_class",
            "duration", "days_left"])
    # Train features were inferred without ever inspecting test features.
    f_test = builder.transform(test.drop(columns=["id", TARGET], errors="ignore"))
    cat_test = make_catboost_features(f_test)
    lgb_test, _ = make_lightgbm_features(f_test, lgb_categories)
    ohe_test = ohe_prep.transform(f_test).tocsr().astype(np.float32)
    ohe_test_scaled = sparse.hstack([
        ohe_test[:, :-len(NUM_COLS)],
        sparse.csr_matrix(ridge_scale.transform(f_test[NUM_COLS]).astype(np.float32))
    ], format="csr")

    # Map by ID only to meet the submission file's requested row ordering.
    id_reference = pd.Index(test["id"])
    if len(id_reference) != len(sample) or not id_reference.is_unique:
        raise ValueError("Unexpected ID count or duplicate test IDs.")
    predictions = {}
    for name, p in MODEL_CONFIGS.items():
        m = models[name]
        if p["family"] == "CatBoost":
            pred = m.predict(cat_test)
        elif p["family"] == "LightGBM":
            pred = m.predict(lgb_test)
        elif p["family"] == "Ridge":
            pred = m.predict(ohe_test_scaled)
        else:
            pred = m.predict(ohe_test)
        pred = np.asarray(pred, dtype=float).reshape(-1) * SCALE
        if len(pred) != len(test) or not np.isfinite(pred).all():
            raise ValueError(f"Invalid prediction dimensions or non-finite values: {name}")
        # Economic domain restriction only (zero is the minimum feasible fare).
        pred = np.maximum(pred, 0.0)
        predictions[name] = pred
    for blend, members in BLENDS.items():
        assert abs(sum(members.values()) - 1.0) < 1e-12
        predictions[blend] = sum(predictions[name] * weight for name, weight in members.items())

    submission_ids = sample["id"].to_numpy()
    reindexer = pd.Series(np.arange(len(test)), index=id_reference).reindex(submission_ids)
    if reindexer.isna().any():
        raise ValueError("Some sample submission IDs are absent from the test file.")
    ordering = reindexer.to_numpy(dtype=int)
    manifest = []
    for name, pred in predictions.items():
        submission = sample[["id", TARGET]].copy()
        submission[TARGET] = pred[ordering]
        file = out_dir / f"{name}.csv"
        submission.to_csv(file, index=False)
        if name in MODEL_CONFIGS:
            p = MODEL_CONFIGS[name]
            manifest.append({"name": name, "type": p["family"],
                 "trees": p.get("n_estimators", p.get("iterations", "N/A")),
                 "max_depth": p.get("max_depth", p.get("depth", "N/A")),
                 "hyperparameters": json.dumps(p, sort_keys=True), "file": file.name})
        else:
            manifest.append({"name": name, "type": "Fixed-weight blend", "trees": "N/A",
                "max_depth": "N/A", "hyperparameters": json.dumps(BLENDS[name]), "file": file.name})
        print(f"Wrote {file.name}: {len(submission)} predictions", flush=True)
    pd.DataFrame(manifest).to_csv(out_dir / "model_manifest.csv", index=False)
    (out_dir / "model_configurations.json").write_text(json.dumps({
        "seed": SEED, "target_scale": SCALE, "scores_computed": False,
        "training_only_features": CAT_COLS + NUM_COLS + ["flight_rare"],
        "models": MODEL_CONFIGS, "blends": BLENDS,
        "metric": "R2 (maximize; use squared-error objectives for fitting)"
    }, indent=2))
    print("Done. No model assessment/score was computed.", flush=True)


if __name__ == "__main__":
    main()
