#!/usr/bin/env python3
"""Describe protected training-only OOF results without fitting any models.

Can be rerun as family results finish. Reads only train and this study's result
JSON/OOF files. It never opens competition test/sample/submission files. The
original equal blend is fixed at 1/3 per family; no weight fitting occurs.
An optional registered exploratory CatBoost one-hot variant is reported along
with its fixed replacement blend. Original candidates are always retained.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/airfare-fresh-diagnostics-mpl")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/airfare-fresh-diagnostics-cache")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
Path(os.environ["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


ROOT = Path(__file__).resolve().parent
DEFAULT_TRAIN = Path("/workspace/attachments/085aeb37-413d-42c9-bb16-baafd968488d/train.csv")
FAMILIES = ["xgb", "lgb", "cat"]
OPTIONAL = ["cat_onehot64"]
ESTIMATORS = FAMILIES + OPTIONAL
BLEND = "equal_blend"
EXPLORATORY_BLEND = "equal_blend_onehot64"
BLENDS = {BLEND, EXPLORATORY_BLEND}
EXPLORATORY = {"cat_onehot64", EXPLORATORY_BLEND}
ORDER = FAMILIES + [BLEND] + OPTIONAL + [EXPLORATORY_BLEND]
LABELS = {"xgb": "XGBoost", "lgb": "LightGBM", "cat": "CatBoost", BLEND: "Original equal blend",
          "cat_onehot64": "Cat onehot64*", EXPLORATORY_BLEND: "Replacement blend*"}
COLORS = {"xgb": "#2166ac", "lgb": "#35978f", "cat": "#d95f02", BLEND: "#762a83",
          "cat_onehot64": "#e6ab02", EXPLORATORY_BLEND: "#e7298a"}


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [clean(v) for v in value]
    if isinstance(value, np.generic):
        return clean(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def save_json(path, value):
    path.write_text(json.dumps(clean(value), indent=2, allow_nan=False) + "\n")


def load_json(path):
    return json.loads(path.read_text())


def markdown_table(frame, columns=None, precision=5):
    columns = list(frame.columns) if columns is None else list(columns)
    def cell(value):
        if pd.isna(value):
            return "—"
        if isinstance(value, (float, np.floating)):
            return f"{value:,.{precision}f}"
        return str(value).replace("|", " / ")
    rows = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    rows.extend("| " + " | ".join(cell(v) for v in row) + " |" for row in frame[columns].itertuples(index=False, name=None))
    return "\n".join(rows)


def metrics(y, pred):
    y, pred = np.asarray(y, float), np.asarray(pred, float)
    error = pred - y
    varying = len(y) >= 2 and np.sum((y - y.mean()) ** 2) > 0
    return {"n": len(y), "r2": float(r2_score(y, pred)) if varying else np.nan,
            "rmse": float(np.sqrt(mean_squared_error(y, pred))),
            "mae": float(mean_absolute_error(y, pred)), "bias": float(error.mean()),
            "sse": float(np.sum(error ** 2))}


def scopes(results):
    """Flatten completed JSON trial archives; never inspect active SQLite DBs."""
    trials, stops, selected, missing = [], [], [], []
    for family in ESTIMATORS:
        family_dir = results / family
        if not family_dir.exists():
            missing.append(f"{family}: no completed result directory")
            continue
        for trial_path in sorted(family_dir.glob("*/*_all_trials.json")):
            scope = trial_path.parent.name
            trial_list = load_json(trial_path)
            selected_path = trial_path.with_name(trial_path.name.replace("_all_trials.json", "_selected.json"))
            chosen = load_json(selected_path) if selected_path.exists() else {}
            cap = int(chosen.get("signature", {}).get("cap", 1200))
            patience = int(chosen.get("signature", {}).get("patience", 70))
            for trial in trial_list:
                attrs = trial.get("attributes", {})
                raw = trial.get("parameters", {})
                actual = attrs.get("configuration", {})
                details = attrs.get("inner_splits", [])
                rounds = [int(d["rounds"]) for d in details]
                number = int(trial["number"])
                requested, effective = raw.get("leaves"), actual.get("leaves")
                record = {"family": family, "scope": scope, "trial": number,
                          "state": trial.get("state"), "selected": number == chosen.get("best_trial"),
                          "inner_tuning_mse": trial.get("value"), "feature_mode": attrs.get("mode", raw.get("feature_mode")),
                          "depth": actual.get("depth", raw.get("depth")),
                          "leaves_requested": requested, "leaves_effective": effective,
                          "leaves_clipped_by_depth": requested is not None and effective is not None and requested != effective,
                          "selected_rounds": attrs.get("selected_rounds"),
                          "inner_best_rounds_min": min(rounds) if rounds else None,
                          "inner_best_rounds_max": max(rounds) if rounds else None,
                          "inner_best_rounds_sd": float(np.std(rounds, ddof=1)) if len(rounds) > 1 else None,
                          "inner_best_at_ceiling": sum(n >= cap for n in rounds),
                          "inner_best_plus_patience_at_ceiling": sum(n + patience >= cap for n in rounds),
                          "ceiling": cap, "patience": patience, "seconds": attrs.get("seconds"),
                          "raw_parameters_json": json.dumps(raw, sort_keys=True),
                          "effective_configuration_json": json.dumps(actual, sort_keys=True)}
                trials.append(record)
                for d in details:
                    stops.append({"family": family, "scope": scope, "trial": number,
                                  "selected": record["selected"], "ceiling": cap, "patience": patience,
                                  "best_at_ceiling": int(d["rounds"]) >= cap,
                                  "best_plus_patience_at_ceiling": int(d["rounds"]) + patience >= cap,
                                  **d})
            if chosen:
                rounds = [int(d["rounds"]) for d in chosen.get("inner_splits", [])]
                raw = chosen.get("raw_optuna_parameters", {})
                actual = chosen.get("configuration", {})
                requested, effective = raw.get("leaves"), actual.get("leaves")
                selected.append({"family": family, "scope": scope,
                                 "best_trial": chosen.get("best_trial"), "trials_completed": chosen.get("trial_count"),
                                 "feature_mode": chosen.get("mode"), "rounds": chosen.get("rounds"),
                                 "inner_tuning_mse": chosen.get("tuning_mse"),
                                 "inner_best_rounds": json.dumps(rounds),
                                 "inner_best_rounds_min": min(rounds) if rounds else None,
                                 "inner_best_rounds_max": max(rounds) if rounds else None,
                                 "inner_best_rounds_sd": float(np.std(rounds, ddof=1)) if len(rounds) > 1 else None,
                                 "inner_best_at_ceiling": sum(n >= cap for n in rounds),
                                 "inner_best_plus_patience_at_ceiling": sum(n + patience >= cap for n in rounds),
                                 "ceiling": cap, "patience": patience,
                                 "leaves_requested": requested, "leaves_effective": effective,
                                 "leaves_clipped_by_depth": requested is not None and effective is not None and requested != effective,
                                 "round_policy": chosen.get("round_policy"),
                                 "configuration_json": json.dumps(actual, sort_keys=True)})
    # The optional representation ablation reuses original configurations and
    # performs no new Optuna search, so it has selected JSON but no trial archive.
    seen = {(record["family"], record["scope"]) for record in selected}
    for family in ESTIMATORS:
        family_dir = results / family
        for selected_path in sorted(family_dir.glob("*/*_selected.json")):
            scope = selected_path.parent.name
            if (family, scope) in seen:
                continue
            chosen = load_json(selected_path)
            cap = int(chosen.get("signature", {}).get("cap", 1200))
            patience = int(chosen.get("signature", {}).get("patience", 70))
            actual = chosen.get("configuration", {})
            raw = chosen.get("raw_optuna_parameters", {})
            override = chosen.get("representation_override", {})
            effective = actual.get("leaves")
            requested = raw.get("leaves", effective)
            rounds = [int(d["rounds"]) for d in chosen.get("inner_splits", [])]
            selected.append({"family": family, "scope": scope,
                             "best_trial": chosen.get("best_trial"), "trials_completed": chosen.get("trial_count", 0),
                             "original_optuna_trial_count": chosen.get("original_optuna_trial_count"),
                             "selection_origin": "reuses original CatBoost settings; rounds reselected" if family in OPTIONAL else "Optuna",
                             "feature_mode": chosen.get("mode"), "rounds": chosen.get("rounds"),
                             "inner_tuning_mse": chosen.get("tuning_mse"),
                             "inner_best_rounds": json.dumps(rounds),
                             "inner_best_rounds_min": min(rounds) if rounds else None,
                             "inner_best_rounds_max": max(rounds) if rounds else None,
                             "inner_best_rounds_sd": float(np.std(rounds, ddof=1)) if len(rounds) > 1 else None,
                             "inner_best_at_ceiling": sum(n >= cap for n in rounds),
                             "inner_best_plus_patience_at_ceiling": sum(n + patience >= cap for n in rounds),
                             "ceiling": cap, "patience": patience,
                             "leaves_requested": requested, "leaves_effective": effective,
                             "leaves_clipped_by_depth": requested is not None and effective is not None and requested != effective,
                             "round_policy": chosen.get("round_policy"),
                             "representation_override_json": json.dumps(override, sort_keys=True),
                             "configuration_json": json.dumps({**actual, **override}, sort_keys=True)})
            seen.add((family, scope))
    return pd.DataFrame(trials), pd.DataFrame(stops), pd.DataFrame(selected), missing


def bootstrap_groups(y, predictions, groups, draws, seed):
    """Resample entire itinerary groups and recompute the resample's target SST."""
    names = list(predictions)
    group_codes, unique_groups = pd.factorize(np.asarray(groups, str), sort=True)
    n_groups = len(unique_groups)
    n = np.bincount(group_codes).astype(float)
    sum_y = np.bincount(group_codes, weights=y)
    sum_y2 = np.bincount(group_codes, weights=y ** 2)
    sse = np.stack([np.bincount(group_codes, weights=(predictions[name] - y) ** 2) for name in names])
    rng = np.random.default_rng(seed)
    boot = np.empty((draws, len(names)))
    for i in range(draws):
        weights = rng.multinomial(n_groups, np.full(n_groups, 1 / n_groups))
        sample_n = weights @ n
        denominator = weights @ sum_y2 - (weights @ sum_y) ** 2 / sample_n
        boot[i] = 1 - (sse @ weights) / denominator
    estimates = {name: metrics(y, predictions[name])["r2"] for name in names}
    absolute = [{"model": name, "estimate": estimates[name],
                 "ci_low": np.quantile(boot[:, i], .025), "ci_high": np.quantile(boot[:, i], .975)}
                for i, name in enumerate(names)]
    paired = []
    for a, b in itertools.combinations(range(len(names)), 2):
        delta = boot[:, a] - boot[:, b]
        paired.append({"model_a": names[a], "model_b": names[b],
                       "delta_r2_a_minus_b": estimates[names[a]] - estimates[names[b]],
                       "ci_low": np.quantile(delta, .025), "ci_high": np.quantile(delta, .975),
                       "bootstrap_fraction_positive": np.mean(delta > 0)})
    return {"draws": draws, "seed": seed, "resampling_unit": "whole itinerary group",
            "n_groups": n_groups, "absolute": absolute, "paired": paired,
            "limitations": "Descriptive OOF prediction uncertainty only; no retraining, repeated tuning, fold-generation, or accumulated model-selection uncertainty. Confidence intervals are not submission eligibility gates."}, pd.DataFrame(boot, columns=names)


def main(train_path, results, out, draws=500, seed=20261021):
    train_path, results, out = Path(train_path), Path(results), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    train = pd.read_csv(train_path)
    assert train.id.is_unique and "price" in train
    y = train.price.to_numpy(float)
    raw = train.drop(columns=["id", "price"])
    expected_groups = pd.util.hash_pandas_object(raw.drop(columns=["airline", "duration"]).astype(str), index=False).astype(str).to_numpy()
    prefix = train.flight.str.split().str[0]
    route = train.source_city + ">" + train.destination_city
    trials, stops, selections, missing = scopes(results)
    trials.to_csv(out / "all_trials.csv", index=False)
    stops.to_csv(out / "all_trial_inner_stopping.csv", index=False)
    selections.to_csv(out / "selected_configurations.csv", index=False)

    predictions, summaries = {}, {}
    folds, group_ids, oof_id = None, None, None
    status = {}
    for family in ESTIMATORS:
        family_dir = results / family
        if not (family_dir / "oof.csv").exists() or not (family_dir / "summary.json").exists():
            complete_folds = len(list(family_dir.glob("outer*/outer_metrics.json")))
            status[family] = {"status": "pending", "outer_folds_completed": complete_folds}
            continue
        frame = pd.read_csv(family_dir / "oof.csv", dtype={"group": str})
        frame = frame.sort_values("row_position").reset_index(drop=True)
        assert len(frame) == len(train)
        assert np.array_equal(frame.row_position.to_numpy(), np.arange(len(train)))
        assert np.array_equal(frame.id.to_numpy(), train.id.to_numpy())
        assert np.array_equal(frame.price.to_numpy(float), y)
        assert np.array_equal(frame.group.to_numpy(str), expected_groups)
        assert frame.prediction.notna().all() and np.isfinite(frame.prediction).all()
        assert (frame.prediction >= 0).all()
        assert frame.groupby("group").fold.nunique().max() == 1
        current_folds = frame.fold.to_numpy(int)
        assert np.array_equal(np.unique(current_folds), np.arange(5))
        if folds is not None:
            assert np.array_equal(current_folds, folds)
            assert np.array_equal(frame.group.to_numpy(str), group_ids)
        else:
            folds, group_ids, oof_id = current_folds, frame.group.to_numpy(str), frame.id.to_numpy()
        summary = load_json(family_dir / "summary.json")
        pred = frame.prediction.to_numpy(float)
        computed = metrics(y, pred)
        assert np.isclose(summary["r2"], computed["r2"], atol=1e-10)
        assert np.isclose(summary["rmse"], computed["rmse"], atol=1e-6)
        predictions[family], summaries[family] = pred, summary
        status[family] = {"status": "complete", "outer_folds_completed": 5,
                          "exploratory_after_original_diagnostics": family in OPTIONAL}
    if all(f in predictions for f in FAMILIES):
        predictions[BLEND] = np.mean(np.stack([predictions[f] for f in FAMILIES]), axis=0)
        status[BLEND] = {"status": "complete", "weights": {f: 1 / 3 for f in FAMILIES}, "weight_search": False}
    else:
        status[BLEND] = {"status": "pending", "reason": "All three predeclared component OOF files are required."}
    replacement_components = ["xgb", "lgb", "cat_onehot64"]
    if all(f in predictions for f in replacement_components):
        predictions[EXPLORATORY_BLEND] = np.mean(np.stack([predictions[f] for f in replacement_components]), axis=0)
        status[EXPLORATORY_BLEND] = {"status": "complete", "weights": {f: 1 / 3 for f in replacement_components},
                                     "weight_search": False, "exploratory_after_original_diagnostics": True}
    else:
        status[EXPLORATORY_BLEND] = {"status": "optional_pending",
                                     "reason": "XGBoost, LightGBM and registered CatBoost onehot64 OOF files are required.",
                                     "exploratory_after_original_diagnostics": True}
    predictions = {name: predictions[name] for name in ORDER if name in predictions}

    manifest = {"training_sha256": hashlib.sha256(train_path.read_bytes()).hexdigest(),
                "training_rows": len(train), "results": str(results), "status": status,
                "kaggle_test_sample_or_submission_opened": False, "models_fitted": 0,
                "trials_archived": len(trials), "selections_archived": len(selections),
                "bootstrap_draws": draws, "bootstrap_seed": seed,
                "interpretation": "Original outer folds evaluate the whole family tuning procedure; selected hyperparameters may differ across folds. Optional onehot64 is exploratory after original diagnostics, on reused outer observations. Final dictionaries do not have separate untouched performance estimates.",
                "exploratory_candidates": sorted(EXPLORATORY),
                "optional_additional_optuna_trials": 0, "optional_planned_additional_fits": 18}
    if not predictions:
        save_json(out / "analysis_metadata.json", manifest)
        (out / "FINDINGS.md").write_text("# Training-only model diagnostics\n\nNo complete protected OOF family results are available yet. Trial/stopping/configuration tables have been written for completed tuning scopes. Rerun diagnostics.py after a family finishes. No models, competition test, sample, or historical predictions were accessed.\n")
        print(json.dumps(clean(manifest), indent=2))
        return

    model_rows, fold_rows, subgroup_rows, tail_rows, worst_rows = [], [], [], [], []
    top_price = y >= np.quantile(y, .99)
    # Price cutoffs define diagnostic bins only, not transforms or fitted features.
    cutoffs = np.quantile(y, [.5, .9, .95, .99])
    percentile_band = pd.cut(y, [-np.inf, *cutoffs, np.inf], labels=["0–50%", "50–90%", "90–95%", "95–99%", "99–100%"])
    subgroup_schemes = {
        "cabin": train.flight_class,
        "prefix": prefix,
        "cabin_prefix": train.flight_class + "|" + prefix,
        "route": route,
        "duration_missing": train.duration.isna().map({True: "missing", False: "observed"}),
        "airline_missing": train.airline.isna().map({True: "missing", False: "observed"}),
        "target_percentile": pd.Series(percentile_band.astype(str)),
    }
    for model, pred in predictions.items():
        pooled = metrics(y, pred)
        by_fold = []
        summary = summaries.get(model, {})
        fold_info = {int(d["fold"]): d for d in summary.get("folds", [])}
        for fold in sorted(np.unique(folds)):
            mask = folds == fold
            score = metrics(y[mask], pred[mask])
            info = fold_info.get(int(fold), {})
            tr2 = info.get("train_r2", np.nan)
            record = {"model": model, "fold": int(fold), **score,
                      "train_r2": tr2, "train_validation_gap": tr2 - score["r2"],
                      "n_fit": info.get("n_fit"), "refit_seconds": info.get("refit_seconds"),
                      "feature_mode": info.get("selected", {}).get("mode"),
                      "rounds": info.get("selected", {}).get("rounds")}
            by_fold.append(record)
            fold_rows.append(record)
        fold_r2 = np.array([d["r2"] for d in by_fold])
        train_r2 = summary.get("mean_train_r2", np.nan)
        model_rows.append({"model": model, "exploratory_after_original_diagnostics": model in EXPLORATORY,
                           **pooled, "mean_fold_r2": fold_r2.mean(),
                           "std_fold_r2": fold_r2.std(ddof=1), "worst_fold_r2": fold_r2.min(),
                           "mean_train_r2": train_r2,
                           "mean_train_validation_gap": train_r2 - fold_r2.mean(),
                           "outer_refit_seconds": sum(d.get("refit_seconds", 0) for d in summary.get("folds", [])) if summary else np.nan,
                           "train_gap_interpretation": "Not available: component train scores cannot be averaged to obtain blend R2." if model in BLENDS else "Mean training R2 minus mean outer-fold R2."})
        error = pred - y
        squared = error ** 2
        sse = squared.sum()
        n_worst = max(1, int(np.ceil(.01 * len(y))))
        worst_idx = np.argsort(squared)[-n_worst:]
        top_metrics = metrics(y[top_price], pred[top_price])
        tail_rows.append({"model": model, "target_p99_threshold": np.quantile(y, .99),
                          "top_1pct_target_n": int(top_price.sum()),
                          "top_1pct_target_sse_share": squared[top_price].sum() / sse,
                          "top_1pct_target_rmse": top_metrics["rmse"],
                          "top_1pct_target_mae": top_metrics["mae"], "top_1pct_target_bias": top_metrics["bias"],
                          "top_1pct_squared_error_rows_n": len(worst_idx),
                          "top_1pct_squared_error_sse_share": squared[worst_idx].sum() / sse,
                          "business_sse_share": squared[train.flight_class.to_numpy() == "Business"].sum() / sse})
        for grouping, labels in subgroup_schemes.items():
            for label in sorted(labels.astype(str).unique()):
                mask = labels.astype(str).to_numpy() == label
                subgroup_rows.append({"model": model, "grouping": grouping, "subgroup": label,
                                      **metrics(y[mask], pred[mask]), "overall_sse_share": squared[mask].sum() / sse})
        for pos in np.argsort(squared)[-40:][::-1]:
            worst_rows.append({"model": model, "row_position": int(pos), "fold": int(folds[pos]),
                               "prediction": pred[pos], "error_prediction_minus_price": error[pos],
                               "squared_error": squared[pos], "prefix": prefix.iloc[pos], "route": route.iloc[pos],
                               **train.iloc[pos].to_dict()})

    model_table, fold_table = pd.DataFrame(model_rows), pd.DataFrame(fold_rows)
    subgroup_table, tail_table = pd.DataFrame(subgroup_rows), pd.DataFrame(tail_rows)
    model_table.to_csv(out / "model_metrics.csv", index=False)
    fold_table.to_csv(out / "fold_metrics.csv", index=False)
    subgroup_table.to_csv(out / "subgroup_metrics.csv", index=False)
    tail_table.to_csv(out / "tail_metrics.csv", index=False)
    pd.DataFrame(worst_rows).to_csv(out / "worst_errors.csv", index=False)
    paired_rows = []
    for a, b in itertools.combinations(predictions, 2):
        fa = fold_table.loc[fold_table.model == a].sort_values("fold")
        fb = fold_table.loc[fold_table.model == b].sort_values("fold")
        for fold, delta in zip(fa.fold, fa.r2.to_numpy() - fb.r2.to_numpy()):
            paired_rows.append({"model_a": a, "model_b": b, "fold": int(fold), "delta_r2_a_minus_b": delta})
    paired_table = pd.DataFrame(paired_rows, columns=["model_a", "model_b", "fold", "delta_r2_a_minus_b"])
    paired_table.to_csv(out / "paired_fold_deltas.csv", index=False)
    residuals = pd.DataFrame({m: p - y for m, p in predictions.items()})
    residuals.corr().to_csv(out / "residual_correlations.csv")
    oof = pd.DataFrame({"row_position": np.arange(len(y)), "id": oof_id, "price": y,
                        "group": group_ids, "fold": folds, **predictions})
    oof.to_csv(out / "aligned_training_oof.csv", index=False)
    boot = None
    if draws > 0:
        boot, samples = bootstrap_groups(y, predictions, group_ids, draws, seed)
        save_json(out / "bootstrap_intervals.json", boot)
        samples.to_csv(out / "bootstrap_r2_draws.csv", index=False)

    # Include exact final dictionaries as they become available, without using
    # those models' in-sample scores as protected validation estimates.
    final_configurations = {}
    for family in ESTIMATORS:
        path = results / family / "full_training" / "final_configuration.json"
        if path.exists():
            config = load_json(path)
            final_configurations[family] = config
    save_json(out / "final_configurations_snapshot.json", final_configurations)
    manifest.update({"completed_oof_models": list(predictions), "unique_itinerary_groups": len(set(group_ids)),
                     "all_component_folds_identical": len(predictions) > 1,
                     "final_model_configurations_available": list(final_configurations)})
    save_json(out / "analysis_metadata.json", manifest)

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "figure.dpi": 150, "savefig.bbox": "tight"})
    names = list(predictions)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2))
    nonblend = [n for n in names if n not in BLENDS]
    for i, name in enumerate(names):
        f = fold_table.loc[fold_table.model == name]
        axes[0].scatter(np.full(len(f), i) + np.linspace(-.12, .12, len(f)), f.r2,
                        color=COLORS[name], alpha=.7, s=30)
        axes[0].scatter(i, model_table.set_index("model").loc[name, "r2"],
                        color=COLORS[name], marker="D", s=65, edgecolor="black", zorder=3)
        if name not in BLENDS:
            row = model_table.set_index("model").loc[name]
            j = nonblend.index(name)
            axes[1].plot([j, j], [row.mean_fold_r2, row.mean_train_r2], color=COLORS[name], lw=3)
            axes[1].scatter(j, row.mean_train_r2, color=COLORS[name], marker="^", s=60)
            axes[1].scatter(j, row.mean_fold_r2, color=COLORS[name], marker="o", s=45)
    axes[0].set_xticks(range(len(names)), [LABELS[n] for n in names])
    axes[0].set(ylabel="Original-price R²", title="Protected outer folds (dots); pooled OOF (diamonds)")
    axes[1].set_xticks(range(len(nonblend)), [LABELS[n] for n in nonblend])
    axes[1].set(ylabel="Original-price R²", title="Mean training (triangles) and outer-fold R² (circles)")
    for ax in axes:
        ax.grid(axis="y", alpha=.2)
        ax.tick_params(axis="x", labelrotation=18, labelsize=9)
    if any(n in EXPLORATORY for n in names):
        fig.suptitle("* Registered exploratory additions after original fit diagnostics; reused outer data")
    fig.tight_layout()
    fig.savefig(out / "outer_r2_and_training_gaps.png")
    plt.close(fig)

    plot_columns = 3 if len(names) > 4 else 2
    fig, axes = plt.subplots(2, plot_columns, figsize=(5 * plot_columns, 9), sharex=True, sharey=True)
    sample = np.random.default_rng(seed).choice(len(y), min(6000, len(y)), replace=False)
    for ax, name in zip(axes.flat, names):
        for klass, color in [("Economy", "#2166ac"), ("Business", "#d95f02")]:
            idx = sample[train.flight_class.to_numpy()[sample] == klass]
            ax.scatter(y[idx] / 1000, predictions[name][idx] / 1000, s=5, alpha=.25, color=color, label=klass, rasterized=True)
        limit = y.max() / 1000 * 1.03
        ax.plot([0, limit], [0, limit], color="black", lw=.8)
        ax.set(title=LABELS[name], xlabel="Actual training fare, thousands", ylabel="Outer OOF prediction, thousands", xlim=(0, limit), ylim=(0, limit))
        ax.grid(alpha=.15)
    for ax in list(axes.flat)[len(names):]:
        ax.set_visible(False)
    axes.flat[0].legend(frameon=False, markerscale=2)
    fig.suptitle("Held-out predictions; identical row sample for every procedure")
    fig.tight_layout()
    fig.savefig(out / "predicted_vs_actual.png")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    bands = ["0–50%", "50–90%", "90–95%", "95–99%", "99–100%"]
    for name in names:
        s = subgroup_table.query("model == @name and grouping == 'target_percentile'").set_index("subgroup").loc[bands]
        axes[0].plot(bands, s.rmse / 1000, marker="o", color=COLORS[name], label=LABELS[name])
        axes[1].plot(bands, s.bias / 1000, marker="o", color=COLORS[name], label=LABELS[name])
    axes[0].set(ylabel="Outer OOF RMSE, thousands", title="Errors by actual-price percentile")
    axes[1].set(ylabel="Prediction minus actual, thousands", title="Bias by actual-price percentile")
    axes[1].axhline(0, color="black", lw=.8)
    for ax in axes:
        ax.set_xlabel("Training target percentile band")
        ax.grid(alpha=.2)
        ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(out / "target_band_error_and_bias.png")
    plt.close(fig)

    s = subgroup_table.query("grouping == 'cabin_prefix' and n >= 30 and subgroup.str.startswith('Business|')", engine="python")
    heat = s.pivot(index="subgroup", columns="model", values="rmse").reindex(columns=names).sort_index()
    if len(heat):
        fig, ax = plt.subplots(figsize=(max(6, len(names) * 1.5), max(4, len(heat) * .4)))
        mat = ax.imshow(heat.to_numpy() / 1000, cmap="YlOrRd", aspect="auto")
        ax.set_xticks(range(len(names)), [LABELS[n] for n in names])
        ax.tick_params(axis="x", labelrotation=18, labelsize=9)
        n_by_group = s.drop_duplicates("subgroup").set_index("subgroup").n
        ax.set_yticks(range(len(heat)), [f"{idx.split('|')[1]} (n={n_by_group[idx]})" for idx in heat.index])
        for row in range(len(heat)):
            for col in range(len(names)):
                ax.text(col, row, f"{heat.iloc[row, col] / 1000:.1f}", ha="center", va="center", fontsize=9)
        ax.set_title("Business-cabin held-out RMSE by prefix")
        fig.colorbar(mat, ax=ax, label="Outer OOF RMSE, thousands")
        fig.tight_layout()
        fig.savefig(out / "business_prefix_rmse.png")
        plt.close(fig)

    table = model_table[["model", "r2", "rmse", "mae", "mean_fold_r2", "std_fold_r2", "worst_fold_r2", "mean_train_r2", "mean_train_validation_gap"]]
    report = ["# Training-only protected model diagnostics", "",
              "This report evaluates the three originally predeclared nested tuning procedures, not a single common hyperparameter dictionary. Each outer fold may choose different settings using only its inner training/tuning data. Original families share the same five itinerary-grouped outer folds. An optional registered CatBoost one-hot representation ablation and its replacement blend use those same folds and are explicitly exploratory after original fit diagnostics. No model was fitted by this analysis and no competition test, sample, or historical submission predictions were opened.", "",
              markdown_table(table), "",
              "R², RMSE, MAE and bias use original ticket-price units. Pooled OOF R² differs from the mean of fold R². Training gap is mean training R² minus mean outer-fold R². Neither equal blend has a measured training R² or gap: averaging component scores would be mathematically invalid.", "",
              "The fixed equal blend, when present, is exactly 1/3 XGBoost + 1/3 LightGBM + 1/3 CatBoost on the price scale. Its weights were not optimized. It uses the same held-out component predictions, and inherits their protected evaluation.", "",
              "The optional `cat_onehot64` changes only CatBoost one_hot_max_size from 2 to 64, reuses each originally selected feature mode/configuration, and reselects rounds on the exact original two inner splits before outer refitting. It introduces zero new Optuna trials, with 18 additional fits planned across five outer scopes and the final scope. Its motivation followed the original CatBoost fit/representation diagnostics, so its evaluation is exploratory on reused outer observations, not pristine confirmation. The original CatBoost and original blend are retained regardless of which scores higher. The additional fixed `equal_blend_onehot64` is exactly 1/3 XGBoost + 1/3 LightGBM + 1/3 cat_onehot64; neither weights nor component membership are searched.", "",
              "## Tail and cabin diagnostics", "",
              markdown_table(tail_table[["model", "top_1pct_target_n", "top_1pct_target_sse_share", "top_1pct_target_rmse", "top_1pct_target_bias", "top_1pct_squared_error_sse_share", "business_sse_share"]], precision=4), "",
              "Top-1% target share means the fraction of total squared error from the highest-priced training observations. Top-1% squared-error share instead sorts rows by their model errors; these are different sets. Negative bias means underprediction. Detailed cabin/prefix/route/missingness/target-band metrics and the 40 worst errors per procedure are saved as CSVs. Small subgroups can have unstable or negative R²; their RMSE, bias and support should be considered together.", "",
              "## Tuning and stopping audit", ""]
    if len(selections):
        report.extend([markdown_table(selections[["family", "scope", "trials_completed", "feature_mode", "rounds", "inner_best_rounds", "inner_best_at_ceiling", "leaves_requested", "leaves_effective"]], precision=0), ""])
        high_disp = selections.loc[(selections.inner_best_rounds_max - selections.inner_best_rounds_min) >= 300]
        report.append(f"Across {len(selections)} completed tuning scopes, {len(trials)} archived trials were inspected. {len(high_disp)} selected scopes have inner best-round ranges of at least 300 trees. This is a warning about round-transfer uncertainty, not an automatic rejection or permission to change the predeclared rule.")
        report.append("")
    report.extend(["`all_trials.csv` records every archived trial, its raw Optuna parameters, effective clipped leaf count, feature option, inner tuning objective and stop dispersion. `selected_configurations.csv` contains per-outer and available full-training choices. `all_trial_inner_stopping.csv` gives each inner split separately. `inner_best_at_ceiling` counts recorded best iterations exactly at the finite ceiling. The additional `best_plus_patience_at_ceiling` flag identifies counts close enough to the cap that the stopping budget may constrain the fit; actual trained iteration histories were not saved, so this is not presented as a measured training-cap hit count.", "",
                   "Final full-training dictionaries, when available, are copied to `final_configurations_snapshot.json`. Their in-sample fit scores are not new validation evidence. Original full-training tuning reruns the registered procedure on all labeled data; the optional representation ablation reuses original settings with fresh inner round selection. Outer scores describe these procedures at the outer-training sample size and do not independently validate each final dictionary. The optional ablation and its replacement blend remain exploratory even if their bootstrap intervals favor them.", "",
                   "## Comparisons and uncertainty", ""])
    if boot:
        report.extend([markdown_table(pd.DataFrame(boot["absolute"])), "",
                       markdown_table(pd.DataFrame(boot["paired"])), "",
                       f"Intervals use {draws} deterministic bootstrap resamples of whole itinerary groups (seed {seed}), recomputing target variance in every resample. They describe the current OOF predictions and do not include retraining, fold-generation or repeated-selection uncertainty. They are not historical-baseline promotion gates. Pairwise differences are descriptive; comparing families also creates selection optimism.", ""])
    else:
        report.append("Bootstrap disabled for this run.\n")
    report.extend(["Validation-supported candidates can be delivered even when a difference between two families is uncertain. None of these scores is a Kaggle score, and none guarantees absence of overfitting, a future score improvement, or a rank. The prior user-reported public R² of 0.95797 comes from different hidden observations and must not be directly compared with these local values.", "",
                   "The experiment remains bounded by its registered families/search spaces. A train–validation gap is an overfitting diagnostic, not a proof that a model is good or bad. Weak tail behavior, fold variability and unstable stopping counts remain explicit risks. Report every registered procedure instead of describing the best observed family as independently proven superior.", "",
                   "## Completion state", "", "```json", json.dumps(status, indent=2), "```", "",
                   "This analysis is idempotent and can be rerun as pending scopes/families finish. Partial reports do not silently invent missing models or redefine the blend. See `analysis_metadata.json` for the training hash and completion state."])
    (out / "FINDINGS.md").write_text("\n".join(report) + "\n")
    print(json.dumps(clean({"models": names, "metrics": model_table.to_dict(orient="records"),
                            "trial_count": len(trials), "selected_scopes": len(selections),
                            "final_configurations_available": list(final_configurations), "out": out}), indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--results", type=Path, default=ROOT / "results")
    parser.add_argument("--out", type=Path, default=ROOT / "analysis")
    parser.add_argument("--bootstrap", type=int, default=500, help="Whole-itinerary group bootstrap draws; 0 disables")
    parser.add_argument("--seed", type=int, default=20261021)
    args = parser.parse_args()
    assert args.bootstrap >= 0
    main(args.train, args.results, args.out, args.bootstrap, args.seed)
