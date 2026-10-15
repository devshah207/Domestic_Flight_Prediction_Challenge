"""Single pre-registered, exploratory CatBoost one-hot representation ablation.

No competition test data is read. No Optuna search is performed. Each original
CatBoost-selected configuration is reused, changing only one_hot_max_size=64;
round counts are reselected on exactly its original two inner splits.
See CAT_REPRESENTATION_ADDENDUM.md for the registration and interpretation.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata as metadata
import json
from pathlib import Path
import time
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

from features import TabularFeatures
import study


ROOT = Path(__file__).resolve().parent
VARIANT = "cat_onehot64"
BASE_FAMILY = "cat"
ONE_HOT_MAX_SIZE = 64
INNER_OUTER_SEED = 20262220
INNER_FULL_SEED = 20263220
INTERPRETATION = (
    "Exploratory hybrid procedure: original 12-trial CatBoost configurations "
    "are reused; one_hot_max_size alone changes to 64 and rounds are reselected. "
    "Earlier outer-fold diagnostics motivated this follow-up, so these reused "
    "outer folds are sensitivity evidence, not a pristine independent test."
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def read_json(path: Path) -> dict[str, Any]:
    obj = json.loads(path.read_text())
    require(isinstance(obj, dict), f"Expected a JSON object: {path}")
    return obj


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(study.plain(value), handle, indent=2, allow_nan=False)
        handle.write("\n")


def signature(raw: pd.DataFrame, y: np.ndarray) -> dict[str, str]:
    return {
        "row_hash": hashlib.sha256(pd.util.hash_pandas_object(raw, index=False).values.tobytes()).hexdigest(),
        "target_hash": hashlib.sha256(y.tobytes()).hexdigest(),
    }


def selected_paths(results: Path) -> dict[str, Path]:
    paths = {f"outer{i}": results / "cat" / f"outer{i}" / f"cat_outer{i}_selected.json" for i in range(5)}
    paths["full"] = results / "cat" / "full_training" / "cat_full_selected.json"
    return paths


def preflight(results: Path) -> dict[str, Any]:
    """Check all original selections before any data-dependent fit can start."""
    paths = selected_paths(results)
    original_final = results / "cat" / "full_training" / "final_configuration.json"
    addendum = ROOT / "CAT_REPRESENTATION_ADDENDUM.md"
    fold_plan = ROOT / "fold_plan.json"
    required = list(paths.values()) + [original_final, addendum, fold_plan]
    missing = [str(path) for path in required if not path.is_file()]
    require(not missing, "Original CatBoost sources are not ready; no fits started. Missing:\n" + "\n".join(missing))
    source_hashes = {name: study.sha(ROOT / name) for name in ("study.py", "features.py", "cat_representation.py")}
    original_hashes = {name: source_hashes[name] for name in ("study.py", "features.py")}
    plan = read_json(fold_plan)
    require(plan.get("code_hashes") == original_hashes, "Original sources differ from their fold plan")
    require(study.CAP == 1200 and study.PATIENCE == 70 and study.SCALE == 10000.0,
            "The registered cap, patience or target scale changed")
    require(study.OUTER_SEED == 20261020 and study.FAMILIES.index("cat") == 2,
            "The registered outer/inner seed mapping changed")
    for package, version in plan.get("versions", {}).items():
        require(metadata.version(package) == version, f"Runtime version differs from original training: {package}")
    sources = {}
    for scope, path in paths.items():
        original = read_json(path)
        sig = original.get("signature", {})
        seed = INNER_FULL_SEED if scope == "full" else INNER_OUTER_SEED + int(scope.removeprefix("outer"))
        expected_study = "cat_full" if scope == "full" else f"cat_{scope}"
        require(original.get("study") == expected_study, f"Wrong original selection identity: {path}")
        require(sig.get("family") == "cat" and sig.get("seed") == seed, f"Original selection seed/family mismatch: {scope}")
        require(sig.get("code") == original_hashes, f"Original selected-configuration source hashes changed: {scope}")
        require(sig.get("cap") == 1200 and sig.get("patience") == 70 and sig.get("trials") == 12,
                f"Original stopping/search protocol mismatch: {scope}")
        require(original.get("trial_count") == 12 and original.get("mode") in ("minimal", "context"),
                f"Original 12-trial selection is incomplete: {scope}")
        require(len(original.get("inner_splits", [])) == 2, f"Expected two original stopping splits: {scope}")
        require(100 <= original.get("rounds", 0) <= 1200, f"Invalid original final round count: {scope}")
        sources[scope] = {"path": str(path.relative_to(results)), "sha256": study.sha(path), "selection": original}
    final = read_json(original_final)
    require(final.get("family") == "cat" and final.get("test_accessed") is False,
            "Original final configuration must be a training-only CatBoost fit")
    require(final.get("selection") == sources["full"]["selection"], "Original final and selected configurations disagree")
    params = final.get("estimator_parameters", {})
    require(params.get("one_hot_max_size") == 2, "The original fitted one-hot setting is not 2")
    require(params.get("ctr_target_border_count") == 1, "The registered original CTR-border observation changed")
    require(params.get("max_ctr_complexity") == 1, "The registered original CTR complexity changed")
    require(params.get("boosting_type") == "Plain", "The registered original boosting type changed")
    return {
        "sources": sources,
        "source_hashes": source_hashes,
        "addendum_sha256": study.sha(addendum),
        "fold_plan_sha256": study.sha(fold_plan),
        "original_final_configuration_sha256": study.sha(original_final),
        "original_final": final,
        "original_ctr_metadata": {key: params.get(key) for key in (
            "one_hot_max_size", "ctr_target_border_count", "max_ctr_complexity",
            "ctr_leaf_count_limit", "boosting_type", "simple_ctr", "combinations_ctr",
        )},
    }


def validate_training(raw: pd.DataFrame, y: np.ndarray, source: dict[str, Any]) -> list[tuple[np.ndarray, np.ndarray]]:
    folds = study.splits(raw, y, study.OUTER_SEED)
    partitions = {f"outer{i}": (raw.iloc[tr].reset_index(drop=True), y[tr]) for i, (tr, _) in enumerate(folds)}
    partitions["full"] = (raw, y)
    # Validate all six partitions before starting the first fit, not midway
    # through a costly experiment after a mismatched source is discovered.
    for scope, (part, target) in partitions.items():
        actual = signature(part, target)
        original = source["sources"][scope]["selection"]["signature"]
        require(all(actual[key] == original[key] for key in actual), f"Training partition differs from original selection: {scope}")
    return folds


def inner_rounds(raw: pd.DataFrame, y: np.ndarray, original: dict[str, Any], seed: int) -> tuple[int, list[dict[str, Any]]]:
    """Only reselect rounds; preserve the original selected parameters/mode."""
    records = []
    for index, (tr, va) in enumerate(study.splits(raw, y, seed)[:2]):
        features = TabularFeatures(original["mode"], "cat")
        xt = features.fit_transform(raw.iloc[tr])
        xv = features.transform(raw.iloc[va])
        estimator = study.model("cat", original["configuration"], study.CAP, stopping=True)
        estimator.set_params(one_hot_max_size=ONE_HOT_MAX_SIZE)
        constructor = estimator.get_params()
        start = time.perf_counter()
        study.fit(estimator, "cat", features, xt, y[tr], xv, y[va])
        predicted = study.predict(estimator, xv)
        rounds = study.rounds_used(estimator, "cat")
        require(1 <= rounds <= study.CAP, "CatBoost returned an invalid stopping iteration")
        records.append(dict(
            split=index, rounds=rounds, n_fit=len(tr), n_tune=len(va),
            fit_positions_sha256=hashlib.sha256(np.asarray(tr, dtype=np.int64).tobytes()).hexdigest(),
            tune_positions_sha256=hashlib.sha256(np.asarray(va, dtype=np.int64).tobytes()).hexdigest(),
            fit_seconds=time.perf_counter() - start,
            constructor_parameters=constructor,
            resolved_parameters=estimator.get_all_params(),
            **study.scores(y[va], predicted),
        ))
    selected = max(100, min(study.CAP, int(np.median([record["rounds"] for record in records]))))
    return selected, records


def selection(raw: pd.DataFrame, y: np.ndarray, scope: str, source: dict[str, Any]) -> dict[str, Any]:
    origin = source["sources"][scope]
    original = origin["selection"]
    seed = INNER_FULL_SEED if scope == "full" else INNER_OUTER_SEED + int(scope.removeprefix("outer"))
    rounds, records = inner_rounds(raw, y, original, seed)
    return dict(
        study=f"{VARIANT}_{scope}", family=VARIANT, base_family=BASE_FAMILY,
        best_trial=original["best_trial"], best_trial_interpretation="identifier from the original 12-trial selection; no new trial",
        mode=original["mode"], configuration=original["configuration"],
        representation_override={"one_hot_max_size": ONE_HOT_MAX_SIZE},
        rounds=rounds, inner_splits=records,
        original_selection=original, original_selected_config_file=origin["path"],
        original_selected_config_sha256=origin["sha256"],
        original_optuna_trial_count=12, trial_count=0, new_optuna_trials=0,
        round_policy="same two original inner80/20 splits; median stops; floor100 ceiling1200; refit100%",
        signature=dict(**signature(raw, y), code=source["source_hashes"], seed=seed,
                       family=VARIANT, base_family=BASE_FAMILY, trials=0,
                       original_trials=12, cap=study.CAP, patience=study.PATIENCE),
        interpretation=INTERPRETATION,
    )


def refit(raw: pd.DataFrame, y: np.ndarray, selected: dict[str, Any]) -> tuple[TabularFeatures, Any, np.ndarray, float]:
    features = TabularFeatures(selected["mode"], "cat")
    matrix = features.fit_transform(raw)
    estimator = study.model("cat", selected["configuration"], selected["rounds"])
    estimator.set_params(one_hot_max_size=ONE_HOT_MAX_SIZE)
    start = time.perf_counter()
    study.fit(estimator, "cat", features, matrix, y)
    predicted = study.predict(estimator, matrix)
    return features, estimator, predicted, time.perf_counter() - start


def outer(frame: pd.DataFrame, raw: pd.DataFrame, y: np.ndarray,
          folds: list[tuple[np.ndarray, np.ndarray]], dest: Path, source: dict[str, Any]) -> None:
    oof = np.full(len(y), np.nan)
    fold_ids = np.full(len(y), -1)
    records = []
    for index, (tr, va) in enumerate(folds):
        scope = f"outer{index}"
        part = raw.iloc[tr].reset_index(drop=True)
        selected = selection(part, y[tr], scope, source)
        write_json(dest / scope / f"{VARIANT}_{scope}_selected.json", selected)
        features, estimator, train_predictions, seconds = refit(part, y[tr], selected)
        predictions = study.predict(estimator, features.transform(raw.iloc[va]))
        oof[va], fold_ids[va] = predictions, index
        record = dict(
            fold=index, family=VARIANT, base_family=BASE_FAMILY, selected=selected,
            train_r2=float(r2_score(y[tr], train_predictions)), n_fit=len(tr), n_validation=len(va),
            refit_seconds=seconds, constructor_parameters=estimator.get_params(),
            resolved_parameters=estimator.get_all_params(), **study.scores(y[va], predictions),
        )
        records.append(record)
        write_json(dest / scope / "outer_metrics.json", record)
        print("OUTER", VARIANT, index, "R2", record["r2"], "train", record["train_r2"], "rounds", selected["rounds"], flush=True)
    require(np.isfinite(oof).all() and (fold_ids >= 0).all(), "Incomplete or invalid variant OOF predictions")
    table = pd.DataFrame(dict(row_position=np.arange(len(y)), id=frame.id, price=y,
                             group=study.group_ids(raw), fold=fold_ids, prediction=oof))
    with (dest / "oof.csv").open("x", encoding="utf-8", newline="") as handle:
        table.to_csv(handle, index=False)
    summary = dict(
        family=VARIANT, base_family=BASE_FAMILY, **study.scores(y, oof), folds=records,
        mean_train_r2=float(np.mean([record["train_r2"] for record in records])),
        mean_fold_r2=float(np.mean([record["r2"] for record in records])),
        std_fold_r2=float(np.std([record["r2"] for record in records], ddof=1)),
        mean_gap=float(np.mean([record["train_r2"] - record["r2"] for record in records])),
        evaluation=INTERPRETATION, original_optuna_trials_per_scope=12, new_optuna_trials=0,
        completed_model_fits=15,
        fixed_alternative_blend={"xgb": 1 / 3, "lgb": 1 / 3, VARIANT: 1 / 3},
        source_hashes=source["source_hashes"],
    )
    write_json(dest / "summary.json", summary)
    print("SUMMARY", VARIANT, summary["r2"], flush=True)


def final_training(raw: pd.DataFrame, y: np.ndarray, train: Path, dest: Path, source: dict[str, Any]) -> None:
    selected = selection(raw, y, "full", source)
    full = dest / "full_training"
    write_json(full / f"{VARIANT}_full_selected.json", selected)
    features, estimator, predicted, seconds = refit(raw, y, selected)
    artifact = full / "final_model.joblib"
    with artifact.open("xb") as handle:
        joblib.dump(dict(features=features, model=estimator, selected=selected, scale=study.SCALE), handle, compress=3)
    write_json(full / "final_configuration.json", dict(
        family=VARIANT, base_family=BASE_FAMILY, selection=selected,
        estimator_parameters=estimator.get_all_params(), constructor_parameters=estimator.get_params(),
        features=features.get_feature_names_out().tolist(), training_rows=len(y),
        training_sha256=study.sha(train), model_sha256=study.sha(artifact),
        source_hashes=source["source_hashes"], addendum_sha256=source["addendum_sha256"],
        original_selected_config_sha256=source["sources"]["full"]["sha256"],
        original_ctr_metadata=source["original_ctr_metadata"],
        full_train_r2=float(r2_score(y, predicted)), refit_seconds=seconds,
        completed_model_fits=3, interpretation=INTERPRETATION, test_accessed=False,
    ))
    print("FINAL_FROZEN", VARIANT, selected["rounds"], selected["mode"], flush=True)


def output_preflight(dest: Path, stage: str) -> None:
    expected = []
    if stage in ("outer", "all"):
        expected += [dest / "summary.json", dest / "oof.csv"]
        for index in range(5):
            expected += [dest / f"outer{index}" / "outer_metrics.json",
                         dest / f"outer{index}" / f"{VARIANT}_outer{index}_selected.json"]
    if stage in ("final", "all"):
        expected += [dest / "full_training" / name for name in (
            f"{VARIANT}_full_selected.json", "final_configuration.json", "final_model.joblib")]
    existing = [str(path) for path in expected if path.exists()]
    require(not existing, "Refusing to overwrite variant outputs; no fits started:\n" + "\n".join(existing))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("outer", "final", "all"), required=True)
    parser.add_argument("--train", type=Path, default=study.TRAIN)
    parser.add_argument("--results", type=Path, default=ROOT / "results")
    args = parser.parse_args()
    source = preflight(args.results)
    dest = args.results / VARIANT
    output_preflight(dest, args.stage)
    frame, raw, y = study.data(args.train)
    require(study.sha(args.train) == source["original_final"]["training_sha256"], "Original training data SHA differs")
    folds = validate_training(raw, y, source)
    dest.mkdir(parents=True, exist_ok=True)
    provenance = dict(
        family=VARIANT, base_family=BASE_FAMILY, interpretation=INTERPRETATION,
        training_sha256=study.sha(args.train), source_hashes=source["source_hashes"],
        addendum_sha256=source["addendum_sha256"], fold_plan_sha256=source["fold_plan_sha256"],
        original_final_configuration_sha256=source["original_final_configuration_sha256"],
        original_selected_configurations={scope: {"file": origin["path"], "sha256": origin["sha256"]}
                                          for scope, origin in source["sources"].items()},
        original_ctr_metadata=source["original_ctr_metadata"],
        one_change={"one_hot_max_size": {"original": 2, "variant": 64}},
        fit_budget={"outer": 15, "final": 3, "total": 18},
        fixed_alternative_blend={"xgb": 1 / 3, "lgb": 1 / 3, VARIANT: 1 / 3},
    )
    provenance_path = dest / "source_provenance.json"
    if provenance_path.exists():
        require(read_json(provenance_path) == provenance, "Variant provenance changed; no fits started")
    else:
        write_json(provenance_path, provenance)
    if args.stage in ("outer", "all"):
        outer(frame, raw, y, folds, dest, source)
    if args.stage in ("final", "all"):
        final_training(raw, y, args.train, dest, source)


if __name__ == "__main__":
    main()
