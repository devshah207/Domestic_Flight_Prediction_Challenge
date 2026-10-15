"""Freeze three completed training-only studies, then perform final inference.

There is deliberately no model-training or model-selection operation here.
Run ``--freeze-only`` for review before supplying Kaggle inference inputs.
Inference reads test.csv once, after verifying the unchanged freeze manifest.
The exclusive marker intentionally prevents retries in an already-used output
directory, including after a failed inference attempt.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata as metadata
import io
import json
import math
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from features import TabularFeatures


ROOT = Path(__file__).resolve().parent
FAMILIES = ("xgb", "lgb", "cat")
OUTPUTS = {
    "xgb": "submission_xgboost.csv",
    "lgb": "submission_lightgbm.csv",
    "cat": "submission_catboost.csv",
    "equal_blend": "submission_equal_blend.csv",
}
VARIANT = "cat_onehot64"
EXTRA_OUTPUTS = {
    VARIANT: "submission_catboost_onehot64.csv",
    "equal_blend_onehot64": "submission_equal_blend_onehot64.csv",
}
EXPECTED_ROWS = 3898
SCALE = 10000.0
MANIFEST_NAME = "frozen_inference_manifest.json"
MARKER_NAME = "INFERENCE_STARTED.json"
AUDIT_NAME = "inference_audit.json"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def parameter_json(value: Any) -> Any:
    """Match study.py's serialization of estimator defaults such as missing=NaN."""
    if isinstance(value, dict):
        return {str(key): parameter_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [parameter_json(item) for item in value]
    if isinstance(value, np.generic):
        return parameter_json(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return "NaN"
    return value


def digest_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def digest_file(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    require(isinstance(value, dict), f"Expected a JSON object: {path.name}")
    return value


def exclusive_json(path: Path, value: dict[str, Any]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write("\n")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def finite_number(value: Any, label: str) -> float:
    require(
        isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value),
        f"Missing or non-finite numeric field: {label}",
    )
    return float(value)


def positive_integer(value: Any, label: str) -> int:
    require(isinstance(value, int) and not isinstance(value, bool) and value > 0,
            f"Expected a positive integer: {label}")
    return value


def validate_outer(summary: dict[str, Any], family: str, plan: dict[str, Any]) -> None:
    require(summary.get("family") == family, f"Outer summary family mismatch: {family}")
    for field in ("r2", "rmse", "mae", "bias", "mean_train_r2", "mean_fold_r2", "std_fold_r2", "mean_gap"):
        finite_number(summary.get(field), f"{family}.{field}")
    folds = summary.get("folds", [])
    require(len(folds) == 5, f"All five completed outer folds are required for {family}")
    require({row.get("fold") for row in folds} == set(range(5)),
            f"Outer fold identifiers must be exactly 0..4 for {family}")
    planned = {row["fold"]: row for row in plan["folds"]}
    for row in folds:
        label = f"{family}.fold{row['fold']}"
        for field in ("r2", "rmse", "mae", "bias", "train_r2", "refit_seconds"):
            finite_number(row.get(field), f"{label}.{field}")
        for field in ("n_fit", "n_validation"):
            positive_integer(row.get(field), f"{label}.{field}")
            require(row[field] == planned[row["fold"]][field], f"Fold size differs from frozen fold plan: {label}")


def checked_snapshot(results: Path, train: Path, include_variant: bool = False) -> tuple[dict[str, Any], dict[str, Any]]:
    """Validate completed artifacts using training files and metadata only."""
    plan_path = ROOT / "fold_plan.json"
    plan = read_json(plan_path)
    require(len(plan.get("folds", [])) == 5, "The fold plan must contain five outer folds")
    require({row.get("fold") for row in plan["folds"]} == set(range(5)), "Invalid fold plan identifiers")
    require(all(row.get("group_overlap") == 0 for row in plan["folds"]), "Fold plan reports group overlap")
    sources = {name: digest_file(ROOT / name) for name in ("study.py", "features.py", "inference.py")}
    require(plan.get("code_hashes") == {name: sources[name] for name in ("study.py", "features.py")},
            "Training source code differs from the fold plan")
    versions = plan.get("versions")
    require(isinstance(versions, dict) and bool(versions), "Missing package versions in fold plan")
    installed = {name: metadata.version(name) for name in versions}
    require(installed == versions, "Runtime package versions differ from the training fold plan")
    installed["joblib"] = metadata.version("joblib")
    train_hash = digest_file(train)
    models: dict[str, Any] = {}
    bundles: dict[str, Any] = {}
    training_rows = None
    for family in FAMILIES:
        model_path = results / family / "full_training" / "final_model.joblib"
        config_path = model_path.with_name("final_configuration.json")
        summary_path = results / family / "summary.json"
        config = read_json(config_path)
        summary = read_json(summary_path)
        validate_outer(summary, family, plan)
        require(config.get("family") == family, f"Final configuration family mismatch: {family}")
        require(config.get("training_sha256") == train_hash, f"Training data hash mismatch: {family}")
        require(config.get("test_accessed") is False, f"Training config must certify no test access: {family}")
        require(config.get("source_hashes") == {name: sources[name] for name in ("study.py", "features.py")},
                f"Final training source hashes mismatch: {family}")
        model_hash = digest_file(model_path)
        require(config.get("model_sha256") == model_hash, f"Serialized model hash mismatch: {family}")
        rows = positive_integer(config.get("training_rows"), f"{family}.training_rows")
        require(sum(row["n_validation"] for row in summary["folds"]) == rows,
                f"Outer validation coverage differs from final training row count: {family}")
        require(training_rows in (None, rows), "Models were trained on different row counts")
        training_rows = rows
        selection = config.get("selection", {})
        rounds = positive_integer(selection.get("rounds"), f"{family}.rounds")
        signature = selection.get("signature", {})
        cap = positive_integer(signature.get("cap"), f"{family}.round_ceiling")
        require(rounds <= cap <= 1200, f"Round count exceeds the predeclared finite ceiling: {family}")
        require(signature.get("family") == family and signature.get("code") == config["source_hashes"],
                f"Final tuning signature mismatch: {family}")
        require(selection.get("mode") in ("minimal", "context"), f"Invalid feature mode: {family}")
        require(signature.get("trials") == selection.get("trial_count") == 12,
                f"The predeclared 12-trial final selection is incomplete: {family}")
        finite_number(selection.get("tuning_mse"), f"{family}.tuning_mse")
        params = config.get("estimator_parameters", {})
        depth_key, leaf_key, rounds_key = (
            ("depth", "max_leaves", "iterations") if family == "cat" else
            ("max_depth", "num_leaves", "n_estimators") if family == "lgb" else
            ("max_depth", "max_leaves", "n_estimators")
        )
        depth = positive_integer(params.get(depth_key), f"{family}.depth")
        leaves = positive_integer(params.get(leaf_key), f"{family}.leaves")
        require(depth <= 7 and leaves <= min(48, 2 ** depth), f"Unexpected final model complexity: {family}")
        require(params.get(rounds_key) == rounds, f"Configured round count mismatch: {family}")
        # These artifacts are locally generated. Their hashes are checked before
        # deserialization, and all validation below still precedes test access.
        bundle = joblib.load(model_path)
        require(isinstance(bundle, dict), f"Invalid serialized model bundle: {family}")
        require(bundle.get("scale") == SCALE, f"Unexpected target scaling: {family}")
        require(bundle.get("selected") == selection, f"Serialized selection mismatch: {family}")
        features = bundle.get("features")
        require(isinstance(features, TabularFeatures), f"Unexpected feature transformer: {family}")
        require(features.family == family and features.mode == selection["mode"],
                f"Serialized feature mode/family mismatch: {family}")
        require(features.get_feature_names_out().tolist() == config.get("features"),
                f"Serialized feature columns mismatch: {family}")
        model = bundle.get("model")
        require(callable(getattr(model, "predict", None)), f"Model lacks prediction method: {family}")
        actual_params = model.get_all_params() if family == "cat" else model.get_params()
        require(canonical(parameter_json(actual_params)) == canonical(params),
                f"Serialized estimator parameters mismatch: {family}")
        actual_rounds = model.tree_count_ if family == "cat" else (
            model.booster_.current_iteration() if family == "lgb" else model.get_booster().num_boosted_rounds()
        )
        require(actual_rounds == rounds, f"Actual fitted tree count mismatch: {family}")
        bundles[family] = bundle
        models[family] = {
            "configuration": config,
            "configuration_sha256": digest_file(config_path),
            "model_file": str(model_path.relative_to(results)),
            "model_sha256": model_hash,
            "outer_summary_sha256": digest_file(summary_path),
            "outer_cv": {key: summary[key] for key in ("r2", "rmse", "mae", "mean_train_r2", "mean_gap")},
            "outer_folds_completed": 5,
        }
    if include_variant:
        module_path = ROOT / "cat_representation.py"
        addendum_path = ROOT / "CAT_REPRESENTATION_ADDENDUM.md"
        sources[module_path.name] = digest_file(module_path)
        model_path = results / VARIANT / "full_training" / "final_model.joblib"
        config_path = model_path.with_name("final_configuration.json")
        summary_path = results / VARIANT / "summary.json"
        config, summary = read_json(config_path), read_json(summary_path)
        validate_outer(summary, VARIANT, plan)
        expected_sources = {name: sources[name] for name in ("study.py", "features.py", module_path.name)}
        require(config.get("family") == VARIANT and config.get("base_family") == "cat", "Variant family mismatch")
        require(config.get("source_hashes") == expected_sources, "Variant source hashes differ")
        require(config.get("training_sha256") == train_hash and config.get("training_rows") == training_rows,
                "Variant training data differs")
        require(config.get("test_accessed") is False, "Variant config must certify training-only fitting")
        require(config.get("addendum_sha256") == digest_file(addendum_path), "Variant registration changed")
        original_path = results / "cat" / "full_training" / "cat_full_selected.json"
        require(config.get("original_selected_config_sha256") == digest_file(original_path),
                "Original Cat selection changed after variant fitting")
        selection = config.get("selection", {})
        original_selection = bundles["cat"]["selected"]
        require(selection.get("original_selection") == original_selection, "Variant did not reuse the original Cat selection")
        require(selection.get("configuration") == original_selection["configuration"]
                and selection.get("mode") == original_selection["mode"], "Variant changed the selected settings or features")
        require(selection.get("representation_override") == {"one_hot_max_size": 64}, "Unexpected representation change")
        require(selection.get("trial_count") == 0 and selection.get("new_optuna_trials") == 0,
                "Variant must not introduce another tuning grid")
        sig = selection.get("signature", {})
        require(sig.get("code") == expected_sources and sig.get("family") == VARIANT
                and sig.get("base_family") == "cat" and sig.get("cap") == 1200
                and sig.get("patience") == 70 and sig.get("seed") == 20263220,
                "Variant fitting signature differs from registration")
        rounds = positive_integer(selection.get("rounds"), "variant.rounds")
        require(100 <= rounds <= 1200 and len(selection.get("inner_splits", [])) == 2, "Invalid variant stopping policy")
        model_hash = digest_file(model_path)
        require(config.get("model_sha256") == model_hash, "Variant serialized artifact changed")
        bundle = joblib.load(model_path)
        require(bundle.get("selected") == selection and bundle.get("scale") == SCALE, "Variant bundle mismatch")
        features, estimator = bundle["features"], bundle["model"]
        require(isinstance(features, TabularFeatures) and features.family == "cat"
                and features.mode == selection["mode"], "Variant transformer changed")
        require(features.get_feature_names_out().tolist() == config.get("features"), "Variant feature columns changed")
        params = config.get("estimator_parameters", {})
        require(canonical(parameter_json(estimator.get_all_params())) == canonical(params), "Variant fitted parameters changed")
        require(params.get("one_hot_max_size") == 64 and params.get("iterations") == rounds
                and estimator.tree_count_ == rounds, "Variant threshold/rounds mismatch")
        depth, leaves = positive_integer(params.get("depth"), "variant.depth"), positive_integer(params.get("max_leaves"), "variant.leaves")
        require(depth <= 7 and leaves <= min(48, 2 ** depth), "Variant complexity exceeds registered bounds")
        excluded = {"one_hot_max_size", "iterations"}
        original_constructor = {k: v for k, v in bundles["cat"]["model"].get_params().items() if k not in excluded}
        variant_constructor = {k: v for k, v in estimator.get_params().items() if k not in excluded}
        require(canonical(parameter_json(original_constructor)) == canonical(parameter_json(variant_constructor)),
                "Variant changed constructor parameters beyond threshold and reselected rounds")
        bundles[VARIANT] = bundle
        models[VARIANT] = {
            "configuration": config, "configuration_sha256": digest_file(config_path),
            "model_file": str(model_path.relative_to(results)), "model_sha256": model_hash,
            "outer_summary_sha256": digest_file(summary_path),
            "outer_cv": {key: summary[key] for key in ("r2", "rmse", "mae", "mean_train_r2", "mean_gap")},
            "outer_folds_completed": 5, "interpretation": "exploratory representation follow-up on reused outer folds",
        }
    output_names = {**OUTPUTS, **(EXTRA_OUTPUTS if include_variant else {})}
    snapshot = {
        "schema_version": 1,
        "models": models,
        "training_sha256": train_hash,
        "training_rows": training_rows,
        "source_sha256": sources,
        "fold_plan_sha256": digest_file(plan_path),
        "package_versions": installed,
        "inference_policy": {
            "target_inverse_scale": SCALE,
            "individual_postprocessing": "clip each original-price prediction at zero",
            "ensemble": {"type": "fixed_arithmetic_mean_after_individual_clipping",
                         "weights": {family: 1 / 3 for family in FAMILIES}},
            "sample_id_order": "preserve the exact sample-submission id sequence; align test predictions by unique id",
            "schema": ["id", "price"],
            "rows": EXPECTED_ROWS,
            "test_content_reads": 1,
            "alternative_ensemble": ({"weights": {"xgb": 1 / 3, "lgb": 1 / 3, VARIANT: 1 / 3},
                                      "interpretation": "predeclared fixed replacement blend; exploratory"} if include_variant else None),
            "outputs": output_names,
            "selection": "final full-training procedures fixed before test access; no inference-time changes",
        },
    }
    canonical(snapshot)  # Reject non-standard NaN/Infinity before writing a freeze.
    return snapshot, bundles


def freeze_or_verify(out: Path, snapshot: dict[str, Any]) -> str:
    manifest_path = out / MANIFEST_NAME
    snapshot_hash = digest_bytes(canonical(snapshot))
    if manifest_path.exists():
        manifest = read_json(manifest_path)
        require(manifest.get("frozen_snapshot_sha256") == snapshot_hash,
                "Existing freeze differs from the current artifacts, sources, or configuration")
        require(canonical(manifest.get("frozen")) == canonical(snapshot), "Existing freeze content mismatch")
    else:
        exclusive_json(manifest_path, {
            "frozen_at_utc": utc_now(),
            "frozen_snapshot_sha256": snapshot_hash,
            "frozen": snapshot,
        })
    return digest_file(manifest_path)


def final_inference(test: Path, sample: Path, out: Path, bundles: dict[str, Any], manifest_hash: str) -> None:
    output_names = {**OUTPUTS, **(EXTRA_OUTPUTS if VARIANT in bundles else {})}
    require(test.resolve() != sample.resolve(), "Test and sample files must be distinct")
    require(not (out / MARKER_NAME).exists(), "This output directory has already started test inference")
    for name in list(output_names.values()) + [AUDIT_NAME]:
        require(not (out / name).exists(), f"Refusing to overwrite existing output: {name}")
    # Atomic exclusive creation protects the single test-content access even if
    # two inference commands are accidentally started at the same time.
    exclusive_json(out / MARKER_NAME, {
        "started_at_utc": utc_now(), "manifest_sha256": manifest_hash,
        "policy": "one attempt in this output directory; do not remove this marker to retry",
    })
    test_bytes = test.read_bytes()  # The only read of the competition test file.
    sample_bytes = sample.read_bytes()
    test_hash, sample_hash = digest_bytes(test_bytes), digest_bytes(sample_bytes)
    test_frame = pd.read_csv(io.BytesIO(test_bytes), dtype={"id": "string"})
    sample_frame = pd.read_csv(io.BytesIO(sample_bytes), dtype={"id": "string"})
    del test_bytes, sample_bytes
    require(list(sample_frame.columns) == ["id", "price"], "Sample submission must have columns id,price")
    require(set(test_frame.columns) == {"id", *TabularFeatures.RAW_COLUMNS},
            "Test predictor schema differs from the frozen feature schema")
    require(len(test_frame) == len(sample_frame) == EXPECTED_ROWS, "Expected exactly 3,898 test and sample rows")
    for label, frame in (("test", test_frame), ("sample", sample_frame)):
        require(frame["id"].notna().all() and frame["id"].is_unique, f"{label} ids must be nonmissing and unique")
    require(set(test_frame["id"]) == set(sample_frame["id"]), "Test and sample ids differ")
    positions = sample_frame["id"].map(pd.Series(np.arange(EXPECTED_ROWS), index=test_frame["id"]))
    require(positions.notna().all(), "Cannot align test predictions to sample ids")
    alignment = positions.to_numpy(dtype=int)
    predictions: dict[str, np.ndarray] = {}
    for family in bundles:
        bundle = bundles[family]
        matrix = bundle["features"].transform(test_frame)
        raw = np.asarray(bundle["model"].predict(matrix), dtype=float).reshape(-1)
        require(raw.shape == (EXPECTED_ROWS,) and np.isfinite(raw).all(), f"Invalid prediction shape or values: {family}")
        prices = raw * SCALE
        require(np.isfinite(prices).all(), f"Non-finite original-price predictions: {family}")
        predictions[family] = np.maximum(0.0, prices)[alignment]
    predictions["equal_blend"] = sum(predictions[family] / 3 for family in FAMILIES)
    if VARIANT in bundles:
        predictions["equal_blend_onehot64"] = (predictions["xgb"] + predictions["lgb"] + predictions[VARIANT]) / 3
    output_hashes = {}
    for name, prices in predictions.items():
        require(prices.shape == (EXPECTED_ROWS,) and np.isfinite(prices).all() and (prices >= 0).all(),
                f"Submission validity check failed: {name}")
        frame = pd.DataFrame({"id": sample_frame["id"], "price": prices})
        require(list(frame.columns) == ["id", "price"] and frame["id"].equals(sample_frame["id"]),
                f"Submission schema/id order check failed: {name}")
        output = out / output_names[name]
        with output.open("x", encoding="utf-8", newline="") as handle:
            frame.to_csv(handle, index=False)
        output_hashes[output.name] = digest_file(output)
    exclusive_json(out / AUDIT_NAME, {
        "completed_at_utc": utc_now(), "manifest_sha256": manifest_hash,
        "test_sha256": test_hash, "sample_sha256": sample_hash,
        "test_content_reads": 1, "rows_per_submission": EXPECTED_ROWS,
        "checks": ["exact id,price schema", "unique nonmissing ids", "exact sample id order",
                   "3,898 rows", "finite nonnegative prices", "frozen artifact/config/source hashes"],
        "submission_sha256": output_hashes,
        "model_training_performed": False, "model_changes_after_test_access": False,
    })


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True, help="Fresh study results directory")
    parser.add_argument("--train", type=Path, required=True, help="Training file, for hash verification only")
    parser.add_argument("--test", type=Path, help="Competition test file; read only after freeze")
    parser.add_argument("--sample", type=Path, help="Competition sample submission; read only after freeze")
    parser.add_argument("--out", type=Path, required=True, help="New directory for the freeze and four submissions")
    parser.add_argument("--freeze-only", action="store_true", help="Validate/save the complete freeze without inference")
    parser.add_argument("--include-cat-onehot64", action="store_true", help="Also freeze the registered exploratory variant and fixed replacement blend")
    args = parser.parse_args()
    if not args.freeze_only:
        if args.test is None or args.sample is None:
            parser.error("Supply --test and --sample for final inference, or use --freeze-only")
        require(args.test.resolve() != args.train.resolve(), "Test and training inputs must be distinct")
        require(args.sample.resolve() != args.train.resolve(), "Sample and training inputs must be distinct")
    args.out.mkdir(parents=True, exist_ok=True)
    snapshot, bundles = checked_snapshot(args.results, args.train, args.include_cat_onehot64)
    manifest_hash = freeze_or_verify(args.out, snapshot)
    if args.freeze_only:
        print(f"Frozen {len(bundles)} completed procedures: {args.out / MANIFEST_NAME}")
        return
    final_inference(args.test, args.sample, args.out, bundles, manifest_hash)
    print(f"Wrote {len(OUTPUTS) + (len(EXTRA_OUTPUTS) if args.include_cat_onehot64 else 0)} validated submissions and inference audit to {args.out}")


if __name__ == "__main__":
    main()
