#!/usr/bin/env python3
"""Freeze exact-850 airfare refits, then perform one final Kaggle inference.

The baseline is always emitted. ``blend_equal850`` additionally emits a frozen
50/50 blend of the baseline and native-categorical XGBoost models. This script
makes no model-selection decision; the caller must select using train-only
validation evidence before invoking it.

Use --freeze-only to fit, serialize, and inspect the complete frozen procedure
without reading either inference CSV. Invoke the same command without that flag
to reuse those artifacts for inference. A persistent marker prevents another
inference or refreeze in the same output directory after test access begins.
"""
from __future__ import annotations

import argparse
from io import BytesIO
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd

from research import BASE, Features, SCALE, make_model
from historical_features import NUM_COLS, OHE_CAT_COLS

EXPECTED_ROWS = 3898
FREEZE_FILE = 'frozen_final_manifest.json'
ACCESS_MARKER = 'KAGGLE_TEST_ACCESSED.json'
CONTROL_FILE = 'xgb850_control_refit.csv'
BLEND_FILE = 'submission_blend_equal850.csv'
CODE_FILES = ('finalize.py', 'research.py', 'historical_features.py')


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def json_value(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f'Unsupported manifest value type: {type(value).__name__}')


def write_json(path, data):
    path = Path(path)
    temporary = path.with_name(path.name + '.writing')
    temporary.write_text(json.dumps(data, indent=2, sort_keys=True,
                                    default=json_value) + '\n')
    temporary.replace(path)


def file_identity(path):
    # Metadata only. Inference data content remains unread until after freezing.
    info = path.stat()
    return dict(path=str(path), size_bytes=info.st_size, mtime_ns=info.st_mtime_ns)


def code_hashes():
    root = Path(__file__).resolve().parent
    return {name: sha256_file(root / name) for name in CODE_FILES}


def package_versions():
    versions = {'python': sys.version.split()[0]}
    for package in ('numpy', 'pandas', 'scipy', 'scikit-learn', 'xgboost-cpu',
                    'xgboost', 'joblib'):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            continue
    return versions


def assert_exact_850(model):
    parameters = model.get_params()
    for key, value in BASE.items():
        if parameters.get(key) != value:
            raise RuntimeError(f'Model differs from the frozen 850-tree configuration: {key}')
    if parameters.get('early_stopping_rounds') is not None:
        raise RuntimeError('Early stopping is not part of this exact-850 finalizer.')
    if len(model.get_booster().get_dump()) != 850:
        raise RuntimeError('The final model must contain exactly 850 trees.')


def verify_frozen(manifest, args, out):
    if manifest.get('schema_version') != 1 or manifest.get('selection') != args.selected:
        raise RuntimeError('Existing freeze does not match the requested selection.')
    if manifest['code_sha256'] != code_hashes():
        raise RuntimeError('Research code changed after freezing. Do not infer with a changed procedure.')
    if manifest['versions'] != package_versions():
        raise RuntimeError('Package versions changed after freezing.')
    if manifest['training_data']['path'] != str(args.train):
        raise RuntimeError('Training path differs from the frozen input.')
    if manifest['training_data']['sha256'] != sha256_file(args.train):
        raise RuntimeError('Training data changed after freezing.')
    for name, path in (('test', args.test), ('sample', args.sample)):
        if manifest['inference_inputs'][name] != file_identity(path):
            raise RuntimeError(f'{name} file identity changed after freezing.')
    for item in manifest['models']:
        artifact = out / item['artifact']
        if not artifact.is_file() or sha256_file(artifact) != item['artifact_sha256']:
            raise RuntimeError('A frozen model artifact is missing or changed.')


def fit_and_freeze(args, out):
    initial_code = code_hashes()
    train_hash = sha256_file(args.train)
    inference_inputs = {'test': file_identity(args.test),
                        'sample': file_identity(args.sample)}
    frame = pd.read_csv(args.train)
    if not {'id', 'price'}.issubset(frame.columns):
        raise ValueError('Training CSV requires id and price columns.')
    if not frame.id.is_unique or frame.id.isna().any():
        raise ValueError('Training IDs must be unique and present.')
    y = frame.price.to_numpy(dtype=float)
    if not np.isfinite(y).all() or (y <= 0).any():
        raise ValueError('Training target must contain finite positive ticket prices.')
    raw = frame.drop(columns=['id', 'price'])
    required = {'airline', 'flight', 'source_city', 'departure_time', 'stops',
                'arrival_time', 'destination_city', 'flight_class',
                'duration', 'days_left'}
    if set(raw.columns) != required:
        raise ValueError('Training predictors do not match the historical feature schema.')
    modes = ['baseline'] + (['native'] if args.selected == 'blend_equal850' else [])
    records = []
    for mode in modes:
        print(f'Fitting full training data: {mode}, exactly 850 rounds.', flush=True)
        features = Features(mode)
        matrix = features.fit_transform(raw)
        model = make_model(mode, rounds=850)
        model.fit(matrix, y / SCALE)
        assert_exact_850(model)
        matrix_columns = (features.prep.get_feature_names_out().tolist()
                          if mode == 'baseline' else matrix.columns.tolist())
        artifact = out / f'frozen_{mode}_850.joblib'
        joblib.dump({'features': features, 'model': model,
                     'raw_feature_columns': raw.columns.tolist(),
                     'target_scale': SCALE, 'feature_mode': mode}, artifact,
                    compress=3)
        resolved = json.loads(model.get_booster().save_config())
        resolved_file = f'frozen_{mode}_850_booster_config.json'
        write_json(out / resolved_file, resolved)
        records.append(dict(
            mode=mode, artifact=artifact.name,
            artifact_sha256=sha256_file(artifact),
            booster_config_file=resolved_file,
            estimator_parameters=model.get_params(),
            resolved_booster_config=resolved,
            raw_feature_columns=raw.columns.tolist(),
            matrix_feature_columns=matrix_columns,
            numeric_features=list(NUM_COLS),
            categorical_features=(list(OHE_CAT_COLS) +
                                  (['flight'] if mode == 'baseline' else ['flight_rare'])),
            native_category_levels=(features.levels if mode == 'native' else None)))
    if initial_code != code_hashes() or train_hash != sha256_file(args.train):
        raise RuntimeError('Code or training data changed while fitting; no freeze was created.')
    manifest = dict(
        schema_version=1, created_utc=utc_now(), selection=args.selected,
        training_data=dict(path=str(args.train), sha256=train_hash, rows=len(frame)),
        inference_inputs=inference_inputs,
        inference_content_hash_policy='Compute SHA256 during the single post-freeze inference read.',
        code_sha256=initial_code, versions=package_versions(),
        training_seed=42, target='price', target_scale=SCALE,
        expected_submission_rows=EXPECTED_ROWS,
        prediction_policy=dict(
            component_prediction='price/10000 prediction multiplied by 10000, then clipped at zero',
            blend_weights=({'baseline': 0.5, 'native': 0.5}
                           if args.selected == 'blend_equal850' else None),
            id_alignment='exact sample ID order', additional_adjustment=None),
        feature_policy=dict(
            implementation='historical_features.TrainingOnlyFeatureBuilder; research.Features',
            baseline_ohe='train-fitted OHE; ordinary unknown ignore; flight min_frequency12, infrequent_if_exist; float32 CSR',
            duration='historical eight-level minimum-count median hierarchy, then global median',
            airline='training-prefix modal airline fill; missingness flag retained',
            id_excluded=True, all_mappings_fit_on_training_only=True,
            native='same numeric/features, flight grouped at12; train-fitted categories; unseen categories missing; max_cat_threshold16; max_cat_to_onehot1'),
        outputs=[CONTROL_FILE] + ([BLEND_FILE] if args.selected == 'blend_equal850' else []),
        models=records)
    write_json(out / FREEZE_FILE, manifest)
    verify_frozen(manifest, args, out)
    return manifest


def final_inference(manifest, args, out):
    verify_frozen(manifest, args, out)
    bundles = {}
    for item in manifest['models']:
        bundle = joblib.load(out / item['artifact'])
        assert_exact_850(bundle['model'])
        bundles[item['mode']] = bundle
    # Mark access before reading. The marker persists even when schema/inference
    # fails, so an inspected test set cannot trigger another model fit here.
    marker = out / ACCESS_MARKER
    descriptor = os.open(marker, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    with os.fdopen(descriptor, 'w') as handle:
        json.dump(dict(status='access_started', utc=utc_now(),
                       frozen_manifest_sha256=sha256_file(out / FREEZE_FILE)), handle)
    # Exactly one read of each inference input; hashes are recorded without a
    # second content read. No fitting/model selection occurs below this point.
    test_bytes = args.test.read_bytes()
    sample_bytes = args.sample.read_bytes()
    test = pd.read_csv(BytesIO(test_bytes))
    sample = pd.read_csv(BytesIO(sample_bytes))
    raw_columns = bundles['baseline']['raw_feature_columns']
    if set(test.columns) != {'id', *raw_columns}:
        raise ValueError('Kaggle test schema does not match frozen training predictors.')
    if list(sample.columns) != ['id', 'price']:
        raise ValueError('Sample submission must have exactly id,price columns.')
    if len(test) != EXPECTED_ROWS or len(sample) != EXPECTED_ROWS:
        raise ValueError('Expected exactly 3898 test and sample rows.')
    if (not test.id.is_unique or not sample.id.is_unique or
            test.id.isna().any() or sample.id.isna().any()):
        raise ValueError('Test and sample IDs must be unique and present.')
    positions = pd.Series(np.arange(len(test)), index=test.id).reindex(sample.id)
    if positions.isna().any():
        raise ValueError('Test and sample IDs do not align.')
    order = positions.to_numpy(dtype=int)
    predictions = {}
    for mode, bundle in bundles.items():
        matrix = bundle['features'].transform(test[raw_columns])
        prediction = np.asarray(bundle['model'].predict(matrix), dtype=float).reshape(-1) * SCALE
        if prediction.shape != (EXPECTED_ROWS,) or not np.isfinite(prediction).all():
            raise ValueError('A frozen model produced an invalid prediction array.')
        predictions[mode] = np.maximum(prediction, 0.)
    outputs = {CONTROL_FILE: predictions['baseline']}
    if args.selected == 'blend_equal850':
        outputs[BLEND_FILE] = 0.5 * predictions['baseline'] + 0.5 * predictions['native']
    for filename, prediction in outputs.items():
        if not np.isfinite(prediction).all() or (prediction < 0).any():
            raise ValueError('Submission predictions must be finite and nonnegative.')
        submission = sample[['id', 'price']].copy()
        submission['price'] = prediction[order]
        submission.to_csv(out / filename, index=False)
        print(f'Wrote {filename}: validated id,price schema and 3898 aligned rows.', flush=True)
    audit = dict(
        status='inference_complete', utc=utc_now(),
        frozen_manifest_sha256=sha256_file(out / FREEZE_FILE),
        test_sha256=hashlib.sha256(test_bytes).hexdigest(),
        sample_sha256=hashlib.sha256(sample_bytes).hexdigest(),
        output_sha256={name: sha256_file(out / name) for name in outputs},
        schema_valid=True, id_alignment_valid=True,
        finite_nonnegative_predictions=True, rows=EXPECTED_ROWS)
    write_json(out / 'final_inference_audit.json', audit)
    write_json(marker, audit)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train', type=Path, required=True)
    parser.add_argument('--test', type=Path, required=True)
    parser.add_argument('--sample', type=Path, required=True)
    parser.add_argument('--selected', choices=('baseline', 'blend_equal850'), required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--freeze-only', action='store_true',
                        help='Fit and serialize the frozen procedure without reading test/sample CSVs.')
    args = parser.parse_args()
    args.train = args.train.resolve()
    args.test = args.test.resolve()
    args.sample = args.sample.resolve()
    out = args.out.resolve()
    for path in (args.train, args.test, args.sample):
        if not path.is_file():
            parser.error(f'Missing input file: {path}')
    if len({args.train, args.test, args.sample}) != 3:
        parser.error('Train, test, and sample must be distinct files.')
    out.mkdir(parents=True, exist_ok=True)
    if (out / ACCESS_MARKER).exists():
        raise SystemExit('Kaggle test access has already begun in this output directory; rerun/refreeze refused.')
    freeze_path = out / FREEZE_FILE
    outputs = [CONTROL_FILE] + ([BLEND_FILE] if args.selected == 'blend_equal850' else [])
    if any((out / name).exists() for name in outputs):
        raise SystemExit('An intended submission output already exists; refusing to overwrite it.')
    if freeze_path.exists():
        manifest = json.loads(freeze_path.read_text())
        verify_frozen(manifest, args, out)
        print('Verified existing frozen training artifacts.', flush=True)
    else:
        if any(out.glob('frozen_*_850*')):
            raise SystemExit('Unmanifested frozen artifacts exist; use a clean output directory.')
        manifest = fit_and_freeze(args, out)
        print(f'Frozen manifest and serialized refits saved to {out}.', flush=True)
    if args.freeze_only:
        print('Freeze-only complete; Kaggle test and sample CSV content was not read.', flush=True)
        return
    final_inference(manifest, args, out)


if __name__ == '__main__':
    main()
