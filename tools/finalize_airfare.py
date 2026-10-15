"""Verify completed artifacts and assemble reporting; never fit or tune a model."""
from pathlib import Path
import hashlib
import json
import sys
import zipfile
import datetime
import numpy as np
import pandas as pd
import joblib

ROOT = Path('/workspace/airfare_fresh')
sys.path.insert(0, str(ROOT))
import study

def read(p):
    return json.loads(p.read_text())

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def save(p, obj):
    p.write_text(json.dumps(study.plain(obj), indent=2, allow_nan=False) + '\n')

manifest = read(ROOT/'submissions/frozen_inference_manifest.json')
frozen = manifest['frozen']
for filename, digest in frozen['source_sha256'].items():
    assert sha(ROOT/filename) == digest, filename
assert sha(ROOT/'fold_plan.json') == frozen['fold_plan_sha256']
audit = read(ROOT/'submissions/inference_audit.json')
assert sha(ROOT/'submissions/frozen_inference_manifest.json') == audit['manifest_sha256']
sample = pd.read_csv('/workspace/attachments/03d4ffea-f3d2-4db0-850d-b7f1a3efa012/sample_submission.csv', dtype={'id': str})
for filename, digest in audit['submission_sha256'].items():
    p = ROOT/'submissions'/filename
    assert sha(p) == digest
    df = pd.read_csv(p, dtype={'id': str})
    assert list(df) == ['id', 'price'] and len(df) == 3898
    assert df.id.is_unique and df.id.notna().all() and df.id.equals(sample.id)
    assert np.isfinite(df.price).all() and (df.price >= 0).all()

families = ['xgb','lgb','cat','cat_onehot64']
configs = {}
trials = []
scopes = []
reference = None
for family in families:
    directory = ROOT/'results'/family
    oof = pd.read_csv(directory/'oof.csv')
    assert len(oof) == 15589 and oof.row_position.is_unique
    assert np.array_equal(oof.row_position, np.arange(15589))
    assert oof.groupby('group').fold.nunique().max() == 1
    assert sorted(oof.fold.unique()) == [0,1,2,3,4]
    identity = oof[['row_position','id','price','group','fold']]
    if reference is None:
        reference = identity
    else:
        pd.testing.assert_frame_equal(reference, identity)
    for fold in range(5):
        p = directory/f'outer{fold}'/'outer_metrics.json'
        assert p.exists()
        scopes.append({'family':family,'scope':f'outer{fold}','metrics':read(p),'sha256':sha(p)})
    p = directory/'full_training'/'final_configuration.json'
    config = read(p)
    bundle_path = directory/'full_training'/'final_model.joblib'
    assert sha(bundle_path) == config['model_sha256']
    bundle = joblib.load(bundle_path)
    model = bundle['model']
    actual = (model.get_booster().num_boosted_rounds() if family=='xgb'
              else model.booster_.current_iteration() if family=='lgb'
              else model.tree_count_)
    assert actual == config['selection']['rounds']
    assert len(config['features']) == 17
    configs[family] = {
        'configuration':config,
        'constructor_parameters':model.get_params(),
        'actual_tree_count':int(actual),
        'configuration_sha256':sha(p),
        'exploratory_after_original_diagnostics':family=='cat_onehot64',
    }
    if family == 'xgb':
        configs[family]['resolved_booster_configuration'] = json.loads(model.get_booster().save_config())
    if family == 'lgb':
        configs[family]['booster_parameters'] = model.booster_.params
    if family.startswith('cat'):
        configs[family]['resolved_parameters'] = model.get_all_params()
    if family != 'cat_onehot64':
        paths = sorted(directory.glob('*/*_all_trials.json'))
        assert len(paths) == 6
        for p in paths:
            records = read(p)
            assert len(records) == 12
            assert all(t['state'] == 'TrialState.COMPLETE' for t in records)
            trials.append({'family':family,'scope':p.parent.name,'path':str(p.relative_to(ROOT)),
                           'sha256':sha(p),'records':records})

assert sum(len(t['records']) for t in trials) == 216
ensembles = {
    'equal_blend':{'xgb':1/3,'lgb':1/3,'cat':1/3},
    'equal_blend_onehot64':{'xgb':1/3,'lgb':1/3,'cat_onehot64':1/3},
}
save(ROOT/'model_configurations.json', {
    'models':configs,'ensembles':ensembles,
    'target_training_scale':10000,'prediction_inverse':'multiply by 10000; clip each component below at zero; then blend',
    'feature_implementation':'features.py; fitted mappings serialized inside each final_model.joblib',
    'package_versions':frozen['package_versions'],
    'interpretation':'Outer CV evaluates complete selection/refit procedures, not an independent test of the full-data dictionaries.',
})
metrics = pd.read_csv(ROOT/'analysis/model_metrics.csv').replace({np.nan:None}).to_dict(orient='records')
save(ROOT/'experiment_ledger.json', {
    'completed_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
    'optuna_trials_completed':216,'original_scopes':18,'representation_followup_scopes':6,
    'model_fits':{'inner':444,'outer_refits':20,'full_refits':4,'total':468,'api_smoke_fits_excluded':3},
    'original_fit_accounting':'216 trials * 2 inner fits + 15 outer refits + 3 full refits = 450',
    'followup_fit_accounting':'6 scopes * (2 inner stopping fits + 1 refit) = 18; no additional Optuna trials',
    'training_sha256':frozen['training_sha256'],'source_sha256':frozen['source_sha256'],
    'fold_plan':read(ROOT/'fold_plan.json'),'outer_metrics':scopes,'trial_archives':trials,
    'metrics':metrics,'ensembles':ensembles,'inference_audit':audit,
    'exploratory_candidates':['cat_onehot64','equal_blend_onehot64'],
    'limitations':['Historical development used these training data before this study.',
        'Outer folds select the preferred candidate; winner selection has optimism.',
        'Representation follow-up reuses outer folds after original diagnostics.',
        'OOF bootstrap intervals condition on fitted predictions and omit retraining uncertainty.',
        'Original-price Kaggle R2 is unknown; no public leaderboard scores were used for selection.'],
})

print('Verified: 216 completed trials, 20 aligned outer folds, four frozen models, six valid CSVs.')
print('Actual tree counts:', {k:v['actual_tree_count'] for k,v in configs.items()})
