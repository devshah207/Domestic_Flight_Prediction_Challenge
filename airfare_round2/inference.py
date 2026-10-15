"""Freeze the registered phase, then write two submissions without model changes."""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
import hashlib
import importlib.metadata as metadata
import io
import json
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
import direct_cat64 as run

OUTPUTS={'cat64_direct':'submission_catboost64_direct_tuned.csv',
         'equal_blend_direct64':'submission_equal_blend_direct64.csv'}
BUSINESS_OUTPUTS={'business_augmented_xgb':'submission_business_augmented_xgb.csv',
                  'business_equal_blend64':'submission_business_equal_blend64.csv'}


def digest(content):return hashlib.sha256(content).hexdigest()
def canonical(value):return json.dumps(run.base.plain(value),sort_keys=True,separators=(',',':'),allow_nan=False).encode()
def read(path):return json.loads(path.read_text())
def write_exclusive(path,data):
    with path.open('x') as f:json.dump(run.base.plain(data),f,indent=2,allow_nan=False);f.write('\n')


def checked_business(train,frozen_old):
    folder=ROOT.parent/'airfare_business';sys.path.insert(0,str(folder))
    import specialist as business
    registration=read(folder/'REGISTRATION.json')
    assert registration['source_hashes']==business.sources()
    assert registration['training_sha256']==run.base.sha(train)
    assert registration['versions']=={p:metadata.version(p) for p in run.VERSIONS}
    completion=read(folder/'analysis/completion.json');decision=read(folder/'sensitivity_decision.json')
    assert completion['source_hashes']==business.sources()
    assert completion['secondary_completed']==decision['run_secondary']
    assert completion['total_trials']==(192 if decision['run_secondary'] else 72)
    assert completion['research_model_fits']==(427 if decision['run_secondary'] else 167)
    assert completion['economy_oof_predictions_unchanged'] is True
    raw=read(folder/'primary/summary.json');assert raw['source_hashes']==business.sources() and len(raw['folds'])==10
    for row in raw['folds']:
        assert np.isfinite([row['r2'],row['train_r2'],row['rmse']]).all()
    c=read(folder/'full_training/final_configuration.json');path=folder/'full_training/final_model.joblib'
    assert c['source_hashes']==business.sources() and c['model_sha256']==run.base.sha(path)
    assert c['training_sha256']==run.base.sha(train) and c['training_rows']==15589 and c['specialist_training_rows']==6380
    assert c['test_accessed'] is False and c['mixture']==.5
    b=joblib.load(path);m=b['model'];f=b['features']
    assert c['constructor_parameters']==run.base.plain(m.get_params())
    assert c['actual_trees']==c['selection']['rounds']==m.get_booster().num_boosted_rounds()
    assert 100<=c['actual_trees']<=1200
    p=c['selection']['configuration'];assert 4<=p['depth']<=6 and 0<p['leaves']<=min(32,2**p['depth'])
    assert f.mode=='context' and f.family=='xgb' and f.get_feature_names_out().tolist()==c['feature_names']
    old_dir=run.FRESH/'results/cat_onehot64/full_training';oc=read(old_dir/'final_configuration.json')
    assert oc==frozen_old['models']['cat_onehot64']['configuration']
    assert oc['model_sha256']==run.base.sha(old_dir/'final_model.joblib')
    ob=joblib.load(old_dir/'final_model.joblib')
    assert oc['estimator_parameters']==ob['model'].get_all_params()
    extra_sources=business.sources()
    extra_sources['airfare_business/analyze.py']=run.base.sha(folder/'analyze.py')
    return {'business_specialist':b,'cat_onehot64':ob},{'business_specialist':c,'cat_onehot64':oc},dict(
        source_hashes=extra_sources,registration_sha256=run.base.sha(folder/'REGISTRATION.json'),
        completion_sha256=run.base.sha(folder/'analysis/completion.json'),decision_sha256=run.base.sha(folder/'sensitivity_decision.json'))


def checked(train,include_business=False):
    frozen_old=run.checked_plan(train)
    registration=read(ROOT/'REGISTRATION.json')
    assert registration['source_hashes']==run.sources()
    assert registration['training_sha256']==run.base.sha(train)
    versions={p:metadata.version(p) for p in run.VERSIONS}
    assert versions==registration['versions']
    completion=read(ROOT/'analysis/completion.json')
    assert completion['source_hashes']==run.sources()
    decision=read(ROOT/'sensitivity_decision.json')
    assert completion['secondary_completed']==decision['run_secondary']
    assert completion['trials_completed']==(192 if decision['run_secondary'] else 72)
    assert completion['research_model_fits']==(415 if decision['run_secondary'] else 150)
    oof=pd.read_csv(ROOT/'primary'/run.VARIANT/'oof.csv')
    control=pd.read_csv(run.FRESH/'results/cat_onehot64/oof.csv')
    pd.testing.assert_frame_equal(oof.drop(columns='prediction'),control.drop(columns='prediction'))
    assert len(oof)==15589 and oof.row_position.is_unique and np.isfinite(oof.prediction).all()
    assert oof.groupby('group').fold.nunique().max()==1
    summary=read(ROOT/'primary'/run.VARIANT/'summary.json')
    assert len(summary['folds'])==5
    for row in summary['folds']:
        assert np.isfinite([row['r2'],row['train_r2'],row['rmse']]).all()
        assert row['source_hashes']==run.sources()
        p=row['selected']['configuration']
        assert 4<=p['depth']<=7 and 0<p['leaves']<=min(48,2**p['depth'])
        assert 100<=row['actual_trees']==row['selected']['rounds']<=1200
    bundles={};configs={}
    for name in ['xgb','lgb',run.VARIANT]:
        d=(ROOT/'primary'/run.VARIANT if name==run.VARIANT else run.FRESH/'results'/name)/'full_training'
        config=read(d/'final_configuration.json');path=d/'final_model.joblib'
        assert config['model_sha256']==run.base.sha(path)
        assert config['training_sha256']==registration['training_sha256']
        assert config['training_rows']==15589 and config['test_accessed'] is False
        bundle=joblib.load(path);m=bundle['model'];f=bundle['features']
        assert bundle['scale']==10000.
        assert f.get_feature_names_out().tolist()==config['features']
        if name==run.VARIANT:
            assert config['source_hashes']==run.sources()
            assert config['resolved_parameters']==run.base.plain(m.get_all_params())
            actual=m.tree_count_;params=m.get_all_params()
            assert params['one_hot_max_size']==64 and params['grow_policy']=='Lossguide'
            assert 4<=params['depth']<=7 and 0<params['max_leaves']<=min(48,2**params['depth'])
            assert f.family=='cat'
        else:
            assert config==frozen_old['models'][name]['configuration']
            assert config['estimator_parameters']==run.base.plain(m.get_params())
            actual=m.get_booster().num_boosted_rounds() if name=='xgb' else m.booster_.current_iteration()
        assert actual==config['selection']['rounds'] and 100<=actual<=1200
        bundles[name]=bundle;configs[name]=config
    sources=run.sources()
    for name in ['inference.py','analyze.py']:
        sources['airfare_round2/'+name]=run.base.sha(ROOT/name)
    business_metadata=None
    outputs=dict(OUTPUTS)
    if include_business:
        extra_bundles,extra_configs,business_metadata=checked_business(train,frozen_old)
        bundles.update(extra_bundles);configs.update(extra_configs);outputs.update(BUSINESS_OUTPUTS)
    snapshot=dict(source_hashes=sources,registration_sha256=run.base.sha(ROOT/'REGISTRATION.json'),
        analysis_completion_sha256=run.base.sha(ROOT/'analysis/completion.json'),
        sensitivity_decision_sha256=run.base.sha(ROOT/'sensitivity_decision.json'),
        training_sha256=registration['training_sha256'],versions=versions,models=configs,business_metadata=business_metadata,
        policy=dict(target_scale=10000.,component_postprocessing='clip original-price predictions below at zero',
            blend_weights={'xgb':1/3,'lgb':1/3,run.VARIANT:1/3},outputs=outputs,rows=3898,
            business_policy='Business: .5 original global XGB + .5 specialist; Economy: global XGB unchanged; optional thirds blend with original LGB and original CatBoost64' if include_business else None,
            schema=['id','price'],id_order='exact sample-submission order',training_during_inference=False))
    return snapshot,bundles


def freeze(train,out,include_business=False):
    out.mkdir(parents=True,exist_ok=True)
    assert not (out/'INFERENCE_STARTED.json').exists(),'Inference already started in this output directory'
    snapshot,bundles=checked(train,include_business);path=out/'frozen_manifest.json'
    if path.exists():
        saved=read(path);assert canonical(saved['snapshot'])==canonical(snapshot),'Freeze changed'
    else:
        saved=dict(frozen_at_utc=datetime.now(timezone.utc).isoformat(),snapshot_sha256=digest(canonical(snapshot)),snapshot=snapshot)
        write_exclusive(path,saved)
    assert saved['snapshot_sha256']==digest(canonical(snapshot))
    return saved,bundles


def infer(train,test,sample,out,include_business=False):
    frozen,bundles=freeze(train,out,include_business)
    write_exclusive(out/'INFERENCE_STARTED.json',dict(started_at_utc=datetime.now(timezone.utc).isoformat(),
        manifest_sha256=run.base.sha(out/'frozen_manifest.json'),test_access='permitted only for final prediction; no new training or selection'))
    # Exactly one content read of the test file, after the complete freeze/marker.
    test_bytes=test.read_bytes();sample_bytes=sample.read_bytes()
    original=read(run.FRESH/'submissions/inference_audit.json')
    assert digest(test_bytes)==original['test_sha256'] and digest(sample_bytes)==original['sample_sha256']
    raw=pd.read_csv(io.BytesIO(test_bytes),dtype={'id':str})
    schema=pd.read_csv(io.BytesIO(sample_bytes),dtype={'id':str})
    assert list(schema)==['id','price'] and len(raw)==len(schema)==3898
    assert raw.id.is_unique and schema.id.is_unique and raw.id.notna().all() and schema.id.notna().all()
    assert set(raw.id)==set(schema.id) and 'price' not in raw
    positions=pd.Series(np.arange(len(raw)),index=raw.id).loc[schema.id].to_numpy()
    predictions={}
    for name,bundle in bundles.items():
        if name=='business_specialist':continue
        x=bundle['features'].transform(raw.drop(columns='id'))
        predictions[name]=run.base.predict(bundle['model'],x)[positions]
    predictions['equal_blend_direct64']=(predictions['xgb']+predictions['lgb']+predictions[run.VARIANT])/3
    outputs=dict(OUTPUTS)
    if include_business:
        assert raw.flight_class.isin(['Business','Economy']).all()
        mask=raw.flight_class.eq('Business').to_numpy()
        specialist=np.zeros(len(raw));b=bundles['business_specialist']
        specialist[mask]=run.base.predict(b['model'],b['features'].transform(raw.loc[mask].drop(columns='id')))
        aligned_mask=mask[positions];augmented=predictions['xgb'].copy()
        augmented[aligned_mask]=.5*augmented[aligned_mask]+.5*specialist[positions][aligned_mask]
        assert np.array_equal(augmented[~aligned_mask],predictions['xgb'][~aligned_mask])
        predictions['business_augmented_xgb']=augmented
        predictions['business_equal_blend64']=(augmented+predictions['lgb']+predictions['cat_onehot64'])/3
        outputs.update(BUSINESS_OUTPUTS)
    hashes={}
    for name,filename in outputs.items():
        p=predictions[name];assert p.shape==(3898,) and np.isfinite(p).all() and (p>=0).all()
        result=pd.DataFrame({'id':schema.id,'price':p})
        with (out/filename).open('x') as f:result.to_csv(f,index=False)
        check=pd.read_csv(out/filename,dtype={'id':str})
        assert list(check)==['id','price'] and check.id.equals(schema.id) and len(check)==3898
        assert np.isfinite(check.price).all() and (check.price>=0).all()
        hashes[filename]=run.base.sha(out/filename)
    write_exclusive(out/'inference_audit.json',dict(completed_at_utc=datetime.now(timezone.utc).isoformat(),
        manifest_sha256=run.base.sha(out/'frozen_manifest.json'),test_content_reads=1,
        test_sha256=digest(test_bytes),sample_sha256=digest(sample_bytes),rows_per_submission=3898,
        checks=['frozen sources, versions and trained-model hashes','id,price schema','sample ID order',
            'unique nonmissing IDs','3898 rows','finite nonnegative prices'],submission_sha256=hashes,
        model_training_performed=False,model_changes_after_test_access=False))
    print('Wrote',len(outputs),'validated submissions and audit:',out,flush=True)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--train',type=Path,default=run.TRAIN)
    ap.add_argument('--out',type=Path,default=ROOT/'submissions');ap.add_argument('--freeze-only',action='store_true')
    ap.add_argument('--test',type=Path);ap.add_argument('--sample',type=Path)
    ap.add_argument('--include-business',action='store_true',help='Freeze both completed experiments and write all four candidate CSVs')
    args=ap.parse_args()
    if args.freeze_only:
        freeze(args.train,args.out,args.include_business);print('Freeze complete; no test/sample data opened.')
    else:
        assert args.test and args.sample,'Provide test and sample only after freezing'
        infer(args.train,args.test,args.sample,args.out,args.include_business)


if __name__=='__main__':main()
