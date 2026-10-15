"""Freeze one exploratory candidate; read test once and emit exactly one CSV."""
from datetime import datetime,timezone
from pathlib import Path
import argparse,hashlib,importlib.metadata as metadata,io,json,sys
import joblib
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT))
import experiment as e

FILENAME='submission_aggressive_business_xgb800.csv'


def read(path):return json.loads(path.read_text())
def canonical(v):return json.dumps(e.base.plain(v),sort_keys=True,separators=(',',':'),allow_nan=False).encode()
def sha_bytes(data):return hashlib.sha256(data).hexdigest()
def write(path,value):
    with path.open('x') as f:json.dump(e.base.plain(value),f,indent=2,allow_nan=False);f.write('\n')


def snapshot(train):
    registration=e.verify(train);old_frozen=e.old.common.checked_plan(train)
    completion=read(ROOT/'analysis/completion.json')
    assert completion['source_hashes']==e.sources() and completion['research_model_fits']==21 and completion['outer_scopes']==10
    assert completion['economy_oof_unchanged'] is True
    for label in e.SEEDS:
        summary=read(ROOT/label/'summary.json');assert summary['source_hashes']==e.sources() and len(summary['folds'])==5
        for r in summary['folds']:
            assert r['group_overlap']==0 and r['global_input_identities_verified'] and r['economy_predictions_unchanged']
            assert r['specialist_trees']==800 and r['global_reconstruction_max_abs_error']<=1e-6
            assert np.isfinite([r['train_r2'],r['r2'],r['rmse'],r['mae']]).all()
    configs={};bundles={}
    for name,directory in [('global_xgb',e.old.common.FRESH/'results/xgb/full_training'),('specialist',ROOT/'full_training')]:
        c=read(directory/'final_configuration.json');path=directory/'final_model.joblib'
        assert e.base.sha(path)==c['model_sha256'] and c['training_sha256']==registration['training_sha256']
        assert c['training_rows']==15589 and c['test_accessed'] is False
        b=joblib.load(path);m=b['model'];f=b['features']
        assert b['scale']==10000. and f.get_feature_names_out().tolist()==c['features']
        if name=='global_xgb':
            assert c==old_frozen['models']['xgb']['configuration']
            assert c['estimator_parameters']==e.base.plain(m.get_params())
            assert m.get_booster().num_boosted_rounds()==454
        else:
            assert c['source_hashes']==e.sources() and c['profile']==e.PROFILE
            assert c['constructor_parameters']==e.base.plain(m.get_params())
            assert c['specialist_training_rows']==6380 and c['actual_trees']==m.get_booster().num_boosted_rounds()==800
            assert f.family=='xgb' and f.mode=='context'
            p=c['constructor_parameters'];assert p['max_depth']==7 and p['max_leaves']==48 and p['min_child_weight']==8
        configs[name]=c;bundles[name]=b
    source=e.sources();source['airfare_aggressive800/inference.py']=e.base.sha(Path(__file__))
    value=dict(source_hashes=source,training_sha256=registration['training_sha256'],registration_sha256=e.base.sha(ROOT/'REGISTRATION.json'),
        completion_sha256=e.base.sha(ROOT/'analysis/completion.json'),versions=registration['versions'],models=configs,
        policy=dict(filename=FILENAME,rows=3898,schema=['id','price'],target_scale=10000.,component_clipping='at zero before mixing',
            weights={'Business':{'global_xgb':.5,'specialist':.5},'Economy':{'global_xgb':1.}},id_order='exact sample order',
            experimental_role='one higher-risk candidate; validation improvement not required under latest user authorization'))
    return value,bundles


def freeze(train,out):
    out.mkdir(parents=True,exist_ok=True)
    assert not (out/'INFERENCE_STARTED.json').exists(),'This directory has already started inference'
    current,bundles=snapshot(train);path=out/'frozen_manifest.json'
    if path.exists():
        saved=read(path);assert canonical(saved['snapshot'])==canonical(current),'Frozen artifacts changed'
    else:
        saved=dict(frozen_at_utc=datetime.now(timezone.utc).isoformat(),snapshot_sha256=sha_bytes(canonical(current)),snapshot=current)
        write(path,saved)
    assert saved['snapshot_sha256']==sha_bytes(canonical(current))
    return saved,bundles


def infer(train,test,sample,out):
    saved,bundles=freeze(train,out)
    write(out/'INFERENCE_STARTED.json',dict(started_at_utc=datetime.now(timezone.utc).isoformat(),manifest_sha256=e.base.sha(out/'frozen_manifest.json')))
    test_data=test.read_bytes();sample_data=sample.read_bytes()
    original=read(e.old.common.FRESH/'submissions/inference_audit.json')
    assert sha_bytes(test_data)==original['test_sha256'] and sha_bytes(sample_data)==original['sample_sha256']
    raw=pd.read_csv(io.BytesIO(test_data),dtype={'id':str});schema=pd.read_csv(io.BytesIO(sample_data),dtype={'id':str})
    assert len(raw)==len(schema)==3898 and list(schema)==['id','price']
    assert raw.id.is_unique and schema.id.is_unique and raw.id.notna().all() and schema.id.notna().all()
    assert set(raw.id)==set(schema.id) and 'price' not in raw and raw.flight_class.isin(['Business','Economy']).all()
    gb=bundles['global_xgb'];global_p=e.base.predict(gb['model'],gb['features'].transform(raw.drop(columns='id')))
    mask=raw.flight_class.eq('Business').to_numpy();sb=bundles['specialist']
    p=global_p.copy();special=e.base.predict(sb['model'],sb['features'].transform(raw.loc[mask].drop(columns='id')))
    p[mask]=.5*global_p[mask]+.5*special
    assert np.array_equal(p[~mask],global_p[~mask])
    positions=pd.Series(np.arange(len(raw)),index=raw.id).loc[schema.id].to_numpy()
    p=p[positions];assert np.isfinite(p).all() and (p>=0).all()
    result=pd.DataFrame({'id':schema.id,'price':p});path=out/FILENAME
    with path.open('x') as f:result.to_csv(f,index=False)
    check=pd.read_csv(path,dtype={'id':str});assert check.id.equals(schema.id) and list(check)==['id','price'] and len(check)==3898
    assert np.isfinite(check.price).all() and (check.price>=0).all()
    write(out/'inference_audit.json',dict(completed_at_utc=datetime.now(timezone.utc).isoformat(),manifest_sha256=e.base.sha(out/'frozen_manifest.json'),
        test_content_reads=1,test_sha256=sha_bytes(test_data),sample_sha256=sha_bytes(sample_data),rows=3898,
        filename=FILENAME,submission_sha256=e.base.sha(path),economy_predictions_unchanged=True,
        checks=['frozen trained-model/source/version identities','id,price schema','exact sample ID order','unique IDs','3898 rows','finite nonnegative prices'],
        model_training_performed=False,model_changes_after_test_access=False))
    print('Wrote one validated exploratory CSV:',path,flush=True)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--train',type=Path,default=e.base.TRAIN)
    ap.add_argument('--out',type=Path,default=ROOT/'submissions');ap.add_argument('--freeze-only',action='store_true')
    ap.add_argument('--test',type=Path);ap.add_argument('--sample',type=Path);a=ap.parse_args()
    if a.freeze_only:freeze(a.train,a.out);print('Frozen before test/sample access.')
    else:
        assert a.test and a.sample
        infer(a.train,a.test,a.sample,a.out)


if __name__=='__main__':main()
