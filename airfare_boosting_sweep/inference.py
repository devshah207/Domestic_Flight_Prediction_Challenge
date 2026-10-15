"""Freeze all eight procedures, then one test read produces ten submissions."""
from datetime import datetime,timezone
from pathlib import Path
import argparse
import hashlib
import importlib.metadata as metadata
import importlib.util
import io
import json
import sys
import joblib
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
spec=importlib.util.spec_from_file_location('boosting_sweep_study',ROOT/'study.py')
e=importlib.util.module_from_spec(spec);spec.loader.exec_module(e)
m=e.m
TEST_SHA='fa76192bf0d6c4186011cb594a300dd40c3418655c16a9844840977043907861'
SAMPLE_SHA='4aab67b24e1a4a56327f54084280fad29d33986c0c3ee1387c5b4f8d6588a75b'
NAMES=m.FAMILIES+list(m.BLENDS)
FILES={name:f'submission_highround_optuna_{name}.csv' for name in NAMES}


def digest(data):return hashlib.sha256(data).hexdigest()
def canonical(value):return json.dumps(e.base.plain(value),sort_keys=True,separators=(',',':'),allow_nan=False).encode()
def exclusive_json(path,value):
    with Path(path).open('x') as f:json.dump(e.base.plain(value),f,indent=2,allow_nan=False);f.write('\n')


def snapshot(train):
    reg=e.verify(train)
    completion=e.read(ROOT/'analysis/completion.json')
    audit=e.read(ROOT/'VALIDATION_AUDIT.json')
    assert completion['source_hashes']==e.sources() and completion['research_fits']==384
    assert completion['optuna_trials']==336 and completion['final_models']==8 and completion['outer_folds']==5
    assert completion['validation_audit_sha256']==e.sha(ROOT/'VALIDATION_AUDIT.json')
    assert audit['source_hashes']==e.sources() and audit['all_partition_and_group_identities_passed']
    assert len(audit['scopes'])==48 and audit['optuna_trials']==336 and audit['research_fits']==384
    assert reg['versions']=={name:metadata.version(name) for name in e.VERSIONS}
    configs={};bundles={}
    for family in m.FAMILIES:
        folder=ROOT/'results'/family/'full_training'
        config=e.read(folder/'configuration.json');modelpath=folder/'model.joblib'
        assert config['source_hashes']==e.sources() and config['training_sha256']==reg['training_sha256']
        assert config['model_sha256']==e.sha(modelpath) and config['test_accessed'] is False
        assert config['selection']['configuration']['rounds'] in m.ROUNDS
        bundle=joblib.load(modelpath)
        assert bundle['family']==family and bundle['scale']==10000.
        assert bundle['features'].description()==config['feature_schema']
        assert config['constructor_parameters']==e.base.plain(m.parameters(bundle['model'],family))
        assert int(m.actual_rounds(bundle['model'],family))==config['actual_rounds']
        configs[family]=config;bundles[family]=bundle
    source=e.sources()
    for name in ['inference.py','analyze.py','check_inference.py']:
        source[str((ROOT/name).relative_to(ROOT.parent))]=e.sha(ROOT/name)
    value=dict(source_hashes=source,registration_sha256=e.sha(ROOT/'REGISTRATION.json'),
        training_sha256=reg['training_sha256'],versions=reg['versions'],models=configs,
        completion_sha256=e.sha(ROOT/'analysis/completion.json'),
        validation_audit_sha256=e.sha(ROOT/'VALIDATION_AUDIT.json'),
        model_configurations_sha256=e.sha(ROOT/'MODEL_CONFIGURATIONS.json'),
        policy=dict(filenames=FILES,blend_weights=m.BLENDS,rows=3898,schema=['id','price'],
            target_scale=10000.,postprocessing='clip individual price predictions below zero before fixed arithmetic blending',
            id_order='exact sample order',experimental_role='multiple high-capacity candidates requested; no public-score guarantee'))
    return value,bundles


def freeze(train,out):
    out.mkdir(parents=True,exist_ok=True)
    assert not (out/'INFERENCE_STARTED.json').exists(),'Inference already started; cannot read test again'
    value,bundles=snapshot(train);path=out/'frozen_manifest.json'
    if path.exists():
        saved=e.read(path);assert canonical(saved['snapshot'])==canonical(value),'Frozen artifact changed'
    else:
        saved=dict(frozen_at_utc=datetime.now(timezone.utc).isoformat(),snapshot_sha256=digest(canonical(value)),snapshot=value)
        exclusive_json(path,saved)
    assert saved['snapshot_sha256']==digest(canonical(value))
    return saved,bundles


def make_outputs(raw,schema,bundles):
    """No fitting, summaries, calibration, clipping selection or weight selection."""
    assert list(schema)==['id','price'] and 'id' in raw and 'price' not in raw
    assert len(raw)==len(schema) and raw.id.is_unique and schema.id.is_unique
    assert raw.id.notna().all() and schema.id.notna().all() and set(raw.id)==set(schema.id)
    positions=pd.Series(np.arange(len(raw)),index=raw.id).loc[schema.id].to_numpy()
    inputs=raw.drop(columns='id');predictions={}
    for family in m.FAMILIES:
        b=bundles[family]
        predictions[family]=m.predict(b['model'],b['features'].transform(inputs))
    for name,weights in m.BLENDS.items():
        predictions[name]=sum(weight*predictions[family] for family,weight in weights.items())
    outputs={}
    for name,p in predictions.items():
        assert len(p)==len(raw) and np.isfinite(p).all() and (p>=0).all()
        outputs[FILES[name]]=pd.DataFrame({'id':schema.id.to_numpy(),'price':p[positions]})
    return outputs


def infer(train,test,sample,out):
    frozen,bundles=freeze(train,out)
    exclusive_json(out/'INFERENCE_STARTED.json',dict(started_at_utc=datetime.now(timezone.utc).isoformat(),
        manifest_sha256=e.sha(out/'frozen_manifest.json')))
    test_bytes=test.read_bytes();sample_bytes=sample.read_bytes()
    assert digest(test_bytes)==TEST_SHA and digest(sample_bytes)==SAMPLE_SHA
    raw=pd.read_csv(io.BytesIO(test_bytes),dtype={'id':str})
    schema=pd.read_csv(io.BytesIO(sample_bytes),dtype={'id':str})
    assert len(raw)==len(schema)==3898
    outputs=make_outputs(raw,schema,bundles);hashes={}
    assert set(outputs)==set(FILES.values()) and len(outputs)==10
    for filename,frame in outputs.items():
        path=out/filename
        with path.open('x') as f:frame.to_csv(f,index=False)
        check=pd.read_csv(path,dtype={'id':str})
        assert list(check)==['id','price'] and len(check)==3898 and check.id.equals(schema.id)
        assert check.id.is_unique and np.isfinite(check.price).all() and check.price.ge(0).all()
        hashes[filename]=e.sha(path)
    # Verify unchanged model/config/source identities after prediction without re-reading test.
    for family,config in frozen['snapshot']['models'].items():
        folder=ROOT/'results'/family/'full_training'
        assert e.sha(folder/'model.joblib')==config['model_sha256']
        assert e.read(folder/'configuration.json')==config
    for name,checksum in frozen['snapshot']['source_hashes'].items():assert e.sha(ROOT.parent/name)==checksum
    exclusive_json(out/'inference_audit.json',dict(completed_at_utc=datetime.now(timezone.utc).isoformat(),
        manifest_sha256=e.sha(out/'frozen_manifest.json'),test_content_reads=1,
        test_sha256=digest(test_bytes),sample_sha256=digest(sample_bytes),rows=3898,files=hashes,
        output_count=10,checks=['frozen sources,models,versions,configs,weights','id,price schema',
            'exact sample ID order','unique IDs','3898 rows','finite nonnegative prices'],
        model_training_performed=False,model_changes_after_test_access=False))
    print('Wrote ten validated high-round submission CSVs.',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--train',type=Path,default=e.base.TRAIN)
    parser.add_argument('--test',type=Path);parser.add_argument('--sample',type=Path)
    parser.add_argument('--out',type=Path,default=ROOT/'submissions');parser.add_argument('--freeze-only',action='store_true')
    args=parser.parse_args()
    if args.freeze_only:freeze(args.train,args.out);print('All eight final models frozen before test access.')
    else:
        assert args.test and args.sample
        infer(args.train,args.test,args.sample,args.out)
