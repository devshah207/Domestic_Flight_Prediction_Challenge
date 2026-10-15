"""Freeze the incomplete peer replication, then generate all seven files once."""
from pathlib import Path
from datetime import datetime,timezone
import argparse,hashlib,io,json
import numpy as np
import pandas as pd
import joblib
import common as c
m=c.m;ROOT=c.ROOT
FILES={name:f'submission_peer_{name}.csv' for name in m.FAMILIES+list(m.BLENDS)}
TEST_SHA='fa76192bf0d6c4186011cb594a300dd40c3418655c16a9844840977043907861'
SAMPLE_SHA='4aab67b24e1a4a56327f54084280fad29d33986c0c3ee1387c5b4f8d6588a75b'

def canonical(x):return json.dumps(c.plain(x),sort_keys=True,separators=(',',':'),allow_nan=False).encode()
def digest(x):return hashlib.sha256(x).hexdigest()
def exclusive(path,value):
    with path.open('x') as f:json.dump(c.plain(value),f,indent=2,allow_nan=False);f.write('\n')

def snapshot(train):
    reg=c.verify(train);done=c.read(ROOT/'analysis/completion.json')
    assert done['regressor_fits']==30 and done['models']==5 and done['blends']==2
    assert done['sources']==c.sources() and done['registration_sha256']==c.sha(ROOT/'REGISTRATION.json')
    assert done['configurations_sha256']==c.sha(ROOT/'MODEL_CONFIGURATIONS.json')
    frame,raw,y=c.base.data(train);groups=c.base.group_ids(raw)
    outer=c.base.splits(raw,y,c.OUTER_SEED)
    for i,(tr,va) in enumerate(outer):
        folder=ROOT/'results'/f'outer{i}';index=c.read(folder/'indices.json')
        assert index['fit']==tr.tolist() and index['validation']==va.tolist()
        assert set(groups[tr]).isdisjoint(groups[va])
    bundles={};configs=c.read(ROOT/'MODEL_CONFIGURATIONS.json')
    for family in m.FAMILIES:
        folder=ROOT/'results/full_training';config=configs['models'][family]
        assert config==c.read(folder/f'{family}_configuration.json')
        assert config['sources']==c.sources() and config['training_rows']==len(y)
        assert config['model_sha256']==c.sha(folder/f'{family}.joblib') and config['test_accessed'] is False
        b=joblib.load(folder/f'{family}.joblib');model=b['model']
        assert b['family']==family and b['scale']==m.SCALE
        assert c.plain(model.get_params())==config['constructor']==reg['constructors'][family]
        assert c.structure(model,family)==config['actual_structure']
        assert b['features'].description()==config['feature_schema'];bundles[family]=b
        if family.endswith('smooth_auto'):
            for scope in [f'outer{i}' for i in range(5)]+['full_training']:
                local=raw if scope=='full_training' else raw.iloc[outer[int(scope[-1])][0]].reset_index(drop=True)
                g=c.base.group_ids(local);seen=[]
                for part in c.read(ROOT/'results'/scope/f'{family}_encoding_folds.json'):
                    it=np.asarray(part['fit']);iv=np.asarray(part['validation'])
                    assert not set(it)&set(iv) and set(it)|set(iv)==set(range(len(local)))
                    assert set(g[it]).isdisjoint(g[iv]);seen.extend(iv.tolist())
                assert sorted(seen)==list(range(len(local)))
    value=dict(sources=c.sources(),versions=c.versions(),training_sha256=reg['training_sha256'],
        registration_sha256=c.sha(ROOT/'REGISTRATION.json'),completion_sha256=c.sha(ROOT/'analysis/completion.json'),
        configurations_sha256=c.sha(ROOT/'MODEL_CONFIGURATIONS.json'),models=configs['models'],
        outer_and_encoding_indices_verified=True,
        policy=dict(files=FILES,blends=m.BLENDS,rows=3898,schema=['id','price'],target_scale=m.SCALE,
            price_policy='inverse10000; clip component prices at zero; fixed arithmetic equal thirds',
            unknown_smooth_implementation=True,exact_peer_pipeline_reproduced=False))
    return value,bundles

def freeze(train,out):
    assert not (out/'INFERENCE_STARTED.json').exists(),'Inference already started; repeat test read blocked'
    out.mkdir(exist_ok=True,parents=True);value,bundles=snapshot(train);path=out/'frozen_manifest.json'
    if path.exists():
        saved=c.read(path);assert canonical(saved['snapshot'])==canonical(value),'Frozen inputs changed'
    else:
        saved=dict(frozen_at_utc=datetime.now(timezone.utc).isoformat(),snapshot_sha256=digest(canonical(value)),snapshot=value)
        exclusive(path,saved)
    assert saved['snapshot_sha256']==digest(canonical(value));return saved,bundles

def outputs(raw,schema,bundles):
    assert list(schema)==['id','price'] and 'price' not in raw and len(raw)==len(schema)
    assert raw.id.notna().all() and schema.id.notna().all() and raw.id.is_unique and schema.id.is_unique
    assert set(raw.id)==set(schema.id)
    positions=pd.Series(np.arange(len(raw)),index=raw.id).loc[schema.id].to_numpy()
    x=raw.drop(columns='id');pred={}
    for family in m.FAMILIES:
        b=bundles[family];pred[family]=m.predict(b['model'],b['features'].transform(x))
    for name,weights in m.BLENDS.items():pred[name]=sum(w*pred[f] for f,w in weights.items())
    result={}
    for name,p in pred.items():
        assert len(p)==len(raw) and np.isfinite(p).all() and (p>=0).all()
        result[FILES[name]]=pd.DataFrame(dict(id=schema.id.to_numpy(),price=p[positions]))
    return result

def infer(train,test,sample,out):
    frozen,bundles=freeze(train,out)
    exclusive(out/'INFERENCE_STARTED.json',dict(started_at_utc=datetime.now(timezone.utc).isoformat(),manifest_sha256=c.sha(out/'frozen_manifest.json')))
    data=test.read_bytes();sampledata=sample.read_bytes()
    assert digest(data)==TEST_SHA and digest(sampledata)==SAMPLE_SHA
    raw=pd.read_csv(io.BytesIO(data),dtype={'id':str});schema=pd.read_csv(io.BytesIO(sampledata),dtype={'id':str})
    assert len(raw)==len(schema)==3898
    files=outputs(raw,schema,bundles);assert len(files)==7;hashes={}
    for filename,frame in files.items():
        path=out/filename
        with path.open('x') as f:frame.to_csv(f,index=False)
        saved=pd.read_csv(path,dtype={'id':str})
        assert saved.id.equals(schema.id) and list(saved)==['id','price'] and len(saved)==3898
        assert np.isfinite(saved.price).all() and saved.price.ge(0).all();hashes[filename]=c.sha(path)
    for name,checksum in frozen['snapshot']['sources'].items():assert c.sha(c.REPO/name)==checksum
    for family,config in frozen['snapshot']['models'].items():
        assert c.sha(ROOT/'results/full_training'/f'{family}.joblib')==config['model_sha256']
    assert c.sha(ROOT/'MODEL_CONFIGURATIONS.json')==frozen['snapshot']['configurations_sha256']
    exclusive(out/'inference_audit.json',dict(completed_at_utc=datetime.now(timezone.utc).isoformat(),
        files=hashes,output_count=7,rows=3898,test_content_reads=1,
        test_sha256=digest(data),sample_sha256=digest(sampledata),model_fits=0,models_changed_after_test=False,
        schema_ids_finite_nonnegative_checks_passed=True))
    print('Seven audited replication submissions saved; no test-dependent revisions.',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--train',type=Path,required=True)
    p.add_argument('--test',type=Path);p.add_argument('--sample',type=Path)
    p.add_argument('--freeze-only',action='store_true');p.add_argument('--out',type=Path,default=ROOT/'submissions');a=p.parse_args()
    if a.freeze_only:freeze(a.train,a.out);print('All five models and both fixed blends frozen.')
    else:
        assert a.test and a.sample
        infer(a.train,a.test,a.sample,a.out)
