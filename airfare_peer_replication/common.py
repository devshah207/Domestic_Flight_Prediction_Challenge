"""Shared identity, scoring and structural checks; never reads test data."""
from pathlib import Path
import hashlib,json,importlib.util,importlib.metadata as md
import numpy as np
import pandas as pd
import models as m
ROOT=Path(__file__).resolve().parent;REPO=ROOT.parent
spec=importlib.util.spec_from_file_location('peer_training_base',REPO/'airfare_fresh/study.py')
base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)
VERSIONS=['numpy','pandas','scipy','scikit-learn','xgboost-cpu','catboost','joblib']
OUTER_SEED=20261020

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def plain(v):
    if isinstance(v,dict):return {str(k):plain(x) for k,x in v.items()}
    if isinstance(v,(list,tuple)):return [plain(x) for x in v]
    if isinstance(v,np.ndarray):return plain(v.tolist())
    if isinstance(v,np.generic):return plain(v.item())
    if isinstance(v,float) and not np.isfinite(v):return None
    if isinstance(v,Path):return str(v)
    return v

def save(p,value):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(plain(value),indent=2,allow_nan=False)+'\n')
def read(p):return json.loads(Path(p).read_text())
def sources():
    names=['models.py','common.py','run.py','inference.py','check_inference.py','PROTOCOL.md']
    paths=[ROOT/n for n in names]+[REPO/'airfare_fresh/features.py',REPO/'airfare_fresh/study.py']
    return {str(p.relative_to(REPO)):sha(p) for p in paths}
def versions():return {name:md.version(name) for name in VERSIONS}
def guard():
    assert not (ROOT/'submissions/INFERENCE_STARTED.json').exists(),'Inference already started; research/test reads blocked'
def verify(train):
    guard();r=read(ROOT/'REGISTRATION.json')
    assert r['sources']==sources() and r['training_sha256']==sha(train) and r['versions']==versions()
    return r

def shape(node,depth=0):
    children=node.get('children',[])
    if not children:return 1,depth
    parts=[shape(x,depth+1) for x in children]
    return sum(x[0] for x in parts),max(x[1] for x in parts)
def structure(model,family):
    if family.startswith('rf'):parts=[(t.get_n_leaves(),t.get_depth()) for t in model.estimators_];depth=2;count=200
    elif family.startswith('xgb'):parts=[shape(json.loads(t)) for t in model.get_booster().get_dump(dump_format='json')];depth=9;count=964
    else:parts=[(int(n),int(round(np.log2(n)))) for n in model.get_tree_leaf_counts()];depth=6;count=1000
    assert len(parts)==m.count(model,family)==count
    assert max(x[0] for x in parts)<=2**depth and max(x[1] for x in parts)<=depth
    return dict(actual_trees=count,max_observed_depth=max(x[1] for x in parts),
        max_observed_leaves=max(x[0] for x in parts),configured_depth_bound=depth,all_checks_passed=True)
