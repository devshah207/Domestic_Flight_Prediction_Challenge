"""Registered direct CatBoost64 study. Development reads training data only."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata as metadata
import json
from pathlib import Path
import sys
import time

import joblib
import numpy as np
import optuna
import pandas as pd

ROOT = Path(__file__).resolve().parent
FRESH = ROOT.parent / 'airfare_fresh'
sys.path.insert(0, str(FRESH))
import study as base
from features import TabularFeatures

TRAIN = base.TRAIN
VARIANT = 'cat64_direct'
PRIMARY = 20261020
SECONDARY = 20261027
VERSIONS = ['numpy', 'pandas', 'scipy', 'scikit-learn', 'xgboost-cpu', 'lightgbm', 'catboost', 'optuna', 'joblib']


def sources():
    return {str(p.relative_to(ROOT.parent)): base.sha(p) for p in
            [Path(__file__), ROOT/'PROTOCOL.md', FRESH/'study.py', FRESH/'features.py']}


def model(p, rounds, threshold, stopping=False):
    return base.model('cat', p, rounds, stopping).set_params(one_hot_max_size=threshold)


def checked_plan(train):
    old = json.loads((FRESH/'submissions/frozen_inference_manifest.json').read_text())['frozen']
    assert base.sha(train) == old['training_sha256'], 'Training identity changed'
    for name, digest in old['source_sha256'].items():
        assert base.sha(FRESH/name) == digest, name
    for family in ['xgb', 'lgb', 'cat', 'cat_onehot64']:
        directory = FRESH/'results'/family/'full_training'
        c = json.loads((directory/'final_configuration.json').read_text())
        assert base.sha(directory/'final_model.joblib') == c['model_sha256']
    return old


def tune(raw, y, out, name, seed, threshold):
    out.mkdir(parents=True, exist_ok=True)
    signature = dict(sources=sources(), input_rows=base.sha(TRAIN),
        raw_sha256=hashlib.sha256(pd.util.hash_pandas_object(raw,index=False).values.tobytes()).hexdigest(),
        target_sha256=hashlib.sha256(y.tobytes()).hexdigest(), seed=seed, threshold=threshold,
        trials=12, cap=1200, patience=70, versions={p:metadata.version(p) for p in VERSIONS})
    selected_path = out/'selected.json'
    if selected_path.exists():
        selected = json.loads(selected_path.read_text())
        assert selected['signature'] == signature, 'Changed study signature'
        return selected
    db = out/'trials.sqlite3'
    sampler = optuna.samplers.TPESampler(seed=seed, n_startup_trials=5)
    sampler_path = out/'sampler.joblib'
    sampler_state = joblib.load(sampler_path) if sampler_path.exists() else None
    if sampler_state is not None:
        sampler = sampler_state['sampler']
    study = optuna.create_study(direction='minimize', study_name=name,
        storage='sqlite:///'+str(db), load_if_exists=True, sampler=sampler)
    previous = study.user_attrs.get('signature')
    assert previous is None or previous == signature, 'Changed study signature'
    study.set_user_attr('signature', signature)
    if len(study.trials) == 0:
        study.enqueue_trial(base.initial('cat','minimal'))
        study.enqueue_trial(base.initial('cat','context'))
    completed = sum(t.state == optuna.trial.TrialState.COMPLETE for t in study.trials)
    assert all(t.state in (optuna.trial.TrialState.COMPLETE,optuna.trial.TrialState.WAITING) for t in study.trials), 'Interrupted/failed trial: stop for diagnosis'
    if completed:
        assert sampler_state is not None and sampler_state['completed'] == completed, 'Sampler checkpoint mismatch'
    inner = base.splits(raw,y,seed)[:2]
    cache = {}
    for mode in ['minimal','context']:
        for k,(tr,va) in enumerate(inner):
            f=TabularFeatures(mode,'cat')
            cache[mode,k]=(f,f.fit_transform(raw.iloc[tr]),f.transform(raw.iloc[va]),tr,va)
    def objective(trial):
        mode,p=base.configuration(trial,'cat'); details=[]; sse=0.; n=0
        for k in range(2):
            f,xt,xv,tr,va=cache[mode,k]
            start=time.perf_counter(); m=model(p,1200,threshold,True)
            base.fit(m,'cat',f,xt,y[tr],xv,y[va])
            pred=base.predict(m,xv)
            sse+=float(np.sum((y[va]-pred)**2)); n+=len(va)
            details.append(dict(split=k,rounds=base.rounds_used(m,'cat'), n_fit=len(tr),n_tune=len(va),
                fit_hash=hashlib.sha256(tr.tobytes()).hexdigest(),tune_hash=hashlib.sha256(va.tobytes()).hexdigest(),
                seconds=time.perf_counter()-start,**base.scores(y[va],pred)))
        rounds=max(100,min(1200,int(np.median([d['rounds'] for d in details]))))
        trial.set_user_attr('configuration',p);trial.set_user_attr('mode',mode)
        trial.set_user_attr('inner_splits',details);trial.set_user_attr('rounds',rounds)
        print(name,'trial',trial.number,'inner_MSE',round(sse/n,2),'rounds',rounds,flush=True)
        return sse/n
    def checkpoint(study,trial):
        state={'sampler':study.sampler,'completed':sum(t.state==optuna.trial.TrialState.COMPLETE for t in study.trials)}
        temporary=sampler_path.with_suffix('.tmp');joblib.dump(state,temporary);temporary.replace(sampler_path)
        base.save(out/'all_trials.json',[dict(number=t.number,state=t.state.name,value=t.value,parameters=t.params,attributes=t.user_attrs) for t in study.trials])
    if completed<12:
        study.optimize(objective,n_trials=12-completed,callbacks=[checkpoint],gc_after_trial=True)
    assert len(study.trials)==12 and all(t.state==optuna.trial.TrialState.COMPLETE for t in study.trials)
    b=study.best_trial
    selected=dict(study=name,trial=b.number,mode=b.user_attrs['mode'],configuration=b.user_attrs['configuration'],
        rounds=b.user_attrs['rounds'],inner_splits=b.user_attrs['inner_splits'],threshold=threshold,
        tuning_mse=b.value,raw_optuna_parameters=b.params,trial_count=12,signature=signature)
    base.save(selected_path,selected)
    return selected


def reselect(raw,y,selected,seed,out):
    path=out/'selected.json'
    if path.exists():
        result=json.loads(path.read_text());assert result['source_hashes']==sources();return result
    details=[]
    for k,(tr,va) in enumerate(base.splits(raw,y,seed)[:2]):
        f=TabularFeatures(selected['mode'],'cat');xt=f.fit_transform(raw.iloc[tr]);xv=f.transform(raw.iloc[va])
        start=time.perf_counter();m=model(selected['configuration'],1200,64,True)
        base.fit(m,'cat',f,xt,y[tr],xv,y[va])
        details.append(dict(split=k,rounds=base.rounds_used(m,'cat'),n_fit=len(tr),n_tune=len(va),
            seconds=time.perf_counter()-start,**base.scores(y[va],base.predict(m,xv))))
    result=dict(selected,threshold=64,inner_splits=details,rounds=max(100,min(1200,int(np.median([d['rounds'] for d in details])))),
        source_hashes=sources(),interpretation='threshold2 selection then threshold64 round reselection; zero new trials')
    base.save(path,result);return result


def refit_outer(raw,y,tr,va,selected,dest,fold):
    metrics_path=dest/'outer_metrics.json';pred_path=dest/'validation_predictions.csv'
    if metrics_path.exists():
        r=json.loads(metrics_path.read_text());assert r['source_hashes']==sources();return r,pd.read_csv(pred_path).prediction.to_numpy()
    start=time.perf_counter();f=TabularFeatures(selected['mode'],'cat')
    xt=f.fit_transform(raw.iloc[tr]);xv=f.transform(raw.iloc[va])
    m=model(selected['configuration'],selected['rounds'],selected['threshold'])
    base.fit(m,'cat',f,xt,y[tr]);p=base.predict(m,xv)
    assert m.tree_count_==selected['rounds']
    r=dict(fold=fold,selected=selected,n_fit=len(tr),n_validation=len(va),
        train_r2=base.scores(y[tr],base.predict(m,xt))['r2'],refit_seconds=time.perf_counter()-start,
        actual_trees=m.tree_count_,source_hashes=sources(),**base.scores(y[va],p))
    pd.DataFrame({'row_position':va,'prediction':p}).to_csv(pred_path,index=False)
    base.save(metrics_path,r)
    print('OUTER',dest.parent.name,fold,'R2',r['r2'],'train',r['train_r2'],'trees',m.tree_count_,flush=True)
    return r,p


def outer(train,out,secondary=False):
    checked_plan(train);frame,raw,y=base.data(train)
    seed=SECONDARY if secondary else PRIMARY
    folds=base.splits(raw,y,seed)
    names=['cat2_control','hybrid64_control',VARIANT] if secondary else [VARIANT]
    predictions={k:np.full(len(y),np.nan) for k in names};records={k:[] for k in names};fold_id=np.full(len(y),-1)
    for i,(tr,va) in enumerate(folds):
        fold_id[va]=i;inner_seed=(20272220 if secondary else 20262220)+i
        local=raw.iloc[tr].reset_index(drop=True)
        if secondary:
            dest=out/'cat2_control'/f'outer{i}'
            selected=tune(local,y[tr],dest,f'secondary_cat2_{i}',inner_seed,2)
            r,p=refit_outer(raw,y,tr,va,selected,dest,i);records['cat2_control'].append(r);predictions['cat2_control'][va]=p
            dest=out/'hybrid64_control'/f'outer{i}';dest.mkdir(parents=True,exist_ok=True)
            hybrid=reselect(local,y[tr],selected,inner_seed,dest)
            r,p=refit_outer(raw,y,tr,va,hybrid,dest,i);records['hybrid64_control'].append(r);predictions['hybrid64_control'][va]=p
        dest=out/VARIANT/f'outer{i}'
        selected=tune(local,y[tr],dest,f'{"secondary" if secondary else "primary"}_direct64_{i}',inner_seed,64)
        r,p=refit_outer(raw,y,tr,va,selected,dest,i);records[VARIANT].append(r);predictions[VARIANT][va]=p
    for name in names:
        assert np.isfinite(predictions[name]).all()
        pd.DataFrame(dict(row_position=np.arange(len(y)),id=frame.id,price=y,group=base.group_ids(raw),fold=fold_id,prediction=predictions[name])).to_csv(out/name/'oof.csv',index=False)
        rows=records[name]
        base.save(out/name/'summary.json',dict(family=name,outer_seed=seed,folds=rows,**base.scores(y,predictions[name]),
            mean_train_r2=float(np.mean([r['train_r2'] for r in rows])),mean_fold_r2=float(np.mean([r['r2'] for r in rows])),
            mean_gap=float(np.mean([r['train_r2']-r['r2'] for r in rows])),source_hashes=sources()))
    print('OUTER_COMPLETE',seed,flush=True)


def gate(out):
    new=pd.read_csv(out/VARIANT/'oof.csv')
    old={k:pd.read_csv(FRESH/'results'/k/'oof.csv') for k in ['xgb','lgb','cat_onehot64']}
    for df in old.values():pd.testing.assert_frame_equal(new.drop(columns='prediction'),df.drop(columns='prediction'))
    y=new.price.to_numpy();a=new.prediction.to_numpy();b=old['cat_onehot64'].prediction.to_numpy()
    shared=old['xgb'].prediction.to_numpy()+old['lgb'].prediction.to_numpy()
    ab=(shared+a)/3;bb=(shared+b)/3
    delta_cat=base.scores(y,a)['r2']-base.scores(y,b)['r2']
    delta_blend=base.scores(y,ab)['r2']-base.scores(y,bb)['r2']
    fold_deltas=[base.scores(y[new.fold==i],ab[new.fold==i])['r2']-base.scores(y[new.fold==i],bb[new.fold==i])['r2'] for i in range(5)]
    decision=bool(delta_cat>=.0005 or (delta_blend>=.0002 and sum(d>0 for d in fold_deltas)>=3))
    result=dict(run_secondary=decision,delta_r2_cat=delta_cat,delta_r2_blend=delta_blend,blend_fold_deltas=fold_deltas,
        note='Registered compute allocation only; not a promotion or statistical significance test')
    base.save(ROOT/'sensitivity_decision.json',result);return result


def final(train,out):
    checked_plan(train);_,raw,y=base.data(train);dest=out/VARIANT/'full_training'
    if (dest/'final_configuration.json').exists():
        c=json.loads((dest/'final_configuration.json').read_text());assert c['source_hashes']==sources();assert c['model_sha256']==base.sha(dest/'final_model.joblib');return
    selected=tune(raw,y,dest,'full_direct64',20263220,64)
    f=TabularFeatures(selected['mode'],'cat');x=f.fit_transform(raw)
    m=model(selected['configuration'],selected['rounds'],64);start=time.perf_counter();base.fit(m,'cat',f,x,y)
    assert m.tree_count_==selected['rounds']
    path=dest/'final_model.joblib';joblib.dump(dict(features=f,model=m,selected=selected,scale=10000.),path,compress=3)
    base.save(dest/'final_configuration.json',dict(family=VARIANT,selection=selected,constructor_parameters=m.get_params(),
        resolved_parameters=m.get_all_params(),actual_trees=m.tree_count_,features=f.get_feature_names_out().tolist(),
        training_rows=len(y),training_sha256=base.sha(train),model_sha256=base.sha(path),source_hashes=sources(),
        full_train_r2=base.scores(y,base.predict(m,x))['r2'],refit_seconds=time.perf_counter()-start,test_accessed=False))
    print('FINAL_COMPLETE',selected['rounds'],selected['mode'],flush=True)


def main():
    global TRAIN
    parser=argparse.ArgumentParser();parser.add_argument('--train',type=Path,default=TRAIN)
    parser.add_argument('--stage',choices=['primary','sensitivity','final','all','smoke'],required=True)
    args=parser.parse_args();TRAIN=args.train
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    if args.stage=='smoke':
        # API-only synthetic smoke; no labeled competition rows or outer score.
        from catboost import CatBoostRegressor
        x=pd.DataFrame({'category':['a','b','c','d']*10,'number':np.arange(40)})
        m=CatBoostRegressor(iterations=4,depth=4,max_leaves=16,grow_policy='Lossguide',one_hot_max_size=64,thread_count=4,verbose=False,allow_writing_files=False)
        m.fit(x,np.arange(40)/100,cat_features=['category']);assert len(m.predict(x))==40;assert m.get_all_params()['one_hot_max_size']==64
        print('SYNTHETIC_SMOKE_PASSED');return
    checked_plan(TRAIN)
    if args.stage in ('primary','all'):outer(TRAIN,ROOT/'primary')
    if args.stage in ('sensitivity','all'):
        decision=gate(ROOT/'primary');print('SENSITIVITY_GATE',decision,flush=True)
        if decision['run_secondary']:outer(TRAIN,ROOT/'sensitivity',True)
    if args.stage in ('final','all'):final(TRAIN,ROOT/'primary')


if __name__=='__main__':main()
