"""Finite Optuna research; no competition-test or submission input path exists."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata as metadata
import json
from pathlib import Path
import sys
import time
import joblib
import numpy as np
import pandas as pd
import optuna
import models as m

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
sys.path.insert(0, str(REPO/'airfare_fresh'))
# Load the preserved splitter under an explicit name, avoiding entrypoint-name ambiguity.
import importlib.util
_spec = importlib.util.spec_from_file_location('airfare_preserved_study', REPO/'airfare_fresh/study.py')
base = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(base)
VERSIONS = ['numpy', 'pandas', 'scipy', 'scikit-learn', 'xgboost-cpu',
            'lightgbm', 'catboost', 'optuna', 'joblib']
OUTER_SEED = 20261020
INNER_SEED = 20264520
FULL_SEED = 20265520


def read(path): return json.loads(Path(path).read_text())
def save(path, value): base.save(path, value)
def sha(path): return base.sha(path)
def hash_rows(raw):
    return hashlib.sha256(pd.util.hash_pandas_object(raw, index=False).values.tobytes()).hexdigest()
def hash_y(y): return hashlib.sha256(np.asarray(y, float).tobytes()).hexdigest()


def sources():
    paths = [ROOT/'models.py', ROOT/'study.py', ROOT/'PROTOCOL.md',
             REPO/'airfare_fresh/study.py', REPO/'airfare_fresh/features.py']
    return {str(p.relative_to(REPO)):sha(p) for p in paths}


def register(train):
    value = dict(created_at_utc=datetime.now(timezone.utc).isoformat(),
        source_hashes=sources(), training_sha256=sha(train), families=m.FAMILIES,
        trials_per_scope=m.TRIALS, rounds=m.ROUNDS, blend_weights=m.BLENDS,
        outer_seed=OUTER_SEED, inner_seed=INNER_SEED, full_seed=FULL_SEED,
        research_trial_budget=336, research_fit_budget=384,
        versions={name:metadata.version(name) for name in VERSIONS},
        test_access_permitted=False, public_scores_are_optimization_inputs=False)
    path=ROOT/'REGISTRATION.json'
    if path.exists():
        old=read(path);value['created_at_utc']=old['created_at_utc']
        assert old==value, 'Registration changed; refusing to overwrite'
    else: save(path,value)
    return value


def verify(train):
    reg=read(ROOT/'REGISTRATION.json')
    assert reg['source_hashes']==sources() and reg['training_sha256']==sha(train)
    assert reg['versions']=={name:metadata.version(name) for name in VERSIONS}
    assert reg['families']==m.FAMILIES and reg['trials_per_scope']==m.TRIALS
    assert reg['rounds']==m.ROUNDS and reg['blend_weights']==m.BLENDS
    assert not (ROOT/'submissions/INFERENCE_STARTED.json').exists(), 'Research is frozen after inference'
    return reg


def tune(raw, y, family, out, seed):
    out.mkdir(parents=True,exist_ok=True)
    signature=dict(source_hashes=sources(),row_hash=hash_rows(raw),target_hash=hash_y(y),
                   family=family,seed=seed,trials=m.TRIALS[family])
    selected_path=out/'selection.json'
    if selected_path.exists():
        result=read(selected_path);assert result['signature']==signature;return result
    tr,va=base.splits(raw,y,seed)[0]
    group=base.group_ids(raw)
    assert set(group[tr]).isdisjoint(group[va])
    save(out/'inner_indices.json',dict(fit=tr.tolist(),validation=va.tolist(),
        group_overlap=0,positions_are_local_to_tuning_input=True,seed=seed))
    features=m.Features(family);x=features.fit_transform(raw.iloc[tr]);xv=features.transform(raw.iloc[va])
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study=optuna.create_study(direction='minimize',study_name=family,
        sampler=optuna.samplers.TPESampler(seed=seed+100*m.FAMILIES.index(family),
            n_startup_trials=4 if m.TRIALS[family]==8 else 3),
        storage='sqlite:///'+str((out/'optuna.sqlite3').resolve()),load_if_exists=True)
    previous=study.user_attrs.get('signature')
    if previous is not None: assert previous==signature, 'Unsafe Optuna resume'
    study.set_user_attr('signature',signature)
    if len(study.trials)==0:
        for index in [0,1]: study.enqueue_trial(m.initial(family,index))
    def objective(trial):
        p=m.propose(trial,family);tick=time.perf_counter()
        estimator=m.estimator(family,p,features)
        m.fit(estimator,family,features,x,y[tr]);pred=m.predict(estimator,xv)
        actual=int(m.actual_rounds(estimator,family))
        trial.set_user_attr('configuration',p)
        trial.set_user_attr('actual_rounds',actual)
        trial.set_user_attr('seconds',time.perf_counter()-tick)
        trial.set_user_attr('tuning_metrics',base.scores(y[va],pred))
        print('TRIAL',family,out.parent.name,trial.number,'R2',round(base.scores(y[va],pred)['r2'],6),
              'requested',p['rounds'],'actual',actual,'seconds',round(time.perf_counter()-tick,2),flush=True)
        return float(np.mean((pred-y[va])**2))
    # Count all attempted trials, including failures, rather than granting extra trials.
    attempted=sum(t.state!=optuna.trial.TrialState.WAITING for t in study.trials)
    remaining=m.TRIALS[family]-attempted
    if remaining>0: study.optimize(objective,n_trials=remaining,gc_after_trial=True)
    trials=[dict(number=t.number,state=t.state.name,value=t.value,
                 parameters=t.params,attributes=t.user_attrs) for t in study.trials]
    save(out/'trials.json',trials)
    assert len(trials)==m.TRIALS[family] and all(t['state']=='COMPLETE' for t in trials)
    best=study.best_trial
    result=dict(signature=signature,best_trial=best.number,configuration=best.user_attrs['configuration'],
        tuning_mse=best.value,tuning_actual_rounds=best.user_attrs['actual_rounds'],
        trial_count=len(trials),n_inner_fit=len(tr),n_inner_validation=len(va),
        selection_policy='lowest single inner80/20 original-price MSE; fixed high round count; fresh outer/full refit')
    save(selected_path,result)
    return result


def run_family(train,family):
    reg=verify(train);frame,raw,y=base.data(train);dest=ROOT/'results'/family
    dest.mkdir(parents=True,exist_ok=True)
    group=base.group_ids(raw);splits=base.splits(raw,y,OUTER_SEED)
    oof=np.full(len(y),np.nan);folds=np.full(len(y),-1);records=[]
    for i,(tr,va) in enumerate(splits):
        folder=dest/f'outer{i}';folder.mkdir(exist_ok=True);folds[va]=i
        path=folder/'metrics.json'
        if path.exists():
            row=read(path);assert row['source_hashes']==sources()
            cached=pd.read_csv(folder/'predictions.csv')
            assert np.array_equal(cached.row_position,va)
            oof[va]=cached.prediction.to_numpy();records.append(row);continue
        save(folder/'outer_indices.json',dict(fit=tr.tolist(),validation=va.tolist(),group_overlap=0))
        assert set(group[tr]).isdisjoint(group[va])
        local=raw.iloc[tr].reset_index(drop=True)
        selected=tune(local,y[tr],family,folder/'tuning',INNER_SEED+i)
        tick=time.perf_counter();features=m.Features(family)
        x=features.fit_transform(raw.iloc[tr]);xv=features.transform(raw.iloc[va])
        model=m.estimator(family,selected['configuration'],features)
        m.fit(model,family,features,x,y[tr]);pred=m.predict(model,xv);pt=m.predict(model,x)
        oof[va]=pred
        row=dict(family=family,fold=i,source_hashes=sources(),selection=selected,
            **base.scores(y[va],pred),train_r2=base.scores(y[tr],pt)['r2'],
            n_fit=len(tr),n_validation=len(va),group_overlap=0,
            actual_rounds=int(m.actual_rounds(model,family)),refit_seconds=time.perf_counter()-tick)
        save(path,row)
        pd.DataFrame(dict(row_position=va,prediction=pred)).to_csv(folder/'predictions.csv',index=False)
        # Training predictions allow honest composite blend training-gap diagnostics.
        np.savez_compressed(folder/'train_predictions.npz',row_position=tr,prediction=pt)
        records.append(row)
        print('OUTER',family,i,'R2',round(row['r2'],7),'train',round(row['train_r2'],7),
              'actual',row['actual_rounds'],flush=True)
    assert np.isfinite(oof).all()
    pd.DataFrame(dict(row_position=np.arange(len(y)),id=frame.id,price=y,group=group,
                      fold=folds,prediction=oof)).to_csv(dest/'oof.csv',index=False)
    save(dest/'summary.json',dict(family=family,source_hashes=sources(),**base.scores(y,oof),folds=records,
         mean_train_r2=float(np.mean([r['train_r2'] for r in records])),
         mean_gap=float(np.mean([r['train_r2']-r['r2'] for r in records]))))
    final=dest/'full_training';final.mkdir(exist_ok=True)
    final_config=final/'configuration.json';artifact=final/'model.joblib'
    if final_config.exists():
        c=read(final_config);assert c['source_hashes']==sources() and sha(artifact)==c['model_sha256'];return
    selected=tune(raw,y,family,final/'tuning',FULL_SEED)
    features=m.Features(family);x=features.fit_transform(raw)
    model=m.estimator(family,selected['configuration'],features)
    m.fit(model,family,features,x,y)
    joblib.dump(dict(features=features,model=model,family=family,scale=m.SCALE),artifact,compress=3)
    config=dict(family=family,selection=selected,constructor_parameters=m.parameters(model,family),
        actual_rounds=int(m.actual_rounds(model,family)),feature_schema=features.description(),
        source_hashes=sources(),training_sha256=reg['training_sha256'],training_rows=len(y),
        training_r2=base.scores(y,m.predict(model,x))['r2'],model_sha256=sha(artifact),
        target_policy='price/10000 squared error; inverse10000; clip below0',test_accessed=False)
    save(final_config,config)
    print('FINAL',family,'OOF_R2',base.scores(y,oof)['r2'],'rounds',config['actual_rounds'],flush=True)


def smoke(train):
    # API checks use training rows only and fixed8 rounds, never tune research choices.
    frame,raw,y=base.data(train)
    fit_indices=np.arange(1200);val_indices=np.arange(1200,1500)
    rows=[]
    for family in m.FAMILIES:
        features=m.Features(family);x=features.fit_transform(raw.iloc[fit_indices]);xv=features.transform(raw.iloc[val_indices])
        import optuna
        p=m.propose(optuna.trial.FixedTrial(m.initial(family,0)),family);p['rounds']=8
        tick=time.perf_counter();model=m.estimator(family,p,features)
        m.fit(model,family,features,x,y[fit_indices]);pred=m.predict(model,xv)
        assert len(pred)==len(val_indices) and (pred>=0).all()
        assert m.actual_rounds(model,family)>0
        before=features.description();changed=raw.iloc[val_indices].copy()
        changed.loc[:,'flight']='NEW-UNSEEN-FLIGHT'
        assert features.transform(changed).shape[0]==len(changed)
        assert before==features.description()
        rows.append(dict(family=family,api_fit_passed=True,unseen_category_transform_passed=True,
                         rounds=int(m.actual_rounds(model,family)),seconds=time.perf_counter()-tick))
        print('API_SMOKE',family,'passed',flush=True)
    save(ROOT/'api_smoke.json',dict(training_data_only=True,smoke_fits=len(rows),checks=rows,
                                   model_choices_not_selected_by_smoke_scores=True))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--train',type=Path,default=base.TRAIN)
    ap.add_argument('--stage',choices=['smoke','register','run'],required=True)
    ap.add_argument('--family',choices=m.FAMILIES+['all'],default='all');args=ap.parse_args()
    if args.stage=='smoke':smoke(args.train)
    elif args.stage=='register':register(args.train)
    else:
        for family in m.FAMILIES if args.family=='all' else [args.family]:run_family(args.train,family)


if __name__=='__main__':main()
