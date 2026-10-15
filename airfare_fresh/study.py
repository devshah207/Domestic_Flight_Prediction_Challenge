"""Independent bounded Optuna study. Kaggle test data is never read here."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import importlib.metadata as metadata
import joblib
import numpy as np
import pandas as pd
import optuna
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import StratifiedGroupKFold
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor, early_stopping, log_evaluation
from catboost import CatBoostRegressor
from features import TabularFeatures

ROOT=Path(__file__).resolve().parent
TRAIN=Path('/workspace/attachments/085aeb37-413d-42c9-bb16-baafd968488d/train.csv')
FAMILIES=['xgb','lgb','cat']
SCALE=10000.
CAP=1200
PATIENCE=70
TRIALS=12
MODEL_SEED=42
OUTER_SEED=20261020
THREADS=4

def plain(v):
    if isinstance(v,dict):return {str(k):plain(x) for k,x in v.items()}
    if isinstance(v,(list,tuple)):return [plain(x) for x in v]
    if isinstance(v,np.generic):return plain(v.item())
    if isinstance(v,float) and not np.isfinite(v):return 'NaN'
    if isinstance(v,Path):return str(v)
    return v

def save(path,v):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(plain(v),indent=2,allow_nan=False)+'\n')

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def scores(y,p):
    return dict(r2=float(r2_score(y,p)),rmse=float(np.sqrt(mean_squared_error(y,p))),
        mae=float(mean_absolute_error(y,p)),bias=float(np.mean(p-y)))

def data(path):
    d=pd.read_csv(path)
    assert d.id.is_unique and np.isfinite(d.price).all() and (d.price>0).all()
    return d,d.drop(columns=['id','price']),d.price.to_numpy(float)

def group_ids(raw):
    # Conservative itinerary grouping prevents missingness from hiding repeats.
    # Duration variants sharing these eight columns are intentionally blocked too.
    return pd.util.hash_pandas_object(raw.drop(columns=['airline','duration']).astype(str),index=False).to_numpy()

def splits(raw,y,seed):
    prefix=raw.flight.astype(str).str.split().str[0]
    base=prefix+'|'+raw.flight_class.astype(str)
    counts=base.value_counts()
    base=base.where(base.map(counts)>=100,'OTHER|'+raw.flight_class.astype(str))
    strata=pd.Series('',index=np.arange(len(raw)),dtype=object)
    for b in base.unique():
        ix=np.flatnonzero(base.to_numpy()==b)
        q=pd.qcut(pd.Series(y[ix]).rank(method='first'),5,labels=False).astype(str)
        strata.iloc[ix]=str(b)+'|'+q.to_numpy()
    g=group_ids(raw)
    out=list(StratifiedGroupKFold(5,shuffle=True,random_state=seed).split(raw,strata,g))
    for tr,va in out:assert set(g[tr]).isdisjoint(g[va])
    return out

def configuration(trial,family):
    mode=trial.suggest_categorical('feature_mode',['minimal','context'])
    depth=trial.suggest_int('depth',4,7)
    p=dict(depth=depth,
        learning_rate=trial.suggest_float('learning_rate',.025,.07,log=True),
        leaves=trial.suggest_categorical('leaves',[16,24,32,48]),
        row_sample=trial.suggest_float('row_sample',.72,.95),
        column_sample=trial.suggest_float('column_sample',.7,.95),
        l2=trial.suggest_float('l2',3.,35.,log=True))
    p['leaves']=min(int(p['leaves']),2**depth)
    if family in ['xgb','lgb']:
        p['l1']=trial.suggest_float('l1',.02,1.5,log=True)
        p['split_gain']=trial.suggest_float('split_gain',.005,.2,log=True)
        p['min_child']=trial.suggest_int('min_child',8 if family=='xgb' else 15,50 if family=='xgb' else 80)
    else:
        p['min_child']=trial.suggest_int('min_child',10,60)
        p['random_strength']=trial.suggest_float('random_strength',.2,2.,log=True)
    return mode,p

def model(family,p,rounds,stopping=False):
    if family=='xgb':
        q=dict(n_estimators=int(rounds),max_depth=p['depth'],max_leaves=p['leaves'],
            learning_rate=p['learning_rate'],min_child_weight=p['min_child'],
            subsample=p['row_sample'],colsample_bytree=p['column_sample'],
            reg_lambda=p['l2'],reg_alpha=p['l1'],gamma=p['split_gain'],
            objective='reg:squarederror',tree_method='hist',max_bin=192,
            grow_policy='depthwise',enable_categorical=True,max_cat_threshold=32,
            max_cat_to_onehot=4,random_state=MODEL_SEED,n_jobs=THREADS,verbosity=0)
        if stopping:q.update(early_stopping_rounds=PATIENCE,eval_metric='rmse')
        return XGBRegressor(**q)
    if family=='lgb':
        return LGBMRegressor(n_estimators=int(rounds),max_depth=p['depth'],num_leaves=p['leaves'],
            learning_rate=p['learning_rate'],min_child_samples=p['min_child'],
            subsample=p['row_sample'],subsample_freq=1,colsample_bytree=p['column_sample'],
            reg_lambda=p['l2'],reg_alpha=p['l1'],min_split_gain=p['split_gain'],
            objective='regression',max_bin=191,max_cat_threshold=32,max_cat_to_onehot=4,
            cat_smooth=20.,cat_l2=10.,min_data_per_group=10,
            random_state=MODEL_SEED,n_jobs=THREADS,verbosity=-1,deterministic=True,force_col_wise=True)
    return CatBoostRegressor(iterations=int(rounds),depth=p['depth'],max_leaves=p['leaves'],
        grow_policy='Lossguide',learning_rate=p['learning_rate'],min_data_in_leaf=p['min_child'],
        l2_leaf_reg=p['l2'],random_strength=p['random_strength'],
        bootstrap_type='Bernoulli',subsample=p['row_sample'],rsm=p['column_sample'],
        loss_function='RMSE',eval_metric='RMSE',random_seed=MODEL_SEED,thread_count=THREADS,
        one_hot_max_size=2,max_ctr_complexity=1,ctr_leaf_count_limit=512,
        model_size_reg=1.,verbose=False,allow_writing_files=False,
        **(dict(early_stopping_rounds=PATIENCE) if stopping else {}))

def fit(m,family,features,x,y,xv=None,yv=None):
    kw={}
    if family=='cat':kw['cat_features']=features.cat_columns_
    if xv is not None:
        kw['eval_set']=(xv,yv/SCALE) if family=='cat' else [(xv,yv/SCALE)]
        if family=='xgb':kw['verbose']=False
        if family=='lgb':kw['callbacks']=[early_stopping(PATIENCE,verbose=False),log_evaluation(0)]
        if family=='cat':kw['use_best_model']=True
    m.fit(x,y/SCALE,**kw)
    return m

def rounds_used(m,family):
    if family=='xgb':return int(m.best_iteration+1)
    if family=='lgb':return int(m.best_iteration_)
    return int(m.get_best_iteration()+1)

def predict(m,x):
    p=np.maximum(0.,np.asarray(m.predict(x),float).reshape(-1)*SCALE)
    assert np.isfinite(p).all()
    return p

def initial(family,mode):
    p=dict(feature_mode=mode,depth=6,learning_rate=.045,leaves=32,
        row_sample=.85,column_sample=.85,l2=12.,min_child=20 if family=='xgb' else 35)
    if family in ['xgb','lgb']:p.update(l1=.2,split_gain=.04)
    else:p.update(random_strength=.7)
    return p

def tune(raw,y,family,name,out,seed):
    inner=splits(raw,y,seed)[:2]
    # Cache only feature matrices independently fitted on each inner fit partition.
    cache={}
    for mode in ['minimal','context']:
        for k,(tr,va) in enumerate(inner):
            f=TabularFeatures(mode,family);xt=f.fit_transform(raw.iloc[tr]);xv=f.transform(raw.iloc[va])
            cache[mode,k]=(f,xt,xv,tr,va)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    database=out/(name+'.sqlite3')
    st=optuna.create_study(direction='minimize',study_name=name,
        storage='sqlite:///'+str(database),load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=seed,n_startup_trials=5))
    signature=dict(code={p.name:sha(p) for p in [ROOT/'study.py',ROOT/'features.py']},
        row_hash=hashlib.sha256(pd.util.hash_pandas_object(raw,index=False).values.tobytes()).hexdigest(),
        target_hash=hashlib.sha256(y.tobytes()).hexdigest(),seed=seed,family=family,trials=TRIALS,cap=CAP,patience=PATIENCE)
    old=st.user_attrs.get('signature')
    if old is not None and old!=signature:raise RuntimeError('Study inputs/code changed; refusing unsafe resume')
    st.set_user_attr('signature',signature)
    if len(st.trials)==0:
        st.enqueue_trial(initial(family,'minimal'));st.enqueue_trial(initial(family,'context'))
    def objective(trial):
        mode,p=configuration(trial,family); details=[];total_sse=0.;count=0
        start=time.perf_counter()
        for k in [0,1]:
            f,xt,xv,tr,va=cache[mode,k]
            m=model(family,p,CAP,True);fit(m,family,f,xt,y[tr],xv,y[va])
            pv=predict(m,xv);bst=rounds_used(m,family)
            total_sse+=float(np.sum((pv-y[va])**2));count+=len(va)
            details.append(dict(split=k,rounds=bst,n_fit=len(tr),n_tune=len(va),**scores(y[va],pv)))
        selected=max(100,min(CAP,int(np.median([d['rounds'] for d in details]))))
        trial.set_user_attr('mode',mode);trial.set_user_attr('configuration',p)
        trial.set_user_attr('inner_splits',details);trial.set_user_attr('selected_rounds',selected)
        trial.set_user_attr('seconds',time.perf_counter()-start)
        print(name,'trial',trial.number,'MSE',round(total_sse/count,2),'rounds',selected,flush=True)
        return total_sse/count
    finished=sum(t.state==optuna.trial.TrialState.COMPLETE for t in st.trials)
    if finished<TRIALS:st.optimize(objective,n_trials=TRIALS-finished,gc_after_trial=True)
    best=st.best_trial
    result=dict(study=name,best_trial=best.number,tuning_mse=best.value,
        mode=best.user_attrs['mode'],configuration=best.user_attrs['configuration'],
        rounds=best.user_attrs['selected_rounds'],inner_splits=best.user_attrs['inner_splits'],
        trial_count=sum(t.state==optuna.trial.TrialState.COMPLETE for t in st.trials),
        raw_optuna_parameters=best.params,round_policy='fresh median of two inner80/20 stops; floor100 ceiling1200; refit100%',
        signature=signature)
    save(out/(name+'_selected.json'),result)
    save(out/(name+'_all_trials.json'),[dict(number=t.number,state=str(t.state),value=t.value,
        parameters=t.params,attributes=t.user_attrs) for t in st.trials])
    return result

def outer(train,family,out):
    frame,raw,y=data(train);folds=splits(raw,y,OUTER_SEED)
    oof=np.full(len(y),np.nan);fold_id=np.full(len(y),-1);records=[]
    family_dir=out/family;family_dir.mkdir(parents=True,exist_ok=True)
    for i,(tr,va) in enumerate(folds):
        dest=family_dir/f'outer{i}';dest.mkdir(exist_ok=True)
        sel=tune(raw.iloc[tr].reset_index(drop=True),y[tr],family,f'{family}_outer{i}',dest,
            20262020+FAMILIES.index(family)*100+i)
        start=time.perf_counter();f=TabularFeatures(sel['mode'],family)
        xt=f.fit_transform(raw.iloc[tr]);xv=f.transform(raw.iloc[va])
        m=model(family,sel['configuration'],sel['rounds']);fit(m,family,f,xt,y[tr])
        pv=predict(m,xv);pt=predict(m,xt);oof[va]=pv;fold_id[va]=i
        record=dict(fold=i,selected=sel,train_r2=float(r2_score(y[tr],pt)),
            n_fit=len(tr),n_validation=len(va),refit_seconds=time.perf_counter()-start,**scores(y[va],pv))
        records.append(record);save(dest/'outer_metrics.json',record)
        print('OUTER',family,i,'R2',record['r2'],'train',record['train_r2'],'rounds',sel['rounds'],flush=True)
    assert np.isfinite(oof).all() and (fold_id>=0).all()
    pd.DataFrame(dict(row_position=np.arange(len(y)),id=frame.id,price=y,group=group_ids(raw),
        fold=fold_id,prediction=oof)).to_csv(family_dir/'oof.csv',index=False)
    summary=dict(family=family,**scores(y,oof),folds=records,
        mean_train_r2=float(np.mean([x['train_r2'] for x in records])),
        mean_fold_r2=float(np.mean([x['r2'] for x in records])),
        std_fold_r2=float(np.std([x['r2'] for x in records],ddof=1)),
        mean_gap=float(np.mean([x['train_r2']-x['r2'] for x in records])),
        evaluation='nested outer performance of the entire12trial tuning procedure; not an independent test of final full-data dict')
    save(family_dir/'summary.json',summary);print('SUMMARY',family,summary['r2'],flush=True)

def final_training(train,family,out):
    frame,raw,y=data(train);dest=out/family/'full_training';dest.mkdir(parents=True,exist_ok=True)
    sel=tune(raw,y,family,f'{family}_full',dest,20263020+FAMILIES.index(family)*100)
    f=TabularFeatures(sel['mode'],family);x=f.fit_transform(raw)
    m=model(family,sel['configuration'],sel['rounds']);fit(m,family,f,x,y)
    artifact=dest/'final_model.joblib';joblib.dump(dict(features=f,model=m,selected=sel,scale=SCALE),artifact,compress=3)
    params=m.get_all_params() if family=='cat' else m.get_params()
    save(dest/'final_configuration.json',dict(family=family,selection=sel,estimator_parameters=params,
        features=f.get_feature_names_out().tolist(),training_rows=len(y),training_sha256=sha(train),
        model_sha256=sha(artifact),source_hashes={p.name:sha(p) for p in [ROOT/'study.py',ROOT/'features.py']},
        full_train_r2=float(r2_score(y,predict(m,x))),test_accessed=False))
    print('FINAL_FROZEN',family,sel['rounds'],sel['mode'],flush=True)

def smoke(train):
    _,raw,y=data(train);tr,va=splits(raw,y,OUTER_SEED)[0]
    it,iv=splits(raw.iloc[tr].reset_index(drop=True),y[tr],20262020)[0]
    for family in FAMILIES:
        trial=optuna.trial.FixedTrial(initial(family,'minimal'));mode,p=configuration(trial,family)
        f=TabularFeatures(mode,family);xt=f.fit_transform(raw.iloc[tr[it]]);xv=f.transform(raw.iloc[tr[iv]])
        tick=time.perf_counter();m=model(family,p,100,True);fit(m,family,f,xt,y[tr[it]],xv,y[tr[iv]])
        assert len(predict(m,xv))==len(iv)
        print('SMOKE',family,'100tree ceiling',round(time.perf_counter()-tick,2),'seconds; APIs and prediction shape verified',flush=True)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--train',type=Path,default=TRAIN)
    ap.add_argument('--family',choices=FAMILIES+['all'],default='all')
    ap.add_argument('--stage',choices=['smoke','outer','final'],required=True)
    ap.add_argument('--out',type=Path,default=ROOT/'results');a=ap.parse_args()
    a.out.mkdir(parents=True,exist_ok=True)
    if a.stage=='smoke':smoke(a.train);return
    for family in FAMILIES if a.family=='all' else [a.family]:
        (outer if a.stage=='outer' else final_training)(a.train,family,a.out)

if __name__=='__main__':main()
