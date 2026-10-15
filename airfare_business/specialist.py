"""Registered bounded Business specialist; training data only during development."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import joblib
import numpy as np
import optuna
import pandas as pd

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent/'airfare_round2'))
import direct_cat64 as common
base=common.base
from features import TabularFeatures


def sources():
    return {str(p.relative_to(ROOT.parent)):base.sha(p) for p in [Path(__file__),ROOT/'PROTOCOL.md',ROOT.parent/'airfare_round2/direct_cat64.py',common.FRESH/'study.py',common.FRESH/'features.py']}


def initial(second=False):
    return dict(depth=5 if second else 6,leaves=24 if second else 32,learning_rate=.035 if second else .045,
        row_sample=.85,column_sample=.85,l2=20. if second else 12.,l1=.2,split_gain=.04,min_child=30 if second else 20)


def specialist_model(p,rounds,stopping=False):
    m=base.model('xgb',p,rounds,stopping)
    # XGBoost3.1.1 callable metrics expect a tag removed by sklearn1.8.
    # Set it only on new model instances, without mutating installed libraries.
    m._estimator_type='regressor'
    return m


def configuration(t):
    depth=t.suggest_int('depth',4,6)
    return dict(depth=depth,leaves=min(t.suggest_categorical('leaves',[16,24,32]),2**depth),
        learning_rate=t.suggest_float('learning_rate',.025,.065,log=True),row_sample=t.suggest_float('row_sample',.75,.95),
        column_sample=t.suggest_float('column_sample',.70,.95),l2=t.suggest_float('l2',5.,40.,log=True),
        l1=t.suggest_float('l1',.05,1.5,log=True),split_gain=t.suggest_float('split_gain',.01,.2,log=True),
        min_child=t.suggest_int('min_child',12,60))


def inner_cache(raw,y,global_selected,inner_seed):
    cache=[]
    for k,(tr,va) in enumerate(base.splits(raw,y,inner_seed)[:2]):
        start=time.perf_counter()
        f=TabularFeatures(global_selected['mode'],'xgb');xt=f.fit_transform(raw.iloc[tr]);xv=f.transform(raw.iloc[va])
        m=base.model('xgb',global_selected['configuration'],global_selected['inner_splits'][k]['rounds'])
        base.fit(m,'xgb',f,xt,y[tr]);global_pred=base.predict(m,xv)
        # Reconstruction should recover the original inner validation score.
        score=base.scores(y[va],global_pred)
        assert abs(score['r2']-global_selected['inner_splits'][k]['r2'])<1e-9
        bt=tr[raw.iloc[tr].flight_class.to_numpy()=='Business']
        bv=va[raw.iloc[va].flight_class.to_numpy()=='Business']
        assert len(bt)>100 and len(bv)>100
        sf=TabularFeatures('context','xgb');sx=sf.fit_transform(raw.iloc[bt]);sv=sf.transform(raw.iloc[bv])
        cache.append(dict(features=sf,x=sx,xv=sv,y=y[bt],yv=y[bv],global_pred=global_pred[raw.iloc[va].flight_class.to_numpy()=='Business'],
            n_fit=len(bt),n_tune=len(bv),fit_positions_sha256=hashlib.sha256(bt.tobytes()).hexdigest(),
            tune_positions_sha256=hashlib.sha256(bv.tobytes()).hexdigest(),global_reconstruction_seconds=time.perf_counter()-start,
            global_r2_reconstruction_error=score['r2']-global_selected['inner_splits'][k]['r2']))
    return cache


def tune(raw,y,global_selected,inner_seed,sampler_seed,dest,train):
    dest.mkdir(parents=True,exist_ok=True)
    signature=dict(source_hashes=sources(),training_sha256=base.sha(train),
        local_raw_sha256=hashlib.sha256(pd.util.hash_pandas_object(raw,index=False).values.tobytes()).hexdigest(),
        local_y_sha256=hashlib.sha256(y.tobytes()).hexdigest(),global_selected_sha256=hashlib.sha256(json.dumps(global_selected,sort_keys=True).encode()).hexdigest(),
        inner_seed=inner_seed,sampler_seed=sampler_seed,trials=12,cap=1200,patience=70,mixture=.5)
    selected_path=dest/'selected.json'
    if selected_path.exists():
        selected=json.loads(selected_path.read_text());assert selected['signature']==signature;return selected
    cache=inner_cache(raw,y,global_selected,inner_seed)
    base.save(dest/'global_inner_reconstructions.json',[{k:v for k,v in c.items() if k in ['n_fit','n_tune','fit_positions_sha256','tune_positions_sha256','global_reconstruction_seconds','global_r2_reconstruction_error']} for c in cache])
    sp=dest/'sampler.joblib';state=joblib.load(sp) if sp.exists() else None
    sampler=state['sampler'] if state else optuna.samplers.TPESampler(seed=sampler_seed,n_startup_trials=5)
    study=optuna.create_study(study_name='business_'+dest.parent.name+'_'+dest.name,direction='minimize',
        storage='sqlite:///'+str(dest/'trials.sqlite3'),load_if_exists=True,sampler=sampler)
    assert study.user_attrs.get('signature') in (None,signature)
    study.set_user_attr('signature',signature)
    if not study.trials:study.enqueue_trial(initial());study.enqueue_trial(initial(True))
    n=sum(t.state==optuna.trial.TrialState.COMPLETE for t in study.trials)
    assert all(t.state in (optuna.trial.TrialState.COMPLETE,optuna.trial.TrialState.WAITING) for t in study.trials)
    if n:assert state and state['completed']==n
    def objective(trial):
        p=configuration(trial);details=[];sse=0.;count=0
        for k,c in enumerate(cache):
            def halfblend_rmse(y_true,y_pred):
                pred=.5*c['global_pred']/10000.+.5*np.maximum(0.,y_pred)
                return float(np.sqrt(np.mean((pred-y_true)**2)))
            m=specialist_model(p,1200,True).set_params(eval_metric=halfblend_rmse)
            start=time.perf_counter();base.fit(m,'xgb',c['features'],c['x'],c['y'],c['xv'],c['yv'])
            specialist=base.predict(m,c['xv']);mixed=.5*c['global_pred']+.5*specialist
            sse+=float(np.sum((mixed-c['yv'])**2));count+=len(mixed)
            details.append(dict(split=k,rounds=base.rounds_used(m,'xgb'),trained_rounds=m.get_booster().num_boosted_rounds(),
                n_fit=c['n_fit'],n_tune=c['n_tune'],seconds=time.perf_counter()-start,
                best_plus_patience_reaches_ceiling=base.rounds_used(m,'xgb')+70>=1200,
                **base.scores(c['yv'],mixed)))
        rounds=max(100,min(1200,int(np.median([d['rounds'] for d in details]))))
        trial.set_user_attr('configuration',p);trial.set_user_attr('inner_splits',details);trial.set_user_attr('rounds',rounds)
        print(dest.parent.name,dest.name,'trial',trial.number,'business_halfblend_MSE',round(sse/count,2),'trees',rounds,flush=True)
        return sse/count
    def checkpoint(study,trial):
        tmp=sp.with_suffix('.tmp');joblib.dump(dict(sampler=study.sampler,completed=sum(t.state==optuna.trial.TrialState.COMPLETE for t in study.trials)),tmp);tmp.replace(sp)
        base.save(dest/'all_trials.json',[dict(number=t.number,state=t.state.name,value=t.value,parameters=t.params,attributes=t.user_attrs) for t in study.trials])
    if n<12:study.optimize(objective,n_trials=12-n,callbacks=[checkpoint],gc_after_trial=True)
    assert len(study.trials)==12 and all(t.state==optuna.trial.TrialState.COMPLETE for t in study.trials)
    b=study.best_trial
    selected=dict(study=study.study_name,trial=b.number,configuration=b.user_attrs['configuration'],mode='context',
        rounds=b.user_attrs['rounds'],inner_splits=b.user_attrs['inner_splits'],raw_optuna_parameters=b.params,
        objective='original-price Business halfblend MSE',tuning_mse=b.value,trial_count=12,signature=signature)
    base.save(selected_path,selected);return selected


def outer(train,secondary=False):
    common.checked_plan(train);frame,raw,y=base.data(train);label='sensitivity' if secondary else 'primary'
    out=ROOT/label;out.mkdir(exist_ok=True)
    seed=20261027 if secondary else 20261020;folds=base.splits(raw,y,seed)
    predictions={k:np.full(len(y),np.nan) for k in ['global_xgb','business_augmented_xgb']}
    fold_id=np.full(len(y),-1);rows=[]
    reference=pd.read_csv(common.FRESH/'results/xgb/oof.csv') if not secondary else None
    for i,(tr,va) in enumerate(folds):
        dest=out/f'outer{i}';dest.mkdir(exist_ok=True);fold_id[va]=i
        if (dest/'outer_metrics.json').exists():
            r=json.loads((dest/'outer_metrics.json').read_text());assert r['source_hashes']==sources()
            cached=pd.read_csv(dest/'validation_predictions.csv')
            assert np.array_equal(cached.row_position,va)
            for k in predictions:predictions[k][va]=cached[k].to_numpy()
            rows.extend(r['rows']);continue
        inner_seed=(20276020 if secondary else 20262020)+i
        local=raw.iloc[tr].reset_index(drop=True)
        if secondary:
            global_selected=base.tune(local,y[tr],'xgb',f'xgb_secondary{i}',dest,inner_seed)
        else:
            global_selected=json.loads((common.FRESH/f'results/xgb/outer{i}/xgb_outer{i}_selected.json').read_text())
            assert np.array_equal(va,np.flatnonzero(reference.fold.to_numpy()==i))
        base.save(dest/'global_selected.json',global_selected)
        sf=tune(local,y[tr],global_selected,inner_seed,(20277020 if secondary else 20274020)+i,dest,train)
        # Reconstruct whole-outer-training global fit and verify the primary OOF.
        start=time.perf_counter();gf=TabularFeatures(global_selected['mode'],'xgb');gx=gf.fit_transform(raw.iloc[tr]);gv=gf.transform(raw.iloc[va])
        gm=base.model('xgb',global_selected['configuration'],global_selected['rounds']);base.fit(gm,'xgb',gf,gx,y[tr])
        gp=base.predict(gm,gv);gpt=base.predict(gm,gx)
        if reference is not None:np.testing.assert_allclose(gp,reference.prediction.to_numpy()[va],rtol=0,atol=1e-6)
        business_tr=raw.iloc[tr].flight_class.to_numpy()=='Business';business_va=raw.iloc[va].flight_class.to_numpy()=='Business'
        f=TabularFeatures('context','xgb');x=f.fit_transform(raw.iloc[tr[business_tr]]);xv=f.transform(raw.iloc[va[business_va]])
        m=specialist_model(sf['configuration'],sf['rounds']);base.fit(m,'xgb',f,x,y[tr[business_tr]])
        p=gp.copy();pt=gpt.copy();p[business_va]=.5*gp[business_va]+.5*base.predict(m,xv);pt[business_tr]=.5*gpt[business_tr]+.5*base.predict(m,x)
        assert np.array_equal(p[~business_va],gp[~business_va])
        local_rows=[]
        for name,pr,ptr in [('global_xgb',gp,gpt),('business_augmented_xgb',p,pt)]:
            predictions[name][va]=pr
            r=dict(model=name,fold=i,n_fit=len(tr),n_validation=len(va),business_fit_rows=int(business_tr.sum()),business_validation_rows=int(business_va.sum()),
                train_r2=base.scores(y[tr],ptr)['r2'],**base.scores(y[va],pr))
            local_rows.append(r);rows.append(r)
        base.save(dest/'outer_metrics.json',dict(source_hashes=sources(),rows=local_rows,specialist_selected=sf,refit_seconds=time.perf_counter()-start,
            global_rounds=global_selected['rounds'],specialist_rounds=sf['rounds'],economy_predictions_unchanged=True))
        pd.DataFrame(dict(row_position=va,**{k:predictions[k][va] for k in predictions})).to_csv(dest/'validation_predictions.csv',index=False)
        print('BUSINESS_OUTER',label,i,'global_R2',local_rows[0]['r2'],'augmented_R2',local_rows[1]['r2'],flush=True)
    for k,p in predictions.items():
        assert np.isfinite(p).all()
        pd.DataFrame(dict(row_position=np.arange(len(y)),id=frame.id,price=y,group=base.group_ids(raw),fold=fold_id,prediction=p)).to_csv(out/(k+'_oof.csv'),index=False)
    base.save(out/'summary.json',dict(source_hashes=sources(),outer_seed=seed,folds=rows,
        metrics={k:base.scores(y,p) for k,p in predictions.items()}))


def gate():
    r=ROOT/'primary';a=pd.read_csv(r/'business_augmented_xgb_oof.csv');b=pd.read_csv(r/'global_xgb_oof.csv')
    controls=[pd.read_csv(common.FRESH/'results'/f/'oof.csv') for f in ['lgb','cat_onehot64']]
    for d in [b]+controls:pd.testing.assert_frame_equal(a.drop(columns='prediction'),d.drop(columns='prediction'))
    y=a.price.to_numpy();ap=a.prediction.to_numpy();bp=b.prediction.to_numpy();shared=sum(d.prediction.to_numpy() for d in controls)
    delta=base.scores(y,ap)['r2']-base.scores(y,bp)['r2'];blend_delta=base.scores(y,(ap+shared)/3)['r2']-base.scores(y,(bp+shared)/3)['r2']
    folds=[base.scores(y[a.fold==i],((ap+shared)/3)[a.fold==i])['r2']-base.scores(y[a.fold==i],((bp+shared)/3)[a.fold==i])['r2'] for i in range(5)]
    decision=dict(run_secondary=bool(delta>=.0005 or (blend_delta>=.0002 and sum(v>0 for v in folds)>=3)),
        augmented_delta_r2=delta,blend_delta_r2=blend_delta,blend_fold_deltas=folds,
        note='Prospective compute rule, not significance or submission eligibility')
    base.save(ROOT/'sensitivity_decision.json',decision);print('BUSINESS_SENSITIVITY_GATE',decision,flush=True);return decision


def final(train):
    common.checked_plan(train);_,raw,y=base.data(train);dest=ROOT/'full_training';dest.mkdir(exist_ok=True)
    if (dest/'final_configuration.json').exists():
        c=json.loads((dest/'final_configuration.json').read_text());assert c['source_hashes']==sources();assert c['model_sha256']==base.sha(dest/'final_model.joblib');return
    global_selected=json.loads((common.FRESH/'results/xgb/full_training/xgb_full_selected.json').read_text())
    selected=tune(raw,y,global_selected,20263020,20275020,dest,train)
    business=raw.flight_class.to_numpy()=='Business';f=TabularFeatures('context','xgb');x=f.fit_transform(raw.loc[business])
    m=specialist_model(selected['configuration'],selected['rounds']);base.fit(m,'xgb',f,x,y[business])
    path=dest/'final_model.joblib';joblib.dump(dict(features=f,model=m,selected=selected,scale=10000.),path,compress=3)
    base.save(dest/'final_configuration.json',dict(family='business_xgb_specialist',selection=selected,constructor_parameters=m.get_params(),
        resolved_booster_configuration=json.loads(m.get_booster().save_config()),actual_trees=m.get_booster().num_boosted_rounds(),
        feature_names=f.get_feature_names_out().tolist(),training_rows=len(y),specialist_training_rows=int(business.sum()),
        training_sha256=base.sha(train),model_sha256=base.sha(path),source_hashes=sources(),
        full_business_train_r2=base.scores(y[business],base.predict(m,x))['r2'],mixture=.5,test_accessed=False,
        compatibility={'_estimator_type':'regressor','reason':'XGBoost3.1.1 callable metric expects estimator tag removed in sklearn1.8'}))
    print('BUSINESS_FINAL_COMPLETE',selected['rounds'],flush=True)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--train',type=Path,default=base.TRAIN)
    ap.add_argument('--stage',choices=['primary','sensitivity','final','all'],required=True);a=ap.parse_args()
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    if a.stage in ['primary','all']:outer(a.train)
    if a.stage in ['sensitivity','all']:
        if gate()['run_secondary']:outer(a.train,True)
    if a.stage in ['final','all']:final(a.train)


if __name__=='__main__':main()
