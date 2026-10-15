"""Fixed-configuration replication: registration, five-fold evaluation, full refits."""
from pathlib import Path
from datetime import datetime,timezone
import argparse,time
import numpy as np
import pandas as pd
import joblib
import common as c
m=c.m;ROOT=c.ROOT

def register(train):
    c.guard();assert not (ROOT/'REGISTRATION.json').exists()
    assert c.sha(train)=='6b0c9793817dc3e1b229ecea9ebbef14eef34452abf05a6f708d197cf6d8cc07'
    c.save(ROOT/'REGISTRATION.json',dict(created_at_utc=datetime.now(timezone.utc).isoformat(),
        sources=c.sources(),training_sha256=c.sha(train),versions=c.versions(),
        constructors={f:m.estimator(f).get_params() for f in m.FAMILIES},
        blends=m.BLENDS,outer_seed=c.OUTER_SEED,research_model_fits=30,target_encoder_fits=72,
        early_stopping=False,tuning_trials=0,test_accessed=False,
        interpretation='smooth=auto interpreted as sklearn TargetEncoder; peer library/formula unconfirmed'))
    print('Registered both encodings and all fixed configurations before model fits.',flush=True)

def train(train):
    reg=c.verify(train);frame,raw,y=c.base.data(train);groups=c.base.group_ids(raw)
    splits=c.base.splits(raw,y,c.OUTER_SEED);folds=np.empty(len(y),int)
    names=m.FAMILIES+list(m.BLENDS);oof={name:np.full(len(y),np.nan) for name in names};foldrows=[]
    for i,(tr,va) in enumerate(splits):
        assert set(groups[tr]).isdisjoint(groups[va]);folds[va]=i
        folder=ROOT/'results'/f'outer{i}';c.save(folder/'indices.json',dict(fit=tr,validation=va,group_overlap=0))
        trainpred={};validpred={}
        for family in m.FAMILIES:
            start=time.monotonic();features=m.Features(family)
            fit_raw=raw.iloc[tr].reset_index(drop=True)
            x=features.fit_transform(fit_raw,y[tr]);v=features.transform(raw.iloc[va])
            model=m.estimator(family);m.fit(model,family,features,x,y[tr])
            trainpred[family]=m.predict(model,x);validpred[family]=m.predict(model,v)
            row=dict(model=family,fold=i,**c.base.scores(y[va],validpred[family]),
                train_r2=c.base.scores(y[tr],trainpred[family])['r2'],
                seconds=time.monotonic()-start,**c.structure(model,family))
            row['gap']=row['train_r2']-row['r2'];foldrows.append(row)
            c.save(folder/f'{family}_metrics.json',row)
            if hasattr(features,'encoding_folds'):c.save(folder/f'{family}_encoding_folds.json',features.encoding_folds)
            print('OUTER',i,family,'R2',round(row['r2'],6),'gap',round(row['gap'],6),flush=True)
        for name,weights in m.BLENDS.items():
            trainpred[name]=sum(w*trainpred[f] for f,w in weights.items())
            validpred[name]=sum(w*validpred[f] for f,w in weights.items())
            row=dict(model=name,fold=i,**c.base.scores(y[va],validpred[name]),train_r2=c.base.scores(y[tr],trainpred[name])['r2'])
            row['gap']=row['train_r2']-row['r2'];foldrows.append(row)
        for name in names:oof[name][va]=validpred[name]
        np.savez_compressed(folder/'training_predictions.npz',row_position=tr,**trainpred)
    assert all(np.isfinite(p).all() for p in oof.values())
    results=ROOT/'analysis';results.mkdir(exist_ok=True)
    aligned=pd.DataFrame(dict(id=frame.id,price=y,group=groups.astype(str),fold=folds,**oof))
    aligned.to_csv(results/'aligned_oof.csv',index=False)
    pd.DataFrame(foldrows).to_csv(results/'fold_metrics.csv',index=False)
    configs={}
    for family in m.FAMILIES:
        start=time.monotonic();features=m.Features(family);x=features.fit_transform(raw,y)
        model=m.estimator(family);m.fit(model,family,features,x,y)
        folder=ROOT/'results/full_training';folder.mkdir(exist_ok=True)
        path=folder/f'{family}.joblib'
        joblib.dump(dict(family=family,features=features,model=model,scale=m.SCALE),path,compress=3)
        config=dict(family=family,constructor=model.get_params(),actual_structure=c.structure(model,family),
            feature_schema=features.description(),model_sha256=c.sha(path),training_rows=len(y),
            training_r2=c.base.scores(y,m.predict(model,x))['r2'],seconds=time.monotonic()-start,
            sources=c.sources(),test_accessed=False)
        if family=='cat':config['resolved_library_parameters']=model.get_all_params()
        if family.startswith('xgb'):config['resolved_library_parameters']=c.json.loads(model.get_booster().save_config())
        configs[family]=config
        c.save(folder/f'{family}_configuration.json',config)
        if hasattr(features,'encoding_folds'):c.save(folder/f'{family}_encoding_folds.json',features.encoding_folds)
        print('FINAL',family,'trees',config['actual_structure']['actual_trees'],flush=True)
    c.save(ROOT/'MODEL_CONFIGURATIONS.json',dict(models=configs,blends=m.BLENDS,
        scale=m.SCALE,unknowns='Peer feature list, exact encoder implementation, other parameters and target scale are unconfirmed.'))
    # Pre-existing controls are descriptive comparisons on verified identical rows.
    for name,path,column in [
        ('control_previous_equal64',c.REPO/'airfare_fresh/analysis/aligned_training_oof.csv','equal_blend_onehot64'),
        ('control_business_augmented',c.REPO/'airfare_business/primary/business_augmented_xgb_oof.csv','prediction')]:
        old=pd.read_csv(path,dtype={'group':str})
        assert np.array_equal(old.id,frame.id) and np.array_equal(old.price,y) and np.array_equal(old.fold,folds)
        assert np.array_equal(old.group.astype(str),groups.astype(str));oof[name]=old[column].to_numpy()
    modelrows=[];tailrows=[];subgroups=[]
    for name,p in oof.items():
        rows=[r for r in foldrows if r['model']==name]
        modelrows.append(dict(model=name,**c.base.scores(y,p),
            mean_train_r2=np.mean([r['train_r2'] for r in rows]) if rows else None,
            mean_gap=np.mean([r['gap'] for r in rows]) if rows else None,
            worst_fold_r2=min(r['r2'] for r in rows) if rows else None))
        err=p-y;sq=err**2;top=y>=np.quantile(y,.99);business=raw.flight_class.eq('Business').to_numpy()
        tailrows.append(dict(model=name,top1_rmse=np.sqrt(sq[top].mean()),top1_bias=err[top].mean(),
            top1_sse_share=sq[top].sum()/sq.sum(),business_sse_share=sq[business].sum()/sq.sum()))
        for value in raw.flight_class.unique():
            mask=raw.flight_class.eq(value).to_numpy()
            subgroups.append(dict(model=name,cabin=value,n=int(mask.sum()),**c.base.scores(y[mask],p[mask])))
    table=pd.DataFrame(modelrows).sort_values('r2',ascending=False)
    table.to_csv(results/'model_metrics.csv',index=False)
    pd.DataFrame(tailrows).to_csv(results/'tail_metrics.csv',index=False)
    pd.DataFrame(subgroups).to_csv(results/'cabin_metrics.csv',index=False)
    pd.DataFrame({n:p-y for n,p in oof.items()}).corr().to_csv(results/'residual_correlations.csv')
    c.save(ROOT/'EXPERIMENT_LEDGER.json',dict(regressor_fits=30,outer_fits=25,full_fits=5,
        target_encoder_mapping_fits=72,tuning_trials=0,versions=c.versions(),test_accessed=False,
        all_outer_groups_disjoint=True,all_target_encoding_groups_disjoint=True,
        inherited_reused_folds=True,public_feedback_used_as_tuning_objective=False))
    lines=['# Disclosed-peer settings: training-only results','',
        'This is an approximate replication. smooth=auto is interpreted as sklearn TargetEncoder, with group cross-fitting to protect labels. One-hot controls are also supplied. Feature lists, other settings and target scale are explicit assumptions; no claim to reproduce the peer score of 0.96080.','',
        '| Model | Outer OOF R² | Mean training R² | Mean gap |','| --- | ---: | ---: | ---: |']
    for r in table.to_dict('records'):
        fmt=lambda v:'—' if v is None or not np.isfinite(v) else f'{v:.6f}'
        lines.append(f"| {r['model']} | {fmt(r['r2'])} | {fmt(r['mean_train_r2'])} | {fmt(r['mean_gap'])} |")
    lines+=['','No hyperparameters, rounds, encodings or mixture weights were selected from these scores. All five component models and both predeclared equal blends are supplied under the explicit replication request. Reused folds and past development limit confidence; these are not Kaggle scores. No extra holdout has been manufactured retrospectively.','',
        'RF: 200 trees, depth2, no bootstrap, .7 features per split. XGB: 964 trees, depth9, lr .054. CatBoost: 1000 trees, depth6, lr .139. All seed42. See ../MODEL_CONFIGURATIONS.json and ../PROTOCOL.md for every setting and assumption.','',
        'For target-encoded models, reported training scores use the cross-fitted training matrix actually passed to the regressor. Validation/inference use the separate full-training categorical mapping. No validation price is used in either mapping. Numeric medians and feature-only mappings fit on outer training.','',
        'Tail/cabin errors and residual correlations are retained in adjacent CSVs. Previous controls have verified identical IDs, labels, folds and groups. The earlier public best 0.95885 for classic gradient remains a user-reported public result, not a comparison on these validation scores.']
    (results/'FINDINGS.md').write_text('\n'.join(lines)+'\n')
    c.save(results/'completion.json',dict(regressor_fits=30,models=5,blends=2,
        registration_sha256=c.sha(ROOT/'REGISTRATION.json'),configurations_sha256=c.sha(ROOT/'MODEL_CONFIGURATIONS.json'),
        sources=c.sources(),training_sha256=reg['training_sha256'],test_accessed=False))
    print(table[['model','r2','mean_gap']].to_string(index=False),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['register','train'],required=True)
    p.add_argument('--train',type=Path,required=True);a=p.parse_args()
    register(a.train) if a.stage=='register' else train(a.train)
