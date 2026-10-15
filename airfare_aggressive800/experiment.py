"""One preregistered aggressive Business profile; development reads train only."""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
import hashlib
import importlib.metadata as metadata
import json
from pathlib import Path
import sys
import time

import joblib
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parent
REPO=ROOT.parent
sys.path.insert(0,str(REPO/'airfare_business'))
import specialist as old
base=old.base
from features import TabularFeatures
import diagnostics as h

PROFILE=dict(depth=7,leaves=48,learning_rate=.045,min_child=8,l2=3.,l1=.05,
    split_gain=.01,row_sample=.95,column_sample=.90)
ROUNDS=800
SEEDS={'primary':20261020,'sensitivity':20261027}


def sources():
    paths=[Path(__file__),ROOT/'PROTOCOL.md',REPO/'airfare_business/specialist.py',
        REPO/'airfare_round2/direct_cat64.py',REPO/'airfare_fresh/study.py',REPO/'airfare_fresh/features.py']
    return {str(p.relative_to(REPO)):base.sha(p) for p in paths}


def read(path):return json.loads(path.read_text())
def local_hash(raw):return hashlib.sha256(pd.util.hash_pandas_object(raw,index=False).values.tobytes()).hexdigest()


def verify(train):
    old.common.checked_plan(train)
    reg=read(ROOT/'REGISTRATION.json')
    assert reg['source_hashes']==sources() and reg['training_sha256']==base.sha(train)
    assert reg['profile']==PROFILE and reg['rounds']==800
    assert reg['versions']=={name:metadata.version(name) for name in old.common.VERSIONS}
    for phase in ['airfare_business','airfare_round2']:
        r=read(REPO/phase/'REGISTRATION.json')
        for name,digest in r['source_hashes'].items():assert base.sha(REPO/name)==digest
    return reg


def evaluate(train,label):
    verify(train);frame,raw,y=base.data(train);dest=ROOT/label;dest.mkdir(exist_ok=True)
    refdir=REPO/'airfare_business'/label
    global_oof=pd.read_csv(refdir/'global_xgb_oof.csv')
    current_oof=pd.read_csv(refdir/'business_augmented_xgb_oof.csv')
    pd.testing.assert_frame_equal(global_oof.drop(columns='prediction'),current_oof.drop(columns='prediction'))
    assert np.array_equal(frame.id,global_oof.id) and np.array_equal(y,global_oof.price)
    prediction=np.full(len(y),np.nan);fold_id=np.full(len(y),-1);rows=[]
    for i,(tr,va) in enumerate(base.splits(raw,y,SEEDS[label])):
        assert np.array_equal(va,np.flatnonzero(global_oof.fold.to_numpy()==i))
        folder=dest/f'outer{i}';folder.mkdir(exist_ok=True)
        metrics=folder/'outer_metrics.json';predfile=folder/'validation_predictions.csv'
        fold_id[va]=i
        if metrics.exists():
            row=read(metrics);assert row['source_hashes']==sources()
            cached=pd.read_csv(predfile);assert np.array_equal(cached.row_position,va)
            prediction[va]=cached.prediction.to_numpy();rows.append(row);continue
        selected=read(refdir/f'outer{i}/global_selected.json');local=raw.iloc[tr].reset_index(drop=True)
        assert selected['signature']['row_hash']==local_hash(local)
        assert selected['signature']['target_hash']==hashlib.sha256(y[tr].tobytes()).hexdigest()
        groups=base.group_ids(raw);assert set(groups[tr]).isdisjoint(groups[va])
        start=time.perf_counter()
        gf=TabularFeatures(selected['mode'],'xgb');gx=gf.fit_transform(raw.iloc[tr]);gv=gf.transform(raw.iloc[va])
        gm=base.model('xgb',selected['configuration'],selected['rounds']);base.fit(gm,'xgb',gf,gx,y[tr])
        gp=base.predict(gm,gv);gt=base.predict(gm,gx)
        np.testing.assert_allclose(gp,global_oof.prediction.to_numpy()[va],rtol=0,atol=1e-6)
        bt=raw.iloc[tr].flight_class.to_numpy()=='Business';bv=raw.iloc[va].flight_class.to_numpy()=='Business'
        f=TabularFeatures('context','xgb');x=f.fit_transform(raw.iloc[tr[bt]]);xv=f.transform(raw.iloc[va[bv]])
        m=old.specialist_model(PROFILE,800);base.fit(m,'xgb',f,x,y[tr[bt]])
        assert m.get_booster().num_boosted_rounds()==800
        p=gp.copy();pt=gt.copy();p[bv]=.5*gp[bv]+.5*base.predict(m,xv);pt[bt]=.5*gt[bt]+.5*base.predict(m,x)
        assert np.array_equal(p[~bv],gp[~bv])
        prediction[va]=p
        prior=read(refdir/f'outer{i}/outer_metrics.json')
        prior_row=next(r for r in prior['rows'] if r['model']=='business_augmented_xgb')
        row=dict(fold=i,**base.scores(y[va],p),train_r2=base.scores(y[tr],pt)['r2'],
            reference_r2=prior_row['r2'],reference_train_r2=prior_row['train_r2'],
            delta_r2=base.scores(y[va],p)['r2']-prior_row['r2'],n_fit=len(tr),n_validation=len(va),
            business_fit_rows=int(bt.sum()),business_validation_rows=int(bv.sum()),specialist_trees=800,
            global_trees=selected['rounds'],global_selection_sha256=base.sha(refdir/f'outer{i}/global_selected.json'),
            group_overlap=0,global_input_identities_verified=True,global_reconstruction_max_abs_error=float(np.max(np.abs(gp-global_oof.prediction.to_numpy()[va]))),
            economy_predictions_unchanged=True,refit_seconds=time.perf_counter()-start,source_hashes=sources())
        pd.DataFrame(dict(row_position=va,prediction=p)).to_csv(predfile,index=False);base.save(metrics,row);rows.append(row)
        print('OUTER',label,i,'candidate_R2',row['r2'],'reference_R2',row['reference_r2'],'train',row['train_r2'],flush=True)
    assert np.isfinite(prediction).all()
    pd.DataFrame(dict(row_position=np.arange(len(y)),id=frame.id,price=y,group=base.group_ids(raw),fold=fold_id,prediction=prediction)).to_csv(dest/'oof.csv',index=False)
    base.save(dest/'summary.json',dict(source_hashes=sources(),outer_seed=SEEDS[label],folds=rows,**base.scores(y,prediction),
        mean_train_r2=float(np.mean([r['train_r2'] for r in rows])),mean_gap=float(np.mean([r['train_r2']-r['r2'] for r in rows]))))


def final(train):
    verify(train);_,raw,y=base.data(train);dest=ROOT/'full_training';dest.mkdir(exist_ok=True)
    path=dest/'final_model.joblib';config_path=dest/'final_configuration.json'
    if config_path.exists():
        c=read(config_path);assert c['source_hashes']==sources() and c['model_sha256']==base.sha(path);return
    business=raw.flight_class.to_numpy()=='Business'
    f=TabularFeatures('context','xgb');x=f.fit_transform(raw.loc[business])
    m=old.specialist_model(PROFILE,800);base.fit(m,'xgb',f,x,y[business])
    assert m.get_booster().num_boosted_rounds()==800
    joblib.dump(dict(features=f,model=m,profile=PROFILE,rounds=800,scale=10000.),path,compress=3)
    base.save(config_path,dict(family='aggressive_business_xgb800',profile=PROFILE,rounds=800,
        actual_trees=m.get_booster().num_boosted_rounds(),constructor_parameters=m.get_params(),
        resolved_booster_configuration=json.loads(m.get_booster().save_config()),features=f.get_feature_names_out().tolist(),
        training_rows=len(y),specialist_training_rows=int(business.sum()),training_sha256=base.sha(train),
        full_business_train_r2=base.scores(y[business],base.predict(m,x))['r2'],source_hashes=sources(),model_sha256=base.sha(path),
        prediction_policy={'Business':{'global_xgb':.5,'specialist':.5},'Economy':{'global_xgb':1.}},test_accessed=False))
    print('FINAL_COMPLETE800',flush=True)


def analyze(train):
    verify(train);trainframe=pd.read_csv(train);out=ROOT/'analysis';out.mkdir(exist_ok=True)
    tables={};tail_tables={};fold_tables={}
    for label in SEEDS:
        d=pd.read_csv(ROOT/label/'oof.csv');old_oof=pd.read_csv(REPO/'airfare_business'/label/'business_augmented_xgb_oof.csv')
        pd.testing.assert_frame_equal(d.drop(columns='prediction'),old_oof.drop(columns='prediction'))
        y=d.price.to_numpy();preds={'current_business_augmented':old_oof.prediction.to_numpy(),'aggressive800':d.prediction.to_numpy()}
        summary=read(ROOT/label/'summary.json');economy=trainframe.flight_class.to_numpy()=='Economy'
        np.testing.assert_allclose(preds['current_business_augmented'][economy],preds['aggressive800'][economy],rtol=0,atol=1e-6)
        metrics=[];foldrows=[];subgroups=[];tails=[];worst=[]
        support=np.zeros(len(y),int)
        for i in range(5):
            mask=d.fold.to_numpy()==i
            support[mask]=trainframe.loc[mask,'flight'].map(trainframe.loc[~mask,'flight'].value_counts()).fillna(0).to_numpy(int)
        groups={'cabin':trainframe.flight_class.astype(str),
            'target_band':pd.cut(y,[-np.inf,np.quantile(y,.9),np.quantile(y,.99),np.inf],labels=['bottom90pct','90-99pct','top1pct']).astype(str),
            'outer_training_flight_support':pd.cut(support,[-1,0,5,20,np.inf],labels=['unseen','1-5','6-20','21+']).astype(str)}
        for name,p in preds.items():
            fit_key='train_r2' if name=='aggressive800' else 'reference_train_r2'
            local=[]
            for row in summary['folds']:
                mask=d.fold.to_numpy()==row['fold'];score=h.metrics(y[mask],p[mask])
                score.update(model=name,fold=row['fold'],train_r2=row[fit_key],gap=row[fit_key]-score['r2']);local.append(score);foldrows.append(score)
            metrics.append(dict(model=name,**h.metrics(y,p),mean_train_r2=float(np.mean([r['train_r2'] for r in local])),
                mean_gap=float(np.mean([r['gap'] for r in local])),worst_fold_r2=min(r['r2'] for r in local),std_fold_r2=float(np.std([r['r2'] for r in local],ddof=1))))
            squared=(p-y)**2;top=y>=np.quantile(y,.99)
            tails.append(dict(model=name,top1pct_target_sse_share=float(squared[top].sum()/squared.sum()),top1pct_target_rmse=float(np.sqrt(squared[top].mean())),
                top1pct_target_bias=float((p-y)[top].mean()),business_sse_share=float(squared[~economy].sum()/squared.sum())))
            for group,values in groups.items():
                values=np.asarray(values,str)
                for value in np.unique(values):
                    mask=values==value;subgroups.append(dict(model=name,group=group,value=value,**h.metrics(y[mask],p[mask])))
            for i in np.argsort(squared)[-30:][::-1]:
                worst.append(dict(model=name,row_position=int(i),**trainframe.iloc[i].to_dict(),prediction=float(p[i]),error=float(p[i]-y[i]),outer_training_flight_support=int(support[i])))
        folder=out/label;folder.mkdir(exist_ok=True)
        for filename,rows in [('model_metrics',metrics),('fold_metrics',foldrows),('tail_metrics',tails),('subgroup_metrics',subgroups),('worst_errors',worst)]:pd.DataFrame(rows).to_csv(folder/(filename+'.csv'),index=False)
        ci,draws=h.bootstrap_groups(y,preds,d.group.to_numpy(),500,20261110 if label=='primary' else 20261111)
        h.save_json(folder/'bootstrap_intervals.json',ci);draws.to_csv(folder/'bootstrap_draws.csv',index=False)
        fig,ax=h.plt.subplots(figsize=(8,4))
        for name in preds:ax.plot(range(5),[r['r2'] for r in foldrows if r['model']==name],marker='o',label=name)
        ax.set(xlabel='Outer fold',ylabel='Original-price R²',title=f'Fixed aggressive candidate: {label}');ax.legend(fontsize=8)
        fig.tight_layout();fig.savefig(folder/'paired_fold_scores.png',dpi=160);h.plt.close(fig)
        tables[label]=pd.DataFrame(metrics);tail_tables[label]=pd.DataFrame(tails);fold_tables[label]=pd.DataFrame(foldrows)
    text=['# Fixed 800-tree exploratory Business candidate',
        'This is one predeclared aggressive profile, not an optimized tree-count search. The user explicitly requested an exploratory portfolio slot with weaker validation acceptable. It changes several capacity/regularization parameters together, so tree-count causality cannot be isolated.']
    for label in SEEDS:
        text.extend([f'## {label.capitalize()} reused-fold evaluation',h.markdown_table(tables[label],['model','r2','rmse','mae','mean_train_r2','mean_gap','worst_fold_r2'],6),
            h.markdown_table(fold_tables[label],['model','fold','r2','train_r2','gap'],6),h.markdown_table(tail_tables[label],precision=5)])
    text.extend(['## Interpretation',
        'Original-price outer validation never chooses this profile, its fixed800 rounds or the mixture. All prior model-selection history and these folds/observations have already informed development; no fresh holdout is claimed. Conditional group-bootstrap intervals omit retraining and accumulated selection uncertainty. Smaller gaps and additional trees do not prove generalization.',
        'This candidate can be delivered despite weaker validation under the user’s explicit exploratory authorization. It is not represented as a stronger conservative finalist, and no public/private score or top-three rank is promised. Preserve earlier submissions. The competition’s final-selection limit remains unverified.',
        'The global branch exactly reconstructs previous per-partition controls. Economy predictions remain unchanged. Every feature mapping is fitted only on the corresponding global or Business fitting rows; no test features/predictions inform model development. Budget21fits: ten reconstructed global outer fits, ten specialist outer fits, one full Business fit. No Optuna trials or early-stopping fits.',
        'Final file uses the unchanged454-tree full global XGBoost and the new800-tree Business specialist in fixed halves on Business; Economy uses global only. Exact configurations, sources, versions and inference audit are delivered separately.'])
    (out/'FINDINGS.md').write_text('\n\n'.join(text)+'\n')
    base.save(out/'completion.json',dict(source_hashes=sources(),outer_scopes=10,reconstructed_global_fits=10,specialist_outer_fits=10,specialist_full_fits=1,
        research_model_fits=21,optuna_trials=0,economy_oof_unchanged=True,test_accessed_by_analysis=False))
    for label,t in tables.items():print(label,t[['model','r2','mean_gap']].to_string(index=False),flush=True)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--train',type=Path,default=base.TRAIN)
    ap.add_argument('--stage',choices=['all','evaluate','final','analyze'],required=True);a=ap.parse_args()
    if a.stage in ['all','evaluate']:
        for label in SEEDS:evaluate(a.train,label)
    if a.stage in ['all','final']:final(a.train)
    if a.stage in ['all','analyze']:analyze(a.train)


if __name__=='__main__':main()
