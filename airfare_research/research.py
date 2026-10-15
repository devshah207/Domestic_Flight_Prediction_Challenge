"""Train-only airfare research. This module never opens Kaggle test data."""
from __future__ import annotations
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import time
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from xgboost import XGBRegressor
from historical_features import TrainingOnlyFeatureBuilder, make_ohe_features, OHE_CAT_COLS, NUM_COLS

ROOT = Path(__file__).resolve().parent
TRAIN = Path('/workspace/attachments/085aeb37-413d-42c9-bb16-baafd968488d/train.csv')
BASE = dict(n_estimators=850, max_depth=7, max_leaves=48, learning_rate=.03,
    min_child_weight=12, subsample=.8, colsample_bytree=.85, reg_lambda=15.,
    reg_alpha=.6, gamma=.08, max_bin=192, objective='reg:squarederror',
    tree_method='hist', random_state=42, n_jobs=4, verbosity=0)
SCALE = 10000.

def save_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, default=lambda v: v.item() if isinstance(v,np.generic) else str(v)))

def load_train(path=TRAIN):
    frame = pd.read_csv(path)
    assert frame.id.is_unique and np.isfinite(frame.price).all() and (frame.price > 0).all()
    return frame, frame.drop(columns=['id','price']), frame.price.to_numpy(float)

def groups_for(raw, scheme='raw'):
    if scheme == 'flight':
        return pd.factorize(raw.flight)[0]
    if scheme == 'semantic':
        # Stress test: missing airline/duration cannot hide otherwise identical records.
        z=raw.drop(columns=['airline','duration'])
    else:
        z=raw
    return pd.util.hash_pandas_object(z.astype(str).fillna('__NA__'),index=False).to_numpy()

def splits_for(raw,y,scheme='raw',seed=20261015,n_splits=5):
    strata=pd.qcut(pd.Series(y).rank(method='first'),10,labels=False).to_numpy()
    if scheme == 'iid':
        splitter=StratifiedKFold(n_splits,shuffle=True,random_state=seed)
        return list(splitter.split(raw,strata))
    g=groups_for(raw,scheme)
    splitter=StratifiedGroupKFold(n_splits,shuffle=True,random_state=seed)
    splits=list(splitter.split(raw,strata,g))
    for tr,va in splits:
        assert set(g[tr]).isdisjoint(g[va])
    return splits

class Features:
    def __init__(self, mode='baseline'):
        if mode not in {'baseline','native','prefix_duration','duration_schedule'}:
            raise ValueError(mode)
        self.mode=mode

    def fit_transform(self,raw):
        self.builder=TrainingOnlyFeatureBuilder()
        feat=self.builder.fit_transform(raw)
        feat=self.extend(feat,raw,fit=True)
        if self.mode=='native':
            self.cats=OHE_CAT_COLS+['flight_rare']
            self.levels={c:list(pd.Index(feat[c].astype(str).unique())) for c in self.cats}
            return self.native(feat)
        self.prep,mat=make_ohe_features(feat)
        if self.mode=='prefix_duration':
            self.prefixes=sorted(feat.flight_prefix.unique())
            mat=self.append_interactions(mat,feat)
        return mat

    def native(self,feat):
        z=feat[self.cats+NUM_COLS].copy()
        for c in self.cats:
            z[c]=pd.Categorical(z[c],categories=self.levels[c])
        return z

    def append_interactions(self,mat,feat):
        from scipy import sparse
        # Deterministic gating of observed/imputed duration; no target statistics.
        d=feat.duration_filled.to_numpy(float)/10.
        b=feat.flight_class.eq('Business').to_numpy(float)
        cols=[d*b*feat.flight_prefix.eq(p).to_numpy(float) for p in self.prefixes]
        return sparse.hstack([mat,sparse.csr_matrix(np.array(cols).T.astype('float32'))],format='csr')

    def extend(self,feat,raw,fit=False):
        if self.mode=='duration_schedule' and fit:
            observed=raw.dropna(subset=['duration'])
            hierarchy=[
                ['flight','source_city','destination_city','stops','departure_time','arrival_time'],
                ['flight','source_city','destination_city','stops','arrival_time'],
                ['flight','source_city','destination_city','stops','departure_time'],
            ]
            tables=[]
            for keys in hierarchy:
                stats=observed.groupby(keys,dropna=False).duration.agg(['median','count'])
                tables.append((keys,stats.loc[stats['count']>=2,'median']))
            self.builder.duration_group_tables=tables+self.builder.duration_group_tables
            feat=self.builder.transform(raw)
        return feat

    def transform(self,raw):
        feat=self.extend(self.builder.transform(raw),raw,fit=False)
        if self.mode=='native': return self.native(feat)
        mat=self.prep.transform(feat).tocsr().astype('float32')
        if self.mode=='prefix_duration': mat=self.append_interactions(mat,feat)
        return mat

def make_model(mode='baseline',rounds=850,early=False):
    p={**BASE,'n_estimators':int(rounds)}
    if mode=='native': p.update(enable_categorical=True,max_cat_threshold=16,max_cat_to_onehot=1)
    if early: p.update(early_stopping_rounds=75,eval_metric='rmse')
    return XGBRegressor(**p)

def metrics(y,p):
    return dict(r2=float(r2_score(y,p)), rmse=float(np.sqrt(mean_squared_error(y,p))),
        mae=float(mean_absolute_error(y,p)))

def evaluate(mode='baseline',scheme='raw',seed=20261015,early=False,out=None,train_path=TRAIN):
    frame,raw,y=load_train(train_path)
    out=Path(out or ROOT/'results'/(f'{scheme}_{seed}_{mode}'+('_nestedstop' if early else '')))
    out.mkdir(parents=True,exist_ok=True)
    splits=splits_for(raw,y,scheme,seed)
    oof=np.full(len(y),np.nan); fold=np.full(len(y),-1)
    rows=[]
    for i,(tr,va) in enumerate(splits):
        tick=time.perf_counter()
        rounds=850; inner_details=[]
        if early:
            # Two fold-pure stopping fits; outer labels are never consulted.
            inner_splits=splits_for(raw.iloc[tr],y[tr],scheme if scheme!='iid' else 'raw',seed+1000+i,n_splits=5)
            for j in [0,1]:
                it,iv=inner_splits[j]
                prep_i=Features(mode); xi=prep_i.fit_transform(raw.iloc[tr[it]])
                xv=prep_i.transform(raw.iloc[tr[iv]])
                model_i=make_model(mode,rounds=1400,early=True)
                model_i.fit(xi,y[tr[it]]/SCALE,eval_set=[(xv,y[tr[iv]]/SCALE)],verbose=False)
                inner_details.append(int(model_i.best_iteration+1))
            rounds=max(100,int(np.median(inner_details)))
        prep=Features(mode); xt=prep.fit_transform(raw.iloc[tr]); xv=prep.transform(raw.iloc[va])
        model=make_model(mode,rounds)
        model.fit(xt,y[tr]/SCALE)
        pv=np.maximum(0.,np.asarray(model.predict(xv),float)*SCALE)
        pt=np.maximum(0.,np.asarray(model.predict(xt),float)*SCALE)
        oof[va]=pv; fold[va]=i
        record=dict(fold=i,rounds=rounds,inner_stopping_rounds=inner_details,
            fitted_parameters=json.dumps({**BASE,'n_estimators':rounds,
                **(dict(enable_categorical=True,max_cat_threshold=16,max_cat_to_onehot=1) if mode=='native' else {})},sort_keys=True),
            n_train=len(tr),n_validation=len(va),train_r2=float(r2_score(y[tr],pt)),
            seconds=time.perf_counter()-tick,**metrics(y[va],pv))
        rows.append(record)
        pd.DataFrame(rows).to_csv(out/'fold_metrics.csv',index=False)
        print(mode,scheme,seed,'fold',i,record,flush=True)
        if i==0:
            (out/'booster_config.json').write_text(model.get_booster().save_config())
            trees=[json.loads(x) for x in model.get_booster().get_dump(dump_format='json')]
            def tree_stats(t,depth=0):
                if 'leaf' in t: return (1,depth)
                stats=[tree_stats(c,depth+1) for c in t['children']]
                return sum(s[0] for s in stats),max(s[1] for s in stats)
            stats=[tree_stats(t) for t in trees]
            save_json(out/'tree_bound_audit.json',dict(trees=len(trees),max_leaves=max(s[0] for s in stats),max_depth=max(s[1] for s in stats)))
    assert np.isfinite(oof).all() and (fold>=0).all()
    pd.DataFrame(dict(row_position=np.arange(len(y)),id=frame.id,price=y,fold=fold,
        raw_group=groups_for(raw),semantic_group=groups_for(raw,'semantic'),prediction=oof)).to_csv(out/'oof.csv',index=False)
    summary=dict(mode=mode,scheme=scheme,seed=seed,early_stopping=early,n=len(y),
        **metrics(y,oof),mean_train_r2=float(np.mean([r['train_r2'] for r in rows])),
        mean_fold_r2=float(np.mean([r['r2'] for r in rows])),
        mean_train_validation_gap=float(np.mean([r['train_r2']-r['r2'] for r in rows])),
        fold_r2_std=float(np.std([r['r2'] for r in rows],ddof=1)),
        worst_fold_r2=float(min(r['r2'] for r in rows)),folds=rows,
        reference_parameters=make_model(mode).get_params(),feature_mode=mode,
        fitted_rounds_by_fold=[r['rounds'] for r in rows],
        round_policy=('median of two fold-pure inner 80/20 stopping fits, ceiling1400 patience75 floor100' if early else 'fixed850'),
        total_seconds=sum(r['seconds'] for r in rows))
    save_json(out/'summary.json',summary)
    print('SUMMARY',json.dumps(summary),flush=True)
    return summary

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--mode',default='baseline')
    ap.add_argument('--scheme',default='raw');ap.add_argument('--seed',type=int,default=20261015)
    ap.add_argument('--early',action='store_true');ap.add_argument('--out',type=Path)
    ap.add_argument('--train',type=Path,default=TRAIN)
    args=ap.parse_args()
    evaluate(args.mode,args.scheme,args.seed,args.early,args.out,args.train)

if __name__=='__main__': main()
