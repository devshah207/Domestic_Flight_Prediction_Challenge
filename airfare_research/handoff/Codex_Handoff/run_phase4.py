#!/usr/bin/env python3
"""Phase 4 leakage-aware Kaggle regression pipeline.

Reads TRAIN only for EDA, CV/model selection, full-data refit.
TEST is read exactly once after ALL final model definitions/weights have been frozen,
and is used only for final prediction and submission-schema checks.

Run from location containing generate_submissions.py:
python phase4/run_phase4.py --train 'train(1).csv' --test 'test(1).csv' \
    --sample 'sample_submission(1).csv' --out phase4/results
"""
from __future__ import annotations
import argparse, json, sys, time, warnings
from pathlib import Path
from collections import Counter
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, GroupKFold
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor, early_stopping, log_evaluation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from generate_submissions import (TrainingOnlyFeatureBuilder, OHE_CAT_COLS, NUM_COLS,
    LGB_CAT_COLS, TARGET, SCALE, SEED)

# Research-controlled small candidate set. All bounds finite, no NN.
XGB_BASE = dict(max_depth=7, max_leaves=48, learning_rate=.030,
    min_child_weight=12, subsample=.80, colsample_bytree=.85,
    reg_lambda=15., reg_alpha=.60, gamma=.08, max_bin=192)
CANDIDATES = {
    'xgb_reference_d7': dict(family='xgb', features='original', **XGB_BASE),
    'xgb_shallower_d6': dict(family='xgb', features='original', **{**XGB_BASE,'max_depth':6,'max_leaves':40}),
    'xgb_stronger_reg_d7': dict(family='xgb', features='original', **{**XGB_BASE,'min_child_weight':18,'reg_lambda':22.,'gamma':.12}),
    'xgb_engineered_d7': dict(family='xgb', features='engineered', **XGB_BASE),
    'xgb_engineered_d6': dict(family='xgb', features='engineered', **{**XGB_BASE,'max_depth':6,'max_leaves':40}),
    'lgb_engineered_d5': dict(family='lgb', features='engineered',max_depth=5,
        num_leaves=15, learning_rate=.045, min_child_samples=65, min_split_gain=.04,
        reg_lambda=20., reg_alpha=.4, subsample=.80, subsample_freq=1,
        colsample_bytree=.80, max_bin=127),
}
MAX_ROUNDS=1450
PATIENCE=75
N_THREADS=4

# Explicitly chosen before CV. No target-leakage encodings.
ENG_CAT = ['class_stops','route_airline','flight_route_rare','class_depart','route_depart']
ENG_NUM = ['business_duration', 'business_days_left', 'economy_duration',
           'economy_days_left', 'duration_per_leg', 'days_left_log1p',
           'days_left_sq', 'duration_sq','duration_days_interaction']

def extend_engineered(feat:pd.DataFrame, training_frequent_routes=None):
    z=feat.copy()
    cl = z['flight_class'].eq('Business').astype(float)
    duration = z['duration_filled'].astype(float)
    days=z['days_left'].astype(float)
    z['business_duration']=duration * cl
    z['business_days_left']=days * cl
    z['economy_duration']=duration * (1.0-cl)
    z['economy_days_left']=days * (1.0-cl)
    z['duration_per_leg']=duration / np.maximum(1., 1.+z['stops_numeric'].clip(lower=0))
    z['days_left_log1p']=np.log1p(days.clip(lower=0))
    z['days_left_sq']=(days / 30)**2
    z['duration_sq']=(duration / 10)**2
    z['duration_days_interaction']=(duration/10)*(days/30)
    z['class_stops']=z['flight_class']+' / '+z['stops']
    z['route_airline']=z['route']+' / '+z['airline_fixed']
    z['class_depart']=z['flight_class']+' / '+z['departure_time']
    z['route_depart']=z['route']+' / '+z['departure_time']
    z['flight_route_rare']=z['flight']+' / '+z['route']
    if training_frequent_routes is not None:
        z['flight_route_rare']=z['flight_route_rare'].where(z['flight_route_rare'].isin(training_frequent_routes),'__RARE_OR_NEW__')
    return z

class FoldFeatures:
    def __init__(self, mode, family):
        self.mode, self.family = mode, family
    def fit_transform(self, x):
        self.builder=TrainingOnlyFeatureBuilder()
        tr=self.builder.fit_transform(x)
        if self.mode=='engineered':
            rr=tr['flight']+' / '+tr['route']
            freq=rr.value_counts()
            self.common_routes=set(freq[freq>=10].index)
            tr=extend_engineered(tr,self.common_routes)
        cats=OHE_CAT_COLS + (ENG_CAT if self.mode=='engineered' else [])
        nums=NUM_COLS + (ENG_NUM if self.mode=='engineered' else [])
        if self.family=='xgb':
            # Group high-cardinality categorical identifiers separately.
            self.cols=ColumnTransformer([
                ('cat',OneHotEncoder(handle_unknown='ignore',sparse_output=True),cats),
                ('flight',OneHotEncoder(handle_unknown='infrequent_if_exist',min_frequency=12,sparse_output=True),['flight']),
                ('numeric','passthrough',nums)], sparse_threshold=1.0)
            return self.cols.fit_transform(tr).tocsr().astype(np.float32)
        self.cat_cols=LGB_CAT_COLS+(ENG_CAT if self.mode=='engineered' else [])
        self.cat_levels={c:pd.Index(tr[c].astype(str).unique()) for c in self.cat_cols}
        return self._make_lgb(tr,nums)
    def _make_lgb(self,tr,nums):
        z=tr[self.cat_cols+nums].copy()
        for c in self.cat_cols:
            z[c]=pd.Categorical(z[c].astype(str), categories=self.cat_levels[c])
        return z
    def transform(self, x):
        feat=self.builder.transform(x)
        if self.mode=='engineered':
            feat=extend_engineered(feat,self.common_routes)
        if self.family=='xgb':
            return self.cols.transform(feat).tocsr().astype(np.float32)
        nums=NUM_COLS+(ENG_NUM if self.mode=='engineered' else [])
        return self._make_lgb(feat,nums)


def make_estimator(cfg, n_estimators, early_stop=False):
    p={k:v for k,v in cfg.items() if k not in ('family','features')}
    if cfg['family']=='xgb':
        return XGBRegressor(**p, n_estimators=n_estimators,
             early_stopping_rounds=PATIENCE if early_stop else None,
             objective='reg:squarederror',tree_method='hist',random_state=SEED,
             n_jobs=N_THREADS,verbosity=0)
    return LGBMRegressor(**p,n_estimators=n_estimators,objective='regression',
        random_state=SEED,n_jobs=N_THREADS,verbosity=-1)


def fold_fit(name,cfg,raw_train,y,tridx,vaidx):
    feat=FoldFeatures(cfg['features'],cfg['family'])
    X_tr=feat.fit_transform(raw_train.iloc[tridx]); X_va=feat.transform(raw_train.iloc[vaidx]);
    train_y=y[tridx] / SCALE
    val_y=y[vaidx] / SCALE
    model=make_estimator(cfg,MAX_ROUNDS,early_stop=(cfg['family']=='xgb'))
    if cfg['family']=='xgb':
        model.fit(X_tr,train_y,eval_set=[(X_va,val_y)],verbose=False)
        iters=model.best_iteration+1
    else:
        model.fit(X_tr,train_y,eval_set=[(X_va,val_y)],
             categorical_feature=feat.cat_cols,callbacks=[early_stopping(PATIENCE,verbose=False),log_evaluation(0)])
        iters=model.best_iteration_
    pred=np.asarray(model.predict(X_va),dtype=float)*SCALE
    pred_train=np.asarray(model.predict(X_tr),dtype=float)*SCALE
    return pred,pred_train,int(iters)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--train',required=True);ap.add_argument('--test',required=True)
    ap.add_argument('--sample',required=True);ap.add_argument('--out',required=True)
    args=ap.parse_args()
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    print('PHASE 4: reading training data ONLY',flush=True)
    t0=time.time()
    train=pd.read_csv(args.train)
    assert 'id' in train and TARGET in train
    raw=train.drop(columns=['id',TARGET]); y=train[TARGET].to_numpy(dtype=float)
    assert np.isfinite(y).all() and (y>=0).all()
    # Targets never used to construct feature-level groups; duplicate records cannot cross folds.
    groups=pd.util.hash_pandas_object(raw.astype(str).fillna('__NA__'),index=False).to_numpy()
    quant=pd.qcut(pd.Series(y).rank(method='first'),q=10,labels=False)
    splitter=StratifiedGroupKFold(n_splits=5,shuffle=True,random_state=SEED)
    splits=list(splitter.split(raw,quant,groups))
    # Write transparent EDA and fold checks, TRAIN ONLY.
    eda={
      'n_rows':len(train),'predictor_columns':raw.columns.tolist(),
      'missing_counts':raw.isna().sum().astype(int).to_dict(),
      'duplicate_feature_rows':int(raw.duplicated().sum()),
      'n_unique_flights':int(raw.flight.nunique()),
      'target_description':{k:float(v) for k,v in train[TARGET].describe(percentiles=[.05,.5,.95,.99]).items()},
      'flight_class_summary':train.groupby('flight_class')[TARGET].agg(['count','mean','median']).round(2).to_dict('index'),
      'target_days_left_pearson':float(train[['days_left','price']].corr().iloc[0,1]),
      'target_duration_pearson_observed':float(train[['duration','price']].corr().iloc[0,1]),
      'fold_summary':[],
    }
    for fi,(tr,va) in enumerate(splits,1):
        assert set(groups[tr]).isdisjoint(set(groups[va]))
        eda['fold_summary'].append({'fold':fi,'n_train':len(tr),'n_validation':len(va),
             'validation_business_pct':round(float(raw.iloc[va].flight_class.eq('Business').mean()*100),2),
             'validation_price_mean':round(float(y[va].mean()),2),
             'validation_price_95pct':round(float(np.quantile(y[va],.95)),2)})
    (out/'training_only_eda.json').write_text(json.dumps(eda,indent=2))
    folds=pd.DataFrame({'row_position':np.arange(len(raw)), 'id':train['id'],
        'cv_fold':np.full(len(raw),-1,dtype=int)})
    for i,(_,va) in enumerate(splits,1):folds.loc[va,'cv_fold']=i
    assert (folds.cv_fold>0).all();folds.to_csv(out/'fold_assignments.csv',index=False)
    print('CV scheme: 5-fold stratified grouped by exact original feature records (zero duplicate overlap)',flush=True)
    cv_records=[];oo_records=[];score_records=[];oof_preds={}
    for name,cfg in CANDIDATES.items():
        print(f'BEGIN {name} / {cfg["features"]}: max_rounds={MAX_ROUNDS}, patience={PATIENCE}',flush=True)
        start=time.time(); preds=np.full(len(y),np.nan); rounds=[];train_scores=[];fold_scores=[]
        for i,(tr,va) in enumerate(splits,1):
            tick=time.time()
            pv,pt,bst=fold_fit(name,cfg,raw,y,tr,va)
            preds[va]=pv
            s=float(r2_score(y[va],pv)); st=float(r2_score(y[tr],pt))
            fold_scores.append(s);train_scores.append(st);rounds.append(bst)
            cv_records.append({'model':name,'fold':i,'n_train':len(tr),'n_validation':len(va),
                'best_rounds':bst,'train_r2':st,'validation_r2':s,
                'validation_rmse':float(np.sqrt(mean_squared_error(y[va],pv))),
                'validation_mae':float(mean_absolute_error(y[va],pv)),
                'seconds':round(time.time()-tick,2)})
            print(f'  fold {i}/5: rounds={bst}, val R2={s:.6f}, train R2={st:.6f}, {time.time()-tick:.1f}s',flush=True)
        assert np.isfinite(preds).all()
        oof_preds[name]=preds
        score_records.append({'model':name,'family':cfg['family'],'feature_mode':cfg['features'],
            'oof_r2':float(r2_score(y,preds)),'mean_fold_r2':float(np.mean(fold_scores)),
            'sd_fold_r2':float(np.std(fold_scores,ddof=1)),
            'pooled_oof_rmse':float(np.sqrt(mean_squared_error(y,preds))),
            'pooled_oof_mae':float(mean_absolute_error(y,preds)),
            'mean_train_r2':float(np.mean(train_scores)),
            'mean_train_minus_val_r2':float(np.mean(np.array(train_scores)-np.array(fold_scores))),
            'median_best_rounds':int(np.median(rounds)), 'rounds_min':int(min(rounds)),
            'rounds_max':int(max(rounds)), 'elapsed_seconds':round(time.time()-start,1)})
        pd.DataFrame(cv_records).to_csv(out/'fold_metrics.csv',index=False)
        pd.DataFrame(score_records).to_csv(out/'cv_comparison_partial.csv',index=False)
        print(f'END {name}: OOF R2={score_records[-1]["oof_r2"]:.6f}, median rounds={int(np.median(rounds))}',flush=True)
    cv=pd.DataFrame(score_records).sort_values('oof_r2',ascending=False).reset_index(drop=True)
    cv.to_csv(out/'cv_comparison.csv',index=False)
    p_oof=pd.DataFrame({'id':train['id'],'price':y,'flight_class':raw.flight_class,'fold':folds.cv_fold})
    for k,pred in oof_preds.items():p_oof[k]=pred
    p_oof.to_csv(out/'oof_predictions.csv',index=False)
    # Candidate selection entirely training-only. If practically tied, choose simpler depth.
    best=cv.iloc[0].to_dict()
    tolerance=0.0003
    eligible=cv[cv.oof_r2 >= best['oof_r2']-tolerance]
    complexity={n:CANDIDATES[n]['max_depth'] for n in CANDIDATES}
    champion=min(eligible['model'],key=lambda n:(complexity[n],-float(cv.loc[cv.model.eq(n),'oof_r2'].iloc[0])))
    # Also keep a distinct strong alternate; never selection on Kaggle public test feedback.
    alternatives=[n for n in cv.model if n!=champion]
    alternate=alternatives[0]
    # Preserve baseline as independent safety/control; maximum three frozen final candidates.
    selected=list(dict.fromkeys([champion,alternate,'xgb_reference_d7']))
    # Blending accepted only if strictly improves pooled OOF R2 meaningfully, no optimization weights.
    blend_name=None;blend_score=None
    if champion!=alternate:
        blended=.5*oof_preds[champion]+.5*oof_preds[alternate]
        blend_score=float(r2_score(y,blended))
        if blend_score >= max(cv.oof_r2)+.0002:
            blend_name='blend_50_50_top2'
    # File records selection BEFORE opening test. Refit entire train for frozen fold-median rounds.
    metadata={'selection_criterion':'highest grouped duplicate-safe 5-fold OOF R2, tie within 0.0003 selects shallower model',
        'validation':'StratifiedGroupKFold on train only, no original predictor-duplicate overlap',
        'early_stopping_patience':PATIENCE,'upper_training_rounds':MAX_ROUNDS,
        'final_round_rule':'median fold best rounds, minimum 100, maximum training ceiling',
        'champion':champion,'alternate':alternate,'selected_candidates':selected,
        'blend_50_50_oof_r2':blend_score,'blend_included':bool(blend_name),
        'testing_leakage':'No test EDA, training, model selection, calibration, or preprocessing fitting.',
        'candidate_configurations':CANDIDATES}
    (out/'frozen_selection_before_test.json').write_text(json.dumps(metadata,indent=2))
    print('SELECTION FROZEN BEFORE TEST:',selected,'blend:',blend_name,flush=True)
    trained={}; feature_builder={}; final_info=[]
    for name in selected:
        cfg=CANDIDATES[name]
        best_i=int(cv.loc[cv.model.eq(name),'median_best_rounds'].iloc[0])
        nrounds=int(np.clip(best_i,100,MAX_ROUNDS))
        feat=FoldFeatures(cfg['features'],cfg['family'])
        X=feat.fit_transform(raw)
        reg=make_estimator(cfg,nrounds,early_stop=False)
        print(f'REFIT full TRAIN {name}: n_rounds={nrounds}',flush=True)
        if cfg['family']=='lgb':reg.fit(X,y/SCALE,categorical_feature=feat.cat_cols)
        else:reg.fit(X,y/SCALE,verbose=False)
        trained[name]=reg;feature_builder[name]=feat
        final_info.append({'model':name,'trees':nrounds,'depth':cfg['max_depth'],
           'features':cfg['features'],'cv_oof_r2':float(cv.loc[cv.model.eq(name),'oof_r2'].iloc[0]),
           'source':'full-training refit with median early-stopping rounds from 5 folds'})
    (out/'final_fit_configurations.json').write_text(json.dumps(final_info,indent=2))
    # *** This is the FIRST test read: no data-derived decision can follow. ***
    print('ALL FITS COMPLETED. Opening test.csv only for frozen-model inference.',flush=True)
    test=pd.read_csv(args.test); sample=pd.read_csv(args.sample)
    must=['id']+list(raw.columns)
    if not set(must).issubset(test.columns):raise ValueError('missing expected test columns')
    if not {'id','price'}.issubset(sample.columns):raise ValueError('wrong sample format')
    if not test.id.is_unique or not sample.id.is_unique or len(test)!=len(sample):raise ValueError('ID mismatch')
    positions=pd.Series(np.arange(len(test)),index=test.id).reindex(sample.id)
    if positions.isna().any():raise ValueError('missing test ids in sample')
    order=positions.to_numpy(dtype=int)
    test_features=test[raw.columns]  # schema only; never analyze test distributions
    final_preds={}
    for name in selected:
        xx=feature_builder[name].transform(test_features)
        pr=np.asarray(trained[name].predict(xx)).ravel()*SCALE
        assert len(pr)==len(test) and np.isfinite(pr).all()
        pr=np.maximum(pr,0)
        final_preds[name]=pr
        result=sample[['id','price']].copy();result['price']=pr[order]
        result.to_csv(out/(name+'.csv'),index=False)
        print('Submission:',name+'.csv',len(result),'rows',flush=True)
    if blend_name:
        pred=.5*final_preds[champion]+.5*final_preds[alternate]
        result=sample[['id','price']].copy();result['price']=pred[order]
        result.to_csv(out/(blend_name+'.csv'),index=False)
        print('Submission:',blend_name+'.csv',len(result),'rows',flush=True)
    # Mean error by cabin and price-decile, strictly OOF / training labels only.
    diagnostic=[]
    for name in selected:
        for cl in sorted(raw.flight_class.unique()):
            mask=raw.flight_class.eq(cl).to_numpy()
            diagnostic.append({'model':name,'segment':cl,'count':int(mask.sum()),
               'mae':float(mean_absolute_error(y[mask],oof_preds[name][mask])),
               'mean_error_pred_minus_true':float(np.mean(oof_preds[name][mask]-y[mask]))})
        p99=np.quantile(y,.99)
        mask=y>=p99
        diagnostic.append({'model':name,'segment':'highest_1pct_prices','count':int(mask.sum()),
           'mae':float(mean_absolute_error(y[mask],oof_preds[name][mask])),
           'mean_error_pred_minus_true':float(np.mean(oof_preds[name][mask]-y[mask]))})
    pd.DataFrame(diagnostic).to_csv(out/'oof_segments.csv',index=False)
    report=[]
    report.append('# Phase 4 – leakage-aware training-only validation and submissions')
    report.append(f'\nTraining rows: **{len(y):,}**. Duplicate feature rows: **{eda["duplicate_feature_rows"]:,}** (excluded from validation overlap). Official competition metric: **R²** on original price scale.')
    report.append('\n## Dataset EDA (training only)')
    report.append(f'- Ticket price: median {np.median(y):,.0f}, mean {np.mean(y):,.0f}, 99th percentile {np.quantile(y,.99):,.0f}.')
    report.append(f'- Airline missing {int(raw.airline.isna().sum()):,} rows; duration missing {int(raw.duration.isna().sum()):,}.')
    report.append(f'- Airline price class difference: economy median {eda["flight_class_summary"]["Economy"]["median"]:,.0f}; business median {eda["flight_class_summary"]["Business"]["median"]:,.0f}.')
    report.append(f'- Observed duration vs ticket price Pearson correlation: {eda["target_duration_pearson_observed"]:.3f}; booking lead days vs ticket price correlation: {eda["target_days_left_pearson"]:.3f}.')
    report.append('- Learned duration imputations, rare-flight thresholds, categorical encodings and any feature frequency mappings are refitted within each CV training fold.')
    report.append('\n## Validation')
    report.append('Five folds: stratified by deciles of target and grouped by exact original predictor rows; each group appears in only one fold. No test features read before selection.')
    report.append('Early stopping on the validation fold: patience 75 boosting rounds, upper ceiling 1450. For final full-data refit, median best fold iterations (bounded 100 to 1450).')
    report.append('\n## Measured results (not Kaggle public scores)')
    report.append('\n'+cv[['model','oof_r2','mean_fold_r2','sd_fold_r2','pooled_oof_rmse','mean_train_r2','mean_train_minus_val_r2','median_best_rounds','rounds_min','rounds_max']].round(5).to_markdown(index=False))
    report.append('\n## Frozen submissions')
    report.append('\n'+pd.DataFrame(final_info).to_markdown(index=False))
    if blend_name:report.append(f'\nOptional predefined 50/50 blend achieved OOF R² {blend_score:.6f} and exceeded the leading individual model by at least 0.0002. No fitted blend weights.')
    else:report.append('\nNo blend met the predefined +0.0002 OOF R² hurdle; not issuing an unvalidated blend.')
    report.append('\n## Interpretation and limitations')
    report.append('- Five-fold validation informs overfitting risk but cannot guarantee hidden-test performance. Selection among multiple models still has some optimism.')
    report.append('- The folds block *identical rows*, not all shared flight IDs; unseen-flight generalization could be harder. Public/private differences remain possible.')
    report.append('- Engineered features are deterministic domain interactions and transformations; no target encoding or test-feature distributions are used.')
    report.append('- Training-vs-validation R² gaps and OOF errors should be interpreted jointly; leakage-safe transforms are refitted separately for each fold.')
    report.append('- Fixed seed, exact parameters, fold assignments, OOF predictions and training-only EDA are included for audit and reproducibility.')
    report.append(f'\nRuntime: {time.time()-t0:.1f} seconds.')
    (out/'PHASE4_REPORT.md').write_text('\n'.join(report)+'\n')
    print('PHASE 4 SUCCESS. Total sec',round(time.time()-t0,1),flush=True)

if __name__=='__main__':main()
