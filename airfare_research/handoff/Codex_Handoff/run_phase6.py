#!/usr/bin/env python3
"""Phase 6: no test access until a predeclared train-only CV gate passes."""
import sys,time,json,zipfile
from pathlib import Path
import numpy as np,pandas as pd
from scipy.sparse import hstack,csr_matrix
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import r2_score,mean_squared_error,mean_absolute_error
from xgboost import XGBRegressor
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'phase4')]
from run_phase4 import FoldFeatures
from generate_phase3_boosting import CONFIGS
OUT=Path(__file__).parent/'results';OUT.mkdir(exist_ok=True)
SCALE=10000.;B={k:v for k,v in CONFIGS['xgb_850_d7_regularized'].items() if k not in ('family','target_transform')}
assert B['n_estimators']==850 and B['max_depth']==7
KEYS={'prefix_class':(['prefix','flight_class'],25.),'route_class':(['source_city','destination_city','flight_class'],40.),'prefix_route_class':(['prefix','source_city','destination_city','flight_class'],85.),'flight_class':(['flight','flight_class'],65.)}
SETS={'baseline':[],'coarse_history':['prefix_class','route_class'],'fine_history':list(KEYS)}
def groups(x):return pd.util.hash_pandas_object(x.astype(str).fillna('__NA__'),index=False).to_numpy()
def bins(y):return pd.qcut(pd.Series(y).rank(method='first'),q=10,labels=False).to_numpy()
def frame(x):
 x=x.copy().reset_index(drop=True);x['prefix']=x.flight.astype(str).str.split().str[0]
 for c in ['flight_class','source_city','destination_city','flight']:x[c]=x[c].fillna('__NA__').astype(str)
 return x
class Enc:
 def __init__(self,which):self.which=which
 def fit(self,x,y):
  x=frame(x);y=np.asarray(y)/SCALE;self.mu=y.mean()
  t=pd.DataFrame({'cls':x.flight_class,'y':y}).groupby('cls').y.agg(['sum','count'])
  self.classmu=((t['sum']+60*self.mu)/(t['count']+60)).to_dict();self.maps={}
  for name in self.which:
   keys,a=KEYS[name];t=x[keys].copy();t['y']=y
   self.maps[name]=(keys,a,t.groupby(keys,dropna=False).y.agg(['sum','count']))
  return self
 def transform(self,x):
  x=frame(x);prior=x.flight_class.map(self.classmu).fillna(self.mu).to_numpy();data={}
  for name in self.which:
   keys,a,m=self.maps[name]; ix=pd.MultiIndex.from_frame(x[keys]);s=m['sum'].reindex(ix).fillna(0).to_numpy();n=m['count'].reindex(ix).fillna(0).to_numpy()
   data['mean_'+name]=(s+a*prior)/(n+a);data['count_'+name]=np.log1p(n)
  return pd.DataFrame(data,index=x.index).to_numpy(dtype=np.float32)
def cv_enc(x,y,v,keys):
 if not keys:return None,None
 m=2*len(keys);o=np.full((len(x),m),np.nan,dtype=np.float32)
 cv=StratifiedGroupKFold(n_splits=4,shuffle=True,random_state=843)
 for tr,va in cv.split(x,bins(y),groups(x)):
  enc=Enc(keys).fit(x.iloc[tr],y[tr]);o[va]=enc.transform(x.iloc[va])
 assert np.isfinite(o).all()
 return o,Enc(keys).fit(x,y).transform(v)
def build():return XGBRegressor(**B,objective='reg:squarederror',tree_method='hist',random_state=42,n_jobs=5,verbosity=0)
def comb(x,e):return x if e is None else hstack([x,csr_matrix(e)],format='csr')
def mets(y,p):return {'r2':float(r2_score(y,p)),'rmse':float(np.sqrt(mean_squared_error(y,p))),'mae':float(mean_absolute_error(y,p))}
def ci(y,p0,p1,g):
 v=pd.DataFrame({'g':g,'z':(y-p0)**2-(y-p1)**2}).groupby('g',sort=False).z.sum().to_numpy()
 r=np.random.default_rng(26);denom=np.sum((y-y.mean())**2)
 a=np.array([v[r.integers(0,len(v),len(v))].sum()/denom for _ in range(800)])
 return np.quantile(a,[.025,.975]).tolist()
def main():
 t=time.time();d=pd.read_csv(ROOT/'train(1).csv');x=d.drop(columns=['id','price']);y=d.price.to_numpy(float);g=groups(x)
 high=y>=np.quantile(y,.99);prefix=d.flight.str.split().str[0]
 eda={'n':len(d),'feature_duplicates':int(x.duplicated().sum()),'99th_price_percentile':float(np.quantile(y,.99)),'top1_class_counts':d.loc[high,'flight_class'].value_counts().to_dict(),'top1_prefix_counts':prefix[high].value_counts().to_dict(),'missing':d.isna().sum().to_dict(),'scope':'training file only'}
 (OUT/'TRAIN_ONLY_EDA.json').write_text(json.dumps(eda,indent=2))
 folds=list(StratifiedGroupKFold(n_splits=5,shuffle=True,random_state=2026).split(x,bins(y),g));p={k:np.full(len(y),np.nan) for k in SETS};foldrows=[];trainrows=[]
 for f,(it,iv) in enumerate(folds,1):
  assert set(g[it]).isdisjoint(g[iv]);print('FOLD',f,'size',len(it),len(iv),flush=True)
  a=x.iloc[it].reset_index(drop=True);v=x.iloc[iv].reset_index(drop=True);fe=FoldFeatures('original','xgb');A=fe.fit_transform(a);V=fe.transform(v)
  for name,keys in SETS.items():
   z=time.time();e,ee=cv_enc(a,y[it],v,keys);u=comb(A,e);w=comb(V,ee)
   model=build().fit(u,y[it]/SCALE);pred=model.predict(w)*SCALE;trainpred=model.predict(u)*SCALE;p[name][iv]=pred
   foldrows.append({'fold':f,'name':name,**mets(y[iv],pred),'seconds':round(time.time()-z,1)})
   trainrows.append({'fold':f,'name':name,**mets(y[it],trainpred)})
   print(' ',name,round(foldrows[-1]['r2'],6),'seconds',foldrows[-1]['seconds'],flush=True)
  pd.DataFrame(foldrows).to_csv(OUT/'cv_progress.csv',index=False)
 pd.DataFrame(foldrows).to_csv(OUT/'fold_scores.csv',index=False);pd.DataFrame(trainrows).to_csv(OUT/'train_scores.csv',index=False)
 pd.DataFrame({'id':d.id,'price':y,**p}).to_csv(OUT/'training_only_oof.csv',index=False)
 bu=p['baseline'];busi=d.flight_class.eq('Business').to_numpy();allmetrics=[]
 for name,z in p.items():
  assert np.isfinite(z).all();q=mets(y,z);fv=pd.DataFrame(foldrows).query('name == @name').sort_values('fold').r2.to_numpy()
  q.update({'name':name,'mean_fold_r2':float(fv.mean()),'fold_r2_std':float(fv.std(ddof=1)),
    'mean_train_r2':float(pd.DataFrame(trainrows).query('name == @name').r2.mean()),
    'business_rmse':float(np.sqrt(np.mean((y[busi]-z[busi])**2))),'top1_rmse':float(np.sqrt(np.mean((y[high]-z[high])**2))),'top1_sse_share':float(np.sum((y[high]-z[high])**2)/np.sum((y-z)**2))})
  if name!='baseline':
   old=pd.DataFrame(foldrows).query('name == "baseline"').sort_values('fold').r2.to_numpy(); delta=fv-old
   q.update({'delta_r2':float(q['r2']-mets(y,bu)['r2']),'positive_folds':int((delta>0).sum()),'fold_delta':delta.tolist(),'bootstrap_ci':ci(y,bu,z,g)})
  allmetrics.append(q)
 (OUT/'cv_results.json').write_text(json.dumps(allmetrics,indent=2))
 pd.DataFrame([{k:v for k,v in u.items() if not isinstance(v,list)} for u in allmetrics]).to_csv(OUT/'cv_summary.csv',index=False)
 base=allmetrics[0];passing=[]
 for q in allmetrics[1:]:
  ok=(q['delta_r2']>=.0015 and q['positive_folds']>=4 and q['bootstrap_ci'][0]>0 and q['top1_rmse']<=base['top1_rmse']*1.015 and q['business_rmse']<=base['business_rmse']*1.01)
  q['passed_gate']=bool(ok)
  if ok:passing.append(q)
 winner=max(passing,key=lambda q:q['delta_r2']) if passing else None
 status={'winner':winner['name'] if winner else None,'gate':'OOF R2 +0.0015, at least 4/5 positive folds, positive grouped bootstrap lower bound, no more than 1.5% top1 RMSE or 1% business RMSE deterioration','train_time_seconds':round(time.time()-t,1),'Kaggle_test_never_used_for_selection':True}
 (OUT/'selection.json').write_text(json.dumps(status,indent=2))
 print('SUMMARY',json.dumps(allmetrics,indent=2),'GATE',status,flush=True)
 if winner:
  print('Passed gate; fitting full training then opening test for prediction only',flush=True)
  fe=FoldFeatures('original','xgb');A=fe.fit_transform(x);ec,_=cv_enc(x,y,x.iloc[0:0],SETS[winner['name']]);model=build().fit(comb(A,ec),y/SCALE);enc=Enc(SETS[winner['name']]).fit(x,y)
  test=pd.read_csv(ROOT/'test(1).csv');sample=pd.read_csv(ROOT/'sample_submission(1).csv')
  assert test.id.is_unique and sample.id.is_unique and len(test)==len(sample) and set(x.columns).issubset(test.columns)
  pred=model.predict(comb(fe.transform(test[x.columns]),enc.transform(test[x.columns])))*SCALE
  assert np.isfinite(pred).all();ix=pd.Series(np.arange(len(test)),index=test.id).reindex(sample.id);assert ix.notna().all()
  output=sample[['id','price']].copy();output['price']=np.maximum(0,pred[ix.to_numpy(dtype=int)]);output.to_csv(OUT/'phase6_selected.csv',index=False)
  (OUT/'frozen_config.json').write_text(json.dumps({'model':B,'target_history_keys':{k:KEYS[k] for k in SETS[winner['name']]}},indent=2))
 else:print('No candidate passed evidence gate. Test file not opened; no submission created.',flush=True)
 report=['# Phase 6 — Smoothed price-history test','','**Decision:** '+(f'Advance {winner["name"]}' if winner else 'No submission: neither new candidate met the predeclared evidence gate.'),'','## Experimental design','- All EDA, preprocessing, feature generation, and CV used the supplied training data only.','- Exact original XGBoost: 850 trees, depth 7, learning rate .03; no model hyperparameter changes.','- 5 outer folds grouped by identical feature rows and stratified by target deciles (seed 2026).','- Four inner folds produce cross-fitted target statistics for the training portion of each outer fold. Outer validation uses mappings fitted only on outer training labels.','- Strong smoothing toward class means for rare flight/route categories.','- Five-fold CV is training-only comparative evidence, **not a fresh untouched external holdout**. Prior work used much of this same training dataset.','','## Results','','| Candidate | OOF R² | Δ vs baseline | Positive folds | Business RMSE | Top 1% RMSE |','|---|---:|---:|---:|---:|---:|']
 for q in allmetrics:report.append(f"| {q['name']} | {q['r2']:.6f} | {q.get('delta_r2',0):+.6f} | {q.get('positive_folds','—')} | {q['business_rmse']:,.0f} | {q['top1_rmse']:,.0f} |")
 report+=['','## Advance criterion','Improvement >=0.0015 pooled OOF R², improvement in >=4/5 folds, positive approximate grouped bootstrap 95% lower confidence bound, top-1% RMSE <=1.015x baseline and business RMSE <=1.01x baseline.','', '## Limitations','A grouped bootstrap on pooled OOF errors is an approximate uncertainty assessment; its resampling does not capture the full uncertainty in retraining. Kaggle public/private rankings cannot be inferred from training-only CV.']
 (OUT/'PHASE6_DECISION_REPORT.md').write_text('\n'.join(report)+'\n')
 with zipfile.ZipFile(OUT/'phase6_analysis_package.zip','w',zipfile.ZIP_DEFLATED) as z:
  for item in sorted(OUT.iterdir()):
   if item.suffix in ('.csv','.json','.md'):z.write(item,item.name)
  z.write(__file__,'run_phase6.py')
 print('DONE',flush=True)
if __name__=='__main__':main()
