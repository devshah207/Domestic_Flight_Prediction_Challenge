#!/usr/bin/env python3
"""Phase 5. All modeling/selection on training data. Kaggle test only for inference if passed strict gates."""
import sys,time,json,zipfile
from pathlib import Path
import pandas as pd,numpy as np
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error
from xgboost import XGBRegressor
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'phase4'))
from run_phase4 import FoldFeatures
from generate_phase3_boosting import CONFIGS
O=Path(__file__).parent/'results';O.mkdir(exist_ok=True)
BASE={k:v for k,v in CONFIGS['xgb_850_d7_regularized'].items() if k not in ('family','n_estimators','target_transform')}
TAIL={'Business':{**BASE,'max_depth':7,'max_leaves':48,'min_child_weight':7,'reg_lambda':12.,'n_estimators':1050},'Economy':{**BASE,'max_depth':5,'max_leaves':32,'min_child_weight':8,'reg_lambda':16.,'n_estimators':650}}
STRATS=['global_850','split_same_850','split_tailored_earlystop'];SEED=42; SCALE=10000.
def grp(x):return pd.util.hash_pandas_object(x.astype(str).fillna('__NA__'),index=False).to_numpy()
def bins(y):return pd.qcut(pd.Series(y).rank(method='first'),q=10,labels=False).to_numpy()
def model_fit(X,y,cfg,n,stopX=None,stopY=None):
 p={k:v for k,v in cfg.items() if k!='n_estimators'}
 m=XGBRegressor(**p,n_estimators=int(n),objective='reg:squarederror',tree_method='hist',random_state=42,n_jobs=5,verbosity=0,early_stopping_rounds=70 if stopX is not None else None)
 extra={} if stopX is None else {'eval_set':[(stopX,stopY/SCALE)],'verbose':False}
 return m.fit(X,y/SCALE,**extra)
def train(strategy,raw,y):
 prep=FoldFeatures('original','xgb'); X=prep.fit_transform(raw)
 models={};counts={}
 if strategy=='global_850': models['ALL']=model_fit(X,y,BASE,850);counts['ALL']=850
 else:
  for cat in ('Business','Economy'):
   idx=np.flatnonzero(raw.flight_class.to_numpy()==cat)
   cfg=BASE if strategy=='split_same_850' else TAIL[cat];n=850 if strategy=='split_same_850' else cfg['n_estimators']
   if strategy=='split_tailored_earlystop':
    part=raw.iloc[idx].reset_index(drop=True)
    it,iv=next(iter(StratifiedGroupKFold(n_splits=5,shuffle=True,random_state=73).split(part,bins(y[idx]),grp(part))))
    pilot=model_fit(X[idx[it]],y[idx[it]],cfg,n,X[idx[iv]],y[idx[iv]]);n=int(pilot.best_iteration+1)
   models[cat]=model_fit(X[idx],y[idx],cfg,n);counts[cat]=n
 def predict(xx):
  xx=xx.reset_index(drop=True);A=prep.transform(xx)
  if strategy=='global_850':return np.asarray(models['ALL'].predict(A))*SCALE
  result=np.zeros(len(xx))
  for cat,m in models.items():
   i=np.flatnonzero(xx.flight_class.to_numpy()==cat)
   if len(i):result[i]=np.asarray(m.predict(A[i]))*SCALE
  return result
 return predict,counts

def metrics(y,p,x,name):
 e=p-y;z=e**2;d={'model':name,'r2':float(r2_score(y,p)),'rmse':float(np.sqrt(np.mean(z))),'mae':float(np.mean(abs(e))),'n':len(y)}
 for cat in ('Business','Economy'):
  mask=x.flight_class.to_numpy()==cat;d[cat+'_rmse']=float(np.sqrt(z[mask].mean()));d[cat+'_sse_share']=float(z[mask].sum()/z.sum())
 hi=y>=np.quantile(y,.99);d['top1_sse_share']=float(z[hi].sum()/z.sum());d['top1_mean_error']=float(e[hi].mean())
 return d

def paired_boot(y,p0,p1,g,N=750):
 z=pd.DataFrame({'g':g,'d':(y-p0)**2-(y-p1)**2}).groupby('g',sort=False)['d'].sum().to_numpy()
 rng=np.random.default_rng(2026);v=np.array([z[rng.integers(0,len(z),size=len(z))].sum() for _ in range(N)])/np.sum((y-y.mean())**2)
 return {'ci_low':float(np.quantile(v,.025)),'ci_high':float(np.quantile(v,.975)),'fraction_positive':float(np.mean(v>0))}
def save(o,name): (O/name).write_text(json.dumps(o,indent=2))

def main():
 print('Reading TRAIN only',flush=True);T=pd.read_csv(ROOT/'train(1).csv');x=T.drop(columns=['id','price']);y=T.price.to_numpy(float);g=grp(x)
 train,hold=next(iter(StratifiedGroupKFold(n_splits=5,shuffle=True,random_state=143).split(x,bins(y),g)))
 assert set(g[train]).isdisjoint(set(g[hold]));xd=x.iloc[train].reset_index(drop=True);yd=y[train];gd=grp(xd)
 save({'train_n':len(T),'dev_n':len(train),'untouched_holdout_n':len(hold),'baseline':{'trees':850,**BASE},'tailored':TAIL,'no_test_data_until_after_selection':True},'design.json')
 folds=list(StratifiedGroupKFold(n_splits=3,shuffle=True,random_state=144).split(xd,bins(yd),gd))
 records=[];iters=[];oo={s:np.full(len(yd),np.nan) for s in STRATS}
 for k,(tr,va) in enumerate(folds,1):
  print('DEVELOPMENT FOLD',k,flush=True)
  for name in STRATS:
   tm=time.time();f,rounds=train_model(name,xd.iloc[tr].reset_index(drop=True),yd[tr]);p=f(xd.iloc[va]);oo[name][va]=p
   score=metrics(yd[va],p,xd.iloc[va],name);score['fold']=k;score['seconds']=round(time.time()-tm,1);records.append(score)
   iters.append({'fold':k,'model':name,**rounds});print('  ',name,round(score['r2'],6),'rounds',rounds,'s',score['seconds'],flush=True)
  pd.DataFrame(records).to_csv(O/'cv_partial.csv',index=False)
 res=[metrics(yd,oo[s],xd,s) for s in STRATS]
 pd.DataFrame(records).to_csv(O/'cv_folds.csv',index=False);pd.DataFrame(res).to_csv(O/'cv_summary.csv',index=False);pd.DataFrame(iters).to_csv(O/'iterations.csv',index=False)
 ptab=pd.DataFrame({'id':T.iloc[train].id.to_numpy(),'actual':yd,**oo});ptab.to_csv(O/'dev_oof.csv',index=False)
 best=max(res[1:],key=lambda d:d['r2']);b=res[0];fdf=pd.DataFrame(records)
 gains=[float(fdf[(fdf.fold==i)&(fdf.model==best['model'])].r2.iloc[0]-fdf[(fdf.fold==i)&(fdf.model=='global_850')].r2.iloc[0]) for i in range(1,4)]
 ci=paired_boot(yd,oo['global_850'],oo[best['model']],gd)
 advance=bool(best['r2']-b['r2']>=.003 and sum(a>0 for a in gains)>=2 and ci['ci_low']>0)
 print('DEV summary',[(z['model'],z['r2']) for z in res],'gains',gains,'CI',ci,'ADVANCE',advance,flush=True)
 # Independently assess both the original benchmark and the preselected best candidate once.
 hs=[];hp={}
 for s in ('global_850',best['model']):
  print('HELD OUT:',s,flush=True);model,_=train_model(s,xd,yd);p=model(x.iloc[hold]);hp[s]=p;hs.append(metrics(y[hold],p,x.iloc[hold],s))
 pd.DataFrame(hs).to_csv(O/'holdout_scores.csv',index=False)
 hci=paired_boot(y[hold],hp['global_850'],hp[best['model']],g[hold]);delta=hs[1]['r2']-hs[0]['r2']
 selected=bool(advance and delta>=.002 and hci['ci_low']>0)
 report={'development':res,'best_dev':best['model'],'dev_delta':best['r2']-b['r2'],'dev_fold_deltas':gains,'dev_ci':ci,'pass_development':advance,'holdout':hs,'holdout_delta':delta,'holdout_ci':hci,'pass_all_gates':selected,'gate':'development delta>=0.003, positive on >=2/3 folds, 95% paired grouped bootstrap CI low >0; independent holdout delta>=0.002 and 95% bootstrap CI low >0','model_submission':('phase5_selected.csv' if selected else None)}
 save(report,'validation_findings.json');print('FINAL DECISION',report['model_submission'] or 'NO NEW MODEL DEMONSTRATED',flush=True)
 if selected:
  print('Training selected model on ALL TRAIN',flush=True);handler,rounds=train_model(best['model'],x,y);save({'strategy':best['model'],'rounds':rounds},'frozen_final_config.json')
  print('Training done. Opening TEST ONLY for inference.',flush=True)
  te=pd.read_csv(ROOT/'test(1).csv');sub=pd.read_csv(ROOT/'sample_submission(1).csv')
  assert set(x.columns).issubset(te.columns) and len(sub)==len(te) and sub.id.is_unique and te.id.is_unique
  pred=handler(te[x.columns]);assert np.isfinite(pred).all() and len(pred)==len(sub)
  idx=pd.Series(np.arange(len(te)),index=te.id).reindex(sub.id).to_numpy();assert np.isfinite(idx).all()
  sub['price']=np.maximum(0,pred[idx.astype(int)]);sub[['id','price']].to_csv(O/'phase5_selected.csv',index=False)
 with zipfile.ZipFile(O/'phase5_analysis.zip','w',zipfile.ZIP_DEFLATED) as z:
  for fn in O.glob('*'):
   if fn.suffix in ['.csv','.json']:z.write(fn,fn.name)
  z.write(__file__,'run_phase5.py')

def train_model(s,x,y):return train(s,x,y)
if __name__=='__main__':main()
