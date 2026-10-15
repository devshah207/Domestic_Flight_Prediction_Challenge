#!/usr/bin/env python3
"""Training-only follow-up: evaluate tail weights, preserving the 850-tree baseline."""
import json,zipfile,time
from pathlib import Path
import numpy as np,pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import r2_score
from run_phase6 import OUT,ROOT,groups,bins,build,mets,ci,SCALE
from run_phase4 import FoldFeatures

def weights(y,mode):
 hi=np.quantile(y,.95);ext=np.quantile(y,.99);w=np.ones(len(y),dtype=float)
 if mode=='tail_up':w[y>=hi]=1.35;w[y>=ext]=1.6
 elif mode=='tail_down':w[y>=hi]=.90;w[y>=ext]=.65
 else:raise ValueError(mode)
 return w/w.mean()

def main():
 d=pd.read_csv(ROOT/'train(1).csv');x=d.drop(columns=['id','price']);y=d.price.to_numpy(float);g=groups(x)
 p0=pd.read_csv(OUT/'training_only_oof.csv').baseline.to_numpy();summary=json.loads((OUT/'cv_results.json').read_text())[0]
 folds=list(StratifiedGroupKFold(n_splits=5,shuffle=True,random_state=2026).split(x,bins(y),g));modes=['tail_up','tail_down']
 oof={k:np.full(len(y),np.nan) for k in modes};frows=[];weights_summary=[]
 for f,(itr,iva) in enumerate(folds,1):
  print('TAIL EXPERIMENT FOLD',f,flush=True)
  fe=FoldFeatures('original','xgb');X=fe.fit_transform(x.iloc[itr]);V=fe.transform(x.iloc[iva]);by=y[itr]
  for mode in modes:
   t=time.monotonic();w=weights(by,mode);mdl=build();mdl.fit(X,by/SCALE,sample_weight=w)
   p=np.asarray(mdl.predict(V))*SCALE;oof[mode][iva]=p
   frows.append({'fold':f,'name':mode,**mets(y[iva],p),'seconds':round(time.monotonic()-t,1)})
   weights_summary.append({'fold':f,'name':mode,'min_weight':float(w.min()),'max_weight':float(w.max()),'mean_weight':float(w.mean())})
   print(' ',mode,round(frows[-1]['r2'],6),flush=True)
  pd.DataFrame(frows).to_csv(OUT/'weight_progress.csv',index=False)
 pd.DataFrame(frows).to_csv(OUT/'weight_folds.csv',index=False)
 pd.DataFrame(weights_summary).to_csv(OUT/'weight_detail.csv',index=False)
 pd.DataFrame({'id':d.id,'price':y,**oof}).to_csv(OUT/'weight_oof.csv',index=False)
 basefold=pd.read_csv(OUT/'fold_scores.csv').query('name == "baseline"').sort_values('fold').r2.to_numpy()
 top=y>=np.quantile(y,.99);busi=(x.flight_class=='Business').to_numpy()
 results=[]
 for mode,p in oof.items():
  assert np.isfinite(p).all()
  fold=np.array([r['r2'] for r in frows if r['name']==mode]);delta=fold-basefold
  m=mets(y,p);m.update({'name':mode,'delta_r2':float(m['r2']-summary['r2']),'fold_deltas':delta.tolist(),'positive_folds':int((delta>0).sum()),'bootstrap_ci':ci(y,p0,p,g),'top1_rmse':float(np.sqrt(np.mean((y[top]-p[top])**2))),'business_rmse':float(np.sqrt(np.mean((y[busi]-p[busi])**2)))})
  m['passed_gate']=bool(m['delta_r2']>=.0015 and m['positive_folds']>=4 and m['bootstrap_ci'][0]>0 and m['top1_rmse']<=summary['top1_rmse']*1.015 and m['business_rmse']<=summary['business_rmse']*1.01)
  results.append(m)
 (OUT/'weight_results.json').write_text(json.dumps(results,indent=2));passed=[c for c in results if c['passed_gate']]
 chosen=max(passed,key=lambda c:c['delta_r2']) if passed else None
 if chosen:
  print('Weighting candidate passed gate; refitting full train then test inference',flush=True)
  prep=FoldFeatures('original','xgb');X=prep.fit_transform(x);mod=build().fit(X,y/SCALE,sample_weight=weights(y,chosen['name']))
  test=pd.read_csv(ROOT/'test(1).csv');sub=pd.read_csv(ROOT/'sample_submission(1).csv')
  assert test.id.is_unique and sub.id.is_unique and len(test)==len(sub)
  p=np.asarray(mod.predict(prep.transform(test[x.columns])))*SCALE
  assert np.isfinite(p).all();idx=pd.Series(np.arange(len(test)),index=test.id).reindex(sub.id);assert idx.notna().all()
  out=sub[['id','price']].copy();out['price']=np.maximum(p[idx.to_numpy(int)],0);out.to_csv(OUT/'phase6_weight_selected.csv',index=False)
 else:print('No weighting variant passed gate. No Kaggle test file opened.',flush=True)
 rfile=OUT/'PHASE6_DECISION_REPORT.md'
 with rfile.open('a') as fp:
  fp.write('\n## Phase 6B — Original-price loss weighting, exploratory follow-up\n\nAfter the encoding variants failed, we separately explored two modest, training-only weighting policies with the unchanged 850-tree XGBoost architecture. The thresholds for the top 5% and top 1% are recomputed within each outer training fold; weights are normalized to mean one. The same CV folds and original feature builder are used. This is not fresh independent evaluation.\n\n| Objective | OOF R² | Δ vs baseline | Positive folds | Business RMSE | Top-1% RMSE |\n|---|---:|---:|---:|---:|---:|\n')
  for c in results:fp.write(f"| {c['name']} | {c['r2']:.6f} | {c['delta_r2']:+.6f} | {c['positive_folds']}/5 | {c['business_rmse']:,.0f} | {c['top1_rmse']:,.0f} |\n")
  fp.write('\n**Decision:** '+('Advanced '+chosen['name'] if chosen else 'Neither objective weighting rule met the evidence gate. Original XGBoost 850 remains the benchmark.')+'\n')
 with zipfile.ZipFile(OUT/'phase6_analysis_package.zip','w',zipfile.ZIP_DEFLATED) as z:
  for file in sorted(OUT.iterdir()):
   if file.suffix in ('.json','.csv','.md'):z.write(file,file.name)
  z.write(Path(__file__),'weight_loss_study.py');z.write(Path(__file__).parent/'run_phase6.py','run_phase6.py')
 print('FINAL WEIGHT RESULTS',json.dumps(results,indent=2),flush=True)
if __name__=='__main__':main()
