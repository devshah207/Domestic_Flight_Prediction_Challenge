"""Training-only diagnostics for the registered Business specialist."""
from pathlib import Path
import argparse,json,sys
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
import specialist as s
import diagnostics as h


def evaluate(seed,train,out):
    d=ROOT/seed
    if not (d/'summary.json').exists():return None
    summary=json.loads((d/'summary.json').read_text())
    reference=pd.read_csv(d/'global_xgb_oof.csv');y=reference.price.to_numpy();fold=reference.fold.to_numpy()
    assert np.array_equal(reference.id,train.id) and np.array_equal(y,train.price)
    assert len(reference)==15589 and reference.row_position.is_unique and reference.groupby('group').fold.nunique().max()==1
    predictions={}
    for name in ['global_xgb','business_augmented_xgb']:
        frame=pd.read_csv(d/(name+'_oof.csv'));pd.testing.assert_frame_equal(frame.drop(columns='prediction'),reference.drop(columns='prediction'))
        predictions[name]=frame.prediction.to_numpy()
    economy=train.flight_class.to_numpy()=='Economy'
    assert np.array_equal(predictions['global_xgb'][economy],predictions['business_augmented_xgb'][economy])
    pairs=[('business_augmented_xgb','global_xgb')]
    if seed=='primary':
        controls={k:pd.read_csv(s.common.FRESH/'results'/k/'oof.csv') for k in ['xgb','lgb','cat_onehot64']}
        for f in controls.values():pd.testing.assert_frame_equal(f.drop(columns='prediction'),reference.drop(columns='prediction'))
        np.testing.assert_allclose(predictions['global_xgb'],controls['xgb'].prediction,rtol=0,atol=1e-6)
        shared=controls['lgb'].prediction.to_numpy()+controls['cat_onehot64'].prediction.to_numpy()
        predictions['frozen_equal_blend64']=(predictions['global_xgb']+shared)/3
        predictions['business_equal_blend64']=(predictions['business_augmented_xgb']+shared)/3
        pairs.append(('business_equal_blend64','frozen_equal_blend64'))
    supports=np.zeros(len(y),int)
    for f in range(5):
        mask=fold==f;supports[mask]=train.loc[mask,'flight'].map(train.loc[~mask,'flight'].value_counts()).fillna(0).to_numpy(int)
    groups={'cabin':train.flight_class.astype(str),'prefix_cabin':train.flight.str.split().str[0]+'|'+train.flight_class,
        'target_band':pd.cut(y,[-np.inf,np.quantile(y,.9),np.quantile(y,.99),np.inf],labels=['bottom90pct','90-99pct','top1pct']).astype(str),
        'outer_training_flight_support':pd.cut(supports,[-1,0,5,20,np.inf],labels=['unseen','1-5','6-20','21+']).astype(str)}
    metrics=[];foldrows=[];subgroups=[];tails=[];worst=[];deltas=[]
    for name,p in predictions.items():
        assert np.isfinite(p).all() and (p>=0).all()
        measured=[r for r in summary['folds'] if r['model']==name]
        for f in range(5):
            mask=fold==f;row=dict(model=name,fold=f,**h.metrics(y[mask],p[mask]))
            fit=next((r for r in measured if r['fold']==f),None)
            row['train_r2']=fit['train_r2'] if fit else None
            row['gap']=row['train_r2']-row['r2'] if fit else None;foldrows.append(row)
        local=[r for r in foldrows if r['model']==name]
        metrics.append(dict(model=name,**h.metrics(y,p),mean_fold_r2=float(np.mean([r['r2'] for r in local])),std_fold_r2=float(np.std([r['r2'] for r in local],ddof=1)),
            worst_fold_r2=min(r['r2'] for r in local),mean_train_r2=float(np.mean([r['train_r2'] for r in local])) if measured else None,
            mean_gap=float(np.mean([r['gap'] for r in local])) if measured else None))
        squared=(p-y)**2;top=y>=np.quantile(y,.99);business=~economy
        tails.append(dict(model=name,business_sse_share=float(squared[business].sum()/squared.sum()),top1pct_target_sse_share=float(squared[top].sum()/squared.sum()),
            top1pct_target_rmse=float(np.sqrt(squared[top].mean())),top1pct_target_bias=float((p-y)[top].mean())))
        for label,values in groups.items():
            values=np.asarray(values,str)
            for value in np.unique(values):
                mask=values==value;subgroups.append(dict(model=name,group=label,value=value,**h.metrics(y[mask],p[mask])))
        for i in np.argsort(squared)[-30:][::-1]:
            worst.append(dict(model=name,row_position=int(i),**train.iloc[i].to_dict(),fold=int(fold[i]),prediction=float(p[i]),error=float(p[i]-y[i]),outer_training_flight_support=int(supports[i])))
    for a,b in pairs:
        for f in list(range(5))+['pooled']:
            mask=np.ones(len(y),bool) if f=='pooled' else fold==f
            deltas.append(dict(candidate=a,control=b,fold=f,delta_r2=h.metrics(y[mask],predictions[a][mask])['r2']-h.metrics(y[mask],predictions[b][mask])['r2'],
                delta_rmse=h.metrics(y[mask],predictions[a][mask])['rmse']-h.metrics(y[mask],predictions[b][mask])['rmse']))
    out.mkdir(parents=True,exist_ok=True)
    for name,rows in [('model_metrics',metrics),('fold_metrics',foldrows),('paired_fold_deltas',deltas),('subgroup_metrics',subgroups),('tail_metrics',tails),('worst_errors',worst)]:pd.DataFrame(rows).to_csv(out/(name+'.csv'),index=False)
    aligned=reference.drop(columns='prediction').copy()
    for name,p in predictions.items():aligned[name]=p
    aligned['outer_training_flight_support']=supports;aligned.to_csv(out/'aligned_oof.csv',index=False)
    pd.DataFrame({name:p-y for name,p in predictions.items()}).corr().to_csv(out/'residual_correlations.csv')
    intervals,draws=h.bootstrap_groups(y,predictions,reference.group.to_numpy(),500,20261031 if seed=='primary' else 20261101)
    h.save_json(out/'bootstrap_intervals.json',intervals);draws.to_csv(out/'bootstrap_draws.csv',index=False)
    fig,ax=h.plt.subplots(figsize=(9,4))
    for name in predictions:
        ax.plot(range(5),[r['r2'] for r in foldrows if r['model']==name],marker='o',label=name)
    ax.set(xlabel='Outer fold',ylabel='Original-price R²',title='Global versus Business-augmented predictions');ax.legend(fontsize=8)
    fig.tight_layout();fig.savefig(out/'paired_fold_scores.png',dpi=160);h.plt.close(fig)
    return pd.DataFrame(metrics),pd.DataFrame(deltas),pd.DataFrame(tails)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--train',type=Path,default=s.base.TRAIN);args=ap.parse_args()
    s.common.checked_plan(args.train);train=pd.read_csv(args.train);out=ROOT/'analysis';out.mkdir(exist_ok=True)
    tables={seed:r for seed in ['primary','sensitivity'] if (r:=evaluate(seed,train,out/seed)) is not None}
    assert 'primary' in tables
    trials=[];global_trials=[];selected=[];cachefits=0;outerfits=0
    scopes=list((ROOT/'primary').glob('outer*'))+list((ROOT/'sensitivity').glob('outer*'))+[ROOT/'full_training']
    for scope in scopes:
        if not (scope/'selected.json').exists():continue
        label='full_training' if scope.name=='full_training' else scope.parent.name+'/'+scope.name
        chosen=json.loads((scope/'selected.json').read_text());chosen['scope']=label;selected.append(chosen)
        rows=json.loads((scope/'all_trials.json').read_text());assert len(rows)==12 and all(t['state']=='COMPLETE' for t in rows)
        trials.extend(dict(scope=label,**t) for t in rows)
        cachefits+=len(json.loads((scope/'global_inner_reconstructions.json').read_text()))
        if (scope/'outer_metrics.json').exists():outerfits+=2
        for p in scope.glob('xgb_secondary*_all_trials.json'):
            rows=json.loads(p.read_text());assert len(rows)==12 and all(t['state']=='TrialState.COMPLETE' for t in rows)
            global_trials.extend(dict(scope=label,**t) for t in rows)
    final=int((ROOT/'full_training/final_configuration.json').exists())
    fits=2*(len(trials)+len(global_trials))+cachefits+outerfits+final
    h.save_json(out/'specialist_trials.json',trials);h.save_json(out/'secondary_global_trials.json',global_trials)
    h.save_json(out/'selected_configurations.json',selected)
    pd.DataFrame([dict(scope=d['scope'],rounds=d['rounds'],trial=d['trial'],inner_rounds=json.dumps([r['rounds'] for r in d['inner_splits']]),configuration=json.dumps(d['configuration'])) for d in selected]).to_csv(out/'selected_configurations.csv',index=False)
    h.save_json(out/'completion.json',dict(source_hashes=s.sources(),specialist_trials=len(trials),secondary_global_trials=len(global_trials),
        total_trials=len(trials)+len(global_trials),research_model_fits=fits,synthetic_smoke_fits_excluded=1,secondary_completed='sensitivity' in tables,
        economy_oof_predictions_unchanged=True,test_accessed_by_analysis=False))
    text=['# Business specialist results','A bounded Business-only XGBoost is mixed 50/50 with the existing global XGBoost on Business rows only. Economy is exactly unchanged. One fixed thirds replacement blend is evaluated; no weights are fitted.']
    for seed,(m,d,t) in tables.items():
        text.extend([f'## {seed.capitalize()} evaluation',h.markdown_table(m,['model','r2','rmse','mae','worst_fold_r2','mean_train_r2','mean_gap'],6),
            'Paired candidate minus control changes:',h.markdown_table(d,precision=6),'High-price tail diagnostics:',h.markdown_table(t,precision=5)])
    text.extend(['## Validation limits','Global and specialist parameters and stopping counts are selected inside outer training partitions. The primary global fits exactly reconstruct the frozen XGBoost OOF predictions. The custom stopping metric evaluates the registered halfblend on Business inner validation rows; those scores remain tuning observations.',
        'Training scores for augmented XGBoost are measured from actual composite predictions, not averaged component R². Training scores for the three-family blends are not measured and remain null. Inspect all folds, cabin support, rare flights, price tails and stopping dispersion; smaller gaps alone do not establish generalization.',
        'These observations and primary folds were used in earlier development. The conditional second seed, when run, is a prescribed sensitivity analysis of global versus augmented XGBoost; no secondary three-family blend is fabricated. It is not an untouched holdout. Group-bootstrap intervals condition on current OOF predictions and omit retraining and accumulated selection uncertainty.',
        f'Completed trials: {len(trials)+len(global_trials)}; research fits: {fits}. One synthetic API smoke fit is separate. No public scores or test features chose parameters, weights or postprocessing. No Kaggle upload or hidden-test improvement is claimed.'])
    (out/'FINDINGS.md').write_text('\n\n'.join(text)+'\n')
    print(tables['primary'][0][['model','r2','mean_gap']].to_string(index=False));print('Trials',len(trials)+len(global_trials),'fits',fits)


if __name__=='__main__':main()
