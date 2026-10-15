"""Training-only diagnostics for the registered direct CatBoost64 phase."""
from pathlib import Path
import argparse
import json
import sys

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
import direct_cat64 as run
import diagnostics as helpers


def inspect(seed_name, train, out):
    path=ROOT/seed_name/run.VARIANT/'oof.csv'
    if not path.exists():return None
    direct=pd.read_csv(path);y=direct.price.to_numpy();fold=direct.fold.to_numpy()
    assert len(direct)==len(train)==15589 and direct.row_position.is_unique
    assert np.array_equal(direct.row_position,np.arange(len(train)))
    assert np.array_equal(direct.id,train.id) and np.array_equal(y,train.price)
    assert direct.groupby('group').fold.nunique().max()==1
    predictions={run.VARIANT:direct.prediction.to_numpy()}
    records={run.VARIANT:json.loads(path.with_name('summary.json').read_text())}
    if seed_name=='primary':
        for family in ['xgb','lgb','cat_onehot64']:
            p=run.FRESH/'results'/family/'oof.csv';df=pd.read_csv(p)
            pd.testing.assert_frame_equal(df.drop(columns='prediction'),direct.drop(columns='prediction'))
            predictions[family]=df.prediction.to_numpy()
            records[family]=json.loads(p.with_name('summary.json').read_text())
        shared=predictions['xgb']+predictions['lgb']
        predictions['frozen_equal_blend64']=(shared+predictions['cat_onehot64'])/3
        predictions['direct_equal_blend64']=(shared+predictions[run.VARIANT])/3
        pairs=[(run.VARIANT,'cat_onehot64'),('direct_equal_blend64','frozen_equal_blend64')]
    else:
        for family in ['cat2_control','hybrid64_control']:
            p=ROOT/seed_name/family/'oof.csv';df=pd.read_csv(p)
            pd.testing.assert_frame_equal(df.drop(columns='prediction'),direct.drop(columns='prediction'))
            predictions[family]=df.prediction.to_numpy()
            records[family]=json.loads(p.with_name('summary.json').read_text())
        pairs=[(run.VARIANT,'hybrid64_control')]
    metric_rows=[];fold_rows=[];paired=[];subgroups=[];tails=[];worst=[]
    prefix=train.flight.str.split().str[0]
    support=np.zeros(len(y),int)
    for f in range(5):
        mask=fold==f
        support[mask]=train.loc[mask,'flight'].map(train.loc[~mask,'flight'].value_counts()).fillna(0).to_numpy(int)
    support_band=pd.cut(support,[-1,0,5,20,np.inf],labels=['unseen','1-5','6-20','21+']).astype(str)
    target_band=pd.cut(y,[-np.inf,np.quantile(y,.9),np.quantile(y,.99),np.inf],labels=['bottom90pct','90-99pct','top1pct']).astype(str)
    groups={'cabin':train.flight_class.astype(str),'target_band':target_band,'outer_train_flight_support':support_band,
        'prefix_cabin':prefix+'|'+train.flight_class.astype(str),'route_cabin':train.source_city+'>'+train.destination_city+'|'+train.flight_class,
        'duration_missing':train.duration.isna().astype(str),'airline_missing':train.airline.isna().astype(str)}
    for name,p in predictions.items():
        assert np.isfinite(p).all() and (p>=0).all()
        row=dict(seed=seed_name,model=name,**helpers.metrics(y,p))
        component_folds=records[name]['folds'] if name in records else None
        train_r2=[r['train_r2'] for r in component_folds] if component_folds else None
        for f in range(5):
            mask=fold==f;fr=dict(seed=seed_name,model=name,fold=f,**helpers.metrics(y[mask],p[mask]))
            fr['train_r2']=train_r2[f] if train_r2 else np.nan
            fr['gap']=fr['train_r2']-fr['r2'];fold_rows.append(fr)
        local=[r for r in fold_rows if r['model']==name]
        row.update(mean_fold_r2=float(np.mean([r['r2'] for r in local])),std_fold_r2=float(np.std([r['r2'] for r in local],ddof=1)),worst_fold_r2=min(r['r2'] for r in local),
            mean_train_r2=float(np.mean(train_r2)) if train_r2 else None,
            mean_gap=float(np.mean([r['gap'] for r in local])) if train_r2 else None)
        metric_rows.append(row)
        squared=(p-y)**2;total=squared.sum();top=y>=np.quantile(y,.99);business=train.flight_class.eq('Business').to_numpy()
        tails.append(dict(seed=seed_name,model=name,business_sse_share=float(squared[business].sum()/total),
            top1pct_target_sse_share=float(squared[top].sum()/total),top1pct_target_bias=float((p-y)[top].mean()),
            top1pct_target_rmse=float(np.sqrt(squared[top].mean())),worst1pct_error_sse_share=float(np.sort(squared)[-int(np.ceil(len(y)*.01)):].sum()/total)))
        for group_name,values in groups.items():
            values=np.asarray(values,str)
            for value in np.unique(values):
                mask=values==value
                subgroups.append(dict(seed=seed_name,model=name,group=group_name,value=value,**helpers.metrics(y[mask],p[mask])))
        for i in np.argsort(squared)[-30:][::-1]:
            worst.append(dict(seed=seed_name,model=name,row_position=int(i),**train.iloc[i].to_dict(),fold=int(fold[i]),
                prediction=float(p[i]),error=float(p[i]-y[i]),outer_training_flight_support=int(support[i])))
    for a,b in pairs:
        for f in list(range(5))+['pooled']:
            mask=np.ones(len(y),bool) if f=='pooled' else fold==f
            paired.append(dict(seed=seed_name,candidate=a,control=b,fold=f,
                delta_r2=helpers.metrics(y[mask],predictions[a][mask])['r2']-helpers.metrics(y[mask],predictions[b][mask])['r2'],
                delta_rmse=helpers.metrics(y[mask],predictions[a][mask])['rmse']-helpers.metrics(y[mask],predictions[b][mask])['rmse']))
    out.mkdir(parents=True,exist_ok=True)
    for filename,rows in [('model_metrics',metric_rows),('fold_metrics',fold_rows),('paired_fold_deltas',paired),('subgroup_metrics',subgroups),('tail_metrics',tails),('worst_errors',worst)]:
        pd.DataFrame(rows).to_csv(out/(filename+'.csv'),index=False)
    oof=direct.drop(columns='prediction').copy()
    for name,p in predictions.items():oof[name]=p
    oof['outer_training_flight_support']=support;oof.to_csv(out/'aligned_oof.csv',index=False)
    pd.DataFrame({name:p-y for name,p in predictions.items()}).corr().to_csv(out/'residual_correlations.csv')
    bootstrap,draws=helpers.bootstrap_groups(y,predictions,direct.group.to_numpy(),500,20261029 if seed_name=='primary' else 20261030)
    helpers.save_json(out/'bootstrap_intervals.json',bootstrap);draws.to_csv(out/'bootstrap_draws.csv',index=False)
    fig,axes=helpers.plt.subplots(1,2,figsize=(12,4))
    for name in predictions:
        local=[r for r in fold_rows if r['model']==name]
        axes[0].plot(range(5),[r['r2'] for r in local],marker='o',label=name)
    axes[0].set(xlabel='Outer fold',ylabel='Original-price R²',title='Paired outer validation')
    axes[0].legend(fontsize=7)
    measured=[r for r in metric_rows if r['mean_gap'] is not None]
    axes[1].bar([r['model'] for r in measured],[r['mean_gap'] for r in measured])
    axes[1].set(ylabel='Mean training R² − validation R²',title='Measured component gaps')
    axes[1].tick_params(axis='x',rotation=25)
    fig.tight_layout();fig.savefig(out/'fold_scores_and_gaps.png',dpi=160);helpers.plt.close(fig)
    fig,ax=helpers.plt.subplots(figsize=(9,4))
    tail_rows=pd.DataFrame(tails)
    ax.bar(tail_rows.model,tail_rows.top1pct_target_sse_share)
    ax.set(ylabel='Fraction of total squared error',title='Error attributable to highest-priced 1% of training rows')
    ax.tick_params(axis='x',rotation=25);fig.tight_layout();fig.savefig(out/'expensive_fare_error_share.png',dpi=160);helpers.plt.close(fig)
    return pd.DataFrame(metric_rows),pd.DataFrame(paired),pd.DataFrame(tails)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--train',type=Path,default=run.TRAIN);args=ap.parse_args()
    run.checked_plan(args.train);train=pd.read_csv(args.train)
    out=ROOT/'analysis';out.mkdir(exist_ok=True)
    tables={}
    for seed in ['primary','sensitivity']:
        result=inspect(seed,train,out/seed)
        if result:tables[seed]=result
    assert 'primary' in tables,'Primary study not yet complete'
    trial_rows=[];selected=[];fit_count=0
    for seed in ['primary','sensitivity']:
        for p in sorted((ROOT/seed).glob('*/*/all_trials.json')):
            family=p.parent.parent.name;scope=p.parent.name
            records=json.loads(p.read_text())
            assert len(records)==12 and all(r['state']=='COMPLETE' for r in records)
            for t in records:
                trial_rows.append(dict(seed=seed,family=family,scope=scope,**t))
        for p in sorted((ROOT/seed).glob('*/*/selected.json')):
            s=json.loads(p.read_text());family=p.parent.parent.name
            selected.append(dict(seed=seed,family=family,scope=p.parent.name,mode=s['mode'],rounds=s['rounds'],
                inner_rounds=[r['rounds'] for r in s['inner_splits']],configuration=s['configuration'],threshold=s['threshold'],
                selected_trial=s['trial'],new_trials=0 if family=='hybrid64_control' else 12))
        fit_count+=len(list((ROOT/seed).glob('*/outer*/outer_metrics.json')))
        fit_count+=2*len(list((ROOT/seed).glob('hybrid64_control/outer*/selected.json')))
        fit_count+=len(list((ROOT/seed).glob('*/full_training/final_configuration.json')))
    fit_count+=2*len(trial_rows)
    helpers.save_json(out/'all_trials.json',trial_rows);helpers.save_json(out/'selected_configurations.json',selected)
    pd.DataFrame([{**s,'inner_rounds':json.dumps(s['inner_rounds']),'configuration':json.dumps(s['configuration'])} for s in selected]).to_csv(out/'selected_configurations.csv',index=False)
    helpers.save_json(out/'completion.json',dict(trials_completed=len(trial_rows),research_model_fits=fit_count,
        synthetic_smoke_fits_excluded=1,secondary_completed='sensitivity' in tables,source_hashes=run.sources(),test_accessed_by_analysis=False))
    sections=['# Direct CatBoost64 tuning results','This phase retunes the entire bounded CatBoost search with threshold 64. No new features, weight search, loss change or competition test data informed development. All comparisons are developmental on previously used training observations. The additional split seed, when run, is a registered robustness check, not a pristine confirmation holdout.']
    for seed,(metrics,paired,tails) in tables.items():
        sections.extend([f'## {seed.capitalize()} evaluation',helpers.markdown_table(metrics,['model','r2','rmse','mae','worst_fold_r2','mean_train_r2','mean_gap'],6),
            'Paired changes use candidate minus control. Negative RMSE change is better.',helpers.markdown_table(paired,precision=6),
            'Tail diagnostics use original-price errors; negative bias means underprediction.',helpers.markdown_table(tails,precision=5)])
    sections.extend(['## Interpretation and limitations',
        'The primary fixed replacement blend uses exactly one third each of the frozen original XGBoost, frozen original LightGBM and the newly tuned CatBoost64. No weights were optimized. The sensitivity evaluation reruns original CatBoost selection, its threshold-64 hybrid and direct threshold-64 tuning on a different prescribed set of partitions. It does not evaluate a secondary three-family blend.',
        'Outer labels never choose per-fold parameters or stopping rounds. However, the research direction follows prior training diagnostics and reported public outcomes. Reusing folds and selecting a preferred candidate across studies introduces selection optimism. Bootstrap intervals condition on current OOF predictions, not retraining, search, folds or accumulated model-selection uncertainty.',
        'A small training gap does not prove generalization. Business/Economy errors, high-price tails, rare-flight support, stopping dispersion and worst-fold behavior must be considered together. Full-data configurations are not independently tested by the outer scores. Do not compare these local R² values numerically with Kaggle public R².',
        f'Completed Optuna trials: {len(trial_rows)}. Research fits: {fit_count}. One synthetic API smoke fit is separate. See PROTOCOL.md for the prospective budget and conditional split-seed compute rule.',
        'The frozen earlier best is retained regardless of this phase’s result. No Kaggle uploads are performed by this research.'])
    (out/'FINDINGS.md').write_text('\n\n'.join(sections)+'\n')
    print(pd.DataFrame(tables['primary'][0])[['model','r2','mean_gap']].to_string(index=False))
    print('Completed',len(trial_rows),'trials;',fit_count,'research fits')


if __name__=='__main__':main()
