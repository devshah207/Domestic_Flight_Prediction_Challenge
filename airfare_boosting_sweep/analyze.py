"""Training-only diagnostics and independent nested-index audit; no new fits."""
from pathlib import Path
import argparse
import importlib.util
import sys
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
spec=importlib.util.spec_from_file_location('boosting_sweep_study',ROOT/'study.py')
e=importlib.util.module_from_spec(spec);spec.loader.exec_module(e)
import diagnostics as h
import inspect_models
m=e.m;base=e.base


def audit(train):
    reg=e.verify(train);frame,raw,y=base.data(train);outer=base.splits(raw,y,e.OUTER_SEED)
    rows=[];trial_total=0
    for family in m.FAMILIES:
        directory=ROOT/'results'/family
        for scope in [f'outer{i}' for i in range(5)]+['full_training']:
            folder=directory/scope
            if scope=='full_training':
                fit_positions=np.arange(len(y));seed=e.FULL_SEED
            else:
                i=int(scope[-1]);tr,va=outer[i]
                recorded=e.read(folder/'outer_indices.json')
                assert recorded['fit']==tr.tolist() and recorded['validation']==va.tolist()
                assert set(base.group_ids(raw)[tr]).isdisjoint(base.group_ids(raw)[va])
                fit_positions=tr;seed=e.INNER_SEED+i
                metrics=e.read(folder/'metrics.json')
                assert metrics['source_hashes']==e.sources() and metrics['group_overlap']==0
                if 'execution_helper_sha256' in metrics:assert metrics['execution_helper_sha256']==e.sha(ROOT/'scope_worker.py')
                assert np.isfinite([metrics['train_r2'],metrics['r2']]).all()
                pred=pd.read_csv(folder/'predictions.csv')
                assert np.array_equal(pred.row_position,va) and np.isfinite(pred.prediction).all()
                np.testing.assert_allclose(base.scores(y[va],pred.prediction)['r2'],metrics['r2'],atol=1e-12,rtol=0)
                saved=np.load(folder/'train_predictions.npz')
                assert np.array_equal(saved['row_position'],tr)
                np.testing.assert_allclose(base.scores(y[tr],saved['prediction'])['r2'],metrics['train_r2'],atol=1e-12,rtol=0)
            local=raw.iloc[fit_positions].reset_index(drop=True);target=y[fit_positions]
            selection=e.read(folder/'tuning/selection.json');s=selection['signature']
            assert s['source_hashes']==e.sources() and s['row_hash']==e.hash_rows(local)
            assert s['target_hash']==e.hash_y(target) and s['seed']==seed and s['family']==family
            it,iv=base.splits(local,target,seed)[0]
            index=e.read(folder/'tuning/inner_indices.json')
            assert index['fit']==it.tolist() and index['validation']==iv.tolist()
            g=base.group_ids(local);assert set(g[it]).isdisjoint(g[iv])
            trials=e.read(folder/'tuning/trials.json')
            assert len(trials)==m.TRIALS[family] and all(t['state']=='COMPLETE' for t in trials)
            chosen=next(t for t in trials if t['number']==selection['best_trial'])
            assert chosen['value']==min(t['value'] for t in trials)
            for t in trials:
                p=t['attributes']['configuration'];assert p['rounds'] in m.ROUNDS
                assert 4<=p['depth']<=8 and np.isfinite(t['value'])
            trial_total+=len(trials)
            rows.append(dict(family=family,scope=scope,inner_fit_rows=len(it),inner_validation_rows=len(iv),
                outer_validation_used_for_tuning=False,group_overlap=0,source_input_identities_verified=True,
                trials=len(trials),requested_rounds=selection['configuration']['rounds']))
        c=e.read(directory/'full_training/configuration.json')
        if 'execution_helper_sha256' in c:assert c['execution_helper_sha256']==e.sha(ROOT/'scope_worker.py')
        assert c['source_hashes']==e.sources() and c['training_sha256']==reg['training_sha256']
        assert c['model_sha256']==e.sha(directory/'full_training/model.joblib')
        assert c['training_rows']==len(y) and c['test_accessed'] is False
    assert len(rows)==48 and trial_total==336
    recovery=e.read(ROOT/'SCHEDULING_RECOVERY.json')
    assert recovery['recovered_trial_complete'] and recovery['duplicate_fit_attempts']==1
    overlap_trials=e.read(ROOT/'results/lgb_gbdt/outer0/tuning/trials.json')
    assert overlap_trials[2]['attributes']['configuration']==overlap_trials[3]['attributes']['configuration']
    assert overlap_trials[2]['value']==overlap_trials[3]['value']
    value=dict(scopes=rows,optuna_trials=trial_total,outer_refits=40,full_refits=8,
        canonical_registered_research_fits=trial_total+48,duplicate_scheduling_fit_attempts=1,
        recovered_trial2_matches_independent_same_configuration_trial3_exactly=True,
        restarted_sampler_repeated_startup_profile_within_registered_trial_budget=[2,3],
        research_fits=trial_total+49,source_hashes=e.sources(),training_sha256=reg['training_sha256'],
        all_partition_and_group_identities_passed=True,test_accessed=False)
    e.save(ROOT/'VALIDATION_AUDIT.json',value)
    return value


def analyze(train):
    completion=audit(train);frame,raw,y=base.data(train);dest=ROOT/'analysis';dest.mkdir(exist_ok=True)
    predictions={};train_predictions={};folds=None;summaries={}
    group=base.group_ids(raw)
    for family in m.FAMILIES:
        d=pd.read_csv(ROOT/'results'/family/'oof.csv',dtype={'group':str})
        assert np.array_equal(d.id,frame.id) and np.array_equal(d.price,y)
        assert np.array_equal(d.group.to_numpy(),group.astype(str))
        if folds is None:folds=d.fold.to_numpy()
        else:assert np.array_equal(d.fold,folds)
        predictions[family]=d.prediction.to_numpy();summaries[family]=e.read(ROOT/'results'/family/'summary.json')
        train_predictions[family]=[np.load(ROOT/'results'/family/f'outer{i}/train_predictions.npz')['prediction'] for i in range(5)]
    # Controls were frozen before this study and use identical primary folds and labels.
    for name,path in [('control_global_xgb',ROOT.parent/'airfare_business/primary/global_xgb_oof.csv'),
                      ('control_business_augmented',ROOT.parent/'airfare_business/primary/business_augmented_xgb_oof.csv')]:
        d=pd.read_csv(path)
        assert np.array_equal(d.id,frame.id) and np.array_equal(d.price,y) and np.array_equal(d.fold,folds)
        assert np.array_equal(d.group.to_numpy(),group)
        predictions[name]=d.prediction.to_numpy()
    prior_blend=pd.read_csv(ROOT.parent/'airfare_fresh/analysis/aligned_training_oof.csv',dtype={'group':str})
    assert np.array_equal(prior_blend.id,frame.id) and np.array_equal(prior_blend.price,y)
    assert np.array_equal(prior_blend.fold,folds) and np.array_equal(prior_blend.group,group.astype(str))
    predictions['control_previous_equal64']=prior_blend.equal_blend_onehot64.to_numpy()
    for name,weights in m.BLENDS.items():
        predictions[name]=sum(weight*predictions[f] for f,weight in weights.items())
        train_predictions[name]=[sum(weight*train_predictions[f][i] for f,weight in weights.items()) for i in range(5)]
    controls={'control_global_xgb':'global_xgb','control_business_augmented':'business_augmented_xgb'}
    modelrows=[];foldrows=[];tailrows=[];subgroups=[];worst=[]
    flight_support=np.zeros(len(y),int)
    for i,(tr,va) in enumerate(base.splits(raw,y,e.OUTER_SEED)):
        flight_support[va]=raw.iloc[va].flight.map(raw.iloc[tr].flight.value_counts()).fillna(0).to_numpy(int)
    bands=pd.cut(y,[-np.inf,*np.quantile(y,[.5,.9,.99]),np.inf],labels=['bottom50','50-90','90-99','top1']).astype(str)
    schemes={'cabin':raw.flight_class.to_numpy(str),'prefix':raw.flight.str.split().str[0].to_numpy(str),
             'target_percentile':bands,'flight_training_support':pd.cut(flight_support,[-1,0,5,20,np.inf],labels=['unseen','1-5','6-20','21+']).astype(str)}
    reference=base.scores(y,predictions['control_business_augmented'])['r2']
    for name,p in predictions.items():
        local=[]
        for i,(tr,va) in enumerate(base.splits(raw,y,e.OUTER_SEED)):
            row=dict(model=name,fold=i,**h.metrics(y[va],p[va]))
            if name in train_predictions:row['train_r2']=base.scores(y[tr],train_predictions[name][i])['r2']
            elif name=='control_previous_equal64':row['train_r2']=float('nan')
            else:
                oldfold=e.read(ROOT.parent/'airfare_business/primary'/f'outer{i}/outer_metrics.json')
                row['train_r2']=next(r['train_r2'] for r in oldfold['rows'] if r['model']==controls[name])
            row['gap']=row['train_r2']-row['r2'];local.append(row);foldrows.append(row)
        record=dict(model=name,**h.metrics(y,p),mean_train_r2=float(np.mean([r['train_r2'] for r in local])),
            mean_gap=float(np.mean([r['gap'] for r in local])),worst_fold_r2=min(r['r2'] for r in local),
            fold_std_r2=float(np.std([r['r2'] for r in local],ddof=1)),delta_vs_business=base.scores(y,p)['r2']-reference,
            folds_won_vs_business=int(sum(base.scores(y[folds==i],p[folds==i])['r2']>
                                         base.scores(y[folds==i],predictions['control_business_augmented'][folds==i])['r2'] for i in range(5))))
        if name in m.FAMILIES:
            config=e.read(ROOT/'results'/name/'full_training/configuration.json')
            record.update(final_requested_rounds=config['selection']['configuration']['rounds'],
                          final_actual_rounds=config['actual_rounds'],full_training_r2=config['training_r2'])
        modelrows.append(record)
        top=y>=np.quantile(y,.99);sq=(p-y)**2;business=raw.flight_class.eq('Business').to_numpy()
        tailrows.append(dict(model=name,top1_rmse=float(np.sqrt(sq[top].mean())),top1_bias=float((p-y)[top].mean()),
                             top1_sse_share=float(sq[top].sum()/sq.sum()),business_sse_share=float(sq[business].sum()/sq.sum())))
        for scheme,values in schemes.items():
            values=np.asarray(values,str)
            for value in np.unique(values):
                mask=values==value;subgroups.append(dict(model=name,group=scheme,value=value,**h.metrics(y[mask],p[mask])))
        for i in np.argsort(sq)[-20:][::-1]:
            worst.append(dict(model=name,row_position=int(i),**frame.iloc[i].to_dict(),prediction=p[i],error=p[i]-y[i]))
    for file,rows in [('model_metrics',modelrows),('fold_metrics',foldrows),('tail_metrics',tailrows),('subgroup_metrics',subgroups),('worst_errors',worst)]:
        pd.DataFrame(rows).to_csv(dest/(file+'.csv'),index=False)
    pd.DataFrame({name:p-y for name,p in predictions.items()}).corr().to_csv(dest/'oof_residual_correlations.csv')
    ci,draws=h.bootstrap_groups(y,predictions,group,500,20261031)
    h.save_json(dest/'bootstrap_intervals.json',ci);draws.to_csv(dest/'bootstrap_draws.csv',index=False)
    trials=[];configs={};counts=[]
    for family in m.FAMILIES:
        configs[family]=e.read(ROOT/'results'/family/'full_training/configuration.json')
        for scope in [f'outer{i}' for i in range(5)]+['full_training']:
            folder=ROOT/'results'/family/scope/'tuning';sel=e.read(folder/'selection.json')
            for trial in e.read(folder/'trials.json'):
                trials.append(dict(family=family,scope=scope,trial=trial['number'],state=trial['state'],
                    selected=trial['number']==sel['best_trial'],tuning_mse=trial['value'],
                    actual_rounds=trial['attributes']['actual_rounds'],seconds=trial['attributes']['seconds'],
                    **trial['attributes']['configuration']))
    pd.DataFrame(trials).to_csv(dest/'all_trials.csv',index=False)
    resolved={};tree_audit=[]
    for family in m.FAMILIES:
        bundle=e.joblib.load(ROOT/'results'/family/'full_training/model.joblib')
        tree_audit.append(inspect_models.inspect(bundle['model'],family,configs[family]))
        if family=='xgb':resolved[family]=e.json.loads(bundle['model'].get_booster().save_config())
        elif family.startswith('lgb'):resolved[family]=e.base.plain(bundle['model'].booster_.params)
    e.save(ROOT/'TREE_COMPLEXITY_AUDIT.json',dict(models=tree_audit,all_checks_passed=True,model_fits=0,test_read=False))
    e.save(ROOT/'MODEL_CONFIGURATIONS.json',dict(models=configs,blends=m.BLENDS,resolved_library_details=resolved,
        note='All final dictionaries selected by fresh full-data inner tuning, not outer validation or public results.'))
    e.save(ROOT/'EXPERIMENT_LEDGER.json',dict(research_optuna_trials=336,research_model_fit_attempts=385,
        canonical_registered_research_fits=384,duplicate_scheduling_fit_attempts=1,
        scheduling_recovery='One overlapping LightGBM outer0 trial was recovered at its exact registered parameters and inner rows. See SCHEDULING_RECOVERY.json; no new parameter proposal or outer/test-based model choice.',
        outer_refits=40,full_refits=8,successful_final_api_smoke_fits=8,earlier_smoke_fits_before_assertion_repair=7,
        smoke_repair='Sparse matrix row-count assertion changed from len() to shape[0] before registration; no model-choice change.',
        all_trial_seconds=float(sum(t['seconds'] for t in trials)),test_accessed=False,
        timing_note='Trial durations are elapsed wall time, include deliberately suspended runners and CPU contention, overlap across workers, and are not CPU time or additive study wall time.',
        user_authorized_aggressive_exploratory_delivery=True,public_scores_used_for_tuning=False))
    table=pd.DataFrame(modelrows).sort_values('r2',ascending=False)
    fig,ax=h.plt.subplots(figsize=(10,6));ax.barh(table.model[::-1],table.r2[::-1],color='#267a9b')
    ax.axvline(reference,color='#b53a36',linestyle='--',label='Prior Business control CV')
    ax.set(xlabel='Original-price grouped outer OOF R²',title='High-round Optuna procedures: training-only validation');ax.legend()
    fig.tight_layout();fig.savefig(dest/'outer_scores.png',dpi=160);h.plt.close(fig)
    text=['# High-round Optuna boosting results',
        'All scores below are training-only estimates. No new Kaggle R² or rank is known. All successful candidates are delivered under the user’s explicit request for multiple aggressive exploratory submissions; weaker results are not concealed.',
        h.markdown_table(table,['model','r2','delta_vs_business','mean_train_r2','mean_gap','worst_fold_r2','folds_won_vs_business','final_actual_rounds'],6),
        '## Tail and subgroup diagnostics',h.markdown_table(pd.DataFrame(tailrows),precision=5),
        '## Interpretation and limits',
        'Each outer fold tunes only inside its fitting partition, then refits at a fixed selected high count before scoring the outer validation fold. Outer folds are shared with previous research and are not a fresh untouched test. Each scope uses only one inner 80/20 split to contain runtime; hyperparameters can be unstable. Selection among eight families also creates optimism. Bootstrap intervals condition on saved OOF predictions and omit retraining, tuning and accumulated research uncertainty.',
        'The previous equal-blend64 OOF is included as an additional descriptive control on verified identical rows/folds, so a gain over the Business control is not presented as beating every earlier CV result. Its composite training predictions were not retained in that phase; its training gap is left missing rather than estimated from component gaps. No new fitting or changes to any submitted blend weights are introduced by this comparison.',
        'The explicit 800-round floor may be suboptimal; these are requested high-capacity candidates, not claims of optimal count. Actual rounds are reported separately because AdaBoost and some libraries can terminate before a requested cap. A larger training gap is a warning, not proof of its cause. Tree count alone does not establish overfitting or improve the public leaderboard.',
        'A scheduling race caused a second runner to enter the existing LightGBM outer0 scope while original trial2 was suspended. That exact trial was recovered using identical parameters/inner rows, without a new suggestion. The original duplicate later fails its second completion update for the already-finished trial, preventing further duplicate suggestions. There are 336 distinct Optuna trials and 384 canonical research fits, plus one duplicate fit attempt (385 total). This one scope includes asynchronous/restarted TPE ordering; saved trials provide the exact parameters. Other scopes follow the registered seeded procedure. See SCHEDULING_RECOVERY.json and logs.',
        'Exact final dictionaries, requested and actual counts, feature schemas and resolved library parameters are in MODEL_CONFIGURATIONS.json. Trials, selections, fold indices, errors, confidence intervals and fit accounting are retained. Both blends have registered fixed equal weights; none were fit to outer or public scores. Existing models and submissions remain unchanged.',
        '## Reproduction',
        'Use the pinned ../airfare_fresh/requirements.txt. From the repository root set OMP_NUM_THREADS=4 OMP_WAIT_POLICY=PASSIVE OPENBLAS_NUM_THREADS=1, then run python airfare_boosting_sweep/study.py --stage register --train TRAIN.csv; python airfare_boosting_sweep/study.py --stage run --train TRAIN.csv; python airfare_boosting_sweep/analyze.py --train TRAIN.csv. Run in a fresh copy without generated study outputs to reproduce fitting. Registration prevents silent code/data/version changes. Inference has a one-use marker and never permits research after test access.']
    (dest/'FINDINGS.md').write_text('\n\n'.join(text)+'\n')
    e.save(dest/'completion.json',dict(research_fits=completion['research_fits'],optuna_trials=336,
        final_models=len(configs),outer_folds=5,validation_audit_sha256=e.sha(ROOT/'VALIDATION_AUDIT.json'),
        source_hashes=e.sources(),test_accessed=False))
    print(table[['model','r2','mean_gap','final_actual_rounds']].to_string(index=False),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--train',type=Path,default=base.TRAIN)
    args=parser.parse_args();analyze(args.train)
