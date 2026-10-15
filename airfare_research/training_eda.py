#!/usr/bin/env python3
"""Train-only descriptive audit and masked-feature reconstruction; never reads Kaggle test.

EDA is descriptive, not independent performance proof. Every learned preprocessing
component used in predictive experiments must be refitted inside its training fold.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.model_selection import GroupKFold
os.environ.setdefault('MPLCONFIGDIR', '/tmp/airfare-matplotlib')
os.environ.setdefault('XDG_CACHE_HOME', '/tmp/airfare-cache')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE_HIERARCHY = [
    (['flight', 'source_city', 'destination_city', 'stops'], 2),
    (['flight', 'source_city', 'destination_city'], 2),
    (['flight', 'stops'], 2), (['flight'], 2),
    (['source_city', 'destination_city', 'stops', 'flight_class'], 5),
    (['source_city', 'destination_city', 'stops'], 5),
    (['source_city', 'destination_city'], 5), (['stops'], 5),
]
TIME_HIERARCHY = [
    (['flight', 'source_city', 'destination_city', 'stops', 'departure_time', 'arrival_time'], 2),
    (['flight', 'source_city', 'destination_city', 'stops', 'arrival_time'], 2),
    (['flight', 'source_city', 'destination_city', 'stops', 'departure_time'], 2),
] + BASE_HIERARCHY
PREFIX_HIERARCHY = [
    (['flight', 'source_city', 'destination_city', 'stops', 'departure_time', 'arrival_time'], 2),
    (['flight', 'source_city', 'destination_city', 'stops'], 2),
    (['flight', 'source_city', 'destination_city'], 2),
    (['flight', 'stops'], 2), (['flight'], 2),
    (['flight_prefix', 'source_city', 'destination_city', 'stops', 'departure_time', 'arrival_time'], 3),
    (['flight_prefix', 'source_city', 'destination_city', 'stops'], 3),
] + BASE_HIERARCHY[4:]


def markdown_table(frame, max_rows=35):
    f = frame.head(max_rows).copy()
    for c in f.columns:
        if pd.api.types.is_float_dtype(f[c]):
            f[c] = f[c].map(lambda x: '' if pd.isna(x) else f'{x:,.4f}')
    rows = ['| ' + ' | '.join(map(str, f.columns)) + ' |',
            '| ' + ' | '.join(['---'] * len(f.columns)) + ' |']
    rows += ['| ' + ' | '.join(map(str, row)) + ' |' for row in f.itertuples(index=False, name=None)]
    return '\n'.join(rows)


def summarize_group(d, keys):
    out = d.groupby(keys, dropna=False).agg(n=('price', 'size'), mean=('price', 'mean'),
                 median=('price', 'median'), std=('price', 'std'), min=('price', 'min'),
                 max=('price', 'max')).reset_index()
    out['cv'] = out['std'] / out['mean']
    return out


def duration_predict(train, valid, hierarchy):
    observed = train.dropna(subset=['duration'])
    pred = pd.Series(np.nan, index=valid.index)
    source = pd.Series('global', index=valid.index)
    for j, (keys, threshold) in enumerate(hierarchy):
        t = observed.groupby(keys, dropna=False).duration.agg(['median', 'count'])
        t = t.loc[t['count'] >= threshold, 'median']
        need = pred.isna()
        if len(keys) == 1:
            values = valid.loc[need, keys[0]].map(t)
        else:
            idx = pd.MultiIndex.from_frame(valid.loc[need, keys])
            values = pd.Series(t.reindex(idx).to_numpy(), index=valid.index[need])
        pred.loc[need] = values
        source.loc[values.dropna().index] = str(j)
    return pred.fillna(observed.duration.median()), source


def duplicate_summary(d, feature_cols, name, out):
    keys = pd.util.hash_pandas_object(d[feature_cols], index=False)
    g = d.groupby(keys).price.agg(['size', 'nunique', 'mean', 'min', 'max'])
    repeated = g[g['size'] > 1]
    conflict = repeated[repeated['nunique'] > 1]
    means = keys.map(g['mean'])
    # In-sample group-mean residual is an observed lower bound for a deterministic
    # predictor on this particular sample, not a population irreducible-noise estimate.
    sse = float(np.square(d.price - means).sum())
    info = dict(groups=int(len(g)), repeated_groups=int(len(repeated)),
                repeated_rows=int(repeated['size'].sum()),
                redundant_rows=int((repeated['size'] - 1).sum()),
                conflicting_groups=int(len(conflict)), conflicting_rows=int(conflict['size'].sum()),
                observed_within_key_sse=sse,
                observed_within_key_sse_fraction_total_variation=sse / float(np.square(d.price-d.price.mean()).sum()))
    temp = d.copy(); temp['group_id'] = keys.astype(str)
    temp[temp.group_id.isin(set(conflict.index.astype(str)))].to_csv(out / f'{name}_conflicts.csv', index=False)
    return info


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--train', default='/workspace/attachments/085aeb37-413d-42c9-bb16-baafd968488d/train.csv')
    p.add_argument('--out', default='/workspace/airfare_research/eda')
    args = p.parse_args(); out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    source = Path(args.train)
    if source.name != 'train.csv':
        raise ValueError('Use the labeled train.csv only.')
    d = pd.read_csv(source)
    if 'price' not in d: raise ValueError('Expected labeled training data.')
    raw_cols = [c for c in d if c not in ['id', 'price']]
    d['flight_prefix'] = d.flight.str.split().str[0]
    d['route'] = d.source_city + ' -> ' + d.destination_city
    d['airline_missing'] = d.airline.isna()
    d['duration_missing'] = d.duration.isna()
    summary = {'data_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
               'n_rows': len(d), 'n_columns_original': 12,
               'missing': d[raw_cols].isna().sum().to_dict(),
               'price_quantiles': d.price.quantile([0,.01,.05,.25,.5,.75,.9,.95,.99,1]).to_dict(),
               'price_mean': d.price.mean(), 'price_sd': d.price.std(), 'price_skew': d.price.skew(),
               'duration_quantiles': d.duration.quantile([0,.001,.01,.25,.5,.75,.99,1]).to_dict(),
               'days_left_range': [int(d.days_left.min()), int(d.days_left.max())]}
    summary['raw_duplicates'] = duplicate_summary(d, raw_cols, 'raw', out)
    # Fill airline using only deterministic feature redundancy; this is EDA only.
    prefix_airline = d.dropna(subset=['airline']).groupby('flight_prefix').airline.agg(lambda x: x.mode().iloc[0])
    prefix_purity = d.dropna(subset=['airline']).groupby('flight_prefix').airline.nunique()
    summary['prefix_airline_max_distinct'] = int(prefix_purity.max())
    canonical = d.copy()
    canonical['airline'] = canonical.airline.fillna(canonical.flight_prefix.map(prefix_airline))
    summary['airline_canonical_duplicates'] = duplicate_summary(canonical, raw_cols, 'airline_canonical', out)
    # A missing duration is not an observed duration; do not group NaN with an imputed
    # value when blocking true duplicates. A separate missing-agnostic key quantifies risk.
    structural = [c for c in raw_cols if c not in ['airline', 'duration']]
    summary['structural_keys_ignoring_duration_airline'] = duplicate_summary(d, structural, 'structural', out)
    for keys, name in [(['flight_class'],'class'), (['flight_class','stops'],'class_stops'),
                      (['flight_prefix','flight_class'],'prefix_class'),
                      (['route','flight_class'],'route_class'),
                      (['airline_missing','flight_class'],'airline_missing_class'),
                      (['duration_missing','flight_class'],'duration_missing_class')]:
        summarize_group(d, keys).to_csv(out / f'{name}_price_summary.csv', index=False)
    sparsity = d.groupby('flight').size()
    summary['flight_sparsity'] = {'n_flights': len(sparsity), 'lt12_flights': int((sparsity<12).sum()),
          'rows_in_lt12_flights': int(sparsity[sparsity<12].sum()),
          'singletons': int((sparsity==1).sum()), 'singleton_rows': int(sparsity[sparsity==1].sum())}
    s = d.groupby(['flight', 'route', 'flight_class']).size()
    summary['flight_route_class_sparsity'] = {'n_keys': len(s),'lt5_keys':int((s<5).sum()),'rows_in_lt5_keys':int(s[s<5].sum())}
    top = d[d.price >= d.price.quantile(.99)]
    summary['top_1pct'] = {'threshold': float(d.price.quantile(.99)), 'n': len(top),
          'classes': top.flight_class.value_counts().to_dict(), 'prefixes': top.flight_prefix.value_counts().to_dict(),
          'duration_missing_fraction': float(top.duration_missing.mean())}
    correlations = []
    for key,z in d.groupby(['flight_prefix','flight_class']):
        q=z.dropna(subset=['duration']); n=len(q)
        if n < 10: continue
        correlations.append(dict(flight_prefix=key[0],flight_class=key[1],n=n,
             price_duration_pearson=q.price.corr(q.duration),
             price_duration_spearman=spearmanr(q.price,q.duration).statistic,
             days_price_pearson=z.price.corr(z.days_left),
             price_cv=z.price.std()/z.price.mean()))
    corr = pd.DataFrame(correlations); corr.to_csv(out/'prefix_class_correlations.csv',index=False)
    # Within-group duration dispersion tests whether departure/arrival identifies
    # itinerary variants hidden by the original first-flight-number grouping.
    dispersion=[]
    for keys in [['flight','route','stops'],['flight','route','stops','departure_time'],
                 ['flight','route','stops','arrival_time'],['flight','route','stops','departure_time','arrival_time']]:
        obs=d.dropna(subset=['duration']); g=obs.groupby(keys).duration
        ag=g.agg(['size','std','nunique','min','max']); ag=ag[ag['size']>=3]
        dispersion.append(dict(keys='+'.join(keys),groups_ge3=len(ag),rows=int(ag['size'].sum()),
             weighted_sd=float(np.sqrt((ag['std'].fillna(0)**2*ag['size']).sum()/ag['size'].sum())),
             weighted_range=float(((ag['max']-ag['min'])*ag['size']).sum()/ag['size'].sum()),
             multiple_durations_groups=int((ag['nunique']>1).sum())))
    pd.DataFrame(dispersion).to_csv(out/'duration_group_dispersion.csv',index=False)
    # Five folds mask known feature values; all identical raw feature rows stay
    # together. No price is used for splitting, imputation, or scoring this diagnostic.
    raw_groups=pd.util.hash_pandas_object(d[raw_cols],index=False).to_numpy()
    imputation_rows=[]; imputation_predictions=[]
    for fold,(tr,va) in enumerate(GroupKFold(5).split(d,groups=raw_groups)):
        train=d.iloc[tr]; valid=d.iloc[va]; valid=valid[valid.duration.notna()]
        for name,hierarchy in [('baseline',BASE_HIERARCHY),('departure_arrival',TIME_HIERARCHY),('time_prefix_fallback',PREFIX_HIERARCHY)]:
            pred,level=duration_predict(train,valid,hierarchy)
            error=pred-valid.duration
            for cls in ['all','Business','Economy']:
                mask=np.ones(len(valid),dtype=bool) if cls=='all' else valid.flight_class.eq(cls).to_numpy()
                e=error.to_numpy()[mask]
                imputation_rows.append(dict(fold=fold,method=name,flight_class=cls,n=len(e),
                       mae=float(np.abs(e).mean()),rmse=float(np.sqrt(np.square(e).mean())),
                       exact_008hr_fraction=float((np.abs(e)<=.08).mean()),large_error_2hr_fraction=float((np.abs(e)>2).mean())))
            imputation_predictions.append(pd.DataFrame({'id':valid.id.to_numpy(),'fold':fold,'method':name,
                  'flight_class':valid.flight_class.to_numpy(),'flight_prefix':valid.flight_prefix.to_numpy(),
                  'true_duration':valid.duration.to_numpy(),'pred_duration':pred.to_numpy(),'level':level.to_numpy(),
                  'error':error.to_numpy()}))
    imputation = pd.DataFrame(imputation_rows); imputation.to_csv(out/'duration_masked_fold_scores.csv',index=False)
    imp_pred=pd.concat(imputation_predictions,ignore_index=True); imp_pred.to_csv(out/'duration_masked_predictions.csv',index=False)
    imp_summary=imp_pred.groupby(['method','flight_class']).agg(n=('error','size'),mae=('error',lambda x:np.abs(x).mean()),
         rmse=('error',lambda x:np.sqrt(np.square(x).mean())),bias=('error','mean')).reset_index()
    imp_summary.to_csv(out/'duration_masked_summary.csv',index=False)
    summary['duration_masking_design']='5 GroupKFold, raw identical feature groups blocked, no prices used; all observed durations masked only in their validation fold'
    # Training-only patterns across lead time, with class/prefix composition retained.
    d['days_bin']=pd.cut(d.days_left,[0,3,7,14,21,30,40,50,60],include_lowest=True).astype(str)
    day_summary=summarize_group(d,['flight_class','days_bin']); day_summary.to_csv(out/'days_class_patterns.csv',index=False)
    prefix_days=summarize_group(d,['flight_prefix','flight_class','days_bin']); prefix_days.to_csv(out/'days_prefix_class_patterns.csv',index=False)
    # Descriptive matched-cell demeaning controls for fixed route/prefix/class/stops
    # composition. This is not an estimated treatment effect or predictive score.
    cell=['flight_prefix','flight_class','route','stops']
    means=d.groupby(cell).price.transform('mean'); d['price_cell_demeaned']=d.price-means
    d.groupby(['flight_class','days_bin']).agg(n=('price','size'),mean_price=('price','mean'),
        mean_within_cell_deviation=('price_cell_demeaned','mean')).to_csv(out/'days_within_cell_patterns.csv')
    # Observed outlier labels; do not trim or downweight based on this table.
    d.nlargest(40,'price').to_csv(out/'top_40_training_prices.csv',index=False)
    d.nlargest(40,'duration').to_csv(out/'top_40_training_durations.csv',index=False)
    summary['invalid']={'nonpositive_prices':int((d.price<=0).sum()),'nonpositive_observed_duration':int((d.duration.dropna()<=0).sum()),
          'negative_days':int((d.days_left<0).sum()),'duplicate_ids':int(d.id.duplicated().sum())}
    # Plots use original observed feature/target values and no model predictions.
    fig,axs=plt.subplots(1,2,figsize=(12,4.2))
    for cls,color in [('Economy','#2878b5'),('Business','#d15a3c')]:
        z=d[d.flight_class.eq(cls)]
        axs[0].hist(z.price,bins=np.linspace(0,d.price.max(),65),alpha=.55,label=cls,color=color)
        observed=z.dropna(subset=['duration'])
        axs[1].scatter(observed.duration,observed.price,s=5,alpha=.18,label=cls,color=color,rasterized=True)
    axs[0].set(xlabel='Ticket price',ylabel='Rows',title='Training target distribution');axs[0].legend()
    axs[1].set(xlabel='Observed duration',ylabel='Ticket price',title='Duration association differs by class');axs[1].legend()
    fig.tight_layout();fig.savefig(out/'target_duration.png',dpi=150);plt.close(fig)
    prefixes=['6E','AI','EK','EY','TG','UL']
    fig,axs=plt.subplots(2,3,figsize=(12,7),sharey=True)
    for ax,prefix in zip(axs.flat,prefixes):
        z=d[d.flight_prefix.eq(prefix)&d.flight_class.eq('Business')].dropna(subset=['duration'])
        ax.scatter(z.duration,z.price,s=6,alpha=.25);ax.set(title=f'{prefix}, Business, n={len(z)}',xlabel='Observed duration',ylabel='Price')
    fig.tight_layout();fig.savefig(out/'business_duration_by_prefix.png',dpi=150);plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(11,4))
    for ax,cls in zip(axs,['Economy','Business']):
        z=d[d.flight_class.eq(cls)].groupby('days_left').price.agg(['mean','median','size'])
        ax.plot(z.index,z['mean'],label='Mean');ax.plot(z.index,z['median'],label='Median');ax.legend()
        ax.set(xlabel='Days left',ylabel='Price',title=f'{cls}: observed daily patterns')
    fig.tight_layout();fig.savefig(out/'days_class.png',dpi=150);plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,4.5))
    for name,group in imp_pred.groupby('method'):
        sorted_e=np.sort(np.abs(group.error));prob=np.arange(1,len(sorted_e)+1)/len(sorted_e)
        ax.plot(sorted_e,prob,label=name)
    ax.set(xlim=(0,8),xlabel='Absolute masked duration error (hours)',ylabel='Empirical cumulative fraction',title='Feature-only reconstruction, identical-row groups blocked');ax.legend()
    fig.tight_layout();fig.savefig(out/'duration_reconstruction.png',dpi=150);plt.close(fig)
    with (out/'training_summary.json').open('w') as f: json.dump(summary,f,indent=2,default=lambda x:x.item() if isinstance(x,np.generic) else str(x))
    report = f'''# Training-only structural audit\n\nThe only data source is labeled `train.csv` (SHA256 `{summary['data_sha256']}`). No Kaggle test, sample submission, previous prediction CSV, external data, or leaderboard data was read by this script.\n\nThis audit is descriptive and hypothesis-generating. Broad EDA using training labels is **not independent proof** of predictive improvement. Every learned imputer, category map, statistic, and encoding used in a predictive comparison must be fitted within that comparison's training partition. No predictive model was fitted in this script.\n\n## Reconciled data and ambiguity\n\nThere are {len(d):,} training observations. Target mean {d.price.mean():,.2f}, median {d.price.median():,.2f}, SD {d.price.std():,.2f}, range {d.price.min():,}–{d.price.max():,}. Airline missing {d.airline_missing.sum():,}; duration missing {d.duration_missing.sum():,}. Every observed prefix has exactly one airline label, so prefix fill is feature-redundancy recovery, not a new target signal.\n\n```json\n{json.dumps({k:v for k,v in summary.items() if 'duplicate' in k or 'structural_keys' in k},indent=2)}\n```\n\nThe within-key SSE fraction is a finite-sample observed ambiguity measure, **not a population irreducible-noise bound**. Grouping after arbitrary duration imputation would manufacture false exact matches. Filling deterministic airline redundancy can nevertheless identify semantic duplicates that raw missingness distinguishes.\n\n## Duration reconstruction (feature-only)\n\nObserved duration is masked in five group folds; all identical raw feature rows are blocked together. Fit-only medians use no `price`. This diagnostic measures reconstruction of observed values, not error on genuinely missing values (which could follow a different missingness mechanism), and does not establish original-price R² improvement.\n\n{markdown_table(imp_summary)}\n\n{markdown_table(pd.DataFrame(dispersion))}\n\nDeparture/arrival-aware keys distinguish itinerary variants that share the first flight identifier, route and stops. The more detailed hierarchy is a candidate only when masked accuracy improves consistently, and must be fitted fold-locally during model evaluation. No inference about Kaggle test missingness is made.\n\n## Heterogeneity and nonlinear effects\n\n{markdown_table(corr)}\n\nGlobal duration/price association is strongly confounded with class and prefix. Within Business, domestic-prefix associations differ from EK/EY/TG/UL; a global monotonic duration constraint is unsupported. Preserve raw observed duration and prefix/class interactions; broad numeric polynomials previously failed and should not be repeated.\n\n{markdown_table(summarize_group(d,['flight_prefix','flight_class']).sort_values('mean',ascending=False))}\n\nThe top-1% price cohort contains {len(top)} rows, all Business. Its prefix distribution is `{top.flight_prefix.value_counts().to_dict()}`. Sparse high-price prefix groups mean a few expensive examples can change pooled R² markedly. Rare-category collapse should therefore be audited: an uncommon airline may describe expensive structural conditions, not arbitrary noise. Avoid direct target-history encoding and price weighting already rejected in prior work.\n\n{markdown_table(day_summary)}\n\nDaily averages mix prefix/route composition and ticket selection, and do not justify causal monotonicity. `days_within_cell_patterns.csv` provides descriptive demeaning by prefix/class/route/stops; these full-training label summaries must never be used as model features. Existing trees already capture thresholds; additional lead-time transforms require isolated ablations rather than repetition of the failed feature bundle.\n\n## Recommendations and limits\n\n1. Compare the exact original XGB850 control with a single departure/arrival-aware duration hierarchy if masked reconstruction supports it. Preserve missingness indicators and raw duration. Add reliability/count features only as a separate justified ablation; no arbitrary target-conditioned imputation.\n2. Audit categorical representation and sparsity: keep interpretable prefix/class information and consider a bounded categorical model only if it addresses OHE rare-flight collapse. Avoid conflating a new model family, target transform and feature bundle.\n3. Use raw identical-row grouping as primary historical comparison and canonical airline-filled grouping as a sensitivity check. Structural keys that omit duration intentionally make a harder task and do not imply identical observations.\n4. Do not remove extreme observations or guarantee gains. No values violate basic positivity checks; high prices occur in recognizable but sparse groups. Validate original-price errors by prefix/class and top-tail contribution.\n5. A fresh fold seed does not create wholly untouched labels after broad historic/EDA use. Use nested inner selection and predeclared outer comparisons, label uncertainty honestly, and keep Kaggle inference until all choices are frozen.\n\nGenerated by `training_eda.py`; CSVs retain exact numerical diagnostics and four PNG figures visualize the evidence.\n'''
    (out/'TRAINING_ONLY_EDA.md').write_text(report)
    print(json.dumps({'summary':summary,'masked_duration':imp_summary.to_dict('records')},indent=2,default=str))

if __name__=='__main__': main()
