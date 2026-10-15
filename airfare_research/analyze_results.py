"""Analyze completed train-only OOF experiments; never fits models or opens test data.

Run: python analyze_results.py [--results results] [--out analysis] [--bootstrap 2000]
Only labeled training data, results/*/oof.csv, summary.json and fold_metrics.csv
are read during analysis. --form-registered-blend additionally checks the saved
BLEND_REGISTRATION.md and constructs its exact fixed average in results/.
Bootstrap intervals condition on saved OOF predictions; they do not
capture model retraining uncertainty or remove historical model-selection bias.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import math

ROOT = Path(__file__).resolve().parent
# Matplotlib's cache belongs to this analyzer's writable output directory.
os.environ.setdefault('MPLCONFIGDIR', str(ROOT/'analysis'/'matplotlib-cache'))
os.environ.setdefault('XDG_CACHE_HOME', str(ROOT/'analysis'/'cache'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DEFAULT_TRAIN = Path('/workspace/attachments/085aeb37-413d-42c9-bb16-baafd968488d/train.csv')
REGISTERED = {'native', 'duration_schedule', 'prefix_duration'}
ALPHA_ADJUSTED = .05 / 3


def save_json(path, value):
    def convert(x):
        if isinstance(x, np.generic):
            return x.item()
        if isinstance(x, Path):
            return str(x)
        raise TypeError(type(x).__name__)
    Path(path).write_text(json.dumps(value, indent=2, default=convert) + '\n')


def metrics(y, pred):
    y = np.asarray(y, dtype=float)
    pred = np.asarray(pred, dtype=float)
    e = pred - y
    sse = float(np.dot(e, e))
    sst = float(np.sum((y - y.mean()) ** 2))
    return dict(n=len(y), r2=(1. - sse / sst) if sst > 0 else np.nan,
                rmse=float(np.sqrt(np.mean(e ** 2))),
                mae=float(np.mean(np.abs(e))), bias=float(e.mean()), sse=sse)


def raw_groups(raw):
    return pd.util.hash_pandas_object(raw.astype(str), index=False).to_numpy()


def clustered_delta_ci(y, p0, p1, groups, n_boot=2000, seed=731):
    """Pairs stay together; denominator uses the resampled y sufficient statistics.

    A draw samples the observed number of clusters, with replacement, then
    includes every row in each sampled cluster. Unequal cluster sizes therefore
    change n as well as sums and variances. No fixed original denominator is used.
    """
    table = pd.DataFrame(dict(group=groups, n=1., y=y, yy=y*y,
                              sse0=(y-p0)**2, sse1=(y-p1)**2))
    stats = table.groupby('group', sort=False, dropna=False).sum()
    values = stats[['n', 'y', 'yy', 'sse0', 'sse1']].to_numpy(float)
    n_groups = len(values)
    rng = np.random.default_rng(seed)
    deltas = []
    # Modest batches bound memory even for one raw-record cluster per row.
    for start in range(0, n_boot, 32):
        idx = rng.integers(n_groups, size=(min(32, n_boot-start), n_groups))
        sums = np.empty((len(idx), values.shape[1]), dtype=float)
        for j in range(values.shape[1]):
            sums[:, j] = values[idx, j].sum(axis=1)
        denom = sums[:, 2] - sums[:, 1] ** 2 / sums[:, 0]
        valid = denom > 0
        deltas.extend(((sums[valid, 3]-sums[valid, 4])/denom[valid]).tolist())
    v = np.asarray(deltas)
    if len(v) == 0:
        raise ValueError('No bootstrap draw had positive target variance')
    return dict(n_clusters=n_groups, n_bootstrap=n_boot, valid_draws=len(v),
                ci95=[float(x) for x in np.quantile(v, [.025, .975])],
                ci98_333=[float(x) for x in np.quantile(v, [ALPHA_ADJUSTED/2, 1-ALPHA_ADJUSTED/2])],
                ci98_75=[float(x) for x in np.quantile(v, [.05/4/2, 1-.05/4/2])],
                ci99=[float(x) for x in np.quantile(v, [.05/5/2, 1-.05/5/2])],
                ci99_1667=[float(x) for x in np.quantile(v, [.05/6/2, 1-.05/6/2])],
                conditional_bootstrap_mean=float(v.mean()),
                conditional_bootstrap_sd=float(v.std(ddof=1)))


def subgroup_dimensions(raw, y, folds):
    freq = np.zeros(len(raw), dtype=int)
    for f in np.unique(folds):
        val = folds == f
        count = raw.loc[~val, 'flight'].value_counts()
        freq[val] = raw.loc[val, 'flight'].map(count).fillna(0).to_numpy(int)
    quantiles = np.quantile(y, [.5, .75, .9, .95, .99])
    # Duplicate target boundaries are allowed and collapse safely if necessary.
    edges = np.unique(np.r_[-np.inf, quantiles, np.inf])
    qlabels = [f'{a:g}..{b:g}' for a, b in zip(edges[:-1], edges[1:])]
    dims = {
        'flight_class': raw.flight_class.astype(str),
        'flight_prefix': raw.flight.astype(str).str.split().str[0],
        'route': raw.source_city.astype(str) + ' -> ' + raw.destination_city.astype(str),
        'stops': raw.stops.astype(str),
        'duration_missing': raw.duration.isna().astype(str),
        'airline_missing': raw.airline.isna().astype(str),
        'days_left': pd.cut(raw.days_left, [-np.inf, 3, 7, 14, 30, np.inf],
                            labels=['<=3','4..7','8..14','15..30','>30']).astype(str),
        'target_quantile_range': pd.cut(y, edges, labels=qlabels, include_lowest=True).astype(str),
        'target_decile': pd.qcut(pd.Series(y).rank(method='first'), 10, labels=False).astype(str),
        'training_fold_flight_frequency': pd.Series(np.select([freq == 0, freq < 12],
                                                     ['unseen', 'rare_1..11'], default='common_>=12')),
    }
    return {k: np.asarray(v) for k, v in dims.items()}, freq


def diagnostic_plot(run, groups, out):
    y, pred = run['y'], run['pred']
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes[0, 0].scatter(y, pred, s=3, alpha=.2)
    lim = max(float(y.max()), float(pred.max()))
    axes[0, 0].plot([0, lim], [0, lim], 'k--', lw=1)
    axes[0, 0].set(xlabel='Observed price', ylabel='OOF prediction', title='Original-price predictions')
    axes[0, 1].scatter(pred, pred-y, s=3, alpha=.2)
    axes[0, 1].axhline(0, color='k', linestyle='--', lw=1)
    axes[0, 1].set(xlabel='OOF prediction', ylabel='Prediction − actual', title='Residuals')
    axes[0, 2].hist(pred-y, bins=80)
    axes[0, 2].set(xlabel='Prediction − actual', ylabel='Rows', title='Residual distribution')
    target = groups[groups.dimension.eq('target_decile')].sort_values('group')
    axes[1, 0].bar(target['group'], target.rmse)
    axes[1, 0].set(xlabel='Training-price decile (0 low)', ylabel='RMSE', title='Errors by target decile')
    prefix = groups[groups.dimension.eq('flight_prefix')].nlargest(10, 'sse_share')
    axes[1, 1].bar(prefix['group'], prefix.sse_share)
    axes[1, 1].set(xlabel='Carrier prefix', ylabel='Share of total squared error', title='Largest carrier error shares')
    missing = groups[groups.dimension.eq('duration_missing')]
    axes[1, 2].bar(missing['group'], missing.rmse)
    axes[1, 2].set(xlabel='Duration missing', ylabel='RMSE', title='Missing-duration diagnosis')
    fig.suptitle(f"{run['name']}: pooled OOF R² {metrics(y, pred)['r2']:.6f}")
    fig.tight_layout()
    fig.savefig(out / f"{run['name']}_diagnostics.png", dpi=130)
    plt.close(fig)


def load_runs(results, train, y):
    runs = []
    skipped = []
    for path in sorted(results.glob('*/summary.json')):
        oof_path = path.parent / 'oof.csv'
        fold_path = path.parent / 'fold_metrics.csv'
        if not oof_path.exists():
            skipped.append(path.parent.name)
            continue
        summary = json.loads(path.read_text())
        frame = pd.read_csv(oof_path).sort_values('row_position').reset_index(drop=True)
        if len(frame) != len(y) or not np.array_equal(frame.row_position, np.arange(len(y))):
            raise ValueError(f'Row positions do not cover training data: {oof_path}')
        if not np.array_equal(frame.id.to_numpy(), train.id.to_numpy()):
            raise ValueError(f'ID alignment error: {oof_path}')
        if not np.array_equal(frame.price.to_numpy(float), y):
            raise ValueError(f'Labels differ from supplied training labels: {oof_path}')
        pred = frame.prediction.to_numpy(float)
        if not np.isfinite(pred).all() or (frame.fold < 0).any():
            raise ValueError(f'Incomplete/invalid OOF predictions: {oof_path}')
        foldmetrics = pd.read_csv(fold_path) if fold_path.exists() else pd.DataFrame(summary['folds'])
        runs.append(dict(name=path.parent.name, summary=summary, y=y, pred=pred,
                         folds=frame.fold.to_numpy(int), foldmetrics=foldmetrics))
    return runs, skipped


def form_registered_blend(results):
    """Materialize the explicitly authorized 50/50 average; no weight search.

    No fitted blend training predictions exist, so train R² is null. This
    function writes only this named blend's OOF artifacts, never constituent files.
    """
    registration = ROOT/'BLEND_REGISTRATION.md'
    if not registration.exists():
        raise ValueError('The required pre-score blend registration is missing')
    registration_hash = hashlib.sha256(registration.read_bytes()).hexdigest()
    for seed in [20261015, 20261016]:
        a_dir = results/f'raw_{seed}_baseline'
        b_dir = results/f'raw_{seed}_native'
        if not all((p/q).exists() for p in [a_dir,b_dir] for q in ['oof.csv','summary.json','fold_metrics.csv']):
            continue
        sa, sb = [json.loads((p/'summary.json').read_text()) for p in [a_dir,b_dir]]
        if any(s.get('early_stopping', False) or s['scheme'] != 'raw' or s['seed'] != seed for s in [sa,sb]):
            raise ValueError('Blend constituents must be the registered fixed850 raw-fold procedures')
        a, b = [pd.read_csv(p/'oof.csv').sort_values('row_position').reset_index(drop=True) for p in [a_dir,b_dir]]
        for c in ['row_position','id','price','fold']:
            if not np.array_equal(a[c].to_numpy(), b[c].to_numpy()):
                raise ValueError(f'Blend constituent alignment mismatch: {c}')
        params = [s.get('reference_parameters', s.get('parameters', {})) for s in [sa,sb]]
        if any(int(p['n_estimators']) != 850 for p in params):
            raise ValueError('Registered constituents must each use exactly850 trees')
        af, bf = [pd.read_csv(p/'fold_metrics.csv') for p in [a_dir,b_dir]]
        if any(not x['rounds'].eq(850).all() for x in [af,bf]):
            raise ValueError('A constituent has fold rounds other than850')
        y = a.price.to_numpy(float)
        a['prediction'] = .5*a.prediction.to_numpy(float)+.5*b.prediction.to_numpy(float)
        pred = a.prediction.to_numpy(float)
        foldrows = []
        for f in sorted(a.fold.unique()):
            mask = a.fold.eq(f).to_numpy()
            saved = af.loc[af.fold.eq(f)].iloc[0]
            foldrows.append(dict(fold=int(f), rounds=850, constituent_rounds=[850,850],
                n_train=int(saved.n_train), n_validation=int(mask.sum()), train_r2=None,
                inner_stopping_rounds=[], seconds=0., **metrics(y[mask],pred[mask])))
        name = f'raw_{seed}_blend_equal850'
        dest = results/name
        dest.mkdir(exist_ok=True,parents=True)
        a.to_csv(dest/'oof.csv',index=False)
        pd.DataFrame(foldrows).to_csv(dest/'fold_metrics.csv',index=False)
        summary = dict(mode='blend_equal850',scheme='raw',seed=seed,early_stopping=False,
            **metrics(y,pred),mean_train_r2=None, mean_train_validation_gap=None,
            mean_fold_r2=float(np.mean([r['r2'] for r in foldrows])),
            fold_r2_std=float(np.std([r['r2'] for r in foldrows],ddof=1)),
            worst_fold_r2=float(min(r['r2'] for r in foldrows)), folds=foldrows,
            parameters=dict(weights=[.5,.5],constituents=[a_dir.name,b_dir.name],
                            constituent_rounds=[850,850],total_trees=1700),
            round_policy='each constituent fixed850',feature_mode='blend_equal850',total_seconds=0.,
            train_metric_status='Unavailable: no blend training predictions saved; not approximated from constituent R².',
            registration_sha256=registration_hash,
            constituent_oof_sha256={p.name:hashlib.sha256((p/'oof.csv').read_bytes()).hexdigest() for p in [a_dir,b_dir]})
        save_json(dest/'summary.json',summary)
        print(f'Formed registered fixed50/50 OOF blend: {dest}',flush=True)


def analyze(train_path, results, out, n_boot=2000):
    if n_boot < 100:
        raise ValueError('Use at least 100 bootstrap draws; 2000 is recommended')
    # This is the sole competition CSV input to the analyzer.
    train = pd.read_csv(train_path)
    raw = train.drop(columns=['id', 'price'])
    y = train.price.to_numpy(float)
    raw_cluster = raw_groups(raw)
    flight_cluster = raw.flight.astype(str).to_numpy()
    out.mkdir(parents=True, exist_ok=True)
    plots = out / 'plots'
    plots.mkdir(exist_ok=True)
    runs, skipped = load_runs(results, train, y)
    pooled, foldrows, subrows, worst, compositions = [], [], [], [], []
    for run in runs:
        name, pred, folds, summary = run['name'], run['pred'], run['folds'], run['summary']
        met = metrics(y, pred)
        squared = (pred-y)**2
        tail = y >= np.quantile(y, .99)
        worst1 = np.argsort(squared)[-max(1, math.ceil(len(y)*.01)):]
        dims, flight_frequency = subgroup_dimensions(raw, y, folds)
        fm = run['foldmetrics']
        tr_mean = float(fm.train_r2.mean()) if 'train_r2' in fm else np.nan
        local_folds = []
        for f in np.unique(folds):
            mask = folds == f
            row = dict(model=name, fold=int(f), **metrics(y[mask], pred[mask]))
            saved = fm.loc[fm.fold.eq(f)]
            if len(saved):
                for c in ['train_r2', 'rounds', 'n_train', 'seconds', 'inner_stopping_rounds']:
                    if c in saved:
                        row[c] = saved.iloc[0][c]
            row['train_validation_gap'] = row.get('train_r2', np.nan)-row['r2']
            local_folds.append(row)
            comp = dict(model=name, fold=int(f), n=int(mask.sum()),
                        target_mean=float(y[mask].mean()), target_std=float(y[mask].std(ddof=1)),
                        business_n=int(np.sum(mask & raw.flight_class.eq('Business').to_numpy())),
                        top1_target_n=int(np.sum(mask & tail)),
                        unseen_flights_n=int(np.sum(mask & (flight_frequency == 0))))
            for dec in range(10):
                comp[f'target_decile_{dec}_n'] = int(np.sum(mask & (dims['target_decile'] == str(dec))))
            for prefix in ['EY', 'EK', 'QR', 'SQ', 'GF']:
                comp[f'prefix_{prefix}_n'] = int(np.sum(mask & (dims['flight_prefix'] == prefix)))
            compositions.append(comp)
        foldrows.extend(local_folds)
        foldr2 = np.array([r['r2'] for r in local_folds])
        pooled.append(dict(model=name, mode=summary['mode'], scheme=summary['scheme'], seed=summary['seed'],
                           **met, mean_fold_r2=float(foldr2.mean()),
                           fold_r2_std=float(foldr2.std(ddof=1)), worst_fold_r2=float(foldr2.min()),
                           mean_train_r2=tr_mean, mean_train_validation_gap=tr_mean-float(foldr2.mean()),
                           top1_target_n=int(tail.sum()), top1_target_rmse=metrics(y[tail], pred[tail])['rmse'],
                           top1_target_bias=float((pred-y)[tail].mean()),
                           top1_target_sse_share=float(squared[tail].sum()/squared.sum()),
                           top1_error_n=len(worst1), top1_error_sse_share=float(squared[worst1].sum()/squared.sum()),
                           min_rounds=int(fm['rounds'].min()) if 'rounds' in fm else np.nan,
                           max_rounds=int(fm['rounds'].max()) if 'rounds' in fm else np.nan,
                           fit_seconds=float(fm.seconds.sum()) if 'seconds' in fm else np.nan))
        run_sub = []
        for dim, group in dims.items():
            for label in np.unique(group):
                mask = group == label
                record = dict(model=name, dimension=dim, group=str(label),
                              **metrics(y[mask], pred[mask]), sse_share=float(squared[mask].sum()/squared.sum()))
                run_sub.append(record)
        subrows.extend(run_sub)
        top = np.argsort(squared)[-30:][::-1]
        cases = train.iloc[top].copy()
        cases.insert(0, 'row_position', top)
        cases.insert(0, 'model', name)
        cases['fold'] = folds[top]
        cases['prediction'] = pred[top]
        cases['residual_prediction_minus_actual'] = pred[top]-y[top]
        cases['squared_error'] = squared[top]
        cases['fold_training_flight_count'] = flight_frequency[top]
        worst.append(cases)
        diagnostic_plot(run, pd.DataFrame(run_sub), plots)

    by_key = {(r['summary']['scheme'], r['summary']['seed']): r for r in runs
              if r['summary']['mode'] == 'baseline' and not r['summary'].get('early_stopping', False)}
    paired, pairfolds, bootstrap, complementarity = [], [], {}, []
    for run in runs:
        s = run['summary']
        if s['mode'] == 'baseline' and not s.get('early_stopping', False):
            continue
        base = by_key.get((s['scheme'], s['seed']))
        if base is None:
            continue
        if not np.array_equal(run['folds'], base['folds']):
            raise ValueError(f"Candidate/baseline fold mismatch: {run['name']}")
        delta = metrics(y, run['pred'])['r2']-metrics(y, base['pred'])['r2']
        if s['mode'] == 'native' and not s.get('early_stopping', False):
            e0, e1 = base['pred']-y, run['pred']-y
            masks = {'all_rows': np.ones(len(y), dtype=bool),
                     'Business': raw.flight_class.eq('Business').to_numpy(),
                     'Economy': raw.flight_class.eq('Economy').to_numpy(),
                     'highest_price_1pct': y >= np.quantile(y, .99)}
            for subgroup, mask in masks.items():
                a, b = e0[mask], e1[mask]
                complementarity.append(dict(model=run['name'], baseline=base['name'], subgroup=subgroup,
                    n=int(mask.sum()), residual_correlation=float(np.corrcoef(a,b)[0,1]),
                    residual_covariance=float(np.cov(a,b,ddof=1)[0,1]),
                    baseline_residual_variance=float(a.var(ddof=1)),
                    candidate_residual_variance=float(b.var(ddof=1)),
                    baseline_bias=float(a.mean()), candidate_bias=float(b.mean()),
                    opposite_error_sign_fraction=float(np.mean(a*b<0))))
        positives = 0
        for f in np.unique(run['folds']):
            mask = run['folds'] == f
            d = metrics(y[mask], run['pred'][mask])['r2']-metrics(y[mask], base['pred'][mask])['r2']
            positives += int(d > 0)
            pairfolds.append(dict(model=run['name'], baseline=base['name'], fold=int(f), delta_r2=d))
        intervals = {}
        for cluster_name, cluster in [('raw_duplicate', raw_cluster), ('flight', flight_cluster)]:
            intervals[cluster_name] = clustered_delta_ci(y, base['pred'], run['pred'], cluster, n_boot)
        bootstrap[run['name']] = dict(baseline=base['name'], delta_r2=delta, clusters=intervals)
        row = dict(model=run['name'], baseline=base['name'], mode=s['mode'], scheme=s['scheme'], seed=s['seed'],
                   delta_r2=delta, positive_folds=positives, n_folds=len(np.unique(run['folds'])),
                   registered_primary_hypothesis=s['mode'] in REGISTERED and not s.get('early_stopping', False),
                   exploratory_followup=s['mode'] not in REGISTERED or s.get('early_stopping', False))
        for cname, ci in intervals.items():
            row[f'{cname}_n_clusters'] = ci['n_clusters']
            for level in ['ci95', 'ci98_333', 'ci98_75', 'ci99', 'ci99_1667']:
                row[f'{cname}_{level}_lower'], row[f'{cname}_{level}_upper'] = ci[level]
        row['passes_primary_numeric_gate'] = bool(delta >= .001 and positives >= 4 and intervals['raw_duplicate']['ci95'][0] > 0)
        for family_size, level in [(3,'ci98_333'),(4,'ci98_75'),(5,'ci99'),(6,'ci99_1667')]:
            row[f'passes_family{family_size}_bonferroni_numeric_gate'] = bool(
                row['passes_primary_numeric_gate'] and intervals['raw_duplicate'][level][0] > 0)
        row['passes_expanded_family6_numeric_gate'] = row['passes_family6_bonferroni_numeric_gate']
        row['repeat_confirmation_required'] = True
        paired.append(row)

    pd.DataFrame(pooled).to_csv(out/'model_metrics.csv', index=False)
    pd.DataFrame(foldrows).to_csv(out/'fold_metrics.csv', index=False)
    pd.DataFrame(subrows).to_csv(out/'subgroup_metrics.csv', index=False)
    pd.DataFrame(compositions).to_csv(out/'fold_composition.csv', index=False)
    pd.concat(worst, ignore_index=True).to_csv(out/'worst_errors.csv', index=False) if worst else None
    pd.DataFrame(paired).to_csv(out/'paired_comparisons.csv', index=False)
    pd.DataFrame(pairfolds).to_csv(out/'paired_fold_deltas.csv', index=False)
    pd.DataFrame(complementarity).to_csv(out/'residual_complementarity.csv', index=False)
    save_json(out/'paired_bootstrap.json', bootstrap)
    save_json(out/'analysis_metadata.json', dict(train_path=str(train_path), results_path=str(results),
        completed_runs=[r['name'] for r in runs], skipped_incomplete=skipped,
        bootstrap_draws=n_boot, bootstrap_seed=731, multiplicity_hypotheses=3,
        exploratory_multiplicity_sensitivity=[4,5,6],
        bonferroni_interval_coverage=1-ALPHA_ADJUSTED,
        limitations='Cluster bootstrap conditions on saved OOF fits; no retraining, selection, or independent-population uncertainty guarantee.',
        early_round_metadata='Actual fold rounds override any summary.parameters.n_estimators=850 default in early-stopped runs.'))

    report = ['# Training-only experiment diagnostics', '',
              f'Completed OOF procedures: {len(runs)}. Each row and label was checked against the supplied labeled training CSV. No Kaggle test/sample/preserved predictions were read.', '',
              '## Measured pooled performance', '',
              '| Procedure | Pooled R² | RMSE | MAE | Mean train−validation R² | Top-price 1% SSE share |',
              '|---|---:|---:|---:|---:|---:|']
    for r in sorted(pooled, key=lambda x: (str(x['scheme']), x['seed'], -x['r2'])):
        gap = f"{r['mean_train_validation_gap']:.5f}" if np.isfinite(r['mean_train_validation_gap']) else 'Unavailable'
        report.append(f"| {r['model']} | {r['r2']:.6f} | {r['rmse']:,.1f} | {r['mae']:,.1f} | {gap} | {r['top1_target_sse_share']:.1%} |")
    report += ['', 'Results from different fold schemes are sensitivity analyses; their differences are not candidate improvement estimates.', '',
               '## Paired comparisons on identical folds', '',
               '| Candidate | Δ R² | Positive folds | Raw-group 95% interval | Raw-group Bonferroni 98.333% interval | Flight-cluster 95% interval | Numeric promotion gate |',
               '|---|---:|---:|---|---|---|---|']
    for r in paired:
        def ci(c, level):
            return f"[{r[f'{c}_{level}_lower']:+.6f}, {r[f'{c}_{level}_upper']:+.6f}]"
        report.append(f"| {r['model']} | {r['delta_r2']:+.6f} | {r['positive_folds']}/{r['n_folds']} | {ci('raw_duplicate','ci95')} | {ci('raw_duplicate','ci98_333')} | {ci('flight','ci95')} | {'Pass' if r['passes_primary_numeric_gate'] else 'Fail'} |")
    report += ['', 'The numeric gate requires Δ R² ≥0.001, at least four positive folds, and a positive raw-duplicate-group 95% lower bound. A passing primary comparison still requires the predeclared unchanged repeated-CV confirmation and subgroup review; this script does not select or fit a model. Family-three adjusted-gate flags describe only that registered primary sensitivity; expanded stopping/blending comparisons additionally carry a family-six strict sensitivity flag.', '',
               '## Concrete findings from completed procedures', '']
    primary = next((r for r in pooled if r['model'] == 'raw_20261015_baseline'), None)
    if primary:
        baseline_sub = [r for r in subrows if r['model'] == primary['model']]
        business = next(r for r in baseline_sub if r['dimension'] == 'flight_class' and r['group'] == 'Business')
        report += [f"The primary fixed850 control has pooled R² {primary['r2']:.6f}. Business rows contribute {business['sse_share']:.1%} of its squared error; the highest-price 1% contribute {primary['top1_target_sse_share']:.1%}. The largest-error 1% (a different set of rows) contribute {primary['top1_error_sse_share']:.1%}. Tail bias is {primary['top1_target_bias']:,.0f} price units.", '']
    for r in paired:
        if r['seed'] != 20261015 or r['scheme'] != 'raw':
            continue
        why = []
        if r['delta_r2'] < .001:
            why.append('material-effect threshold')
        if r['positive_folds'] < 4:
            why.append('four-positive-fold threshold')
        if r['raw_duplicate_ci95_lower'] <= 0:
            why.append('positive raw-group confidence bound')
        status = 'Fails '+', '.join(why)+'.' if why else 'Passes the initial numeric thresholds; repeat confirmation is still needed.'
        report += [f"- `{r['model']}`: Δ R² {r['delta_r2']:+.6f}, {r['positive_folds']}/{r['n_folds']} positive folds. {status}"]
    report += ['',
               '## Residual complementarity', '',
               'No blend weights or blend predictions are computed by this section. Native850/control residual covariance and correlations are saved to `residual_complementarity.csv` for all rows, cabins, and expensive-target rows. Covariance below both constituent residual variances suggests a fixed average may reduce variance; it does not establish a total squared-error gain in the presence of bias or justify a submission without the predeclared confirmation.', '',
               '## Error and validation interpretation', '',
               '- `subgroup_metrics.csv` includes class, prefix, route, stops, missingness, booking horizon, target percentiles/deciles, and flight frequency measured within each corresponding training fold. Unseen and rare flights are separate.',
               '- `top1_target_sse_share` selects the highest actual-price 1%; `top1_error_sse_share` selects the largest squared-error 1%. They answer different questions. Their exact selected row counts are saved.',
               '- Fold composition includes target-decile, rare expensive-prefix, Business, and extreme-target counts. A percentile-specific R² can be negative or unstable because target variance within that narrow range is small; use RMSE/bias for those diagnostics.',
               '- Native representation may reduce MAE while worsening R²; the official original-price squared-error objective decides promotion. Error concentration and train–validation gaps help diagnose the tradeoff.',
               '- Each cluster bootstrap resamples whole groups, recomputes resampled n/sum(y)/sum(y²), and uses the resulting denominator for the paired R² difference. Raw-duplicate and flight-cluster intervals assess different dependence assumptions.',
               '- These are conditional intervals on existing OOF predictions. Folds/repeats overlap in training data; neither bootstrap nor repeated CV captures full retraining uncertainty or erases prior selection on this labeled dataset. No reused Phase5 holdout is called independent proof.',
               '- Bonferroni intervals use family size three (98.333% coverage), as registered, even if fewer candidates have finished. Sensitivity intervals for family sizes four (98.75%), five (99%), and six (99.1667%) also appear in the comparison CSV/JSON when stopping/blending procedures expand the exploratory family. They supplement the predeclared 95% promotion criterion. With 2,000 draws, extreme adjusted tails use relatively few draws; treat those quantiles as approximate.',
               '- For early-stopped procedures use actual per-fold round counts in `fold_metrics.csv`; an older summary parameters dictionary can retain the default 850 even when the fitting policy differs.',
               '- Historical public Kaggle 0.95797 is user-reported and concerns another sample. No hidden-test score or rank improvement is measured here.', '',
               'All artifacts can be regenerated by rerunning this analyzer as additional completed OOF results arrive.']
    (out/'FINDINGS.md').write_text('\n'.join(report)+'\n')
    print(json.dumps(dict(completed_runs=len(runs), comparisons=len(paired), output=str(out)), indent=2))
    return pooled, paired


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--train', type=Path, default=DEFAULT_TRAIN)
    p.add_argument('--results', type=Path, default=ROOT/'results')
    p.add_argument('--out', type=Path, default=ROOT/'analysis')
    p.add_argument('--bootstrap', type=int, default=2000)
    p.add_argument('--form-registered-blend', action='store_true')
    a = p.parse_args()
    if a.form_registered_blend:
        form_registered_blend(a.results)
    analyze(a.train, a.results, a.out, a.bootstrap)


if __name__ == '__main__':
    main()
