"""Aggregate runs/*/summary.json into the paper's comparison (Tables III-IV style).

    python scripts/summarize.py --runs ./runs [--dataset agedb] [--model resnet18]

Searches --runs recursively, so it works on ./runs (all datasets) or ./runs/agedb (one dataset).

Reports mean +- std of test MAE and ADC for normal, two-stage and skewed, then paired
Wilcoxon signed-rank and paired t-tests between two-stage and skewed, matching runs by seed.
"""
import argparse
import glob
import json
import os

import numpy as np
import pandas as pd
from scipy import stats


def load(runs_dir, dataset=None, model=None):
    rows = []
    for path in glob.glob(os.path.join(runs_dir, '**', 'summary.json'), recursive=True):
        with open(path) as f:
            s = json.load(f)
        s.setdefault('dataset', 'agedb')  # runs from before the --dataset option
        s.setdefault('regularizer', 'weight_decay')  # runs from before --l2sp
        if dataset and s['dataset'] != dataset:
            continue
        if model and s['model'] != model:
            continue
        for method, r in s['test'].items():
            rows.append({'dataset': s['dataset'], 'model': s['model'], 'reg': s['regularizer'], 'method': method, 'seed': s['seed'],
                         'mae': r['mae'], 'adc': r['adc'], 'run': s['run_name']})
    return pd.DataFrame(rows)


def paired_tests(df, a='two_stage', b='skewed'):
    out = []
    for (dataset, model, reg), g in df.groupby(['dataset', 'model', 'reg']):
        pa = g[g.method == a].set_index('seed')
        pb = g[g.method == b].set_index('seed')
        seeds = pa.index.intersection(pb.index)
        for metric in ('mae', 'adc'):
            x, y = pa.loc[seeds, metric].to_numpy(), pb.loc[seeds, metric].to_numpy()
            row = {'dataset': dataset, 'model': model, 'reg': reg, 'metric': metric, 'n_pairs': len(seeds),
                   'wilcoxon_p': np.nan, 'paired_t_p': np.nan}
            if len(seeds) >= 2 and np.any(x != y):
                row['wilcoxon_p'] = stats.wilcoxon(x, y).pvalue
                row['paired_t_p'] = stats.ttest_rel(x, y).pvalue
            out.append(row)
    return pd.DataFrame(out)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--runs', default='./runs')
    p.add_argument('--dataset', default=None)
    p.add_argument('--model', default=None)
    p.add_argument('--out', default=None, help='optional CSV path for the per-run table')
    args = p.parse_args()

    df = load(args.runs, args.dataset, args.model)
    if df.empty:
        print('no summary.json files found')
        return
    order = {'normal': 0, 'two_stage': 1, 'skewed': 2}
    agg = (df.groupby(['dataset', 'model', 'reg', 'method'])
             .agg(n=('mae', 'size'), mae_mean=('mae', 'mean'), mae_std=('mae', 'std'),
                  adc_mean=('adc', 'mean'), adc_std=('adc', 'std'))
             .reset_index())
    agg = agg.sort_values(['dataset', 'model', 'reg', 'method'], key=lambda c: c.map(order) if c.name == 'method' else c)
    print('\nTest-set performance (mean +- std over seeds)')
    for _, r in agg.iterrows():
        print(f"  {r.dataset:10s} {r.model:9s} {r.reg:12s} {r.method:9s} n={r.n:<3d} MAE {r.mae_mean:6.3f} +- {np.nan_to_num(r.mae_std):.3f}"
              f"   ADC {r.adc_mean:+.3f} +- {np.nan_to_num(r.adc_std):.3f}")
    tests = paired_tests(df)
    if not tests.empty:
        print('\nTwo-stage vs skewed, paired by seed (p-values)')
        print(tests.to_string(index=False, float_format=lambda v: f'{v:.4f}'))
    if args.out:
        df.to_csv(args.out, index=False)


if __name__ == '__main__':
    main()
