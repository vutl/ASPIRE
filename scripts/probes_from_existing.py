#!/usr/bin/env python3
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split

def compute_ccc(y_true, y_pred):
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()
    mean_t, mean_p = y_true.mean(), y_pred.mean()
    var_t, var_p = y_true.var(), y_pred.var()
    cov = np.mean((y_true-mean_t)*(y_pred-mean_p))
    return float(2*cov / (var_t + var_p + (mean_t-mean_p)**2 + 1e-8))

def load_fold_arrays(outdir, prefix, folds):
    arrs = []
    for f in folds:
        p = outdir / f'{prefix}_fold{f}.npy'
        if p.exists():
            arrs.append(np.load(p))
    if not arrs:
        return None
    return np.concatenate(arrs, axis=0)

def gather_labels(csv_path, folds):
    df = pd.read_csv(csv_path)
    labels_ar = []
    labels_va = []
    for f in folds:
        max_fold = df['fold'].max()
        if max_fold == 10:
            fold_to_session = {1:1,2:1,3:2,4:2,5:3,6:3,7:4,8:4,9:5,10:5}
            sel = df[df['fold'].map(fold_to_session)==f]
        else:
            sel = df[df['fold']==f]
        labels_ar.append(sel['arousal'].values)
        labels_va.append(sel['valence'].values)
    if not labels_ar:
        return None, None
    return np.concatenate(labels_ar, axis=0), np.concatenate(labels_va, axis=0)

def fit_and_eval(X, y, out_prefix):
    if X is None or y is None or len(y)==0:
        return None
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    reg = Ridge(alpha=1.0)
    reg.fit(X_train, y_train)
    y_pred = reg.predict(X_test)
    mse = float(((y_pred - y_test)**2).mean())
    ccc = compute_ccc(y_test, y_pred)
    # scatter
    plt.figure(figsize=(5,5))
    plt.scatter(y_test, y_pred, s=6, alpha=0.6)
    mn = min(y_test.min(), y_pred.min())
    mx = max(y_test.max(), y_pred.max())
    plt.plot([mn,mx],[mn,mx],'k--',linewidth=0.8)
    plt.xlabel('True')
    plt.ylabel('Pred')
    plt.title(f'{out_prefix} (MSE={mse:.4f}, CCC={ccc:.4f})')
    plt.tight_layout()
    plt.savefig(str(out_prefix) + '.pdf')
    plt.close()
    return {'mse': mse, 'ccc': ccc, 'n_test': int(len(y_test))}

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--results', default='results/compare_full_vs_noav')
    p.add_argument('--csv', default='datasets/IEMOCAP_KFOLD/all.csv')
    p.add_argument('--folds', nargs='+', type=int, default=[1,2,3,4,5])
    args = p.parse_args()

    out = Path(args.results)
    folds = args.folds

    ar_full = load_fold_arrays(out, 'arousal_full', folds)
    ar_noav = load_fold_arrays(out, 'arousal_noav', folds)
    va_full = load_fold_arrays(out, 'valence_full', folds)
    va_noav = load_fold_arrays(out, 'valence_noav', folds)

    y_ar, y_va = gather_labels(args.csv, folds)

    # align lengths
    def align(X, y):
        if X is None or y is None:
            return X, y
        n = min(len(X), len(y))
        return X[:n], y[:n]

    ar_full, y_ar = align(ar_full, y_ar)
    ar_noav, _ = align(ar_noav, y_ar)
    va_full, y_va = align(va_full, y_va)
    va_noav, _ = align(va_noav, y_va)

    results = {}

    res = fit_and_eval(ar_full, y_ar, out / 'probe_arousal_full')
    results['arousal_full'] = res
    res = fit_and_eval(ar_noav, y_ar, out / 'probe_arousal_noav')
    results['arousal_noav'] = res

    res = fit_and_eval(va_full, y_va, out / 'probe_valence_full')
    results['valence_full'] = res
    res = fit_and_eval(va_noav, y_va, out / 'probe_valence_noav')
    results['valence_noav'] = res

    with open(out / 'probe_results.json', 'w') as fh:
        json.dump(results, fh, indent=2)

    print('Probes finished, results saved to', out / 'probe_results.json')

if __name__ == '__main__':
    main()
