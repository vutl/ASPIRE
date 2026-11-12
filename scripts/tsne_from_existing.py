#!/usr/bin/env python3
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE

results = Path('results/compare_full_vs_noav')

def load_prefix(prefix):
    arrs = []
    for p in sorted(results.glob(prefix + '_fold*.npy')):
        try:
            arrs.append(np.load(p))
        except Exception:
            continue
    if not arrs:
        return None
    return np.concatenate(arrs, axis=0)

def run_tsne(X, random_state=42):
    if X is None:
        return None
    tsne = TSNE(n_components=2, random_state=random_state)
    return tsne.fit_transform(X)

def plot(Z, labels, outname, discrete=True):
    plt.figure(figsize=(6,5))
    if discrete:
        sns.scatterplot(x=Z[:,0], y=Z[:,1], hue=labels, palette='tab10', s=10, linewidth=0)
        plt.legend(bbox_to_anchor=(1.05,1), loc='upper left')
    else:
        plt.scatter(Z[:,0], Z[:,1], c=labels, cmap='RdYlBu', s=8)
        plt.colorbar()
    plt.tight_layout()
    plt.savefig(str(outname) + '.pdf')
    plt.close()

def main():
    fused_full = load_prefix('fused_full')
    fused_noav = load_prefix('fused_noav')
    ar_full = load_prefix('arousal_full')
    ar_noav = load_prefix('arousal_noav')
    va_full = load_prefix('valence_full')
    va_noav = load_prefix('valence_noav')

    # For discrete coloring use per_class_rates if available; else cluster
    labels = None
    per_csv = results / 'per_class_rates.csv'
    if per_csv.exists():
        try:
            import pandas as pd
            df = pd.read_csv(per_csv)
            # cannot recover per-sample labels; fallback to clusters
        except Exception:
            pass

    from sklearn.cluster import KMeans

    if fused_full is not None:
        Z = run_tsne(fused_full)
        k = min(4, fused_full.shape[0])
        km = KMeans(n_clusters=k, random_state=0).fit_predict(fused_full)
        plot(Z, km, results / 'tsne_fused_full')

    if fused_noav is not None:
        Z = run_tsne(fused_noav)
        k = min(4, fused_noav.shape[0])
        km = KMeans(n_clusters=k, random_state=0).fit_predict(fused_noav)
        plot(Z, km, results / 'tsne_fused_noav')

    if ar_full is not None:
        Z = run_tsne(ar_full)
        vals = ar_full.mean(axis=1) if ar_full.ndim>1 else ar_full
        plot(Z, vals, results / 'tsne_arousal_full', discrete=False)

    if ar_noav is not None:
        Z = run_tsne(ar_noav)
        vals = ar_noav.mean(axis=1) if ar_noav.ndim>1 else ar_noav
        plot(Z, vals, results / 'tsne_arousal_noav', discrete=False)

    if va_full is not None:
        Z = run_tsne(va_full)
        vals = va_full.mean(axis=1) if va_full.ndim>1 else va_full
        plot(Z, vals, results / 'tsne_valence_full', discrete=False)

    if va_noav is not None:
        Z = run_tsne(va_noav)
        vals = va_noav.mean(axis=1) if va_noav.ndim>1 else va_noav
        plot(Z, vals, results / 'tsne_valence_noav', discrete=False)

    print('t-SNE plots saved to', results)

if __name__ == '__main__':
    main()
