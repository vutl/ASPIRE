#!/usr/bin/env python3
import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
try:
    from umap import UMAP
    _HAS_UMAP = True
except Exception:
    from sklearn.manifold import TSNE
    _HAS_UMAP = False

results = Path('results/compare_full_vs_noav')
out = results
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

def run_umap(X):
    if X is None:
        return None
    if _HAS_UMAP:
        reducer = UMAP(n_components=2, random_state=42)
        return reducer.fit_transform(X)
    else:
        from sklearn.manifold import TSNE
        tsne = TSNE(n_components=2, random_state=42)
        return tsne.fit_transform(X)

def plot(Z, labels, fname, discrete=True):
    plt.figure(figsize=(6,5))
    if discrete:
        sns.scatterplot(x=Z[:,0], y=Z[:,1], hue=labels, palette='tab10', s=10, linewidth=0)
        plt.legend(bbox_to_anchor=(1.05,1), loc='upper left')
    else:
        plt.scatter(Z[:,0], Z[:,1], c=labels, cmap='RdYlBu', s=8)
        plt.colorbar()
    plt.tight_layout()
    plt.savefig(str(fname) + '.pdf')
    plt.close()

fused_full = load_prefix('fused_full')
fused_noav = load_prefix('fused_noav')
ar_full = load_prefix('arousal_full')
ar_noav = load_prefix('arousal_noav')
va_full = load_prefix('valence_full')
va_noav = load_prefix('valence_noav')

from sklearn.cluster import KMeans

if fused_full is not None:
    Z = run_umap(fused_full)
    # if no per-sample labels available, color by KMeans clusters (k=4)
    try:
        k = min(4, fused_full.shape[0])
        km = KMeans(n_clusters=k, random_state=0).fit_predict(fused_full)
        plot(Z, km, results / 'umap_fused_full')
    except Exception:
        plot(Z, np.zeros(Z.shape[0], dtype=int), results / 'umap_fused_full')

if fused_noav is not None:
    Z = run_umap(fused_noav)
    try:
        k = min(4, fused_noav.shape[0])
        km = KMeans(n_clusters=k, random_state=0).fit_predict(fused_noav)
        plot(Z, km, results / 'umap_fused_noav')
    except Exception:
        plot(Z, np.zeros(Z.shape[0], dtype=int), results / 'umap_fused_noav')

if ar_full is not None:
    Z = run_umap(ar_full)
    vals = np.mean(ar_full, axis=1) if ar_full.ndim>1 else ar_full
    plot(Z, vals, results / 'umap_arousal_full', discrete=False)

if ar_noav is not None:
    Z = run_umap(ar_noav)
    vals = np.mean(ar_noav, axis=1) if ar_noav.ndim>1 else ar_noav
    plot(Z, vals, results / 'umap_arousal_noav', discrete=False)

if va_full is not None:
    Z = run_umap(va_full)
    vals = np.mean(va_full, axis=1) if va_full.ndim>1 else va_full
    plot(Z, vals, results / 'umap_valence_full', discrete=False)

if va_noav is not None:
    Z = run_umap(va_noav)
    vals = np.mean(va_noav, axis=1) if va_noav.ndim>1 else va_noav
    plot(Z, vals, results / 'umap_valence_noav', discrete=False)

print('UMAP-from-existing finished')
