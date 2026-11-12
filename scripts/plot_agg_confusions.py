#!/usr/bin/env python3
"""
Plot aggregated confusion matrices for Full vs No-AV models.

Behavior:
 - If `confusion_full_agg.npy` and `confusion_noav_agg.npy` exist in `--results`, load them.
 - Otherwise, looks for per-fold files named `confusion_full_fold{f}.npy` and `confusion_noav_fold{f}.npy` for folds in `--folds` and sums them.
 - Saves: `confusion_full_agg.npy`, `confusion_noav_agg.npy`, `confusion_delta_agg.npy`, and PNG heatmaps (raw counts and normalized delta).

Usage:
  python scripts/plot_agg_confusions.py --results results/compare_full_vs_noav --folds 1 2 3 4 5
"""
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

LABELS = ['Happy', 'Sad', 'Neutral', 'Angry']


def save_confusion(cm, labels, outpath, title=None, fmt='d'):
    os.makedirs(outpath.parent, exist_ok=True)
    np.save(str(outpath) + '.npy', cm)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt=fmt, xticklabels=labels, yticklabels=labels, cmap='Blues')
    if title:
        plt.title(title)
    plt.xlabel('Pred')
    plt.ylabel('True')
    plt.tight_layout()
    plt.savefig(str(outpath) + '.pdf')
    plt.close()


def save_heatmap_float(mat, labels, outpath, title=None):
    os.makedirs(outpath.parent, exist_ok=True)
    plt.figure(figsize=(6,5))
    sns.heatmap(mat, annot=True, fmt='.3f', xticklabels=labels, yticklabels=labels, cmap='coolwarm', center=0)
    if title:
        plt.title(title)
    plt.xlabel('Pred')
    plt.ylabel('True')
    plt.tight_layout()
    plt.savefig(str(outpath) + '.pdf')
    plt.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--results', default='results/compare_full_vs_noav')
    p.add_argument('--folds', nargs='+', type=int, default=[1,2,3,4,5])
    args = p.parse_args()

    out = Path(args.results)
    full_agg_path = out / 'confusion_full_agg.npy'
    noav_agg_path = out / 'confusion_noav_agg.npy'

    if full_agg_path.exists() and noav_agg_path.exists():
        cm_full = np.load(full_agg_path)
        cm_noav = np.load(noav_agg_path)
    else:
        cm_full = np.zeros((4,4), dtype=int)
        cm_noav = np.zeros((4,4), dtype=int)
        for f in args.folds:
            pfull = out / f'confusion_full_fold{f}.npy'
            pnoav = out / f'confusion_noav_fold{f}.npy'
            if pfull.exists():
                cm_full += np.load(pfull)
            else:
                print(f'Warning: missing {pfull}, skipping')
            if pnoav.exists():
                cm_noav += np.load(pnoav)
            else:
                print(f'Warning: missing {pnoav}, skipping')

    # Save aggregated npy
    os.makedirs(out, exist_ok=True)
    np.save(out / 'confusion_full_agg.npy', cm_full)
    np.save(out / 'confusion_noav_agg.npy', cm_noav)
    np.save(out / 'confusion_delta_agg.npy', cm_full - cm_noav)

    # Raw plots
    save_confusion(cm_full, LABELS, out / 'confusion_full_agg', title='Full - aggregated (counts)')
    save_confusion(cm_noav, LABELS, out / 'confusion_noav_agg', title='No-AV - aggregated (counts)')
    save_confusion(cm_full - cm_noav, LABELS, out / 'confusion_delta_agg', title='Delta (Full - NoAV) (counts)')

    # Normalized and delta heatmap (row-normalized)
    row_sum_full = cm_full.sum(axis=1, keepdims=True)
    row_sum_noav = cm_noav.sum(axis=1, keepdims=True)
    norm_full = cm_full.astype(float) / (row_sum_full + 1e-8)
    norm_noav = cm_noav.astype(float) / (row_sum_noav + 1e-8)
    delta_norm = norm_full - norm_noav

    save_heatmap_float(norm_full, LABELS, out / 'confusion_full_norm', title='Full - normalized')
    save_heatmap_float(norm_noav, LABELS, out / 'confusion_noav_norm', title='NoAV - normalized')
    save_heatmap_float(delta_norm, LABELS, out / 'confusion_delta_norm', title='Delta (Full - NoAV) normalized')

    # Also save a simple CSV summary of diagonal rates
    diag_full = np.diag(cm_full) / (cm_full.sum(axis=1) + 1e-8)
    diag_noav = np.diag(cm_noav) / (cm_noav.sum(axis=1) + 1e-8)
    import csv
    with open(out / 'per_class_rates.csv', 'w', newline='') as fh:
        writer = csv.writer(fh)
        writer.writerow(['label','full_correct_rate','noav_correct_rate','delta'])
        for i, lbl in enumerate(LABELS):
            writer.writerow([lbl, float(diag_full[i]), float(diag_noav[i]), float(diag_full[i]-diag_noav[i])])

    print('Saved aggregated confusion matrices and plots in', out)


if __name__ == '__main__':
    main()
