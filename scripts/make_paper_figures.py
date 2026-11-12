#!/usr/bin/env python3
"""Make paper-ready figures in ./figures from existing analysis caches/tables."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba_cache")

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from umap import UMAP

sys.path.append(str(Path(__file__).resolve().parent.parent))

from scripts.full_vs_noav_analysis_utils import LABEL_NAMES


FULL_CM = np.array(
    [
        [1229, 87, 220, 100],
        [45, 850, 145, 44],
        [210, 180, 1213, 105],
        [51, 21, 127, 904],
    ],
    dtype=np.int64,
)


def per_class_f1_from_confusion(cm: np.ndarray) -> np.ndarray:
    f1s = []
    for idx in range(cm.shape[0]):
        tp = float(cm[idx, idx])
        fp = float(cm[:, idx].sum() - tp)
        fn = float(cm[idx, :].sum() - tp)
        denom = 2 * tp + fp + fn
        f1s.append(0.0 if denom == 0 else (2 * tp) / denom)
    return np.asarray(f1s, dtype=np.float64)


def save_confusion_full(figures_dir: Path, cm_full: np.ndarray):
    fig, ax = plt.subplots(figsize=(5.05, 4.35))
    sns.heatmap(
        cm_full,
        annot=True,
        fmt="d",
        xticklabels=LABEL_NAMES,
        yticklabels=LABEL_NAMES,
        cmap="Blues",
        cbar=False,
        square=True,
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    plt.tight_layout(pad=0.15)
    plt.savefig(figures_dir / "confusion_full_counts.pdf", bbox_inches="tight", pad_inches=0.02)
    plt.close()


def save_per_class_f1(figures_dir: Path, cm_full: np.ndarray, cm_noav: np.ndarray):
    f1_full = per_class_f1_from_confusion(cm_full)
    f1_noav = per_class_f1_from_confusion(cm_noav)
    x = np.arange(len(LABEL_NAMES))

    fig, ax = plt.subplots(figsize=(5.9, 4.1))
    ax.bar(x - 0.18, f1_noav, width=0.36, label="No Dim Guidance")
    ax.bar(x + 0.18, f1_full, width=0.36, label="ASPIRE")
    ax.set_xticks(x, LABEL_NAMES)
    ax.set_ylabel("Per-class F1")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.10), ncol=2, frameon=False)
    plt.tight_layout(rect=[0, 0, 1, 0.95], pad=0.15)
    plt.savefig(figures_dir / "per_class_f1_comparison.pdf", bbox_inches="tight", pad_inches=0.02)
    plt.close()


def preprocess_features(features: np.ndarray):
    return StandardScaler().fit_transform(features)


def project_umap(features: np.ndarray, random_state: int, n_neighbors: int, min_dist: float):
    reducer = UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        random_state=random_state,
    )
    return reducer.fit_transform(features)


def stretch_embedding(z: np.ndarray, lower: float = 1.0, upper: float = 99.0) -> np.ndarray:
    z = np.asarray(z, dtype=np.float64)
    out = z.copy()
    for dim in range(2):
        lo = np.percentile(out[:, dim], lower)
        hi = np.percentile(out[:, dim], upper)
        if hi - lo < 1e-8:
            lo = float(np.min(out[:, dim]))
            hi = float(np.max(out[:, dim]))
        center = 0.5 * (lo + hi)
        scale = max(hi - lo, 1e-8)
        out[:, dim] = (out[:, dim] - center) / scale
    return out


def set_tight_bounds(ax, z: np.ndarray, pad_ratio: float = 0.06):
    x_min, x_max = float(np.min(z[:, 0])), float(np.max(z[:, 0]))
    y_min, y_max = float(np.min(z[:, 1])), float(np.max(z[:, 1]))
    x_pad = max((x_max - x_min) * pad_ratio, 1e-3)
    y_pad = max((y_max - y_min) * pad_ratio, 1e-3)
    ax.set_xlim(x_min - x_pad, x_max + x_pad)
    ax.set_ylim(y_min - y_pad, y_max + y_pad)


def label_names_from_ids(label_ids: np.ndarray):
    return np.array([LABEL_NAMES[int(i)] for i in label_ids])


def save_umap_fold_grid(figures_dir: Path, analysis_root: Path):
    tables_dir = analysis_root / "tables"
    full_items = []
    for fold_id in [1, 2, 3, 4, 5]:
        full_cache = np.load(analysis_root / "cache" / f"full_fold{fold_id}.npz", allow_pickle=True)

        z_full = project_umap(preprocess_features(full_cache["fused_features"]), 42, 15, 0.3)
        z_full = stretch_embedding(z_full)
        full_items.append((z_full, label_names_from_ids(full_cache["y_true"])))

        np.save(tables_dir / f"paper_umap_full_fold{fold_id}.npy", z_full)

    fig, axes = plt.subplots(1, 5, figsize=(16.2, 3.35))
    palette = sns.color_palette("tab10", n_colors=len(LABEL_NAMES))
    palette_map = dict(zip(LABEL_NAMES, palette))

    legend_handles = None
    legend_labels = None
    for idx, fold_id in enumerate([1, 2, 3, 4, 5]):
        ax = axes[idx]
        z, labels = full_items[idx]
        sns.scatterplot(
            x=z[:, 0],
            y=z[:, 1],
            hue=labels,
            hue_order=LABEL_NAMES,
            palette=palette_map,
            s=12,
            linewidth=0,
            alpha=0.85,
            ax=ax,
            legend=(idx == 4),
        )
        set_tight_bounds(ax, z)
        ax.set_title(f"Fold {fold_id}", fontsize=11)
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_xticks([])
        ax.set_yticks([])
        if idx == 4 and ax.legend_ is not None:
            legend_handles, legend_labels = ax.get_legend_handles_labels()
            ax.legend_.remove()

    fig.text(0.017, 0.50, "ASPIRE", rotation=90, va="center", ha="center", fontsize=12, weight="bold")
    if legend_handles:
        fig.legend(
            legend_handles,
            legend_labels,
            loc="center left",
            bbox_to_anchor=(0.905, 0.5),
            frameon=True,
            fancybox=True,
            borderpad=0.6,
            labelspacing=0.45,
            handletextpad=0.5,
            fontsize=10,
        )
    fig.subplots_adjust(left=0.045, right=0.89, top=0.90, bottom=0.12, wspace=0.08)
    plt.savefig(figures_dir / "umap_fold_grid.pdf", bbox_inches="tight", pad_inches=0.02)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis_root", default="results/full_vs_noav_analysis")
    parser.add_argument("--figures_dir", default="figures")
    args = parser.parse_args()

    analysis_root = Path(args.analysis_root)
    figures_dir = Path(args.figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)

    cm_noav = np.load(analysis_root / "tables" / "confusion_noav_counts.npy")
    save_confusion_full(figures_dir, FULL_CM)
    save_per_class_f1(figures_dir, FULL_CM, cm_noav)
    save_umap_fold_grid(figures_dir, analysis_root)

    print("Saved:")
    print(figures_dir / "confusion_full_counts.pdf")
    print(figures_dir / "per_class_f1_comparison.pdf")
    print(figures_dir / "umap_fold_grid.pdf")


if __name__ == "__main__":
    main()
