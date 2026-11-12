#!/usr/bin/env python3
"""Create a 2x3 paper-ready grid: 5 IEMOCAP folds + 1 MSP panel."""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba_cache")

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from umap import UMAP

sys.path.append(str(Path(__file__).resolve().parent.parent))

from scripts.full_vs_noav_analysis_utils import LABEL_NAMES


def preprocess_features(features: np.ndarray) -> np.ndarray:
    return StandardScaler().fit_transform(features)


def project_umap(features: np.ndarray, random_state: int = 42, n_neighbors: int = 15, min_dist: float = 0.3):
    reducer = UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        random_state=random_state,
    )
    return reducer.fit_transform(features)


def project_umap_fast(
    features: np.ndarray,
    random_state: int = 42,
    n_neighbors: int = 15,
    min_dist: float = 0.3,
    pca_dim: int = 50,
    sample_size: int = 5000,
) -> np.ndarray:
    if features.shape[1] > pca_dim:
        features = PCA(n_components=pca_dim, random_state=random_state).fit_transform(features)

    if features.shape[0] <= sample_size:
        return project_umap(features, random_state=random_state, n_neighbors=n_neighbors, min_dist=min_dist)

    rng = np.random.default_rng(random_state)
    sample_idx = np.sort(rng.choice(features.shape[0], size=sample_size, replace=False))
    reducer = UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        random_state=random_state,
    )
    reducer.fit(features[sample_idx])
    return reducer.transform(features)


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


def set_tight_bounds(ax, z: np.ndarray, pad_ratio: float = 0.055):
    x_min, x_max = float(np.min(z[:, 0])), float(np.max(z[:, 0]))
    y_min, y_max = float(np.min(z[:, 1])), float(np.max(z[:, 1]))
    x_pad = max((x_max - x_min) * pad_ratio, 1e-3)
    y_pad = max((y_max - y_min) * pad_ratio, 1e-3)
    ax.set_xlim(x_min - x_pad, x_max + x_pad)
    ax.set_ylim(y_min - y_pad, y_max + y_pad)


def label_names_from_ids(label_ids: np.ndarray):
    return np.array([LABEL_NAMES[int(i)] for i in label_ids])


def main():
    root = Path("/home4/quanpn/ICME2026/AVD_Test")
    figures_dir = root / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    panels = []
    for fold_id in [1, 2, 3, 4, 5]:
        cache = np.load(root / "results" / "full_vs_noav_analysis" / "cache" / f"full_fold{fold_id}.npz", allow_pickle=True)
        z = project_umap(preprocess_features(cache["fused_features"]))
        z = stretch_embedding(z)
        labels = label_names_from_ids(cache["y_true"])
        panels.append((f"IEMOCAP Fold {fold_id}", z, labels))

    d1 = np.load(root / "results" / "msp_wpro_manifold" / "tables" / "features_test1.npz")
    d2 = np.load(root / "results" / "msp_wpro_manifold" / "tables" / "features_test2.npz")
    msp_features = np.concatenate([d1["fused_features"], d2["fused_features"]], axis=0)
    z_msp = stretch_embedding(project_umap_fast(preprocess_features(msp_features)))
    labels_msp = label_names_from_ids(np.concatenate([d1["y_true"], d2["y_true"]], axis=0))
    panels.append(("MSP Test1 + Test2", z_msp, labels_msp))

    fig, axes = plt.subplots(2, 3, figsize=(11.8, 7.2))
    palette = sns.color_palette("tab10", n_colors=len(LABEL_NAMES))
    palette_map = dict(zip(LABEL_NAMES, palette))

    legend_handles = None
    legend_labels = None
    for ax, (title, z, labels) in zip(axes.flatten(), panels):
        sns.scatterplot(
            x=z[:, 0],
            y=z[:, 1],
            hue=labels,
            hue_order=LABEL_NAMES,
            palette=palette_map,
            s=11,
            linewidth=0,
            alpha=0.82,
            ax=ax,
            legend=(title == "MSP Test1 + Test2"),
        )
        set_tight_bounds(ax, z)
        ax.set_title(title, fontsize=11.5)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel("")
        ax.set_ylabel("")
        if title == "MSP Test1 + Test2" and ax.legend_ is not None:
            legend_handles, legend_labels = ax.get_legend_handles_labels()
            ax.legend_.remove()

    if legend_handles:
        fig.legend(
            legend_handles,
            legend_labels,
            loc="center left",
            bbox_to_anchor=(0.905, 0.5),
            frameon=True,
            fancybox=True,
            borderpad=0.55,
            labelspacing=0.4,
            handletextpad=0.45,
            fontsize=10,
        )

    fig.subplots_adjust(left=0.04, right=0.88, top=0.94, bottom=0.06, wspace=0.10, hspace=0.15)
    out_path = figures_dir / "iemocap_msp_umap_grid.pdf"
    plt.savefig(out_path, bbox_inches="tight", pad_inches=0.02)
    plt.close()
    print(out_path)


if __name__ == "__main__":
    main()
