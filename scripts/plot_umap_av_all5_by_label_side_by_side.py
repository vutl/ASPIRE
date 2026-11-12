#!/usr/bin/env python3
"""Plot all-5-fold paired UMAPs from A/V embeddings, colored by discrete emotion labels."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba_cache")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from sklearn.preprocessing import StandardScaler
from umap import UMAP

sys.path.append(str(Path(__file__).resolve().parent.parent))

from scripts.full_vs_noav_analysis_utils import add_shared_args, aggregate_cached, ensure_cache_for_models, ensure_output_dirs  # noqa: E402


LABEL_ID_TO_NAME = {0: "Happy", 1: "Sad", 2: "Neutral", 3: "Angry"}

AROUSAL_COLOR_BY_LABEL = {
    1: "#c62828",
    2: "#f2a3a3",
    0: "#8ecae6",
    3: "#1565c0",
}
VALENCE_COLOR_BY_LABEL = {
    1: "#c62828",
    3: "#f2a3a3",
    2: "#8ecae6",
    0: "#1565c0",
}

AROUSAL_LEGEND_ORDER = [1, 2, 0, 3]
VALENCE_LEGEND_ORDER = [1, 3, 2, 0]


def preprocess_features(features: np.ndarray) -> np.ndarray:
    return StandardScaler().fit_transform(features)


def project_umap(features: np.ndarray, random_state: int, n_neighbors: int, min_dist: float) -> np.ndarray:
    reducer = UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        random_state=random_state,
    )
    return reducer.fit_transform(features)


def stretch_embedding(z: np.ndarray, scale: float = 1.6, lower: float = 1.0, upper: float = 99.0) -> np.ndarray:
    z = np.asarray(z, dtype=np.float64)
    out = z.copy()
    for dim in range(2):
        lo = np.percentile(out[:, dim], lower)
        hi = np.percentile(out[:, dim], upper)
        if hi - lo < 1e-8:
            lo = float(np.min(out[:, dim]))
            hi = float(np.max(out[:, dim]))
        center = 0.5 * (lo + hi)
        denom = max(hi - lo, 1e-8)
        out[:, dim] = scale * (out[:, dim] - center) / denom
    return out


def inflate_all5_islands(
    z: np.ndarray,
    fold_ids: np.ndarray,
    cluster_scale: float = 2.95,
    center_pull: float = 0.55,
) -> np.ndarray:
    z = np.asarray(z, dtype=np.float64)
    fold_ids = np.asarray(fold_ids)
    out = z.copy()
    global_center = np.mean(out, axis=0, keepdims=True)

    for fold_id in np.unique(fold_ids):
        mask = fold_ids == fold_id
        cluster = out[mask]
        centroid = np.mean(cluster, axis=0, keepdims=True)
        out[mask] = global_center + center_pull * (cluster - global_center)
        out[mask] = centroid + cluster_scale * (out[mask] - centroid)
    return out


def set_tight_bounds(ax, z: np.ndarray, pad_ratio: float = 0.08) -> None:
    x_min, x_max = float(np.min(z[:, 0])), float(np.max(z[:, 0]))
    y_min, y_max = float(np.min(z[:, 1])), float(np.max(z[:, 1]))
    x_pad = max((x_max - x_min) * pad_ratio, 1e-3)
    y_pad = max((y_max - y_min) * pad_ratio, 1e-3)
    ax.set_xlim(x_min - x_pad, x_max + x_pad)
    ax.set_ylim(y_min - y_pad, y_max + y_pad)


def make_legend_handles(order, color_by_label):
    return [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=color_by_label[label_id],
            markersize=9,
            label=LABEL_ID_TO_NAME[label_id],
        )
        for label_id in order
    ]


def scatter_by_label(ax, z, labels, color_by_label):
    for label_id in [0, 1, 2, 3]:
        mask = labels == label_id
        if np.any(mask):
            ax.scatter(
                z[mask, 0],
                z[mask, 1],
                s=16,
                alpha=0.90,
                linewidths=0,
                c=color_by_label[label_id],
            )


def build_entry(agg, feature_key: str, args):
    z = project_umap(
        preprocess_features(agg[feature_key]),
        random_state=args.random_state,
        n_neighbors=args.umap_neighbors,
        min_dist=args.umap_min_dist,
    )
    z = stretch_embedding(z, scale=args.stretch_scale)
    fold_ids = np.repeat(np.asarray(args.folds), len(z) // len(args.folds)) if False else None
    return z


def main():
    parser = argparse.ArgumentParser()
    add_shared_args(parser)
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--umap_neighbors", type=int, default=15)
    parser.add_argument("--umap_min_dist", type=float, default=0.3)
    parser.add_argument("--stretch_scale", type=float, default=1.72)
    parser.add_argument("--all5_cluster_scale", type=float, default=2.95)
    parser.add_argument("--all5_center_pull", type=float, default=0.55)
    args = parser.parse_args()

    dirs = ensure_output_dirs(args.output)
    ensure_cache_for_models(args)

    jobs = [
        ("full", "full_av_all5_by_label_side_by_side.pdf", "Full ASPIRE"),
        ("noav", "noav_av_all5_by_label_side_by_side.pdf", "No-AV-Sup"),
    ]

    for model_key, out_name, model_title in jobs:
        agg = aggregate_cached(dirs["cache"], model_key, args.folds)
        labels = agg["y_true"].reshape(-1)
        fold_ids = np.concatenate(
            [
                np.full(load_cached_len, fold_id, dtype=np.int64)
                for fold_id, load_cached_len in zip(
                    args.folds,
                    [aggregate_cached(dirs["cache"], model_key, [fold_id])["y_true"].shape[0] for fold_id in args.folds],
                )
            ],
            axis=0,
        )

        z_arousal = project_umap(
            preprocess_features(agg["arousal_embedding"]),
            random_state=args.random_state,
            n_neighbors=args.umap_neighbors,
            min_dist=args.umap_min_dist,
        )
        z_valence = project_umap(
            preprocess_features(agg["valence_embedding"]),
            random_state=args.random_state,
            n_neighbors=args.umap_neighbors,
            min_dist=args.umap_min_dist,
        )
        z_arousal = stretch_embedding(z_arousal, scale=args.stretch_scale)
        z_valence = stretch_embedding(z_valence, scale=args.stretch_scale)
        z_arousal = inflate_all5_islands(
            z_arousal,
            fold_ids,
            cluster_scale=args.all5_cluster_scale,
            center_pull=args.all5_center_pull,
        )
        z_valence = inflate_all5_islands(
            z_valence,
            fold_ids,
            cluster_scale=args.all5_cluster_scale,
            center_pull=args.all5_center_pull,
        )

        fig, axes = plt.subplots(1, 2, figsize=(13.5, 6.1))

        scatter_by_label(axes[0], z_arousal, labels, AROUSAL_COLOR_BY_LABEL)
        set_tight_bounds(axes[0], z_arousal)
        axes[0].set_title("Arousal embedding", fontsize=13)
        axes[0].set_xticks([])
        axes[0].set_yticks([])

        scatter_by_label(axes[1], z_valence, labels, VALENCE_COLOR_BY_LABEL)
        set_tight_bounds(axes[1], z_valence)
        axes[1].set_title("Valence embedding", fontsize=13)
        axes[1].set_xticks([])
        axes[1].set_yticks([])

        fig.suptitle(f"{model_title}: all 5 folds", fontsize=16, y=0.99)
        fig.legend(
            handles=make_legend_handles(AROUSAL_LEGEND_ORDER, AROUSAL_COLOR_BY_LABEL),
            loc="upper center",
            bbox_to_anchor=(0.28, 0.93),
            ncol=4,
            frameon=False,
            title="Arousal palette",
        )
        fig.legend(
            handles=make_legend_handles(VALENCE_LEGEND_ORDER, VALENCE_COLOR_BY_LABEL),
            loc="upper center",
            bbox_to_anchor=(0.73, 0.93),
            ncol=4,
            frameon=False,
            title="Valence palette",
        )
        fig.subplots_adjust(top=0.80, wspace=0.08)
        plt.savefig(dirs["plots"] / out_name)
        plt.close()

    print("Saved:")
    print(dirs["plots"] / "full_av_all5_by_label_side_by_side.pdf")
    print(dirs["plots"] / "noav_av_all5_by_label_side_by_side.pdf")


if __name__ == "__main__":
    main()
