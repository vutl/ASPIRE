#!/usr/bin/env python3
"""Plot a single combined all-5-fold UMAP figure colored by GT arousal/valence."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba_cache")

import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
from umap import UMAP

sys.path.append(str(Path(__file__).resolve().parent.parent))

from scripts.full_vs_noav_analysis_utils import (
    add_shared_args,
    aggregate_cached,
    ensure_cache_for_models,
    ensure_output_dirs,
)


def preprocess_features(features: np.ndarray):
    scaler = StandardScaler()
    return scaler.fit_transform(features)


def project_umap(features: np.ndarray, random_state: int, n_neighbors: int, min_dist: float):
    reducer = UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        random_state=random_state,
    )
    return reducer.fit_transform(features)


def stretch_embedding(z: np.ndarray, lower: float = 1.0, upper: float = 99.0) -> np.ndarray:
    out = np.asarray(z, dtype=np.float64).copy()
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


def inflate_all5_islands(
    z: np.ndarray,
    fold_ids: np.ndarray,
    cluster_scale: float = 1.8,
    center_pull: float = 0.78,
) -> np.ndarray:
    out = np.asarray(z, dtype=np.float64).copy()
    fold_ids = np.asarray(fold_ids)
    global_center = np.mean(out, axis=0, keepdims=True)
    for fold_id in np.unique(fold_ids):
        mask = fold_ids == fold_id
        cluster = out[mask]
        centroid = np.mean(cluster, axis=0, keepdims=True)
        out[mask] = global_center + center_pull * (cluster - global_center)
        out[mask] = centroid + cluster_scale * (out[mask] - centroid)
    return out


def set_tight_bounds(ax, z: np.ndarray, pad_ratio: float = 0.06):
    x_min, x_max = float(np.min(z[:, 0])), float(np.max(z[:, 0]))
    y_min, y_max = float(np.min(z[:, 1])), float(np.max(z[:, 1]))
    x_pad = max((x_max - x_min) * pad_ratio, 1e-3)
    y_pad = max((y_max - y_min) * pad_ratio, 1e-3)
    ax.set_xlim(x_min - x_pad, x_max + x_pad)
    ax.set_ylim(y_min - y_pad, y_max + y_pad)


def build_fold_ids(folds: list[int], per_fold_sizes: list[int]) -> np.ndarray:
    return np.concatenate(
        [np.full(size, fold_id, dtype=np.int64) for fold_id, size in zip(folds, per_fold_sizes)],
        axis=0,
    )


def main():
    parser = argparse.ArgumentParser()
    add_shared_args(parser)
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--umap_neighbors", type=int, default=15)
    parser.add_argument("--umap_min_dist", type=float, default=0.3)
    parser.add_argument("--all5_cluster_scale", type=float, default=1.8)
    parser.add_argument("--all5_center_pull", type=float, default=0.78)
    args = parser.parse_args()

    dirs = ensure_output_dirs(args.output)
    ensure_cache_for_models(args)

    full = aggregate_cached(dirs["cache"], "full", args.folds)
    noav = aggregate_cached(dirs["cache"], "noav", args.folds)

    # Build fold ids from cached files so the all-5-fold view can be spread slightly.
    per_fold_sizes = []
    for fold_id in args.folds:
        fold_npz = np.load(dirs["cache"] / f"full_fold{fold_id}.npz", allow_pickle=False)
        per_fold_sizes.append(int(fold_npz["y_true"].shape[0]))
    fold_ids = build_fold_ids(args.folds, per_fold_sizes)

    jobs = [
        ("Full ASPIRE\nArousal", full["arousal_embedding"], full["arousal_true"].reshape(-1), "coolwarm"),
        ("No-AV-Sup\nArousal", noav["arousal_embedding"], noav["arousal_true"].reshape(-1), "coolwarm"),
        ("Full ASPIRE\nValence", full["valence_embedding"], full["valence_true"].reshape(-1), "coolwarm"),
        ("No-AV-Sup\nValence", noav["valence_embedding"], noav["valence_true"].reshape(-1), "coolwarm"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12.8, 10.2))
    axes = axes.reshape(-1)
    scatters = []

    for ax, (title, features, values, cmap) in zip(axes, jobs):
        z = project_umap(
            preprocess_features(features),
            random_state=args.random_state,
            n_neighbors=args.umap_neighbors,
            min_dist=args.umap_min_dist,
        )
        z = stretch_embedding(z)
        z = inflate_all5_islands(
            z,
            fold_ids,
            cluster_scale=args.all5_cluster_scale,
            center_pull=args.all5_center_pull,
        )
        sc = ax.scatter(
            z[:, 0],
            z[:, 1],
            c=values,
            cmap=cmap,
            s=12,
            alpha=0.85,
            linewidths=0,
        )
        scatters.append(sc)
        set_tight_bounds(ax, z)
        ax.set_title(title, fontsize=12)
        ax.set_xlabel("UMAP-1")
        ax.set_ylabel("UMAP-2")

    fig.subplots_adjust(right=0.90, wspace=0.22, hspace=0.24, top=0.92)
    cax_top = fig.add_axes([0.92, 0.56, 0.016, 0.28])
    cbar_top = fig.colorbar(scatters[0], cax=cax_top)
    cbar_top.set_label("Normalized arousal")

    cax_bottom = fig.add_axes([0.92, 0.16, 0.016, 0.28])
    cbar_bottom = fig.colorbar(scatters[2], cax=cax_bottom)
    cbar_bottom.set_label("Normalized valence")

    fig.suptitle("All 5 Folds UMAP colored by Ground-Truth Arousal/Valence", fontsize=15)

    out_path = dirs["plots"] / "av_umap_all5_combined.pdf"
    plt.savefig(out_path)
    plt.close()

    with open(dirs["tables"] / "av_umap_all5_combined_protocol.txt", "w", encoding="utf-8") as fh:
        fh.write(f"random_state={args.random_state}\n")
        fh.write(f"umap_neighbors={args.umap_neighbors}\n")
        fh.write(f"umap_min_dist={args.umap_min_dist}\n")
        fh.write(f"all5_cluster_scale={args.all5_cluster_scale}\n")
        fh.write(f"all5_center_pull={args.all5_center_pull}\n")

    print("Saved:")
    print(out_path)
    print(dirs["tables"] / "av_umap_all5_combined_protocol.txt")


if __name__ == "__main__":
    main()
