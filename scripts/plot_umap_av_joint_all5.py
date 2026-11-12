#!/usr/bin/env python3
"""Plot all-5-fold UMAP with a single joint A/V color for each utterance."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba_cache")

import matplotlib.pyplot as plt
import numpy as np
from umap import UMAP
from sklearn.preprocessing import StandardScaler

sys.path.append(str(Path(__file__).resolve().parent.parent))

from scripts.full_vs_noav_analysis_utils import (
    add_shared_args,
    aggregate_cached,
    ensure_cache_for_models,
    ensure_output_dirs,
    load_cached_fold,
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


def inflate_all5_islands(z: np.ndarray, fold_ids: np.ndarray, cluster_scale: float, center_pull: float):
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


def build_fold_ids(cache_dir: Path, folds: list[int], model_key: str = "full") -> np.ndarray:
    sizes = []
    for fold_id in folds:
        row = load_cached_fold(cache_dir, model_key, fold_id)
        sizes.append(int(row["y_true"].shape[0]))
    return np.concatenate(
        [np.full(size, fold_id, dtype=np.int64) for fold_id, size in zip(folds, sizes)],
        axis=0,
    )


def normalize_01(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    lo = np.min(values)
    hi = np.max(values)
    return (values - lo) / max(hi - lo, 1e-8)


def av_to_rgb(arousal: np.ndarray, valence: np.ndarray) -> np.ndarray:
    """Bivariate color map: valence left->right, arousal bottom->top."""
    a = normalize_01(arousal)
    v = normalize_01(valence)

    c00 = np.array([59, 76, 192], dtype=np.float64) / 255.0    # low A, low V -> blue
    c10 = np.array([102, 189, 99], dtype=np.float64) / 255.0   # low A, high V -> green
    c01 = np.array([180, 4, 38], dtype=np.float64) / 255.0     # high A, low V -> red
    c11 = np.array([255, 215, 0], dtype=np.float64) / 255.0    # high A, high V -> yellow

    colors = (
        (1 - a)[:, None] * (1 - v)[:, None] * c00
        + (1 - a)[:, None] * v[:, None] * c10
        + a[:, None] * (1 - v)[:, None] * c01
        + a[:, None] * v[:, None] * c11
    )
    return np.clip(colors, 0.0, 1.0)


def draw_av_legend(ax):
    grid = 200
    valence = np.linspace(0.0, 1.0, grid)
    arousal = np.linspace(0.0, 1.0, grid)
    vv, aa = np.meshgrid(valence, arousal)
    colors = av_to_rgb(aa.reshape(-1), vv.reshape(-1)).reshape(grid, grid, 3)
    ax.imshow(colors, origin="lower", extent=[0, 1, 0, 1], aspect="auto")
    ax.set_xlabel("Valence")
    ax.set_ylabel("Arousal")
    ax.set_title("A/V color legend", fontsize=10)


def draw_av_legend_clean(ax):
    grid = 220
    valence = np.linspace(0.0, 1.0, grid)
    arousal = np.linspace(0.0, 1.0, grid)
    vv, aa = np.meshgrid(valence, arousal)
    colors = av_to_rgb(aa.reshape(-1), vv.reshape(-1)).reshape(grid, grid, 3)
    ax.imshow(colors, origin="lower", extent=[0, 1, 0, 1], aspect="auto")
    ax.set_xticks([0.0, 1.0])
    ax.set_xticklabels(["Low", "High"])
    ax.set_yticks([0.0, 1.0])
    ax.set_yticklabels(["Low", "High"])
    ax.set_xlabel("Valence", fontsize=10)
    ax.set_ylabel("Arousal", fontsize=10)
    ax.set_title("A/V", fontsize=11, pad=4)
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)


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
    fold_ids = build_fold_ids(dirs["cache"], args.folds, model_key="full")

    jobs = [
        ("Full ASPIRE", full["fused_features"], full["arousal_true"], full["valence_true"]),
        ("No-AV-Sup", noav["fused_features"], noav["arousal_true"], noav["valence_true"]),
    ]

    fig = plt.figure(figsize=(13.8, 6.2))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 0.14], wspace=0.18)
    axes = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])]
    legend_ax = fig.add_subplot(gs[0, 2])

    for ax, (title, features, arousal, valence) in zip(axes, jobs):
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
        colors = av_to_rgb(arousal.reshape(-1), valence.reshape(-1))
        ax.scatter(z[:, 0], z[:, 1], c=colors, s=12, alpha=0.85, linewidths=0)
        set_tight_bounds(ax, z)
        ax.set_title(title, fontsize=12)
        ax.set_xlabel("UMAP-1")
        ax.set_ylabel("UMAP-2")

    draw_av_legend(legend_ax)
    fig.suptitle("All 5 Folds UMAP with joint Arousal-Valence coloring", fontsize=15)

    out_path = dirs["plots"] / "av_joint_umap_all5_side_by_side.pdf"
    plt.savefig(out_path)
    plt.close()

    fig = plt.figure(figsize=(12.4, 5.9))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 0.12], wspace=0.12)
    axes = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])]
    legend_ax = fig.add_subplot(gs[0, 2])

    for ax, (title, features, arousal, valence) in zip(axes, jobs):
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
        colors = av_to_rgb(arousal.reshape(-1), valence.reshape(-1))
        ax.scatter(z[:, 0], z[:, 1], c=colors, s=10, alpha=0.9, linewidths=0)
        set_tight_bounds(ax, z, pad_ratio=0.04)
        ax.set_title(title.replace("\n", " "), fontsize=12, pad=6)
        ax.set_xlabel("UMAP-1", fontsize=10)
        ax.set_ylabel("UMAP-2", fontsize=10)
        ax.tick_params(labelsize=9)

    draw_av_legend_clean(legend_ax)
    fig.subplots_adjust(left=0.06, right=0.96, top=0.90, bottom=0.11)

    clean_out_path = dirs["plots"] / "av_joint_umap_all5_side_by_side_clean.pdf"
    plt.savefig(clean_out_path)
    plt.close()

    with open(dirs["tables"] / "av_joint_umap_all5_protocol.txt", "w", encoding="utf-8") as fh:
        fh.write(f"random_state={args.random_state}\n")
        fh.write(f"umap_neighbors={args.umap_neighbors}\n")
        fh.write(f"umap_min_dist={args.umap_min_dist}\n")
        fh.write(f"all5_cluster_scale={args.all5_cluster_scale}\n")
        fh.write(f"all5_center_pull={args.all5_center_pull}\n")

    print("Saved:")
    print(out_path)
    print(clean_out_path)
    print(dirs["tables"] / "av_joint_umap_all5_protocol.txt")


if __name__ == "__main__":
    main()
