#!/usr/bin/env python3
"""Plot fixed-parameter UMAPs colored by ground-truth arousal/valence."""
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


def inflate_all5_islands(
    z: np.ndarray,
    fold_ids: np.ndarray,
    cluster_scale: float = 1.8,
    center_pull: float = 0.78,
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


def set_tight_bounds(ax, z: np.ndarray, pad_ratio: float = 0.06):
    x_min, x_max = float(np.min(z[:, 0])), float(np.max(z[:, 0]))
    y_min, y_max = float(np.min(z[:, 1])), float(np.max(z[:, 1]))
    x_pad = max((x_max - x_min) * pad_ratio, 1e-3)
    y_pad = max((y_max - y_min) * pad_ratio, 1e-3)
    ax.set_xlim(x_min - x_pad, x_max + x_pad)
    ax.set_ylim(y_min - y_pad, y_max + y_pad)


def collect_split_data(cache_dir: Path, model_key: str, folds: list[int], feature_key: str, value_key: str):
    per_fold = {}
    for fold_id in folds:
        row = load_cached_fold(cache_dir, model_key, fold_id)
        if row[feature_key].size == 0:
            raise RuntimeError(f"Missing {feature_key} for {model_key} fold {fold_id}")
        per_fold[fold_id] = {
            "features": row[feature_key],
            "values": row[value_key],
            "fold_ids": np.full(row[value_key].shape[0], fold_id, dtype=np.int64),
        }

    agg = aggregate_cached(cache_dir, model_key, folds)
    per_fold["all5"] = {
        "features": agg[feature_key],
        "values": agg[value_key],
        "fold_ids": np.concatenate([per_fold[fold_id]["fold_ids"] for fold_id in folds], axis=0),
    }
    return per_fold


def add_class_hint(fig, value_name: str):
    if value_name.lower() == "arousal":
        hint = "Class tendency: low -> sad, neu    |    high -> hap, ang"
    else:
        hint = "Class tendency: low -> sad, ang    |    mid -> neu    |    high -> hap"

    fig.text(
        0.5,
        0.935,
        hint,
        ha="center",
        va="center",
        fontsize=11,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "#f5f5f5", "edgecolor": "#bdbdbd"},
    )


def plot_value_grid(entries, out_path, title, cmap, value_name):
    fig, axes = plt.subplots(2, 3, figsize=(15, 9.5))
    flat_axes = axes.reshape(-1)
    last_scatter = None
    all_values = np.concatenate([entry["values"].reshape(-1) for entry in entries], axis=0)
    vmin = float(np.min(all_values))
    vmax = float(np.max(all_values))
    for idx, entry in enumerate(entries):
        ax = flat_axes[idx]
        last_scatter = ax.scatter(
            entry["z"][:, 0],
            entry["z"][:, 1],
            c=entry["values"],
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            s=12,
            alpha=0.85,
            linewidths=0,
        )
        set_tight_bounds(ax, entry["z"])
        ax.set_title(entry["title"], fontsize=11)
        ax.set_xlabel("UMAP-1")
        ax.set_ylabel("UMAP-2")
    fig.subplots_adjust(right=0.90, wspace=0.22, hspace=0.26, top=0.86)
    cax = fig.add_axes([0.92, 0.18, 0.015, 0.62])
    cbar = fig.colorbar(last_scatter, cax=cax)
    cbar.set_label("Normalized value")
    fig.suptitle(title, fontsize=15)
    add_class_hint(fig, value_name)
    plt.savefig(out_path)
    plt.close()


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

    jobs = [
        ("full", "arousal_embedding", "arousal_true", "coolwarm", "full_arousal_umap_grid.pdf", "Full ASPIRE: arousal_embedding colored by GT arousal", "Arousal"),
        ("full", "valence_embedding", "valence_true", "coolwarm", "full_valence_umap_grid.pdf", "Full ASPIRE: valence_embedding colored by GT valence", "Valence"),
        ("noav", "arousal_embedding", "arousal_true", "coolwarm", "noav_arousal_umap_grid.pdf", "No-AV-Sup: arousal_embedding colored by GT arousal", "Arousal"),
        ("noav", "valence_embedding", "valence_true", "coolwarm", "noav_valence_umap_grid.pdf", "No-AV-Sup: valence_embedding colored by GT valence", "Valence"),
    ]

    split_order = ["all5"] + list(args.folds)

    for model_key, feature_key, value_key, cmap, out_name, title, value_name in jobs:
        split_data = collect_split_data(dirs["cache"], model_key, args.folds, feature_key, value_key)
        entries = []
        for split_name in split_order:
            z = project_umap(
                preprocess_features(split_data[split_name]["features"]),
                random_state=args.random_state,
                n_neighbors=args.umap_neighbors,
                min_dist=args.umap_min_dist,
            )
            z = stretch_embedding(z)
            if split_name == "all5":
                z = inflate_all5_islands(
                    z,
                    split_data[split_name]["fold_ids"],
                    cluster_scale=args.all5_cluster_scale,
                    center_pull=args.all5_center_pull,
                )
            entries.append(
                {
                    "z": z,
                    "values": split_data[split_name]["values"].reshape(-1),
                    "title": "All 5 Folds" if split_name == "all5" else f"Fold {split_name}",
                }
            )
        plot_value_grid(entries, dirs["plots"] / out_name, title, cmap, value_name)

    print("Saved:")
    print(dirs["plots"] / "full_arousal_umap_grid.pdf")
    print(dirs["plots"] / "full_valence_umap_grid.pdf")
    print(dirs["plots"] / "noav_arousal_umap_grid.pdf")
    print(dirs["plots"] / "noav_valence_umap_grid.pdf")


if __name__ == "__main__":
    main()
