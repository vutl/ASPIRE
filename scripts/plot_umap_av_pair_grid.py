#!/usr/bin/env python3
"""Plot paired UMAP grids: left=arousal, right=valence for all5 + each fold."""
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

from scripts.full_vs_noav_analysis_utils import (  # noqa: E402
    add_shared_args,
    aggregate_cached,
    ensure_cache_for_models,
    ensure_output_dirs,
    load_cached_fold,
)


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


def stretch_embedding(z: np.ndarray, scale: float = 1.45, lower: float = 1.0, upper: float = 99.0) -> np.ndarray:
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
    cluster_scale: float = 2.2,
    center_pull: float = 0.72,
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


def collect_split_data(cache_dir: Path, model_key: str, folds: list[int], feature_key: str, value_key: str):
    per_fold = {}
    for fold_id in folds:
        row = load_cached_fold(cache_dir, model_key, fold_id)
        per_fold[fold_id] = {
            "features": row[feature_key],
            "values": row[value_key].reshape(-1),
            "fold_ids": np.full(row[value_key].shape[0], fold_id, dtype=np.int64),
        }

    agg = aggregate_cached(cache_dir, model_key, folds)
    per_fold["all5"] = {
        "features": agg[feature_key],
        "values": agg[value_key].reshape(-1),
        "fold_ids": np.concatenate([per_fold[f]["fold_ids"] for f in folds], axis=0),
    }
    return per_fold


def make_entries(split_data, split_order, args):
    entries = {}
    for split_name in split_order:
        z = project_umap(
            preprocess_features(split_data[split_name]["features"]),
            random_state=args.random_state,
            n_neighbors=args.umap_neighbors,
            min_dist=args.umap_min_dist,
        )
        z = stretch_embedding(z, scale=args.stretch_scale)
        if split_name == "all5":
            z = inflate_all5_islands(
                z,
                split_data[split_name]["fold_ids"],
                cluster_scale=args.all5_cluster_scale,
                center_pull=args.all5_center_pull,
            )
        entries[split_name] = {
            "z": z,
            "values": split_data[split_name]["values"],
        }
    return entries


def add_class_hint(fig):
    fig.text(
        0.25,
        0.975,
        "Arousal tendency: low -> Sad, Neutral | high -> Happy, Angry",
        ha="center",
        va="center",
        fontsize=11,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "#f5f5f5", "edgecolor": "#bdbdbd"},
    )
    fig.text(
        0.75,
        0.975,
        "Valence tendency: low -> Sad, Angry | mid -> Neutral | high -> Happy",
        ha="center",
        va="center",
        fontsize=11,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "#f5f5f5", "edgecolor": "#bdbdbd"},
    )


def plot_pair_grid(arousal_entries, valence_entries, out_path: Path, title: str, split_order):
    rows = len(split_order)
    fig, axes = plt.subplots(rows, 2, figsize=(13.5, 3.35 * rows))
    if rows == 1:
        axes = np.asarray([axes])

    arousal_values = np.concatenate([arousal_entries[s]["values"] for s in split_order], axis=0)
    valence_values = np.concatenate([valence_entries[s]["values"] for s in split_order], axis=0)
    a_vmin, a_vmax = float(np.min(arousal_values)), float(np.max(arousal_values))
    v_vmin, v_vmax = float(np.min(valence_values)), float(np.max(valence_values))

    left_scatter = None
    right_scatter = None
    for row_idx, split_name in enumerate(split_order):
        row_title = "All 5 Folds" if split_name == "all5" else f"Fold {split_name}"

        ax_left = axes[row_idx, 0]
        entry_left = arousal_entries[split_name]
        left_scatter = ax_left.scatter(
            entry_left["z"][:, 0],
            entry_left["z"][:, 1],
            c=entry_left["values"],
            cmap="coolwarm",
            vmin=a_vmin,
            vmax=a_vmax,
            s=14,
            alpha=0.88,
            linewidths=0,
        )
        set_tight_bounds(ax_left, entry_left["z"])
        ax_left.set_title(f"{row_title} - Arousal", fontsize=11)
        ax_left.set_xticks([])
        ax_left.set_yticks([])

        ax_right = axes[row_idx, 1]
        entry_right = valence_entries[split_name]
        right_scatter = ax_right.scatter(
            entry_right["z"][:, 0],
            entry_right["z"][:, 1],
            c=entry_right["values"],
            cmap="coolwarm",
            vmin=v_vmin,
            vmax=v_vmax,
            s=14,
            alpha=0.88,
            linewidths=0,
        )
        set_tight_bounds(ax_right, entry_right["z"])
        ax_right.set_title(f"{row_title} - Valence", fontsize=11)
        ax_right.set_xticks([])
        ax_right.set_yticks([])

    fig.subplots_adjust(right=0.90, top=0.94, wspace=0.10, hspace=0.18)
    cax_left = fig.add_axes([0.915, 0.54, 0.012, 0.28])
    cax_right = fig.add_axes([0.945, 0.54, 0.012, 0.28])
    cbar_left = fig.colorbar(left_scatter, cax=cax_left)
    cbar_right = fig.colorbar(right_scatter, cax=cax_right)
    cbar_left.set_label("Normalized arousal", rotation=90)
    cbar_right.set_label("Normalized valence", rotation=90)

    fig.suptitle(title, fontsize=16, y=0.995)
    add_class_hint(fig)
    plt.savefig(out_path)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    add_shared_args(parser)
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--umap_neighbors", type=int, default=15)
    parser.add_argument("--umap_min_dist", type=float, default=0.3)
    parser.add_argument("--stretch_scale", type=float, default=1.45)
    parser.add_argument("--all5_cluster_scale", type=float, default=2.2)
    parser.add_argument("--all5_center_pull", type=float, default=0.72)
    args = parser.parse_args()

    dirs = ensure_output_dirs(args.output)
    ensure_cache_for_models(args)

    split_order = ["all5"] + list(args.folds)

    jobs = [
        ("full", "full_av_pair_umap_grid.pdf", "Full ASPIRE: paired UMAP projections by fold"),
        ("noav", "noav_av_pair_umap_grid.pdf", "No-AV-Sup: paired UMAP projections by fold"),
    ]

    for model_key, out_name, title in jobs:
        arousal_data = collect_split_data(dirs["cache"], model_key, args.folds, "arousal_embedding", "arousal_true")
        valence_data = collect_split_data(dirs["cache"], model_key, args.folds, "valence_embedding", "valence_true")
        arousal_entries = make_entries(arousal_data, split_order, args)
        valence_entries = make_entries(valence_data, split_order, args)
        plot_pair_grid(arousal_entries, valence_entries, dirs["plots"] / out_name, title, split_order)

    print("Saved:")
    print(dirs["plots"] / "full_av_pair_umap_grid.pdf")
    print(dirs["plots"] / "noav_av_pair_umap_grid.pdf")


if __name__ == "__main__":
    main()
