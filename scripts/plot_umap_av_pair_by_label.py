
#!/usr/bin/env python3
"""Plot paired UMAP grids from A/V embeddings, colored by discrete emotion labels."""
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

from scripts.full_vs_noav_analysis_utils import (  # noqa: E402
    LABEL_NAMES,
    add_shared_args,
    aggregate_cached,
    ensure_cache_for_models,
    ensure_output_dirs,
    load_cached_fold,
)


LABEL_ID_TO_NAME = {0: "Happy", 1: "Sad", 2: "Neutral", 3: "Angry"}

AROUSAL_COLOR_BY_LABEL = {
    1: "#c62828",  # Sad: low arousal
    2: "#f2a3a3",  # Neutral: lower-mid arousal (light red)
    0: "#8ecae6",  # Happy: high arousal
    3: "#1565c0",  # Angry: highest arousal
}
VALENCE_COLOR_BY_LABEL = {
    1: "#c62828",  # Sad: low valence
    3: "#f2a3a3",  # Angry: low-mid valence (light red)
    2: "#8ecae6",  # Neutral: mid valence
    0: "#1565c0",  # Happy: high valence
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
    cluster_scale: float = 2.8,
    center_pull: float = 0.58,
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


def collect_split_data(cache_dir: Path, model_key: str, folds: list[int], feature_key: str):
    per_fold = {}
    for fold_id in folds:
        row = load_cached_fold(cache_dir, model_key, fold_id)
        per_fold[fold_id] = {
            "features": row[feature_key],
            "labels": row["y_true"].reshape(-1),
            "fold_ids": np.full(row["y_true"].shape[0], fold_id, dtype=np.int64),
        }

    agg = aggregate_cached(cache_dir, model_key, folds)
    per_fold["all5"] = {
        "features": agg[feature_key],
        "labels": agg["y_true"].reshape(-1),
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
            "labels": split_data[split_name]["labels"],
        }
    return entries


def scatter_by_label(ax, z, labels, color_by_label):
    for label_id in [0, 1, 2, 3]:
        mask = labels == label_id
        if np.any(mask):
            ax.scatter(
                z[mask, 0],
                z[mask, 1],
                s=14,
                alpha=0.88,
                linewidths=0,
                c=color_by_label[label_id],
            )


def make_legend_handles(order, color_by_label):
    handles = []
    for label_id in order:
        handles.append(
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor=color_by_label[label_id],
                markersize=8,
                label=LABEL_ID_TO_NAME[label_id],
            )
        )
    return handles


def plot_pair_grid(arousal_entries, valence_entries, out_path: Path, title: str, split_order):
    rows = len(split_order)
    fig, axes = plt.subplots(rows, 2, figsize=(13.5, 3.35 * rows))
    if rows == 1:
        axes = np.asarray([axes])

    for row_idx, split_name in enumerate(split_order):
        row_title = "All 5 Folds" if split_name == "all5" else f"Fold {split_name}"

        ax_left = axes[row_idx, 0]
        entry_left = arousal_entries[split_name]
        scatter_by_label(ax_left, entry_left["z"], entry_left["labels"], AROUSAL_COLOR_BY_LABEL)
        set_tight_bounds(ax_left, entry_left["z"])
        ax_left.set_title(f"{row_title} - Arousal embedding", fontsize=11)
        ax_left.set_xticks([])
        ax_left.set_yticks([])

        ax_right = axes[row_idx, 1]
        entry_right = valence_entries[split_name]
        scatter_by_label(ax_right, entry_right["z"], entry_right["labels"], VALENCE_COLOR_BY_LABEL)
        set_tight_bounds(ax_right, entry_right["z"])
        ax_right.set_title(f"{row_title} - Valence embedding", fontsize=11)
        ax_right.set_xticks([])
        ax_right.set_yticks([])

    fig.subplots_adjust(top=0.92, hspace=0.18, wspace=0.08)
    fig.suptitle(title, fontsize=16, y=0.985)

    left_handles = make_legend_handles(AROUSAL_LEGEND_ORDER, AROUSAL_COLOR_BY_LABEL)
    right_handles = make_legend_handles(VALENCE_LEGEND_ORDER, VALENCE_COLOR_BY_LABEL)
    fig.legend(
        handles=left_handles,
        loc="upper center",
        bbox_to_anchor=(0.28, 0.945),
        ncol=4,
        frameon=False,
        title="Arousal palette",
    )
    fig.legend(
        handles=right_handles,
        loc="upper center",
        bbox_to_anchor=(0.73, 0.945),
        ncol=4,
        frameon=False,
        title="Valence palette",
    )

    plt.savefig(out_path)
    plt.close()


def plot_pair_single(arousal_entry, valence_entry, out_path: Path, title: str):
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 6.1))

    scatter_by_label(axes[0], arousal_entry["z"], arousal_entry["labels"], AROUSAL_COLOR_BY_LABEL)
    set_tight_bounds(axes[0], arousal_entry["z"])
    axes[0].set_title("Arousal Embedding", fontsize=13)
    axes[0].set_xticks([])
    axes[0].set_yticks([])
    axes[0].legend(
        handles=make_legend_handles(AROUSAL_LEGEND_ORDER, AROUSAL_COLOR_BY_LABEL),
        loc="upper center",
        bbox_to_anchor=(0.5, 1.01),
        ncol=4,
        frameon=False,
        fontsize=9,
        handletextpad=0.4,
        columnspacing=0.9,
    )

    scatter_by_label(axes[1], valence_entry["z"], valence_entry["labels"], VALENCE_COLOR_BY_LABEL)
    set_tight_bounds(axes[1], valence_entry["z"])
    axes[1].set_title("Valence Embedding", fontsize=13)
    axes[1].set_xticks([])
    axes[1].set_yticks([])
    axes[1].legend(
        handles=make_legend_handles(VALENCE_LEGEND_ORDER, VALENCE_COLOR_BY_LABEL),
        loc="upper center",
        bbox_to_anchor=(0.5, 1.01),
        ncol=4,
        frameon=False,
        fontsize=9,
        handletextpad=0.4,
        columnspacing=0.9,
    )

    fig.suptitle(title, fontsize=16, y=0.99)
    fig.subplots_adjust(top=0.84, wspace=0.08)
    plt.savefig(out_path, bbox_inches="tight", pad_inches=0.02)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    add_shared_args(parser)
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--umap_neighbors", type=int, default=15)
    parser.add_argument("--umap_min_dist", type=float, default=0.3)
    parser.add_argument("--stretch_scale", type=float, default=1.55)
    parser.add_argument("--all5_cluster_scale", type=float, default=2.8)
    parser.add_argument("--all5_center_pull", type=float, default=0.58)
    args = parser.parse_args()

    dirs = ensure_output_dirs(args.output)
    ensure_cache_for_models(args)

    split_order = ["all5"] + list(args.folds)
    jobs = [
        ("full", "full_av_pair_umap_by_label_grid.pdf", "Full ASPIRE: A/V embeddings colored by emotion labels"),
        ("noav", "noav_av_pair_umap_by_label_grid.pdf", "No-AV-Sup: A/V embeddings colored by emotion labels"),
    ]

    for model_key, out_name, title in jobs:
        arousal_data = collect_split_data(dirs["cache"], model_key, args.folds, "arousal_embedding")
        valence_data = collect_split_data(dirs["cache"], model_key, args.folds, "valence_embedding")
        arousal_entries = make_entries(arousal_data, split_order, args)
        valence_entries = make_entries(valence_data, split_order, args)
        plot_pair_grid(arousal_entries, valence_entries, dirs["plots"] / out_name, title, split_order)

        for split_name in split_order:
            split_tag = "all5" if split_name == "all5" else f"fold{split_name}"
            split_title = "All 5 folds" if split_name == "all5" else f"Fold {split_name}"
            plot_pair_single(
                arousal_entries[split_name],
                valence_entries[split_name],
                dirs["plots"] / f"{model_key}_av_pair_umap_by_label_side_by_side_{split_tag}.pdf",
                split_title,
            )

    print("Saved:")
    print(dirs["plots"] / "full_av_pair_umap_by_label_grid.pdf")
    print(dirs["plots"] / "noav_av_pair_umap_by_label_grid.pdf")


if __name__ == "__main__":
    main()
