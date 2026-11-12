#!/usr/bin/env python3
"""Plot fixed-parameter UMAPs per fold and all-5-fold for Full vs No-AV-Sup."""
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

sys.path.append(str(Path(__file__).resolve().parent.parent))

from scripts.full_vs_noav_analysis_utils import (
    LABEL_NAMES,
    add_shared_args,
    aggregate_cached,
    ensure_cache_for_models,
    ensure_output_dirs,
    load_cached_fold,
)

from umap import UMAP


def label_names_from_ids(label_ids: np.ndarray):
    return np.array([LABEL_NAMES[int(i)] for i in label_ids])


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
    """Affine-stretch each axis independently so the panel uses space better."""
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
    """For the aggregated 5-fold view only, enlarge each fold island and pull centroids inward."""
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


def plot_panel(ax, z, labels, title, show_legend=False):
    sns.scatterplot(
        x=z[:, 0],
        y=z[:, 1],
        hue=labels,
        hue_order=LABEL_NAMES,
        palette="tab10",
        s=12,
        linewidth=0,
        alpha=0.85,
        ax=ax,
        legend=show_legend,
    )
    set_tight_bounds(ax, z)
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")


def plot_side_by_side(z_full, labels_full, z_noav, labels_noav, out_path, title_suffix):
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.8))
    plot_panel(axes[0], z_full, labels_full, f"Full ASPIRE\n{title_suffix}", show_legend=False)
    plot_panel(axes[1], z_noav, labels_noav, f"No-AV-Sup\n{title_suffix}", show_legend=True)
    handles, labels = axes[1].get_legend_handles_labels()
    if handles:
        axes[1].legend(handles=handles, labels=labels, bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_model_grid(entries, out_path, model_title):
    fig, axes = plt.subplots(2, 3, figsize=(15, 9.5))
    flat_axes = axes.reshape(-1)
    for idx, entry in enumerate(entries):
        show_legend = idx == len(entries) - 1
        plot_panel(flat_axes[idx], entry["z"], entry["labels"], entry["title"], show_legend=show_legend)
        if show_legend:
            handles, labels = flat_axes[idx].get_legend_handles_labels()
            if handles:
                flat_axes[idx].legend(handles=handles, labels=labels, bbox_to_anchor=(1.02, 1), loc="upper left")
        elif flat_axes[idx].legend_ is not None:
            flat_axes[idx].legend_.remove()
    fig.suptitle(model_title, fontsize=15)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(out_path)
    plt.close()


def collect_split_data(cache_dir: Path, model_key: str, folds: list[int], feature_key: str):
    per_fold = {}
    for fold_id in folds:
        row = load_cached_fold(cache_dir, model_key, fold_id)
        if row[feature_key].size == 0:
            raise RuntimeError(f"Missing {feature_key} for {model_key} fold {fold_id}")
        per_fold[fold_id] = {
            "features": row[feature_key],
            "labels": row["y_true"],
            "fold_ids": np.full(row["y_true"].shape[0], fold_id, dtype=np.int64),
        }

    agg = aggregate_cached(cache_dir, model_key, folds)
    per_fold["all5"] = {
        "features": agg[feature_key],
        "labels": agg["y_true"],
        "fold_ids": np.concatenate(
            [per_fold[fold_id]["fold_ids"] for fold_id in folds],
            axis=0,
        ),
    }
    return per_fold


def main():
    parser = argparse.ArgumentParser()
    add_shared_args(parser)
    parser.add_argument("--feature_key", default="fused_features", choices=["fused_features", "final_representation"])
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--umap_neighbors", type=int, default=15)
    parser.add_argument("--umap_min_dist", type=float, default=0.3)
    parser.add_argument("--no_stretch_axes", action="store_true")
    parser.add_argument("--all5_cluster_scale", type=float, default=1.8)
    parser.add_argument("--all5_center_pull", type=float, default=0.78)
    args = parser.parse_args()

    dirs = ensure_output_dirs(args.output)
    ensure_cache_for_models(args)

    full_data = collect_split_data(dirs["cache"], "full", args.folds, args.feature_key)
    noav_data = collect_split_data(dirs["cache"], "noav", args.folds, args.feature_key)

    split_order = ["all5"] + list(args.folds)
    full_grid_entries = []
    noav_grid_entries = []

    for split_name in split_order:
        title_suffix = "All 5 Folds" if split_name == "all5" else f"Fold {split_name}"

        full_scaled = preprocess_features(full_data[split_name]["features"])
        noav_scaled = preprocess_features(noav_data[split_name]["features"])

        z_full = project_umap(
            full_scaled,
            random_state=args.random_state,
            n_neighbors=args.umap_neighbors,
            min_dist=args.umap_min_dist,
        )
        z_noav = project_umap(
            noav_scaled,
            random_state=args.random_state,
            n_neighbors=args.umap_neighbors,
            min_dist=args.umap_min_dist,
        )
        if not args.no_stretch_axes:
            z_full = stretch_embedding(z_full)
            z_noav = stretch_embedding(z_noav)
        if split_name == "all5":
            z_full = inflate_all5_islands(
                z_full,
                full_data[split_name]["fold_ids"],
                cluster_scale=args.all5_cluster_scale,
                center_pull=args.all5_center_pull,
            )
            z_noav = inflate_all5_islands(
                z_noav,
                noav_data[split_name]["fold_ids"],
                cluster_scale=args.all5_cluster_scale,
                center_pull=args.all5_center_pull,
            )

        labels_full = label_names_from_ids(full_data[split_name]["labels"])
        labels_noav = label_names_from_ids(noav_data[split_name]["labels"])

        np.save(dirs["tables"] / f"umap_fixed_full_{split_name}.npy", z_full)
        np.save(dirs["tables"] / f"umap_fixed_noav_{split_name}.npy", z_noav)

        out_name = f"umap_fixed_side_by_side_{split_name}.pdf"
        plot_side_by_side(
            z_full,
            labels_full,
            z_noav,
            labels_noav,
            dirs["plots"] / out_name,
            title_suffix,
        )

        full_grid_entries.append(
            {
                "z": z_full,
                "labels": labels_full,
                "title": title_suffix,
            }
        )
        noav_grid_entries.append(
            {
                "z": z_noav,
                "labels": labels_noav,
                "title": title_suffix,
            }
        )

    plot_model_grid(
        full_grid_entries,
        dirs["plots"] / "umap_fixed_full_grid.pdf",
        f"Full ASPIRE UMAP ({args.feature_key}, n_neighbors={args.umap_neighbors}, min_dist={args.umap_min_dist})",
    )
    plot_model_grid(
        noav_grid_entries,
        dirs["plots"] / "umap_fixed_noav_grid.pdf",
        f"No-AV-Sup UMAP ({args.feature_key}, n_neighbors={args.umap_neighbors}, min_dist={args.umap_min_dist})",
    )

    with open(dirs["tables"] / "umap_fixed_protocol.txt", "w", encoding="utf-8") as fh:
        fh.write(f"feature_key={args.feature_key}\n")
        fh.write(f"random_state={args.random_state}\n")
        fh.write(f"umap_neighbors={args.umap_neighbors}\n")
        fh.write(f"umap_min_dist={args.umap_min_dist}\n")
        fh.write(f"stretch_axes={not args.no_stretch_axes}\n")
        fh.write(f"all5_cluster_scale={args.all5_cluster_scale}\n")
        fh.write(f"all5_center_pull={args.all5_center_pull}\n")
        fh.write(f"folds={args.folds}\n")

    print("Saved:")
    print(dirs["plots"] / "umap_fixed_side_by_side_all5.pdf")
    for fold_id in args.folds:
        print(dirs["plots"] / f"umap_fixed_side_by_side_{fold_id}.pdf")
    print(dirs["plots"] / "umap_fixed_full_grid.pdf")
    print(dirs["plots"] / "umap_fixed_noav_grid.pdf")
    print(dirs["tables"] / "umap_fixed_protocol.txt")


if __name__ == "__main__":
    main()
