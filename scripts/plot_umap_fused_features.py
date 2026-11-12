#!/usr/bin/env python3
"""Plot UMAP and t-SNE for fused features of Full ASPIRE vs No-AV-Sup."""
from __future__ import annotations

import argparse
import itertools
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

sys.path.append(str(Path(__file__).resolve().parent.parent))

from scripts.full_vs_noav_analysis_utils import (
    LABEL_IDS,
    LABEL_NAMES,
    add_shared_args,
    aggregate_cached,
    ensure_cache_for_models,
    ensure_output_dirs,
)

try:
    from umap import UMAP

    HAS_UMAP = True
except Exception:
    HAS_UMAP = False


def label_names_from_ids(label_ids: np.ndarray):
    return np.array([LABEL_NAMES[int(i)] for i in label_ids])


def slugify_float(value: float) -> str:
    return str(value).replace(".", "p")


def plot_scatter(z, labels, out_path, title, xlabel="Dim 1", ylabel="Dim 2"):
    plt.figure(figsize=(7, 6))
    sns.scatterplot(
        x=z[:, 0],
        y=z[:, 1],
        hue=labels,
        hue_order=LABEL_NAMES,
        palette="tab10",
        s=10,
        linewidth=0,
    )
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_panel(ax, z, labels, title, xlabel="Dim 1", ylabel="Dim 2", show_legend=False):
    sns.scatterplot(
        x=z[:, 0],
        y=z[:, 1],
        hue=labels,
        hue_order=LABEL_NAMES,
        palette="tab10",
        s=10,
        linewidth=0,
        ax=ax,
        legend=show_legend,
    )
    ax.set_title(title, fontsize=10)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)


def preprocess_features(full_features: np.ndarray, noav_features: np.ndarray):
    all_features = np.concatenate([full_features, noav_features], axis=0)
    scaler = StandardScaler()
    scaler.fit(all_features)
    return scaler.transform(full_features), scaler.transform(noav_features)


def fit_joint_pca(
    full_features: np.ndarray,
    noav_features: np.ndarray,
    n_components: int,
):
    all_features = np.concatenate([full_features, noav_features], axis=0)
    max_components = min(n_components, all_features.shape[0], all_features.shape[1])
    pca = PCA(n_components=max_components, random_state=42)
    pca.fit(all_features)
    return pca.transform(full_features), pca.transform(noav_features), max_components


def project_umap(
    features: np.ndarray,
    random_state: int = 42,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
):
    if not HAS_UMAP:
        return None
    reducer = UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        random_state=random_state,
    )
    return reducer.fit_transform(features)


def project_tsne(
    features: np.ndarray,
    random_state: int = 42,
    perplexity: float = 30.0,
    max_iter: int = 1000,
):
    reducer = TSNE(
        n_components=2,
        perplexity=perplexity,
        init="pca",
        learning_rate="auto",
        max_iter=max_iter,
        random_state=random_state,
    )
    return reducer.fit_transform(features)


def plot_side_by_side(
    z_full,
    labels_full,
    z_noav,
    labels_noav,
    out_path,
    title_left,
    title_right,
    xlabel="Dim 1",
    ylabel="Dim 2",
):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), sharex=False, sharey=False)
    for ax, z, labels, title in [
        (axes[0], z_full, labels_full, title_left),
        (axes[1], z_noav, labels_noav, title_right),
    ]:
        plot_panel(ax, z, labels, title, xlabel=xlabel, ylabel=ylabel, show_legend=ax is axes[1])
    handles, labels = axes[1].get_legend_handles_labels()
    if handles:
        axes[1].legend(handles=handles, labels=labels, bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_sweep_grid(
    entries,
    out_path,
    super_title,
    xlabel,
    ylabel,
):
    if not entries:
        return

    n_panels = len(entries)
    ncols = 2
    nrows = int(np.ceil(n_panels / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(13, 4.8 * nrows), squeeze=False)
    flat_axes = axes.reshape(-1)

    for idx, entry in enumerate(entries):
        ax = flat_axes[idx]
        plot_panel(
            ax,
            entry["z"],
            entry["labels"],
            entry["title"],
            xlabel=xlabel,
            ylabel=ylabel,
            show_legend=idx == n_panels - 1,
        )
        if idx == n_panels - 1:
            handles, labels = ax.get_legend_handles_labels()
            if handles:
                ax.legend(handles=handles, labels=labels, bbox_to_anchor=(1.02, 1), loc="upper left")
        elif ax.legend_ is not None:
            ax.legend_.remove()

    for idx in range(n_panels, len(flat_axes)):
        flat_axes[idx].axis("off")

    fig.suptitle(super_title, fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(out_path)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    add_shared_args(parser)
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--umap_neighbors", type=int, default=15)
    parser.add_argument("--umap_min_dist", type=float, default=0.1)
    parser.add_argument("--umap_neighbors_list", nargs="+", type=int, default=[15, 30])
    parser.add_argument("--umap_min_dist_list", nargs="+", type=float, default=[0.1, 0.3])
    parser.add_argument("--tsne_perplexity", type=float, default=30.0)
    parser.add_argument("--tsne_perplexity_list", nargs="+", type=float, default=[10.0, 30.0, 50.0])
    parser.add_argument("--tsne_pca_dim", type=int, default=50)
    parser.add_argument("--tsne_max_iter", type=int, default=1000)
    parser.add_argument("--run_sweep", action="store_true")
    args = parser.parse_args()

    dirs = ensure_output_dirs(args.output)
    ensure_cache_for_models(args)

    full = aggregate_cached(dirs["cache"], "full", args.folds)
    noav = aggregate_cached(dirs["cache"], "noav", args.folds)

    if full["fused_features"].size == 0 or noav["fused_features"].size == 0:
        raise RuntimeError("Missing fused_features in cache. Re-run with checkpoints that support return_intermediate=True")

    labels_full = label_names_from_ids(full["y_true"])
    labels_noav = label_names_from_ids(noav["y_true"])

    full_scaled, noav_scaled = preprocess_features(full["fused_features"], noav["fused_features"])
    saved_paths = []

    if HAS_UMAP:
        z_umap_full = project_umap(
            full_scaled,
            random_state=args.random_state,
            n_neighbors=args.umap_neighbors,
            min_dist=args.umap_min_dist,
        )
        z_umap_noav = project_umap(
            noav_scaled,
            random_state=args.random_state,
            n_neighbors=args.umap_neighbors,
            min_dist=args.umap_min_dist,
        )
        np.save(dirs["tables"] / "umap_fused_full.npy", z_umap_full)
        np.save(dirs["tables"] / "umap_fused_noav.npy", z_umap_noav)
        saved_paths.extend(
            [
                dirs["tables"] / "umap_fused_full.npy",
                dirs["tables"] / "umap_fused_noav.npy",
            ]
        )
        plot_scatter(
            z_umap_full,
            labels_full,
            dirs["plots"] / "umap_fused_full.pdf",
            "Full ASPIRE: fused_features",
            xlabel="UMAP-1",
            ylabel="UMAP-2",
        )
        saved_paths.append(dirs["plots"] / "umap_fused_full.pdf")
        plot_scatter(
            z_umap_noav,
            labels_noav,
            dirs["plots"] / "umap_fused_noav.pdf",
            "No-AV-Sup: fused_features",
            xlabel="UMAP-1",
            ylabel="UMAP-2",
        )
        saved_paths.append(dirs["plots"] / "umap_fused_noav.pdf")
        plot_side_by_side(
            z_umap_full,
            labels_full,
            z_umap_noav,
            labels_noav,
            dirs["plots"] / "umap_fused_side_by_side.pdf",
            "Full ASPIRE",
            "No-AV-Sup",
            xlabel="UMAP-1",
            ylabel="UMAP-2",
        )
        saved_paths.append(dirs["plots"] / "umap_fused_side_by_side.pdf")

    full_pca, noav_pca, used_pca_dim = fit_joint_pca(
        full_scaled,
        noav_scaled,
        n_components=args.tsne_pca_dim,
    )
    z_tsne_full = project_tsne(
        full_pca,
        random_state=args.random_state,
        perplexity=args.tsne_perplexity,
        max_iter=args.tsne_max_iter,
    )
    z_tsne_noav = project_tsne(
        noav_pca,
        random_state=args.random_state,
        perplexity=args.tsne_perplexity,
        max_iter=args.tsne_max_iter,
    )
    np.save(dirs["tables"] / "tsne_fused_full.npy", z_tsne_full)
    np.save(dirs["tables"] / "tsne_fused_noav.npy", z_tsne_noav)
    saved_paths.extend(
        [
            dirs["tables"] / "tsne_fused_full.npy",
            dirs["tables"] / "tsne_fused_noav.npy",
        ]
    )
    plot_scatter(
        z_tsne_full,
        labels_full,
        dirs["plots"] / "tsne_fused_full.pdf",
        f"Full ASPIRE: fused_features (PCA-{used_pca_dim} -> t-SNE)",
        xlabel="t-SNE-1",
        ylabel="t-SNE-2",
    )
    saved_paths.append(dirs["plots"] / "tsne_fused_full.pdf")
    plot_scatter(
        z_tsne_noav,
        labels_noav,
        dirs["plots"] / "tsne_fused_noav.pdf",
        f"No-AV-Sup: fused_features (PCA-{used_pca_dim} -> t-SNE)",
        xlabel="t-SNE-1",
        ylabel="t-SNE-2",
    )
    saved_paths.append(dirs["plots"] / "tsne_fused_noav.pdf")
    plot_side_by_side(
        z_tsne_full,
        labels_full,
        z_tsne_noav,
        labels_noav,
        dirs["plots"] / "tsne_fused_side_by_side.pdf",
        "Full ASPIRE",
        "No-AV-Sup",
        xlabel="t-SNE-1",
        ylabel="t-SNE-2",
    )
    saved_paths.append(dirs["plots"] / "tsne_fused_side_by_side.pdf")

    if args.run_sweep:
        if HAS_UMAP:
            umap_entries_full = []
            umap_entries_noav = []
            for n_neighbors, min_dist in itertools.product(args.umap_neighbors_list, args.umap_min_dist_list):
                z_full_sweep = project_umap(
                    full_scaled,
                    random_state=args.random_state,
                    n_neighbors=n_neighbors,
                    min_dist=min_dist,
                )
                z_noav_sweep = project_umap(
                    noav_scaled,
                    random_state=args.random_state,
                    n_neighbors=n_neighbors,
                    min_dist=min_dist,
                )
                suffix = f"n{n_neighbors}_d{slugify_float(min_dist)}"
                np.save(dirs["tables"] / f"umap_fused_full_{suffix}.npy", z_full_sweep)
                np.save(dirs["tables"] / f"umap_fused_noav_{suffix}.npy", z_noav_sweep)
                plot_side_by_side(
                    z_full_sweep,
                    labels_full,
                    z_noav_sweep,
                    labels_noav,
                    dirs["plots"] / f"umap_fused_side_by_side_{suffix}.pdf",
                    f"Full ASPIRE\nn_neighbors={n_neighbors}, min_dist={min_dist}",
                    f"No-AV-Sup\nn_neighbors={n_neighbors}, min_dist={min_dist}",
                    xlabel="UMAP-1",
                    ylabel="UMAP-2",
                )
                saved_paths.extend(
                    [
                        dirs["tables"] / f"umap_fused_full_{suffix}.npy",
                        dirs["tables"] / f"umap_fused_noav_{suffix}.npy",
                        dirs["plots"] / f"umap_fused_side_by_side_{suffix}.pdf",
                    ]
                )
                umap_entries_full.append(
                    {
                        "z": z_full_sweep,
                        "labels": labels_full,
                        "title": f"n_neighbors={n_neighbors}, min_dist={min_dist}",
                    }
                )
                umap_entries_noav.append(
                    {
                        "z": z_noav_sweep,
                        "labels": labels_noav,
                        "title": f"n_neighbors={n_neighbors}, min_dist={min_dist}",
                    }
                )

            plot_sweep_grid(
                umap_entries_full,
                dirs["plots"] / "umap_sweep_full_grid.pdf",
                "Full ASPIRE UMAP Sweep",
                xlabel="UMAP-1",
                ylabel="UMAP-2",
            )
            plot_sweep_grid(
                umap_entries_noav,
                dirs["plots"] / "umap_sweep_noav_grid.pdf",
                "No-AV-Sup UMAP Sweep",
                xlabel="UMAP-1",
                ylabel="UMAP-2",
            )
            saved_paths.extend(
                [
                    dirs["plots"] / "umap_sweep_full_grid.pdf",
                    dirs["plots"] / "umap_sweep_noav_grid.pdf",
                ]
            )

        tsne_entries_full = []
        tsne_entries_noav = []
        for perplexity in args.tsne_perplexity_list:
            z_full_sweep = project_tsne(
                full_pca,
                random_state=args.random_state,
                perplexity=perplexity,
                max_iter=args.tsne_max_iter,
            )
            z_noav_sweep = project_tsne(
                noav_pca,
                random_state=args.random_state,
                perplexity=perplexity,
                max_iter=args.tsne_max_iter,
            )
            suffix = f"p{slugify_float(perplexity)}"
            np.save(dirs["tables"] / f"tsne_fused_full_{suffix}.npy", z_full_sweep)
            np.save(dirs["tables"] / f"tsne_fused_noav_{suffix}.npy", z_noav_sweep)
            plot_side_by_side(
                z_full_sweep,
                labels_full,
                z_noav_sweep,
                labels_noav,
                dirs["plots"] / f"tsne_fused_side_by_side_{suffix}.pdf",
                f"Full ASPIRE\nperplexity={perplexity}",
                f"No-AV-Sup\nperplexity={perplexity}",
                xlabel="t-SNE-1",
                ylabel="t-SNE-2",
            )
            saved_paths.extend(
                [
                    dirs["tables"] / f"tsne_fused_full_{suffix}.npy",
                    dirs["tables"] / f"tsne_fused_noav_{suffix}.npy",
                    dirs["plots"] / f"tsne_fused_side_by_side_{suffix}.pdf",
                ]
            )
            tsne_entries_full.append(
                {
                    "z": z_full_sweep,
                    "labels": labels_full,
                    "title": f"perplexity={perplexity}",
                }
            )
            tsne_entries_noav.append(
                {
                    "z": z_noav_sweep,
                    "labels": labels_noav,
                    "title": f"perplexity={perplexity}",
                }
            )

        plot_sweep_grid(
            tsne_entries_full,
            dirs["plots"] / "tsne_sweep_full_grid.pdf",
            f"Full ASPIRE t-SNE Sweep (PCA-{used_pca_dim})",
            xlabel="t-SNE-1",
            ylabel="t-SNE-2",
        )
        plot_sweep_grid(
            tsne_entries_noav,
            dirs["plots"] / "tsne_sweep_noav_grid.pdf",
            f"No-AV-Sup t-SNE Sweep (PCA-{used_pca_dim})",
            xlabel="t-SNE-1",
            ylabel="t-SNE-2",
        )
        saved_paths.extend(
            [
                dirs["plots"] / "tsne_sweep_full_grid.pdf",
                dirs["plots"] / "tsne_sweep_noav_grid.pdf",
            ]
        )

    config_lines = [
        f"random_state={args.random_state}",
        f"umap_available={HAS_UMAP}",
        f"umap_neighbors={args.umap_neighbors}",
        f"umap_min_dist={args.umap_min_dist}",
        f"umap_neighbors_list={args.umap_neighbors_list}",
        f"umap_min_dist_list={args.umap_min_dist_list}",
        f"tsne_perplexity={args.tsne_perplexity}",
        f"tsne_perplexity_list={args.tsne_perplexity_list}",
        f"tsne_pca_dim={used_pca_dim}",
        f"tsne_max_iter={args.tsne_max_iter}",
        f"run_sweep={args.run_sweep}",
    ]
    with open(dirs["tables"] / "manifold_protocol.txt", "w", encoding="utf-8") as fh:
        fh.write("\n".join(config_lines) + "\n")
    saved_paths.append(dirs["tables"] / "manifold_protocol.txt")

    print("Saved:")
    for path in saved_paths:
        print(path)


if __name__ == "__main__":
    main()
