#!/usr/bin/env python3
"""Plot UMAP/t-SNE for MSP-Podcast Full ASPIRE on test1 and test2 separately."""
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
import torch
import yaml
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

sys.path.append(str(Path(__file__).resolve().parent.parent))

from data.msp_podcast_dataset import get_msp_podcast_dataloaders
from models.method2_crossattn_av import Method2CrossAttentionAV

try:
    from umap import UMAP

    HAS_UMAP = True
except Exception:
    HAS_UMAP = False


LABEL_NAMES = ["Happy", "Sad", "Neutral", "Angry"]


def load_config(config_path: str):
    with open(config_path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def slugify_float(value: float) -> str:
    return str(value).replace(".", "p")


def ensure_output_dirs(output_dir: str | Path):
    out = Path(output_dir)
    plots = out / "plots"
    tables = out / "tables"
    out.mkdir(parents=True, exist_ok=True)
    plots.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)
    return {"root": out, "plots": plots, "tables": tables}


def resolve_checkpoint_path(args_checkpoint: str | None, config: dict) -> str:
    if args_checkpoint:
        return args_checkpoint

    candidates = [
        Path(config["paths"]["checkpoint_dir"]) / "best_model.pt",
        Path("checkpoints/method2_msp_wpro_best_bak.pt"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)

    return str(candidates[0])


def build_model(config):
    return Method2CrossAttentionAV(
        num_emotions=config["model"]["num_emotions"],
        wavlm_model_name=config["model"]["wavlm_model_name"],
        wavlm_prosody_model_name=config["model"].get("wavlm_prosody_model_name", None),
        roberta_model_name=config["model"]["roberta_model_name"],
        hidden_dim=config["model"]["hidden_dim"],
        av_refinement_layers=config["model"]["av_refinement_layers"],
        freeze_wavlm=config["model"]["freeze_wavlm"],
        freeze_roberta=config["model"]["freeze_roberta"],
        num_cross_attn_heads=config["model"]["num_cross_attn_heads"],
        post_attention_layers=config["model"]["post_attention_layers"],
        av_prediction_layers=config["model"]["av_prediction_layers"],
    )


def safe_load_checkpoint(model, checkpoint_path: str, device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint

    model_state = model.state_dict()
    filtered = {}
    skipped = []
    for key, value in state_dict.items():
        if key in model_state and isinstance(value, torch.Tensor) and value.shape == model_state[key].shape:
            filtered[key] = value
        else:
            skipped.append(key)
    model.load_state_dict(filtered, strict=False)
    return checkpoint, skipped


def collect_features(model, loader, device: torch.device):
    model.eval()
    y_true = []
    fused_features = []
    final_representation = []

    with torch.no_grad():
        for batch in loader:
            audio = batch["audio"].to(device)
            text_ids = batch["text_input_ids"].to(device)
            text_mask = batch["text_attention_mask"].to(device)

            out = model(
                audio_input=audio,
                text_input=text_ids,
                text_attention_mask=text_mask,
                return_intermediate=True,
            )

            y_true.append(batch["emotion_label"].cpu().numpy())
            fused_features.append(out["fused_features"].detach().cpu().numpy())
            final_representation.append(out["final_representation"].detach().cpu().numpy())

    return {
        "y_true": np.concatenate(y_true, axis=0),
        "fused_features": np.concatenate(fused_features, axis=0),
        "final_representation": np.concatenate(final_representation, axis=0),
    }


def label_names_from_ids(label_ids: np.ndarray):
    return np.array([LABEL_NAMES[int(i)] for i in label_ids])


def preprocess_features(features: np.ndarray):
    scaler = StandardScaler()
    return scaler.fit_transform(features)


def fit_pca(features: np.ndarray, n_components: int):
    used_dim = min(n_components, features.shape[0], features.shape[1])
    pca = PCA(n_components=used_dim, random_state=42)
    z = pca.fit_transform(features)
    return z, used_dim


def project_umap(features: np.ndarray, random_state: int, n_neighbors: int, min_dist: float):
    if not HAS_UMAP:
        return None
    reducer = UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        random_state=random_state,
    )
    return reducer.fit_transform(features)


def project_tsne(features: np.ndarray, random_state: int, perplexity: float, max_iter: int):
    reducer = TSNE(
        n_components=2,
        perplexity=perplexity,
        init="pca",
        learning_rate="auto",
        max_iter=max_iter,
        random_state=random_state,
    )
    return reducer.fit_transform(features)


def plot_scatter(z, labels, out_path, title, xlabel, ylabel):
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


def plot_panel(ax, z, labels, title, xlabel, ylabel, show_legend=False):
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


def plot_sweep_grid(entries, out_path, super_title, xlabel, ylabel):
    if not entries:
        return

    n_panels = len(entries)
    ncols = 2
    nrows = int(np.ceil(n_panels / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(13, 4.8 * nrows), squeeze=False)
    flat_axes = axes.reshape(-1)

    for idx, entry in enumerate(entries):
        ax = flat_axes[idx]
        plot_panel(ax, entry["z"], entry["labels"], entry["title"], xlabel, ylabel, show_legend=idx == n_panels - 1)
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


def make_plots_for_split(split_name, features, label_names, dirs, args):
    scaled = preprocess_features(features)
    saved = []

    if HAS_UMAP:
        z_umap = project_umap(
            scaled,
            random_state=args.random_state,
            n_neighbors=args.umap_neighbors,
            min_dist=args.umap_min_dist,
        )
        np.save(dirs["tables"] / f"umap_fused_{split_name}.npy", z_umap)
        plot_scatter(
            z_umap,
            label_names,
            dirs["plots"] / f"umap_fused_{split_name}.pdf",
            f"MSP {split_name}: Full ASPIRE fused_features",
            "UMAP-1",
            "UMAP-2",
        )
        saved.extend(
            [
                dirs["tables"] / f"umap_fused_{split_name}.npy",
                dirs["plots"] / f"umap_fused_{split_name}.pdf",
            ]
        )

        if args.run_sweep:
            umap_entries = []
            for n_neighbors, min_dist in itertools.product(args.umap_neighbors_list, args.umap_min_dist_list):
                z_sweep = project_umap(
                    scaled,
                    random_state=args.random_state,
                    n_neighbors=n_neighbors,
                    min_dist=min_dist,
                )
                suffix = f"n{n_neighbors}_d{slugify_float(min_dist)}"
                np.save(dirs["tables"] / f"umap_fused_{split_name}_{suffix}.npy", z_sweep)
                plot_scatter(
                    z_sweep,
                    label_names,
                    dirs["plots"] / f"umap_fused_{split_name}_{suffix}.pdf",
                    f"MSP {split_name}: UMAP n_neighbors={n_neighbors}, min_dist={min_dist}",
                    "UMAP-1",
                    "UMAP-2",
                )
                saved.extend(
                    [
                        dirs["tables"] / f"umap_fused_{split_name}_{suffix}.npy",
                        dirs["plots"] / f"umap_fused_{split_name}_{suffix}.pdf",
                    ]
                )
                umap_entries.append(
                    {
                        "z": z_sweep,
                        "labels": label_names,
                        "title": f"n_neighbors={n_neighbors}, min_dist={min_dist}",
                    }
                )

            plot_sweep_grid(
                umap_entries,
                dirs["plots"] / f"umap_sweep_{split_name}_grid.pdf",
                f"MSP {split_name} UMAP Sweep",
                "UMAP-1",
                "UMAP-2",
            )
            saved.append(dirs["plots"] / f"umap_sweep_{split_name}_grid.pdf")

    pca_features, used_pca_dim = fit_pca(scaled, args.tsne_pca_dim)
    z_tsne = project_tsne(
        pca_features,
        random_state=args.random_state,
        perplexity=args.tsne_perplexity,
        max_iter=args.tsne_max_iter,
    )
    np.save(dirs["tables"] / f"tsne_fused_{split_name}.npy", z_tsne)
    plot_scatter(
        z_tsne,
        label_names,
        dirs["plots"] / f"tsne_fused_{split_name}.pdf",
        f"MSP {split_name}: Full ASPIRE fused_features (PCA-{used_pca_dim} -> t-SNE)",
        "t-SNE-1",
        "t-SNE-2",
    )
    saved.extend(
        [
            dirs["tables"] / f"tsne_fused_{split_name}.npy",
            dirs["plots"] / f"tsne_fused_{split_name}.pdf",
        ]
    )

    if args.run_sweep:
        tsne_entries = []
        for perplexity in args.tsne_perplexity_list:
            z_sweep = project_tsne(
                pca_features,
                random_state=args.random_state,
                perplexity=perplexity,
                max_iter=args.tsne_max_iter,
            )
            suffix = f"p{slugify_float(perplexity)}"
            np.save(dirs["tables"] / f"tsne_fused_{split_name}_{suffix}.npy", z_sweep)
            plot_scatter(
                z_sweep,
                label_names,
                dirs["plots"] / f"tsne_fused_{split_name}_{suffix}.pdf",
                f"MSP {split_name}: t-SNE perplexity={perplexity}",
                "t-SNE-1",
                "t-SNE-2",
            )
            saved.extend(
                [
                    dirs["tables"] / f"tsne_fused_{split_name}_{suffix}.npy",
                    dirs["plots"] / f"tsne_fused_{split_name}_{suffix}.pdf",
                ]
            )
            tsne_entries.append(
                {
                    "z": z_sweep,
                    "labels": label_names,
                    "title": f"perplexity={perplexity}",
                }
            )

        plot_sweep_grid(
            tsne_entries,
            dirs["plots"] / f"tsne_sweep_{split_name}_grid.pdf",
            f"MSP {split_name} t-SNE Sweep (PCA-{used_pca_dim})",
            "t-SNE-1",
            "t-SNE-2",
        )
        saved.append(dirs["plots"] / f"tsne_sweep_{split_name}_grid.pdf")

    return saved, used_pca_dim


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/msp_podcast_wpro_config.yaml")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--tsv", default=None)
    parser.add_argument("--output", default="results/msp_wpro_manifold")
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=2)
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

    config = load_config(args.config)
    checkpoint_path = resolve_checkpoint_path(args.checkpoint, config)
    tsv_path = args.tsv or config["data"]["tsv_path"]
    device_name = args.device or config.get("device", "cuda:0")
    device = torch.device(device_name if torch.cuda.is_available() else "cpu")

    dirs = ensure_output_dirs(args.output)

    batch_size = args.batch_size or config["training"]["batch_size"]
    num_workers = args.num_workers if args.num_workers is not None else config["training"].get("num_workers", 4)

    train_loader, dev_loader, test1_loader, test2_loader = get_msp_podcast_dataloaders(
        tsv_path=tsv_path,
        batch_size=batch_size,
        num_workers=num_workers,
        audio_max_length=config["data"]["max_audio_length"],
        tokenizer_name=config["model"]["roberta_model_name"],
        max_text_length=config["data"]["max_text_length"],
    )
    del train_loader, dev_loader

    model = build_model(config).to(device)
    checkpoint, skipped = safe_load_checkpoint(model, checkpoint_path, device)

    saved_paths = []
    pca_dims = {}

    for split_name, loader in [("test1", test1_loader), ("test2", test2_loader)]:
        result = collect_features(model, loader, device)
        np.savez_compressed(
            dirs["tables"] / f"features_{split_name}.npz",
            y_true=result["y_true"],
            fused_features=result["fused_features"],
            final_representation=result["final_representation"],
        )
        saved_paths.append(dirs["tables"] / f"features_{split_name}.npz")
        labels = label_names_from_ids(result["y_true"])
        split_saved, used_pca_dim = make_plots_for_split(split_name, result["fused_features"], labels, dirs, args)
        saved_paths.extend(split_saved)
        pca_dims[split_name] = used_pca_dim

    protocol_lines = [
        f"checkpoint={checkpoint_path}",
        f"checkpoint_epoch={checkpoint.get('epoch', 'N/A') if isinstance(checkpoint, dict) else 'N/A'}",
        f"device={device}",
        f"random_state={args.random_state}",
        f"umap_available={HAS_UMAP}",
        f"umap_neighbors={args.umap_neighbors}",
        f"umap_min_dist={args.umap_min_dist}",
        f"umap_neighbors_list={args.umap_neighbors_list}",
        f"umap_min_dist_list={args.umap_min_dist_list}",
        f"tsne_perplexity={args.tsne_perplexity}",
        f"tsne_perplexity_list={args.tsne_perplexity_list}",
        f"tsne_pca_dim={args.tsne_pca_dim}",
        f"tsne_max_iter={args.tsne_max_iter}",
        f"run_sweep={args.run_sweep}",
        f"skipped_checkpoint_keys={len(skipped)}",
        f"test1_pca_dim={pca_dims['test1']}",
        f"test2_pca_dim={pca_dims['test2']}",
    ]
    with open(dirs["tables"] / "manifold_protocol.txt", "w", encoding="utf-8") as fh:
        fh.write("\n".join(protocol_lines) + "\n")
    saved_paths.append(dirs["tables"] / "manifold_protocol.txt")

    print("Saved:")
    for path in saved_paths:
        print(path)


if __name__ == "__main__":
    main()
