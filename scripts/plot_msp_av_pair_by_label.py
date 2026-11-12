#!/usr/bin/env python3
"""Plot MSP Arousal/Valence embedding UMAPs colored by emotion labels."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba_cache")

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from matplotlib.lines import Line2D
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from umap import UMAP

sys.path.append(str(Path(__file__).resolve().parent.parent))

from data.msp_podcast_dataset import get_msp_podcast_dataloaders
from models.method2_crossattn_av import Method2CrossAttentionAV


LABEL_NAMES = ["Happy", "Sad", "Neutral", "Angry"]
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


def load_config(path: str):
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


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
    filtered = {
        k: v
        for k, v in state_dict.items()
        if k in model_state and isinstance(v, torch.Tensor) and v.shape == model_state[k].shape
    }
    model.load_state_dict(filtered, strict=False)
    return checkpoint


def cache_path(cache_dir: Path, split_name: str) -> Path:
    return cache_dir / f"msp_av_umap_{split_name}.npz"


def partial_cache_path(cache_dir: Path, split_name: str) -> Path:
    return cache_dir / f"msp_av_umap_{split_name}.partial.npz"


def save_cache(path: Path, payload: dict, num_done_samples: int | None = None):
    data = dict(payload)
    if num_done_samples is not None:
        data["num_done_samples"] = np.array([num_done_samples], dtype=np.int64)
    np.savez_compressed(path, **data)


def load_cache(path: Path) -> dict:
    d = np.load(path)
    return {k: d[k] for k in d.files}


def collect_embeddings(
    model,
    loader,
    device: torch.device,
    split_name: str,
    partial_path: Path | None = None,
    existing: dict | None = None,
    save_every_batches: int = 25,
):
    model.eval()
    labels = [] if existing is None else [existing["labels"]]
    arousal = [] if existing is None else [existing["arousal_embedding"]]
    valence = [] if existing is None else [existing["valence_embedding"]]
    done_samples = 0 if existing is None else int(existing["num_done_samples"][0])
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(loader, desc=f"MSP {split_name}", leave=True), start=1):
            audio = batch["audio"].to(device)
            text_ids = batch["text_input_ids"].to(device)
            text_mask = batch["text_attention_mask"].to(device)
            out = model(audio_input=audio, text_input=text_ids, text_attention_mask=text_mask)
            labels.append(batch["emotion_label"].cpu().numpy())
            arousal.append(out["arousal_embedding"].detach().cpu().numpy())
            valence.append(out["valence_embedding"].detach().cpu().numpy())
            done_samples += int(batch["emotion_label"].shape[0])
            if partial_path is not None and batch_idx % save_every_batches == 0:
                payload = {
                    "labels": np.concatenate(labels, axis=0),
                    "arousal_embedding": np.concatenate(arousal, axis=0),
                    "valence_embedding": np.concatenate(valence, axis=0),
                }
                save_cache(partial_path, payload, num_done_samples=done_samples)
                print(f"Saved partial MSP UMAP cache ({split_name}): {partial_path} [{done_samples} samples]")
    payload = {
        "labels": np.concatenate(labels, axis=0),
        "arousal_embedding": np.concatenate(arousal, axis=0),
        "valence_embedding": np.concatenate(valence, axis=0),
    }
    if partial_path is not None and partial_path.exists():
        partial_path.unlink()
    return payload


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
):
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
        out[:, dim] = scale * (out[:, dim] - center) / max(hi - lo, 1e-8)
    return out


def set_tight_bounds(ax, z: np.ndarray, pad_ratio: float = 0.08):
    x_min, x_max = float(np.min(z[:, 0])), float(np.max(z[:, 0]))
    y_min, y_max = float(np.min(z[:, 1])), float(np.max(z[:, 1]))
    x_pad = max((x_max - x_min) * pad_ratio, 1e-3)
    y_pad = max((y_max - y_min) * pad_ratio, 1e-3)
    ax.set_xlim(x_min - x_pad, x_max + x_pad)
    ax.set_ylim(y_min - y_pad, y_max + y_pad)


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
    return [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=color_by_label[label_id],
            markersize=8,
            label=LABEL_ID_TO_NAME[label_id],
        )
        for label_id in order
    ]


def plot_pair_single(arousal_z, valence_z, labels, out_path: Path, title: str):
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 6.1))

    scatter_by_label(axes[0], arousal_z, labels, AROUSAL_COLOR_BY_LABEL)
    set_tight_bounds(axes[0], arousal_z)
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

    scatter_by_label(axes[1], valence_z, labels, VALENCE_COLOR_BY_LABEL)
    set_tight_bounds(axes[1], valence_z)
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


def build_resume_loader(loader, start_idx: int):
    if start_idx <= 0:
        return loader
    subset = torch.utils.data.Subset(loader.dataset, list(range(start_idx, len(loader.dataset))))
    return torch.utils.data.DataLoader(
        subset,
        batch_size=loader.batch_size,
        shuffle=False,
        num_workers=loader.num_workers,
        pin_memory=getattr(loader, "pin_memory", False),
        drop_last=False,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/msp_podcast_wpro_config.yaml")
    parser.add_argument("--checkpoint", default="checkpoints/method2_msp_wpro_best_bak.pt")
    parser.add_argument("--figures_dir", default="figures")
    parser.add_argument("--cache_dir", default="results/full_vs_noav_analysis/tables")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--umap_neighbors", type=int, default=15)
    parser.add_argument("--umap_min_dist", type=float, default=0.3)
    parser.add_argument("--split", choices=["test1", "test2", "all"], default="all")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    figures_dir = Path(args.figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    batch_size = args.batch_size or config["training"]["batch_size"]

    _, _, test1_loader, test2_loader = get_msp_podcast_dataloaders(
        tsv_path=config["data"]["tsv_path"],
        batch_size=batch_size,
        num_workers=args.num_workers,
        audio_max_length=config["data"]["max_audio_length"],
        tokenizer_name=config["model"]["roberta_model_name"],
        max_text_length=config["data"]["max_text_length"],
        pin_memory=False,
    )

    need_test1 = args.split in {"test1", "all"}
    need_test2 = args.split in {"test2", "all"}
    test1 = test2 = None

    cache1 = cache_path(cache_dir, "test1")
    cache2 = cache_path(cache_dir, "test2")
    partial1 = partial_cache_path(cache_dir, "test1")
    partial2 = partial_cache_path(cache_dir, "test2")

    if need_test1 and cache1.exists() and not args.force:
        print(f"Loading MSP UMAP cache: {cache1}")
        test1 = load_cache(cache1)
    if need_test2 and cache2.exists() and not args.force:
        print(f"Loading MSP UMAP cache: {cache2}")
        test2 = load_cache(cache2)

    if (need_test1 and test1 is None) or (need_test2 and test2 is None):
        print(f"MSP device: {device}")
        print(f"MSP checkpoint: {args.checkpoint}")
        model = build_model(config).to(device)
        safe_load_checkpoint(model, args.checkpoint, device)

        if need_test1 and test1 is None:
            existing1 = load_cache(partial1) if partial1.exists() and not args.force else None
            if existing1 is not None:
                print(f"Resuming MSP UMAP test1 from sample {int(existing1['num_done_samples'][0])}")
                loader1 = build_resume_loader(test1_loader, int(existing1["num_done_samples"][0]))
            else:
                loader1 = test1_loader
            test1 = collect_embeddings(model, loader1, device, "test1", partial_path=partial1, existing=existing1)
            save_cache(cache1, test1)
            print(f"Saved MSP UMAP cache: {cache1}")

        if need_test2 and test2 is None:
            existing2 = load_cache(partial2) if partial2.exists() and not args.force else None
            if existing2 is not None:
                print(f"Resuming MSP UMAP test2 from sample {int(existing2['num_done_samples'][0])}")
                loader2 = build_resume_loader(test2_loader, int(existing2["num_done_samples"][0]))
            else:
                loader2 = test2_loader
            test2 = collect_embeddings(model, loader2, device, "test2", partial_path=partial2, existing=existing2)
            save_cache(cache2, test2)
            print(f"Saved MSP UMAP cache: {cache2}")

    if need_test1:
        arousal_test1 = stretch_embedding(project_umap(preprocess_features(test1["arousal_embedding"]), args.random_state, args.umap_neighbors, args.umap_min_dist))
        valence_test1 = stretch_embedding(project_umap(preprocess_features(test1["valence_embedding"]), args.random_state, args.umap_neighbors, args.umap_min_dist))
        plot_pair_single(arousal_test1, valence_test1, test1["labels"], figures_dir / "msp_full_av_pair_umap_by_label_side_by_side_test1.pdf", "MSP Test1")

    if need_test2:
        arousal_test2 = stretch_embedding(project_umap(preprocess_features(test2["arousal_embedding"]), args.random_state, args.umap_neighbors, args.umap_min_dist))
        valence_test2 = stretch_embedding(project_umap(preprocess_features(test2["valence_embedding"]), args.random_state, args.umap_neighbors, args.umap_min_dist))
        plot_pair_single(arousal_test2, valence_test2, test2["labels"], figures_dir / "msp_full_av_pair_umap_by_label_side_by_side_test2.pdf", "MSP Test2")

    if args.split == "all":
        labels_all = np.concatenate([test1["labels"], test2["labels"]], axis=0)
        arousal_all_features = np.concatenate([test1["arousal_embedding"], test2["arousal_embedding"]], axis=0)
        valence_all_features = np.concatenate([test1["valence_embedding"], test2["valence_embedding"]], axis=0)
        arousal_all = stretch_embedding(project_umap_fast(preprocess_features(arousal_all_features), args.random_state, args.umap_neighbors, args.umap_min_dist))
        valence_all = stretch_embedding(project_umap_fast(preprocess_features(valence_all_features), args.random_state, args.umap_neighbors, args.umap_min_dist))
        plot_pair_single(arousal_all, valence_all, labels_all, figures_dir / "msp_full_av_pair_umap_by_label_side_by_side_all.pdf", "MSP Test1 + Test2")

    print("Saved:")
    if need_test1:
        print(figures_dir / "msp_full_av_pair_umap_by_label_side_by_side_test1.pdf")
    if need_test2:
        print(figures_dir / "msp_full_av_pair_umap_by_label_side_by_side_test2.pdf")
    if args.split == "all":
        print(figures_dir / "msp_full_av_pair_umap_by_label_side_by_side_all.pdf")


if __name__ == "__main__":
    main()
