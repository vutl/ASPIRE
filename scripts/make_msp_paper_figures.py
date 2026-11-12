#!/usr/bin/env python3
"""Make paper-ready MSP-Podcast UMAP and AV-scatter figures."""
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
import torch
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from umap import UMAP

sys.path.append(str(Path(__file__).resolve().parent.parent))


LABEL_NAMES = ["Happy", "Sad", "Neutral", "Angry"]


def label_names_from_ids(label_ids: np.ndarray):
    return np.array([LABEL_NAMES[int(i)] for i in label_ids])


def preprocess_features(features: np.ndarray):
    return StandardScaler().fit_transform(features)


def project_umap(features: np.ndarray, random_state: int = 42, n_neighbors: int = 15, min_dist: float = 0.3):
    reducer = UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        random_state=random_state,
    )
    return reducer.fit_transform(features)


def project_joint_umap(
    features1: np.ndarray,
    features2: np.ndarray,
    random_state: int = 42,
    n_neighbors: int = 15,
    min_dist: float = 0.3,
    pca_dim: int = 50,
    sample_size: int = 12000,
):
    all_features = np.concatenate([features1, features2], axis=0)
    scaler = StandardScaler()
    all_scaled = scaler.fit_transform(all_features)

    if all_scaled.shape[1] > pca_dim:
        pca = PCA(n_components=pca_dim, random_state=random_state)
        all_scaled = pca.fit_transform(all_scaled)

    n_total = all_scaled.shape[0]
    rng = np.random.default_rng(random_state)
    if n_total > sample_size:
        fit_idx = np.sort(rng.choice(n_total, size=sample_size, replace=False))
        fit_data = all_scaled[fit_idx]
    else:
        fit_data = all_scaled

    reducer = UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        random_state=random_state,
    )
    reducer.fit(fit_data)
    z_all = reducer.transform(all_scaled)
    n1 = features1.shape[0]
    return z_all[:n1], z_all[n1:], z_all


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


def set_tight_bounds(ax, z: np.ndarray, pad_ratio: float = 0.06):
    x_min, x_max = float(np.min(z[:, 0])), float(np.max(z[:, 0]))
    y_min, y_max = float(np.min(z[:, 1])), float(np.max(z[:, 1]))
    x_pad = max((x_max - x_min) * pad_ratio, 1e-3)
    y_pad = max((y_max - y_min) * pad_ratio, 1e-3)
    ax.set_xlim(x_min - x_pad, x_max + x_pad)
    ax.set_ylim(y_min - y_pad, y_max + y_pad)


def plot_umap_single(z, labels, out_path: Path, title: str):
    fig, ax = plt.subplots(figsize=(5.4, 5.1))
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
    )
    set_tight_bounds(ax, z)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title(title, fontsize=13)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=True, fancybox=True)
    plt.tight_layout(pad=0.15)
    plt.savefig(out_path, bbox_inches="tight", pad_inches=0.02)
    plt.close()


def plot_umap_combined(z1, labels1, z2, labels2, out_path: Path):
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 5.0))
    z_all = np.concatenate([z1, z2], axis=0)
    for ax, z, labels, title, show_legend in [
        (axes[0], z1, labels1, "MSP Test1", False),
        (axes[1], z2, labels2, "MSP Test2", True),
    ]:
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
        set_tight_bounds(ax, z_all)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_title(title, fontsize=13)
    if axes[1].legend_ is not None:
        handles, labels = axes[1].get_legend_handles_labels()
        axes[1].legend(handles=handles, labels=labels, loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=True, fancybox=True)
    plt.tight_layout(pad=0.15)
    plt.savefig(out_path, bbox_inches="tight", pad_inches=0.02)
    plt.close()


def plot_umap_all(z1, labels1, z2, labels2, out_path: Path):
    fig, ax = plt.subplots(figsize=(5.6, 5.2))
    z = np.concatenate([z1, z2], axis=0)
    labels = np.concatenate([labels1, labels2], axis=0)
    sns.scatterplot(
        x=z[:, 0],
        y=z[:, 1],
        hue=labels,
        hue_order=LABEL_NAMES,
        palette="tab10",
        s=11,
        linewidth=0,
        alpha=0.78,
        ax=ax,
    )
    set_tight_bounds(ax, z)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title("MSP Test1 + Test2", fontsize=13)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=True, fancybox=True)
    plt.tight_layout(pad=0.15)
    plt.savefig(out_path, bbox_inches="tight", pad_inches=0.02)
    plt.close()


def build_model(config):
    from models.method2_crossattn_av import Method2CrossAttentionAV

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


def safe_load_checkpoint(model, checkpoint_path: str):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
    model_state = model.state_dict()
    filtered = {
        k: v
        for k, v in state_dict.items()
        if k in model_state and isinstance(v, torch.Tensor) and v.shape == model_state[k].shape
    }
    model.load_state_dict(filtered, strict=False)
    return checkpoint, len(filtered), len(state_dict) - len(filtered)


def collect_av_predictions(model, loader, device: torch.device):
    model = model.to(device)
    model.eval()
    y_pred = []
    arousal_pred = []
    valence_pred = []
    with torch.no_grad():
        for batch in loader:
            audio = batch["audio"].to(device)
            text_ids = batch["text_input_ids"].to(device)
            text_mask = batch["text_attention_mask"].to(device)
            out = model(audio_input=audio, text_input=text_ids, text_attention_mask=text_mask)
            y_pred.append(torch.argmax(out["emotion_logits"], dim=1).cpu().numpy())
            arousal_pred.append(out["arousal_value"].detach().cpu().numpy().reshape(-1))
            valence_pred.append(out["valence_value"].detach().cpu().numpy().reshape(-1))
    return {
        "y_pred": np.concatenate(y_pred, axis=0),
        "arousal_pred": np.concatenate(arousal_pred, axis=0),
        "valence_pred": np.concatenate(valence_pred, axis=0),
    }


def plot_av_scatter_single(arousal, valence, labels, out_path: Path, title: str):
    fig, ax = plt.subplots(figsize=(5.4, 5.1))
    sns.scatterplot(
        x=arousal,
        y=valence,
        hue=labels,
        hue_order=LABEL_NAMES,
        palette="tab10",
        s=14,
        linewidth=0,
        alpha=0.75,
        ax=ax,
    )
    ax.set_xlabel("Predicted Arousal")
    ax.set_ylabel("Predicted Valence")
    ax.set_title(title, fontsize=13)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=True, fancybox=True)
    plt.tight_layout(pad=0.15)
    plt.savefig(out_path, bbox_inches="tight", pad_inches=0.02)
    plt.close()


def plot_av_scatter_combined(a1, v1, l1, a2, v2, l2, out_path: Path):
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 5.0))
    for ax, arousal, valence, labels, title, show_legend in [
        (axes[0], a1, v1, l1, "MSP Test1", False),
        (axes[1], a2, v2, l2, "MSP Test2", True),
    ]:
        sns.scatterplot(
            x=arousal,
            y=valence,
            hue=labels,
            hue_order=LABEL_NAMES,
            palette="tab10",
            s=14,
            linewidth=0,
            alpha=0.75,
            ax=ax,
            legend=show_legend,
        )
        ax.set_xlabel("Predicted Arousal")
        ax.set_ylabel("Predicted Valence")
        ax.set_title(title, fontsize=13)
    if axes[1].legend_ is not None:
        handles, labels = axes[1].get_legend_handles_labels()
        axes[1].legend(handles=handles, labels=labels, loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=True, fancybox=True)
    plt.tight_layout(pad=0.15)
    plt.savefig(out_path, bbox_inches="tight", pad_inches=0.02)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifold_root", default="results/msp_wpro_manifold")
    parser.add_argument("--figures_dir", default="figures")
    parser.add_argument("--make_av", action="store_true", help="Also infer AV predictions and draw AV scatter figures")
    parser.add_argument("--config", default="configs/msp_podcast_wpro_config.yaml")
    parser.add_argument("--checkpoint", default="checkpoints/method2_msp_wpro_best_bak.pt")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch_size", type=int, default=None)
    args = parser.parse_args()

    manifold_root = Path(args.manifold_root)
    figures_dir = Path(args.figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)

    d1 = np.load(manifold_root / "tables" / "features_test1.npz")
    d2 = np.load(manifold_root / "tables" / "features_test2.npz")
    labels1 = label_names_from_ids(d1["y_true"])
    labels2 = label_names_from_ids(d2["y_true"])
    print("Computing joint UMAP on MSP test1+test2...")
    z1, z2, z_all = project_joint_umap(d1["fused_features"], d2["fused_features"])
    z_all = stretch_embedding(z_all)
    z1 = z_all[: len(labels1)]
    z2 = z_all[len(labels1) :]

    plot_umap_single(z1, labels1, figures_dir / "msp_umap_test1.pdf", "MSP Test1")
    plot_umap_single(z2, labels2, figures_dir / "msp_umap_test2.pdf", "MSP Test2")
    plot_umap_all(z1, labels1, z2, labels2, figures_dir / "msp_umap_all.pdf")
    plot_umap_combined(z1, labels1, z2, labels2, figures_dir / "msp_umap_test1_test2_side_by_side.pdf")

    if not args.make_av:
        print("Saved:")
        print(figures_dir / "msp_umap_test1.pdf")
        print(figures_dir / "msp_umap_test2.pdf")
        print(figures_dir / "msp_umap_all.pdf")
        print(figures_dir / "msp_umap_test1_test2_side_by_side.pdf")
        return

    from data.msp_podcast_dataset import get_msp_podcast_dataloaders
    from train.train_method2_msp_wpro import load_config

    config = load_config(args.config)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    batch_size = args.batch_size or config["training"]["batch_size"]
    _, _, test1_loader, test2_loader = get_msp_podcast_dataloaders(
        tsv_path=config["data"]["tsv_path"],
        batch_size=batch_size,
        num_workers=0,
        audio_max_length=config["data"]["max_audio_length"],
        tokenizer_name=config["model"]["roberta_model_name"],
        max_text_length=config["data"]["max_text_length"],
        pin_memory=False,
    )
    model = build_model(config)
    safe_load_checkpoint(model, args.checkpoint)

    pred1 = collect_av_predictions(model, test1_loader, device)
    pred2 = collect_av_predictions(model, test2_loader, device)
    pred_all = {
        "y_pred": np.concatenate([pred1["y_pred"], pred2["y_pred"]], axis=0),
        "arousal_pred": np.concatenate([pred1["arousal_pred"], pred2["arousal_pred"]], axis=0),
        "valence_pred": np.concatenate([pred1["valence_pred"], pred2["valence_pred"]], axis=0),
    }

    plot_av_scatter_single(
        pred1["arousal_pred"],
        pred1["valence_pred"],
        label_names_from_ids(pred1["y_pred"]),
        figures_dir / "msp_av_pred_by_pred_emotion_test1.pdf",
        "MSP Test1",
    )
    plot_av_scatter_single(
        pred2["arousal_pred"],
        pred2["valence_pred"],
        label_names_from_ids(pred2["y_pred"]),
        figures_dir / "msp_av_pred_by_pred_emotion_test2.pdf",
        "MSP Test2",
    )
    plot_av_scatter_single(
        pred_all["arousal_pred"],
        pred_all["valence_pred"],
        label_names_from_ids(pred_all["y_pred"]),
        figures_dir / "msp_av_pred_by_pred_emotion_all.pdf",
        "MSP Test1 + Test2",
    )
    plot_av_scatter_combined(
        pred1["arousal_pred"],
        pred1["valence_pred"],
        label_names_from_ids(pred1["y_pred"]),
        pred2["arousal_pred"],
        pred2["valence_pred"],
        label_names_from_ids(pred2["y_pred"]),
        figures_dir / "msp_av_pred_by_pred_emotion_test1_test2_side_by_side.pdf",
    )

    print("Saved:")
    print(figures_dir / "msp_umap_test1.pdf")
    print(figures_dir / "msp_umap_test2.pdf")
    print(figures_dir / "msp_umap_all.pdf")
    print(figures_dir / "msp_umap_test1_test2_side_by_side.pdf")
    print(figures_dir / "msp_av_pred_by_pred_emotion_test1.pdf")
    print(figures_dir / "msp_av_pred_by_pred_emotion_test2.pdf")
    print(figures_dir / "msp_av_pred_by_pred_emotion_all.pdf")
    print(figures_dir / "msp_av_pred_by_pred_emotion_test1_test2_side_by_side.pdf")


if __name__ == "__main__":
    main()
