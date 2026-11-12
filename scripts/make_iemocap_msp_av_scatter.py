#!/usr/bin/env python3
"""Create side-by-side AV scatter: IEMOCAP aggregate vs MSP aggregate."""
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
import yaml
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parent.parent))

from data.msp_podcast_dataset import get_msp_podcast_dataloaders
from models.method2_crossattn_av import Method2CrossAttentionAV
from scripts.full_vs_noav_analysis_utils import LABEL_NAMES, aggregate_cached, ensure_cache_for_models, ensure_output_dirs

LABEL_REMAP_OLD_TO_NEW = {
    0: 3,
    1: 0,
    2: 2,
    3: 1,
}


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


def collect_msp_predictions(
    model,
    loader,
    device: torch.device,
    split_name: str,
    partial_cache_path: Path | None = None,
    existing_pred: dict | None = None,
    save_every_batches: int = 25,
):
    model.eval()
    y_pred = [] if existing_pred is None else [existing_pred["y_pred"]]
    arousal_pred = [] if existing_pred is None else [existing_pred["arousal_pred"]]
    valence_pred = [] if existing_pred is None else [existing_pred["valence_pred"]]
    done_samples = 0 if existing_pred is None else int(existing_pred["num_done_samples"][0])
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(loader, desc=f"MSP {split_name}", leave=True), start=1):
            audio = batch["audio"].to(device)
            text_ids = batch["text_input_ids"].to(device)
            text_mask = batch["text_attention_mask"].to(device)
            out = model(audio_input=audio, text_input=text_ids, text_attention_mask=text_mask)
            y_pred.append(torch.argmax(out["emotion_logits"], dim=1).cpu().numpy())
            arousal_pred.append(out["arousal_value"].detach().cpu().numpy().reshape(-1))
            valence_pred.append(out["valence_value"].detach().cpu().numpy().reshape(-1))
            done_samples += int(batch["emotion_label"].shape[0])
            if partial_cache_path is not None and batch_idx % save_every_batches == 0:
                pred = {
                    "y_pred": np.concatenate(y_pred, axis=0),
                    "arousal_pred": np.concatenate(arousal_pred, axis=0),
                    "valence_pred": np.concatenate(valence_pred, axis=0),
                }
                save_prediction_cache(partial_cache_path, pred, num_done_samples=done_samples)
                print(f"Saved partial MSP cache ({split_name}): {partial_cache_path} [{done_samples} samples]")
    pred = {
        "y_pred": np.concatenate(y_pred, axis=0),
        "arousal_pred": np.concatenate(arousal_pred, axis=0),
        "valence_pred": np.concatenate(valence_pred, axis=0),
    }
    if partial_cache_path is not None and partial_cache_path.exists():
        partial_cache_path.unlink()
    return pred


def label_names_from_ids(label_ids):
    return [LABEL_NAMES[int(i)] for i in label_ids]


def remap_old_msp_labels(y_pred: np.ndarray) -> np.ndarray:
    y_pred = np.asarray(y_pred).reshape(-1)
    return np.asarray([LABEL_REMAP_OLD_TO_NEW[int(x)] for x in y_pred], dtype=np.int64)


def msp_cache_path(output_dir: Path, split_name: str) -> Path:
    return output_dir / "tables" / f"msp_av_pred_{split_name}.npz"


def msp_partial_cache_path(output_dir: Path, split_name: str) -> Path:
    return output_dir / "tables" / f"msp_av_pred_{split_name}.partial.npz"


def save_prediction_cache(path: Path, pred: dict, num_done_samples: int | None = None):
    payload = dict(pred)
    if num_done_samples is not None:
        payload["num_done_samples"] = np.array([num_done_samples], dtype=np.int64)
    np.savez_compressed(path, **payload)


def load_prediction_cache(path: Path) -> dict:
    d = np.load(path)
    return {k: d[k] for k in d.files}


def build_resume_loader(loader: DataLoader, start_idx: int) -> DataLoader:
    if start_idx <= 0:
        return loader
    subset = Subset(loader.dataset, list(range(start_idx, len(loader.dataset))))
    return DataLoader(
        subset,
        batch_size=loader.batch_size,
        shuffle=False,
        num_workers=loader.num_workers,
        pin_memory=getattr(loader, "pin_memory", False),
        drop_last=False,
    )


def get_split_ground_truth_av(loader: DataLoader) -> tuple[np.ndarray, np.ndarray]:
    dataset = loader.dataset
    audio_paths = dataset.df["path"].tolist()
    arousal = []
    valence = []
    for audio_path in audio_paths:
        audio_filename = os.path.basename(audio_path)
        if audio_filename in dataset.av_dict:
            arousal.append(dataset.av_dict[audio_filename]["arousal"])
            valence.append(dataset.av_dict[audio_filename]["valence"])
        else:
            arousal.append(dataset.arousal_mean)
            valence.append(dataset.valence_mean)

    arousal = np.asarray(arousal, dtype=np.float64)
    valence = np.asarray(valence, dtype=np.float64)
    if dataset.normalize_av:
        arousal = (arousal - dataset.arousal_mean) / (dataset.arousal_std + 1e-8)
        valence = (valence - dataset.valence_mean) / (dataset.valence_std + 1e-8)
    return arousal, valence


def plot_single(arousal, valence, labels, title, out_path: Path):
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
    ax.set_xlim(-1.0, 1.0)
    ax.set_ylim(-1.0, 1.0)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Predicted Arousal")
    ax.set_ylabel("Predicted Valence")
    ax.set_title(title, fontsize=13)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=True, fancybox=True)
    plt.tight_layout(pad=0.15)
    plt.savefig(out_path, bbox_inches="tight", pad_inches=0.02)
    plt.close()


def compute_symmetric_limit(arousal, valence, min_lim: float = 1.0, pad: float = 0.05) -> float:
    lim = float(max(np.max(np.abs(arousal)), np.max(np.abs(valence))))
    return max(min_lim, round(lim + pad, 2))


def map_to_minus1_1(x, source_min, source_max, eps=1e-6):
    x = np.asarray(x, dtype=np.float64)
    if source_min is None or source_max is None or np.isclose(source_min, source_max):
        mx = np.max(np.abs(x)) if x.size else 1.0
        if mx < 1e-12:
            return np.zeros_like(x)
        return np.clip(x / mx, -1.0 + eps, 1.0 - eps)
    mapped = -1.0 + 2.0 * (x - source_min) / (source_max - source_min + 1e-12)
    return np.clip(mapped, -1.0 + eps, 1.0 - eps)


def remap_pred_with_gt_if_available(item: dict):
    out = dict(item)
    a_true = item.get("arousal_true")
    v_true = item.get("valence_true")
    if a_true is not None and np.asarray(a_true).size:
        a_min, a_max = np.min(a_true), np.max(a_true)
    else:
        a_min, a_max = np.min(item["arousal_pred"]), np.max(item["arousal_pred"])
    if v_true is not None and np.asarray(v_true).size:
        v_min, v_max = np.min(v_true), np.max(v_true)
    else:
        v_min, v_max = np.min(item["valence_pred"]), np.max(item["valence_pred"])
    out["arousal_pred"] = map_to_minus1_1(item["arousal_pred"], a_min, a_max)
    out["valence_pred"] = map_to_minus1_1(item["valence_pred"], v_min, v_max)
    return out


def plot_pair(left, right, left_title: str, right_title: str, out_path: Path):
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 6.2))
    legend_handles = legend_labels = None
    for ax, item, title, show_legend in [
        (axes[0], left, left_title, False),
        (axes[1], right, right_title, True),
    ]:
        sns.scatterplot(
            x=item["arousal_pred"],
            y=item["valence_pred"],
            hue=label_names_from_ids(item["y_pred"]),
            hue_order=LABEL_NAMES,
            palette="tab10",
            s=14,
            linewidth=0,
            alpha=0.75,
            ax=ax,
            legend=show_legend,
        )
        lim = compute_symmetric_limit(item["arousal_pred"], item["valence_pred"])
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("Predicted Arousal")
        ax.set_ylabel("Predicted Valence")
        ax.set_title(title, fontsize=13)
        if ax.legend_ is not None:
            legend_handles, legend_labels = ax.get_legend_handles_labels()
            ax.legend_.remove()
    if legend_handles is not None:
        fig.legend(
            legend_handles,
            legend_labels,
            loc="center left",
            bbox_to_anchor=(0.905, 0.5),
            frameon=True,
            fancybox=True,
        )
    fig.subplots_adjust(left=0.07, right=0.86, top=0.93, bottom=0.11, wspace=0.28)
    plt.savefig(out_path, bbox_inches="tight", pad_inches=0.02)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--full_dir", default="checkpoints/method2_kfold_wpro")
    parser.add_argument("--noav_dir", default="checkpoints/method2_kfold_wpro_ablation_no_avsup")
    parser.add_argument("--csv", default="datasets/IEMOCAP_KFOLD/tenfolds/all.csv")
    parser.add_argument("--output", default="results/full_vs_noav_analysis")
    parser.add_argument("--folds", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--max_audio_length", type=int, default=160000)
    parser.add_argument("--max_text_length", type=int, default=128)
    parser.add_argument("--roberta_model_name", default="roberta-base")
    parser.add_argument("--wavlm_model_name", default="microsoft/wavlm-base-plus")
    parser.add_argument("--wavlm_prosody_model_name", default="Orange/Speaker-wavLM-pro")
    parser.add_argument("--num_emotions", type=int, default=4)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--num_cross_attn_heads", type=int, default=8)
    parser.add_argument("--post_attention_layers", type=int, default=2)
    parser.add_argument("--av_prediction_layers", type=int, default=3)
    parser.add_argument("--av_refinement_layers", type=int, default=2)
    parser.add_argument("--freeze_wavlm", action="store_true")
    parser.add_argument("--freeze_roberta", action="store_true")
    parser.add_argument("--which", choices=["best", "last"], default="best")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--figures_dir", default="figures")
    parser.add_argument("--msp_config", default="configs/msp_podcast_wpro_config.yaml")
    parser.add_argument("--msp_checkpoint", default="checkpoints/method2_msp_wpro_best_bak.pt")
    parser.add_argument("--msp_device", default="cuda:0")
    parser.add_argument("--msp_batch_size", type=int, default=16)
    parser.add_argument("--msp_num_workers", type=int, default=2)
    parser.add_argument("--msp_split", choices=["test1", "test2", "all"], default="all")
    parser.add_argument("--remap_old_msp_labels", action="store_true")
    args = parser.parse_args()

    dirs = ensure_output_dirs(args.output)
    figures_dir = Path(args.figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)

    ensure_cache_for_models(args)
    full = aggregate_cached(dirs["cache"], "full", args.folds)
    if full["arousal_pred"].size == 0 or full["valence_pred"].size == 0:
        raise RuntimeError("Full model cache is missing arousal/valence predictions.")

    msp_config = load_config(args.msp_config)
    msp_device = torch.device(args.msp_device if torch.cuda.is_available() else "cpu")
    msp_batch_size = args.msp_batch_size or msp_config["training"]["batch_size"]

    _, _, test1_loader, test2_loader = get_msp_podcast_dataloaders(
        tsv_path=msp_config["data"]["tsv_path"],
        batch_size=msp_batch_size,
        num_workers=args.msp_num_workers,
        audio_max_length=msp_config["data"]["max_audio_length"],
        tokenizer_name=msp_config["model"]["roberta_model_name"],
        max_text_length=msp_config["data"]["max_text_length"],
        pin_memory=False,
    )

    need_test1 = args.msp_split in {"test1", "all"}
    need_test2 = args.msp_split in {"test2", "all"}

    pred1 = pred2 = None
    cache1 = msp_cache_path(dirs["root"], "test1")
    cache2 = msp_cache_path(dirs["root"], "test2")
    partial1 = msp_partial_cache_path(dirs["root"], "test1")
    partial2 = msp_partial_cache_path(dirs["root"], "test2")

    if need_test1 and cache1.exists() and not args.force:
        print(f"Loading MSP cache: {cache1}")
        pred1 = load_prediction_cache(cache1)
    if need_test2 and cache2.exists() and not args.force:
        print(f"Loading MSP cache: {cache2}")
        pred2 = load_prediction_cache(cache2)

    if (need_test1 and pred1 is None) or (need_test2 and pred2 is None):
        print(f"MSP device: {msp_device}")
        print(f"MSP checkpoint: {args.msp_checkpoint}")
        model = build_model(msp_config).to(msp_device)
        safe_load_checkpoint(model, args.msp_checkpoint, msp_device)

        if need_test1 and pred1 is None:
            existing1 = None
            resume_loader1 = test1_loader
            if partial1.exists() and not args.force:
                existing1 = load_prediction_cache(partial1)
                start_idx = int(existing1["num_done_samples"][0])
                print(f"Resuming MSP test1 from sample {start_idx}")
                resume_loader1 = build_resume_loader(test1_loader, start_idx)
            pred1 = collect_msp_predictions(
                model,
                resume_loader1,
                msp_device,
                "test1",
                partial_cache_path=partial1,
                existing_pred=existing1,
            )
            pred1["arousal_true"], pred1["valence_true"] = get_split_ground_truth_av(test1_loader)
            save_prediction_cache(cache1, pred1)
            print(f"Saved MSP cache: {cache1}")
        if need_test2 and pred2 is None:
            existing2 = None
            resume_loader2 = test2_loader
            if partial2.exists() and not args.force:
                existing2 = load_prediction_cache(partial2)
                start_idx = int(existing2["num_done_samples"][0])
                print(f"Resuming MSP test2 from sample {start_idx}")
                resume_loader2 = build_resume_loader(test2_loader, start_idx)
            pred2 = collect_msp_predictions(
                model,
                resume_loader2,
                msp_device,
                "test2",
                partial_cache_path=partial2,
                existing_pred=existing2,
            )
            pred2["arousal_true"], pred2["valence_true"] = get_split_ground_truth_av(test2_loader)
            save_prediction_cache(cache2, pred2)
            print(f"Saved MSP cache: {cache2}")

    msp = None
    if pred1 is not None and pred2 is not None:
        msp = {
            "y_pred": np.concatenate([pred1["y_pred"], pred2["y_pred"]], axis=0),
            "arousal_pred": np.concatenate([pred1["arousal_pred"], pred2["arousal_pred"]], axis=0),
            "valence_pred": np.concatenate([pred1["valence_pred"], pred2["valence_pred"]], axis=0),
            "arousal_true": np.concatenate([pred1.get("arousal_true", np.array([])), pred2.get("arousal_true", np.array([]))], axis=0),
            "valence_true": np.concatenate([pred1.get("valence_true", np.array([])), pred2.get("valence_true", np.array([]))], axis=0),
        }
    if pred1 is not None:
        pred1_plot = remap_pred_with_gt_if_available(pred1)
        if args.remap_old_msp_labels:
            pred1_plot["y_pred"] = remap_old_msp_labels(pred1_plot["y_pred"])
        plot_single(
            pred1_plot["arousal_pred"],
            pred1_plot["valence_pred"],
            label_names_from_ids(pred1_plot["y_pred"]),
            "MSP Test1",
            figures_dir / "msp_av_pred_by_pred_emotion_test1.pdf",
        )
        plot_pair(
            full,
            pred1_plot,
            "IEMOCAP",
            "MSP Test1",
            figures_dir / "av_pred_by_pred_emotion_iemocap_msp_test1_side_by_side.pdf",
        )
        print(figures_dir / "msp_av_pred_by_pred_emotion_test1.pdf")
        print(figures_dir / "av_pred_by_pred_emotion_iemocap_msp_test1_side_by_side.pdf")

    if pred2 is not None:
        pred2_plot = remap_pred_with_gt_if_available(pred2)
        if args.remap_old_msp_labels:
            pred2_plot["y_pred"] = remap_old_msp_labels(pred2_plot["y_pred"])
        plot_single(
            pred2_plot["arousal_pred"],
            pred2_plot["valence_pred"],
            label_names_from_ids(pred2_plot["y_pred"]),
            "MSP Test2",
            figures_dir / "msp_av_pred_by_pred_emotion_test2.pdf",
        )
        print(figures_dir / "msp_av_pred_by_pred_emotion_test2.pdf")

    if msp is not None:
        msp_plot = remap_pred_with_gt_if_available(msp)
        if args.remap_old_msp_labels:
            msp_plot["y_pred"] = remap_old_msp_labels(msp_plot["y_pred"])
        plot_single(
            msp_plot["arousal_pred"],
            msp_plot["valence_pred"],
            label_names_from_ids(msp_plot["y_pred"]),
            "MSP Test1 + Test2",
            figures_dir / "msp_av_pred_by_pred_emotion_all.pdf",
        )
        plot_pair(
            full,
            msp_plot,
            "IEMOCAP",
            "MSP Test1 + Test2",
            figures_dir / "av_pred_by_pred_emotion_iemocap_msp_side_by_side.pdf",
        )
        print(figures_dir / "msp_av_pred_by_pred_emotion_all.pdf")
        print(figures_dir / "av_pred_by_pred_emotion_iemocap_msp_side_by_side.pdf")


if __name__ == "__main__":
    main()
