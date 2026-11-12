#!/usr/bin/env python3
"""Evaluate an IEMOCAP No-AV-Sup checkpoint on MSP-Podcast test sets."""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import torch

sys.path.append(str(Path(__file__).resolve().parent.parent))

from data.msp_podcast_dataset import get_msp_podcast_dataloaders
from models.method2_crossattn_av_ablation_no_avsup import Method2CrossAttentionAV_AblationNoAVSup
from train.train_method2_msp_wpro_ablation_no_avsup import build_criterion, load_config, validate


def pick_best_fold(ckpt_root: Path) -> int:
    csv_path = ckpt_root / "kfold_results.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing {csv_path}")
    best_fold = None
    best_acc = float("-inf")
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            acc = float(row["accuracy"])
            fold = int(row["fold"])
            if acc > best_acc:
                best_acc = acc
                best_fold = fold
    if best_fold is None:
        raise RuntimeError(f"No rows found in {csv_path}")
    return best_fold


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/msp_podcast_wpro_ablation_no_avsup_config.yaml")
    parser.add_argument("--checkpoint_root", default="checkpoints/method2_kfold_wpro_ablation_no_avsup")
    parser.add_argument("--fold", type=int, default=None, help="Fold to use; default picks best fold from kfold_results.csv")
    parser.add_argument("--checkpoint", default=None, help="Explicit checkpoint path; overrides --fold")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch_size", type=int, default=8)
    args = parser.parse_args()

    config = load_config(args.config)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    ckpt_root = Path(args.checkpoint_root)

    if args.checkpoint:
        checkpoint_path = Path(args.checkpoint)
        fold_used = args.fold
    else:
        fold_used = args.fold or pick_best_fold(ckpt_root)
        checkpoint_path = ckpt_root / f"fold_{fold_used}" / "best_model.pt"

    batch_size = args.batch_size or config["training"]["batch_size"]
    _, _, test1_loader, test2_loader = get_msp_podcast_dataloaders(
        tsv_path=config["data"]["tsv_path"],
        batch_size=batch_size,
        num_workers=2,
        audio_max_length=config["data"]["max_audio_length"],
        tokenizer_name=config["model"]["roberta_model_name"],
        max_text_length=config["data"]["max_text_length"],
        pin_memory=False,
    )

    model = Method2CrossAttentionAV_AblationNoAVSup(
        num_emotions=config["model"]["num_emotions"],
        wavlm_model_name=config["model"]["wavlm_model_name"],
        wavlm_prosody_model_name=config["model"].get("wavlm_prosody_model_name", None),
        roberta_model_name=config["model"]["roberta_model_name"],
        hidden_dim=config["model"]["hidden_dim"],
        av_refinement_layers=config["model"].get("av_refinement_layers", 2),
        freeze_wavlm=config["model"]["freeze_wavlm"],
        freeze_roberta=config["model"]["freeze_roberta"],
        num_cross_attn_heads=config["model"]["num_cross_attn_heads"],
        post_attention_layers=config["model"]["post_attention_layers"],
        av_prediction_layers=config["model"].get("av_prediction_layers", 3),
    ).to(device)

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
    model_state = model.state_dict()
    filtered = {
        k: v
        for k, v in state_dict.items()
        if k in model_state and isinstance(v, torch.Tensor) and v.shape == model_state[k].shape
    }
    skipped = len(state_dict) - len(filtered)
    model.load_state_dict(filtered, strict=False)

    criterion = build_criterion(config)

    print(f"Device: {device}")
    print(f"Checkpoint: {checkpoint_path}")
    if fold_used is not None:
        print(f"Fold used: {fold_used}")
    print(f"Loaded params: {len(filtered)} | Skipped: {skipped}")

    print("\n=== MSP Test1 ===")
    test1_metrics = validate(model, test1_loader, criterion, device, writer=None, epoch=0, phase="Test1", logger=None)
    print(test1_metrics)

    print("\n=== MSP Test2 ===")
    test2_metrics = validate(model, test2_loader, criterion, device, writer=None, epoch=0, phase="Test2", logger=None)
    print(test2_metrics)


if __name__ == "__main__":
    main()
