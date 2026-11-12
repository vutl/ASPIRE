#!/usr/bin/env python3
"""Evaluate fold 5 test set for the no-AV-supervision, no-global-audio ablation."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

sys.path.append(str(Path(__file__).resolve().parent.parent))

from data.iemocap_dataset import IEMOCAPDataset
from models.method2_crossattn_av_ablation_no_avsup_no_global import (
    Method2CrossAttentionAV_AblationNoAVSupNoGlobal,
)
from train.train_method2_kfold_wpro_ablation_no_avsup_no_global import (
    prepare_folds,
    validate,
)


def load_filtered_state(model, checkpoint_path: str):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
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
    return checkpoint, len(filtered), len(skipped)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/method2_wpro_ablation_no_avsup_no_global_config.yaml",
    )
    parser.add_argument(
        "--checkpoint",
        default="checkpoints/method2_kfold_wpro_ablation_no_avsup_no_global/fold_5/best_model.pt",
    )
    parser.add_argument("--fold", type=int, default=5)
    parser.add_argument("--device", default=None)
    parser.add_argument("--num_workers", type=int, default=0)
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    folds = prepare_folds(config["data"]["csv"])
    fold_data = folds[args.fold - 1]
    test_df = fold_data["test"]

    device_name = args.device or config.get("device", "cuda:0")
    device = torch.device(device_name if torch.cuda.is_available() else "cpu")

    print(f"Device: {device}")
    print(f"Config: {args.config}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Fold: {args.fold}")
    print(f"Test samples: {len(test_df)}")

    model = Method2CrossAttentionAV_AblationNoAVSupNoGlobal(
        num_emotions=config["model"]["num_emotions"],
        wavlm_model_name=config["model"]["wavlm_model_name"],
        wavlm_prosody_model_name=config["model"].get("wavlm_prosody_model_name", None),
        roberta_model_name=config["model"].get("roberta_model_name", "roberta-base"),
        hidden_dim=config["model"]["hidden_dim"],
        av_refinement_layers=config["model"].get("av_refinement_layers", 2),
        freeze_wavlm=config["model"]["freeze_wavlm"],
        freeze_roberta=config["model"].get("freeze_roberta", False),
        num_cross_attn_heads=config["model"]["num_cross_attn_heads"],
        post_attention_layers=config["model"].get("post_attention_layers", 2),
        av_prediction_layers=config["model"].get("av_prediction_layers", 3),
    ).to(device)

    checkpoint, loaded_count, skipped_count = load_filtered_state(model, args.checkpoint)
    print(f"Loaded epoch: {checkpoint.get('epoch')}")
    print(f"Stored val_accuracy: {checkpoint.get('val_accuracy')}")
    print(f"Loaded params: {loaded_count} | Skipped params: {skipped_count}")

    test_dataset = IEMOCAPDataset(
        test_df,
        audio_max_length=config["data"].get("max_audio_length", 160000),
        max_text_length=config["data"].get("max_text_length", 128),
        tokenizer_name=config["model"].get("roberta_model_name", "roberta-base"),
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    criterion = torch.nn.CrossEntropyLoss()
    metrics = validate(model, test_loader, criterion, device, writer=None, epoch=0, phase="Test")

    print("\n=== Fold Test Metrics ===")
    for key, value in metrics.items():
        print(f"{key}: {value:.6f}")


if __name__ == "__main__":
    main()
