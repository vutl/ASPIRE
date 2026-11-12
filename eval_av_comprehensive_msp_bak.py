#!/usr/bin/env python3
"""
Comprehensive Arousal-Valence Evaluation for MSP-Podcast (Backup Checkpoint)
Runs full Test1 and Test2 splits using the November backup checkpoint.
Prints metrics and saves plots with the same logic as eval_av_comprehensive_msp.py,
while allowing an explicit checkpoint path.
"""
import os
import argparse
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from data.msp_podcast_dataset import MSPPodcastDataset, collate_fn as msp_collate_fn
from models.method2_crossattn_av import Method2CrossAttentionAV

# Reuse the exact metric/plotting logic from the existing MSP comprehensive script
from eval_av_comprehensive_msp import (
    EMOTION_MAP_PLOT,
    compute_all_metrics,
    plot_av_comprehensive,
    print_metrics_report,
    load_checkpoint,
    evaluate_model,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Comprehensive A-V Evaluation for MSP-Podcast using backup checkpoint"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/method2_msp_wpro_best_bak.pt",
        help="Path to checkpoint file (default: November backup)",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="method2_msp_wpro_best_bak",
        help="Name used for logging/plotting outputs",
    )
    parser.add_argument(
        "--tsv",
        type=str,
        default="datasets/MSP-Podcast/msp_podcast.tsv",
        help="Path to MSP-Podcast TSV",
    )
    parser.add_argument(
        "--splits",
        type=str,
        nargs="+",
        default=["test1", "test2"],
        help="Splits to evaluate",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/av_comprehensive",
        help="Directory to save plots",
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        default="logs/eval_av_comprehensive",
        help="Directory to save metric logs",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device to use",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Batch size for inference",
    )
    parser.add_argument(
        "--use_wpro",
        action="store_true",
        default=None,
        help="Force enable W-PRO",
    )
    parser.add_argument(
        "--no_wpro",
        action="store_true",
        default=False,
        help="Force disable W-PRO",
    )
    return parser.parse_args()


def detect_wpro(args) -> bool:
    if args.no_wpro:
        return False
    if args.use_wpro is not None:
        return args.use_wpro
    # Auto-detect from model name or checkpoint path
    hint = (args.model_name + str(args.checkpoint)).lower()
    return "wpro" in hint


def write_metrics_log(log_path: Path, split: str, metrics: dict, true_counts: dict, pred_counts: dict):
    log_lines = []
    log_lines.append(f"Split: {split}")
    log_lines.append("CLASS COUNTS")
    log_lines.append(f"  True: {true_counts}")
    log_lines.append(f"  Pred: {pred_counts}")
    log_lines.append("CLASSIFICATION")
    log_lines.append(f"  accuracy      = {metrics['accuracy']:.4f}")
    log_lines.append(f"  f1_macro      = {metrics['f1_macro']:.4f}")
    log_lines.append(f"  f1_weighted   = {metrics['f1_weighted']:.4f}")
    log_lines.append("REGRESSION (A-V)")
    log_lines.append(f"  arousal_ccc   = {metrics['arousal_ccc']:.4f}")
    log_lines.append(f"  valence_ccc   = {metrics['valence_ccc']:.4f}")
    log_lines.append(f"  arousal_pearson = {metrics['arousal_pearson']:.4f}")
    log_lines.append(f"  valence_pearson = {metrics['valence_pearson']:.4f}")
    log_lines.append(f"  arousal_mae   = {metrics['arousal_mae']:.4f}")
    log_lines.append(f"  valence_mae   = {metrics['valence_mae']:.4f}")
    log_lines.append("CLUSTERING (pred A-V)")
    log_lines.append(f"  silhouette    = {metrics['silhouette_pred']:.4f}")
    log_lines.append(f"  davies_bouldin= {metrics['davies_bouldin_pred']:.4f}")
    log_lines.append("CIRCUMPLEX ALIGNMENT")
    log_lines.append(f"  overall       = {metrics['circumplex_alignment']:.4f}")
    for emo_name, details in metrics['circumplex_details'].items():
        pred_a, pred_v = details['predicted']
        exp_a, exp_v = details['expected']
        dev = details['deviation']
        log_lines.append(
            f"  {emo_name:8s}: pred=(A:{pred_a:+.2f}, V:{pred_v:+.2f}), "
            f"theory=(A:{exp_a:+.2f}, V:{exp_v:+.2f}), dev={dev:.3f}"
        )
    log_path.write_text("\n".join(log_lines))


def main():
    args = parse_args()
    use_wpro = detect_wpro(args)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        print(f"Error: checkpoint not found at {ckpt_path}")
        return

    output_dir = Path(args.output_dir) / args.model_name
    log_dir = Path(args.log_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'#'*80}")
    print(f" MSP-Podcast A-V Comprehensive (backup): {args.model_name}")
    print(f" Checkpoint: {ckpt_path}")
    print(f" Splits: {args.splits}")
    print(f" Use W-PRO: {use_wpro}")
    print(f" Device: {device}")
    print(f"{'#'*80}\n")

    model = Method2CrossAttentionAV(
        num_emotions=4,
        wavlm_model_name="microsoft/wavlm-base-plus",
        wavlm_prosody_model_name="Orange/Speaker-wavLM-pro" if use_wpro else None,
        roberta_model_name="roberta-base",
        hidden_dim=256,
        freeze_wavlm=True,
        freeze_roberta=True,
    ).to(device)
    model = load_checkpoint(model, ckpt_path)
    model.eval()

    all_metrics = []

    for split in args.splits:
        print(f"[{split.upper()}] loading data...")
        dataset = MSPPodcastDataset(
            args.tsv,
            split=split,
            audio_max_length=16000 * 10,
            sample_rate=16000,
            normalize_av=True,
            four_class_only=True,
        )
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=2,
            collate_fn=msp_collate_fn,
        )
        print(f"  Samples: {len(dataset)}")

        results = evaluate_model(model, loader, device)
        metrics = compute_all_metrics(results)
        all_metrics.append(metrics)

        # Helpful counts to spot class collapse
        true_counts = {EMOTION_MAP_PLOT[i]: int((results['emotion_true'] == i).sum()) for i in EMOTION_MAP_PLOT}
        pred_counts = {EMOTION_MAP_PLOT[i]: int((results['emotion_pred'] == i).sum()) for i in EMOTION_MAP_PLOT}

        print_metrics_report(metrics, args.model_name, split)

        plot_av_comprehensive(results, args.model_name, split, str(output_dir))

        log_path = log_dir / f"{args.model_name}_{split}.log"
        write_metrics_log(log_path, split, metrics, true_counts, pred_counts)
        print(f"  ✓ Logged metrics to {log_path}")

    if all_metrics:
        metric_keys = [
            "accuracy",
            "f1_macro",
            "f1_weighted",
            "arousal_ccc",
            "valence_ccc",
            "arousal_pearson",
            "valence_pearson",
            "arousal_mae",
            "valence_mae",
            "silhouette_pred",
            "davies_bouldin_pred",
            "circumplex_alignment",
        ]
        summary = {"metric": []}
        for split in args.splits:
            summary[split] = []

        for key in metric_keys:
            summary["metric"].append(key)
            for split, metrics in zip(args.splits, all_metrics):
                summary[split].append(metrics.get(key, np.nan))

        summary_path = output_dir / "summary_metrics.csv"
        pd.DataFrame(summary).to_csv(summary_path, index=False)
        print(f"\n✓ Summary saved to: {summary_path}\n")

    del model
    torch.cuda.empty_cache()
    print("✅ Evaluation complete.")


if __name__ == "__main__":
    main()
