#!/usr/bin/env python3
"""
Comprehensive Multi-Checkpoint Evaluation for MSP-Podcast (Full Splits)
=========================================================================
Evaluates 5 checkpoint variants on full Test1 and Test2 splits.
Uses the same comprehensive metrics and plots as eval_av_comprehensive_msp.py.

Checkpoints evaluated:
  1. method2_msp_wpro_continue/last_model.pt (Nov 27 08:06)
  2. best_model.pt.20251128_114858.bak (Nov 28 11:49, epoch 7)
  3. best_model.pt.20251128_115036.bak (Nov 28 11:50)
  4. best_model.pt.20251128_115150.bak (Nov 28 11:50)
  5. best_model.pt.bak_20251128_113034 (Nov 28 11:31)

Outputs: 
  - Per-checkpoint plots for test1/test2
  - Per-checkpoint summary CSV
  - Comparison table across all 5 checkpoints
  - Per-split/checkpoint detailed logs
"""
import os
import sys
import argparse
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.msp_podcast_dataset import MSPPodcastDataset, collate_fn as msp_collate_fn
from models.method2_crossattn_av import Method2CrossAttentionAV

# Import utilities from existing comprehensive script
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
        description="Comprehensive multi-checkpoint evaluation on full test1+test2"
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
        help="Directory to save logs",
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
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Checkpoints to evaluate (in order: older -> newest)
    checkpoints = [
        ("continue_last", "checkpoints/method2_msp_wpro_continue/last_model.pt"),
        ("bak_113034", "checkpoints/method2_msp_wpro/best_model.pt.bak_20251128_113034"),
        ("bak_114858", "checkpoints/method2_msp_wpro/best_model.pt.20251128_114858.bak"),
        ("bak_115036", "checkpoints/method2_msp_wpro/best_model.pt.20251128_115036.bak"),
        ("bak_115150", "checkpoints/method2_msp_wpro/best_model.pt.20251128_115150.bak"),
    ]

    # Verify all checkpoints exist
    available_checkpoints = []
    for name, path in checkpoints:
        if Path(path).exists():
            available_checkpoints.append((name, path))
        else:
            print(f"Warning: {name} not found at {path}")

    if not available_checkpoints:
        print("Error: No checkpoints found!")
        return

    output_base = Path(args.output_dir) / "multi_checkpoint_full"
    log_dir = Path(args.log_dir)
    output_base.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'#'*80}")
    print(f" Comprehensive Multi-Checkpoint Evaluation: MSP-Podcast Full Splits")
    print(f" Checkpoints: {len(available_checkpoints)}")
    print(f" Splits: {args.splits}")
    print(f" Device: {device}")
    print(f"{'#'*80}\n")

    # Store results: {ckpt_name: {split: metrics}}
    all_results = {}

    for ckpt_name, ckpt_path in available_checkpoints:
        print(f"\n{'='*80}")
        print(f" [{ckpt_name}] Loading checkpoint: {ckpt_path}")
        print(f"{'='*80}")

        try:
            model = Method2CrossAttentionAV(
                num_emotions=4,
                wavlm_model_name="microsoft/wavlm-base-plus",
                wavlm_prosody_model_name="Orange/Speaker-wavLM-pro",
                roberta_model_name="roberta-base",
                hidden_dim=256,
                freeze_wavlm=True,
                freeze_roberta=True,
            ).to(device)
            model = load_checkpoint(model, ckpt_path)
            model.eval()
        except Exception as e:
            print(f"  ✗ Failed to load checkpoint: {e}")
            continue

        all_results[ckpt_name] = {}
        ckpt_metrics = {}

        for split in args.splits:
            print(f"\n  [{split.upper()}] Loading dataset...")
            try:
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
                print(f"    Samples: {len(dataset)}")
                print(f"    Distribution: {dataset.df['label'].value_counts().to_dict()}")

                # Evaluate
                results = evaluate_model(model, loader, device)
                metrics = compute_all_metrics(results)
                all_results[ckpt_name][split] = metrics

                ckpt_metrics[split] = metrics

                # Print report
                print_metrics_report(metrics, ckpt_name, split)

                # Plot
                output_dir = output_base / ckpt_name
                output_dir.mkdir(parents=True, exist_ok=True)
                plot_av_comprehensive(results, ckpt_name, split, str(output_dir))

                # Log detailed metrics
                true_counts = {
                    EMOTION_MAP_PLOT[i]: int((results['emotion_true'] == i).sum())
                    for i in EMOTION_MAP_PLOT
                }
                pred_counts = {
                    EMOTION_MAP_PLOT[i]: int((results['emotion_pred'] == i).sum())
                    for i in EMOTION_MAP_PLOT
                }

                log_lines = [
                    f"Checkpoint: {ckpt_path}",
                    f"Split: {split}",
                    f"Samples: {len(dataset)}",
                    f"True counts: {true_counts}",
                    f"Pred counts: {pred_counts}",
                    "",
                    "METRICS",
                    f"  accuracy           = {metrics['accuracy']:.4f}",
                    f"  f1_macro           = {metrics['f1_macro']:.4f}",
                    f"  f1_weighted        = {metrics['f1_weighted']:.4f}",
                    f"  arousal_ccc        = {metrics['arousal_ccc']:.4f}",
                    f"  valence_ccc        = {metrics['valence_ccc']:.4f}",
                    f"  arousal_pearson    = {metrics['arousal_pearson']:.4f}",
                    f"  valence_pearson    = {metrics['valence_pearson']:.4f}",
                    f"  arousal_mae        = {metrics['arousal_mae']:.4f}",
                    f"  valence_mae        = {metrics['valence_mae']:.4f}",
                    f"  silhouette_pred    = {metrics['silhouette_pred']:.4f}",
                    f"  davies_bouldin_pred= {metrics['davies_bouldin_pred']:.4f}",
                    f"  circumplex_align   = {metrics['circumplex_alignment']:.4f}",
                ]
                log_path = log_dir / f"comprehensive_{ckpt_name}_{split}.log"
                log_path.write_text("\n".join(log_lines))
                print(f"    ✓ Logged to {log_path}")

            except Exception as e:
                print(f"    ✗ {split} evaluation failed: {e}")
                import traceback
                traceback.print_exc()
                continue

        # Save per-checkpoint summary CSV
        if ckpt_metrics:
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

            summary_data = {"metric": metric_keys}
            for split in args.splits:
                if split in ckpt_metrics:
                    summary_data[split] = [
                        ckpt_metrics[split].get(k, np.nan) for k in metric_keys
                    ]
                else:
                    summary_data[split] = [np.nan] * len(metric_keys)

            summary_path = (output_base / ckpt_name) / f"summary_{ckpt_name}.csv"
            pd.DataFrame(summary_data).to_csv(summary_path, index=False)
            print(f"\n  ✓ Summary saved to {summary_path}")

        del model
        torch.cuda.empty_cache()

    # === CROSS-CHECKPOINT COMPARISON TABLE ===
    if all_results:
        print(f"\n{'#'*80}")
        print(f" CROSS-CHECKPOINT COMPARISON")
        print(f"{'#'*80}\n")

        metric_keys = [
            "accuracy",
            "f1_macro",
            "arousal_ccc",
            "valence_ccc",
            "silhouette_pred",
            "circumplex_alignment",
        ]

        for split in args.splits:
            print(f"\n{split.upper()} METRICS")
            print("-" * 100)

            comparison_data = {"checkpoint": list(all_results.keys())}
            for key in metric_keys:
                comparison_data[key] = [
                    all_results[ckpt].get(split, {}).get(key, np.nan)
                    for ckpt in all_results
                ]

            df_comparison = pd.DataFrame(comparison_data)
            print(df_comparison.to_string(index=False))

            # Save comparison CSV
            comparison_csv = output_base / f"comparison_{split}.csv"
            df_comparison.to_csv(comparison_csv, index=False)
            print(f"\n✓ Saved to {comparison_csv}")

            # Find best by accuracy
            valid_idx = [
                i
                for i, v in enumerate(comparison_data["accuracy"])
                if not np.isnan(v)
            ]
            if valid_idx:
                best_idx = valid_idx[
                    np.argmax([comparison_data["accuracy"][i] for i in valid_idx])
                ]
                best_ckpt = comparison_data["checkpoint"][best_idx]
                best_acc = comparison_data["accuracy"][best_idx]
                print(f"🏆 Best by accuracy: {best_ckpt} ({best_acc:.4f})")

    print(f"\n✅ Comprehensive multi-checkpoint evaluation complete.")
    print(f"   Outputs: {output_base}")


if __name__ == "__main__":
    main()
