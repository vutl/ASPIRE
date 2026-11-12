#!/usr/bin/env python3
"""
Multi-Checkpoint Evaluation for MSP-Podcast (20% Test1 Stratified)
===================================================================
Evaluates 5 checkpoint variants on 20% stratified test1 subset.
Identifies which checkpoint is most reliable before running full evaluation.

Checkpoints evaluated:
  1. method2_msp_wpro_continue/last_model.pt (Nov 27 08:06)
  2. best_model.pt.20251128_114858.bak (Nov 28 11:49, epoch 7)
  3. best_model.pt.20251128_115036.bak (Nov 28 11:50)
  4. best_model.pt.20251128_115150.bak (Nov 28 11:50)
  5. best_model.pt.bak_20251128_113034 (Nov 28 11:31)

Outputs: metrics per checkpoint, plots, and comparison CSV.
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
    EMOTION_COLORS,
    EMOTION_MARKERS,
    CIRCUMPLEX_RUSSELL,
    compute_ccc,
    compute_cluster_metrics,
    compute_circumplex_alignment,
    load_checkpoint,
    evaluate_model,
    compute_all_metrics,
    plot_av_comprehensive,
    print_metrics_report,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Multi-checkpoint evaluation on 20% test1 subset"
    )
    parser.add_argument(
        "--tsv",
        type=str,
        default="datasets/MSP-Podcast/msp_podcast.tsv",
        help="Path to MSP-Podcast TSV",
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
        help="Batch size",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for 20% stratified sampling",
    )
    return parser.parse_args()


def get_stratified_subset_utt_ids(tsv_path: str, split: str, fraction: float = 0.2, seed: int = 42) -> list:
    """
    Load TSV, filter to 4-class, stratify by emotion, and return a list of utt_ids.
    """
    df = pd.read_csv(tsv_path, sep='\t')
    df = df[df['split'] == split]
    df = df[df['label'].isin(['ang', 'sad', 'hap', 'neu'])]
    
    # Stratified sampling
    np.random.seed(seed)
    n_subset = int(len(df) * fraction)
    sampled_df = df.groupby('label', group_keys=False).apply(
        lambda x: x.sample(n=max(1, int(len(x) * fraction)), random_state=seed)
    )
    
    return sampled_df['utt_id'].tolist()


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Checkpoints to evaluate
    checkpoints = [
        ("continue_last", "checkpoints/method2_msp_wpro_continue/last_model.pt"),
        ("bak_114858", "checkpoints/method2_msp_wpro/best_model.pt.20251128_114858.bak"),
        ("bak_115036", "checkpoints/method2_msp_wpro/best_model.pt.20251128_115036.bak"),
        ("bak_115150", "checkpoints/method2_msp_wpro/best_model.pt.20251128_115150.bak"),
        ("bak_113034", "checkpoints/method2_msp_wpro/best_model.pt.bak_20251128_113034"),
    ]

    # Verify all checkpoints exist
    for name, path in checkpoints:
        if not Path(path).exists():
            print(f"Warning: {name} not found at {path}")

    output_dir = Path(args.output_dir) / "multi_checkpoint_20pct"
    log_dir = Path(args.log_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'#'*80}")
    print(f" Multi-Checkpoint Evaluation: MSP-Podcast 20% Test1")
    print(f" Device: {device}")
    print(f"{'#'*80}\n")

    # Get 20% stratified subset
    print(f"[Preparation] Loading 20% stratified test1 subset (seed={args.seed})...")
    subset_utt_ids = get_stratified_subset_utt_ids(
        args.tsv, "test1", fraction=0.2, seed=args.seed
    )
    print(f"  Subset size: {len(subset_utt_ids)}")

    # Load dataset with subset filter
    dataset = MSPPodcastDataset(
        args.tsv,
        split="test1",
        audio_max_length=16000 * 10,
        sample_rate=16000,
        normalize_av=True,
        four_class_only=True,
        filter_utt_ids=subset_utt_ids,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        collate_fn=msp_collate_fn,
    )
    print(f"  Emotion distribution: {dataset.df['label'].value_counts().to_dict()}")

    # Evaluate all checkpoints
    all_results = {}
    all_metrics = {}

    for ckpt_name, ckpt_path in checkpoints:
        if not Path(ckpt_path).exists():
            print(f"\n✗ Skipping {ckpt_name}: {ckpt_path} not found")
            continue

        print(f"\n[{ckpt_name}] Loading checkpoint: {ckpt_path}")
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

        print(f"  Running inference on {len(dataset)} samples...")
        try:
            results = evaluate_model(model, loader, device)
            metrics = compute_all_metrics(results)
            all_results[ckpt_name] = results
            all_metrics[ckpt_name] = metrics

            # Print report
            print_metrics_report(metrics, "multi_ckpt", ckpt_name)

            # Plot
            plot_av_comprehensive(
                results, "multi_ckpt", ckpt_name, str(output_dir)
            )

            # Log metrics
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
                f"Samples: {len(dataset)}",
                f"True counts: {true_counts}",
                f"Pred counts: {pred_counts}",
                "METRICS",
                f"  accuracy      = {metrics['accuracy']:.4f}",
                f"  f1_macro      = {metrics['f1_macro']:.4f}",
                f"  f1_weighted   = {metrics['f1_weighted']:.4f}",
                f"  arousal_ccc   = {metrics['arousal_ccc']:.4f}",
                f"  valence_ccc   = {metrics['valence_ccc']:.4f}",
                f"  arousal_mae   = {metrics['arousal_mae']:.4f}",
                f"  valence_mae   = {metrics['valence_mae']:.4f}",
                f"  silhouette    = {metrics['silhouette_pred']:.4f}",
                f"  davies_bouldin= {metrics['davies_bouldin_pred']:.4f}",
                f"  circumplex_align = {metrics['circumplex_alignment']:.4f}",
            ]
            log_path = log_dir / f"multi_ckpt_{ckpt_name}_20pct.log"
            log_path.write_text("\n".join(log_lines))
            print(f"  ✓ Logged to {log_path}")

        except Exception as e:
            print(f"  ✗ Evaluation failed: {e}")
            import traceback
            traceback.print_exc()
            continue
        finally:
            del model
            torch.cuda.empty_cache()

    # Comparison summary
    if all_metrics:
        print(f"\n{'#'*80}")
        print(f" COMPARISON SUMMARY")
        print(f"{'#'*80}\n")

        metric_keys = [
            "accuracy",
            "f1_macro",
            "arousal_ccc",
            "valence_ccc",
            "silhouette_pred",
            "circumplex_alignment",
        ]

        comparison_data = {"checkpoint": list(all_metrics.keys())}
        for key in metric_keys:
            comparison_data[key] = [all_metrics[ckpt].get(key, np.nan) for ckpt in all_metrics]

        df_comparison = pd.DataFrame(comparison_data)
        print(df_comparison.to_string(index=False))

        # Save comparison
        csv_path = output_dir / "checkpoint_comparison_20pct.csv"
        df_comparison.to_csv(csv_path, index=False)
        print(f"\n✓ Comparison saved to: {csv_path}\n")

        # Identify best checkpoint
        best_ckpt = df_comparison.loc[df_comparison['accuracy'].idxmax(), 'checkpoint']
        best_acc = df_comparison.loc[df_comparison['accuracy'].idxmax(), 'accuracy']
        print(f"🏆 Best by accuracy: {best_ckpt} ({best_acc:.4f})")

    print("✅ Multi-checkpoint evaluation complete.")


if __name__ == "__main__":
    main()
