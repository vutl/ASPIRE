#!/usr/bin/env python3
"""
Generate heatmap of logit shifts (Δz) induced by AV guidance in Method2 W-PRO.

Visualizes how guidance selectively boosts correct class logits and suppresses
confusable alternatives at the decision level.

Usage:
  python3 scripts/heatmap_guidance_logits.py \
    --checkpoints_dir checkpoints/method2_kfold_wpro \
    --config configs/method2_wpro_config.yaml \
    --csv datasets/IEMOCAP_KFOLD/tenfolds/all.csv \
    --fold 1 2 3 4 5 \
    --which best \
    --device cuda:0 \
    --top_k 100 \
    --out_prefix logs/heatmap_guidance_logits
"""
import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

# Local imports
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
from data.iemocap_dataset import IEMOCAPDataset, collate_fn
from models.method2_crossattn_av import Method2CrossAttentionAV


def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def prepare_folds(csv_path):
    df = pd.read_csv(csv_path)
    max_fold = df["fold"].max()
    if max_fold == 10:
        fold_to_session = {1: 1, 2: 1, 3: 2, 4: 2, 5: 3, 6: 3, 7: 4, 8: 4, 9: 5, 10: 5}
        df["session"] = df["fold"].map(fold_to_session)
    else:
        df["session"] = df["fold"]

    folds = []
    for s in range(1, 6):
        folds.append({"session": s, "test": df[df["session"] == s], "train_val": df[df["session"] != s]})
    return folds


def load_checkpoint(model, ckpt_path, device):
    torch.serialization.add_safe_globals([np._core.multiarray.scalar])
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)


def compute_logit_differences(model, loader, device, max_samples=None):
    """
    Compute Δz = z_full - z_mask for each sample.
    Args:
        max_samples: optional int, if set>0 will cap processed samples (use 0/None for full dataset).
    Returns:
        delta_logits: [N, 4] - logit differences
        labels: [N] - ground truth labels
        corrected_mask: [N] - samples corrected by guidance (wrong→right)
        degraded_mask: [N] - samples degraded by guidance (right→wrong)
    """
    
    model.eval()
    
    all_delta_logits = []
    all_labels = []
    all_preds_full = []
    all_preds_masked = []
    
    processed = 0
    limit = max_samples if (max_samples is not None and max_samples > 0) else None

    with torch.no_grad():
        for batch in tqdm(loader, desc="Computing logit differences"):
            audio = batch["audio"].to(device)
            text_input_ids = batch["text_input_ids"].to(device)
            text_attention_mask = batch["text_attention_mask"].to(device)
            labels = batch["emotion_label"].to(device)
            
            # Full guidance
            out_full = model(
                audio_input=audio,
                text_input=text_input_ids,
                text_attention_mask=text_attention_mask,
                use_guidance=True
            )
            
            # Masked guidance
            out_masked = model(
                audio_input=audio,
                text_input=text_input_ids,
                text_attention_mask=text_attention_mask,
                use_guidance=False
            )
            
            # Compute differences
            logits_full = out_full["emotion_logits"]
            logits_masked = out_masked["emotion_logits"]
            delta_logits = logits_full - logits_masked
            
            preds_full = torch.argmax(logits_full, dim=1)
            preds_masked = torch.argmax(logits_masked, dim=1)
            
            all_delta_logits.append(delta_logits.cpu())
            all_labels.append(labels.cpu())
            all_preds_full.append(preds_full.cpu())
            all_preds_masked.append(preds_masked.cpu())

            processed += labels.size(0)
            if limit is not None and processed >= limit:
                break
    
    # Concatenate all batches
    delta_logits = torch.cat(all_delta_logits).numpy()  # [N, 4]
    labels = torch.cat(all_labels).numpy()  # [N]
    preds_full = torch.cat(all_preds_full).numpy()  # [N]
    preds_masked = torch.cat(all_preds_masked).numpy()  # [N]
    
    # Identify corrected and degraded samples
    correct_full = (preds_full == labels)
    correct_masked = (preds_masked == labels)
    corrected_mask = (~correct_masked) & correct_full  # wrong → right
    degraded_mask = correct_masked & (~correct_full)   # right → wrong
    
    return delta_logits, labels, corrected_mask, degraded_mask


def create_logit_heatmap(delta_logits, labels, sample_mask, class_names, title, out_path):
    """
    Create heatmap of logit shifts
    Args:
        delta_logits: [N, 4] - logit differences
        labels: [N] - ground truth labels
        sample_mask: [N] - boolean mask for samples to include
        class_names: list of class names
        title: plot title
        out_path: output file path
    """
    # Filter samples
    selected_deltas = delta_logits[sample_mask]  # [K, 4]
    selected_labels = labels[sample_mask]  # [K]
    
    if len(selected_deltas) == 0:
        print(f"No samples selected for {title}")
        return
    
    # Sort by ground truth class for better visualization
    sort_idx = np.argsort(selected_labels)
    selected_deltas = selected_deltas[sort_idx]
    selected_labels = selected_labels[sort_idx]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(8, max(6, len(selected_deltas) * 0.1)))
    
    # Create heatmap
    im = ax.imshow(selected_deltas, cmap='RdBu_r', aspect='auto', vmin=-2, vmax=2)
    
    # Set ticks and labels
    ax.set_xticks(range(len(class_names)))
    ax.set_xticklabels(class_names)
    ax.set_xlabel('Emotion Classes')
    ax.set_ylabel('Samples (sorted by ground truth)')
    
    # Add ground truth indicators on the right
    for i, gt_label in enumerate(selected_labels):
        ax.text(len(class_names) + 0.1, i, f'GT:{gt_label}', 
                va='center', ha='left', fontsize=8, 
                color=plt.cm.tab10(gt_label))
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Δlogit (Full - Masked)', rotation=270, labelpad=20)
    
    # Title
    ax.set_title(f'{title}\n(N={len(selected_deltas)} samples)', fontsize=12, pad=20)
    
    plt.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved heatmap to {out_path}")


def create_summary_statistics(delta_logits, labels, corrected_mask, degraded_mask, out_path):
    """
    Create summary statistics and bar plots
    """
    # Overall statistics
    delta_norms = np.linalg.norm(delta_logits, axis=1)
    
    stats = {
        'total_samples': int(len(delta_logits)),
        'corrected_samples': int(np.sum(corrected_mask)),
        'degraded_samples': int(np.sum(degraded_mask)),
        'unchanged_samples': int(np.sum(~corrected_mask & ~degraded_mask)),
        'mean_delta_norm': float(np.mean(delta_norms)),
        'std_delta_norm': float(np.std(delta_norms)),
        'max_delta_norm': float(np.max(delta_norms))
    }
    
    # Per-class impact
    class_stats = {}
    class_names = ['Angry', 'Happy', 'Neutral', 'Sad']
    
    for c in range(4):
        class_mask = (labels == c)
        if np.sum(class_mask) > 0:
            class_delta = delta_logits[class_mask]
            class_stats[f'class_{c}'] = {
                'n_samples': int(np.sum(class_mask)),
                'mean_self_boost': float(np.mean(class_delta[:, c])),  # boost for own class
                'std_self_boost': float(np.std(class_delta[:, c])),
                'corrected_ratio': float(np.mean(corrected_mask[class_mask])),
                'degraded_ratio': float(np.mean(degraded_mask[class_mask]))
            }
    
    # Save stats
    stats['per_class'] = class_stats
    
    import json
    with open(out_path.replace('.png', '_stats.json'), 'w') as f:
        json.dump(stats, f, indent=2)
    
    # Create summary plot
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # 1. Correction/degradation by class
    classes = list(range(4))
    corrected_ratios = [class_stats[f'class_{c}']['corrected_ratio'] for c in classes]
    degraded_ratios = [class_stats[f'class_{c}']['degraded_ratio'] for c in classes]
    
    x = np.arange(len(classes))
    width = 0.35
    axes[0, 0].bar(x - width/2, corrected_ratios, width, label='Corrected (Wrong→Right)', color='green', alpha=0.7)
    axes[0, 0].bar(x + width/2, degraded_ratios, width, label='Degraded (Right→Wrong)', color='red', alpha=0.7)
    axes[0, 0].set_xlabel('Emotion Class')
    axes[0, 0].set_ylabel('Ratio')
    axes[0, 0].set_title('Guidance Impact by Class')
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(class_names)
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)
    
    # 2. Self-boost distribution
    self_boosts = [delta_logits[labels == c, c] for c in classes]
    axes[0, 1].boxplot(self_boosts, labels=class_names)
    axes[0, 1].set_xlabel('Emotion Class')
    axes[0, 1].set_ylabel('Δlogit (self-class)')
    axes[0, 1].set_title('Self-class Logit Boost Distribution')
    axes[0, 1].grid(alpha=0.3)
    axes[0, 1].axhline(y=0, color='black', linestyle='--', alpha=0.5)
    
    # 3. Delta norm histogram
    axes[1, 0].hist(delta_norms, bins=50, alpha=0.7, color='blue')
    axes[1, 0].axvline(np.mean(delta_norms), color='red', linestyle='--', label=f'Mean: {np.mean(delta_norms):.3f}')
    axes[1, 0].set_xlabel('||Δlogit||')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Distribution of Logit Change Magnitude')
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.3)
    
    # 4. Confusion matrix of changes
    change_matrix = np.zeros((3, 4))  # [corrected, degraded, unchanged] x [class]
    for c in classes:
        class_mask = (labels == c)
        n_class = np.sum(class_mask)
        if n_class > 0:
            change_matrix[0, c] = np.sum(corrected_mask[class_mask]) / n_class
            change_matrix[1, c] = np.sum(degraded_mask[class_mask]) / n_class
            change_matrix[2, c] = 1 - change_matrix[0, c] - change_matrix[1, c]
    
    im = axes[1, 1].imshow(change_matrix, cmap='RdYlBu_r', aspect='auto')
    axes[1, 1].set_xticks(range(4))
    axes[1, 1].set_xticklabels(class_names)
    axes[1, 1].set_yticks(range(3))
    axes[1, 1].set_yticklabels(['Corrected', 'Degraded', 'Unchanged'])
    axes[1, 1].set_title('Guidance Impact Matrix')
    
    # Add text annotations
    for i in range(3):
        for j in range(4):
            axes[1, 1].text(j, i, f'{change_matrix[i, j]:.2f}', 
                           ha='center', va='center', color='black', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved summary statistics to {out_path}")
    return stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints_dir", type=str, required=False,
                        help='Path to checkpoints directory (for IEMOCAP kfold) or file for MSP')
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--csv", type=str, required=False,
                        help='IEMOCAP csv (used when --dataset iemocap)')
    parser.add_argument("--dataset", type=str, choices=['iemocap', 'msp'], default='iemocap',
                        help='Which dataset to run the heatmap on')
    parser.add_argument("--tsv", type=str, required=False,
                        help='MSP-Podcast TSV path (used when --dataset msp)')
    parser.add_argument("--checkpoint", type=str, required=False,
                        help='Single checkpoint file to use for MSP runs (overrides --checkpoints_dir)')
    parser.add_argument("--fold", type=int, nargs="*", default=[1,2,3,4,5])
    parser.add_argument("--which", type=str, choices=["best", "last"], default="best")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_samples", type=int, default=0, help="Limit samples processed (0 = full)")
    parser.add_argument("--top_k", type=int, default=100, help="Top K samples by ||Δlogit||")
    parser.add_argument("--out_prefix", type=str, default="logs/heatmap_guidance_logits")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    cfg = load_config(args.config)
    folds = None
    if args.dataset == 'iemocap':
        if not args.csv:
            raise SystemExit('When --dataset iemocap you must provide --csv path')
        folds = prepare_folds(args.csv)
    
    class_names = ['Angry', 'Happy', 'Neutral', 'Sad']
    
    # Collect data from all folds
    all_delta_logits = []
    all_labels = []
    all_corrected = []
    all_degraded = []
    
    if args.dataset == 'iemocap':
        for fold_idx in args.fold:
            if fold_idx < 1 or fold_idx > 5:
                continue

            print(f"\n=== Processing Fold {fold_idx} ===")

            fold_data = folds[fold_idx - 1]
            test_df = fold_data["test"]

            test_dataset = IEMOCAPDataset(
                test_df,
                audio_max_length=cfg.get("data", {}).get("max_audio_length", 160000),
                tokenizer_name=cfg.get("model", {}).get("roberta_model_name", "roberta-base"),
                max_text_length=cfg.get("data", {}).get("max_text_length", 128),
            )
            test_loader = DataLoader(
                test_dataset,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=2,
                collate_fn=collate_fn,
                pin_memory=True,
            )

            # Load model
            model = Method2CrossAttentionAV(
                num_emotions=cfg["model"].get("num_emotions", 4),
                wavlm_model_name=cfg["model"].get("wavlm_model_name"),
                wavlm_prosody_model_name=cfg["model"].get("wavlm_prosody_model_name"),
                roberta_model_name=cfg["model"].get("roberta_model_name", "roberta-base"),
                hidden_dim=cfg["model"].get("hidden_dim", 256),
                av_refinement_layers=cfg["model"].get("av_refinement_layers", 2),
                freeze_wavlm=cfg["model"].get("freeze_wavlm", False),
                freeze_roberta=cfg["model"].get("freeze_roberta", False),
                num_cross_attn_heads=cfg["model"].get("num_cross_attn_heads", 8),
                post_attention_layers=cfg["model"].get("post_attention_layers", 2),
                av_prediction_layers=cfg["model"].get("av_prediction_layers", 3),
            )

            ckpt_path = Path(args.checkpoints_dir) / f"fold_{fold_idx}" / ("best_model.pt" if args.which == "best" else "last_model.pt")
            if not ckpt_path.exists():
                print(f"Checkpoint not found: {ckpt_path}, skip")
                continue

            load_checkpoint(model, ckpt_path, device)
            model.to(device)

            # Compute logit differences for this fold
            delta_logits, labels, corrected_mask, degraded_mask = compute_logit_differences(
                model, test_loader, device, max_samples=args.max_samples
            )

            all_delta_logits.append(delta_logits)
            all_labels.append(labels)
            all_corrected.append(corrected_mask)
            all_degraded.append(degraded_mask)

            print(f"Fold {fold_idx}: {len(delta_logits)} samples, {np.sum(corrected_mask)} corrected, {np.sum(degraded_mask)} degraded")
    else:
        # MSP mode: use MSP dataloaders and a single checkpoint
        from data.msp_podcast_dataset import get_msp_podcast_dataloaders

        if not args.tsv:
            raise SystemExit('When --dataset msp you must provide --tsv path to MSP-Podcast TSV')

        # Build dataloaders
        train_loader, dev_loader, test1_loader, test2_loader = get_msp_podcast_dataloaders(
            tsv_path=args.tsv,
            batch_size=args.batch_size,
            num_workers=0,
            audio_max_length=cfg.get('data', {}).get('max_audio_length', 160000),
            tokenizer_name=cfg.get('model', {}).get('roberta_model_name', 'roberta-base'),
            max_text_length=cfg.get('data', {}).get('max_text_length', 128)
        )

        # Create combined dataset and loader
        combined_list = [test1_loader.dataset[i] for i in range(len(test1_loader.dataset))] + [test2_loader.dataset[i] for i in range(len(test2_loader.dataset))]
        combined_loader = DataLoader(combined_list, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=collate_fn)

        # Instantiate model
        model = Method2CrossAttentionAV(
            num_emotions=cfg['model'].get('num_emotions', 4),
            wavlm_model_name=cfg['model'].get('wavlm_model_name'),
            wavlm_prosody_model_name=cfg['model'].get('wavlm_prosody_model_name'),
            roberta_model_name=cfg['model'].get('roberta_model_name', 'roberta-base'),
            hidden_dim=cfg['model'].get('hidden_dim', 256),
            av_refinement_layers=cfg['model'].get('av_refinement_layers', 2),
            freeze_wavlm=cfg['model'].get('freeze_wavlm', False),
            freeze_roberta=cfg['model'].get('freeze_roberta', False),
            num_cross_attn_heads=cfg['model'].get('num_cross_attn_heads', 8),
            post_attention_layers=cfg['model'].get('post_attention_layers', 2),
            av_prediction_layers=cfg['model'].get('av_prediction_layers', 3),
        )
        model.to(device)

        # Determine checkpoint path
        if args.checkpoint:
            ckpt_path = Path(args.checkpoint)
        elif args.checkpoints_dir and Path(args.checkpoints_dir).is_file():
            ckpt_path = Path(args.checkpoints_dir)
        else:
            ckpt_path = Path('checkpoints/method2_msp_wpro_best_bak.pt')

        if not ckpt_path.exists():
            raise SystemExit(f'Checkpoint not found: {ckpt_path}')

        load_checkpoint(model, str(ckpt_path), device)

        # Compute logit differences for combined MSP test set
        delta_logits, labels, corrected_mask, degraded_mask = compute_logit_differences(
            model, combined_loader, device, max_samples=args.max_samples
        )

        all_delta_logits.append(delta_logits)
        all_labels.append(labels)
        all_corrected.append(corrected_mask)
        all_degraded.append(degraded_mask)
    
    if not all_delta_logits:
        print("No data processed!")
        return
    
    # Combine all folds
    combined_delta_logits = np.concatenate(all_delta_logits, axis=0)
    combined_labels = np.concatenate(all_labels, axis=0)
    combined_corrected = np.concatenate(all_corrected, axis=0)
    combined_degraded = np.concatenate(all_degraded, axis=0)
    
    print(f"\n=== Combined Results ===")
    print(f"Total samples: {len(combined_delta_logits)}")
    print(f"Corrected samples: {np.sum(combined_corrected)}")
    print(f"Degraded samples: {np.sum(combined_degraded)}")
    
    # Create heatmaps
    out_prefix = Path(args.out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    
    # 1. Top-K samples by ||Δlogit||
    delta_norms = np.linalg.norm(combined_delta_logits, axis=1)
    top_k_mask = np.argsort(-delta_norms)[:args.top_k]  # Top K indices
    top_k_sample_mask = np.zeros(len(combined_delta_logits), dtype=bool)
    top_k_sample_mask[top_k_mask] = True
    
    create_logit_heatmap(
        combined_delta_logits, combined_labels, top_k_sample_mask, class_names,
        f'Top-{args.top_k} Samples by ||Δlogit||',
        f'{out_prefix}_topk.png'
    )
    
    # 2. Corrected samples (wrong → right)
    create_logit_heatmap(
        combined_delta_logits, combined_labels, combined_corrected, class_names,
        'Corrected Samples (Wrong→Right)',
        f'{out_prefix}_corrected.png'
    )
    
    # 3. Degraded samples (right → wrong) 
    create_logit_heatmap(
        combined_delta_logits, combined_labels, combined_degraded, class_names,
        'Degraded Samples (Right→Wrong)',
        f'{out_prefix}_degraded.png'
    )
    
    # 4. Summary statistics
    stats = create_summary_statistics(
        combined_delta_logits, combined_labels, combined_corrected, combined_degraded,
        f'{out_prefix}_summary.png'
    )
    
    print(f"\n=== Summary ===")
    print(f"Correction rate: {stats['corrected_samples']}/{stats['total_samples']} = {stats['corrected_samples']/stats['total_samples']:.3f}")
    print(f"Degradation rate: {stats['degraded_samples']}/{stats['total_samples']} = {stats['degraded_samples']/stats['total_samples']:.3f}")
    print(f"Mean ||Δlogit||: {stats['mean_delta_norm']:.3f} ± {stats['std_delta_norm']:.3f}")


if __name__ == "__main__":
    main()