#!/usr/bin/env python3
"""
Comprehensive Arousal-Valence Evaluation Script for MSP-Podcast
================================================================
Evaluates A-V predictions on MSP-Podcast test sets (Test1, Test2).

Uses the same metrics as IEMOCAP evaluation:
1. Classification metrics: Accuracy, F1
2. Regression metrics: CCC, MAE, Pearson
3. Clustering metrics: Silhouette, Davies-Bouldin
4. Circumplex alignment

Author: AVD_Test Evaluation
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
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    silhouette_score, 
    davies_bouldin_score,
    accuracy_score, 
    f1_score,
    confusion_matrix
)
from scipy.stats import pearsonr

# Add parent directory to path
sys.path.append(str(Path(__file__).resolve().parent))

from data.msp_podcast_dataset import MSPPodcastDataset, collate_fn as msp_collate_fn
from models.method2_crossattn_av import Method2CrossAttentionAV

# ====================================================================
# Emotion mapping (same as MSPPodcastDataset)
# ====================================================================
# Current dataset mapping: hap=0, sad=1, neu=2, ang=3
EMOTION_MAP_PLOT = {0: 'Happy', 1: 'Sad', 2: 'Neutral', 3: 'Angry'}

# OLD model mapping (if model was trained with embeddings dataset): ang=0, hap=1, neu=2, sad=3
# Conversion map: model_output -> correct_label
# If model predicts 0 (thinks Angry) -> should be 3
# If model predicts 1 (thinks Happy) -> should be 0
# If model predicts 2 (thinks Neutral) -> should be 2
# If model predicts 3 (thinks Sad) -> should be 1
LABEL_REMAP_OLD_TO_NEW = {
    0: 3,  # model's 0 (ang) -> new 3 (ang)
    1: 0,  # model's 1 (hap) -> new 0 (hap)
    2: 2,  # model's 2 (neu) -> new 2 (neu)
    3: 1   # model's 3 (sad) -> new 1 (sad)
}

# Russell's Circumplex Model - THEORETICAL positions
CIRCUMPLEX_RUSSELL = {
    'Happy':   {'arousal': 0.5,  'valence': 0.8},
    'Angry':   {'arousal': 0.8,  'valence': -0.6},
    'Sad':     {'arousal': -0.5, 'valence': -0.6},
    'Neutral': {'arousal': 0.0,  'valence': 0.0}
}

EMOTION_COLORS = {
    'Happy': '#2ecc71',
    'Sad': '#3498db',
    'Neutral': '#95a5a6',
    'Angry': '#e74c3c'
}

EMOTION_MARKERS = {
    'Happy': 'o',
    'Sad': 's',
    'Neutral': '^',
    'Angry': 'D'
}


def compute_ccc(y_true, y_pred):
    """Compute Concordance Correlation Coefficient"""
    mean_true = np.mean(y_true)
    mean_pred = np.mean(y_pred)
    var_true = np.var(y_true)
    var_pred = np.var(y_pred)
    covariance = np.mean((y_true - mean_true) * (y_pred - mean_pred))
    ccc = (2 * covariance) / (var_true + var_pred + (mean_true - mean_pred) ** 2 + 1e-8)
    return ccc


def compute_cluster_metrics(arousal, valence, labels):
    """Compute clustering quality metrics"""
    X = np.column_stack([arousal, valence])
    unique_labels = np.unique(labels)
    if len(unique_labels) < 2:
        return {'silhouette': 0.0, 'davies_bouldin': float('inf')}
    
    try:
        silhouette = silhouette_score(X, labels)
    except:
        silhouette = 0.0
    
    try:
        db_index = davies_bouldin_score(X, labels)
    except:
        db_index = float('inf')
    
    return {'silhouette': silhouette, 'davies_bouldin': db_index}


def compute_circumplex_alignment(arousal_by_emotion, valence_by_emotion):
    """Compute alignment with Russell's Circumplex Model"""
    alignment_scores = {}
    total_deviation = 0.0
    
    for emotion, expected in CIRCUMPLEX_RUSSELL.items():
        if emotion not in arousal_by_emotion:
            continue
        
        pred_a = np.mean(arousal_by_emotion[emotion])
        pred_v = np.mean(valence_by_emotion[emotion])
        exp_a = expected['arousal']
        exp_v = expected['valence']
        
        deviation = np.sqrt((pred_a - exp_a)**2 + (pred_v - exp_v)**2)
        alignment_scores[emotion] = {
            'predicted': (pred_a, pred_v),
            'expected': (exp_a, exp_v),
            'deviation': deviation
        }
        total_deviation += deviation
    
    max_deviation = 2.83 * len(alignment_scores)
    alignment_score = 1 - (total_deviation / max_deviation) if max_deviation > 0 else 0
    
    return {'overall_alignment': alignment_score, 'per_emotion': alignment_scores}


def load_checkpoint(model, ckpt_path):
    """Load model checkpoint"""
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
        state_dict = ckpt['model_state_dict']
    elif isinstance(ckpt, dict) and 'state_dict' in ckpt:
        state_dict = ckpt['state_dict']
    else:
        state_dict = ckpt
    state_dict = {k: v for k, v in state_dict.items() 
                  if not k.endswith(('total_ops', 'total_params'))}
    model.load_state_dict(state_dict, strict=False)
    return model


def evaluate_model(model, loader, device):
    """Run inference and collect predictions"""
    model.eval()
    
    all_emotion_preds = []
    all_emotion_true = []
    all_arousal_pred = []
    all_arousal_true = []
    all_valence_pred = []
    all_valence_true = []
    
    with torch.no_grad():
        for batch in tqdm(loader, desc='Inference', leave=False):
            audio = batch['audio'].to(device)
            text_ids = batch['text_input_ids'].to(device)
            text_mask = batch['text_attention_mask'].to(device)
            
            outputs = model(audio, text_ids, text_attention_mask=text_mask)
            
            preds = torch.argmax(outputs['emotion_logits'], dim=1)
            all_emotion_preds.extend(preds.cpu().numpy())
            all_emotion_true.extend(batch['emotion_label'].numpy())
            
            all_arousal_pred.extend(outputs['arousal_value'].cpu().numpy().flatten())
            all_valence_pred.extend(outputs['valence_value'].cpu().numpy().flatten())
            
            all_arousal_true.extend(batch['arousal_value'].numpy().flatten())
            all_valence_true.extend(batch['valence_value'].numpy().flatten())
    
    # CRITICAL: Remap predicted labels from old model space to new dataset space
    all_emotion_preds = np.array(all_emotion_preds)
    all_emotion_preds_remapped = np.array([LABEL_REMAP_OLD_TO_NEW[p] for p in all_emotion_preds])
    
    return {
        'emotion_pred': all_emotion_preds_remapped,
        'emotion_true': np.array(all_emotion_true),
        'arousal_pred': np.array(all_arousal_pred),
        'arousal_true': np.array(all_arousal_true),
        'valence_pred': np.array(all_valence_pred),
        'valence_true': np.array(all_valence_true)
    }


def compute_all_metrics(results):
    """Compute all evaluation metrics"""
    metrics = {}
    
    emotion_pred = results['emotion_pred']
    emotion_true = results['emotion_true']
    metrics['accuracy'] = accuracy_score(emotion_true, emotion_pred)
    metrics['f1_macro'] = f1_score(emotion_true, emotion_pred, average='macro')
    metrics['f1_weighted'] = f1_score(emotion_true, emotion_pred, average='weighted')
    
    metrics['arousal_ccc'] = compute_ccc(results['arousal_true'], results['arousal_pred'])
    metrics['valence_ccc'] = compute_ccc(results['valence_true'], results['valence_pred'])
    metrics['arousal_pearson'], _ = pearsonr(results['arousal_true'], results['arousal_pred'])
    metrics['valence_pearson'], _ = pearsonr(results['valence_true'], results['valence_pred'])
    metrics['arousal_mae'] = np.mean(np.abs(results['arousal_true'] - results['arousal_pred']))
    metrics['valence_mae'] = np.mean(np.abs(results['valence_true'] - results['valence_pred']))
    
    cluster_metrics = compute_cluster_metrics(
        results['arousal_pred'], results['valence_pred'], emotion_true
    )
    metrics['silhouette_pred'] = cluster_metrics['silhouette']
    metrics['davies_bouldin_pred'] = cluster_metrics['davies_bouldin']
    
    arousal_by_emotion = {}
    valence_by_emotion = {}
    for emo_id in sorted(EMOTION_MAP_PLOT.keys()):
        emo_name = EMOTION_MAP_PLOT[emo_id]
        mask = emotion_true == emo_id
        if mask.sum() > 0:
            arousal_by_emotion[emo_name] = results['arousal_pred'][mask]
            valence_by_emotion[emo_name] = results['valence_pred'][mask]
    
    circumplex = compute_circumplex_alignment(arousal_by_emotion, valence_by_emotion)
    metrics['circumplex_alignment'] = circumplex['overall_alignment']
    metrics['circumplex_details'] = circumplex['per_emotion']
    
    return metrics


def plot_av_comprehensive(results, model_name, split_name, output_dir, show_theory=True):
    """Create comprehensive A-V visualization"""
    os.makedirs(output_dir, exist_ok=True)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f'A-V Comprehensive Evaluation: {model_name} ({split_name})', 
                 fontsize=16, fontweight='bold')
    
    emotion_pred = results['emotion_pred']
    emotion_true = results['emotion_true']
    arousal_pred = results['arousal_pred']
    arousal_true = results['arousal_true']
    valence_pred = results['valence_pred']
    valence_true = results['valence_true']
    
    # Panel 1: Ground Truth A-V
    ax = axes[0, 0]
    for emo_id in sorted(EMOTION_MAP_PLOT.keys()):
        emo_name = EMOTION_MAP_PLOT[emo_id]
        mask = emotion_true == emo_id
        if mask.sum() > 0:
            ax.scatter(arousal_true[mask], valence_true[mask],
                      c=EMOTION_COLORS[emo_name], 
                      marker=EMOTION_MARKERS[emo_name],
                      label=f'{emo_name} (n={mask.sum()})',
                      alpha=0.4, s=30, edgecolors='white', linewidth=0.3)
    
    if show_theory:
        for emo_name, pos in CIRCUMPLEX_RUSSELL.items():
            ax.scatter(pos['arousal'], pos['valence'], 
                      marker='X', s=200, c='black', zorder=10,
                      edgecolors='yellow', linewidth=2)
    
    ax.set_xlabel('Arousal (normalized)', fontsize=11)
    ax.set_ylabel('Valence (normalized)', fontsize=11)
    ax.set_title('Ground Truth A-V\n(colored by TRUE emotion)', fontsize=12, fontweight='bold')
    ax.legend(loc='best', fontsize=8)
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
    ax.grid(True, alpha=0.3)
    
    # Panel 2: Predicted A-V (by PREDICTED emotion)
    ax = axes[0, 1]
    for emo_id in sorted(EMOTION_MAP_PLOT.keys()):
        emo_name = EMOTION_MAP_PLOT[emo_id]
        mask = emotion_pred == emo_id
        if mask.sum() > 0:
            ax.scatter(arousal_pred[mask], valence_pred[mask],
                      c=EMOTION_COLORS[emo_name],
                      marker=EMOTION_MARKERS[emo_name],
                      label=f'{emo_name} (n={mask.sum()})',
                      alpha=0.4, s=30, edgecolors='white', linewidth=0.3)
    
    if show_theory:
        for emo_name, pos in CIRCUMPLEX_RUSSELL.items():
            ax.scatter(pos['arousal'], pos['valence'], 
                      marker='X', s=200, c='black', zorder=10,
                      edgecolors='yellow', linewidth=2)
    
    ax.set_xlabel('Arousal (predicted)', fontsize=11)
    ax.set_ylabel('Valence (predicted)', fontsize=11)
    ax.set_title('Predicted A-V\n(colored by PREDICTED emotion)', fontsize=12, fontweight='bold')
    ax.legend(loc='best', fontsize=8)
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
    ax.grid(True, alpha=0.3)
    
    # Panel 3: Predicted A-V (by TRUE emotion)
    ax = axes[0, 2]
    for emo_id in sorted(EMOTION_MAP_PLOT.keys()):
        emo_name = EMOTION_MAP_PLOT[emo_id]
        mask = emotion_true == emo_id
        if mask.sum() > 0:
            ax.scatter(arousal_pred[mask], valence_pred[mask],
                      c=EMOTION_COLORS[emo_name],
                      marker=EMOTION_MARKERS[emo_name],
                      label=f'{emo_name} (n={mask.sum()})',
                      alpha=0.4, s=30, edgecolors='white', linewidth=0.3)
    
    if show_theory:
        for emo_name, pos in CIRCUMPLEX_RUSSELL.items():
            ax.scatter(pos['arousal'], pos['valence'], 
                      marker='X', s=200, c='black', zorder=10,
                      edgecolors='yellow', linewidth=2)
    
    ax.set_xlabel('Arousal (predicted)', fontsize=11)
    ax.set_ylabel('Valence (predicted)', fontsize=11)
    ax.set_title('Predicted A-V\n(colored by TRUE emotion)', fontsize=12, fontweight='bold')
    ax.legend(loc='best', fontsize=8)
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
    ax.grid(True, alpha=0.3)
    
    # Panel 4: Arousal Scatter
    ax = axes[1, 0]
    ax.scatter(arousal_true, arousal_pred, alpha=0.3, s=15, c='steelblue')
    ax.plot([-1, 1], [-1, 1], 'r--', linewidth=2, label='Perfect')
    ccc_a = compute_ccc(arousal_true, arousal_pred)
    r_a, _ = pearsonr(arousal_true, arousal_pred)
    ax.set_xlabel('Arousal (true)', fontsize=11)
    ax.set_ylabel('Arousal (predicted)', fontsize=11)
    ax.set_title(f'Arousal Regression\nCCC={ccc_a:.3f}, r={r_a:.3f}', fontsize=12, fontweight='bold')
    ax.legend(loc='best')
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    ax.grid(True, alpha=0.3)
    
    # Panel 5: Valence Scatter
    ax = axes[1, 1]
    ax.scatter(valence_true, valence_pred, alpha=0.3, s=15, c='coral')
    ax.plot([-1, 1], [-1, 1], 'r--', linewidth=2, label='Perfect')
    ccc_v = compute_ccc(valence_true, valence_pred)
    r_v, _ = pearsonr(valence_true, valence_pred)
    ax.set_xlabel('Valence (true)', fontsize=11)
    ax.set_ylabel('Valence (predicted)', fontsize=11)
    ax.set_title(f'Valence Regression\nCCC={ccc_v:.3f}, r={r_v:.3f}', fontsize=12, fontweight='bold')
    ax.legend(loc='best')
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    ax.grid(True, alpha=0.3)
    
    # Panel 6: Confusion Matrix
    ax = axes[1, 2]
    cm = confusion_matrix(emotion_true, emotion_pred)
    emotion_names = [EMOTION_MAP_PLOT[i] for i in range(4)]
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=emotion_names, yticklabels=emotion_names)
    ax.set_xlabel('Predicted Emotion', fontsize=11)
    ax.set_ylabel('True Emotion', fontsize=11)
    acc = accuracy_score(emotion_true, emotion_pred)
    f1 = f1_score(emotion_true, emotion_pred, average='macro')
    ax.set_title(f'Confusion Matrix\nAcc={acc:.3f}, F1={f1:.3f}', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    save_path = os.path.join(output_dir, f'av_comprehensive_{model_name}_{split_name}.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Saved: {save_path}")
    return save_path


def print_metrics_report(metrics, model_name, split_name):
    """Print formatted metrics report"""
    print(f"\n{'='*70}")
    print(f" METRICS REPORT: {model_name} ({split_name})")
    print(f"{'='*70}")
    
    print("\n📊 CLASSIFICATION METRICS")
    print(f"  Accuracy:      {metrics['accuracy']:.4f}")
    print(f"  F1 (macro):    {metrics['f1_macro']:.4f}")
    print(f"  F1 (weighted): {metrics['f1_weighted']:.4f}")
    
    print("\n📈 REGRESSION METRICS (A-V)")
    print(f"  Arousal CCC:     {metrics['arousal_ccc']:.4f}")
    print(f"  Valence CCC:     {metrics['valence_ccc']:.4f}")
    print(f"  Arousal Pearson: {metrics['arousal_pearson']:.4f}")
    print(f"  Valence Pearson: {metrics['valence_pearson']:.4f}")
    print(f"  Arousal MAE:     {metrics['arousal_mae']:.4f}")
    print(f"  Valence MAE:     {metrics['valence_mae']:.4f}")
    
    print("\n🔮 CLUSTERING METRICS (on predicted A-V)")
    print(f"  Silhouette Score:     {metrics['silhouette_pred']:.4f}")
    print(f"  Davies-Bouldin Index: {metrics['davies_bouldin_pred']:.4f}")
    
    print("\n🎯 CIRCUMPLEX ALIGNMENT")
    print(f"  Overall Alignment: {metrics['circumplex_alignment']:.4f}")
    print("  Per-emotion deviation from theory:")
    for emo_name, details in metrics['circumplex_details'].items():
        pred = details['predicted']
        exp = details['expected']
        dev = details['deviation']
        print(f"    {emo_name:8s}: pred=(A:{pred[0]:+.2f}, V:{pred[1]:+.2f}), "
              f"theory=(A:{exp[0]:+.2f}, V:{exp[1]:+.2f}), deviation={dev:.3f}")
    
    print(f"\n{'='*70}\n")


def main():
    parser = argparse.ArgumentParser(description='Comprehensive A-V Evaluation for MSP-Podcast')
    parser.add_argument('--model', type=str, default='method2_msp_wpro',
                        help='Model checkpoint directory name')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to specific checkpoint file (overrides --model)')
    parser.add_argument('--tsv', type=str, default='datasets/MSP-Podcast/msp_podcast.tsv',
                        help='Path to MSP-Podcast TSV file')
    parser.add_argument('--splits', type=str, nargs='+', default=['test1', 'test2'],
                        help='Splits to evaluate')
    parser.add_argument('--output_dir', type=str, default='outputs/av_comprehensive',
                        help='Output directory')
    parser.add_argument('--use_wpro', action='store_true', default=None,
                        help='Use W-PRO model (auto-detected from model name)')
    parser.add_argument('--no_wpro', action='store_true', default=False,
                        help='Force disable W-PRO')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='Device to use')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size')
    args = parser.parse_args()
    
    # Auto-detect W-PRO
    if args.no_wpro:
        use_wpro = False
    elif args.use_wpro is not None:
        use_wpro = args.use_wpro
    else:
        use_wpro = 'wpro' in args.model.lower()
    
    print(f"W-PRO mode: {use_wpro} (auto-detected from model name: '{args.model}')")
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Checkpoint path - use --checkpoint if provided, otherwise use default path
    if args.checkpoint:
        ckpt_path = Path(args.checkpoint)
    else:
        ckpt_dir = Path('checkpoints') / args.model
        ckpt_path = ckpt_dir / 'best_model.pt'
    
    if not ckpt_path.exists():
        print(f"Error: Checkpoint not found: {ckpt_path}")
        return
    
    # Output directory - use checkpoint name if custom checkpoint provided
    if args.checkpoint:
        ckpt_name = Path(args.checkpoint).stem
        output_dir = Path(args.output_dir) / f"{args.model}_{ckpt_name}"
    else:
        output_dir = Path(args.output_dir) / args.model
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n{'#'*80}")
    print(f" COMPREHENSIVE A-V EVALUATION (MSP-Podcast): {args.model}")
    print(f"{'#'*80}")
    
    # Load model once
    print(f"\nLoading model from: {ckpt_path}")
    print(f"Using W-PRO: {use_wpro}")
    model = Method2CrossAttentionAV(
        num_emotions=4,
        wavlm_model_name="microsoft/wavlm-base-plus",
        wavlm_prosody_model_name="Orange/Speaker-wavLM-pro" if use_wpro else None,
        roberta_model_name="roberta-base",
        hidden_dim=256,
        freeze_wavlm=True,
        freeze_roberta=True
    ).to(device)
    model = load_checkpoint(model, ckpt_path)
    model.eval()
    
    all_results = []
    all_metrics = []
    
    for split in args.splits:
        print(f"\n[{split.upper()}]")
        
        # Load dataset
        dataset = MSPPodcastDataset(
            args.tsv,
            split=split,
            audio_max_length=16000 * 10,
            sample_rate=16000,
            normalize_av=True,
            four_class_only=True
        )
        
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=2,
            collate_fn=msp_collate_fn
        )
        
        print(f"  Samples: {len(dataset)}")
        
        # Evaluate
        results = evaluate_model(model, loader, device)
        all_results.append(results)
        
        # Compute metrics
        metrics = compute_all_metrics(results)
        all_metrics.append(metrics)
        
        # Print report
        print_metrics_report(metrics, args.model, split)
        
        # Plot
        plot_av_comprehensive(results, args.model, split, str(output_dir))
    
    # Save summary
    if len(all_metrics) > 0:
        metric_keys = ['accuracy', 'f1_macro', 'f1_weighted', 
                       'arousal_ccc', 'valence_ccc',
                       'arousal_pearson', 'valence_pearson',
                       'arousal_mae', 'valence_mae',
                       'silhouette_pred', 'davies_bouldin_pred',
                       'circumplex_alignment']
        
        summary_data = {'metric': [], **{s: [] for s in args.splits}}
        for key in metric_keys:
            summary_data['metric'].append(key)
            for i, split in enumerate(args.splits):
                summary_data[split].append(all_metrics[i].get(key, np.nan))
        
        summary_path = output_dir / 'summary_metrics.csv'
        pd.DataFrame(summary_data).to_csv(summary_path, index=False)
        print(f"\n✓ Summary saved to: {summary_path}")
    
    # Clean up
    del model
    torch.cuda.empty_cache()
    
    print("\n✅ Evaluation complete!")


if __name__ == '__main__':
    main()
