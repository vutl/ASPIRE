#!/usr/bin/env python3
"""
Comprehensive Arousal-Valence Evaluation Script
==============================================
Evaluates A-V predictions with multiple metrics:
1. Visual metrics: Separability, Compactness, Overlap visualization
2. Clustering metrics: Silhouette Score, Davies-Bouldin Index
3. Regression metrics: CCC, MAE, Pearson correlation  
4. Russell's Circumplex Model alignment
5. Classification metrics per emotion

CRITICAL FIX: Uses CORRECT emotion mapping!
- Dataset: {'hap': 0, 'sad': 1, 'neu': 2, 'ang': 3}
- Plots: {0: 'Happy', 1: 'Sad', 2: 'Neutral', 3: 'Angry'}

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
import matplotlib.patches as mpatches
from matplotlib.patches import Ellipse
import seaborn as sns

from sklearn.metrics import (
    silhouette_score, 
    davies_bouldin_score,
    accuracy_score, 
    f1_score,
    confusion_matrix,
    classification_report
)
from scipy.stats import pearsonr
from scipy.spatial import ConvexHull

# Add parent directory to path
sys.path.append(str(Path(__file__).resolve().parent))

from data.iemocap_dataset import IEMOCAPDataset, collate_fn as iemocap_collate_fn
from models.method2_crossattn_av import Method2CrossAttentionAV
from models.method2_crossattn_av_ablation_no_eav import Method2CrossAttentionAV_AblationNoEAV

# ====================================================================
# CRITICAL: CORRECT EMOTION MAPPING (matching IEMOCAPDataset code!)
# NOTE: IEMOCAPDataset uses its OWN mapping, NOT the emotion_id from CSV!
# CSV has: {'hap': 0, 'neu': 1, 'sad': 2, 'ang': 3}
# But IEMOCAPDataset uses: {'hap': 0, 'sad': 1, 'neu': 2, 'ang': 3}
# ====================================================================
EMOTION_MAP_TRAIN = {'hap': 0, 'sad': 1, 'neu': 2, 'ang': 3}  # From IEMOCAPDataset code
EMOTION_MAP_PLOT = {0: 'Happy', 1: 'Sad', 2: 'Neutral', 3: 'Angry'}  # For plots - MATCHES dataset code

# ====================================================================
# Russell's Circumplex Model - THEORETICAL positions (normalized to [-1, 1])
# Reference: Russell, J.A. (1980). A circumplex model of affect.
# These are the STANDARD theoretical expectations for each emotion quadrant.
# ====================================================================
CIRCUMPLEX_RUSSELL = {
    'Happy':   {'arousal': 0.5,  'valence': 0.8},   # High A, High V (Q1: excited/happy)
    'Angry':   {'arousal': 0.8,  'valence': -0.6},  # High A, Low V  (Q2: angry/tense)
    'Sad':     {'arousal': -0.5, 'valence': -0.6},  # Low A, Low V   (Q3: sad/depressed)
    'Neutral': {'arousal': 0.0,  'valence': 0.0}    # Center         (neutral)
}

# IEMOCAP Ground Truth positions (empirical from annotations)
# NOTE: These deviate from Russell's theory - this is a dataset characteristic
CIRCUMPLEX_IEMOCAP_GT = {
    'Happy':   {'arousal': 0.06,  'valence': 0.48},   # IEMOCAP: Lower arousal than theory
    'Angry':   {'arousal': 0.32,  'valence': -0.55},  # IEMOCAP: Lower arousal than theory
    'Sad':     {'arousal': -0.22, 'valence': -0.37},  # IEMOCAP: Moderate values
    'Neutral': {'arousal': -0.14, 'valence': -0.01}   # IEMOCAP: Close to theory
}

# DEFAULT: Use Russell's theoretical positions for paper reporting
# This allows fair comparison across different datasets
CIRCUMPLEX_THEORY = CIRCUMPLEX_RUSSELL

EMOTION_COLORS = {
    'Happy': '#2ecc71',     # Green
    'Sad': '#3498db',       # Blue  
    'Neutral': '#95a5a6',   # Gray
    'Angry': '#e74c3c'      # Red
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
    """
    Compute clustering quality metrics for A-V space
    
    Args:
        arousal: array of arousal values
        valence: array of valence values
        labels: array of emotion labels (0-3)
    
    Returns:
        dict with silhouette_score, davies_bouldin_index
    """
    X = np.column_stack([arousal, valence])
    
    # Need at least 2 unique labels for clustering metrics
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
    
    return {
        'silhouette': silhouette,
        'davies_bouldin': db_index
    }


def compute_circumplex_alignment(arousal_by_emotion, valence_by_emotion):
    """
    Compute alignment with Russell's Circumplex Model
    
    Returns:
        dict with alignment_score (0-1), per-emotion deviations
    """
    alignment_scores = {}
    total_deviation = 0.0
    
    for emotion, expected in CIRCUMPLEX_THEORY.items():
        if emotion not in arousal_by_emotion:
            continue
        
        pred_a = np.mean(arousal_by_emotion[emotion])
        pred_v = np.mean(valence_by_emotion[emotion])
        exp_a = expected['arousal']
        exp_v = expected['valence']
        
        # Euclidean distance in A-V space
        deviation = np.sqrt((pred_a - exp_a)**2 + (pred_v - exp_v)**2)
        alignment_scores[emotion] = {
            'predicted': (pred_a, pred_v),
            'expected': (exp_a, exp_v),
            'deviation': deviation
        }
        total_deviation += deviation
    
    # Normalize to [0, 1] where 1 is perfect alignment
    # Max possible deviation is sqrt(8) ≈ 2.83 (diagonal of [-1,1]x[-1,1])
    max_deviation = 2.83 * len(alignment_scores)
    alignment_score = 1 - (total_deviation / max_deviation) if max_deviation > 0 else 0
    
    return {
        'overall_alignment': alignment_score,
        'per_emotion': alignment_scores
    }


def compute_overlap_area(arousal_by_emotion, valence_by_emotion):
    """
    Compute approximate overlap between emotion clusters using convex hulls
    """
    try:
        hulls = {}
        for emotion in arousal_by_emotion:
            points = np.column_stack([
                arousal_by_emotion[emotion],
                valence_by_emotion[emotion]
            ])
            if len(points) >= 3:
                try:
                    hulls[emotion] = ConvexHull(points)
                except:
                    pass
        
        # Compute pairwise overlap (simplified)
        total_area = sum(h.volume for h in hulls.values()) if hulls else 0
        return {'total_hull_area': total_area, 'num_valid_hulls': len(hulls)}
    except:
        return {'total_hull_area': 0, 'num_valid_hulls': 0}


def load_checkpoint(model, ckpt_path):
    """Load model checkpoint"""
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
        state_dict = ckpt['model_state_dict']
    elif isinstance(ckpt, dict) and 'state_dict' in ckpt:
        state_dict = ckpt['state_dict']
    else:
        state_dict = ckpt
    # Remove profiling keys
    state_dict = {k: v for k, v in state_dict.items() 
                  if not k.endswith(('total_ops', 'total_params'))}
    model.load_state_dict(state_dict, strict=False)
    return model


def evaluate_model(model, loader, device, disable_eav=False):
    """
    Run inference and collect predictions
    
    Returns:
        dict with predictions and ground truth
    """
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
            
            outputs = model(
                audio,
                text_ids,
                text_attention_mask=text_mask,
                use_guidance=not disable_eav,
            )
            
            # Emotion predictions
            preds = torch.argmax(outputs['emotion_logits'], dim=1)
            all_emotion_preds.extend(preds.cpu().numpy())
            all_emotion_true.extend(batch['emotion_label'].numpy())
            
            # A-V predictions (already in [-1, 1] from model)
            all_arousal_pred.extend(outputs['arousal_value'].cpu().numpy().flatten())
            all_valence_pred.extend(outputs['valence_value'].cpu().numpy().flatten())
            
            # A-V ground truth (already normalized in dataset)
            all_arousal_true.extend(batch['arousal_value'].numpy().flatten())
            all_valence_true.extend(batch['valence_value'].numpy().flatten())
    
    return {
        'emotion_pred': np.array(all_emotion_preds),
        'emotion_true': np.array(all_emotion_true),
        'arousal_pred': np.array(all_arousal_pred),
        'arousal_true': np.array(all_arousal_true),
        'valence_pred': np.array(all_valence_pred),
        'valence_true': np.array(all_valence_true)
    }


def plot_av_comprehensive(results, model_name, fold_info, output_dir, show_theory=True):
    """
    Create comprehensive A-V visualization with multiple panels
    """
    os.makedirs(output_dir, exist_ok=True)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f'A-V Comprehensive Evaluation: {model_name} ({fold_info})', 
                 fontsize=16, fontweight='bold')
    
    emotion_pred = results['emotion_pred']
    emotion_true = results['emotion_true']
    arousal_pred = results['arousal_pred']
    arousal_true = results['arousal_true']
    valence_pred = results['valence_pred']
    valence_true = results['valence_true']
    
    # =========== Panel 1: Ground Truth A-V (colored by TRUE emotion) ===========
    ax = axes[0, 0]
    for emo_id in sorted(EMOTION_MAP_PLOT.keys()):
        emo_name = EMOTION_MAP_PLOT[emo_id]
        mask = emotion_true == emo_id
        if mask.sum() > 0:
            ax.scatter(arousal_true[mask], valence_true[mask],
                      c=EMOTION_COLORS[emo_name], 
                      marker=EMOTION_MARKERS[emo_name],
                      label=f'{emo_name} (n={mask.sum()})',
                      alpha=0.6, s=50, edgecolors='white', linewidth=0.5)
    
    if show_theory:
        for emo_name, pos in CIRCUMPLEX_THEORY.items():
            ax.scatter(pos['arousal'], pos['valence'], 
                      marker='X', s=200, c='black', zorder=10,
                      edgecolors='yellow', linewidth=2)
            ax.annotate(f'{emo_name}\n(theory)', 
                       (pos['arousal'], pos['valence']),
                       fontsize=8, ha='center', va='bottom')
    
    ax.set_xlabel('Arousal (normalized)', fontsize=11)
    ax.set_ylabel('Valence (normalized)', fontsize=11)
    ax.set_title('Ground Truth A-V\n(colored by TRUE emotion)', fontsize=12, fontweight='bold')
    ax.legend(loc='best', fontsize=9)
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
    ax.grid(True, alpha=0.3)
    
    # =========== Panel 2: Predicted A-V (colored by PREDICTED emotion) ===========
    ax = axes[0, 1]
    for emo_id in sorted(EMOTION_MAP_PLOT.keys()):
        emo_name = EMOTION_MAP_PLOT[emo_id]
        mask = emotion_pred == emo_id
        if mask.sum() > 0:
            ax.scatter(arousal_pred[mask], valence_pred[mask],
                      c=EMOTION_COLORS[emo_name],
                      marker=EMOTION_MARKERS[emo_name],
                      label=f'{emo_name} (n={mask.sum()})',
                      alpha=0.6, s=50, edgecolors='white', linewidth=0.5)
    
    if show_theory:
        for emo_name, pos in CIRCUMPLEX_THEORY.items():
            ax.scatter(pos['arousal'], pos['valence'], 
                      marker='X', s=200, c='black', zorder=10,
                      edgecolors='yellow', linewidth=2)
    
    ax.set_xlabel('Arousal (predicted)', fontsize=11)
    ax.set_ylabel('Valence (predicted)', fontsize=11)
    ax.set_title('Predicted A-V\n(colored by PREDICTED emotion)', fontsize=12, fontweight='bold')
    ax.legend(loc='best', fontsize=9)
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
    ax.grid(True, alpha=0.3)
    
    # =========== Panel 3: Predicted A-V (colored by TRUE emotion) ===========
    ax = axes[0, 2]
    for emo_id in sorted(EMOTION_MAP_PLOT.keys()):
        emo_name = EMOTION_MAP_PLOT[emo_id]
        mask = emotion_true == emo_id
        if mask.sum() > 0:
            ax.scatter(arousal_pred[mask], valence_pred[mask],
                      c=EMOTION_COLORS[emo_name],
                      marker=EMOTION_MARKERS[emo_name],
                      label=f'{emo_name} (n={mask.sum()})',
                      alpha=0.6, s=50, edgecolors='white', linewidth=0.5)
    
    if show_theory:
        for emo_name, pos in CIRCUMPLEX_THEORY.items():
            ax.scatter(pos['arousal'], pos['valence'], 
                      marker='X', s=200, c='black', zorder=10,
                      edgecolors='yellow', linewidth=2)
    
    ax.set_xlabel('Arousal (predicted)', fontsize=11)
    ax.set_ylabel('Valence (predicted)', fontsize=11)
    ax.set_title('Predicted A-V\n(colored by TRUE emotion)', fontsize=12, fontweight='bold')
    ax.legend(loc='best', fontsize=9)
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
    ax.grid(True, alpha=0.3)
    
    # =========== Panel 4: Arousal Scatter (pred vs true) ===========
    ax = axes[1, 0]
    ax.scatter(arousal_true, arousal_pred, alpha=0.4, s=20, c='steelblue')
    ax.plot([-1, 1], [-1, 1], 'r--', linewidth=2, label='Perfect prediction')
    ccc_a = compute_ccc(arousal_true, arousal_pred)
    r_a, _ = pearsonr(arousal_true, arousal_pred)
    ax.set_xlabel('Arousal (true)', fontsize=11)
    ax.set_ylabel('Arousal (predicted)', fontsize=11)
    ax.set_title(f'Arousal Regression\nCCC={ccc_a:.3f}, r={r_a:.3f}', fontsize=12, fontweight='bold')
    ax.legend(loc='best')
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    ax.grid(True, alpha=0.3)
    
    # =========== Panel 5: Valence Scatter (pred vs true) ===========
    ax = axes[1, 1]
    ax.scatter(valence_true, valence_pred, alpha=0.4, s=20, c='coral')
    ax.plot([-1, 1], [-1, 1], 'r--', linewidth=2, label='Perfect prediction')
    ccc_v = compute_ccc(valence_true, valence_pred)
    r_v, _ = pearsonr(valence_true, valence_pred)
    ax.set_xlabel('Valence (true)', fontsize=11)
    ax.set_ylabel('Valence (predicted)', fontsize=11)
    ax.set_title(f'Valence Regression\nCCC={ccc_v:.3f}, r={r_v:.3f}', fontsize=12, fontweight='bold')
    ax.legend(loc='best')
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    ax.grid(True, alpha=0.3)
    
    # =========== Panel 6: Confusion Matrix ===========
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
    save_path = os.path.join(output_dir, f'av_comprehensive_{model_name}_{fold_info}.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Saved: {save_path}")
    return save_path


def compute_all_metrics(results):
    """
    Compute all evaluation metrics
    """
    metrics = {}
    
    # Classification metrics
    emotion_pred = results['emotion_pred']
    emotion_true = results['emotion_true']
    metrics['accuracy'] = accuracy_score(emotion_true, emotion_pred)
    metrics['f1_macro'] = f1_score(emotion_true, emotion_pred, average='macro')
    metrics['f1_weighted'] = f1_score(emotion_true, emotion_pred, average='weighted')
    
    # Regression metrics
    metrics['arousal_ccc'] = compute_ccc(results['arousal_true'], results['arousal_pred'])
    metrics['valence_ccc'] = compute_ccc(results['valence_true'], results['valence_pred'])
    metrics['arousal_pearson'], _ = pearsonr(results['arousal_true'], results['arousal_pred'])
    metrics['valence_pearson'], _ = pearsonr(results['valence_true'], results['valence_pred'])
    metrics['arousal_mae'] = np.mean(np.abs(results['arousal_true'] - results['arousal_pred']))
    metrics['valence_mae'] = np.mean(np.abs(results['valence_true'] - results['valence_pred']))
    
    # Clustering metrics on predicted A-V space
    cluster_metrics = compute_cluster_metrics(
        results['arousal_pred'], 
        results['valence_pred'], 
        emotion_true  # Use true labels for cluster evaluation
    )
    metrics['silhouette_pred'] = cluster_metrics['silhouette']
    metrics['davies_bouldin_pred'] = cluster_metrics['davies_bouldin']
    
    # Circumplex alignment
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


def print_metrics_report(metrics, model_name, fold_info):
    """
    Print formatted metrics report
    """
    print(f"\n{'='*70}")
    print(f" METRICS REPORT: {model_name} ({fold_info})")
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
    print(f"  Silhouette Score:     {metrics['silhouette_pred']:.4f}  (range: -1 to 1, higher=better)")
    print(f"  Davies-Bouldin Index: {metrics['davies_bouldin_pred']:.4f}  (lower=better)")
    
    print("\n🎯 CIRCUMPLEX ALIGNMENT")
    print(f"  Overall Alignment: {metrics['circumplex_alignment']:.4f}  (0-1, 1=perfect)")
    print("  Per-emotion deviation from theory:")
    for emo_name, details in metrics['circumplex_details'].items():
        pred = details['predicted']
        exp = details['expected']
        dev = details['deviation']
        print(f"    {emo_name:8s}: pred=(A:{pred[0]:+.2f}, V:{pred[1]:+.2f}), "
              f"theory=(A:{exp[0]:+.2f}, V:{exp[1]:+.2f}), deviation={dev:.3f}")
    
    print(f"\n{'='*70}\n")


def get_test_df(csv_path, fold_id):
    """Get test dataframe for a specific fold"""
    df = pd.read_csv(csv_path)
    max_fold = df['fold'].max()
    
    if max_fold == 10:
        # 10-fold: map to 5 sessions
        fold_to_session = {1:1, 2:1, 3:2, 4:2, 5:3, 6:3, 7:4, 8:4, 9:5, 10:5}
        df['session'] = df['fold'].map(fold_to_session)
        test_df = df[df['session'] == fold_id].reset_index(drop=True)
    else:
        test_df = df[df['fold'] == fold_id].reset_index(drop=True)
    
    return test_df


def main():
    parser = argparse.ArgumentParser(description='Comprehensive A-V Evaluation')
    parser.add_argument('--model', type=str, default='method2_kfold_wpro',
                        help='Model checkpoint directory name')
    parser.add_argument('--csv', type=str, default='datasets/IEMOCAP_KFOLD/tenfolds/all.csv',
                        help='Path to dataset CSV (10-fold format)')
    parser.add_argument('--folds', type=int, nargs='+', default=[1, 2, 3, 4, 5],
                        help='Folds to evaluate')
    parser.add_argument('--output_dir', type=str, default='outputs/av_comprehensive',
                        help='Output directory for plots and metrics')
    parser.add_argument('--use_wpro', action='store_true', default=None,
                        help='Use W-PRO model for prosody (auto-detected from model name if not specified)')
    parser.add_argument('--no_wpro', action='store_true', default=False,
                        help='Force disable W-PRO (use standard WavLM)')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='Device to use')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size for inference')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Custom checkpoint path (overrides default path for single fold eval)')
    parser.add_argument('--disable_eav', action='store_true',
                        help='Disable AV_refined in final fusion during inference (use_guidance=False)')
    args = parser.parse_args()
    
    # Auto-detect W-PRO usage from model name
    if args.no_wpro:
        use_wpro = False
    elif args.use_wpro is not None:
        use_wpro = args.use_wpro
    else:
        # Auto-detect: if model name contains 'wpro', use W-PRO
        use_wpro = 'wpro' in args.model.lower()
    
    # Auto-detect ablation model
    use_ablation_no_eav = 'ablation_no_eav' in args.model.lower()
    
    print(f"W-PRO mode: {use_wpro} (auto-detected from model name: '{args.model}')")
    print(f"Ablation no-eAV: {use_ablation_no_eav}")
    print(f"Disable eAV at inference: {args.disable_eav}")
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Checkpoint directory
    ckpt_dir = Path('checkpoints') / args.model
    if not ckpt_dir.exists():
        print(f"Error: Checkpoint directory not found: {ckpt_dir}")
        return
    
    # Output directory
    output_dir = Path(args.output_dir) / args.model
    os.makedirs(output_dir, exist_ok=True)
    
    # Store all fold results
    all_results = []
    all_metrics = []
    
    print(f"\n{'#'*80}")
    print(f" COMPREHENSIVE A-V EVALUATION: {args.model}")
    print(f"{'#'*80}")
    
    for fold in args.folds:
        print(f"\n[Fold {fold}]")
        
        # Find checkpoint - use custom path if provided, otherwise default
        if args.checkpoint:
            ckpt_path = Path(args.checkpoint)
        else:
            ckpt_path = ckpt_dir / f'fold_{fold}' / 'best_model.pt'
        
        if not ckpt_path.exists():
            print(f"  ✗ Checkpoint not found: {ckpt_path}")
            continue
        
        # Load model with EXACT same parameters as training config (method2_config.yaml)
        print(f"  Loading model from: {ckpt_path}")
        print(f"  Using W-PRO: {use_wpro}")
        print(f"  Using ablation (no-eAV): {use_ablation_no_eav}")
        
        # Choose model class based on ablation flag
        ModelClass = Method2CrossAttentionAV_AblationNoEAV if use_ablation_no_eav else Method2CrossAttentionAV
        
        model = ModelClass(
            num_emotions=4,
            wavlm_model_name="microsoft/wavlm-base-plus",
            wavlm_prosody_model_name="Orange/Speaker-wavLM-pro" if use_wpro else None,
            roberta_model_name="roberta-base",
            hidden_dim=256,
            av_refinement_layers=2,
            freeze_wavlm=True,
            freeze_roberta=True,
            num_cross_attn_heads=8,
            post_attention_layers=2,
            av_prediction_layers=3
        ).to(device)
        model = load_checkpoint(model, ckpt_path)
        model.eval()
        
        # Load test data
        test_df = get_test_df(args.csv, fold)
        print(f"  Test samples: {len(test_df)}")
        
        dataset = IEMOCAPDataset(
            test_df,
            audio_max_length=16000 * 10,
            sample_rate=16000,
            normalize_av=True
        )
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=2,
            collate_fn=iemocap_collate_fn
        )
        
        # Evaluate
        results = evaluate_model(model, loader, device, disable_eav=args.disable_eav)
        all_results.append(results)
        
        # Compute metrics
        metrics = compute_all_metrics(results)
        all_metrics.append(metrics)
        
        # Print report
        print_metrics_report(metrics, args.model, f'Fold {fold}')
        
        # Plot
        plot_av_comprehensive(results, args.model, f'fold_{fold}', str(output_dir))
        
        # Clean up
        del model
        torch.cuda.empty_cache()
    
    # =========== Aggregate Results ===========
    if len(all_metrics) > 0:
        print(f"\n{'#'*80}")
        print(f" AGGREGATED RESULTS ACROSS {len(all_metrics)} FOLDS")
        print(f"{'#'*80}")
        
        # Average metrics
        avg_metrics = {}
        metric_keys = ['accuracy', 'f1_macro', 'f1_weighted', 
                       'arousal_ccc', 'valence_ccc',
                       'arousal_pearson', 'valence_pearson',
                       'arousal_mae', 'valence_mae',
                       'silhouette_pred', 'davies_bouldin_pred',
                       'circumplex_alignment']
        
        for key in metric_keys:
            values = [m[key] for m in all_metrics if key in m]
            if values:
                avg_metrics[key] = {
                    'mean': np.mean(values),
                    'std': np.std(values)
                }
        
        print("\n📊 AVERAGE METRICS (mean ± std)")
        print("-" * 50)
        for key, stats in avg_metrics.items():
            print(f"  {key:25s}: {stats['mean']:.4f} ± {stats['std']:.4f}")
        
        # Save summary to CSV
        summary_path = output_dir / 'summary_metrics.csv'
        summary_data = {
            'metric': [],
            'mean': [],
            'std': [],
            **{f'fold_{i+1}': [] for i in range(len(all_metrics))}
        }
        
        for key in metric_keys:
            summary_data['metric'].append(key)
            values = [m.get(key, np.nan) for m in all_metrics]
            summary_data['mean'].append(np.nanmean(values))
            summary_data['std'].append(np.nanstd(values))
            for i, v in enumerate(values):
                summary_data[f'fold_{i+1}'].append(v)
        
        pd.DataFrame(summary_data).to_csv(summary_path, index=False)
        print(f"\n✓ Summary saved to: {summary_path}")
        
        # Combined plot (all folds)
        if len(all_results) > 1:
            combined = {
                'emotion_pred': np.concatenate([r['emotion_pred'] for r in all_results]),
                'emotion_true': np.concatenate([r['emotion_true'] for r in all_results]),
                'arousal_pred': np.concatenate([r['arousal_pred'] for r in all_results]),
                'arousal_true': np.concatenate([r['arousal_true'] for r in all_results]),
                'valence_pred': np.concatenate([r['valence_pred'] for r in all_results]),
                'valence_true': np.concatenate([r['valence_true'] for r in all_results]),
            }
            plot_av_comprehensive(combined, args.model, 'all_folds', str(output_dir))
    
    print("\n✅ Evaluation complete!")


if __name__ == '__main__':
    main()
