#!/usr/bin/env python3
"""
IEMOCAP Method2 Inference & Evaluation (5-Fold, No W-PRO)
===========================================================
Evaluates checkpoints/method2_kfold/fold_*/best_model.pt on IEMOCAP test sets.
Computes: accuracy, F1 (macro/weighted), arousal/valence CCC, Pearson correlation, MAE.

Architecture: WavLM (standard) + RoBERTa cross-attention (no W-PRO prosody)
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
from sklearn.metrics import (
    accuracy_score, f1_score, confusion_matrix, classification_report,
    silhouette_score, davies_bouldin_score
)
from scipy.stats import pearsonr

# Local imports
sys.path.append(str(Path(__file__).resolve().parent.parent))
from data.iemocap_dataset import IEMOCAPDataset, collate_fn as iemocap_collate_fn
from models.method2_crossattn_av import Method2CrossAttentionAV


# IEMOCAP emotion mapping (must match dataset code)
EMOTION_MAP = {0: 'Happy', 1: 'Sad', 2: 'Neutral', 3: 'Angry'}

# Russell's Circumplex Model - theoretical positions
CIRCUMPLEX_RUSSELL = {
    'Happy':   {'arousal': 0.5,  'valence': 0.8},
    'Angry':   {'arousal': 0.8,  'valence': -0.6},
    'Sad':     {'arousal': -0.5, 'valence': -0.6},
    'Neutral': {'arousal': 0.0,  'valence': 0.0}
}


def compute_ccc(y_true, y_pred):
    """Compute Concordance Correlation Coefficient"""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_true.size == 0 or y_pred.size == 0:
        return np.nan
    mean_true = np.mean(y_true)
    mean_pred = np.mean(y_pred)
    var_true = np.var(y_true)
    var_pred = np.var(y_pred)
    covariance = np.mean((y_true - mean_true) * (y_pred - mean_pred))
    ccc = (2 * covariance) / (var_true + var_pred + (mean_true - mean_pred) ** 2 + 1e-8)
    return float(ccc)


def compute_cluster_metrics(arousal, valence, labels):
    """Compute clustering quality metrics for A-V space"""
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


def get_test_df(csv_path, fold_id):
    """Get test dataframe for a specific fold (5-session split)"""
    df = pd.read_csv(csv_path)
    max_fold = df['fold'].max()
    
    if max_fold == 10:
        # 10-fold structure: map to 5 sessions
        fold_to_session = {1: 1, 2: 1, 3: 2, 4: 2, 5: 3, 6: 3, 7: 4, 8: 4, 9: 5, 10: 5}
        df['session'] = df['fold'].map(fold_to_session)
        test_df = df[df['session'] == fold_id].reset_index(drop=True)
    else:
        # Already 5-fold
        test_df = df[df['fold'] == fold_id].reset_index(drop=True)
    
    return test_df


def evaluate_fold(model, loader, device):
    """
    Run inference on a fold and collect predictions & ground truth
    """
    model.eval()
    
    all_emotion_pred = []
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
            
            # Emotion predictions
            preds = torch.argmax(outputs['emotion_logits'], dim=1)
            all_emotion_pred.extend(preds.cpu().numpy())
            all_emotion_true.extend(batch['emotion_label'].numpy())
            
            # A-V predictions (already normalized in [-1, 1] from model)
            all_arousal_pred.extend(outputs['arousal_value'].cpu().numpy().flatten())
            all_valence_pred.extend(outputs['valence_value'].cpu().numpy().flatten())
            
            # A-V ground truth (normalized in dataset)
            all_arousal_true.extend(batch['arousal_value'].numpy().flatten())
            all_valence_true.extend(batch['valence_value'].numpy().flatten())
    
    return {
        'emotion_pred': np.array(all_emotion_pred),
        'emotion_true': np.array(all_emotion_true),
        'arousal_pred': np.array(all_arousal_pred),
        'arousal_true': np.array(all_arousal_true),
        'valence_pred': np.array(all_valence_pred),
        'valence_true': np.array(all_valence_true),
    }


def compute_metrics(results):
    """Compute all evaluation metrics"""
    metrics = {}
    
    # Classification metrics
    emotion_pred = results['emotion_pred']
    emotion_true = results['emotion_true']
    
    metrics['accuracy'] = accuracy_score(emotion_true, emotion_pred)
    metrics['f1_macro'] = f1_score(emotion_true, emotion_pred, average='macro')
    metrics['f1_weighted'] = f1_score(emotion_true, emotion_pred, average='weighted')
    
    # Confusion matrix & per-class metrics
    cm = confusion_matrix(emotion_true, emotion_pred, labels=[0, 1, 2, 3])
    metrics['confusion_matrix'] = cm
    
    # Regression metrics
    arousal_pred = results['arousal_pred']
    arousal_true = results['arousal_true']
    valence_pred = results['valence_pred']
    valence_true = results['valence_true']
    
    metrics['arousal_ccc'] = compute_ccc(arousal_true, arousal_pred)
    metrics['valence_ccc'] = compute_ccc(valence_true, valence_pred)
    
    if len(arousal_true) > 1:
        metrics['arousal_pearson'], _ = pearsonr(arousal_true, arousal_pred)
    else:
        metrics['arousal_pearson'] = np.nan
    
    if len(valence_true) > 1:
        metrics['valence_pearson'], _ = pearsonr(valence_true, valence_pred)
    else:
        metrics['valence_pearson'] = np.nan
    
    metrics['arousal_mae'] = np.mean(np.abs(arousal_true - arousal_pred))
    metrics['valence_mae'] = np.mean(np.abs(valence_true - valence_pred))
    
    # Clustering metrics on predicted A-V space
    cluster_metrics = compute_cluster_metrics(arousal_pred, valence_pred, emotion_true)
    metrics['silhouette_pred'] = cluster_metrics['silhouette']
    metrics['davies_bouldin_pred'] = cluster_metrics['davies_bouldin']
    
    # Circumplex alignment
    arousal_by_emotion = {}
    valence_by_emotion = {}
    for emo_id in range(4):
        emo_name = EMOTION_MAP[emo_id]
        mask = emotion_true == emo_id
        if mask.sum() > 0:
            arousal_by_emotion[emo_name] = arousal_pred[mask]
            valence_by_emotion[emo_name] = valence_pred[mask]
    
    circumplex = compute_circumplex_alignment(arousal_by_emotion, valence_by_emotion)
    metrics['circumplex_alignment'] = circumplex['overall_alignment']
    metrics['circumplex_details'] = circumplex['per_emotion']
    
    # Per-class accuracy
    for emo_id in range(4):
        mask = emotion_true == emo_id
        if mask.sum() > 0:
            emo_name = EMOTION_MAP[emo_id]
            metrics[f'accuracy_{emo_name}'] = accuracy_score(
                emotion_true[mask], emotion_pred[mask]
            )
    
    return metrics


def print_fold_report(fold_id, metrics, dataset_size):
    """Print formatted report for a fold"""
    print(f"\n{'='*70}")
    print(f" FOLD {fold_id} RESULTS")
    print(f"{'='*70}")
    print(f"Test samples: {dataset_size}\n")
    
    print("📊 CLASSIFICATION METRICS")
    print(f"  Accuracy:       {metrics['accuracy']:.4f}")
    print(f"  F1 (macro):     {metrics['f1_macro']:.4f}")
    print(f"  F1 (weighted):  {metrics['f1_weighted']:.4f}")
    
    print("\n  Per-emotion accuracy:")
    for emo_id, emo_name in EMOTION_MAP.items():
        key = f'accuracy_{emo_name}'
        if key in metrics:
            print(f"    {emo_name:8s}: {metrics[key]:.4f}")
    
    print("\n📈 REGRESSION METRICS (A-V)")
    print(f"  Arousal CCC:    {metrics['arousal_ccc']:.4f}")
    print(f"  Valence CCC:    {metrics['valence_ccc']:.4f}")
    print(f"  Arousal Pearson:{metrics['arousal_pearson']:.4f}")
    print(f"  Valence Pearson:{metrics['valence_pearson']:.4f}")
    print(f"  Arousal MAE:    {metrics['arousal_mae']:.4f}")
    print(f"  Valence MAE:    {metrics['valence_mae']:.4f}")
    
    print("\n🔮 CLUSTERING METRICS (on predicted A-V)")
    print(f"  Silhouette Score:     {metrics['silhouette_pred']:.4f}")
    print(f"  Davies-Bouldin Index: {metrics['davies_bouldin_pred']:.4f}")
    
    print("\n🎯 CIRCUMPLEX ALIGNMENT")
    print(f"  Overall Alignment: {metrics['circumplex_alignment']:.4f}")
    if metrics.get('circumplex_details'):
        print("  Per-emotion deviation from theory:")
        for emo_name, details in metrics['circumplex_details'].items():
            pred = details['predicted']
            exp = details['expected']
            dev = details['deviation']
            print(f"    {emo_name:8s}: pred=(A:{pred[0]:+.2f}, V:{pred[1]:+.2f}), "
                  f"theory=(A:{exp[0]:+.2f}, V:{exp[1]:+.2f}), deviation={dev:.3f}")
    
    print("\n🎯 CONFUSION MATRIX")
    emotion_names = [EMOTION_MAP[i] for i in range(4)]
    cm = metrics['confusion_matrix']
    print("     ", end="")
    for name in emotion_names:
        print(f"{name:8s}", end="")
    print()
    for i, name in enumerate(emotion_names):
        print(f"{name:4s}:", end="")
        for j in range(4):
            print(f"{cm[i, j]:8d}", end="")
        print()
    
    print(f"\n{'='*70}\n")


def parse_args():
    parser = argparse.ArgumentParser(
        description='IEMOCAP Method2 (no W-PRO) 5-fold inference'
    )
    parser.add_argument(
        '--checkpoint_dir',
        type=str,
        default='checkpoints/method2_kfold',
        help='Directory containing fold_*/best_model.pt checkpoints'
    )
    parser.add_argument(
        '--csv',
        type=str,
        default='datasets/IEMOCAP_KFOLD/tenfolds/all.csv',
        help='Path to IEMOCAP CSV file'
    )
    parser.add_argument(
        '--folds',
        type=int,
        nargs='+',
        default=[1, 2, 3, 4, 5],
        help='Folds to evaluate'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='logs/eval_method2_kfold',
        help='Directory to save detailed logs'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda:0',
        help='Device to use'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=4,
        help='Batch size for inference'
    )
    parser.add_argument(
        '--wavlm_prosody_model_name',
        type=str,
        default=None,
        help='Optional prosody WavLM model name (e.g., Orange/Speaker-wavLM-pro) to match checkpoints'
    )
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'#'*70}")
    print(f" IEMOCAP Method2 (No W-PRO) 5-Fold Evaluation")
    print(f"{'#'*70}")
    print(f"Checkpoint dir: {args.checkpoint_dir}")
    print(f"CSV: {args.csv}")
    print(f"Folds: {args.folds}")
    print(f"Device: {device}")
    print(f"Output dir: {output_dir}\n")
    
    all_metrics = []
    summary_data = []
    
    for fold_id in args.folds:
        print(f"\n[FOLD {fold_id}] Loading checkpoint...")
        
        ckpt_path = Path(args.checkpoint_dir) / f'fold_{fold_id}' / 'best_model.pt'
        if not ckpt_path.exists():
            print(f"  ✗ Checkpoint not found: {ckpt_path}")
            continue
        
        # Load model (optionally using W-PRO if provided)
        try:
            model = Method2CrossAttentionAV(
                num_emotions=4,
                wavlm_model_name="microsoft/wavlm-base-plus",
                wavlm_prosody_model_name=args.wavlm_prosody_model_name,  # Use provided prosody model name if set
                roberta_model_name="roberta-base",
                hidden_dim=256,
                freeze_wavlm=True,
                freeze_roberta=True
            ).to(device)
            model = load_checkpoint(model, ckpt_path)
            print(f"  ✓ Model loaded from: {ckpt_path}")
        except Exception as e:
            print(f"  ✗ Failed to load model: {e}")
            continue
        
        # Load test data
        print(f"  Loading test data for fold {fold_id}...")
        test_df = get_test_df(args.csv, fold_id)
        print(f"    Test samples: {len(test_df)}")
        
        try:
            dataset = IEMOCAPDataset(
                test_df,
                audio_max_length=16000 * 10,
                sample_rate=16000,
                normalize_av=True,
                tokenizer_name='roberta-base',
                max_text_length=128
            )
            loader = DataLoader(
                dataset,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=2,
                collate_fn=iemocap_collate_fn,
                pin_memory=True
            )
            print(f"    Dataset size: {len(dataset)}")
        except Exception as e:
            print(f"  ✗ Failed to load dataset: {e}")
            continue
        
        # Evaluate
        print(f"  Evaluating...")
        try:
            results = evaluate_fold(model, loader, device)
            metrics = compute_metrics(results)
            all_metrics.append(metrics)
            
            # Print report
            print_fold_report(fold_id, metrics, len(dataset))
            
            # Save detailed log
            log_lines = [
                f"Fold: {fold_id}",
                f"Test samples: {len(dataset)}",
                f"Checkpoint: {ckpt_path}",
                "",
                "CLASSIFICATION",
                f"  accuracy       = {metrics['accuracy']:.4f}",
                f"  f1_macro       = {metrics['f1_macro']:.4f}",
                f"  f1_weighted    = {metrics['f1_weighted']:.4f}",
                "",
                "REGRESSION (A-V)",
                f"  arousal_ccc    = {metrics['arousal_ccc']:.4f}",
                f"  valence_ccc    = {metrics['valence_ccc']:.4f}",
                f"  arousal_pearson= {metrics['arousal_pearson']:.4f}",
                f"  valence_pearson= {metrics['valence_pearson']:.4f}",
                f"  arousal_mae    = {metrics['arousal_mae']:.4f}",
                f"  valence_mae    = {metrics['valence_mae']:.4f}",
                "",
                "CLUSTERING (A-V)",
                f"  silhouette_pred    = {metrics['silhouette_pred']:.4f}",
                f"  davies_bouldin_pred= {metrics['davies_bouldin_pred']:.4f}",
                "",
                "CIRCUMPLEX ALIGNMENT",
                f"  circumplex_alignment = {metrics['circumplex_alignment']:.4f}",
            ]
            log_path = output_dir / f"fold_{fold_id}_metrics.log"
            log_path.write_text("\n".join(log_lines))
            print(f"  ✓ Metrics logged to: {log_path}")
            
            # Record for summary
            summary_data.append({
                'fold': fold_id,
                'accuracy': metrics['accuracy'],
                'f1_macro': metrics['f1_macro'],
                'f1_weighted': metrics['f1_weighted'],
                'arousal_ccc': metrics['arousal_ccc'],
                'valence_ccc': metrics['valence_ccc'],
                'arousal_pearson': metrics['arousal_pearson'],
                'valence_pearson': metrics['valence_pearson'],
                'arousal_mae': metrics['arousal_mae'],
                'valence_mae': metrics['valence_mae'],
                'silhouette_pred': metrics['silhouette_pred'],
                'davies_bouldin_pred': metrics['davies_bouldin_pred'],
                'circumplex_alignment': metrics['circumplex_alignment'],
            })
            
        except Exception as e:
            print(f"  ✗ Evaluation failed: {e}")
            import traceback
            traceback.print_exc()
            continue
        finally:
            del model
            torch.cuda.empty_cache()
    
    # === AGGREGATE RESULTS ===
    if all_metrics:
        print(f"\n{'#'*70}")
        print(f" AGGREGATE RESULTS (Folds {args.folds})")
        print(f"{'#'*70}\n")
        
        metric_keys = [
            'accuracy', 'f1_macro', 'f1_weighted',
            'arousal_ccc', 'valence_ccc',
            'arousal_pearson', 'valence_pearson',
            'arousal_mae', 'valence_mae',
            'silhouette_pred', 'davies_bouldin_pred',
            'circumplex_alignment'
        ]
        
        print("📊 AVERAGE METRICS (mean ± std)")
        print("-" * 60)
        for key in metric_keys:
            values = [m[key] for m in all_metrics if not np.isnan(m.get(key, np.nan))]
            if values:
                mean_val = np.mean(values)
                std_val = np.std(values)
                print(f"  {key:20s}: {mean_val:.4f} ± {std_val:.4f}")
        
        # Save summary CSV
        df_summary = pd.DataFrame(summary_data)
        csv_path = output_dir / 'summary_all_folds.csv'
        df_summary.to_csv(csv_path, index=False)
        print(f"\n✓ Summary saved to: {csv_path}")
        print("\nSummary:")
        print(df_summary.to_string(index=False))
    
    print(f"\n✅ Evaluation complete!")


if __name__ == '__main__':
    main()
