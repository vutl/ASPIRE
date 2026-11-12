#!/usr/bin/env python3
"""
Evaluate Method 2 W-PRO Fold 1: Confusion Matrix + Scatter Plot
Diagnose Neutral class prediction quality
"""
import os
import sys
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

sys.path.append(str(Path(__file__).resolve().parent.parent))

from data.iemocap_dataset import IEMOCAPDataset, collate_fn as iemocap_collate_fn
from models.method2_crossattn_av import Method2CrossAttentionAV

# Settings - FIXED: Match IEMOCAPDataset mapping
EMOTION_NAMES = {0: 'Happy', 1: 'Sad', 2: 'Neutral', 3: 'Angry'}
EMOTION_COLORS = {
    'Happy': '#2ecc71',
    'Sad': '#3498db',
    'Neutral': '#95a5a6',
    'Angry': '#e74c3c'
}


def load_checkpoint(ckpt_path, device):
    """Load model and checkpoint"""
    model = Method2CrossAttentionAV(
        num_emotions=4,
        wavlm_model_name="microsoft/wavlm-base-plus",
        wavlm_prosody_model_name="Orange/Speaker-wavLM-pro",
        roberta_model_name="roberta-base",
        hidden_dim=256,
        freeze_wavlm=True,
        freeze_roberta=True
    ).to(device)
    
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    state_dict = ckpt.get('model_state_dict', ckpt)
    state_dict = {k: v for k, v in state_dict.items() if not k.endswith(('total_ops', 'total_params'))}
    model.load_state_dict(state_dict, strict=False)
    return model


def get_fold1_test_df(csv_path):
    """Load Fold 1 test data"""
    df = pd.read_csv(csv_path)
    fold_to_session = {1: 1, 2: 1, 3: 2, 4: 2, 5: 3, 6: 3, 7: 4, 8: 4, 9: 5, 10: 5}
    df['session'] = df['fold'].map(fold_to_session)
    return df[df['session'] == 1].reset_index(drop=True)


def run_inference(model, loader, device):
    """Run inference and get predictions + A-V values"""
    model.eval()
    all_preds = []
    all_labels = []
    all_arousal = []
    all_valence = []
    
    with torch.no_grad():
        for batch in tqdm(loader, desc='Inference', leave=False):
            audio = batch['audio'].to(device)
            text_input_ids = batch['text_input_ids'].to(device)
            text_attention_mask = batch['text_attention_mask'].to(device)
            
            outputs = model(audio, text_input_ids, text_attention_mask=text_attention_mask)
            
            preds = torch.argmax(outputs['emotion_logits'], dim=1).cpu().numpy()
            labels = batch['emotion_label'].cpu().numpy()
            arousal = outputs['arousal_value'].cpu().numpy().flatten()
            valence = outputs['valence_value'].cpu().numpy().flatten()
            
            all_preds.extend(preds)
            all_labels.extend(labels)
            all_arousal.extend(arousal)
            all_valence.extend(valence)
    
    return np.array(all_preds), np.array(all_labels), np.array(all_arousal), np.array(all_valence)


def plot_confusion_matrix(y_true, y_pred, save_path):
    """Plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3])
    
    plt.figure(figsize=(8, 7))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=[EMOTION_NAMES[i] for i in range(4)],
        yticklabels=[EMOTION_NAMES[i] for i in range(4)],
        cbar_kws={'label': 'Count'}
    )
    plt.xlabel('Predicted Emotion', fontsize=12)
    plt.ylabel('Ground Truth Emotion', fontsize=12)
    plt.title('Method 2 W-PRO Fold 1: Confusion Matrix', fontsize=14, fontweight='bold')
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {save_path}")


def plot_scatter(arousal, valence, emotion_preds, save_path):
    """Plot A-V scatter with predicted emotions"""
    labels = [EMOTION_NAMES.get(int(e), 'Unknown') for e in emotion_preds]
    all_emotions = [EMOTION_NAMES[i] for i in sorted(EMOTION_NAMES.keys())]
    
    # Print class counts
    counts = {emo: 0 for emo in all_emotions}
    for emo in labels:
        counts[emo] = counts.get(emo, 0) + 1
    print(f"  Pred counts: {counts}")
    
    plt.figure(figsize=(8, 8))
    for emo in all_emotions:
        mask = np.array([e == emo for e in labels])
        label_text = f"{emo} ({mask.sum()})"
        plt.scatter(arousal[mask], valence[mask], label=label_text, alpha=0.6, s=50,
                    color=EMOTION_COLORS.get(emo, '#888888'), edgecolors='black', linewidth=0.5)
    
    plt.xlabel('Arousal (predicted, normalized)', fontsize=12)
    plt.ylabel('Valence (predicted, normalized)', fontsize=12)
    plt.title('Method 2 W-PRO Fold 1: Arousal-Valence Space', fontsize=14, fontweight='bold')
    plt.xlim(-0.1, 1.1)
    plt.ylim(-0.1, 1.1)
    plt.grid(True, alpha=0.3)
    plt.legend(loc='best', fontsize=10)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {save_path}")


def print_classification_report(y_true, y_pred):
    """Print detailed classification report per class"""
    print("\n" + "="*80)
    print("DETAILED CLASSIFICATION REPORT (Per-Class Metrics)")
    print("="*80)
    print(classification_report(
        y_true, y_pred,
        target_names=[EMOTION_NAMES[i] for i in range(4)],
        digits=4
    ))


def main():
    device = torch.device('cuda:0')
    
    print("\n" + "="*80)
    print("🔍 METHOD 2 W-PRO FOLD 1 EVALUATION")
    print("="*80)
    
    # Load data
    csv_path = "datasets/IEMOCAP_KFOLD/tenfolds/all.csv"
    test_df = get_fold1_test_df(csv_path)
    
    print(f"\nFold 1 Test Set:")
    print(f"  Total samples: {len(test_df)}")
    print(f"\n  GT Distribution:")
    for emo in ['neu', 'hap', 'sad', 'ang']:
        count = (test_df['emotion'] == emo).sum()
        print(f"    {emo}: {count}")
    
    # Create dataset & loader
    dataset = IEMOCAPDataset(test_df, audio_max_length=16000*10, sample_rate=16000, normalize_av=True)
    loader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=2, collate_fn=iemocap_collate_fn)
    
    # Load checkpoint
    ckpt_path = Path("checkpoints/method2_kfold_wpro/fold_1/best_model.pt")
    print(f"\n📦 Loading checkpoint: {ckpt_path}")
    model = load_checkpoint(ckpt_path, device)
    
    # Run inference
    print(f"\n▶️  Running inference...")
    preds, labels, arousal, valence = run_inference(model, loader, device)
    
    # Normalize A-V to [0, 1]
    arousal_norm = (arousal - arousal.min() + 1e-8) / (arousal.max() - arousal.min() + 1e-8)
    valence_norm = (valence - valence.min() + 1e-8) / (valence.max() - valence.min() + 1e-8)
    
    # Compute metrics
    acc = accuracy_score(labels, preds)
    
    print(f"\n📊 OVERALL METRICS")
    print(f"{'='*80}")
    print(f"Accuracy: {acc:.4f}")
    print(f"{'='*80}")
    
    print(f"\n  Predicted Distribution:")
    for i in range(4):
        count = (preds == i).sum()
        print(f"    {EMOTION_NAMES[i]}: {count}")
    
    # Per-class metrics
    print_classification_report(labels, preds)
    
    # Plot confusion matrix
    cm_path = "outputs/eval_method2_wpro_fold1/confusion_matrix.png"
    print(f"\n📈 Plotting confusion matrix...")
    plot_confusion_matrix(labels, preds, cm_path)
    
    # Plot scatter
    scatter_path = "outputs/eval_method2_wpro_fold1/av_scatter.png"
    print(f"\n📊 Plotting A-V scatter...")
    plot_scatter(arousal_norm, valence_norm, preds, scatter_path)
    
    print(f"\n✅ Evaluation complete!")


if __name__ == '__main__':
    main()
