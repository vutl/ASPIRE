#!/usr/bin/env python3
"""
Evaluation script for trained W-PRO models
Load checkpoints and run real inference on test data
Models: method2_kfold_wpro (IEMOCAP) and method2_msp_wpro (MSP-Podcast)
"""
import os
import sys
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import DataLoader
import warnings
warnings.filterwarnings('ignore')

sys.path.append(str(Path(__file__).resolve().parent))

from models.method2_crossattn_av import Method2CrossAttentionAV
from data.iemocap_dataset import IEMOCAPDataset, collate_fn as iemocap_collate_fn
from data.msp_podcast_dataset import MSPPodcastDataset, collate_fn as msp_collate_fn

EMOTION_COLORS = {
    'Happy': '#2ecc71',     # Green (was Happiness)
    'Sad': '#3498db',       # Blue (was Sadness)
    'Neutral': '#95a5a6',   # Gray
    'Angry': '#e74c3c'      # Red (was Anger)
}

# FIXED: Match IEMOCAPDataset mapping: {'hap': 0, 'sad': 1, 'neu': 2, 'ang': 3}
EMOTION_NAMES = {0: 'Happy', 1: 'Sad', 2: 'Neutral', 3: 'Angry'}

def get_emotion_label(emotion_id):
    return EMOTION_NAMES.get(int(emotion_id), 'Unknown')

def compute_ccc(y_true, y_pred):
    """Compute Concordance Correlation Coefficient"""
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()
    if len(y_true) < 2:
        return 0.0
    mean_true, mean_pred = np.mean(y_true), np.mean(y_pred)
    var_true, var_pred = np.var(y_true), np.var(y_pred)
    cov = np.mean((y_true - mean_true) * (y_pred - mean_pred))
    return float(2 * cov / (var_true + var_pred + (mean_true - mean_pred) ** 2 + 1e-8))

def load_model(checkpoint_path, device):
    """Load Method2 W-PRO model from checkpoint"""
    print(f"  Loading: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    model = Method2CrossAttentionAV(
        num_emotions=4,
        wavlm_model_name="microsoft/wavlm-base-plus",
        wavlm_prosody_model_name="Orange/Speaker-wavLM-pro",
        roberta_model_name="roberta-base",
        hidden_dim=256,
        freeze_wavlm=True,
        freeze_roberta=True
    ).to(device)
    
    # Extract state dict and filter out thop keys
    if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
        state_dict = ckpt['model_state_dict']
    elif isinstance(ckpt, dict) and 'state_dict' in ckpt:
        state_dict = ckpt['state_dict']
    else:
        state_dict = ckpt
    
    # Filter out thop profiling keys (total_ops, total_params)
    state_dict = {k: v for k, v in state_dict.items() if not k.endswith(('total_ops', 'total_params'))}
    model.load_state_dict(state_dict, strict=False)
    
    model.eval()
    print(f"  ✓ Loaded")
    return model

def evaluate(model, dataloader, device):
    """Run inference"""
    preds, labels, a_pred, a_true, v_pred, v_true = [], [], [], [], [], []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="  Inference", leave=False):
            audio = batch['audio'].to(device)
            text_input_ids = batch['text_input_ids'].to(device)
            text_attention_mask = batch['text_attention_mask'].to(device)
            
            outputs = model(audio, text_input_ids, text_attention_mask=text_attention_mask)
            
            preds.extend(torch.argmax(outputs['emotion_logits'], dim=1).cpu().numpy())
            labels.extend(batch['emotion_label'].cpu().numpy())
            a_pred.extend(outputs['arousal_value'].cpu().numpy().flatten())
            a_true.extend(batch['arousal_value'].cpu().numpy().flatten())
            v_pred.extend(outputs['valence_value'].cpu().numpy().flatten())
            v_true.extend(batch['valence_value'].cpu().numpy().flatten())
    
    results = {
        'emotion_preds': np.array(preds),
        'emotion_labels': np.array(labels),
        'arousal_preds': np.array(a_pred),
        'arousal_true': np.array(a_true),
        'valence_preds': np.array(v_pred),
        'valence_true': np.array(v_true)
    }
    
    acc = np.mean(results['emotion_preds'] == results['emotion_labels'])
    a_ccc = compute_ccc(results['arousal_true'], results['arousal_preds'])
    v_ccc = compute_ccc(results['valence_true'], results['valence_preds'])
    
    print(f"  Acc: {acc:.4f} | A-CCC: {a_ccc:.4f} | V-CCC: {v_ccc:.4f}")
    return results

def plot_scatter(results, model_name, split, output_dir):
    """Plot A-V scatter"""
    os.makedirs(output_dir, exist_ok=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f'{model_name} - {split}', fontsize=14, fontweight='bold')
    
    emotion_labels = [get_emotion_label(e) for e in results['emotion_labels']]
    unique = sorted(set(emotion_labels))
    
    for ax, (a_data, v_data, title) in enumerate([
        (results['arousal_true'], results['valence_true'], 'Ground Truth'),
        (results['arousal_preds'], results['valence_preds'], 'Predicted')
    ]):
        for emo in unique:
            mask = np.array([e == emo for e in emotion_labels])
            axes[ax].scatter(a_data[mask], v_data[mask], 
                           label=emo, alpha=0.6, s=50, 
                           color=EMOTION_COLORS[emo], 
                           edgecolors='black', linewidth=0.5)
        
        axes[ax].set_xlabel('Arousal', fontsize=12)
        axes[ax].set_ylabel('Valence', fontsize=12)
        axes[ax].set_title(title, fontsize=12, fontweight='bold')
        axes[ax].legend(loc='best', fontsize=10)
        axes[ax].grid(True, alpha=0.3)
        axes[ax].set_xlim(0.5, 5.5)
        axes[ax].set_ylim(0.5, 5.5)
    
    path = os.path.join(output_dir, f'av_{model_name}_{split}.png')
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {path}")

def eval_iemocap(checkpoint_dir, device, output_dir):
    """Evaluate IEMOCAP 5-fold"""
    print("\n" + "="*80)
    print("IEMOCAP 5-Fold (method2_kfold_wpro)")
    print("="*80)
    
    for fold in range(1, 6):
        print(f"\n[Fold {fold}]")
        ckpt_path = os.path.join(checkpoint_dir, f'fold_{fold}', 'best_model.pt')
        
        if not os.path.exists(ckpt_path):
            print(f"  ✗ Not found: {ckpt_path}")
            continue
        
        model = load_model(ckpt_path, device)
        
        # Load and filter test data for this fold
        df = pd.read_csv('datasets/IEMOCAP_KFOLD/tenfolds/all.csv')
        test_df = df[df['fold'] == fold].reset_index(drop=True)
        
        dataset = IEMOCAPDataset(
            dataframe_or_csv=test_df,
            audio_max_length=16000*10,
            sample_rate=16000,
            normalize_av=True
        )
        
        loader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=2, collate_fn=iemocap_collate_fn)
        print(f"  Samples: {len(dataset)}")
        
        results = evaluate(model, loader, device)
        plot_scatter(results, 'wpro_iemocap', f'fold_{fold}', output_dir)
        
        del model
        torch.cuda.empty_cache()

def eval_msp(checkpoint_dir, device, output_dir):
    """Evaluate MSP-Podcast"""
    print("\n" + "="*80)
    print("MSP-Podcast (method2_msp_wpro)")
    print("="*80)
    
    ckpt_path = os.path.join(checkpoint_dir, 'best_model.pt')
    if not os.path.exists(ckpt_path):
        print(f"  ✗ Not found: {ckpt_path}")
        return
    
    model = load_model(ckpt_path, device)
    
    for split in ['test1', 'test2']:
        print(f"\n[{split.upper()}]")
        
        dataset = MSPPodcastDataset(
            tsv_path='datasets/MSP-Podcast/msp_podcast.tsv',
            split=split,
            audio_max_length=16000*10,
            sample_rate=16000,
            normalize_av=True
        )
        
        loader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=2, collate_fn=msp_collate_fn)
        print(f"  Samples: {len(dataset)}")
        
        results = evaluate(model, loader, device)
        plot_scatter(results, 'wpro_msp', split, output_dir)
    
    del model
    torch.cuda.empty_cache()

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--output_dir', default='results/av_evaluation_wpro')
    args = parser.parse_args()
    
    print("="*80)
    print("A-V EVALUATION - W-PRO MODELS")
    print("="*80)
    
    if os.path.exists('checkpoints/method2_kfold_wpro'):
        eval_iemocap('checkpoints/method2_kfold_wpro', args.device, args.output_dir)

    
    print("\n" + "="*80)
    print(f"✓ Done! Plots: {args.output_dir}")
    print("="*80)

if __name__ == '__main__':
    main()
