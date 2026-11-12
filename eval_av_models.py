#!/usr/bin/env python3
"""
Evaluate OpenSMILE + Method2 models on IEMOCAP & MSP-Podcast
Plot A-V predictions colored by emotion label
"""
import os
import sys
import json
import yaml
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Patch

sys.path.append(str(Path(__file__).resolve().parent))

from models.opensmile_emotion_av import OpenSMILEEmotionAVClassifier
from models.method2_crossattn_av_opensmile import Method2CrossAttentionAVOpenSMILE


class OpenSMILEEvalDataset(Dataset):
    """OpenSMILE evaluation dataset"""
    def __init__(self, df, opensmile_dir, mapping_json=None):
        self.df = df.reset_index(drop=True)
        self.opensmile_dir = opensmile_dir
        self.mapping = {}
        if mapping_json and os.path.exists(mapping_json):
            with open(mapping_json, 'r') as f:
                self.mapping = json.load(f)
        self.emotion_to_label = {'hap': 0, 'sad': 1, 'neu': 2, 'ang': 3}
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        sample_name = row['sample_name']
        emotion = row['emotion']
        arousal = float(row.get('arousal', 0.0))
        valence = float(row.get('valence', 0.0))
        
        if sample_name in self.mapping:
            npy_file = self.mapping[sample_name]
        else:
            npy_file = os.path.join(self.opensmile_dir, f"{sample_name}.npy")
        
        try:
            embedding = np.load(npy_file).squeeze()
        except:
            embedding = np.zeros(88, dtype=np.float32)
        
        emotion_label = self.emotion_to_label.get(emotion, 2)
        arousal_norm = (arousal - 3.0) / 2.0
        valence_norm = (valence - 3.0) / 2.0
        
        return {
            'embedding': embedding.astype(np.float32),
            'emotion_label': emotion_label,
            'arousal': np.array([arousal_norm], dtype=np.float32),
            'valence': np.array([valence_norm], dtype=np.float32),
            'emotion': emotion
        }


class Method2EvalDataset(Dataset):
    """Method2 evaluation dataset"""
    def __init__(self, df, wavlm_dir, roberta_dir, opensmile_dir, mapping_json=None):
        self.df = df.reset_index(drop=True)
        self.wavlm_dir = wavlm_dir
        self.roberta_dir = roberta_dir
        self.opensmile_dir = opensmile_dir
        self.mapping = {}
        if mapping_json and os.path.exists(mapping_json):
            with open(mapping_json, 'r') as f:
                self.mapping = json.load(f)
        self.emotion_to_label = {'hap': 0, 'sad': 1, 'neu': 2, 'ang': 3}
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        sample_name = row['sample_name']
        emotion = row['emotion']
        arousal = float(row.get('arousal', 0.0))
        valence = float(row.get('valence', 0.0))
        
        try:
            wavlm = np.load(os.path.join(self.wavlm_dir, f"{sample_name}.npy")).squeeze()
            if wavlm.ndim == 1:
                wavlm = wavlm.reshape(1, -1)
        except:
            wavlm = np.zeros((1, 768), dtype=np.float32)
        
        try:
            roberta = np.load(os.path.join(self.roberta_dir, f"{sample_name}.npy")).squeeze()
            if roberta.ndim == 1:
                roberta = roberta.reshape(1, -1)
        except:
            roberta = np.zeros((1, 768), dtype=np.float32)
        
        try:
            if sample_name in self.mapping:
                opensmile_file = self.mapping[sample_name]
            else:
                opensmile_file = os.path.join(self.opensmile_dir, f"{sample_name}.npy")
            opensmile = np.load(opensmile_file).squeeze()
            if opensmile.ndim == 0:
                opensmile = np.array([opensmile])
        except:
            opensmile = np.zeros(88, dtype=np.float32)
        
        emotion_label = self.emotion_to_label.get(emotion, 2)
        
        return {
            'wavlm': wavlm.astype(np.float32),
            'roberta': roberta.astype(np.float32),
            'opensmile': opensmile.astype(np.float32),
            'emotion_label': emotion_label,
            'arousal': arousal,
            'valence': valence,
            'emotion': emotion
        }


def collate_opensmile(batch):
    embeddings = np.stack([item['embedding'] for item in batch])
    arousals = np.concatenate([item['arousal'] for item in batch]).squeeze()
    valences = np.concatenate([item['valence'] for item in batch]).squeeze()
    return {
        'embedding': embeddings,
        'arousal': arousals,
        'valence': valences,
        'emotion_label': np.array([item['emotion_label'] for item in batch]),
        'emotion': [item['emotion'] for item in batch]
    }


def collate_method2(batch):
    wavlms = [item['wavlm'] for item in batch]
    robertas = [item['roberta'] for item in batch]
    opensmiles = [item['opensmile'] for item in batch]
    
    max_len_wavlm = max(w.shape[0] for w in wavlms)
    max_len_roberta = max(r.shape[0] for r in robertas)
    
    wavlms_padded = []
    robertas_padded = []
    
    for w in wavlms:
        if w.shape[0] < max_len_wavlm:
            w = np.vstack([w, np.zeros((max_len_wavlm - w.shape[0], w.shape[1]))])
        wavlms_padded.append(w)
    
    for r in robertas:
        if r.shape[0] < max_len_roberta:
            r = np.vstack([r, np.zeros((max_len_roberta - r.shape[0], r.shape[1]))])
        robertas_padded.append(r)
    
    return {
        'wavlm': np.array(wavlms_padded),
        'roberta': np.array(robertas_padded),
        'opensmile': np.array(opensmiles),
        'arousal': np.array([item['arousal'] for item in batch]),
        'valence': np.array([item['valence'] for item in batch]),
        'emotion_label': np.array([item['emotion_label'] for item in batch]),
        'emotion': [item['emotion'] for item in batch]
    }


def evaluate_opensmile(model, loader, device):
    """Evaluate OpenSMILE model"""
    model.eval()
    all_preds = {'arousal': [], 'valence': [], 'emotion': [], 'emotion_label': []}
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating OpenSMILE", leave=False):
            embeddings = torch.from_numpy(batch['embedding']).to(device)
            outputs = model(embeddings)
            
            all_preds['arousal'].extend(outputs['arousal'].cpu().numpy().squeeze())
            all_preds['valence'].extend(outputs['valence'].cpu().numpy().squeeze())
            all_preds['emotion'].extend(batch['emotion'])
            all_preds['emotion_label'].extend(batch['emotion_label'])
    
    return all_preds


def evaluate_method2(model, loader, device):
    """Evaluate Method2 model"""
    model.eval()
    all_preds = {'arousal': [], 'valence': [], 'emotion': [], 'emotion_label': []}
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating Method2", leave=False):
            wavlm = torch.from_numpy(batch['wavlm']).to(device)
            roberta = torch.from_numpy(batch['roberta']).to(device)
            opensmile = torch.from_numpy(batch['opensmile']).to(device)
            
            outputs = model(wavlm, roberta, opensmile)
            
            all_preds['arousal'].extend(outputs['arousal'].cpu().numpy().squeeze())
            all_preds['valence'].extend(outputs['valence'].cpu().numpy().squeeze())
            all_preds['emotion'].extend(batch['emotion'])
            all_preds['emotion_label'].extend(batch['emotion_label'])
    
    return all_preds


def plot_av_scatter(opensmile_preds, method2_preds, dataset_name, output_dir):
    """Plot A-V scatter colored by emotion"""
    emotion_colors = {'hap': '#FF6B6B', 'sad': '#4ECDC4', 'neu': '#FFE66D', 'ang': '#95E1D3'}
    emotion_names = {'hap': 'Happy', 'sad': 'Sad', 'neu': 'Neutral', 'ang': 'Angry'}
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # OpenSMILE
    for emotion in ['neu', 'sad', 'hap', 'ang']:
        mask = np.array(opensmile_preds['emotion']) == emotion
        axes[0].scatter(
            np.array(opensmile_preds['arousal'])[mask],
            np.array(opensmile_preds['valence'])[mask],
            c=emotion_colors[emotion],
            label=emotion_names[emotion],
            alpha=0.6,
            s=30
        )
    axes[0].set_xlabel('Arousal', fontsize=12)
    axes[0].set_ylabel('Valence', fontsize=12)
    axes[0].set_title(f'OpenSMILE-only ({dataset_name})', fontsize=13, fontweight='bold')
    axes[0].set_xlim(-1.2, 1.2)
    axes[0].set_ylim(-1.2, 1.2)
    axes[0].axhline(y=0, color='gray', linestyle='--', alpha=0.3)
    axes[0].axvline(x=0, color='gray', linestyle='--', alpha=0.3)
    axes[0].legend(loc='upper right')
    axes[0].grid(alpha=0.2)
    
    # Method2
    for emotion in ['neu', 'sad', 'hap', 'ang']:
        mask = np.array(method2_preds['emotion']) == emotion
        axes[1].scatter(
            np.array(method2_preds['arousal'])[mask],
            np.array(method2_preds['valence'])[mask],
            c=emotion_colors[emotion],
            label=emotion_names[emotion],
            alpha=0.6,
            s=30
        )
    axes[1].set_xlabel('Arousal', fontsize=12)
    axes[1].set_ylabel('Valence', fontsize=12)
    axes[1].set_title(f'Method2 + OpenSMILE ({dataset_name})', fontsize=13, fontweight='bold')
    axes[1].set_xlim(-1.2, 1.2)
    axes[1].set_ylim(-1.2, 1.2)
    axes[1].axhline(y=0, color='gray', linestyle='--', alpha=0.3)
    axes[1].axvline(x=0, color='gray', linestyle='--', alpha=0.3)
    axes[1].legend(loc='upper right')
    axes[1].grid(alpha=0.2)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, f'av_scatter_{dataset_name}.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--output_dir', type=str, default='results/av_evaluation')
    args = parser.parse_args()
    
    device = torch.device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("="*80)
    print("🎬 A-V Model Evaluation: OpenSMILE vs Method2")
    print("="*80)
    
    # ============ IEMOCAP Evaluation ============
    print("\n🔹 IEMOCAP 5-Fold Evaluation\n")
    
    for fold_idx in range(1, 6):
        print(f"--- Fold {fold_idx} ---")
        
        # Load test CSV
        csv_path = 'datasets/IEMOCAP_KFOLD/all.csv'
        df = pd.read_csv(csv_path)
        test_df = df[df['fold'] == fold_idx]
        
        # OpenSMILE model
        opensmile_ckpt = f'checkpoints/iemocap_opensmile_emotion_av_kfold/fold_{fold_idx}/best_model.pth'
        if os.path.exists(opensmile_ckpt):
            opensmile_model = OpenSMILEEmotionAVClassifier().to(device)
            opensmile_model.load_state_dict(torch.load(opensmile_ckpt, map_location=device))
            
            opensmile_dataset = OpenSMILEEvalDataset(
                test_df,
                'outputs/iemocap/egemaps_npy',
                'outputs/iemocap/egemaps_mapping.json'
            )
            opensmile_loader = DataLoader(opensmile_dataset, batch_size=64, collate_fn=collate_opensmile)
            opensmile_preds = evaluate_opensmile(opensmile_model, opensmile_loader, device)
            print(f"  ✓ OpenSMILE evaluated ({len(opensmile_preds['arousal'])} samples)")
        else:
            print(f"  ✗ OpenSMILE checkpoint not found: {opensmile_ckpt}")
            continue
        
        # Method2 model
        method2_ckpt = f'checkpoints/iemocap_method2_opensmile_kfold/fold_{fold_idx}/best_model.pth'
        if os.path.exists(method2_ckpt):
            with open('configs/iemocap_method2_opensmile_kfold_config.yaml') as f:
                config = yaml.safe_load(f)
            
            method2_model = Method2CrossAttentionAVOpenSMILE(
                audio_dim=config['audio_dim'],
                text_dim=config['text_dim'],
                prosody_dim=config['prosody_dim'],
                hidden_dim=config['hidden_dim']
            ).to(device)
            method2_model.load_state_dict(torch.load(method2_ckpt, map_location=device))
            
            method2_dataset = Method2EvalDataset(
                test_df,
                config['wavlm_dir'],
                config['roberta_dir'],
                config['opensmile_dir'],
                config.get('mapping_json')
            )
            method2_loader = DataLoader(method2_dataset, batch_size=64, collate_fn=collate_method2)
            method2_preds = evaluate_method2(method2_model, method2_loader, device)
            print(f"  ✓ Method2 evaluated ({len(method2_preds['arousal'])} samples)")
        else:
            print(f"  ✗ Method2 checkpoint not found: {method2_ckpt}")
            continue
        
        # Plot
        plot_av_scatter(opensmile_preds, method2_preds, f'IEMOCAP_Fold{fold_idx}', args.output_dir)
    
    print("\n✓ IEMOCAP evaluation complete!")
    print(f"\n📊 Results saved to: {args.output_dir}")
    print("="*80)


if __name__ == '__main__':
    main()
