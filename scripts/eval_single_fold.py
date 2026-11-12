#!/usr/bin/env python3
"""
Evaluate a single fold checkpoint on its test set
"""
import os
import sys
from pathlib import Path
import argparse

import torch
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, classification_report
import numpy as np

sys.path.append(str(Path(__file__).resolve().parent.parent))

from data.iemocap_dataset import IEMOCAPDataset, collate_fn
from models.method2_crossattn_av import Method2CrossAttentionAV


def compute_ccc(y_true, y_pred):
    """Compute Concordance Correlation Coefficient"""
    mean_true = np.mean(y_true)
    mean_pred = np.mean(y_pred)
    var_true = np.var(y_true)
    var_pred = np.var(y_pred)
    covariance = np.mean((y_true - mean_true) * (y_pred - mean_pred))
    ccc = (2 * covariance) / (var_true + var_pred + (mean_true - mean_pred) ** 2 + 1e-8)
    return ccc


def load_checkpoint(model, ckpt_path):
    """Load checkpoint into model"""
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
        state_dict = ckpt['model_state_dict']
    elif isinstance(ckpt, dict) and 'state_dict' in ckpt:
        state_dict = ckpt['state_dict']
    else:
        state_dict = ckpt
    
    # Remove hook stats if present
    state_dict = {k: v for k, v in state_dict.items() if not k.endswith(('total_ops', 'total_params'))}
    model.load_state_dict(state_dict, strict=False)
    return model


def get_fold_test_df(csv_path, fold_id):
    """Get test DataFrame for a specific fold"""
    df = pd.read_csv(csv_path)
    max_fold = df['fold'].max()
    
    if max_fold == 10:
        # 10-fold CSV: Map to 5 sessions
        fold_to_session = {1:1, 2:1, 3:2, 4:2, 5:3, 6:3, 7:4, 8:4, 9:5, 10:5}
        df['session'] = df['fold'].map(fold_to_session)
        session_id = fold_id
        test_df = df[df['session'] == session_id].reset_index(drop=True)
    else:
        # 5-fold CSV
        test_df = df[df['fold'] == fold_id].reset_index(drop=True)
    
    return test_df


def evaluate(model, loader, device):
    """Evaluate model on test set"""
    model.eval()
    
    all_preds = []
    all_labels = []
    all_arousal_pred = []
    all_arousal_true = []
    all_valence_pred = []
    all_valence_true = []
    
    with torch.no_grad():
        for batch in loader:
            audio = batch['audio'].to(device)
            text_input_ids = batch['text_input_ids'].to(device)
            text_attention_mask = batch['text_attention_mask'].to(device)
            emotion_label = batch['emotion_label'].to(device)
            arousal = batch['arousal_value'].to(device)
            valence = batch['valence_value'].to(device)
            
            outputs = model(audio, text_input_ids, text_attention_mask=text_attention_mask)
            
            preds = torch.argmax(outputs['emotion_logits'], dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(emotion_label.cpu().numpy())
            
            all_arousal_pred.extend(outputs['arousal_value'].cpu().numpy().flatten())
            all_arousal_true.extend(arousal.cpu().numpy().flatten())
            
            all_valence_pred.extend(outputs['valence_value'].cpu().numpy().flatten())
            all_valence_true.extend(valence.cpu().numpy().flatten())
    
    # Compute metrics
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_arousal_pred = np.array(all_arousal_pred)
    all_arousal_true = np.array(all_arousal_true)
    all_valence_pred = np.array(all_valence_pred)
    all_valence_true = np.array(all_valence_true)
    
    accuracy = accuracy_score(all_labels, all_preds)
    f1_macro = f1_score(all_labels, all_preds, average='macro')
    f1_weighted = f1_score(all_labels, all_preds, average='weighted')
    
    arousal_ccc = compute_ccc(all_arousal_true, all_arousal_pred)
    valence_ccc = compute_ccc(all_valence_true, all_valence_pred)
    avg_ccc = (arousal_ccc + valence_ccc) / 2
    
    return {
        'accuracy': accuracy,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted,
        'arousal_ccc': arousal_ccc,
        'valence_ccc': valence_ccc,
        'avg_ccc': avg_ccc,
        'predictions': all_preds,
        'labels': all_labels,
        'arousal_pred': all_arousal_pred,
        'arousal_true': all_arousal_true,
        'valence_pred': all_valence_pred,
        'valence_true': all_valence_true
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to checkpoint')
    parser.add_argument('--fold', type=int, required=True, help='Fold number (1-5)')
    parser.add_argument('--csv', type=str, default='datasets/IEMOCAP_KFOLD/tenfolds/all.csv')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--use_wpro', action='store_true', help='Use W-PRO model')
    args = parser.parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    print(f"\nLoading model from: {args.checkpoint}")
    model = Method2CrossAttentionAV(
        num_emotions=4,
        wavlm_model_name="microsoft/wavlm-base-plus",
        wavlm_prosody_model_name="Orange/Speaker-wavLM-pro" if args.use_wpro else None,
        roberta_model_name="roberta-base",
        hidden_dim=256,
        freeze_wavlm=True,
        freeze_roberta=True
    ).to(device)
    
    model = load_checkpoint(model, args.checkpoint)
    print("✓ Model loaded")
    
    # Get test data for this fold
    print(f"\nPreparing test data for Fold {args.fold}")
    test_df = get_fold_test_df(args.csv, args.fold)
    print(f"Test samples: {len(test_df)}")
    
    test_dataset = IEMOCAPDataset(
        test_df,
        audio_max_length=16000*10,
        sample_rate=16000,
        normalize_av=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=4,
        shuffle=False,
        num_workers=2,
        collate_fn=collate_fn
    )
    
    # Evaluate
    print("\nEvaluating...")
    results = evaluate(model, test_loader, device)
    
    # Print results
    print(f"\n{'='*60}")
    print(f"FOLD {args.fold} TEST RESULTS")
    print(f"{'='*60}")
    print(f"Accuracy:     {results['accuracy']:.4f}")
    print(f"F1 Macro:     {results['f1_macro']:.4f}")
    print(f"F1 Weighted:  {results['f1_weighted']:.4f}")
    print(f"Arousal CCC:  {results['arousal_ccc']:.4f}")
    print(f"Valence CCC:  {results['valence_ccc']:.4f}")
    print(f"Avg CCC:      {results['avg_ccc']:.4f}")
    print(f"{'='*60}")
    
    # Print classification report
    print("\nClassification Report:")
    emotion_names = ['Happy', 'Sad', 'Neutral', 'Angry']
    print(classification_report(results['labels'], results['predictions'], 
                                target_names=emotion_names, digits=4))


if __name__ == '__main__':
    main()
