#!/usr/bin/env python3
"""Evaluate best checkpoints for W-PRO emotion-only across folds and print Acc and Macro F1."""
import os
import sys
import argparse
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import torch
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader

from models.wpro_emotion_only import WProEmotionOnly
from data.iemocap_dataset import IEMOCAPDataset, collate_fn


def prepare_folds(csv_path):
    df = pd.read_csv(csv_path)
    fold_to_session = {1:1,2:1,3:2,4:2,5:3,6:3,7:4,8:4,9:5,10:5}
    if 'fold' in df.columns:
        maxf = df['fold'].max()
        if maxf == 10:
            df['session'] = df['fold'].map(fold_to_session)
        else:
            df['session'] = df['fold']
    else:
        raise RuntimeError('CSV missing fold column')

    folds = []
    for session in range(1, 6):
        test_df = df[df['session'] == session].reset_index(drop=True)
        train_val_df = df[df['session'] != session].reset_index(drop=True)
        folds.append({'train_val': train_val_df, 'test': test_df, 'session': session})
    return folds


def evaluate_one_fold(checkpoint_path, fold_data, cfg, device):
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(checkpoint_path)

    test_df = fold_data['test']
    test_dataset = IEMOCAPDataset(test_df.reset_index(drop=True), sample_rate=cfg.get('data', {}).get('sample_rate', 16000), audio_max_length=cfg.get('data', {}).get('max_audio_length', 16000*10))
    test_loader = DataLoader(test_dataset, batch_size=cfg.get('training', {}).get('batch_size', 8), shuffle=False, num_workers=cfg.get('training', {}).get('num_workers', 2), collate_fn=collate_fn)

    model = WProEmotionOnly(num_emotions=cfg.get('model', {}).get('num_emotions', 4), wpro_model_name=cfg.get('model', {}).get('wpro_model_name', 'Orange/Speaker-wavLM-pro'), hidden_dim=cfg.get('model', {}).get('hidden_dim', 256), freeze_wpro=cfg.get('model', {}).get('freeze_wpro', True))
    model.to(device)

    ckpt = torch.load(checkpoint_path, map_location=device)
    # checkpoint may be dict with model_state_dict or direct state_dict
    if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
        state = ckpt['model_state_dict']
    else:
        state = ckpt
    model.load_state_dict(state)
    model.eval()

    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in test_loader:
            audio = batch['audio'].to(device)
            labels = batch['emotion_label'].to(device)
            out = model(audio)
            logits = out['emotion_logits']
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    f1_macro = f1_score(all_labels, all_preds, average='macro')
    return acc, f1_macro


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoints_dir', default='checkpoints/method2_kfold_wpro_emotion_only_bs8')
    parser.add_argument('--csv', default='datasets/IEMOCAP_KFOLD/tenfolds/all.csv')
    parser.add_argument('--config', default=None, help='Optional yaml config to read training params')
    parser.add_argument('--device', default='cuda:1')
    args = parser.parse_args()

    # minimal cfg
    cfg = {
        'data': {'sample_rate': 16000, 'max_audio_length': 16000*10},
        'training': {'batch_size': 8, 'num_workers': 2},
        'model': {'num_emotions':4}
    }
    # try to read config if provided
    if args.config and os.path.exists(args.config):
        import yaml
        with open(args.config, 'r') as f:
            cfg = yaml.safe_load(f)

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    folds = prepare_folds(args.csv)

    per_fold = []
    for i in range(1, 6):
        # support folder name patterns
        p1 = os.path.join(args.checkpoints_dir, f'fold{i}', 'checkpoint_best.pt')
        p2 = os.path.join(args.checkpoints_dir, f'fold_{i}', 'checkpoint_best.pt')
        p3 = os.path.join(args.checkpoints_dir, f'fold{i}', 'checkpoint_best.pt')
        p4 = os.path.join(args.checkpoints_dir, f'fold_{i}', 'checkpoint_best.pt')
        ckpt = None
        for p in (p1, p2, p3, p4):
            if os.path.exists(p):
                ckpt = p
                break
        if ckpt is None:
            print(f"Fold {i}: no best checkpoint found in {args.checkpoints_dir}")
            continue
        print(f"Evaluating Fold {i} using {ckpt}")
        acc, f1 = evaluate_one_fold(ckpt, folds[i-1], cfg, device)
        print(f"Fold {i} -> Acc: {acc:.4f}, F1_macro: {f1:.4f}\n")
        per_fold.append({'fold': i, 'acc': acc, 'f1': f1})

    # compute averages
    if per_fold:
        avg_acc = np.mean([x['acc'] for x in per_fold])
        avg_f1 = np.mean([x['f1'] for x in per_fold])
        print(f"Average Acc (evaluated folds): {avg_acc:.4f}")
        print(f"Average F1_macro (evaluated folds): {avg_f1:.4f}")


if __name__ == '__main__':
    main()
