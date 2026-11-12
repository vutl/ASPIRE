#!/usr/bin/env python3
"""
Evaluate Method2 W-PRO checkpoints per-fold.

Usage examples:
  python3 scripts/infer_method2_wpro_checkpoints.py \
    --checkpoints_dir checkpoints/method2_kfold_wpro \
    --config configs/method2_wpro_config.yaml \
    --folds 3 4 5 \
    --which best last all \
    --device cuda:1

Use `--dry_run` to only print the planned actions.
"""
import argparse
import os
from pathlib import Path
import yaml
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))

from models.method2_crossattn_av import Method2CrossAttentionAV
from train.losses import CombinedEmotionLossCCC
from data.iemocap_dataset import IEMOCAPDataset, collate_fn
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def compute_ccc(y_true, y_pred):
    mean_true = np.mean(y_true)
    mean_pred = np.mean(y_pred)
    var_true = np.var(y_true)
    var_pred = np.var(y_pred)
    covariance = np.mean((y_true - mean_true) * (y_pred - mean_pred))
    ccc = (2 * covariance) / (var_true + var_pred + (mean_true - mean_pred) ** 2 + 1e-8)
    return ccc


def prepare_folds(csv_path):
    df = pd.read_csv(csv_path)
    max_fold = df['fold'].max()
    if max_fold == 10:
        fold_to_session = {1:1,2:1,3:2,4:2,5:3,6:3,7:4,8:4,9:5,10:5}
        df['session'] = df['fold'].map(fold_to_session)
    else:
        df['session'] = df['fold']

    folds = []
    for s in range(1,6):
        folds.append({'session': s, 'test': df[df['session']==s], 'train_val': df[df['session']!=s]})
    return folds


def test_checkpoint(model, ckpt_path, test_loader, device):
    # load safely
    torch.serialization.add_safe_globals([np._core.multiarray.scalar])
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    # some checkpoints include extra profiling keys (total_ops/total_params).
    # load with strict=False to ignore unexpected keys.
    model.load_state_dict(ckpt['model_state_dict'], strict=False)
    model.to(device)
    model.eval()

    # Test both full and masked guidance
    results = {}
    
    for guidance_mode in ['full', 'masked']:
        use_guidance = (guidance_mode == 'full')
        
        all_preds = []
        all_labels = []
        all_arousal_pred = []
        all_arousal_true = []
        all_valence_pred = []
        all_valence_true = []

        with torch.no_grad():
            for batch in test_loader:
                audio = batch['audio'].to(device)
                text_input_ids = batch['text_input_ids'].to(device)
                text_attention_mask = batch['text_attention_mask'].to(device)
                emotion_label = batch['emotion_label'].to(device)
                arousal = batch['arousal_value'].to(device)
                valence = batch['valence_value'].to(device)

                outputs = model(audio_input=audio, text_input=text_input_ids, 
                               text_attention_mask=text_attention_mask, use_guidance=use_guidance)

                preds = torch.argmax(outputs['emotion_logits'], dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(emotion_label.cpu().numpy())

                all_arousal_pred.extend(outputs['arousal_value'].cpu().numpy())
                all_arousal_true.extend(arousal.cpu().numpy())
                all_valence_pred.extend(outputs['valence_value'].cpu().numpy())
                all_valence_true.extend(valence.cpu().numpy())

        accuracy = accuracy_score(all_labels, all_preds)
        precision_macro = precision_score(all_labels, all_preds, average='macro', zero_division=0)
        recall_macro = recall_score(all_labels, all_preds, average='macro', zero_division=0)
        f1_macro = f1_score(all_labels, all_preds, average='macro')
        f1_weighted = f1_score(all_labels, all_preds, average='weighted')

        all_arousal_pred = np.array(all_arousal_pred).flatten()
        all_arousal_true = np.array(all_arousal_true).flatten()
        all_valence_pred = np.array(all_valence_pred).flatten()
        all_valence_true = np.array(all_valence_true).flatten()

        arousal_ccc = compute_ccc(all_arousal_true, all_arousal_pred)
        valence_ccc = compute_ccc(all_valence_true, all_valence_pred)
        avg_ccc = (arousal_ccc + valence_ccc) / 2.0

        results[guidance_mode] = {
            'accuracy': float(accuracy),
            'precision_macro': float(precision_macro),
            'recall_macro': float(recall_macro),
            'f1_macro': float(f1_macro),
            'f1_weighted': float(f1_weighted),
            'arousal_ccc': float(arousal_ccc),
            'valence_ccc': float(valence_ccc),
            'avg_ccc': float(avg_ccc)
        }

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoints_dir', type=str, required=True)
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--csv', type=str, required=True)
    parser.add_argument('--folds', type=int, nargs='+', default=[1,2,3,4,5])
    parser.add_argument('--which', type=str, choices=['best','last','all'], default='best')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--dry_run', action='store_true')
    parser.add_argument('--out_csv', type=str, default='logs/inference_results_method2_wpro.csv')
    args = parser.parse_args()

    cfg = load_config(args.config)
    folds = prepare_folds(args.csv)

    results = []
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    for fold_idx in args.folds:
        if fold_idx < 1 or fold_idx > 5:
            print(f"Skipping invalid fold {fold_idx}")
            continue

        fold_data = folds[fold_idx-1]
        test_df = fold_data['test']

        test_dataset = IEMOCAPDataset(test_df, audio_max_length=cfg.get('data',{}).get('max_audio_length',160000), tokenizer_name=cfg.get('model',{}).get('roberta_model_name','roberta-base'), max_text_length=cfg.get('data',{}).get('max_text_length',128))
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2, collate_fn=collate_fn, pin_memory=True)

        # instantiate model from config
        model = Method2CrossAttentionAV(
            num_emotions=cfg['model'].get('num_emotions',4),
            wavlm_model_name=cfg['model'].get('wavlm_model_name'),
            wavlm_prosody_model_name=cfg['model'].get('wavlm_prosody_model_name'),
            roberta_model_name=cfg['model'].get('roberta_model_name','roberta-base'),
            hidden_dim=cfg['model'].get('hidden_dim',256),
            av_refinement_layers=cfg['model'].get('av_refinement_layers',2),
            freeze_wavlm=cfg['model'].get('freeze_wavlm',False),
            freeze_roberta=cfg['model'].get('freeze_roberta',False),
            num_cross_attn_heads=cfg['model'].get('num_cross_attn_heads',8),
            post_attention_layers=cfg['model'].get('post_attention_layers',2),
            av_prediction_layers=cfg['model'].get('av_prediction_layers',3)
        )

        ckpt_folder = Path(args.checkpoints_dir) / f'fold_{fold_idx}'
        if not ckpt_folder.exists():
            print(f"Fold folder not found: {ckpt_folder}, skipping")
            continue

        ckpt_candidates = []
        if args.which in ('best','all'):
            p = ckpt_folder / 'best_model.pt'
            if p.exists():
                ckpt_candidates.append(('best_model.pt', str(p)))
        if args.which in ('last','all'):
            p = ckpt_folder / 'last_model.pt'
            if p.exists():
                ckpt_candidates.append(('last_model.pt', str(p)))
        if args.which == 'all':
            for p in ckpt_folder.glob('*.pt'):
                name = p.name
                if name not in ('best_model.pt','last_model.pt'):
                    ckpt_candidates.append((name, str(p)))

        if len(ckpt_candidates) == 0:
            print(f"No checkpoints found for fold {fold_idx} with --which {args.which}")
            continue

        for name, ckpt_path in ckpt_candidates:
            print(f"Fold {fold_idx} -> evaluating checkpoint: {ckpt_path}")
            if args.dry_run:
                results.append({'fold': fold_idx, 'checkpoint': name, 'mode': 'full', 'accuracy': None, 'f1_macro': None, 'avg_ccc': None})
                results.append({'fold': fold_idx, 'checkpoint': name, 'mode': 'masked', 'accuracy': None, 'f1_macro': None, 'avg_ccc': None})
                continue

            metrics_both = test_checkpoint(model, ckpt_path, test_loader, device)
            
            # Save results for both modes
            for mode in ['full', 'masked']:
                row = {'fold': fold_idx, 'checkpoint': name, 'mode': mode}
                row.update(metrics_both[mode])
                results.append(row)
            
            # lightweight save after each checkpoint
            pd.DataFrame(results).to_csv(args.out_csv, index=False)

    print(f"Saved inference results to {args.out_csv}")


if __name__ == '__main__':
    main()
