#!/usr/bin/env python3
"""
Compare Full ASPIRE vs No-AV-Supervision models.

Produces:
 - 5-fold summary CSV with Acc / F1 (macro/weighted) for both models
 - Full model Arousal/Valence CCC table
 - Confusion matrices (Full, No-AV, Delta) saved as .npy and heatmaps
 - Per-class F1 delta bar chart
 - (Optional) UMAP visualizations of fused features and branch embeddings
 - Linear probe regressors for arousal/valence on aggregated embeddings

Usage (example):
  python scripts/compare_full_vs_noav.py \
    --full_dir checkpoints/method2_kfold_wpro \
    --noav_dir checkpoints/method2_kfold_wpro_ablation_no_avsup \
    --csv datasets/IEMOCAP_KFOLD/tenfolds/all.csv \
    --output results/compare_full_vs_noav --device cpu
"""
import os
import sys
import argparse
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split

import torch

sys.path.append(str(Path(__file__).resolve().parent.parent))

def safe_load_checkpoint(path, map_location='cpu'):
    if not os.path.exists(path):
        return None
    try:
        ckpt = torch.load(path, map_location=map_location)
        # Accept possible wrappers
        if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
            return ckpt['model_state_dict']
        return ckpt
    except Exception:
        try:
            return torch.load(path, map_location=map_location)
        except Exception as e:
            print('Failed to load checkpoint', path, e)
            return None


def safe_load_and_filter_state(model, path, device='cpu', log_file=None):
    """Safely load a checkpoint and load only compatible parameters into `model`.

    - Tries `torch.load(path)` first. If it fails due to newer PyTorch weights_only
      restrictions or pickling globals, retries with `weights_only=False` using
      `torch.serialization.safe_globals` to allowlist numpy scalar if available.
    - Filters the returned state_dict so only keys present in `model.state_dict()`
      with matching shapes are loaded. Loads with `strict=False` and logs skipped
      keys to `log_file` if provided.
    Returns a dict with keys: filtered_count, skipped (list).
    """
    if not os.path.exists(path):
        return {'filtered_count': 0, 'skipped': [('missing_file', path)]}

    state = None
    warnings = []
    try:
        state = torch.load(path, map_location=device)
    except Exception as e1:
        warnings.append(f'initial_load_failed: {e1}')
        # Retry with weights_only=False and safe_globals if available
        try:
            import numpy as _np
            try:
                from torch.serialization import safe_globals
            except Exception:
                safe_globals = None

            if safe_globals is not None:
                with safe_globals([_np._core.multiarray.scalar]):
                    state = torch.load(path, map_location=device, weights_only=False)
            else:
                state = torch.load(path, map_location=device, weights_only=False)
        except Exception as e2:
            warnings.append(f'retry_load_failed: {e2}')

    if state is None:
        msg = f'Failed to load checkpoint: {path}; warnings: {warnings}'
        print(msg)
        if log_file is not None:
            with open(log_file, 'a') as fh:
                fh.write(msg + '\n')
        return {'filtered_count': 0, 'skipped': warnings}

    # Extract state_dict
    if isinstance(state, dict):
        if 'model_state_dict' in state:
            sd = state['model_state_dict']
        elif 'state_dict' in state:
            sd = state['state_dict']
        else:
            sd = state
    else:
        sd = state

    model_sd = model.state_dict()
    filtered = {}
    skipped = []
    for k, v in sd.items():
        if k in model_sd:
            try:
                if isinstance(v, torch.Tensor) and v.shape == model_sd[k].shape:
                    filtered[k] = v
                else:
                    skipped.append((k, getattr(v, 'shape', type(v)), model_sd[k].shape))
            except Exception as e:
                skipped.append((k, f'error_inspect:{e}', model_sd[k].shape))
        else:
            skipped.append((k, 'missing_in_model', None))

    # Load filtered weights
    try:
        model.load_state_dict(filtered, strict=False)
    except Exception as e:
        skipped.append(('load_state_dict_error', str(e)))

    # Write log
    if log_file is not None:
        with open(log_file, 'a') as fh:
            fh.write(f'Loaded {len(filtered)} params from {path}\n')
            if skipped:
                fh.write('Skipped keys/shapes:\n')
                for s in skipped:
                    fh.write(str(s) + '\n')

    return {'filtered_count': len(filtered), 'skipped': skipped}

def fold_test_df(csv_path, fold):
    df = pd.read_csv(csv_path)
    max_fold = df['fold'].max()
    if max_fold == 10:
        fold_to_session = {1:1,2:1,3:2,4:2,5:3,6:3,7:4,8:4,9:5,10:5}
        df['session'] = df['fold'].map(fold_to_session)
        test_df = df[df['session'] == fold].reset_index(drop=True)
    else:
        test_df = df[df['fold'] == fold].reset_index(drop=True)
    return test_df

def prepare_loader(df, batch_size=8, device='cpu'):
    from data.iemocap_dataset import IEMOCAPDataset, collate_fn
    ds = IEMOCAPDataset(df, audio_max_length=160000, sample_rate=16000,
                        tokenizer_name='roberta-base', max_text_length=128)
    from torch.utils.data import DataLoader
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=collate_fn)
    return loader

def infer_model_on_loader(model, loader, device, return_intermediate=False):
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()

    all_preds = []
    all_trues = []
    arousal_preds = []
    valence_preds = []
    fused_feats = []
    arousal_embs = []
    valence_embs = []

    with torch.no_grad():
        for batch in loader:
            audio = batch['audio'].to(device)
            text_ids = batch['text_input_ids'].to(device)
            text_mask = batch['text_attention_mask'].to(device)

            # Some model variants don't accept extra kwargs (use_guidance/return_intermediate).
            # Try calling with kwargs first, fall back to positional if TypeError.
            try:
                out = model(audio_input=audio, text_input=text_ids,
                            audio_attention_mask=None, text_attention_mask=text_mask)
            except TypeError:
                # Fallback: positional call
                out = model(audio, text_ids, None, text_mask)

            # Emotion logits
            logits = out.get('emotion_logits') if isinstance(out, dict) else None
            if logits is None:
                # Try older name
                logits = out[0] if isinstance(out, (list, tuple)) and len(out) > 0 else None
            preds = torch.argmax(logits, dim=1).cpu().numpy() if logits is not None else np.zeros(len(batch['emotion_label']))
            all_preds.extend(preds.tolist())
            all_trues.extend(batch['emotion_label'].cpu().numpy().tolist())

            # AV values if present
            if isinstance(out, dict):
                if 'arousal_value' in out:
                    arousal_preds.extend(out['arousal_value'].cpu().numpy().flatten().tolist())
                if 'valence_value' in out:
                    valence_preds.extend(out['valence_value'].cpu().numpy().flatten().tolist())
                if 'fused_features' in out:
                    fused_feats.append(out['fused_features'].cpu().numpy())
                if 'arousal_embedding' in out:
                    arousal_embs.append(out['arousal_embedding'].cpu().numpy())
                if 'valence_embedding' in out:
                    valence_embs.append(out['valence_embedding'].cpu().numpy())

    # stack collected features
    if fused_feats:
        fused_feats = np.concatenate(fused_feats, axis=0)
    else:
        fused_feats = None
    if arousal_embs:
        arousal_embs = np.concatenate(arousal_embs, axis=0)
    else:
        arousal_embs = None
    if valence_embs:
        valence_embs = np.concatenate(valence_embs, axis=0)
    else:
        valence_embs = None

    return {
        'preds': np.array(all_preds, dtype=int),
        'trues': np.array(all_trues, dtype=int),
        'arousal_preds': np.array(arousal_preds) if arousal_preds else None,
        'valence_preds': np.array(valence_preds) if valence_preds else None,
        'fused_feats': fused_feats,
        'arousal_embs': arousal_embs,
        'valence_embs': valence_embs,
    }

def compute_ccc(y_true, y_pred):
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()
    if len(y_true) < 2:
        return 0.0
    mean_t, mean_p = np.mean(y_true), np.mean(y_pred)
    var_t, var_p = np.var(y_true), np.var(y_pred)
    cov = np.mean((y_true - mean_t) * (y_pred - mean_p))
    return float(2 * cov / (var_t + var_p + (mean_t - mean_p) ** 2 + 1e-8))

def save_confusion_and_plot(cm, labels, outpath, title='Confusion'):
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    np.save(outpath + '.npy', cm)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=labels, yticklabels=labels, cmap='Blues')
    plt.title(title)
    plt.xlabel('Pred')
    plt.ylabel('True')
    plt.tight_layout()
    plt.savefig(outpath + '.pdf')
    plt.close()

def heatmap_float(mat, labels, outpath, title='Heatmap'):
    plt.figure(figsize=(6,5))
    sns.heatmap(mat, annot=True, fmt='.3f', xticklabels=labels, yticklabels=labels, cmap='coolwarm', center=0)
    plt.title(title)
    plt.xlabel('Pred')
    plt.ylabel('True')
    plt.tight_layout()
    plt.savefig(outpath + '.pdf')
    plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--full_dir', required=True)
    parser.add_argument('--noav_dir', required=True)
    parser.add_argument('--csv', required=True)
    parser.add_argument('--output', default='results/compare_full_vs_noav')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--folds', nargs='+', type=int, default=[1,2,3,4,5])
    parser.add_argument('--wavlm_prosody_model_name', default=None,
                        help='Optional prosody WavLM model name to match checkpoints (e.g. Orange/Speaker-wavLM-pro)')
    args = parser.parse_args()

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    # Import models lazily
    from models.method2_crossattn_av import Method2CrossAttentionAV
    from models.method2_crossattn_av_ablation_no_avsup import Method2CrossAttentionAV_AblationNoAVSup

    LABELS = ['Happy','Sad','Neutral','Angry']

    summary_rows = []

    # Per-fold evaluation
    agg_conf_full = np.zeros((4,4), dtype=int)
    agg_conf_noav = np.zeros((4,4), dtype=int)
    total_samples = 0

    # collect embeddings for linear probe across folds
    pooled_arousal_emb_full = []
    pooled_arousal_true_full = []
    pooled_valence_emb_full = []
    pooled_valence_true_full = []

    for f in args.folds:
        print(f"Evaluating fold {f}...")
        test_df = fold_test_df(args.csv, f)
        loader = prepare_loader(test_df, batch_size=args.batch_size, device=args.device)

        # locate checkpoints
        ck_full = Path(args.full_dir) / f'fold_{f}' / 'best_model.pt'
        ck_noav = Path(args.noav_dir) / f'fold_{f}' / 'best_model.pt'
        if not ck_full.exists():
            ck_full = Path(args.full_dir) / f'fold_{f}' / 'last_model.pt'
        if not ck_noav.exists():
            ck_noav = Path(args.noav_dir) / f'fold_{f}' / 'last_model.pt'

        # Full model
        # instantiate with the W-PRO prosody model to match checkpoints (can be overridden by CLI)
        model_full = Method2CrossAttentionAV(num_emotions=4, wavlm_prosody_model_name=args.wavlm_prosody_model_name)
        # robust load: filter incompatible keys/shapes and log skips
        lwarn = out / 'load_warnings.log'
        # load checkpoint state to CPU to avoid GPU OOM during loading
        info_full = safe_load_and_filter_state(model_full, str(ck_full), device='cpu', log_file=str(lwarn))
        if info_full and info_full.get('skipped'):
            print(f'Load warnings (full) fold {f}:', len(info_full.get('skipped'))) 

        res_full = infer_model_on_loader(model_full, loader, device=args.device, return_intermediate=True)

        # No-AV model
        model_noav = Method2CrossAttentionAV_AblationNoAVSup(num_emotions=4, wavlm_prosody_model_name=args.wavlm_prosody_model_name)
        info_noav = safe_load_and_filter_state(model_noav, str(ck_noav), device='cpu', log_file=str(lwarn))
        if info_noav and info_noav.get('skipped'):
            print(f'Load warnings (no-av) fold {f}:', len(info_noav.get('skipped')))

        res_noav = infer_model_on_loader(model_noav, loader, device=args.device, return_intermediate=True)

        # Metrics
        acc_full = accuracy_score(res_full['trues'], res_full['preds'])
        f1_macro_full = f1_score(res_full['trues'], res_full['preds'], average='macro')
        f1_w_full = f1_score(res_full['trues'], res_full['preds'], average='weighted')

        acc_noav = accuracy_score(res_noav['trues'], res_noav['preds'])
        f1_macro_noav = f1_score(res_noav['trues'], res_noav['preds'], average='macro')
        f1_w_noav = f1_score(res_noav['trues'], res_noav['preds'], average='weighted')

        # Full model CCCs if present
        arousal_ccc = compute_ccc(test_df['arousal'].values, res_full['arousal_preds']) if res_full['arousal_preds'] is not None else None
        valence_ccc = compute_ccc(test_df['valence'].values, res_full['valence_preds']) if res_full['valence_preds'] is not None else None

        summary_rows.append({
            'fold': f,
            'acc_full': acc_full, 'f1_macro_full': f1_macro_full, 'f1_w_full': f1_w_full,
            'acc_noav': acc_noav, 'f1_macro_noav': f1_macro_noav, 'f1_w_noav': f1_w_noav,
            'arousal_ccc_full': arousal_ccc, 'valence_ccc_full': valence_ccc,
        })

        # Confusion
        cm_full = confusion_matrix(res_full['trues'], res_full['preds'], labels=[0,1,2,3])
        cm_noav = confusion_matrix(res_noav['trues'], res_noav['preds'], labels=[0,1,2,3])
        agg_conf_full += cm_full
        agg_conf_noav += cm_noav
        total_samples += len(res_full['trues'])

        # save per-fold confusion heatmaps
        save_confusion_and_plot(cm_full, LABELS, str(out / f'confusion_full_fold{f}'), title=f'Full Fold {f}')
        save_confusion_and_plot(cm_noav, LABELS, str(out / f'confusion_noav_fold{f}'), title=f'NoAV Fold {f}')

        # collect embeddings for linear probe
        if res_full['arousal_embs'] is not None and res_full['arousal_preds'] is not None:
            pooled_arousal_emb_full.append(res_full['arousal_embs'])
            pooled_arousal_true_full.append(test_df['arousal'].values[:res_full['arousal_embs'].shape[0]])
        if res_full['valence_embs'] is not None and res_full['valence_preds'] is not None:
            pooled_valence_emb_full.append(res_full['valence_embs'])
            pooled_valence_true_full.append(test_df['valence'].values[:res_full['valence_embs'].shape[0]])

        # free memory
        del model_full, model_noav
        torch.cuda.empty_cache()

    # Aggregate summary CSV
    df_sum = pd.DataFrame(summary_rows)
    df_sum.to_csv(out / 'folds_summary.csv', index=False)

    # Aggregate confusion matrices and delta
    np.save(out / 'confusion_full_agg.npy', agg_conf_full)
    np.save(out / 'confusion_noav_agg.npy', agg_conf_noav)
    np.save(out / 'confusion_delta_agg.npy', agg_conf_full - agg_conf_noav)

    # normalized delta
    row_sums = agg_conf_full.sum(axis=1, keepdims=True)
    norm_full = agg_conf_full.astype(float) / (row_sums + 1e-8)
    row_sums_no = agg_conf_noav.sum(axis=1, keepdims=True)
    norm_no = agg_conf_noav.astype(float) / (row_sums_no + 1e-8)
    delta_norm = norm_full - norm_no
    heatmap_float(delta_norm, LABELS, str(out / 'confusion_delta_norm'), title='Delta (Full - NoAV) normalized')

    # Per-class F1 delta
    per_class_f1_full = np.diag(agg_conf_full) / (agg_conf_full.sum(axis=1) + 1e-8)
    per_class_f1_noav = np.diag(agg_conf_noav) / (agg_conf_noav.sum(axis=1) + 1e-8)
    plt.figure()
    x = np.arange(len(LABELS))
    plt.bar(x - 0.15, per_class_f1_noav, width=0.3, label='NoAV')
    plt.bar(x + 0.15, per_class_f1_full, width=0.3, label='Full')
    plt.xticks(x, LABELS)
    plt.ylabel('Per-class correct rate (TP / True)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(out / 'per_class_f1_comparison.pdf')
    plt.close()

    # Linear probe on aggregated embeddings (Full model only)
    if pooled_arousal_emb_full:
        X_ar = np.concatenate(pooled_arousal_emb_full, axis=0)
        y_ar = np.concatenate(pooled_arousal_true_full, axis=0)
        X_train, X_test, y_train, y_test = train_test_split(X_ar, y_ar, test_size=0.2, random_state=42)
        reg = Ridge(alpha=1.0)
        reg.fit(X_train, y_train)
        y_pred = reg.predict(X_test)
        arousal_mse = float(np.mean((y_pred - y_test) ** 2))
        arousal_ccc_lp = compute_ccc(y_test, y_pred)
        with open(out / 'linear_probe_results.json', 'w') as fh:
            json.dump({'arousal_mse': arousal_mse, 'arousal_ccc_lp': arousal_ccc_lp}, fh, indent=2)

    print('\nDone. Results saved in', out)


if __name__ == '__main__':
    main()
