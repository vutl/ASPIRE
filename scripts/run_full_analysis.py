#!/usr/bin/env python3
"""
Run end-to-end comparison: safe checkpoint load, per-fold inference (Full vs No-AV),
save per-fold embeddings, aggregate, run UMAP + t-SNE, and linear probes.

Usage (example):
  nohup python3 scripts/run_full_analysis.py \
    --full_dir checkpoints/method2_kfold_wpro \
    --noav_dir checkpoints/method2_kfold_wpro_ablation_no_avsup \
    --csv datasets/IEMOCAP_KFOLD/all.csv \
    --folds 1 2 3 4 5 --device cuda:1 --batch_size 4 \
    --wavlm_prosody_model_name Orange/Speaker-wavLM-pro \
    --output results/compare_full_vs_noav > logs/run_full_analysis.out 2>&1 &
"""
import os
import argparse
from pathlib import Path
import json
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split

try:
    from umap import UMAP
    _HAS_UMAP = True
except Exception:
    from sklearn.manifold import TSNE
    _HAS_UMAP = False

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))

from data.iemocap_dataset import IEMOCAPDataset, collate_fn
from models.method2_crossattn_av import Method2CrossAttentionAV
from models.method2_crossattn_av_ablation_no_avsup import Method2CrossAttentionAV_AblationNoAVSup


def safe_load_checkpoint_to_model(model, ckpt_path, log_file=None):
    """Load checkpoint to model robustly: load on CPU, filter keys by shape, load strict=False."""
    if not os.path.exists(ckpt_path):
        return {'loaded': 0, 'skipped': [('missing_file', ckpt_path)]}
    # add safe globals for numpy scalars (match infer script)
    try:
        torch.serialization.add_safe_globals([np._core.multiarray.scalar])
    except Exception:
        pass
    state = None
    try:
        state = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    except Exception as e:
        try:
            state = torch.load(ckpt_path, map_location='cpu')
        except Exception as e2:
            if log_file:
                with open(log_file,'a') as fh:
                    fh.write(f'Failed to load {ckpt_path}: {e} | {e2}\n')
            return {'loaded': 0, 'skipped': [('load_failed', str(e))]}

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
                    skipped.append((k, getattr(v,'shape',type(v)), model_sd[k].shape))
            except Exception as e:
                skipped.append((k, f'err:{e}', model_sd[k].shape))
        else:
            skipped.append((k, 'missing_in_model', None))

    try:
        model.load_state_dict(filtered, strict=False)
    except Exception as e:
        skipped.append(('load_state_error', str(e)))

    if log_file:
        with open(log_file,'a') as fh:
            fh.write(f'Loaded {len(filtered)} params from {ckpt_path}\n')
            if skipped:
                fh.write('Skipped keys:\n')
                for s in skipped:
                    fh.write(str(s)+'\n')

    return {'loaded': len(filtered), 'skipped': skipped}


def prepare_loader(df, batch_size=8):
    ds = IEMOCAPDataset(df, audio_max_length=160000, sample_rate=16000,
                        tokenizer_name='roberta-base', max_text_length=128)
    from torch.utils.data import DataLoader
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=2, collate_fn=collate_fn)
    return loader


def infer_collect(model, loader, device, return_intermediate=True, use_guidance=True):
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()

    fused_list = []
    final_list = []
    arousal_list = []
    valence_list = []
    preds = []
    trues = []

    with torch.no_grad():
        for batch in loader:
            audio = batch['audio'].to(device)
            text = batch['text_input_ids'].to(device)
            text_mask = batch['text_attention_mask'].to(device)
            try:
                out = model(audio_input=audio, text_input=text, text_attention_mask=text_mask, use_guidance=use_guidance, return_intermediate=return_intermediate)
            except TypeError:
                out = model(audio, text, None, text_mask, return_intermediate=return_intermediate)

            # outputs
            if isinstance(out, dict):
                if 'fused_features' in out and out['fused_features'] is not None:
                    fused_list.append(out['fused_features'].cpu().numpy())
                if 'final_representation' in out and out['final_representation'] is not None:
                    final_list.append(out['final_representation'].cpu().numpy())
                if 'arousal_embedding' in out and out['arousal_embedding'] is not None:
                    arousal_list.append(out['arousal_embedding'].cpu().numpy())
                if 'valence_embedding' in out and out['valence_embedding'] is not None:
                    valence_list.append(out['valence_embedding'].cpu().numpy())
                if 'emotion_logits' in out and out['emotion_logits'] is not None:
                    p = torch.argmax(out['emotion_logits'], dim=1).cpu().numpy()
                    preds.extend(p.tolist())
            else:
                # fallback: assume model returns tuple (logits, ...)
                logits = out[0]
                p = torch.argmax(logits, dim=1).cpu().numpy()
                preds.extend(p.tolist())

            trues.extend(batch['emotion_label'].numpy().tolist())

    def cat(lst):
        return np.concatenate(lst, axis=0) if lst else None

    return {
        'preds': np.array(preds, dtype=int) if preds else None,
        'trues': np.array(trues, dtype=int) if trues else None,
        'fused': cat(fused_list),
        'final': cat(final_list),
        'arousal': cat(arousal_list),
        'valence': cat(valence_list),
    }


def compute_ccc(y_true, y_pred):
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()
    if y_true.size < 2:
        return float('nan')
    mean_t, mean_p = y_true.mean(), y_pred.mean()
    var_t, var_p = y_true.var(), y_pred.var()
    cov = np.mean((y_true-mean_t)*(y_pred-mean_p))
    return float(2*cov / (var_t + var_p + (mean_t-mean_p)**2 + 1e-8))


def run_umap_or_tsne(X, method='umap', random_state=42):
    if X is None: return None
    if method == 'umap' and _HAS_UMAP:
        reducer = UMAP(n_components=2, random_state=random_state)
        return reducer.fit_transform(X)
    else:
        from sklearn.manifold import TSNE
        tsne = TSNE(n_components=2, random_state=random_state)
        return tsne.fit_transform(X)


def plot_scatter(Z, color, outpath, discrete=True, title=None):
    plt.figure(figsize=(6,5))
    if discrete:
        sns.scatterplot(x=Z[:,0], y=Z[:,1], hue=color, palette='tab10', s=8, linewidth=0)
        plt.legend(bbox_to_anchor=(1.05,1), loc='upper left')
    else:
        plt.scatter(Z[:,0], Z[:,1], c=color, cmap='RdYlBu', s=8)
        plt.colorbar()
    if title:
        plt.title(title)
    plt.tight_layout()
    plt.savefig(str(outpath) + '.pdf')
    plt.close()


def linear_probe_eval(X, y, outprefix):
    if X is None or y is None: return None
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    reg = Ridge(alpha=1.0)
    reg.fit(X_train, y_train)
    y_pred = reg.predict(X_test)
    mse = float(((y_pred - y_test)**2).mean())
    ccc = compute_ccc(y_test, y_pred)
    # scatter
    plt.figure(figsize=(5,5))
    plt.scatter(y_test, y_pred, s=6, alpha=0.6)
    mn = min(y_test.min(), y_pred.min())
    mx = max(y_test.max(), y_pred.max())
    plt.plot([mn,mx],[mn,mx],'k--',linewidth=0.8)
    plt.xlabel('True')
    plt.ylabel('Pred')
    plt.title(f'{outprefix} (MSE={mse:.4f}, CCC={ccc:.4f})')
    plt.tight_layout()
    plt.savefig(str(outprefix) + '.pdf')
    plt.close()
    return {'mse': mse, 'ccc': ccc}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--full_dir', required=True)
    p.add_argument('--noav_dir', required=True)
    p.add_argument('--csv', required=True)
    p.add_argument('--folds', nargs='+', type=int, default=[1,2,3,4,5])
    p.add_argument('--device', default='cuda:1')
    p.add_argument('--batch_size', type=int, default=4)
    p.add_argument('--output', default='results/compare_full_vs_noav')
    p.add_argument('--wavlm_prosody_model_name', default=None)
    args = p.parse_args()

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    logf = out / 'run_full_analysis.log'

    df_all = pd.read_csv(args.csv)

    LABELS = ['Happy','Sad','Neutral','Angry']

    for f in args.folds:
        print(f'Fold {f}:')
        # select fold df (session mapping if 10-fold csv)
        if df_all['fold'].max() == 10:
            fold_to_session = {1:1,2:1,3:2,4:2,5:3,6:3,7:4,8:4,9:5,10:5}
            test_df = df_all[df_all['fold'].map(fold_to_session)==f].reset_index(drop=True)
        else:
            test_df = df_all[df_all['fold']==f].reset_index(drop=True)

        loader = prepare_loader(test_df, batch_size=args.batch_size)

        # checkpoint paths
        ck_full = Path(args.full_dir) / f'fold_{f}' / 'best_model.pt'
        ck_noav = Path(args.noav_dir) / f'fold_{f}' / 'best_model.pt'

        # instantiate models with prosody option
        model_full = Method2CrossAttentionAV(num_emotions=4, wavlm_prosody_model_name=args.wavlm_prosody_model_name)
        safe_load_checkpoint_to_model(model_full, str(ck_full), log_file=str(logf))
        res_full = infer_collect(model_full, loader, device=args.device, return_intermediate=True, use_guidance=True)
        # save per-fold arrays
        if res_full['fused'] is not None:
            np.save(out / f'fused_full_fold{f}.npy', res_full['fused'])
        if res_full['final'] is not None:
            np.save(out / f'final_full_fold{f}.npy', res_full['final'])
        if res_full['arousal'] is not None:
            np.save(out / f'arousal_full_fold{f}.npy', res_full['arousal'])
        if res_full['valence'] is not None:
            np.save(out / f'valence_full_fold{f}.npy', res_full['valence'])

        model_noav = Method2CrossAttentionAV_AblationNoAVSup(num_emotions=4, wavlm_prosody_model_name=args.wavlm_prosody_model_name)
        safe_load_checkpoint_to_model(model_noav, str(ck_noav), log_file=str(logf))
        res_noav = infer_collect(model_noav, loader, device=args.device, return_intermediate=True, use_guidance=False)
        if res_noav['fused'] is not None:
            np.save(out / f'fused_noav_fold{f}.npy', res_noav['fused'])
        if res_noav['final'] is not None:
            np.save(out / f'final_noav_fold{f}.npy', res_noav['final'])
        if res_noav['arousal'] is not None:
            np.save(out / f'arousal_noav_fold{f}.npy', res_noav['arousal'])
        if res_noav['valence'] is not None:
            np.save(out / f'valence_noav_fold{f}.npy', res_noav['valence'])

        # small cleanup
        del model_full, model_noav
        torch.cuda.empty_cache()

    # Aggregate arrays across folds
    def agg(prefix):
        arrs = []
        for f in args.folds:
            p = out / f'{prefix}_fold{f}.npy'
            if p.exists():
                arrs.append(np.load(p))
        return np.concatenate(arrs, axis=0) if arrs else None

    fused_full = agg('fused_full')
    fused_noav = agg('fused_noav')
    final_full = agg('final_full')
    final_noav = agg('final_noav')
    ar_full = agg('arousal_full')
    ar_noav = agg('arousal_noav')
    va_full = agg('valence_full')
    va_noav = agg('valence_noav')

    # UMAP + t-SNE and plotting (use same random_state)
    random_state = 42
    # fused features, color by emotion label (aggregate labels)
    labels = []
    for f in args.folds:
        if df_all['fold'].max() == 10:
            fold_to_session = {1:1,2:1,3:2,4:2,5:3,6:3,7:4,8:4,9:5,10:5}
            df_fold = df_all[df_all['fold'].map(fold_to_session)==f]
        else:
            df_fold = df_all[df_all['fold']==f]
        labels.extend(df_fold['emotion'].values.tolist())
    labels = np.array(labels, dtype=int)

    def maybe_plot(X, name, discrete=True, method='umap'):
        if X is None:
            return
        Z = run_umap_or_tsne(X, method=method, random_state=random_state)
        if Z is None:
            return
        color = labels if discrete else None
        plot_scatter(Z, color if color is not None else np.zeros(Z.shape[0]), out / f'{method}_{name}', discrete=discrete, title=f'{method.upper()} {name}')

    maybe_plot(fused_full, 'fused_full', discrete=True, method='umap')
    maybe_plot(fused_noav, 'fused_noav', discrete=True, method='umap')
    maybe_plot(final_full, 'final_full', discrete=True, method='umap')
    maybe_plot(final_noav, 'final_noav', discrete=True, method='umap')
    maybe_plot(ar_full, 'arousal_full', discrete=False, method='umap')
    maybe_plot(ar_noav, 'arousal_noav', discrete=False, method='umap')
    maybe_plot(va_full, 'valence_full', discrete=False, method='umap')
    maybe_plot(va_noav, 'valence_noav', discrete=False, method='umap')

    # also t-SNE
    for m in ['fused_full','fused_noav','arousal_full','arousal_noav','valence_full','valence_noav']:
        X = locals().get(m)
        maybe_plot(X, m, discrete=(m.startswith('fused')), method='tsne')

    # Linear probes
    probe_results = {}

    probe_results['arousal_full'] = linear_probe_eval(ar_full, None if ar_full is None else np.loadtxt('') if False else (None), out / 'probe_arousal_full')
    # Instead of trying to read labels from file again, align using aggregated label arrays
    # Build aggregated arousal/valence arrays from csv
    ar_labels = []
    va_labels = []
    for f in args.folds:
        if df_all['fold'].max() == 10:
            fold_to_session = {1:1,2:1,3:2,4:2,5:3,6:3,7:4,8:4,9:5,10:5}
            df_fold = df_all[df_all['fold'].map(fold_to_session)==f]
        else:
            df_fold = df_all[df_all['fold']==f]
        ar_labels.extend(df_fold['arousal'].values.tolist())
        va_labels.extend(df_fold['valence'].values.tolist())
    ar_labels = np.array(ar_labels)
    va_labels = np.array(va_labels)

    # align and run probes
    def align_and_probe(X, y, prefix):
        if X is None or y is None:
            return None
        n = min(len(X), len(y))
        return linear_probe_eval(X[:n], y[:n], out / prefix)

    probe_results['arousal_full'] = align_and_probe(ar_full, ar_labels, 'probe_arousal_full')
    probe_results['arousal_noav'] = align_and_probe(ar_noav, ar_labels, 'probe_arousal_noav')
    probe_results['valence_full'] = align_and_probe(va_full, va_labels, 'probe_valence_full')
    probe_results['valence_noav'] = align_and_probe(va_noav, va_labels, 'probe_valence_noav')

    with open(out / 'probe_results.json','w') as fh:
        json.dump(probe_results, fh, indent=2)

    print('All done. Results in', out)


if __name__ == '__main__':
    main()
