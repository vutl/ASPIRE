#!/usr/bin/env python3
"""
Run inference to collect fused/final/arousal/valence embeddings, run UMAP and linear probes.

Outputs (in --output dir):
 - per-fold .npy: fused_full_fold{f}.npy, arousal_full_fold{f}.npy, valence_full_fold{f}.npy and _noav
 - aggregated npy: fused_full.npy, fused_noav.npy, arousal_full.npy, valence_full.npy
 - UMAP PDF plots for fused features and arousal/valence embeddings
 - probe_results.json with MSE and CCC
"""
import os
import sys
from pathlib import Path
import argparse
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

sys.path.append(str(Path(__file__).resolve().parent.parent))

from data.iemocap_dataset import IEMOCAPDataset, collate_fn
from models.method2_crossattn_av import Method2CrossAttentionAV
from models.method2_crossattn_av_ablation_no_avsup import Method2CrossAttentionAV_AblationNoAVSup


def safe_load_and_filter_state(model, path, device='cpu', log_file=None):
    if not os.path.exists(path):
        return {'filtered_count': 0, 'skipped': [('missing_file', path)]}
    state = None
    warnings = []
    try:
        state = torch.load(path, map_location=device)
    except Exception as e1:
        warnings.append(f'initial_load_failed: {e1}')
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
        if log_file:
            with open(log_file,'a') as fh:
                fh.write(msg + '\n')
        return {'filtered_count': 0, 'skipped': warnings}
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
                skipped.append((k, f'error:{e}', model_sd[k].shape))
        else:
            skipped.append((k, 'missing_in_model', None))
    try:
        model.load_state_dict(filtered, strict=False)
    except Exception as e:
        skipped.append(('load_state_error', str(e)))
    if log_file:
        with open(log_file,'a') as fh:
            fh.write(f'Loaded {len(filtered)} params from {path}\n')
            if skipped:
                fh.write('Skipped keys:\n')
                for s in skipped:
                    fh.write(str(s)+'\n')
    return {'filtered_count': len(filtered), 'skipped': skipped}


def prepare_loader(df, batch_size=8):
    ds = IEMOCAPDataset(df, audio_max_length=160000, sample_rate=16000,
                        tokenizer_name='roberta-base', max_text_length=128)
    from torch.utils.data import DataLoader
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=2, collate_fn=collate_fn)
    return loader


def infer_collect(model, loader, device):
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    fused_list = []
    final_list = []
    arousal_list = []
    valence_list = []
    trues = []
    with torch.no_grad():
        for batch in loader:
            audio = batch['audio'].to(device)
            text = batch['text_input_ids'].to(device)
            text_mask = batch['text_attention_mask'].to(device)
            try:
                out = model(audio_input=audio, text_input=text, text_attention_mask=text_mask, return_intermediate=True)
            except TypeError:
                out = model(audio, text, None, text_mask, return_intermediate=True)
            fused = out.get('fused_features') if isinstance(out, dict) else None
            final = out.get('final_representation') if isinstance(out, dict) else None
            ar_emb = out.get('arousal_embedding') if isinstance(out, dict) else None
            va_emb = out.get('valence_embedding') if isinstance(out, dict) else None
            if fused is not None:
                fused_list.append(fused.cpu().numpy())
            if final is not None:
                final_list.append(final.cpu().numpy())
            if ar_emb is not None:
                arousal_list.append(ar_emb.cpu().numpy())
            if va_emb is not None:
                valence_list.append(va_emb.cpu().numpy())
            trues.extend(batch['emotion_label'].numpy().tolist())
    def cat_or_none(lst):
        return np.concatenate(lst, axis=0) if lst else None
    return {
        'fused': cat_or_none(fused_list),
        'final': cat_or_none(final_list),
        'arousal': cat_or_none(arousal_list),
        'valence': cat_or_none(valence_list),
        'trues': np.array(trues, dtype=int)
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


def run_umap(X, n_components=2, random_state=42):
    if X is None or X.shape[0] == 0:
        return None
    if _HAS_UMAP:
        reducer = UMAP(n_components=n_components, random_state=random_state)
        Z = reducer.fit_transform(X)
    else:
        tsne = TSNE(n_components=n_components, random_state=random_state)
        Z = tsne.fit_transform(X)
    return Z


def plot_scatter(Z, labels, title, outpath, cmap='tab10'):
    plt.figure(figsize=(6,5))
    sns.scatterplot(x=Z[:,0], y=Z[:,1], hue=labels, palette='tab10', s=10, linewidth=0)
    plt.title(title)
    plt.legend(bbox_to_anchor=(1.05,1), loc='upper left')
    plt.tight_layout()
    plt.savefig(outpath + '.pdf')
    plt.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--full_dir', required=True)
    p.add_argument('--noav_dir', required=True)
    p.add_argument('--csv', required=True)
    p.add_argument('--folds', nargs='+', type=int, default=[1,2,3,4,5])
    p.add_argument('--device', default='cuda:1')
    p.add_argument('--batch_size', type=int, default=8)
    p.add_argument('--output', default='results/compare_full_vs_noav')
    p.add_argument('--wavlm_prosody_model_name', default=None)
    args = p.parse_args()

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    logf = out / 'umap_probes.log'

    df_all = pd.read_csv(args.csv)

    for f in args.folds:
        print(f'Processing fold {f}...')
        if df_all['fold'].max() == 10:
            fold_to_session = {1:1,2:1,3:2,4:2,5:3,6:3,7:4,8:4,9:5,10:5}
            test_df = df_all[df_all['fold'].map(fold_to_session)==f].reset_index(drop=True)
        else:
            test_df = df_all[df_all['fold']==f].reset_index(drop=True)
        loader = prepare_loader(test_df, batch_size=args.batch_size)

        ck_full = Path(args.full_dir)/f'fold_{f}'/'best_model.pt'
        ck_noav = Path(args.noav_dir)/f'fold_{f}'/'best_model.pt'

        model_full = Method2CrossAttentionAV(num_emotions=4, wavlm_prosody_model_name=args.wavlm_prosody_model_name)
        safe_load_and_filter_state(model_full, str(ck_full), device='cpu', log_file=str(logf))
        res_full = infer_collect(model_full, loader, device=args.device)
        if res_full['fused'] is not None:
            np.save(out / f'fused_full_fold{f}.npy', res_full['fused'])
        if res_full['arousal'] is not None:
            np.save(out / f'arousal_full_fold{f}.npy', res_full['arousal'])
        if res_full['valence'] is not None:
            np.save(out / f'valence_full_fold{f}.npy', res_full['valence'])

        model_noav = Method2CrossAttentionAV_AblationNoAVSup(num_emotions=4, wavlm_prosody_model_name=args.wavlm_prosody_model_name)
        safe_load_and_filter_state(model_noav, str(ck_noav), device='cpu', log_file=str(logf))
        res_noav = infer_collect(model_noav, loader, device=args.device)
        if res_noav['fused'] is not None:
            np.save(out / f'fused_noav_fold{f}.npy', res_noav['fused'])
        if res_noav['arousal'] is not None:
            np.save(out / f'arousal_noav_fold{f}.npy', res_noav['arousal'])
        if res_noav['valence'] is not None:
            np.save(out / f'valence_noav_fold{f}.npy', res_noav['valence'])

        del model_full, model_noav
        torch.cuda.empty_cache()

    def agg(prefix):
        arrs = []
        for f in args.folds:
            p = out / f'{prefix}_fold{f}.npy'
            if p.exists():
                arrs.append(np.load(p))
        return np.concatenate(arrs, axis=0) if arrs else None

    fused_full = agg('fused_full')
    fused_noav = agg('fused_noav')
    ar_full = agg('arousal_full')
    ar_noav = agg('arousal_noav')
    va_full = agg('valence_full')
    va_noav = agg('valence_noav')

    if fused_full is not None:
        np.save(out / 'fused_full.npy', fused_full)
    if fused_noav is not None:
        np.save(out / 'fused_noav.npy', fused_noav)
    if ar_full is not None:
        np.save(out / 'arousal_full.npy', ar_full)
    if va_full is not None:
        np.save(out / 'valence_full.npy', va_full)

    labels = []
    ar_labels = []
    va_labels = []
    for f in args.folds:
        if df_all['fold'].max() == 10:
            fold_to_session = {1:1,2:1,3:2,4:2,5:3,6:3,7:4,8:4,9:5,10:5}
            df_fold = df_all[df_all['fold'].map(fold_to_session)==f]
        else:
            df_fold = df_all[df_all['fold']==f]
        labels.extend(df_fold['emotion'].values.tolist())
        ar_labels.extend(df_fold['arousal'].values.tolist())
        va_labels.extend(df_fold['valence'].values.tolist())

    labels = np.array(labels, dtype=int)
    ar_labels = np.array(ar_labels)
    va_labels = np.array(va_labels)

    def do_umap_and_plot(X, name, color_values, discrete=True):
        if X is None:
            print(f'Skipping {name}, no data')
            return
        Z = run_umap(X)
        if Z is None:
            print('UMAP/TSNE failed for', name)
            return
        outpath = out / f'umap_{name}'
        if discrete:
            plot_scatter(Z, color_values, f'UMAP {name}', str(outpath))
        else:
            plt.figure(figsize=(6,5))
            plt.scatter(Z[:,0], Z[:,1], c=color_values, cmap='RdYlBu', s=8)
            plt.colorbar()
            plt.title(f'UMAP {name}')
            plt.tight_layout()
            plt.savefig(str(outpath) + '.pdf')
            plt.close()

    do_umap_and_plot(fused_full, 'fused_full', labels, discrete=True)
    do_umap_and_plot(fused_noav, 'fused_noav', labels, discrete=True)
    do_umap_and_plot(ar_full, 'arousal_full', ar_labels, discrete=False)
    do_umap_and_plot(ar_noav, 'arousal_noav', ar_labels, discrete=False)
    do_umap_and_plot(va_full, 'valence_full', va_labels, discrete=False)
    do_umap_and_plot(va_noav, 'valence_noav', va_labels, discrete=False)

    probe_results = {}
    def probe(X_full, X_noav, y):
        if X_full is None or y is None:
            return None
        Xf_train, Xf_test, yf_train, yf_test = train_test_split(X_full, y, test_size=0.2, random_state=42)
        regf = Ridge(alpha=1.0)
        regf.fit(Xf_train, yf_train)
        yf_pred = regf.predict(Xf_test)
        mse_f = float(np.mean((yf_pred - yf_test)**2))
        ccc_f = compute_ccc(yf_test, yf_pred)
        res = {'mse': mse_f, 'ccc': ccc_f}
        if X_noav is not None:
            Xn_train, Xn_test, yn_train, yn_test = train_test_split(X_noav, y, test_size=0.2, random_state=42)
            regn = Ridge(alpha=1.0)
            regn.fit(Xn_train, yn_train)
            yn_pred = regn.predict(Xn_test)
            mse_n = float(np.mean((yn_pred - yn_test)**2))
            ccc_n = compute_ccc(yn_test, yn_pred)
            res.update({'mse_noav': mse_n, 'ccc_noav': ccc_n})
        return res

    probe_ar = probe(ar_full, ar_noav, ar_labels)
    probe_va = probe(va_full, va_noav, va_labels)
    probe_results['arousal_probe'] = probe_ar
    probe_results['valence_probe'] = probe_va

    with open(out / 'probe_results.json', 'w') as fh:
        json.dump(probe_results, fh, indent=2)

    print('UMAP and probes finished. Results in', out)


if __name__ == '__main__':
    main()
