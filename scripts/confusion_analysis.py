#!/usr/bin/env python3
"""
Confusion-matrix analysis for IEMOCAP (5-fold) comparing checkpoints.

Usage (dry-run, prints plan):
  python scripts/confusion_analysis.py --model_dirs checkpoints/method2_kfold checkpoints/method2_kfold_wpro

To run actual inference (may require GPU and model weights):
  python scripts/confusion_analysis.py --model_dirs checkpoints/method2_kfold checkpoints/method2_kfold_wpro --run --device cuda:0

Outputs:
  - saved confusion matrices and summary CSV in `logs/confusion_analysis/`
"""
import os
import sys
import argparse
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

def parse_args():
    parser = argparse.ArgumentParser(description='Confusion matrix analysis (IEMOCAP 5-fold)')
    parser.add_argument('--model_dirs', nargs='+', required=True,
                        help='One or more checkpoint directories containing fold_*/best_model.pt')
    parser.add_argument('--csv', type=str, default='datasets/IEMOCAP_KFOLD/tenfolds/all.csv')
    parser.add_argument('--folds', nargs='+', type=int, default=[1,2,3,4,5])
    parser.add_argument('--output_dir', type=str, default='logs/confusion_analysis')
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--run', action='store_true', help='Execute full inference (may be slow)')
    return parser.parse_args()


def find_checkpoints(model_dir, folds):
    paths = []
    for f in folds:
        p = Path(model_dir) / f'fold_{f}' / 'best_model.pt'
        if p.exists():
            paths.append(str(p))
    return paths


def dry_run_report(model_dirs, csv_path, folds, out_dir):
    print('\nDry-run: planned actions')
    print('CSV:', csv_path)
    print('Folds:', folds)
    print('Models:')
    for md in model_dirs:
        print(' -', md)
        cps = find_checkpoints(md, folds)
        print(f'   Found {len(cps)} checkpoint(s); example: {cps[0] if cps else "(none)"}')
    print('\nTo run real inference, re-run with --run --device cuda:0')
    print('Results will be saved to', out_dir)


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Simple dry-run to inspect available checkpoints
    if not args.run:
        dry_run_report(args.model_dirs, args.csv, args.folds, out_dir)
        return

    # Full run: Import heavy modules only when requested
    import torch
    from torch.utils.data import DataLoader
    from tqdm import tqdm
    from sklearn.metrics import confusion_matrix

    # Local imports
    from data.iemocap_dataset import IEMOCAPDataset, collate_fn as iemocap_collate_fn
    from models.method2_crossattn_av import Method2CrossAttentionAV

    def _safe_torch_load(path):
        try:
            return torch.load(path, map_location='cpu', weights_only=False)
        except TypeError:
            return torch.load(path, map_location='cpu')
        except Exception:
            return torch.load(path, map_location='cpu')

    def _get_state_dict(obj):
        # Accept various checkpoint layouts and return raw state_dict
        if isinstance(obj, dict):
            if 'model_state_dict' in obj:
                return obj['model_state_dict']
            if 'state_dict' in obj:
                return obj['state_dict']
        return obj

    def _inspect_state_dict(state):
        """Inspect state_dict to detect prosody input dim and av input dim (2 vs 3)
        Returns (prosody_input_dim_or_None, av_input_dim_or_None)
        """
        prosody_in = None
        av_in = None
        # prosody_projection.weight shape is [hidden_dim, wavlm_dim]
        k = 'prosody_projection.weight'
        if k in state:
            t = state[k]
            try:
                prosody_in = int(t.shape[1])
            except Exception:
                prosody_in = None

        # av_refinement first linear weight is at 'av_refinement.mlp.0.weight'
        k2 = 'av_refinement.mlp.0.weight'
        if k2 in state:
            t2 = state[k2]
            try:
                av_in = int(t2.shape[1])
            except Exception:
                av_in = None

        return prosody_in, av_in

    EMOTION_MAP = {0: 'Happy', 1: 'Sad', 2: 'Neutral', 3: 'Angry'}

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    summary_rows = []
    # For each model dir, aggregate confusion across folds
    for md in args.model_dirs:
        print('\nProcessing model_dir:', md)
        agg_cm = np.zeros((4,4), dtype=int)
        total_samples = 0

        for f in args.folds:
            ckpt = Path(md) / f'fold_{f}' / 'best_model.pt'
            if not ckpt.exists():
                print('  Skipping missing checkpoint:', ckpt)
                continue

            # Load model with W-PRO if present in name
            use_wpro = 'wpro' in md.lower() or 'wpro' in str(ckpt).lower()
            wavlm_prosody = "Orange/Speaker-wavLM-pro" if use_wpro else None

            print('  Loading checkpoint:', ckpt.name, ' use_wpro=', use_wpro)
            # Load checkpoint state first to inspect shapes and decide model config
            raw = _safe_torch_load(str(ckpt))
            state = _get_state_dict(raw)
            prosody_in_dim, av_input_dim = _inspect_state_dict(state)

            # Decide whether to use W-PRO based on checkpoint prosody input dim
            if prosody_in_dim is not None and prosody_in_dim >= 1000:
                wavlm_prosody = "Orange/Speaker-wavLM-pro"
                print(f'   Checkpoint prosody input_dim={prosody_in_dim} -> using W-PRO')
            else:
                wavlm_prosody = None

            # Instantiate model with matching prosody backbone when possible
            try:
                model = Method2CrossAttentionAV(
                    num_emotions=4,
                    wavlm_model_name='microsoft/wavlm-base-plus',
                    wavlm_prosody_model_name=wavlm_prosody,
                    roberta_model_name='roberta-base',
                    hidden_dim=256,
                    freeze_wavlm=True,
                    freeze_roberta=True
                ).to(device)
            except RuntimeError as e:
                # Potential OOM during .to(device); retry on CPU
                print('   Warning: model.to(device) failed, retrying on CPU:', e)
                device_cpu = torch.device('cpu')
                model = Method2CrossAttentionAV(
                    num_emotions=4,
                    wavlm_model_name='microsoft/wavlm-base-plus',
                    wavlm_prosody_model_name=wavlm_prosody,
                    roberta_model_name='roberta-base',
                    hidden_dim=256,
                    freeze_wavlm=True,
                    freeze_roberta=True
                ).to(device_cpu)
                device = device_cpu

            # state already loaded above as 'state'
            if state is None:
                raw = _safe_torch_load(str(ckpt))
                state = _get_state_dict(raw)

            # Handle AV-dimension mismatch by slicing checkpoint weights if needed
            if av_input_dim is not None and av_input_dim != 2:
                print(f'   Checkpoint AV input dim = {av_input_dim}; adapting to 2')
                # av_refinement.mlp.0.weight shape is [hidden_dim, av_input_dim]
                w_key = 'av_refinement.mlp.0.weight'
                rp_key = 'av_refinement.residual_proj.weight'
                if w_key in state:
                    w = state[w_key]
                    if w.ndim == 2 and w.shape[1] >= 2:
                        state[w_key] = w[:, :2].clone()
                if rp_key in state:
                    rp = state[rp_key]
                    if rp.ndim == 2 and rp.shape[1] >= 2:
                        state[rp_key] = rp[:, :2].clone()

            # Finally attempt to load state dict
            try:
                model.load_state_dict(state, strict=False)
            except Exception as e:
                print('   Warning: state_dict load issue:', e)

            # Prepare test df for this fold
            df = pd.read_csv(args.csv)
            max_fold = df['fold'].max()
            if max_fold == 10:
                fold_to_session = {1:1,2:1,3:2,4:2,5:3,6:3,7:4,8:4,9:5,10:5}
                df['session'] = df['fold'].map(fold_to_session)
                test_df = df[df['session'] == f].reset_index(drop=True)
            else:
                test_df = df[df['fold'] == f].reset_index(drop=True)

            dataset = IEMOCAPDataset(
                test_df,
                audio_max_length=16000*10,
                sample_rate=16000,
                normalize_av=True,
                tokenizer_name='roberta-base',
                max_text_length=128
            )
            # Use single-process DataLoader to avoid worker crashes on limited servers
            loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                                num_workers=0, collate_fn=iemocap_collate_fn)

            # Inference (catch errors per-fold so other folds/models continue)
            try:
                model.eval()
                with torch.no_grad():
                    for batch in tqdm(loader, desc=f'Fold {f}', leave=False):
                        # Move tensors safely (handle CPU/GPU device mismatch)
                        audio = batch['audio'].to(device)
                        text_ids = batch['text_input_ids'].to(device)
                        text_mask = batch['text_attention_mask'].to(device)
                        try:
                            outputs = model(audio, text_ids, text_attention_mask=text_mask)
                        except RuntimeError as e:
                            # Likely OOM; try moving model to CPU and re-run this batch
                            if 'out of memory' in str(e).lower():
                                print('    OOM during forward; retrying on CPU for this fold')
                                torch.cuda.empty_cache()
                                model_cpu = model.to(torch.device('cpu'))
                                device = torch.device('cpu')
                                outputs = model_cpu(audio.cpu(), text_ids.cpu(), text_attention_mask=text_mask.cpu())
                                model = model_cpu
                            else:
                                raise
                        preds = torch.argmax(outputs['emotion_logits'], dim=1).cpu().numpy()
                        trues = batch['emotion_label'].numpy()
                        cm = confusion_matrix(trues, preds, labels=[0,1,2,3])
                        agg_cm += cm
                        total_samples += len(trues)
            except Exception as e:
                import traceback
                print(f"  ✗ Inference failed for fold {f}: {e}")
                traceback.print_exc()
                # Skip this fold and continue
                del model
                torch.cuda.empty_cache()
                continue

            del model
            torch.cuda.empty_cache()

        # Save aggregate confusion matrix
        if total_samples == 0:
            print('No samples processed for', md)
            continue

        cm_path = out_dir / f"confusion_{Path(md).name}.npy"
        np.save(str(cm_path), agg_cm)
        print(' Saved confusion matrix to', cm_path)

        # Compute normalized confusion (row-normalized)
        row_sums = agg_cm.sum(axis=1, keepdims=True)
        norm_cm = agg_cm.astype(float) / (row_sums + 1e-8)
        # Extract target confusions of interest
        # Angry (3) predicted as Happy (0), Happy (0) predicted as Angry (3)
        angry_as_happy = norm_cm[3,0]
        happy_as_angry = norm_cm[0,3]

        summary_rows.append({
            'model_dir': md,
            'total_samples': int(total_samples),
            'angry_as_happy_rate': float(angry_as_happy),
            'happy_as_angry_rate': float(happy_as_angry),
            'confusion_matrix_path': str(cm_path)
        })

    # Save summary
    df_sum = pd.DataFrame(summary_rows)
    summary_csv = out_dir / 'confusion_summary.csv'
    df_sum.to_csv(summary_csv, index=False)
    print('\nSummary saved to', summary_csv)

    # If we have exactly two models, compute percent reduction
    if len(summary_rows) == 2:
        a = summary_rows[0]
        b = summary_rows[1]
        def pct_reduction(x,y):
            if x == 0:
                return np.nan
            return float((x - y) / x * 100.0)
        red_angry = pct_reduction(a['angry_as_happy_rate'], b['angry_as_happy_rate'])
        red_happy = pct_reduction(a['happy_as_angry_rate'], b['happy_as_angry_rate'])
        print(f"\nPercent reduction (model {Path(args.model_dirs[0]).name} -> {Path(args.model_dirs[1]).name}):")
        print(f"  Angry->Happy reduction: {red_angry:.2f}%")
        print(f"  Happy->Angry reduction: {red_happy:.2f}%")


if __name__ == '__main__':
    main()
