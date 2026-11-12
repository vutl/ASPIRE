#!/usr/bin/env python3
"""
Evaluate checkpoints and plot Arousal–Valence scatter colored by predicted emotions.
Supports:
- IEMOCAP 5-fold (standard, W-PRO, W-PRO-ASR, W-PRO-AVD)
- MSP-Podcast fixed splits (standard, W-PRO)

Output figures are saved under outputs/av_scatter/<model_name>/.
"""
import os
import argparse
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score

# Local imports
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))

from data.iemocap_dataset import IEMOCAPDataset, collate_fn as iemocap_collate_fn
from data.msp_podcast_dataset import MSPPodcastDataset, collate_fn as msp_collate_fn
from models.method2_crossattn_av import Method2CrossAttentionAV
from models.method2_crossattn_avd import Method2CrossAttentionAVD

# ---------- Plot settings ----------
# FIXED: Match IEMOCAPDataset mapping: {'hap': 0, 'sad': 1, 'neu': 2, 'ang': 3}
EMOTION_NAMES = {0: 'Happy', 1: 'Sad', 2: 'Neutral', 3: 'Angry'}
EMOTION_COLORS = {
    'Happy': '#2ecc71',     # Green
    'Sad': '#3498db',       # Blue
    'Neutral': '#95a5a6',   # Gray
    'Angry': '#e74c3c'      # Red
}


def load_state(model, ckpt_path):
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
        state_dict = ckpt['model_state_dict']
    elif isinstance(ckpt, dict) and 'state_dict' in ckpt:
        state_dict = ckpt['state_dict']
    else:
        state_dict = ckpt
    state_dict = {k: v for k, v in state_dict.items() if not k.endswith(('total_ops', 'total_params'))}
    model.load_state_dict(state_dict, strict=False)
    return model


def get_iemocap_test_df(csv_path, fold_id, asr_column=None):
    df = pd.read_csv(csv_path)
    max_fold = df['fold'].max()
    if max_fold == 10:
        fold_to_session = {1:1,2:1,3:2,4:2,5:3,6:3,7:4,8:4,9:5,10:5}
        df['session'] = df['fold'].map(fold_to_session)
        session_id = fold_id
        test_df = df[df['session'] == session_id].reset_index(drop=True)
    else:
        test_df = df[df['fold'] == fold_id].reset_index(drop=True)
    if asr_column is not None and asr_column in test_df.columns:
        test_df['transcript'] = test_df[asr_column].fillna('')
    return test_df


def evaluate_av(model, loader, device):
    model.eval()
    preds, a_pred, v_pred = [], [], []
    true_labels, a_true, v_true = [], [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc='  Inference', leave=False):
            audio = batch['audio'].to(device)
            text_input_ids = batch['text_input_ids'].to(device)
            text_attention_mask = batch['text_attention_mask'].to(device)
            outputs = model(audio, text_input_ids, text_attention_mask=text_attention_mask)
            preds.extend(torch.argmax(outputs['emotion_logits'], dim=1).cpu().numpy())
            a_pred.extend(outputs['arousal_value'].cpu().numpy().flatten())
            v_pred.extend(outputs['valence_value'].cpu().numpy().flatten())

            # collect GT if present in batch
            if 'emotion_label' in batch:
                true_labels.extend(batch['emotion_label'].cpu().numpy())
            if 'arousal_value' in batch:
                a_true.extend(batch['arousal_value'].cpu().numpy().flatten())
            if 'valence_value' in batch:
                v_true.extend(batch['valence_value'].cpu().numpy().flatten())

    return np.array(preds), np.array(a_pred), np.array(v_pred), (np.array(true_labels) if len(true_labels) else None), (np.array(a_true) if len(a_true) else None), (np.array(v_true) if len(v_true) else None)


def evaluate_avd(model, loader, device):
    model.eval()
    preds, a_pred, v_pred = [], [], []
    true_labels, a_true, v_true = [], [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc='  Inference', leave=False):
            audio = batch['audio'].to(device)
            text_input_ids = batch['text_input_ids'].to(device)
            text_attention_mask = batch['text_attention_mask'].to(device)
            outputs = model(audio, text_input_ids, text_attention_mask=text_attention_mask)
            preds.extend(torch.argmax(outputs['emotion_logits'], dim=1).cpu().numpy())
            a_pred.extend(outputs['arousal_value'].cpu().numpy().flatten())
            v_pred.extend(outputs['valence_value'].cpu().numpy().flatten())

            # collect GT if present in batch
            if 'emotion_label' in batch:
                true_labels.extend(batch['emotion_label'].cpu().numpy())
            if 'arousal_value' in batch:
                a_true.extend(batch['arousal_value'].cpu().numpy().flatten())
            if 'valence_value' in batch:
                v_true.extend(batch['valence_value'].cpu().numpy().flatten())

    return np.array(preds), np.array(a_pred), np.array(v_pred), (np.array(true_labels) if len(true_labels) else None), (np.array(a_true) if len(a_true) else None), (np.array(v_true) if len(v_true) else None)


def plot_scatter(arousal, valence, emotion_preds, title, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    labels = [EMOTION_NAMES.get(int(e), 'Unknown') for e in emotion_preds]
    all_emotions = [EMOTION_NAMES[i] for i in sorted(EMOTION_NAMES.keys())]

    # Print class counts so collapsed classes are obvious
    counts = {emo: 0 for emo in all_emotions}
    for emo in labels:
        counts[emo] = counts.get(emo, 0) + 1
    print("  Pred counts:", counts)

    plt.figure(figsize=(6, 6))
    for emo in all_emotions:
        mask = np.array([e == emo for e in labels])
        label_text = f"{emo} ({mask.sum()})"
        # Even if a class has zero predictions, keep it in the legend for clarity
        plt.scatter(arousal[mask], valence[mask], label=label_text, alpha=0.6, s=40,
                    color=EMOTION_COLORS.get(emo, '#888888'), edgecolors='black', linewidth=0.4)
    plt.xlabel('Arousal (predicted, normalized to [0,1])')
    plt.ylabel('Valence (predicted, normalized to [0,1])')
    plt.title(title)
    plt.xlim(-0.05, 1.05)
    plt.ylim(-0.05, 1.05)
    plt.grid(True, alpha=0.3)
    plt.legend(loc='best', fontsize=9)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {save_path}")


def compute_ccc(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_true.size == 0 or y_pred.size == 0:
        return float('nan')
    mean_true = np.mean(y_true)
    mean_pred = np.mean(y_pred)
    var_true = np.var(y_true)
    var_pred = np.var(y_pred)
    covariance = np.mean((y_true - mean_true) * (y_pred - mean_pred))
    ccc = (2 * covariance) / (var_true + var_pred + (mean_true - mean_pred) ** 2 + 1e-8)
    return float(ccc)


def run_iemocap_av(model_name, checkpoint_dir, csv_path, device, asr_column=None, use_wpro=False):
    print(f"\n=== {model_name} (IEMOCAP AV) ===")
    for fold in range(1, 6):
        ckpt_path = Path(checkpoint_dir) / f"fold_{fold}" / "best_model.pt"
        if not ckpt_path.exists():
            print(f"  ✗ Missing {ckpt_path}")
            continue
        print(f"[Fold {fold}] loading checkpoint: {ckpt_path}")
        model = Method2CrossAttentionAV(
            num_emotions=4,
            wavlm_model_name="microsoft/wavlm-base-plus",
            wavlm_prosody_model_name="Orange/Speaker-wavLM-pro" if use_wpro else None,
            roberta_model_name="roberta-base",
            hidden_dim=256,
            freeze_wavlm=True,
            freeze_roberta=True
        ).to(device)
        model = load_state(model, ckpt_path)

        test_df = get_iemocap_test_df(csv_path, fold, asr_column=asr_column)
        dataset = IEMOCAPDataset(test_df, audio_max_length=16000*10, sample_rate=16000, normalize_av=True)
        loader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=2, collate_fn=iemocap_collate_fn)
        print(f"  Samples: {len(dataset)}")

        preds, a_pred, v_pred, true_labels, a_true, v_true = evaluate_av(model, loader, device)

        # Plot predictions (map to [0,1] for plotting convenience)
        out_dir = f"outputs/av_scatter/{model_name}"
        os.makedirs(out_dir, exist_ok=True)
        save_path = os.path.join(out_dir, f"fold_{fold}.png")
        # For plotting we normalize predictions to [0,1]
        if a_pred.size:
            a_plot = (a_pred - a_pred.min()) / (a_pred.max() - a_pred.min() + 1e-8)
            v_plot = (v_pred - v_pred.min()) / (v_pred.max() - v_pred.min() + 1e-8)
        else:
            a_plot, v_plot = a_pred, v_pred
        plot_scatter(a_plot, v_plot, preds, f"{model_name} Fold {fold}", save_path)

        # Compute metrics and log
        log_dir = Path('logs') / 'eval_av_scatter'
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / f"{model_name}_fold{fold}.log"
        with open(log_file, 'w') as f:
            f.write(f"Model: {model_name} \nFold: {fold} \nSamples: {len(dataset)}\n")
            if true_labels is not None and len(true_labels):
                acc = accuracy_score(true_labels, preds)
                f1_macro = f1_score(true_labels, preds, average='macro')
                f1_weighted = f1_score(true_labels, preds, average='weighted')
                f.write(f"acc={acc:.4f} | f1_macro={f1_macro:.4f} | f1_weighted={f1_weighted:.4f}\n")
            else:
                f.write("No emotion ground-truth available.\n")

            if a_true is not None and v_true is not None and a_true.size and v_true.size:
                a_ccc = compute_ccc(a_true, a_pred)
                v_ccc = compute_ccc(v_true, v_pred)
                avg_ccc = (a_ccc + v_ccc) / 2.0
                f.write(f"arousal_ccc={a_ccc:.4f} | valence_ccc={v_ccc:.4f} | avg_ccc={avg_ccc:.4f}\n")
            else:
                f.write("No A/V ground-truth available.\n")
        print(f"  ✓ Logged metrics to {log_file}")

        del model
        torch.cuda.empty_cache()


def run_iemocap_avd(model_name, checkpoint_dir, csv_path, device):
    print(f"\n=== {model_name} (IEMOCAP AVD) ===")
    for fold in range(1, 6):
        ckpt_path = Path(checkpoint_dir) / f"fold_{fold}" / "best_model.pt"
        if not ckpt_path.exists():
            print(f"  ✗ Missing {ckpt_path}")
            continue
        print(f"[Fold {fold}] loading checkpoint: {ckpt_path}")
        model = Method2CrossAttentionAVD(
            num_emotions=4,
            wavlm_model_name="microsoft/wavlm-base-plus",
            wavlm_prosody_model_name="Orange/Speaker-wavLM-pro",
            roberta_model_name="roberta-base",
            hidden_dim=256,
            av_refinement_layers=2,
            freeze_wavlm=True,
            freeze_roberta=True,
            num_cross_attn_heads=8,
            post_attention_layers=2,
            av_prediction_layers=3
        ).to(device)
        model = load_state(model, ckpt_path)

        test_df = get_iemocap_test_df(csv_path, fold, asr_column=None)
        dataset = IEMOCAPDataset(test_df, audio_max_length=16000*10, sample_rate=16000, normalize_av=True)
        loader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=2, collate_fn=iemocap_collate_fn)
        print(f"  Samples: {len(dataset)}")

        preds, a_pred, v_pred, true_labels, a_true, v_true = evaluate_avd(model, loader, device)

        out_dir = f"outputs/av_scatter/{model_name}"
        os.makedirs(out_dir, exist_ok=True)
        save_path = os.path.join(out_dir, f"fold_{fold}.png")
        if a_pred.size:
            a_plot = (a_pred - a_pred.min()) / (a_pred.max() - a_pred.min() + 1e-8)
            v_plot = (v_pred - v_pred.min()) / (v_pred.max() - v_pred.min() + 1e-8)
        else:
            a_plot, v_plot = a_pred, v_pred
        plot_scatter(a_plot, v_plot, preds, f"{model_name} Fold {fold}", save_path)

        log_dir = Path('logs') / 'eval_av_scatter'
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / f"{model_name}_fold{fold}.log"
        with open(log_file, 'w') as f:
            f.write(f"Model: {model_name} \nFold: {fold} \nSamples: {len(dataset)}\n")
            if true_labels is not None and len(true_labels):
                acc = accuracy_score(true_labels, preds)
                f1_macro = f1_score(true_labels, preds, average='macro')
                f1_weighted = f1_score(true_labels, preds, average='weighted')
                f.write(f"acc={acc:.4f} | f1_macro={f1_macro:.4f} | f1_weighted={f1_weighted:.4f}\n")
            else:
                f.write("No emotion ground-truth available.\n")

            if a_true is not None and v_true is not None and a_true.size and v_true.size:
                a_ccc = compute_ccc(a_true, a_pred)
                v_ccc = compute_ccc(v_true, v_pred)
                avg_ccc = (a_ccc + v_ccc) / 2.0
                f.write(f"arousal_ccc={a_ccc:.4f} | valence_ccc={v_ccc:.4f} | avg_ccc={avg_ccc:.4f}\n")
            else:
                f.write("No A/V ground-truth available.\n")
        print(f"  ✓ Logged metrics to {log_file}")

        del model
        torch.cuda.empty_cache()


def run_msp_av(model_name, checkpoint_path, tsv_path, device, use_wpro=False):
    print(f"\n=== {model_name} (MSP-Podcast AV) ===")
    if not Path(checkpoint_path).exists():
        print(f"  ✗ Missing {checkpoint_path}")
        return
    model = Method2CrossAttentionAV(
        num_emotions=4,
        wavlm_model_name="microsoft/wavlm-base-plus",
        wavlm_prosody_model_name="Orange/Speaker-wavLM-pro" if use_wpro else None,
        roberta_model_name="roberta-base",
        hidden_dim=256,
        freeze_wavlm=True,
        freeze_roberta=True
    ).to(device)
    model = load_state(model, checkpoint_path)

    # For MSP-Podcast we want to plot ground-truth and predictions side-by-side
    for split in ['test1', 'test2']:
        dataset = MSPPodcastDataset(
            tsv_path=tsv_path,
            split=split,
            audio_max_length=16000*10,
            tokenizer_name='roberta-base',
            max_text_length=128,
            four_class_only=True
        )
        loader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=2, collate_fn=msp_collate_fn)
        print(f"  {split}: {len(dataset)} samples")

        # Collect predictions and ground-truth
        preds, a_pred, v_pred, true_labels, a_true, v_true = evaluate_av(model, loader, device)

        # Map predictions and GT to [-1, 1] for plotting.
        # If GT exists and its range is not already [-1,1], map GT from its min/max to [-1,1].
        eps = 1e-6
        def map_to_minus1_1(x, source_min=None, source_max=None):
            if x is None or len(x) == 0:
                return x
            if source_min is None or source_max is None or source_max - source_min < 1e-8:
                # If we don't know source range, scale by max abs then shrink into (-1,1)
                mx = np.max(np.abs(x))
                if mx < 1e-8:
                    return np.zeros_like(x)
                scaled = x / mx
                return np.clip(scaled, -1.0 + eps, 1.0 - eps)
            # linear map from [source_min, source_max] -> (-1,1)
            mapped = -1.0 + 2.0 * (x - source_min) / (source_max - source_min + 1e-12)
            return np.clip(mapped, -1.0 + eps, 1.0 - eps)

        # Determine source ranges: prefer GT range if available, else use predictions' max abs
        if a_true is not None and v_true is not None:
            a_min, a_max = a_true.min(), a_true.max()
            v_min, v_max = v_true.min(), v_true.max()
        else:
            # fallback to prediction ranges
            a_min, a_max = a_pred.min(), a_pred.max()
            v_min, v_max = v_pred.min(), v_pred.max()

        a_pred_m = map_to_minus1_1(a_pred, a_min, a_max)
        v_pred_m = map_to_minus1_1(v_pred, v_min, v_max)
        a_true_m = map_to_minus1_1(a_true, a_min, a_max) if a_true is not None else None
        v_true_m = map_to_minus1_1(v_true, v_min, v_max) if v_true is not None else None

        # Save two plots: left = GT (if available), right = Pred
        out_dir = f"outputs/av_scatter/{model_name}"
        os.makedirs(out_dir, exist_ok=True)

        if a_true_m is not None and v_true_m is not None:
            # Plot GT
            save_path_gt = os.path.join(out_dir, f"{split}_gt.png")
            plot_scatter(a_true_m, v_true_m, preds, f"{model_name} {split} (GT)", save_path_gt)

        # Plot predictions
        save_path_pred = os.path.join(out_dir, f"{split}_pred.png")
        plot_scatter(a_pred_m, v_pred_m, preds, f"{model_name} {split} (Pred)", save_path_pred)

        # Save metrics to log
        log_dir = Path('logs') / 'eval_av_scatter'
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / f"{model_name}_{split}.log"
        with open(log_file, 'w') as f:
            f.write(f"Model: {model_name} \nSplit: {split} \nSamples: {len(dataset)}\n")
            if true_labels is not None and len(true_labels):
                acc = accuracy_score(true_labels, preds)
                f1_macro = f1_score(true_labels, preds, average='macro')
                f1_weighted = f1_score(true_labels, preds, average='weighted')
                f.write(f"acc={acc:.4f} | f1_macro={f1_macro:.4f} | f1_weighted={f1_weighted:.4f}\n")
            else:
                f.write("No emotion ground-truth available.\n")

            if a_true is not None and v_true is not None and a_true.size and v_true.size:
                a_ccc = compute_ccc(a_true, a_pred)
                v_ccc = compute_ccc(v_true, v_pred)
                avg_ccc = (a_ccc + v_ccc) / 2.0
                f.write(f"arousal_ccc={a_ccc:.4f} | valence_ccc={v_ccc:.4f} | avg_ccc={avg_ccc:.4f}\n")
            else:
                f.write("No A/V ground-truth available.\n")
        print(f"  ✓ Logged metrics to {log_file}")

    del model
    torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description='Evaluate checkpoints and plot A-V scatter')
    parser.add_argument('--device', type=str, default='cuda:0')
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # IEMOCAP AV
    run_iemocap_av(
        model_name='method2_kfold',
        checkpoint_dir='checkpoints/method2_kfold',
        csv_path='datasets/IEMOCAP_KFOLD/tenfolds/all.csv',
        device=device,
        asr_column=None,
        use_wpro=False
    )
    run_iemocap_av(
        model_name='method2_kfold_wpro',
        checkpoint_dir='checkpoints/method2_kfold_wpro',
        csv_path='datasets/IEMOCAP_KFOLD/tenfolds/all.csv',
        device=device,
        asr_column=None,
        use_wpro=True
    )
    run_iemocap_av(
        model_name='method2_kfold_wpro_asr',
        checkpoint_dir='checkpoints/method2_kfold_wpro_asr',
        csv_path='datasets/IEMOCAP_KFOLD/tenfolds/all_asr.csv',
        device=device,
        asr_column='asr_transcript',
        use_wpro=True
    )

    # IEMOCAP AVD
    run_iemocap_avd(
        model_name='method2_kfold_wpro_avd',
        checkpoint_dir='checkpoints/method2_kfold_wpro_avd',
        csv_path='datasets/IEMOCAP_KFOLD/tenfolds/all.csv',
        device=device
    )

    # MSP-Podcast AV
    run_msp_av(
        model_name='method2_msp',
        checkpoint_path='checkpoints/method2_msp/best_model.pt',
        tsv_path='datasets/MSP-Podcast/msp_podcast.tsv',
        device=device,
        use_wpro=False
    )
    run_msp_av(
        model_name='method2_msp_wpro',
        checkpoint_path='checkpoints/method2_msp_wpro/best_model.pt',
        tsv_path='datasets/MSP-Podcast/msp_podcast.tsv',
        device=device,
        use_wpro=True
    )


if __name__ == '__main__':
    main()
