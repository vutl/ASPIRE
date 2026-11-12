"""Train four IEMOCAP configurations sequentially and produce paired Table 11 statistics."""
from __future__ import annotations

import argparse
import itertools
import json
import random
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from data.iemocap_dataset import IEMOCAPDataset
from models.method2_crossattn_av import Method2CrossAttentionAV
from models.method2_crossattn_av_ablation_no_eav import Method2CrossAttentionAV_AblationNoEAV
from models.method2_crossattn_av_ablation_no_avsup import Method2CrossAttentionAV_AblationNoAVSup
from models.wavlm_roberta_guided import WavLMRobertaGuided
from train.losses import CombinedEmotionLossCCC

MODELS = ('full', 'no_eav', 'no_dgm', 'wavlm_roberta_guided')
TABLE11_MODELS = ('full', 'no_dgm', 'wavlm_roberta_guided')
LABELS = {'hap': 0, 'sad': 1, 'neu': 2, 'ang': 3}


def make_model(name):
    shared = dict(num_emotions=4, wavlm_model_name='microsoft/wavlm-base-plus',
                  roberta_model_name='roberta-base', hidden_dim=256,
                  freeze_wavlm=True, freeze_roberta=True, num_cross_attn_heads=8,
                  post_attention_layers=2, av_prediction_layers=3, av_refinement_layers=2)
    if name == 'wavlm_roberta_guided':
        return WavLMRobertaGuided(**shared)
    cls = {'full': Method2CrossAttentionAV, 'no_eav': Method2CrossAttentionAV_AblationNoEAV,
           'no_dgm': Method2CrossAttentionAV_AblationNoAVSup}[name]
    return cls(wavlm_prosody_model_name='Orange/Speaker-wavLM-pro', **shared)


def metrics(y, pred):
    return {'accuracy': float(accuracy_score(y, pred)),
            'macro_f1': float(f1_score(y, pred, labels=[0, 1, 2, 3], average='macro', zero_division=0))}


def validate_manifest(path):
    df = pd.read_csv(path)
    required = {'sample_name', 'path', 'fold', 'emotion', 'arousal', 'valence', 'transcript'}
    if not required.issubset(df.columns):
        raise ValueError(f'Missing columns: {sorted(required - set(df.columns))}')
    if len(df) != 5531 or df.sample_name.duplicated().any():
        raise ValueError('Expected 5531 distinct utterance IDs')
    if set(df.emotion) != set(LABELS) or set(df.fold) != set(range(1, 11)):
        raise ValueError('Expected four labels and speaker folds 1-10')
    bad = [p for p in df.path if not Path(p).is_file()]
    if bad:
        raise FileNotFoundError(f'{len(bad)} audio files missing, e.g. {bad[:3]}')
    df['session'] = (df.fold.astype(int) + 1) // 2
    for session, name in zip(df.session, df.sample_name):
        if not name.startswith(f'Ses0{session}'):
            raise ValueError(f'Fold/session mismatch: {name}')
    return df


def run_epoch(model, loader, device, train, optimizer=None, no_dgm=False):
    model.train(train)
    predictions = []
    loss_fn = CombinedEmotionLossCCC(emotion_weight=1.0, arousal_weight=0.5, valence_weight=0.5)
    with torch.set_grad_enabled(train):
        started = time.monotonic()
        for step, batch in enumerate(loader, 1):
            labels = batch['emotion_label'].to(device)
            output = model(audio_input=batch['audio'].to(device),
                           text_input=batch['text_input_ids'].to(device),
                           text_attention_mask=batch['text_attention_mask'].to(device))
            if no_dgm:
                loss = torch.nn.functional.cross_entropy(output['emotion_logits'], labels)
            else:
                loss = loss_fn(output, {'emotion_label': labels,
                                       'arousal_value': batch['arousal_value'].to(device).view(-1, 1),
                                       'valence_value': batch['valence_value'].to(device).view(-1, 1)})['total_loss']
            if not torch.isfinite(loss):
                raise ValueError('Non-finite loss')
            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            pred = output['emotion_logits'].argmax(dim=1).cpu().tolist()
            for sample_id, true, predicted in zip(batch['sample_name'], labels.cpu().tolist(), pred):
                predictions.append({'sample_name': sample_id, 'y_true': true, 'y_pred': predicted})
            if step == 1 or step % 25 == 0 or step == len(loader):
                phase = 'train' if train else 'eval'
                print(f'  {phase} batch {step}/{len(loader)} loss={loss.item():.5f} '
                      f'elapsed={time.monotonic() - started:.1f}s', flush=True)
    result = pd.DataFrame(predictions)
    if result.sample_name.duplicated().any():
        raise ValueError('Duplicate prediction IDs')
    return result


def compare_predictions(predictions, reference, seed, bootstrap_iterations):
    a = predictions[predictions.model == 'full']
    b = predictions[predictions.model == reference]
    paired = a.merge(b, on=['session', 'sample_name'], suffixes=('_full', '_ref'), validate='one_to_one')
    if len(paired) != len(a) or len(paired) != len(b) or not (paired.y_true_full == paired.y_true_ref).all():
        raise ValueError(f'Unpaired predictions for {reference}')
    differences = []
    for _, group in paired.groupby('session', sort=True):
        full = metrics(group.y_true_full, group.y_pred_full)
        ref = metrics(group.y_true_ref, group.y_pred_ref)
        differences.append([full['accuracy'] - ref['accuracy'], full['macro_f1'] - ref['macro_f1']])
    diffs = np.array(differences)
    if diffs.shape != (5, 2):
        raise ValueError('Expected predictions for all five sessions')
    rng = np.random.default_rng(seed)
    boot = diffs[rng.integers(0, 5, size=(bootstrap_iterations, 5))].mean(axis=1)
    permutations = np.array(list(itertools.product([-1, 1], repeat=5))) @ diffs / 5
    observed = diffs.mean(axis=0)
    result = {'comparison': f'full vs {reference}', 'n_utterances': len(paired), 'n_sessions': 5}
    for j, metric in enumerate(('accuracy', 'macro_f1')):
        result[f'delta_{metric}_pp'] = float(100 * observed[j])
        result[f'{metric}_ci_low_pp'], result[f'{metric}_ci_high_pp'] = (100 * np.percentile(boot[:, j], [2.5, 97.5])).tolist()
        result[f'{metric}_p_two_sided'] = float(np.mean(np.abs(permutations[:, j]) >= abs(observed[j]) - 1e-12))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--csv', default=str(ROOT / 'datasets/IEMOCAP_KFOLD/tenfolds/all.local.csv'))
    parser.add_argument('--output', default=None, help='New output directory (must not exist)')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--models', nargs='+', choices=MODELS, default=list(TABLE11_MODELS))
    parser.add_argument('--smoke-test', action='store_true', help='Real backbone forward/backward on one batch per model, then exit')
    parser.add_argument('--bootstrap-iterations', type=int, default=10000)
    parser.add_argument('--dry-run', action='store_true', help='Validate dataset and print plan without training')
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 2 or args.bootstrap_iterations < 1:
        parser.error('epochs and bootstrap iterations must be positive; batch size must be at least 2 for CCC')
    df = validate_manifest(args.csv)
    print(f'Validated {len(df)} samples: {df.session.value_counts().sort_index().to_dict()}', flush=True)
    if len(set(args.models)) != len(args.models):
        parser.error('Models must not be repeated')
    print(f'Model order: {args.models}; epochs={args.epochs}; batch size={args.batch_size}', flush=True)
    print('Baseline is a reconstructed guided two-stream model, not the original checkpoint.', flush=True)
    if args.dry_run:
        return
    if args.device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable; pass --device cpu explicitly to run on CPU')
    device = torch.device(args.device)
    if args.smoke_test:
        dataset = IEMOCAPDataset(df.iloc[:args.batch_size], tokenizer_name='roberta-base')
        loader = DataLoader(dataset, batch_size=args.batch_size)
        for name in args.models:
            print(f'Smoke test: {name}', flush=True)
            model = make_model(name).to(device)
            optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=1e-4)
            run_epoch(model, loader, device, True, optimizer, name == 'no_dgm')
            run_epoch(model, loader, device, False, no_dgm=name == 'no_dgm')
            print(f'Passed {name}; trainable parameters={sum(p.numel() for p in model.parameters() if p.requires_grad)}', flush=True)
            del model, optimizer
            if device.type == 'cuda': torch.cuda.empty_cache()
        return
    out = Path(args.output) if args.output else ROOT / 'outputs' / ('table11_' + datetime.now().strftime('%Y%m%d_%H%M%S'))
    out.mkdir(parents=True, exist_ok=False)
    (out / 'run_config.json').write_text(json.dumps(vars(args), indent=2), encoding='utf-8')
    predictions = []
    for name in args.models:
        for session in range(1, 6):
            random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
            if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.seed)
            trainval, test = df[df.session != session], df[df.session == session]
            train, val = train_test_split(trainval, test_size=0.2, random_state=args.seed, stratify=trainval.emotion)
            datasets = [IEMOCAPDataset(part, tokenizer_name='roberta-base') for part in (train, val, test)]
            if any(len(d) != len(part) for d, part in zip(datasets, (train, val, test))):
                raise ValueError('Dataset silently filtered an input file')
            loaders = [DataLoader(d, batch_size=args.batch_size, shuffle=(i == 0), num_workers=args.num_workers)
                       for i, d in enumerate(datasets)]
            model = make_model(name).to(device)
            optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=1e-4, weight_decay=1e-5)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
            fold_dir = out / name / f'fold_{session}'
            fold_dir.mkdir(parents=True)
            best = -1.0
            for epoch in range(1, args.epochs + 1):
                print(f'Starting {name} fold={session} epoch={epoch}/{args.epochs}', flush=True)
                run_epoch(model, loaders[0], device, True, optimizer, name == 'no_dgm')
                val_pred = run_epoch(model, loaders[1], device, False, no_dgm=name == 'no_dgm')
                val_acc = metrics(val_pred.y_true, val_pred.y_pred)['accuracy']
                scheduler.step(val_acc)
                print(f'{name} fold={session} epoch={epoch} val_acc={val_acc:.5f}', flush=True)
                if val_acc > best:
                    best = val_acc
                    torch.save({'model_state_dict': model.state_dict(), 'epoch': epoch, 'val_accuracy': val_acc}, fold_dir / 'best_model.pt')
            checkpoint = torch.load(fold_dir / 'best_model.pt', map_location=device, weights_only=True)
            model.load_state_dict(checkpoint['model_state_dict'], strict=True)
            pred = run_epoch(model, loaders[2], device, False, no_dgm=name == 'no_dgm')
            if set(pred.sample_name) != set(test.sample_name):
                raise ValueError('Test predictions do not match held-out IDs')
            pred['session'] = session; pred['model'] = name
            pred.to_csv(fold_dir / 'predictions.csv', index=False)
            predictions.append(pred)
            del model, optimizer, checkpoint, loaders, datasets
            if device.type == 'cuda': torch.cuda.empty_cache()
    all_pred = pd.concat(predictions, ignore_index=True)
    all_pred.to_csv(out / 'all_predictions.csv', index=False)
    results = [compare_predictions(all_pred, r, args.seed, args.bootstrap_iterations)
               for r in ('no_dgm', 'wavlm_roberta_guided')]
    pd.DataFrame(results).to_csv(out / 'table11_statistics.csv', index=False)
    (out / 'statistics_method.txt').write_text(
        'Mean of five session-level metric differences. 95% percentile bootstrap CI resamples the five sessions. '
        'Two-sided exact sign-flip permutation p-value over 32 sign patterns. With five sessions, the smallest '
        'possible two-sided p-value is 0.0625. Fold training sets overlap; results do not establish independence. '
        'The guided WavLM+RoBERTa reference is reconstructed, not the original paper implementation.\n', encoding='utf-8')
    print(f'Saved results to {out}', flush=True)


if __name__ == '__main__':
    main()
