"""Continue Table 11 with validated prior Full predictions and one process per fold."""
from __future__ import annotations

import argparse
import atexit
import ctypes
import json
import os
import random
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from data.iemocap_dataset import IEMOCAPDataset
from scripts.run_iemocap_table11 import LABELS, compare_predictions, make_model, metrics, run_epoch, validate_manifest

ORIGINAL = ROOT / 'outputs/table11_20261031_025259'
MANIFEST = ROOT / 'datasets/IEMOCAP_KFOLD/tenfolds/all.local.csv'
MODELS = ('no_dgm', 'wavlm_roberta_guided')


def prevent_idle_sleep():
    """Keep this Windows worker orchestration awake until it exits.

    This does not override an explicit sleep request or change the power plan.
    """
    if os.name != 'nt':
        return
    continuous = 0x80000000
    system_required = 0x00000001
    if not ctypes.windll.kernel32.SetThreadExecutionState(continuous | system_required):
        raise OSError('Could not request that Windows prevent idle sleep')
    atexit.register(lambda: ctypes.windll.kernel32.SetThreadExecutionState(continuous))
    print('Requested Windows to prevent idle sleep while this run is active.', flush=True)


def validate_prediction(path, df, session, model):
    pred = pd.read_csv(path)
    expected = df[df.session == session]
    required = {'sample_name', 'y_true', 'y_pred', 'session', 'model'}
    if not required.issubset(pred.columns) or pred.sample_name.duplicated().any():
        raise ValueError(f'Invalid prediction columns or duplicate IDs: {path}')
    aligned = expected[['sample_name', 'emotion']].merge(pred, on='sample_name', validate='one_to_one')
    if len(aligned) != len(expected) or len(pred) != len(expected):
        raise ValueError(f'Missing or extra prediction IDs: {path}')
    if not (aligned.y_true == aligned.emotion.map(LABELS)).all():
        raise ValueError(f'Labels do not match manifest: {path}')
    if not (pred.session == session).all() or not (pred.model == model).all() or not pred.y_pred.isin(range(4)).all():
        raise ValueError(f'Wrong model/session/predictions: {path}')
    return pred


def atomic_save(obj, path):
    tmp = path.with_name(path.name + '.tmp')
    torch.save(obj, tmp)
    os.replace(tmp, path)


def run_fold(out, df, name, session, config):
    random.seed(config['seed']); np.random.seed(config['seed']); torch.manual_seed(config['seed'])
    torch.cuda.manual_seed_all(config['seed'])
    device = torch.device(config['device'])
    trainval, test = df[df.session != session], df[df.session == session]
    train, val = train_test_split(trainval, test_size=0.2, random_state=config['seed'], stratify=trainval.emotion)
    datasets = [IEMOCAPDataset(part, tokenizer_name='roberta-base') for part in (train, val, test)]
    if any(len(dataset) != len(part) for dataset, part in zip(datasets, (train, val, test))):
        raise ValueError('A dataset silently filtered input files')
    loaders = [DataLoader(dataset, batch_size=config['batch_size'], shuffle=(i == 0), num_workers=config['num_workers'])
               for i, dataset in enumerate(datasets)]
    model = make_model(name).to(device)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=1e-4, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
    fold_dir = out / name / f'fold_{session}'
    fold_dir.mkdir(parents=True, exist_ok=True)
    last, best_path = fold_dir / 'last_model.pt', fold_dir / 'best_model.pt'
    best, start = -1.0, 1
    if last.exists():
        state = torch.load(last, map_location='cpu', weights_only=True)
        model.load_state_dict(state['model_state_dict'], strict=True)
        optimizer.load_state_dict(state['optimizer_state_dict'])
        scheduler.load_state_dict(state['scheduler_state_dict'])
        torch.set_rng_state(state['torch_rng_state'])
        torch.cuda.set_rng_state_all(state['cuda_rng_state_all'])
        best, start = state['best_val_accuracy'], state['epoch'] + 1
        print(f'Resuming {name} fold={session} from epoch {start}', flush=True)
        del state
    for epoch in range(start, config['epochs'] + 1):
        print(f'Starting {name} fold={session} epoch={epoch}/{config["epochs"]}', flush=True)
        run_epoch(model, loaders[0], device, True, optimizer, name == 'no_dgm')
        val_pred = run_epoch(model, loaders[1], device, False, no_dgm=name == 'no_dgm')
        score = metrics(val_pred.y_true, val_pred.y_pred)['accuracy']
        scheduler.step(score)
        print(f'{name} fold={session} epoch={epoch} val_acc={score:.5f}', flush=True)
        if score > best:
            best = score
            atomic_save({'model_state_dict': model.state_dict(), 'epoch': epoch, 'val_accuracy': score}, best_path)
        atomic_save({'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(),
                     'scheduler_state_dict': scheduler.state_dict(), 'epoch': epoch, 'best_val_accuracy': best,
                     'torch_rng_state': torch.get_rng_state(), 'cuda_rng_state_all': torch.cuda.get_rng_state_all()}, last)
    checkpoint = torch.load(best_path, map_location='cpu', weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    del checkpoint
    pred = run_epoch(model, loaders[2], device, False, no_dgm=name == 'no_dgm')
    pred['session'] = session; pred['model'] = name
    pred_path = fold_dir / 'predictions.csv'
    pred.to_csv(pred_path, index=False)
    validate_prediction(pred_path, df, session, name)
    print(f'Completed {name} fold={session}: {metrics(pred.y_true, pred.y_pred)}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--worker', choices=MODELS)
    parser.add_argument('--fold', type=int, choices=range(1, 6))
    parser.add_argument('--plan', action='store_true')
    args = parser.parse_args()
    if bool(args.worker) != bool(args.fold):
        parser.error('--worker and --fold must be specified together')
    out = Path(args.output).resolve()
    df = validate_manifest(MANIFEST)
    config = json.loads((ORIGINAL / 'run_config.json').read_text(encoding='utf-8'))
    for field, expected in [('epochs', 30), ('batch_size', 8), ('seed', 42), ('num_workers', 0), ('device', 'cuda:0')]:
        if config[field] != expected:
            raise ValueError(f'Original configuration differs: {field}={config[field]!r}')
    if args.worker:
        run_fold(out, df, args.worker, args.fold, config)
        return
    for session in range(1, 6):
        validate_prediction(ORIGINAL / 'full' / f'fold_{session}' / 'predictions.csv', df, session, 'full')
    print('All five Full prediction files match the manifest; keeping the original outputs.', flush=True)
    print('Will retrain no_dgm fold 1 from scratch, then continue all remaining folds in separate processes.', flush=True)
    print(f'Output: {out}; epochs={config["epochs"]}; batch_size={config["batch_size"]}; seed={config["seed"]}', flush=True)
    if args.plan:
        return
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable')
    prevent_idle_sleep()
    if out.exists() and not args.resume:
        raise FileExistsError(f'Output already exists: {out}; use --resume only for this run')
    out.mkdir(parents=True, exist_ok=args.resume)
    run_config = out / 'run_config.json'
    provenance = {'original_run': str(ORIGINAL), 'csv': str(MANIFEST), 'original_config': config,
                  'models_to_train': list(MODELS), 'baseline': 'reconstructed AV-guided two-stream'}
    if run_config.exists():
        if json.loads(run_config.read_text(encoding='utf-8')) != provenance:
            raise ValueError('Existing run configuration does not match')
    else:
        run_config.write_text(json.dumps(provenance, indent=2), encoding='utf-8')
    for session in range(1, 6):
        src = ORIGINAL / 'full' / f'fold_{session}' / 'predictions.csv'
        dst = out / 'full' / f'fold_{session}' / 'predictions.csv'
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not dst.exists(): shutil.copy2(src, dst)
        validate_prediction(dst, df, session, 'full')
    for name in MODELS:
        for session in range(1, 6):
            path = out / name / f'fold_{session}' / 'predictions.csv'
            if path.exists():
                validate_prediction(path, df, session, name)
                print(f'Skipping completed {name} fold={session}', flush=True)
                continue
            cmd = [sys.executable, '-u', str(Path(__file__).resolve()), '--output', str(out), '--worker', name, '--fold', str(session)]
            print(f'Launching isolated worker: {name} fold={session}', flush=True)
            subprocess.run(cmd, cwd=ROOT, check=True)
    all_pred = pd.concat([pd.read_csv(out / name / f'fold_{session}' / 'predictions.csv')
                          for name in ('full', *MODELS) for session in range(1, 6)], ignore_index=True)
    all_pred.to_csv(out / 'all_predictions.csv', index=False)
    stats = [compare_predictions(all_pred, name, config['seed'], config['bootstrap_iterations']) for name in MODELS]
    pd.DataFrame(stats).to_csv(out / 'table11_statistics.csv', index=False)
    (out / 'statistics_method.txt').write_text(
        'Mean of five session-level metric differences; 95% percentile bootstrap resampling sessions; '
        'two-sided exact sign-flip over 32 patterns (minimum p=0.0625). Overlapping training folds are not independent. '
        'WavLM+RoBERTa is a reconstructed guided model. Full predictions are from the original run.\n', encoding='utf-8')
    print(f'Finished: {out}', flush=True)


if __name__ == '__main__':
    main()
