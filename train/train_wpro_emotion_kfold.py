"""
5-Fold k-fold trainer for W-PRO emotion-only model.
Uses CrossEntropyLoss and logs accuracy / F1 (emotion labels) only.
Output/logging format follows existing wpro k-fold script.
"""
import os
import sys
import yaml
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import logging
import argparse
from datetime import datetime
from pathlib import Path
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

sys.path.append(str(Path(__file__).resolve().parent.parent))

from models.wpro_emotion_only import WProEmotionOnly
from data.iemocap_dataset import IEMOCAPDataset


def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def prepare_folds(csv_path):
    df = pd.read_csv(csv_path)
    fold_to_session = {1:1,2:1,3:2,4:2,5:3,6:3,7:4,8:4,9:5,10:5}
    df['session'] = df['fold'].map(fold_to_session)
    folds = []
    for session in range(1,6):
        test_mask = df['session']==session
        folds.append({
            'train_indices': df[~test_mask].index.tolist(),
            'test_indices': df[test_mask].index.tolist(),
            'session': session
        })
    return folds


def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def train_one_epoch(model, loader, criterion, optimizer, device, writer, global_step, epoch):
    model.train()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    pbar = tqdm(loader, desc=f"Epoch {epoch} [Train]", leave=False)
    for batch in pbar:
        audio = batch['audio'].to(device)
        labels = batch['emotion_label'].to(device)
        outputs = model(audio)
        logits = outputs['emotion_logits']
        loss = criterion(logits, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        preds = torch.argmax(logits, dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())
        if writer:
            writer.add_scalar('Train/Batch Loss', loss.item(), global_step)
            global_step += 1
    avg_loss = total_loss / len(loader)
    acc = accuracy_score(all_labels, all_preds)
    f1_macro = f1_score(all_labels, all_preds, average='macro')
    return global_step, { 'loss': avg_loss, 'accuracy': acc, 'f1_macro': f1_macro }


def validate(model, loader, criterion, device, writer, epoch, phase='Val'):
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in tqdm(loader, desc=f"Epoch {epoch} [{phase}]", leave=False):
            audio = batch['audio'].to(device)
            labels = batch['emotion_label'].to(device)
            outputs = model(audio)
            logits = outputs['emotion_logits']
            loss = criterion(logits, labels)
            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
    avg_loss = total_loss / len(loader)
    acc = accuracy_score(all_labels, all_preds)
    f1_macro = f1_score(all_labels, all_preds, average='macro')
    if writer:
        writer.add_scalar(f'{phase}/Loss', avg_loss, epoch)
        writer.add_scalar(f'{phase}/Accuracy', acc, epoch)
        writer.add_scalar(f'{phase}/F1 Macro', f1_macro, epoch)
    return { 'loss': avg_loss, 'accuracy': acc, 'f1_macro': f1_macro }


def train_fold(fold_data, config, fold_idx, output_dir, device, resume_checkpoint=None):
    df = pd.read_csv(config['data']['train_csv'])
    train_idx = fold_data['train_indices']
    test_idx = fold_data['test_indices']
    train_idx, val_idx = train_test_split(train_idx, test_size=0.2, random_state=42)
    train_ds = IEMOCAPDataset(df.iloc[train_idx].reset_index(drop=True), sample_rate=config['data']['sample_rate'], audio_max_length=config['data']['max_audio_length'])
    val_ds = IEMOCAPDataset(df.iloc[val_idx].reset_index(drop=True), sample_rate=config['data']['sample_rate'], audio_max_length=config['data']['max_audio_length'])
    test_ds = IEMOCAPDataset(df.iloc[test_idx].reset_index(drop=True), sample_rate=config['data']['sample_rate'], audio_max_length=config['data']['max_audio_length'])
    train_loader = DataLoader(train_ds, batch_size=config['training']['batch_size'], shuffle=True, num_workers=config['training'].get('num_workers', 2), collate_fn=None)
    val_loader = DataLoader(val_ds, batch_size=config['training']['batch_size'], shuffle=False, num_workers=config['training'].get('num_workers', 2), collate_fn=None)
    test_loader = DataLoader(test_ds, batch_size=config['training']['batch_size'], shuffle=False, num_workers=config['training'].get('num_workers', 2), collate_fn=None)

    model = WProEmotionOnly(num_emotions=config['model'].get('num_emotions',4), wpro_model_name=config['model'].get('wpro_model_name','Orange/Speaker-wavLM-pro'), hidden_dim=config['model'].get('hidden_dim',256), freeze_wpro=config['model'].get('freeze_wpro',True))
    model.to(device)
    total, trainable = count_params(model)
    logger = logging.getLogger(__name__)
    logger.info(f"Fold {fold_idx}: Total params: {total:,}, Trainable: {trainable:,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=config['training'].get('learning_rate',1e-4), weight_decay=config['training'].get('weight_decay',1e-5))

    fold_output_dir = Path(output_dir) / f'fold{fold_idx}'
    fold_output_dir.mkdir(parents=True, exist_ok=True)
    tb_dir = Path(config['paths'].get('tensorboard_dir','logs/tensorboard')) / f'wpro_emotion_fold{fold_idx}'
    tb_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(str(tb_dir))

    epochs = config['training'].get('epochs',30)
    global_step = 0
    best_val = -1.0

    if resume_checkpoint:
        ckpt = torch.load(resume_checkpoint, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        start_epoch = ckpt.get('epoch',0)+1
    else:
        start_epoch = 0

    for epoch in range(start_epoch, epochs):
        logger.info(f"Fold {fold_idx} Epoch {epoch+1}/{epochs}")
        global_step, train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, device, writer, global_step, epoch+1)
        logger.info(f"Train - Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['accuracy']:.4f}, F1: {train_metrics['f1_macro']:.4f}")
        val_metrics = validate(model, val_loader, criterion, device, writer, epoch+1, 'Val')
        test_metrics = validate(model, test_loader, criterion, device, writer, epoch+1, 'Test')

        is_best = val_metrics['f1_macro'] > best_val
        if is_best:
            best_val = val_metrics['f1_macro']
        checkpoint = {
            'epoch': epoch,
            'fold': fold_idx,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_val_f1': best_val,
            'global_step': global_step
        }
        torch.save(checkpoint, fold_output_dir / f'checkpoint_epoch{epoch+1}.pt')
        if is_best:
            torch.save(checkpoint, fold_output_dir / 'checkpoint_best.pt')

    writer.close()
    return { 'fold': fold_idx, 'best_val_f1': best_val, 'final_test': test_metrics }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/method2_wpro_ablation_roberta_bs8.yaml')
    parser.add_argument('--csv', default='datasets/IEMOCAP_KFOLD/tenfolds/all.csv')
    parser.add_argument('--output_dir', default='checkpoints/method2_kfold_wpro_emotion_only')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--resume_fold', type=int, default=None)
    parser.add_argument('--resume_checkpoint', type=str, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    config['data']['train_csv'] = args.csv

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    log_file = output_dir / f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', handlers=[logging.FileHandler(log_file), logging.StreamHandler()])
    logger = logging.getLogger(__name__)

    logger.info("Starting W-PRO emotion-only 5-fold k-fold training")
    logger.info(f"Device: {device}")
    logger.info(f"Config: {args.config}")
    logger.info(f"CSV: {args.csv}")

    folds = prepare_folds(args.csv)
    start_fold = args.resume_fold if args.resume_fold else 1
    results = []
    for fi in range(start_fold, 6):
        resume_ckpt = args.resume_checkpoint if fi == start_fold else None
        res = train_fold(folds[fi-1], config, fi, output_dir, device, resume_ckpt)
        results.append(res)
        logger.info(f"Fold {fi} best val F1: {res['best_val_f1']:.4f}")

    logger.info("Training complete")


if __name__ == '__main__':
    main()
