"""
5-Fold Training for Method2 W-PRO (No Validation)

This variant uses the entire `train_val` set as training data (no train/val split).
One session is held out as test. Behaviour and checkpoints mirror the original
trainer but skip validation steps.

Usage (example):
  python3 train/train_method2_kfold_wpro_no_val.py \
    --config configs/method2_wpro_config.yaml \
    --csv datasets/IEMOCAP_KFOLD/tenfolds/all.csv \
    --output_dir checkpoints/method2_kfold_wpro_no_val \
    --device cuda:1 \
    --resume_fold 1 \
    --resume_checkpoint checkpoints/method2_kfold_wpro/fold_1/last_model.pt
"""
import os
import sys
import yaml
import argparse
from pathlib import Path
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import accuracy_score, f1_score

# Add repo root
sys.path.append(str(Path(__file__).resolve().parent.parent))

from models.method2_crossattn_av import Method2CrossAttentionAV
from train.losses import CombinedEmotionLossCCC
from data.iemocap_dataset import IEMOCAPDataset


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


def train_one_epoch(model, loader, criterion, optimizer, device, writer, global_step, epoch):
    model.train()
    total_loss = 0.0
    all_preds = []
    all_labels = []

    pbar = tqdm(loader, desc=f"Epoch {epoch} [Train]", leave=False)
    for batch_idx, batch in enumerate(pbar):
        audio = batch['audio'].to(device)
        text_input_ids = batch['text_input_ids'].to(device)
        text_attention_mask = batch['text_attention_mask'].to(device)
        emotion_label = batch['emotion_label'].to(device)
        arousal = batch['arousal_value'].to(device)
        valence = batch['valence_value'].to(device)

        optimizer.zero_grad()
        outputs = model(audio_input=audio, text_input=text_input_ids, text_attention_mask=text_attention_mask)

        loss_dict = criterion(outputs, {'emotion_label': emotion_label, 'arousal_value': arousal, 'valence_value': valence})
        loss = loss_dict['total_loss']
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        preds = torch.argmax(outputs['emotion_logits'], dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(emotion_label.cpu().numpy())

        pbar.set_postfix({'loss': f"{loss.item():.4f}", 'acc': f"{accuracy_score(all_labels, all_preds):.3f}"})

        if writer is not None and batch_idx % 50 == 0:
            writer.add_scalar('Train/Batch/Total_Loss', loss.item(), global_step)
            global_step += 1

    avg_loss = total_loss / len(loader)
    accuracy = accuracy_score(all_labels, all_preds)
    f1_macro = f1_score(all_preds, all_labels, average='macro')
    return {'loss': avg_loss, 'accuracy': accuracy, 'f1_macro': f1_macro}, global_step


def test_model(model, loader, criterion, device, writer, epoch=0):
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    all_arousal_pred = []
    all_arousal_true = []
    all_valence_pred = []
    all_valence_true = []

    with torch.no_grad():
        pbar = tqdm(loader, desc="[Test]", leave=False)
        for batch in pbar:
            audio = batch['audio'].to(device)
            text_input_ids = batch['text_input_ids'].to(device)
            text_attention_mask = batch['text_attention_mask'].to(device)
            emotion_label = batch['emotion_label'].to(device)
            arousal = batch['arousal_value'].to(device)
            valence = batch['valence_value'].to(device)

            outputs = model(audio_input=audio, text_input=text_input_ids, text_attention_mask=text_attention_mask)
            loss_dict = criterion(outputs, {'emotion_label': emotion_label, 'arousal_value': arousal, 'valence_value': valence})
            loss = loss_dict['total_loss']
            total_loss += loss.item()

            preds = torch.argmax(outputs['emotion_logits'], dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(emotion_label.cpu().numpy())

            all_arousal_pred.extend(outputs['arousal_value'].cpu().numpy())
            all_arousal_true.extend(arousal.cpu().numpy())
            all_valence_pred.extend(outputs['valence_value'].cpu().numpy())
            all_valence_true.extend(valence.cpu().numpy())

    avg_loss = total_loss / len(loader)
    accuracy = accuracy_score(all_labels, all_preds)
    f1_macro = f1_score(all_preds, all_labels, average='macro')
    f1_weighted = f1_score(all_labels, all_preds, average='weighted')

    all_arousal_pred = np.array(all_arousal_pred).flatten()
    all_arousal_true = np.array(all_arousal_true).flatten()
    all_valence_pred = np.array(all_valence_pred).flatten()
    all_valence_true = np.array(all_valence_true).flatten()

    arousal_ccc = compute_ccc(all_arousal_true, all_arousal_pred)
    valence_ccc = compute_ccc(all_valence_true, all_valence_pred)
    avg_ccc = (arousal_ccc + valence_ccc) / 2

    return {
        'loss': avg_loss,
        'accuracy': accuracy,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted,
        'arousal_ccc': arousal_ccc,
        'valence_ccc': valence_ccc,
        'avg_ccc': avg_ccc
    }


def train_fold_no_val(fold_data, config, fold_idx, output_dir, device, resume_checkpoint=None):
    print(f"\n{'='*80}")
    print(f"🏁 Training Fold {fold_idx} (Test Session {fold_data['session']}) - NO VALIDATION")
    print(f"{'='*80}")

    train_df = fold_data['train_val']
    test_df = fold_data['test']

    print(f"Train (full): {len(train_df)} samples")
    print(f"Test:         {len(test_df)} samples")

    train_dataset = IEMOCAPDataset(train_df, audio_max_length=config.get('max_audio_length',160000), max_text_length=config.get('max_text_length',128))
    test_dataset = IEMOCAPDataset(test_df, audio_max_length=config.get('max_audio_length',160000), max_text_length=config.get('max_text_length',128))

    train_loader = DataLoader(train_dataset, batch_size=config['training']['batch_size'], shuffle=True, num_workers=config['training']['num_workers'], pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=config['training']['batch_size'], shuffle=False, num_workers=config['training']['num_workers'], pin_memory=True)

    model = Method2CrossAttentionAV(
        num_emotions=config['model']['num_emotions'],
        wavlm_model_name=config['model']['wavlm_model_name'],
        wavlm_prosody_model_name=config['model'].get('wavlm_prosody_model_name', None),
        roberta_model_name=config['model'].get('roberta_model_name','roberta-base'),
        hidden_dim=config['model']['hidden_dim'],
        av_refinement_layers=config['model']['av_refinement_layers'],
        freeze_wavlm=config['model']['freeze_wavlm'],
        freeze_roberta=config['model'].get('freeze_roberta', False),
        num_cross_attn_heads=config['model']['num_cross_attn_heads'],
        post_attention_layers=config['model'].get('post_attention_layers',2),
        av_prediction_layers=config['model'].get('av_prediction_layers',3)
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params: {total_params:,}, trainable: {trainable_params:,}")

    criterion = CombinedEmotionLossCCC(emotion_weight=config['loss']['emotion_weight'], arousal_weight=config['loss']['arousal_weight'], valence_weight=config['loss']['valence_weight'])

    optimizer = torch.optim.AdamW(model.parameters(), lr=config['training']['learning_rate'], weight_decay=config['training']['weight_decay'])

    start_epoch = 1
    best_train_acc = 0.0

    if resume_checkpoint and os.path.exists(resume_checkpoint):
        print(f"Resuming from checkpoint: {resume_checkpoint}")
        ckpt = torch.load(resume_checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
        if 'optimizer_state_dict' in ckpt:
            try:
                optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            except Exception:
                print("Warning: couldn't load optimizer state")
        start_epoch = ckpt.get('epoch', 0) + 1
        best_train_acc = ckpt.get('train_accuracy', 0.0)

    writer = SummaryWriter(os.path.join(output_dir, f'logs/fold_{fold_idx}'))

    checkpoint_dir = os.path.join(output_dir, f'fold_{fold_idx}')
    os.makedirs(checkpoint_dir, exist_ok=True)

    global_step = 0
    epochs = config['training']['epochs']

    for epoch in range(start_epoch, epochs + 1):
        print(f"\n--- Fold {fold_idx} Epoch {epoch}/{epochs} ---")
        train_metrics, global_step = train_one_epoch(model, train_loader, criterion, optimizer, device, writer, global_step, epoch)

        # Save checkpoint (last)
        torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'train_accuracy': train_metrics['accuracy']}, os.path.join(checkpoint_dir, 'last_model.pt'))

        # Save best by training accuracy
        if train_metrics['accuracy'] > best_train_acc:
            best_train_acc = train_metrics['accuracy']
            torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'train_accuracy': train_metrics['accuracy']}, os.path.join(checkpoint_dir, 'best_model.pt'))
            print(f"New best training accuracy: {best_train_acc:.4f} (saved)")

    # After training, load best and test
    best_path = os.path.join(checkpoint_dir, 'best_model.pt')
    if not os.path.exists(best_path):
        best_path = os.path.join(checkpoint_dir, 'last_model.pt')
    ckpt = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])

    test_metrics = test_model(model, test_loader, criterion, device, writer)

    print(f"\n=== Fold {fold_idx} Test Results ===")
    print(f"Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"F1 Macro: {test_metrics['f1_macro']:.4f}")
    print(f"Arousal CCC: {test_metrics['arousal_ccc']:.4f}")
    print(f"Valence CCC: {test_metrics['valence_ccc']:.4f}")
    print(f"Avg CCC: {test_metrics['avg_ccc']:.4f}")

    writer.close()
    return test_metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/method2_wpro_config.yaml')
    parser.add_argument('--csv', type=str, default='datasets/IEMOCAP_KFOLD/tenfolds/all.csv')
    parser.add_argument('--output_dir', type=str, default='checkpoints/method2_kfold_wpro_no_val')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--resume_fold', type=int, default=None)
    parser.add_argument('--resume_checkpoint', type=str, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    os.makedirs(args.output_dir, exist_ok=True)

    folds = prepare_folds(args.csv)
    start_fold = args.resume_fold if args.resume_fold else 1

    all_results = []
    for fold_idx, fold_data in enumerate(folds[start_fold-1:], start_fold):
        resume_ckpt = args.resume_checkpoint if fold_idx == start_fold else None
        metrics = train_fold_no_val(fold_data, config, fold_idx, args.output_dir, device, resume_ckpt)
        metrics['fold'] = fold_idx
        metrics['session'] = fold_data['session']
        all_results.append(metrics)

    results_df = pd.DataFrame(all_results)
    results_path = os.path.join(args.output_dir, 'kfold_results_no_val.csv')
    results_df.to_csv(results_path, index=False)
    print(f"Results saved to {results_path}")


if __name__ == '__main__':
    main()
