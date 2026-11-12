"""
5-Fold k-fold trainer for WavLM-Base-Plus + AV refinement on IEMOCAP

Follows logging/printing style of other k-fold trainers in the repo.
Outputs emotion logits + arousal/valence regression; uses CombinedEmotionLossCCC.
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

# Add workspace root
sys.path.append(str(Path(__file__).resolve().parent.parent))

from models.wavlm_base_av import WavLMBaseAV
from train.losses import CombinedEmotionLossCCC
from data.iemocap_dataset import IEMOCAPDataset, collate_fn


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


def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def prepare_folds(csv_path):
    df = pd.read_csv(csv_path)
    # map 10-fold -> 5-session if necessary
    max_fold = df['fold'].max()
    if max_fold == 10:
        fold_to_session = {1:1,2:1,3:2,4:2,5:3,6:3,7:4,8:4,9:5,10:5}
        df['session'] = df['fold'].map(fold_to_session)
    else:
        df['session'] = df['fold']

    folds = []
    for session in range(1, 6):
        test_df = df[df['session'] == session].reset_index(drop=True)
        train_val_df = df[df['session'] != session].reset_index(drop=True)
        folds.append({'train_val': train_val_df, 'test': test_df, 'session': session})
    return folds


def train_one_epoch(model, loader, criterion, optimizer, device, writer, global_step, epoch):
    model.train()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    pbar = tqdm(loader, desc=f"Epoch {epoch} [Train]", leave=False)
    for batch in pbar:
        audio = batch['audio'].to(device)
        labels = batch['emotion_label'].to(device)
        arousal_targets = batch['arousal_value'].to(device)
        valence_targets = batch['valence_value'].to(device)

        outputs = model(audio)
        targets = {
            'emotion_label': labels,
            'arousal_value': arousal_targets,
            'valence_value': valence_targets
        }
        loss_dict = criterion(outputs, targets)
        loss = loss_dict['total_loss']

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        preds = torch.argmax(outputs['emotion_logits'], dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())

        if writer:
            writer.add_scalar('Train/Batch Loss', loss.item(), global_step)
            global_step += 1

    avg_loss = total_loss / len(loader)
    acc = accuracy_score(all_labels, all_preds)
    f1_macro = f1_score(all_labels, all_preds, average='macro')
    # return (train_metrics, global_step) to match method2_kfold_wpro
    return {'loss': avg_loss, 'accuracy': acc, 'f1_macro': f1_macro}, global_step


def validate(model, loader, criterion, device, writer, epoch, phase='Val'):
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    all_arousal_pred = []
    all_arousal_true = []
    all_valence_pred = []
    all_valence_true = []

    with torch.no_grad():
        for batch in tqdm(loader, desc=f"Epoch {epoch} [{phase}]", leave=False):
            audio = batch['audio'].to(device)
            labels = batch['emotion_label'].to(device)
            arousal_targets = batch['arousal_value'].to(device)
            valence_targets = batch['valence_value'].to(device)

            outputs = model(audio)
            targets = {
                'emotion_label': labels,
                'arousal_value': arousal_targets,
                'valence_value': valence_targets
            }
            loss_dict = criterion(outputs, targets)
            loss = loss_dict['total_loss']
            total_loss += loss.item()

            preds = torch.argmax(outputs['emotion_logits'], dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

            all_arousal_pred.extend(outputs['arousal_value'].squeeze(-1).cpu().numpy())
            all_arousal_true.extend(arousal_targets.squeeze(-1).cpu().numpy())
            all_valence_pred.extend(outputs['valence_value'].squeeze(-1).cpu().numpy())
            all_valence_true.extend(valence_targets.squeeze(-1).cpu().numpy())

    avg_loss = total_loss / len(loader)
    acc = accuracy_score(all_labels, all_preds)
    f1_macro = f1_score(all_labels, all_preds, average='macro')
    arousal_ccc = compute_ccc(np.array(all_arousal_true), np.array(all_arousal_pred))
    valence_ccc = compute_ccc(np.array(all_valence_true), np.array(all_valence_pred))
    avg_ccc = (arousal_ccc + valence_ccc) / 2

    if writer:
        writer.add_scalar(f'{phase}/Loss', avg_loss, epoch)
        writer.add_scalar(f'{phase}/Accuracy', acc, epoch)
        writer.add_scalar(f'{phase}/F1 Macro', f1_macro, epoch)
        writer.add_scalar(f'{phase}/Arousal CCC', arousal_ccc, epoch)
        writer.add_scalar(f'{phase}/Valence CCC', valence_ccc, epoch)
        writer.add_scalar(f'{phase}/Avg CCC', avg_ccc, epoch)

    # mirror print/log style of method2_kfold_wpro
    print(f"Epoch {epoch} {phase} - Loss: {avg_loss:.4f}, Acc: {acc:.4f}, F1: {f1_macro:.4f}, Arousal CCC: {arousal_ccc:.4f}, Valence CCC: {valence_ccc:.4f}")

    return {
        'loss': avg_loss,
        'accuracy': acc,
        'f1_macro': f1_macro,
        'arousal_ccc': arousal_ccc,
        'valence_ccc': valence_ccc,
        'avg_ccc': avg_ccc
    }


def train_fold(fold_data, config, fold_idx, output_dir, device, resume_checkpoint=None, continue_epochs=0):
    print(f"\n{'='*80}")
    print(f"🏁 Training Fold {fold_idx} (Test Session {fold_data['session']})")
    print(f"{'='*80}")

    # split train/val
    train_df, val_df = train_test_split(fold_data['train_val'], test_size=0.2, random_state=42, stratify=fold_data['train_val']['emotion'])
    test_df = fold_data['test']

    print(f"Train: {len(train_df)} samples")
    print(f"Val:   {len(val_df)} samples")
    print(f"Test:  {len(test_df)} samples")

    train_dataset = IEMOCAPDataset(train_df.reset_index(drop=True), sample_rate=config['data']['sample_rate'], audio_max_length=config['data']['max_audio_length'])
    val_dataset = IEMOCAPDataset(val_df.reset_index(drop=True), sample_rate=config['data']['sample_rate'], audio_max_length=config['data']['max_audio_length'])
    test_dataset = IEMOCAPDataset(test_df.reset_index(drop=True), sample_rate=config['data']['sample_rate'], audio_max_length=config['data']['max_audio_length'])

    train_loader = DataLoader(train_dataset, batch_size=config['training']['batch_size'], shuffle=True, num_workers=config['training'].get('num_workers',2), collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=config['training']['batch_size'], shuffle=False, num_workers=config['training'].get('num_workers',2), collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=config['training']['batch_size'], shuffle=False, num_workers=config['training'].get('num_workers',2), collate_fn=collate_fn)

    model = WavLMBaseAV(
        num_emotions=config['model'].get('num_emotions',4),
        wavlm_model_name=config['model'].get('wavlm_model_name','microsoft/wavlm-base-plus'),
        hidden_dim=config['model'].get('hidden_dim',256),
        av_refinement_layers=config['model'].get('av_refinement_layers',2),
        freeze_wavlm=config['model'].get('freeze_wavlm',True)
    )
    model.to(device)

    total_params, trainable_params = count_parameters(model)
    print(f"Fold {fold_idx}: Total params: {total_params:,}, Trainable: {trainable_params:,}")

    criterion = CombinedEmotionLossCCC(
        emotion_weight=config['loss'].get('emotion_weight',1.0),
        arousal_weight=config['loss'].get('arousal_weight',1.0),
        valence_weight=config['loss'].get('valence_weight',1.0)
    )

    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=config['training'].get('learning_rate',1e-4), weight_decay=config['training'].get('weight_decay',1e-5))
    # Scheduler (same behavior as method2_kfold_wpro)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',
        factor=0.5,
        patience=5
    )

    # Make checkpoint dir consistent with method2_kfold_wpro (fold_{n})
    checkpoint_dir = os.path.join(output_dir, f'fold_{fold_idx}')
    os.makedirs(checkpoint_dir, exist_ok=True)
    log_dir = os.path.join(output_dir, f'logs/fold_{fold_idx}')
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir)

    epochs = config['training'].get('epochs',30)
    start_epoch = 1
    best_val_acc = 0.0
    global_step = 0

    # Resume/continuation semantics aligned with method2_kfold_wpro
    if resume_checkpoint and os.path.exists(resume_checkpoint):
        print(f"\n🔄 Resuming from checkpoint: {resume_checkpoint}")
        checkpoint = torch.load(resume_checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint.get('epoch',0) + 1
        best_val_acc = checkpoint.get('val_accuracy', 0.0)
        print(f"   Starting from epoch {start_epoch}")
        print(f"   Previous best val accuracy: {best_val_acc:.4f}\n")
    elif resume_checkpoint is None and fold_idx == 1 and continue_epochs > 0:
        # Check for last_model.pt for continuation mode (fold 1)
        last_model_path = os.path.join(os.path.dirname(output_dir), f'fold_{fold_idx}', 'last_model.pt')
        if os.path.exists(last_model_path):
            print(f"\n⏱️  CONTINUATION MODE - Loading last_model.pt for +{continue_epochs} epochs")
            checkpoint = torch.load(last_model_path, map_location=device, weights_only=False)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            best_val_acc = checkpoint.get('val_accuracy', 0.0)
            print(f"   Starting from epoch {start_epoch}")
            print(f"   Will train until epoch {min(start_epoch + continue_epochs - 1, config['training']['epochs'])}\n")

    for epoch in range(start_epoch, config['training']['epochs'] + 1):
        print(f"\n{'='*60}")
        print(f"Fold {fold_idx} - Epoch {epoch}/{config['training']['epochs']}")
        print(f"{'='*60}")

        # Train
        train_metrics, global_step = train_one_epoch(model, train_loader, criterion, optimizer, device, writer, global_step, epoch)
        print(f"Fold {fold_idx} Epoch {epoch} Train - Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['accuracy']:.4f}, F1: {train_metrics['f1_macro']:.4f}")

        # Validate
        val_metrics = validate(model, val_loader, criterion, device, writer, epoch, phase='Val')

        # Scheduler step
        try:
            scheduler.step(val_metrics['accuracy'])
        except Exception:
            pass

        # Log learning rate
        current_lr = optimizer.param_groups[0]['lr']
        writer.add_scalar('Train/Learning_Rate', current_lr, epoch)

        # Print metrics
        print(f"\n[Train] Loss: {train_metrics['loss']:.4f} | Acc: {train_metrics['accuracy']:.4f} | F1: {train_metrics['f1_macro']:.4f}")
        print(f"[Val]   Loss: {val_metrics['loss']:.4f} | Acc: {val_metrics['accuracy']:.4f} | F1: {val_metrics['f1_macro']:.4f} | CCC: {val_metrics['avg_ccc']:.4f}")

        # Save best model (use accuracy like method2_kfold_wpro)
        if val_metrics['accuracy'] > best_val_acc:
            best_val_acc = val_metrics['accuracy']
            best_epoch = epoch
            checkpoint_path = os.path.join(checkpoint_dir, 'best_model.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_accuracy': val_metrics['accuracy'],
                'val_f1_macro': val_metrics['f1_macro'],
                'val_avg_ccc': val_metrics['avg_ccc']
            }, checkpoint_path)
            print(f"🔥 New best validation accuracy: {best_val_acc:.4f} (saved)")

        # Always save last_model.pt for continuation training
        last_model_path = os.path.join(checkpoint_dir, 'last_model.pt')
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_accuracy': val_metrics['accuracy'],
            'val_f1_macro': val_metrics['f1_macro'],
            'val_avg_ccc': val_metrics['avg_ccc']
        }, last_model_path)

    writer.close()
    return {'fold': fold_idx, 'best_val_acc': best_val_acc, 'checkpoint_dir': checkpoint_dir}


def test_only_fold(fold_data, config, fold_idx, output_dir, device, checkpoint_path):
    """Load checkpoint and run test-only evaluation for a single fold."""
    print(f"\n{'='*80}")
    print(f"🔎 Test Only - Fold {fold_idx} (Test Session {fold_data['session']})")
    print(f"{'='*80}")

    test_df = fold_data['test']
    test_dataset = IEMOCAPDataset(test_df.reset_index(drop=True), sample_rate=config['data']['sample_rate'], audio_max_length=config['data']['max_audio_length'])
    test_loader = DataLoader(test_dataset, batch_size=config['training']['batch_size'], shuffle=False, num_workers=config['training'].get('num_workers',2), collate_fn=collate_fn)

    model = WavLMBaseAV(
        num_emotions=config['model'].get('num_emotions',4),
        wavlm_model_name=config['model'].get('wavlm_model_name','microsoft/wavlm-base-plus'),
        hidden_dim=config['model'].get('hidden_dim',256),
        av_refinement_layers=config['model'].get('av_refinement_layers',2),
        freeze_wavlm=config['model'].get('freeze_wavlm',True)
    )
    model.to(device)

    criterion = CombinedEmotionLossCCC(
        emotion_weight=config['loss'].get('emotion_weight',1.0),
        arousal_weight=config['loss'].get('arousal_weight',1.0),
        valence_weight=config['loss'].get('valence_weight',1.0)
    )

    if not checkpoint_path or not os.path.exists(checkpoint_path):
        print(f"Checkpoint not found: {checkpoint_path}")
        return None

    print(f"Loading checkpoint for test: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

    test_metrics = validate(model, test_loader, criterion, device, writer=None, epoch=0, phase='Test')
    print(f"\nTest metrics for Fold {fold_idx}: {test_metrics}")
    return test_metrics


def main():
    parser = argparse.ArgumentParser(description='5-Fold Cross-Validation Training for WavLM AV (aligned to W-PRO kfold)')
    parser.add_argument('--config', default='configs/method2_config_bs8.yaml')
    parser.add_argument('--csv', default='datasets/IEMOCAP_KFOLD/tenfolds/all.csv')
    parser.add_argument('--output_dir', default='checkpoints/method2_kfold_wavlm_av')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--resume_fold', type=int, default=None)
    parser.add_argument('--resume_checkpoint', type=str, default=None)
    parser.add_argument('--continue_epochs', type=int, default=0)
    args = parser.parse_args()

    config = load_config(args.config)
    config['data']['train_csv'] = args.csv

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*80}")
    print(f"🚀 STARTING 5-FOLD CROSS-VALIDATION TRAINING")
    print(f"{'='*80}")
    print(f"Device: {device}")
    print(f"Config: {args.config}")
    print(f"CSV: {args.csv}")
    print(f"Output: {args.output_dir}")
    if args.resume_fold:
        print(f"Resume from fold: {args.resume_fold}")
    if args.resume_checkpoint:
        print(f"Resume checkpoint: {args.resume_checkpoint}")
    print(f"{'='*80}\n")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'logs'), exist_ok=True)

    folds = prepare_folds(args.csv)
    start_fold = args.resume_fold if args.resume_fold else 1
    results = []
    for fi in range(start_fold, 6):
        resume_ckpt = args.resume_checkpoint if fi == start_fold else None
        res = train_fold(folds[fi-1], config, fi, args.output_dir, device, resume_ckpt, continue_epochs=args.continue_epochs)
        # After training the fold, load best_model.pt and run Test once (match wpro logic)
        checkpoint_dir = os.path.join(args.output_dir, f'fold_{fi}')
        best_path = os.path.join(checkpoint_dir, 'best_model.pt')
        if not os.path.exists(best_path):
            if resume_ckpt and os.path.exists(resume_ckpt):
                best_path = resume_ckpt
            else:
                raise FileNotFoundError(f"No best_model.pt found in {checkpoint_dir} and no resume checkpoint available.")

        # Rebuild model to run test
        model = WavLMBaseAV(
            num_emotions=config['model'].get('num_emotions',4),
            wavlm_model_name=config['model'].get('wavlm_model_name','microsoft/wavlm-base-plus'),
            hidden_dim=config['model'].get('hidden_dim',256),
            av_refinement_layers=config['model'].get('av_refinement_layers',2),
            freeze_wavlm=config['model'].get('freeze_wavlm',True)
        ).to(device)
        checkpoint = torch.load(best_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])

        criterion = CombinedEmotionLossCCC(
            emotion_weight=config['loss'].get('emotion_weight',1.0),
            arousal_weight=config['loss'].get('arousal_weight',1.0),
            valence_weight=config['loss'].get('valence_weight',1.0)
        )

        # Prepare test loader
        test_df = folds[fi-1]['test']
        test_dataset = IEMOCAPDataset(test_df.reset_index(drop=True), sample_rate=config['data']['sample_rate'], audio_max_length=config['data']['max_audio_length'])
        test_loader = DataLoader(test_dataset, batch_size=config['training']['batch_size'], shuffle=False, num_workers=config['training'].get('num_workers',2), collate_fn=collate_fn)

        test_metrics = validate(model, test_loader, criterion, device, writer=None, epoch=0, phase='Test')
        res['final_test_metrics'] = test_metrics
        results.append(res)
        print(f"Fold {fi} - Best Val Acc: {res['best_val_acc']:.4f}; Test Acc: {test_metrics['accuracy']:.4f} F1: {test_metrics['f1_macro']:.4f}")

    print("Training Summary")
    for r in results:
        print(f"Fold {r['fold']}: Best Val Acc = {r['best_val_acc']:.4f}, Test Acc = {r['final_test_metrics']['accuracy']:.4f}, Test F1 = {r['final_test_metrics']['f1_macro']:.4f}")
    print(f"Average Val Acc (5-Fold): {np.mean([r['best_val_acc'] for r in results]):.4f}")
    print(f"Average Test Acc (5-Fold): {np.mean([r['final_test_metrics']['accuracy'] for r in results]):.4f}")


if __name__ == '__main__':
    main()
