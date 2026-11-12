"""
Training script for Method 2 (Cross-Attention A-V)
More efficient: WavLM used only 2 times
"""
import os
import sys
import argparse
import logging
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.method2_crossattn_av import Method2CrossAttentionAV
from train.losses import CombinedEmotionLoss, CombinedEmotionLossCCC
from data.iemocap_dataset import get_iemocap_dataloaders


def setup_logging(log_dir, exp_name):
    """Setup logging"""
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"{exp_name}.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def compute_metrics(all_preds, all_labels):
    """Compute classification metrics"""
    acc = accuracy_score(all_labels, all_preds)
    f1_weighted = f1_score(all_labels, all_preds, average='weighted')
    f1_macro = f1_score(all_labels, all_preds, average='macro')
    
    return {
        'accuracy': acc,
        'f1_weighted': f1_weighted,
        'f1_macro': f1_macro
    }


def compute_ccc(pred, target):
    """Compute Concordance Correlation Coefficient"""
    pred_mean = np.mean(pred)
    target_mean = np.mean(target)
    pred_var = np.var(pred)
    target_var = np.var(target)
    covariance = np.mean((pred - pred_mean) * (target - target_mean))
    
    ccc = (2 * covariance) / (pred_var + target_var + (pred_mean - target_mean) ** 2 + 1e-8)
    return ccc


def train_epoch(model, train_loader, criterion, optimizer, device, epoch, logger):
    """Train for one epoch"""
    model.train()
    
    total_loss = 0
    emotion_losses = []
    arousal_losses = []
    valence_losses = []
    
    all_emotion_preds = []
    all_emotion_labels = []
    all_arousal_preds = []
    all_arousal_targets = []
    all_valence_preds = []
    all_valence_targets = []
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch} [Train]")
    for batch_idx, batch in enumerate(pbar):
        audio = batch['audio'].to(device)
        emotion_labels = batch['emotion_label'].to(device)
        arousal_values = batch['arousal_value'].to(device)
        valence_values = batch['valence_value'].to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(audio)
        
        # Compute loss
        targets = {
            'emotion_label': emotion_labels,
            'arousal_value': arousal_values,
            'valence_value': valence_values
        }
        
        losses = criterion(outputs, targets)
        loss = losses['total_loss']
        
        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        # Accumulate
        total_loss += loss.item()
        emotion_losses.append(losses['emotion_loss'].item())
        arousal_losses.append(losses['arousal_loss'].item())
        valence_losses.append(losses['valence_loss'].item())
        
        emotion_preds = torch.argmax(outputs['emotion_logits'], dim=1)
        all_emotion_preds.extend(emotion_preds.cpu().numpy())
        all_emotion_labels.extend(emotion_labels.cpu().numpy())
        all_arousal_preds.extend(outputs['arousal_value'].detach().cpu().numpy().flatten())
        all_arousal_targets.extend(arousal_values.cpu().numpy().flatten())
        all_valence_preds.extend(outputs['valence_value'].detach().cpu().numpy().flatten())
        all_valence_targets.extend(valence_values.cpu().numpy().flatten())
        
        pbar.set_postfix({
            'loss': loss.item(),
            'emo': losses['emotion_loss'].item(),
            'ar': losses['arousal_loss'].item(),
            'val': losses['valence_loss'].item()
        })
    
    # Compute metrics
    metrics = compute_metrics(all_emotion_preds, all_emotion_labels)
    arousal_ccc = compute_ccc(np.array(all_arousal_preds), np.array(all_arousal_targets))
    valence_ccc = compute_ccc(np.array(all_valence_preds), np.array(all_valence_targets))
    
    avg_loss = total_loss / len(train_loader)
    
    logger.info(f"Epoch {epoch} [Train] - Loss: {avg_loss:.4f}, "
                f"Acc: {metrics['accuracy']:.4f}, F1: {metrics['f1_weighted']:.4f}, "
                f"A-CCC: {arousal_ccc:.4f}, V-CCC: {valence_ccc:.4f}")
    
    return {
        'loss': avg_loss,
        'emotion_loss': np.mean(emotion_losses),
        'arousal_loss': np.mean(arousal_losses),
        'valence_loss': np.mean(valence_losses),
        'accuracy': metrics['accuracy'],
        'f1_weighted': metrics['f1_weighted'],
        'f1_macro': metrics['f1_macro'],
        'arousal_ccc': arousal_ccc,
        'valence_ccc': valence_ccc
    }


def validate(model, val_loader, criterion, device, epoch, logger):
    """Validate the model"""
    model.eval()
    
    total_loss = 0
    emotion_losses = []
    arousal_losses = []
    valence_losses = []
    
    all_emotion_preds = []
    all_emotion_labels = []
    all_arousal_preds = []
    all_arousal_targets = []
    all_valence_preds = []
    all_valence_targets = []
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc=f"Epoch {epoch} [Val]")
        for batch in pbar:
            audio = batch['audio'].to(device)
            emotion_labels = batch['emotion_label'].to(device)
            arousal_values = batch['arousal_value'].to(device)
            valence_values = batch['valence_value'].to(device)
            
            outputs = model(audio)
            
            targets = {
                'emotion_label': emotion_labels,
                'arousal_value': arousal_values,
                'valence_value': valence_values
            }
            
            losses = criterion(outputs, targets)
            loss = losses['total_loss']
            
            total_loss += loss.item()
            emotion_losses.append(losses['emotion_loss'].item())
            arousal_losses.append(losses['arousal_loss'].item())
            valence_losses.append(losses['valence_loss'].item())
            
            emotion_preds = torch.argmax(outputs['emotion_logits'], dim=1)
            all_emotion_preds.extend(emotion_preds.cpu().numpy())
            all_emotion_labels.extend(emotion_labels.cpu().numpy())
            all_arousal_preds.extend(outputs['arousal_value'].cpu().numpy().flatten())
            all_arousal_targets.extend(arousal_values.cpu().numpy().flatten())
            all_valence_preds.extend(outputs['valence_value'].cpu().numpy().flatten())
            all_valence_targets.extend(valence_values.cpu().numpy().flatten())
    
    metrics = compute_metrics(all_emotion_preds, all_emotion_labels)
    arousal_ccc = compute_ccc(np.array(all_arousal_preds), np.array(all_arousal_targets))
    valence_ccc = compute_ccc(np.array(all_valence_preds), np.array(all_valence_targets))
    
    avg_loss = total_loss / len(val_loader)
    
    logger.info(f"Epoch {epoch} [Val] - Loss: {avg_loss:.4f}, "
                f"Acc: {metrics['accuracy']:.4f}, F1: {metrics['f1_weighted']:.4f}, "
                f"A-CCC: {arousal_ccc:.4f}, V-CCC: {valence_ccc:.4f}")
    
    return {
        'loss': avg_loss,
        'emotion_loss': np.mean(emotion_losses),
        'arousal_loss': np.mean(arousal_losses),
        'valence_loss': np.mean(valence_losses),
        'accuracy': metrics['accuracy'],
        'f1_weighted': metrics['f1_weighted'],
        'f1_macro': metrics['f1_macro'],
        'arousal_ccc': arousal_ccc,
        'valence_ccc': valence_ccc
    }


def main(args):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = f"method2_crossattn_av_{timestamp}"
    
    logger = setup_logging(args.log_dir, exp_name)
    logger.info(f"Experiment: {exp_name}")
    logger.info(f"Arguments: {args}")
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Data
    logger.info("Loading data...")
    train_loader, test_loader = get_iemocap_dataloaders(
        train_csv=args.train_csv,
        test_csv=args.test_csv,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        audio_max_length=args.audio_max_length
    )
    
    # Model
    logger.info("Creating model...")
    model = Method2CrossAttentionAV(
        num_emotions=args.num_emotions,
        wavlm_model_name=args.wavlm_model,
        hidden_dim=args.hidden_dim,
        av_refinement_layers=args.av_refinement_layers,
        freeze_wavlm=args.freeze_wavlm,
        num_cross_attn_heads=args.num_cross_attn_heads
    )
    model = model.to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    
    # Loss
    if args.loss_type == 'mse':
        criterion = CombinedEmotionLoss(
            emotion_weight=args.emotion_weight,
            arousal_weight=args.arousal_weight,
            valence_weight=args.valence_weight,
            regression_loss='mse'
        )
    elif args.loss_type == 'ccc':
        criterion = CombinedEmotionLossCCC(
            emotion_weight=args.emotion_weight,
            arousal_weight=args.arousal_weight,
            valence_weight=args.valence_weight
        )
    
    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5, verbose=True
    )
    
    writer = SummaryWriter(log_dir=os.path.join(args.log_dir, 'tensorboard', exp_name))
    
    best_f1 = 0
    best_epoch = 0
    checkpoint_dir = os.path.join(args.checkpoint_dir, exp_name)
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    for epoch in range(1, args.num_epochs + 1):
        logger.info(f"\n{'='*50}")
        logger.info(f"Epoch {epoch}/{args.num_epochs}")
        logger.info(f"{'='*50}")
        
        train_metrics = train_epoch(model, train_loader, criterion, optimizer, device, epoch, logger)
        val_metrics = validate(model, test_loader, criterion, device, epoch, logger)
        
        scheduler.step(val_metrics['f1_weighted'])
        
        for key, value in train_metrics.items():
            writer.add_scalar(f'train/{key}', value, epoch)
        for key, value in val_metrics.items():
            writer.add_scalar(f'val/{key}', value, epoch)
        writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], epoch)
        
        if val_metrics['f1_weighted'] > best_f1:
            best_f1 = val_metrics['f1_weighted']
            best_epoch = epoch
            
            checkpoint_path = os.path.join(checkpoint_dir, 'best_model.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_metrics': val_metrics,
                'args': args
            }, checkpoint_path)
            logger.info(f"Saved best model to {checkpoint_path}")
        
        checkpoint_path = os.path.join(checkpoint_dir, 'latest_model.pt')
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_metrics': val_metrics,
            'args': args
        }, checkpoint_path)
    
    logger.info(f"\nBest F1: {best_f1:.4f} at epoch {best_epoch}")
    writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train Method 2 (Cross-Attention A-V)')
    
    # Data
    parser.add_argument('--train_csv', type=str,
                        default='/home4/quanpn/ICME2026/AVD_Test/datasets/IEMOCAP_KFOLD/IEMOCAP_train4classes.csv')
    parser.add_argument('--test_csv', type=str,
                        default='/home4/quanpn/ICME2026/AVD_Test/datasets/IEMOCAP_KFOLD/IEMOCAP_test4classes.csv')
    parser.add_argument('--audio_max_length', type=int, default=16000 * 10)
    
    # Model
    parser.add_argument('--wavlm_model', type=str, default='microsoft/wavlm-base-plus')
    parser.add_argument('--num_emotions', type=int, default=4)
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--av_refinement_layers', type=int, default=2)
    parser.add_argument('--freeze_wavlm', action='store_true')
    parser.add_argument('--num_cross_attn_heads', type=int, default=8)
    
    # Training
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_epochs', type=int, default=50)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--num_workers', type=int, default=4)
    
    # Loss
    parser.add_argument('--loss_type', type=str, default='ccc', choices=['mse', 'ccc'],
                       help='Loss type for A-V regression. CCC is recommended.')
    parser.add_argument('--emotion_weight', type=float, default=1.0)
    parser.add_argument('--arousal_weight', type=float, default=1.0)
    parser.add_argument('--valence_weight', type=float, default=1.0)
    
    # Paths
    parser.add_argument('--log_dir', type=str, default='logs')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints')
    parser.add_argument('--device', type=str, default='cuda:0')
    
    args = parser.parse_args()
    main(args)
