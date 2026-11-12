"""
Training script for Method 1 (WavLM + Cross-Attention)
Multi-task learning: Emotion classification + A-V regression
"""
import os
import sys
import argparse
import logging
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from thop import profile, clever_format

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.method1_triple_wavlm import Method1TripleWavLM
from train.losses import CombinedEmotionLoss, CombinedEmotionLossCCC
from data.iemocap_dataset import get_iemocap_dataloaders


def setup_logging(log_file):
    """Setup logging to file and console"""
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def count_parameters(model):
    """Count total and trainable parameters"""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = total - trainable
    return total, trainable, frozen


def compute_flops(model, audio_length=160000, device='cpu'):
    """Compute FLOPs for the model"""
    try:
        # Create dummy input
        audio_input = torch.randn(1, audio_length).to(device)
        
        # Compute FLOPs
        flops, params = profile(
            model, 
            inputs=(audio_input,),
            verbose=False
        )
        flops_str, params_str = clever_format([flops, params], "%.3f")
        return flops, flops_str, params_str
    except Exception as e:
        print(f"Warning: FLOPs computation failed: {e}")
        return None, "N/A", "N/A"


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
        # Move to device
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
        
        # Accumulate losses
        total_loss += loss.item()
        emotion_losses.append(losses['emotion_loss'].item())
        arousal_losses.append(losses['arousal_loss'].item())
        valence_losses.append(losses['valence_loss'].item())
        
        # Accumulate predictions
        emotion_preds = torch.argmax(outputs['emotion_logits'], dim=1)
        all_emotion_preds.extend(emotion_preds.cpu().numpy())
        all_emotion_labels.extend(emotion_labels.cpu().numpy())
        all_arousal_preds.extend(outputs['arousal_value'].detach().cpu().numpy().flatten())
        all_arousal_targets.extend(arousal_values.cpu().numpy().flatten())
        all_valence_preds.extend(outputs['valence_value'].detach().cpu().numpy().flatten())
        all_valence_targets.extend(valence_values.cpu().numpy().flatten())
        
        # Update progress bar
        pbar.set_postfix({
            'loss': loss.item(),
            'emo_loss': losses['emotion_loss'].item(),
            'ar_loss': losses['arousal_loss'].item(),
            'val_loss': losses['valence_loss'].item()
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
            
            # Forward pass
            outputs = model(audio)
            
            # Compute loss
            targets = {
                'emotion_label': emotion_labels,
                'arousal_value': arousal_values,
                'valence_value': valence_values
            }
            
            losses = criterion(outputs, targets)
            loss = losses['total_loss']
            
            # Accumulate losses
            total_loss += loss.item()
            emotion_losses.append(losses['emotion_loss'].item())
            arousal_losses.append(losses['arousal_loss'].item())
            valence_losses.append(losses['valence_loss'].item())
            
            # Accumulate predictions
            emotion_preds = torch.argmax(outputs['emotion_logits'], dim=1)
            all_emotion_preds.extend(emotion_preds.cpu().numpy())
            all_emotion_labels.extend(emotion_labels.cpu().numpy())
            all_arousal_preds.extend(outputs['arousal_value'].cpu().numpy().flatten())
            all_arousal_targets.extend(arousal_values.cpu().numpy().flatten())
            all_valence_preds.extend(outputs['valence_value'].cpu().numpy().flatten())
            all_valence_targets.extend(valence_values.cpu().numpy().flatten())
    
    # Compute metrics
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
    # Setup
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = f"method1_{timestamp}"
    
    # Create log directory
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"method1_{timestamp}.log"
    
    logger = setup_logging(str(log_file))
    logger.info("="*60)
    logger.info("Training Method 1 (WavLM + Cross-Attention)")
    logger.info("="*60)
    logger.info(f"Log file: {log_file}")
    logger.info(f"Arguments: {args}")
    
    # Device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Data
    logger.info("\nLoading data...")
    train_loader, test_loader = get_iemocap_dataloaders(
        train_csv=args.train_csv,
        test_csv=args.test_csv,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        audio_max_length=args.audio_max_length,
        use_text=False  # Method 1 is audio-only
    )
    logger.info(f"Train: {len(train_loader.dataset)} samples")
    logger.info(f"Test:  {len(test_loader.dataset)} samples")
    
    # Model
    logger.info("\nCreating model...")
    model = Method1TripleWavLM(
        num_emotions=args.num_emotions,
        wavlm_model_name=args.wavlm_model,
        hidden_dim=args.hidden_dim,
        av_refinement_layers=args.av_refinement_layers,
        freeze_wavlm=args.freeze_wavlm
    )
    model = model.to(device)
    
    # Count parameters
    total_params, trainable_params, frozen_params = count_parameters(model)
    logger.info("\n" + "="*60)
    logger.info("📊 MODEL STATISTICS")
    logger.info("="*60)
    logger.info(f"Total parameters:     {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    logger.info(f"Frozen parameters:    {frozen_params:,}")
    
    # Compute FLOPs
    logger.info("\n⚡ Computing FLOPs...")
    flops, flops_str, params_str = compute_flops(model, args.audio_max_length, device)
    if flops is not None:
        logger.info(f"FLOPs:  {flops_str}")
        logger.info(f"Params: {params_str}")
    
    # Architecture summary
    logger.info("\n" + "="*60)
    logger.info("🏗️  ARCHITECTURE SUMMARY")
    logger.info("="*60)
    logger.info("1. WavLM:                Extract features from audio")
    logger.info("2. Prosody Encoder:      Process WavLM → Cross-Attn → Fusion → Arousal")
    logger.info("3. Text Encoder:         Process WavLM → Cross-Attn → Fusion → Valence")
    logger.info("4. A-V Refinement:       Refine arousal/valence values")
    logger.info("5. Final Fusion:         WavLM + fusion(W+Prosody) + fusion(W+Text) + AV_refined")
    logger.info("6. Classification:       Final layers → Emotion")
    logger.info("="*60)
    
    # Loss function info
    logger.info("\n" + "="*60)
    logger.info("📉 LOSS FUNCTION")
    logger.info("="*60)
    if args.loss_type == 'ccc':
        logger.info("Type: CombinedEmotionLossCCC")
        logger.info(f"  L_total = {args.emotion_weight} * L_emotion")
        logger.info(f"          + {args.arousal_weight} * L_arousal_CCC")
        logger.info(f"          + {args.valence_weight} * L_valence_CCC")
        logger.info("")
        logger.info("Where:")
        logger.info("  L_emotion     = CrossEntropyLoss(logits, labels)")
        logger.info("  L_arousal_CCC = 1 - CCC(arousal_pred, arousal_true)")
        logger.info("  L_valence_CCC = 1 - CCC(valence_pred, valence_true)")
        logger.info("  CCC = (2*cov) / (var_pred + var_true + (mean_pred - mean_true)²)")
    else:
        logger.info("Type: CombinedEmotionLoss (MSE)")
        logger.info(f"  L_total = {args.emotion_weight} * L_emotion")
        logger.info(f"          + {args.arousal_weight} * MSE(arousal)")
        logger.info(f"          + {args.valence_weight} * MSE(valence)")
    logger.info("="*60 + "\n")
    
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
    else:
        raise ValueError(f"Unknown loss_type: {args.loss_type}")
    
    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )
    
    # Scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',
        factor=0.5,
        patience=5
    )
    
    # Tensorboard
    tb_dir = log_dir / 'tensorboard' / exp_name
    tb_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(tb_dir))
    logger.info(f"TensorBoard logs: {tb_dir}")
    
    # Training loop
    best_f1 = 0
    best_epoch = 0
    
    checkpoint_dir = Path(args.checkpoint_dir) / exp_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Checkpoints: {checkpoint_dir}\n")
    
    # Resume from checkpoint if provided
    start_epoch = 1
    if args.resume_checkpoint and os.path.exists(args.resume_checkpoint):
        logger.info(f"Resuming from checkpoint: {args.resume_checkpoint}")
        checkpoint = torch.load(args.resume_checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint.get('epoch', 0) + 1
        best_f1 = checkpoint.get('val_metrics', {}).get('f1_weighted', 0)
        logger.info(f"Resumed from epoch {start_epoch-1}, best F1: {best_f1:.4f}\n")
    
    for epoch in range(start_epoch, args.num_epochs + 1):
        logger.info(f"\n{'='*60}")
        logger.info(f"Epoch {epoch}/{args.num_epochs}")
        logger.info(f"{'='*60}")
        
        # Train
        train_metrics = train_epoch(model, train_loader, criterion, optimizer, device, epoch, logger)
        
        # Validate
        val_metrics = validate(model, test_loader, criterion, device, epoch, logger)
        
        # Scheduler step
        scheduler.step(val_metrics['f1_weighted'])
        
        # Tensorboard logging
        for key, value in train_metrics.items():
            writer.add_scalar(f'train/{key}', value, epoch)
        for key, value in val_metrics.items():
            writer.add_scalar(f'val/{key}', value, epoch)
        writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], epoch)
        
        # Save best model
        if val_metrics['f1_weighted'] > best_f1:
            best_f1 = val_metrics['f1_weighted']
            best_epoch = epoch
            
            checkpoint_path = checkpoint_dir / 'best_model.pt'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_metrics': val_metrics,
                'args': args
            }, checkpoint_path)
            logger.info(f"✓ Saved best model to {checkpoint_path}")
        
        # Save latest model
        checkpoint_path = checkpoint_dir / 'latest_model.pt'
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'val_metrics': val_metrics,
            'args': args
        }, checkpoint_path)
    
    logger.info(f"\n{'='*60}")
    logger.info(f"🎉 Training completed!")
    logger.info(f"Best F1: {best_f1:.4f} at epoch {best_epoch}")
    logger.info(f"{'='*60}")
    
    writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train Method 1 (Triple WavLM)')
    
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
    parser.add_argument('--device', type=str, default='cuda:3')
    parser.add_argument('--resume_checkpoint', type=str, default=None,
                       help='Path to checkpoint to resume from')
    
    args = parser.parse_args()
    main(args)
