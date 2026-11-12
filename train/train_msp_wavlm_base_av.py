"""
Training script for WavLM-Base-Plus A-V Model on MSP-Podcast dataset
Simple single-branch architecture with frozen WavLM backbone
Fixed splits: train/dev/test1/test2 (no k-fold)
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
from sklearn.metrics import accuracy_score, f1_score, classification_report
from torch.utils.tensorboard import SummaryWriter
from thop import profile, clever_format

# Add parent directory to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from models.wavlm_base_av import WavLMBaseAV
from train.losses import CombinedEmotionLossCCC
from data.msp_podcast_dataset import get_msp_podcast_dataloaders


def load_config(config_path):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def compute_ccc(y_true, y_pred):
    """Compute Concordance Correlation Coefficient"""
    mean_true = np.mean(y_true)
    mean_pred = np.mean(y_pred)
    var_true = np.var(y_true)
    var_pred = np.var(y_pred)
    covariance = np.mean((y_true - mean_true) * (y_pred - mean_pred))
    ccc = (2 * covariance) / (var_true + var_pred + (mean_true - mean_pred) ** 2 + 1e-8)
    return ccc


def count_parameters(model):
    """Count total and trainable parameters"""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def compute_flops(model, audio_length=16000*10, device='cpu'):
    """Compute FLOPs for the model"""
    try:
        audio_input = torch.randn(1, audio_length, device=device)
        macs, params = profile(model, inputs=(audio_input,), verbose=False)
        flops = macs * 2
        flops_str = clever_format([flops], "%.2f")
        return flops_str[0]
    except Exception as e:
        return "N/A"


def train_one_epoch(model, loader, criterion, optimizer, device, writer, global_step, epoch, logger):
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
    
    pbar = tqdm(loader, desc=f"Epoch {epoch} [Train]")
    for batch_idx, batch in enumerate(pbar):
        audio = batch['audio'].to(device)
        emotion_labels = batch['emotion_label'].to(device)
        arousal_targets = batch['arousal_value'].to(device)
        valence_targets = batch['valence_value'].to(device)
        
        # Forward pass
        outputs = model(audio)
        
        # Compute loss
        targets = {
            'emotion_label': emotion_labels,
            'arousal_value': arousal_targets,
            'valence_value': valence_targets
        }
        
        loss_dict = criterion(outputs, targets)
        loss = loss_dict['total_loss']
        losses_dict = loss_dict
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        # Record metrics
        total_loss += loss.item()
        emotion_losses.append(losses_dict['emotion_loss'].item())
        arousal_losses.append(losses_dict['arousal_loss'].item())
        valence_losses.append(losses_dict['valence_loss'].item())
        
        # Get predictions
        emotion_pred = torch.argmax(outputs['emotion_logits'], dim=1).cpu().numpy()
        emotion_label = emotion_labels.cpu().numpy()
        arousal_pred = outputs['arousal_value'].squeeze(-1).detach().cpu().numpy()
        arousal_target = arousal_targets.squeeze(-1).cpu().numpy()
        valence_pred = outputs['valence_value'].squeeze(-1).detach().cpu().numpy()
        valence_target = valence_targets.squeeze(-1).cpu().numpy()
        
        all_emotion_preds.extend(emotion_pred)
        all_emotion_labels.extend(emotion_label)
        all_arousal_preds.extend(arousal_pred)
        all_arousal_targets.extend(arousal_target)
        all_valence_preds.extend(valence_pred)
        all_valence_targets.extend(valence_target)
        
        pbar.update(1)
        pbar.set_postfix({'loss': loss.item()})
        
        if writer:
            writer.add_scalar('Train/Loss', loss.item(), global_step)
            writer.add_scalar('Train/Emotion Loss', losses_dict['emotion_loss'], global_step)
            writer.add_scalar('Train/Arousal Loss', losses_dict['arousal_loss'], global_step)
            writer.add_scalar('Train/Valence Loss', losses_dict['valence_loss'], global_step)
            global_step += 1
    
    # Compute epoch metrics
    avg_loss = total_loss / len(loader)
    accuracy = accuracy_score(all_emotion_labels, all_emotion_preds)
    f1_macro = f1_score(all_emotion_labels, all_emotion_preds, average='macro')
    f1_weighted = f1_score(all_emotion_labels, all_emotion_preds, average='weighted')
    
    arousal_ccc = compute_ccc(np.array(all_arousal_targets), np.array(all_arousal_preds))
    valence_ccc = compute_ccc(np.array(all_valence_targets), np.array(all_valence_preds))
    avg_ccc = (arousal_ccc + valence_ccc) / 2
    
    # Log epoch metrics
    logger.info(f"Epoch {epoch} Train - Loss: {avg_loss:.4f}, Acc: {accuracy:.4f}, "
                f"F1 Macro: {f1_macro:.4f}, Arousal CCC: {arousal_ccc:.4f}, "
                f"Valence CCC: {valence_ccc:.4f}, Avg CCC: {avg_ccc:.4f}")
    
    if writer:
        writer.add_scalar('Train/Epoch Loss', avg_loss, epoch)
        writer.add_scalar('Train/Epoch Accuracy', accuracy, epoch)
        writer.add_scalar('Train/Epoch F1 Macro', f1_macro, epoch)
        writer.add_scalar('Train/Epoch Arousal CCC', arousal_ccc, epoch)
        writer.add_scalar('Train/Epoch Valence CCC', valence_ccc, epoch)
        writer.add_scalar('Train/Epoch Avg CCC', avg_ccc, epoch)
    
    return global_step


def validate(model, loader, criterion, device, writer, epoch, phase='Dev', logger=None):
    """Validate/Test the model"""
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
        pbar = tqdm(loader, desc=f"Epoch {epoch} [{phase}]")
        for batch in pbar:
            audio = batch['audio'].to(device)
            emotion_labels = batch['emotion_label'].to(device)
            arousal_targets = batch['arousal_value'].to(device)
            valence_targets = batch['valence_value'].to(device)
            
            # Forward pass
            outputs = model(audio)
            
            # Compute loss
            targets = {
                'emotion_label': emotion_labels,
                'arousal_value': arousal_targets,
                'valence_value': valence_targets
            }
            
            loss_dict = criterion(outputs, targets)
            loss = loss_dict['total_loss']
            losses_dict = loss_dict
            
            total_loss += loss.item()
            emotion_losses.append(losses_dict['emotion_loss'].item())
            arousal_losses.append(losses_dict['arousal_loss'].item())
            valence_losses.append(losses_dict['valence_loss'].item())
            
            # Get predictions
            emotion_pred = torch.argmax(outputs['emotion_logits'], dim=1).cpu().numpy()
            emotion_label = emotion_labels.cpu().numpy()
            arousal_pred = outputs['arousal_value'].squeeze(-1).detach().cpu().numpy()
            arousal_target = arousal_targets.squeeze(-1).cpu().numpy()
            valence_pred = outputs['valence_value'].squeeze(-1).detach().cpu().numpy()
            valence_target = valence_targets.squeeze(-1).cpu().numpy()
            
            all_emotion_preds.extend(emotion_pred)
            all_emotion_labels.extend(emotion_label)
            all_arousal_preds.extend(arousal_pred)
            all_arousal_targets.extend(arousal_target)
            all_valence_preds.extend(valence_pred)
            all_valence_targets.extend(valence_target)
            
            pbar.update(1)
    
    # Compute metrics
    avg_loss = total_loss / len(loader)
    accuracy = accuracy_score(all_emotion_labels, all_emotion_preds)
    f1_macro = f1_score(all_emotion_labels, all_emotion_preds, average='macro')
    f1_weighted = f1_score(all_emotion_labels, all_emotion_preds, average='weighted')
    
    arousal_ccc = compute_ccc(np.array(all_arousal_targets), np.array(all_arousal_preds))
    valence_ccc = compute_ccc(np.array(all_valence_targets), np.array(all_valence_preds))
    avg_ccc = (arousal_ccc + valence_ccc) / 2
    
    # Log metrics
    if logger:
        logger.info(f"Epoch {epoch} {phase} - Loss: {avg_loss:.4f}, Acc: {accuracy:.4f}, "
                    f"F1 Macro: {f1_macro:.4f}, Arousal CCC: {arousal_ccc:.4f}, "
                    f"Valence CCC: {valence_ccc:.4f}, Avg CCC: {avg_ccc:.4f}")
    
    if writer:
        writer.add_scalar(f'{phase}/Loss', avg_loss, epoch)
        writer.add_scalar(f'{phase}/Accuracy', accuracy, epoch)
        writer.add_scalar(f'{phase}/F1 Macro', f1_macro, epoch)
        writer.add_scalar(f'{phase}/Arousal CCC', arousal_ccc, epoch)
        writer.add_scalar(f'{phase}/Valence CCC', valence_ccc, epoch)
        writer.add_scalar(f'{phase}/Avg CCC', avg_ccc, epoch)
    
    return {
        'loss': avg_loss,
        'accuracy': accuracy,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted,
        'arousal_ccc': arousal_ccc,
        'valence_ccc': valence_ccc,
        'avg_ccc': avg_ccc
    }


def main():
    parser = argparse.ArgumentParser(description='Train WavLM-Base A-V on MSP-Podcast')
    parser.add_argument('--config', type=str, default='configs/msp_podcast_config.yaml',
                        help='Path to config file')
    parser.add_argument('--tsv', type=str, default='datasets/MSP-Podcast/msp_podcast.tsv',
                        help='Path to MSP-Podcast TSV file')
    parser.add_argument('--output_dir', type=str, default='checkpoints/msp_wavlm_base_av',
                        help='Output directory for checkpoints')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='Device to use')
    parser.add_argument('--resume', type=str, default=None,
                        help='Resume from checkpoint')
    
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Setup device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
    # Create output directories
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir = output_dir / 'logs'
    log_dir.mkdir(exist_ok=True)
    
    # Setup logging
    log_file = log_dir / f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)
    
    logger.info("="*80)
    logger.info("WavLM-Base A-V Model Training on MSP-Podcast")
    logger.info("="*80)
    logger.info(f"Device: {device}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Config file: {args.config}")
    logger.info(f"TSV file: {args.tsv}")
    
    # Create model
    model = WavLMBaseAV(
        num_emotions=config['model'].get('num_emotions', 4),
        wavlm_model_name=config['model'].get('wavlm_model_name', 'microsoft/wavlm-base-plus'),
        hidden_dim=config['model'].get('hidden_dim', 256),
        av_refinement_layers=config['model'].get('av_refinement_layers', 2),
        freeze_wavlm=config['model'].get('freeze_wavlm', True)
    )
    
    model.to(device)
    
    # Count parameters
    total_params, trainable_params = count_parameters(model)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    logger.info(f"Frozen parameters: {total_params - trainable_params:,}")
    
    # Compute FLOPs
    flops = compute_flops(model, device=device)
    logger.info(f"FLOPs: {flops}")
    
    # Create dataloaders
    logger.info("Loading MSP-Podcast dataset...")
    train_loader, dev_loader, test1_loader, test2_loader = get_msp_podcast_dataloaders(
        tsv_path=args.tsv,
        batch_size=config['training'].get('batch_size', 4),
        num_workers=config['training'].get('num_workers', 4),
        audio_max_length=config['data'].get('max_audio_length', 160000)
    )
    
    logger.info(f"Train samples: {len(train_loader.dataset)}")
    logger.info(f"Dev samples: {len(dev_loader.dataset)}")
    logger.info(f"Test1 samples: {len(test1_loader.dataset)}")
    logger.info(f"Test2 samples: {len(test2_loader.dataset)}")
    
    # Create loss function
    criterion = CombinedEmotionLossCCC(
        emotion_weight=config['loss'].get('emotion_weight', 1.0),
        arousal_weight=config['loss'].get('arousal_weight', 1.0),
        valence_weight=config['loss'].get('valence_weight', 1.0)
    )
    
    # Create optimizer
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config['training'].get('learning_rate', 0.0001),
        weight_decay=config['training'].get('weight_decay', 0.00001),
        betas=config['optimizer'].get('betas', [0.9, 0.999])
    )
    
    # Create scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode=config['scheduler'].get('mode', 'max'),
        factor=config['scheduler'].get('factor', 0.5),
        patience=config['scheduler'].get('patience', 3),
        min_lr=config['scheduler'].get('min_lr', 1e-6)
    )
    
    # TensorBoard writer
    tb_dir = Path(config['paths'].get('tensorboard_dir', 'logs/tensorboard')) / 'msp_wavlm_base_av'
    tb_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(str(tb_dir))
    
    # Training loop
    epochs = config['training'].get('epochs', 10)
    start_epoch = 0
    best_avg_ccc = -1.0
    global_step = 0
    
    # Load checkpoint if resuming
    if args.resume:
        logger.info(f"Loading checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_avg_ccc = checkpoint.get('best_avg_ccc', -1.0)
        global_step = checkpoint.get('global_step', 0)
        logger.info(f"Resumed from epoch {start_epoch}, best CCC: {best_avg_ccc:.4f}")
    
    logger.info("="*80)
    logger.info("Starting training...")
    logger.info("="*80)
    
    for epoch in range(start_epoch, epochs):
        logger.info(f"\nEpoch {epoch+1}/{epochs}")
        
        # Train
        global_step = train_one_epoch(model, train_loader, criterion, optimizer, device, 
                                     writer, global_step, epoch+1, logger)
        
        # Validate on dev set
        dev_metrics = validate(model, dev_loader, criterion, device, writer, epoch+1, 'Dev', logger)
        
        # Test on test1 set
        test1_metrics = validate(model, test1_loader, criterion, device, writer, epoch+1, 'Test1', logger)
        
        # Test on test2 set
        test2_metrics = validate(model, test2_loader, criterion, device, writer, epoch+1, 'Test2', logger)
        
        # Update scheduler
        scheduler.step(dev_metrics['avg_ccc'])
        
        # Save checkpoint
        is_best = dev_metrics['avg_ccc'] > best_avg_ccc
        if is_best:
            best_avg_ccc = dev_metrics['avg_ccc']
            logger.info(f"New best CCC: {best_avg_ccc:.4f}")
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_avg_ccc': best_avg_ccc,
            'global_step': global_step,
            'config': config
        }
        
        checkpoint_path = output_dir / f'checkpoint_epoch{epoch+1}.pt'
        torch.save(checkpoint, checkpoint_path)
        
        if is_best:
            best_checkpoint_path = output_dir / 'checkpoint_best.pt'
            torch.save(checkpoint, best_checkpoint_path)
            logger.info(f"Saved best checkpoint: {best_checkpoint_path}")
    
    logger.info("="*80)
    logger.info("Training completed!")
    logger.info(f"Best Dev CCC: {best_avg_ccc:.4f}")
    logger.info("="*80)
    
    writer.close()


if __name__ == '__main__':
    main()
