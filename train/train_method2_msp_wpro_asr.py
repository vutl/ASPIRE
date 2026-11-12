"""
Training script for Method 2 W-PRO with ASR on MSP-Podcast
Uses Whisper ASR transcripts instead of ground-truth transcripts
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
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.tensorboard import SummaryWriter

# Add parent directory to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from models.method2_crossattn_av import Method2CrossAttentionAV
from train.losses import CombinedEmotionLossCCC
from data.msp_podcast_asr_dataset import get_msp_podcast_asr_dataloaders


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


def train_one_epoch(model, loader, criterion, optimizer, device, writer, global_step, epoch, logger):
    """Train for one epoch"""
    model.train()
    
    total_loss = 0
    all_emotion_preds = []
    all_emotion_labels = []
    all_arousal_preds = []
    all_arousal_targets = []
    all_valence_preds = []
    all_valence_targets = []
    
    pbar = tqdm(loader, desc=f"Epoch {epoch} [Train]")
    for batch_idx, batch in enumerate(pbar):
        audio = batch['audio'].to(device)
        text_input_ids = batch['text_input_ids'].to(device)
        text_attention_mask = batch['text_attention_mask'].to(device)
        emotion_labels = batch['emotion_label'].to(device)
        arousal_values = batch['arousal_value'].to(device)
        valence_values = batch['valence_value'].to(device)
        
        optimizer.zero_grad()
        outputs = model(
            audio_input=audio,
            text_input=text_input_ids,
            text_attention_mask=text_attention_mask
        )
        
        targets = {
            'emotion_label': emotion_labels,
            'arousal_value': arousal_values,
            'valence_value': valence_values
        }
        
        loss_dict = criterion(outputs, targets)
        loss = loss_dict['total_loss']
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
        
        preds = torch.argmax(outputs['emotion_logits'], dim=1)
        all_emotion_preds.extend(preds.cpu().numpy())
        all_emotion_labels.extend(emotion_labels.cpu().numpy())
        all_arousal_preds.extend(outputs['arousal_value'].detach().cpu().numpy().flatten())
        all_arousal_targets.extend(arousal_values.cpu().numpy().flatten())
        all_valence_preds.extend(outputs['valence_value'].detach().cpu().numpy().flatten())
        all_valence_targets.extend(valence_values.cpu().numpy().flatten())
        
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{accuracy_score(all_emotion_labels, all_emotion_preds):.4f}'
        })
        
        if writer:
            writer.add_scalar('Train/batch_loss', loss.item(), global_step)
            global_step += 1
    
    avg_loss = total_loss / len(loader)
    accuracy = accuracy_score(all_emotion_labels, all_emotion_preds)
    f1_macro = f1_score(all_emotion_labels, all_emotion_preds, average='macro')
    arousal_ccc = compute_ccc(np.array(all_arousal_targets), np.array(all_arousal_preds))
    valence_ccc = compute_ccc(np.array(all_valence_targets), np.array(all_valence_preds))
    
    logger.info(f"Epoch {epoch} Train - Loss: {avg_loss:.4f}, Acc: {accuracy:.4f}, "
                f"F1: {f1_macro:.4f}, A-CCC: {arousal_ccc:.4f}, V-CCC: {valence_ccc:.4f}")
    
    if writer:
        writer.add_scalar('Train/loss', avg_loss, epoch)
        writer.add_scalar('Train/accuracy', accuracy, epoch)
        writer.add_scalar('Train/f1_macro', f1_macro, epoch)
        writer.add_scalar('Train/arousal_ccc', arousal_ccc, epoch)
        writer.add_scalar('Train/valence_ccc', valence_ccc, epoch)
    
    return avg_loss, accuracy, global_step


def validate(model, loader, criterion, device, writer, epoch, phase='Dev', logger=None):
    """Validate model"""
    model.eval()
    
    total_loss = 0
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
            text_input_ids = batch['text_input_ids'].to(device)
            text_attention_mask = batch['text_attention_mask'].to(device)
            emotion_labels = batch['emotion_label'].to(device)
            arousal_values = batch['arousal_value'].to(device)
            valence_values = batch['valence_value'].to(device)
            
            outputs = model(
                audio_input=audio,
                text_input=text_input_ids,
                text_attention_mask=text_attention_mask
            )
            
            targets = {
                'emotion_label': emotion_labels,
                'arousal_value': arousal_values,
                'valence_value': valence_values
            }
            
            loss_dict = criterion(outputs, targets)
            loss = loss_dict['total_loss']
            
            total_loss += loss.item()
            
            preds = torch.argmax(outputs['emotion_logits'], dim=1)
            all_emotion_preds.extend(preds.cpu().numpy())
            all_emotion_labels.extend(emotion_labels.cpu().numpy())
            all_arousal_preds.extend(outputs['arousal_value'].cpu().numpy().flatten())
            all_arousal_targets.extend(arousal_values.cpu().numpy().flatten())
            all_valence_preds.extend(outputs['valence_value'].cpu().numpy().flatten())
            all_valence_targets.extend(valence_values.cpu().numpy().flatten())
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{accuracy_score(all_emotion_labels, all_emotion_preds):.4f}'
            })
    
    avg_loss = total_loss / len(loader)
    accuracy = accuracy_score(all_emotion_labels, all_emotion_preds)
    f1_macro = f1_score(all_emotion_labels, all_emotion_preds, average='macro')
    f1_weighted = f1_score(all_emotion_labels, all_emotion_preds, average='weighted')
    arousal_ccc = compute_ccc(np.array(all_arousal_targets), np.array(all_arousal_preds))
    valence_ccc = compute_ccc(np.array(all_valence_targets), np.array(all_valence_preds))
    avg_ccc = (arousal_ccc + valence_ccc) / 2
    
    if logger:
        logger.info(f"Epoch {epoch} {phase} - Loss: {avg_loss:.4f}, Acc: {accuracy:.4f}, "
                    f"F1: {f1_macro:.4f}, A-CCC: {arousal_ccc:.4f}, V-CCC: {valence_ccc:.4f}")
    
    if writer:
        writer.add_scalar(f'{phase}/loss', avg_loss, epoch)
        writer.add_scalar(f'{phase}/accuracy', accuracy, epoch)
        writer.add_scalar(f'{phase}/f1_macro', f1_macro, epoch)
        writer.add_scalar(f'{phase}/arousal_ccc', arousal_ccc, epoch)
        writer.add_scalar(f'{phase}/valence_ccc', valence_ccc, epoch)
    
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
    parser = argparse.ArgumentParser(description='Train Method 2 W-PRO with ASR on MSP-Podcast')
    parser.add_argument('--config', type=str, default='configs/method2_wpro_asr_msp_config.yaml')
    parser.add_argument('--tsv', type=str, default='datasets/MSP-Podcast/msp_podcast_asr.tsv')
    parser.add_argument('--output_dir', type=str, default='checkpoints/method2_msp_wpro_asr')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--resume', type=str, default=None)
    
    args = parser.parse_args()
    
    config = load_config(args.config)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    log_dir = output_dir / 'logs'
    log_dir.mkdir(exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_dir / f'train_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)
    
    logger.info(f"Config: {args.config}")
    logger.info(f"TSV: {args.tsv}")
    logger.info(f"Output: {args.output_dir}")
    logger.info(f"Device: {device}")
    
    # Create dataloaders
    logger.info("Creating dataloaders...")
    train_loader, dev_loader, test1_loader, test2_loader = get_msp_podcast_asr_dataloaders(
        tsv_path=args.tsv,
        batch_size=config['training']['batch_size'],
        num_workers=config['training']['num_workers'],
        audio_max_length=config['data']['max_audio_length'],
        max_text_length=config['data']['max_text_length'],
        asr_column=config['data'].get('asr_column', 'asr_transcript')
    )
    
    # Create model
    logger.info("Creating model...")
    model = Method2CrossAttentionAV(
        num_emotions=config['model']['num_emotions'],
        wavlm_model_name=config['model']['wavlm_model_name'],
        wavlm_prosody_model_name=config['model'].get('wavlm_prosody_model_name'),
        roberta_model_name=config['model']['roberta_model_name'],
        hidden_dim=config['model']['hidden_dim'],
        av_refinement_layers=config['model']['av_refinement_layers'],
        freeze_wavlm=config['model']['freeze_wavlm'],
        freeze_roberta=config['model']['freeze_roberta'],
        num_cross_attn_heads=config['model']['num_cross_attn_heads'],
        post_attention_layers=config['model']['post_attention_layers'],
        av_prediction_layers=config['model']['av_prediction_layers']
    ).to(device)
    
    # Loss and optimizer
    criterion = CombinedEmotionLossCCC(
        emotion_weight=config['loss']['emotion_weight'],
        arousal_weight=config['loss']['arousal_weight'],
        valence_weight=config['loss']['valence_weight']
    )
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',
        factor=config['scheduler']['factor'],
        patience=config['scheduler']['patience'],
        min_lr=config['scheduler']['min_lr']
    )
    
    # Resume if needed
    start_epoch = 1
    best_val_acc = 0.0
    if args.resume and os.path.exists(args.resume):
        logger.info(f"Resuming from {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_acc = checkpoint.get('best_val_acc', 0.0)
    
    # TensorBoard
    writer = SummaryWriter(log_dir / 'tensorboard')
    
    # Training loop
    global_step = 0
    best_epoch = 0
    
    for epoch in range(start_epoch, config['training']['epochs'] + 1):
        logger.info(f"\n{'='*80}")
        logger.info(f"Epoch {epoch}/{config['training']['epochs']}")
        logger.info(f"{'='*80}")
        
        # Train
        train_loss, train_acc, global_step = train_one_epoch(
            model, train_loader, criterion, optimizer, device, writer, global_step, epoch, logger
        )
        
        # Validate
        dev_metrics = validate(
            model, dev_loader, criterion, device, writer, epoch, 'Dev', logger
        )
        
        # Scheduler step
        scheduler.step(dev_metrics['accuracy'])
        
        # Save checkpoint
        if dev_metrics['accuracy'] > best_val_acc:
            best_val_acc = dev_metrics['accuracy']
            best_epoch = epoch
            
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_acc': best_val_acc,
                'config': config
            }
            torch.save(checkpoint, output_dir / 'best_model.pt')
            logger.info(f"✓ Best model saved (Acc: {best_val_acc:.4f})")
        
        # Save last model
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_val_acc': best_val_acc,
            'config': config
        }
        torch.save(checkpoint, output_dir / 'last_model.pt')
    
    # Final test
    logger.info(f"\n{'='*80}")
    logger.info(f"Loading best model from epoch {best_epoch}")
    logger.info(f"{'='*80}")
    
    checkpoint = torch.load(output_dir / 'best_model.pt', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    test1_metrics = validate(model, test1_loader, criterion, device, writer, 0, 'Test1', logger)
    test2_metrics = validate(model, test2_loader, criterion, device, writer, 0, 'Test2', logger)
    
    logger.info(f"\n{'='*80}")
    logger.info(f"Final Results:")
    logger.info(f"{'='*80}")
    logger.info(f"Test1 - Acc: {test1_metrics['accuracy']:.4f}, F1: {test1_metrics['f1_macro']:.4f}, "
                f"A-CCC: {test1_metrics['arousal_ccc']:.4f}, V-CCC: {test1_metrics['valence_ccc']:.4f}")
    logger.info(f"Test2 - Acc: {test2_metrics['accuracy']:.4f}, F1: {test2_metrics['f1_macro']:.4f}, "
                f"A-CCC: {test2_metrics['arousal_ccc']:.4f}, V-CCC: {test2_metrics['valence_ccc']:.4f}")
    
    writer.close()
    logger.info("\n✅ Training complete!")


if __name__ == '__main__':
    main()
