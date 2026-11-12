"""
Training script for Method 2 (ASPIRE Cross-Attention A-V) on MSP-Podcast dataset with 8-class emotion classification
Emotions (8): Anger, Contempt, Disgust, Fear, Happiness, Neutral, Sadness, Surprise
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

from models.method2_crossattn_av import Method2CrossAttentionAV
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


def compute_flops(model, audio_length=16000*10, text_length=128, device='cpu'):
    """Compute FLOPs for the model"""
    try:
        model = model.to(device)
        audio_input = torch.randn(1, audio_length).to(device)
        text_input = torch.randint(0, 1000, (1, text_length)).to(device)
        text_mask = torch.ones(1, text_length).to(device)
        
        flops, params = profile(model, inputs=(audio_input, text_input, None, text_mask), verbose=False)
        flops, params = clever_format([flops, params], "%.3f")
        return flops, params
    except Exception as e:
        return "N/A", "N/A"


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
    
    pbar = tqdm(loader, desc=f"Epoch {epoch} [Train 8-cls]")
    for batch_idx, batch in enumerate(pbar):
        # Move to device
        audio = batch['audio'].to(device)
        text_input_ids = batch['text_input_ids'].to(device)
        text_attention_mask = batch['text_attention_mask'].to(device)
        emotion_labels = batch['emotion_label'].to(device)
        arousal_values = batch['arousal_value'].to(device)
        valence_values = batch['valence_value'].to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(
            audio_input=audio,
            text_input=text_input_ids,
            text_attention_mask=text_attention_mask
        )
        
        # Compute loss
        targets = {
            'emotion_label': emotion_labels,
            'arousal_value': arousal_values,
            'valence_value': valence_values
        }
        
        loss_dict = criterion(outputs, targets)
        loss = loss_dict['total_loss']
        
        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        # Track metrics
        total_loss += loss.item()
        if not torch.isnan(loss_dict['emotion_loss']):
            emotion_losses.append(loss_dict['emotion_loss'].item())
        if not torch.isnan(loss_dict['arousal_loss']):
            arousal_losses.append(loss_dict['arousal_loss'].item())
        if not torch.isnan(loss_dict['valence_loss']):
            valence_losses.append(loss_dict['valence_loss'].item())
        
        # Predictions
        emotion_preds = torch.argmax(outputs['emotion_logits'], dim=1)
        all_emotion_preds.extend(emotion_preds.cpu().numpy())
        all_emotion_labels.extend(emotion_labels.cpu().numpy())
        all_arousal_preds.extend(outputs['arousal_value'].detach().cpu().numpy().flatten())
        all_arousal_targets.extend(arousal_values.cpu().numpy().flatten())
        all_valence_preds.extend(outputs['valence_value'].detach().cpu().numpy().flatten())
        all_valence_targets.extend(valence_values.cpu().numpy().flatten())
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'war': f'{accuracy_score(all_emotion_labels, all_emotion_preds):.4f}'
        })
        
        # Log to tensorboard
        if writer:
            writer.add_scalar('Train/batch_loss', loss.item(), global_step)
            global_step += 1
    
    # Compute epoch metrics (WAR = Weighted Accuracy / Standard Accuracy)
    avg_loss = total_loss / len(loader)
    accuracy = accuracy_score(all_emotion_labels, all_emotion_preds)
    f1_macro = f1_score(all_emotion_labels, all_emotion_preds, average='macro')
    f1_weighted = f1_score(all_emotion_labels, all_emotion_preds, average='weighted')
    
    arousal_ccc = compute_ccc(np.array(all_arousal_targets), np.array(all_arousal_preds))
    valence_ccc = compute_ccc(np.array(all_valence_targets), np.array(all_valence_preds))
    avg_ccc = (arousal_ccc + valence_ccc) / 2
    
    # Log epoch metrics
    logger.info(f"Epoch {epoch} Train - Loss: {avg_loss:.4f}, WAR (Acc): {accuracy:.4f}, "
                f"F1 Macro: {f1_macro:.4f}, F1 Weighted: {f1_weighted:.4f}, "
                f"Arousal CCC: {arousal_ccc:.4f}, Valence CCC: {valence_ccc:.4f}, Avg CCC: {avg_ccc:.4f}")
    
    if writer:
        writer.add_scalar('Train/loss', avg_loss, epoch)
        writer.add_scalar('Train/accuracy_war', accuracy, epoch)
        writer.add_scalar('Train/f1_macro', f1_macro, epoch)
        writer.add_scalar('Train/f1_weighted', f1_weighted, epoch)
        writer.add_scalar('Train/arousal_ccc', arousal_ccc, epoch)
        writer.add_scalar('Train/valence_ccc', valence_ccc, epoch)
        writer.add_scalar('Train/avg_ccc', avg_ccc, epoch)
    
    return avg_loss, accuracy, global_step


def validate(model, loader, criterion, device, writer, epoch, phase='Dev', logger=None):
    """Validate/Test the model for 8-class classification"""
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
        pbar = tqdm(loader, desc=f"Epoch {epoch} [{phase} 8-cls]")
        for batch in pbar:
            # Move to device
            audio = batch['audio'].to(device)
            text_input_ids = batch['text_input_ids'].to(device)
            text_attention_mask = batch['text_attention_mask'].to(device)
            emotion_labels = batch['emotion_label'].to(device)
            arousal_values = batch['arousal_value'].to(device)
            valence_values = batch['valence_value'].to(device)
            
            # Forward pass
            outputs = model(
                audio_input=audio,
                text_input=text_input_ids,
                text_attention_mask=text_attention_mask
            )
            
            # Compute loss
            targets = {
                'emotion_label': emotion_labels,
                'arousal_value': arousal_values,
                'valence_value': valence_values
            }
            
            loss_dict = criterion(outputs, targets)
            loss = loss_dict['total_loss']
            
            # Track metrics
            total_loss += loss.item()
            if not torch.isnan(loss_dict['emotion_loss']):
                emotion_losses.append(loss_dict['emotion_loss'].item())
            if not torch.isnan(loss_dict['arousal_loss']):
                arousal_losses.append(loss_dict['arousal_loss'].item())
            if not torch.isnan(loss_dict['valence_loss']):
                valence_losses.append(loss_dict['valence_loss'].item())
            
            # Predictions
            emotion_preds = torch.argmax(outputs['emotion_logits'], dim=1)
            all_emotion_preds.extend(emotion_preds.cpu().numpy())
            all_emotion_labels.extend(emotion_labels.cpu().numpy())
            all_arousal_preds.extend(outputs['arousal_value'].cpu().numpy().flatten())
            all_arousal_targets.extend(arousal_values.cpu().numpy().flatten())
            all_valence_preds.extend(outputs['valence_value'].cpu().numpy().flatten())
            all_valence_targets.extend(valence_values.cpu().numpy().flatten())
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'war': f'{accuracy_score(all_emotion_labels, all_emotion_preds):.4f}'
            })
    
    # Compute metrics (WAR = Weighted Accuracy)
    avg_loss = total_loss / len(loader)
    accuracy = accuracy_score(all_emotion_labels, all_emotion_preds)
    f1_macro = f1_score(all_emotion_labels, all_emotion_preds, average='macro')
    f1_weighted = f1_score(all_emotion_labels, all_emotion_preds, average='weighted')
    
    arousal_ccc = compute_ccc(np.array(all_arousal_targets), np.array(all_arousal_preds))
    valence_ccc = compute_ccc(np.array(all_valence_targets), np.array(all_valence_preds))
    avg_ccc = (arousal_ccc + valence_ccc) / 2
    
    # Log metrics
    if logger:
        logger.info(f"Epoch {epoch} {phase} - Loss: {avg_loss:.4f}, WAR (Acc): {accuracy:.4f}, "
                    f"F1 Macro: {f1_macro:.4f}, F1 Weighted: {f1_weighted:.4f}")
        logger.info(f"  Arousal CCC: {arousal_ccc:.4f}, Valence CCC: {valence_ccc:.4f}, "
                    f"Avg CCC: {avg_ccc:.4f}")
    
    if writer:
        writer.add_scalar(f'{phase}/loss', avg_loss, epoch)
        writer.add_scalar(f'{phase}/accuracy_war', accuracy, epoch)
        writer.add_scalar(f'{phase}/f1_macro', f1_macro, epoch)
        writer.add_scalar(f'{phase}/f1_weighted', f1_weighted, epoch)
        writer.add_scalar(f'{phase}/arousal_ccc', arousal_ccc, epoch)
        writer.add_scalar(f'{phase}/valence_ccc', valence_ccc, epoch)
        writer.add_scalar(f'{phase}/avg_ccc', avg_ccc, epoch)
    
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
    parser = argparse.ArgumentParser(description='Train Method 2 on MSP-Podcast (8-Class Emotion Recognition)')
    parser.add_argument('--config', type=str, default='configs/msp_podcast_8class_wpro_config.yaml',
                        help='Path to config file')
    parser.add_argument('--tsv', type=str, default='datasets/MSP-Podcast/msp_podcast.tsv',
                        help='Path to MSP-Podcast TSV file')
    parser.add_argument('--output_dir', type=str, default='checkpoints/method2_msp_wpro_8class',
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
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f'train_msp_8class_{timestamp}.log'
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)
    
    # Setup tensorboard
    tb_dir = log_dir / 'tensorboard'
    writer = SummaryWriter(str(tb_dir))
    
    logger.info("="*80)
    logger.info("Training Method 2 (Cross-Attention A-V) on MSP-Podcast (8-Class Emotion Recognition)")
    logger.info("="*80)
    logger.info(f"Config: {args.config}")
    logger.info(f"TSV: {args.tsv}")
    logger.info(f"Output: {args.output_dir}")
    logger.info(f"Device: {device}")
    logger.info(f"Log file: {log_file}")
    
    # Set seed
    seed = config.get('seed', 42)
    torch.manual_seed(seed)
    np.random.seed(seed)
    logger.info(f"Random seed: {seed}")

    try:
        torch.multiprocessing.set_sharing_strategy("file_system")
        logger.info("Torch multiprocessing sharing strategy: file_system")
    except RuntimeError:
        pass
    
    # Create dataloaders for 8-class (four_class_only=False)
    four_class_only = config['data'].get('four_class_only', False)
    logger.info(f"\nCreating 8-class dataloaders (four_class_only={four_class_only})...")
    train_loader, dev_loader, test1_loader, test2_loader = get_msp_podcast_dataloaders(
        tsv_path=args.tsv,
        batch_size=config['training']['batch_size'],
        num_workers=config['training'].get('num_workers', 4),
        audio_max_length=config['data']['max_audio_length'],
        tokenizer_name=config['model']['roberta_model_name'],
        max_text_length=config['data']['max_text_length'],
        pin_memory=config['training'].get('pin_memory', True) and torch.cuda.is_available(),
        persistent_workers=config['training'].get('persistent_workers', None),
        prefetch_factor=config['training'].get('prefetch_factor', None),
        four_class_only=four_class_only
    )
    
    logger.info(f"Train batches: {len(train_loader)}")
    logger.info(f"Dev batches: {len(dev_loader)}")
    logger.info(f"Test1 batches: {len(test1_loader)}")
    logger.info(f"Test2 batches: {len(test2_loader)}")
    
    # Create model with num_emotions=8
    logger.info(f"\nCreating Method2 model with {config['model']['num_emotions']} emotion outputs...")
    model = Method2CrossAttentionAV(
        num_emotions=config['model']['num_emotions'],
        wavlm_model_name=config['model']['wavlm_model_name'],
        wavlm_prosody_model_name=config['model'].get('wavlm_prosody_model_name', None),
        roberta_model_name=config['model']['roberta_model_name'],
        hidden_dim=config['model']['hidden_dim'],
        av_refinement_layers=config['model']['av_refinement_layers'],
        freeze_wavlm=config['model']['freeze_wavlm'],
        freeze_roberta=config['model']['freeze_roberta'],
        num_cross_attn_heads=config['model']['num_cross_attn_heads'],
        post_attention_layers=config['model']['post_attention_layers'],
        av_prediction_layers=config['model']['av_prediction_layers']
    )
    model = model.to(device)
    
    total_params, trainable_params = count_parameters(model)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    
    # Compute FLOPs
    flops, params = compute_flops(model, 
                                   audio_length=config['data']['max_audio_length'],
                                   text_length=config['data']['max_text_length'],
                                   device=device)
    logger.info(f"FLOPs: {flops}, Params: {params}")
    
    # Create loss function
    criterion = CombinedEmotionLossCCC(
        emotion_weight=config['loss']['emotion_weight'],
        arousal_weight=config['loss']['arousal_weight'],
        valence_weight=config['loss']['valence_weight']
    )
    
    # Create optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay'],
        betas=config['optimizer']['betas'],
        eps=config['optimizer']['eps']
    )
    
    # Create scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode=config['scheduler']['mode'],
        factor=config['scheduler']['factor'],
        patience=config['scheduler']['patience'],
        min_lr=config['scheduler']['min_lr']
    )
    
    # Resume from checkpoint if specified
    start_epoch = 0
    best_dev_acc = 0.0
    global_step = 0
    
    if args.resume:
        logger.info(f"\nResuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_dev_acc = checkpoint.get('best_dev_acc', 0.0)
        global_step = checkpoint.get('global_step', 0)
        logger.info(f"Resumed from epoch {start_epoch}, best dev acc (WAR): {best_dev_acc:.4f}")
    
    # Training loop
    logger.info("\n" + "="*80)
    logger.info("Starting Training (8-Class Emotion Recognition)")
    logger.info("="*80)
    
    num_epochs = config['training']['epochs']
    
    for epoch in range(start_epoch, num_epochs):
        logger.info(f"\nEpoch {epoch}/{num_epochs-1}")
        logger.info("-" * 40)
        
        # Train
        train_loss, train_acc, global_step = train_one_epoch(
            model, train_loader, criterion, optimizer, device,
            writer, global_step, epoch, logger
        )
        
        # Validate on dev set
        dev_metrics = validate(
            model, dev_loader, criterion, device, writer, epoch, 'Dev', logger
        )
        
        # Update scheduler
        scheduler.step(dev_metrics['accuracy'])
        
        # Save checkpoint if best
        if dev_metrics['accuracy'] > best_dev_acc:
            best_dev_acc = dev_metrics['accuracy']
            checkpoint_path = output_dir / 'best_model.pt'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_dev_acc': best_dev_acc,
                'dev_metrics': dev_metrics,
                'global_step': global_step,
                'config': config
            }, checkpoint_path)
            logger.info(f"✅ Saved best 8-class model (dev WAR/acc: {best_dev_acc:.4f}) to {checkpoint_path}")
        
        # Save last checkpoint
        last_checkpoint_path = output_dir / 'last_model.pt'
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_dev_acc': best_dev_acc,
            'dev_metrics': dev_metrics,
            'global_step': global_step,
            'config': config
        }, last_checkpoint_path)
    
    # Final evaluation on test sets
    logger.info("\n" + "="*80)
    logger.info("Final Evaluation on Test Sets (8-Class WAR & F1)")
    logger.info("="*80)
    
    # Load best model
    logger.info(f"\nLoading best model from {output_dir / 'best_model.pt'}")
    checkpoint = torch.load(output_dir / 'best_model.pt', map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    logger.info(f"Best model from epoch {checkpoint['epoch']} (dev WAR/acc: {checkpoint['best_dev_acc']:.4f})")
    
    # Test on Test1
    logger.info("\n--- MSP Test1 Evaluation (8-cls WAR) ---")
    test1_metrics = validate(
        model, test1_loader, criterion, device, None, checkpoint['epoch'], 'Test1', logger
    )
    
    # Test on Test2
    logger.info("\n--- MSP Test2 Evaluation (8-cls WAR) ---")
    test2_metrics = validate(
        model, test2_loader, criterion, device, None, checkpoint['epoch'], 'Test2', logger
    )
    
    # Print final summary
    logger.info("\n" + "="*80)
    logger.info("Training Complete - Final 8-Class Results")
    logger.info("="*80)
    logger.info(f"Best Dev WAR (Acc): {checkpoint['best_dev_acc']:.4f}")
    logger.info(f"\nMSP-Test1 Results (8-class):")
    logger.info(f"  WAR (Accuracy): {test1_metrics['accuracy']:.4f}")
    logger.info(f"  F1 Macro: {test1_metrics['f1_macro']:.4f}")
    logger.info(f"  F1 Weighted: {test1_metrics['f1_weighted']:.4f}")
    logger.info(f"  Arousal CCC: {test1_metrics['arousal_ccc']:.4f}")
    logger.info(f"  Valence CCC: {test1_metrics['valence_ccc']:.4f}")
    logger.info(f"  Avg CCC: {test1_metrics['avg_ccc']:.4f}")
    logger.info(f"\nMSP-Test2 Results (8-class):")
    logger.info(f"  WAR (Accuracy): {test2_metrics['accuracy']:.4f}")
    logger.info(f"  F1 Macro: {test2_metrics['f1_macro']:.4f}")
    logger.info(f"  F1 Weighted: {test2_metrics['f1_weighted']:.4f}")
    logger.info(f"  Arousal CCC: {test2_metrics['arousal_ccc']:.4f}")
    logger.info(f"  Valence CCC: {test2_metrics['valence_ccc']:.4f}")
    logger.info(f"  Avg CCC: {test2_metrics['avg_ccc']:.4f}")
    logger.info("="*80)
    
    writer.close()
    logger.info(f"\nLog file: {log_file}")
    logger.info(f"Tensorboard: tensorboard --logdir {tb_dir}")


if __name__ == '__main__':
    main()
