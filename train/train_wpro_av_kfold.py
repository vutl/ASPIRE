"""
5-Fold Cross-Validation Training for W-PRO A-V Model on IEMOCAP
Speaker-independent: Each fold uses 1 session for test, 4 sessions for train/val
Simple single-branch architecture with frozen W-PRO backbone
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
from sklearn.metrics import accuracy_score, f1_score, classification_report
from scipy.stats import pearsonr
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from transformers import WavLMModel
from thop import profile, clever_format

# Add parent directory to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from models.wpro_av_kfold import WProAVKfold
from train.losses import CombinedEmotionLossCCC
from data.iemocap_dataset import IEMOCAPDataset


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


def prepare_folds(csv_path):
    """
    Prepare 5-fold split based on sessions
    Fold structure:
    - Fold 1: Test Session1 (folds 1,2), Train Sessions 2,3,4,5 (folds 3-10)
    - Fold 2: Test Session2 (folds 3,4), Train Sessions 1,3,4,5 (folds 1,2,5-10)
    - Fold 3: Test Session3 (folds 5,6), Train Sessions 1,2,4,5 (folds 1-4,7-10)
    - Fold 4: Test Session4 (folds 7,8), Train Sessions 1,2,3,5 (folds 1-6,9-10)
    - Fold 5: Test Session5 (folds 9,10), Train Sessions 1,2,3,4 (folds 1-8)
    """
    df = pd.read_csv(csv_path)
    
    # Map 10 speaker folds to 5 session folds
    fold_to_session = {
        1: 1, 2: 1,
        3: 2, 4: 2,
        5: 3, 6: 3,
        7: 4, 8: 4,
        9: 5, 10: 5
    }
    
    df['session'] = df['fold'].map(fold_to_session)
    
    folds = []
    for session in range(1, 6):
        test_mask = df['session'] == session
        train_mask = df['session'] != session
        folds.append({
            'train_indices': df[train_mask].index.tolist(),
            'test_indices': df[test_mask].index.tolist(),
            'session': session
        })
    
    return folds


def train_one_epoch(model, loader, criterion, optimizer, device, writer, global_step, epoch, logger):
    """Train for one epoch"""
    model.train()
    
    total_loss = 0.0
    total_emotion_loss = 0.0
    total_arousal_loss = 0.0
    total_valence_loss = 0.0
    
    all_preds = []
    all_labels = []
    all_arousal_pred = []
    all_arousal_true = []
    all_valence_pred = []
    all_valence_true = []
    
    pbar = tqdm(loader, desc=f"Epoch {epoch} [Train]", leave=False)
    
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
        
        # Record losses
        total_loss += loss.item()
        total_emotion_loss += losses_dict['emotion_loss'].item()
        total_arousal_loss += losses_dict['arousal_loss'].item()
        total_valence_loss += losses_dict['valence_loss'].item()
        
        # Get predictions
        emotion_pred = torch.argmax(outputs['emotion_logits'], dim=1).cpu().numpy()
        emotion_label = emotion_labels.cpu().numpy()
        arousal_pred = outputs['arousal_value'].squeeze(-1).detach().cpu().numpy()
        arousal_true = arousal_targets.squeeze(-1).cpu().numpy()
        valence_pred = outputs['valence_value'].squeeze(-1).detach().cpu().numpy()
        valence_true = valence_targets.squeeze(-1).cpu().numpy()
        
        all_preds.extend(emotion_pred)
        all_labels.extend(emotion_label)
        all_arousal_pred.extend(arousal_pred)
        all_arousal_true.extend(arousal_true)
        all_valence_pred.extend(valence_pred)
        all_valence_true.extend(valence_true)
        
        pbar.update(1)
        
        if writer:
            writer.add_scalar('Train/Loss', loss.item(), global_step)
            global_step += 1
    
    # Compute epoch metrics
    avg_loss = total_loss / len(loader)
    avg_emotion_loss = total_emotion_loss / len(loader)
    avg_arousal_loss = total_arousal_loss / len(loader)
    avg_valence_loss = total_valence_loss / len(loader)
    
    accuracy = accuracy_score(all_labels, all_preds)
    f1_macro = f1_score(all_labels, all_preds, average='macro')
    f1_weighted = f1_score(all_labels, all_preds, average='weighted')
    
    arousal_ccc = compute_ccc(np.array(all_arousal_true), np.array(all_arousal_pred))
    valence_ccc = compute_ccc(np.array(all_valence_true), np.array(all_valence_pred))
    avg_ccc = (arousal_ccc + valence_ccc) / 2
    
    if writer is not None:
        writer.add_scalar('Train/Epoch Loss', avg_loss, epoch)
        writer.add_scalar('Train/Epoch Emotion Loss', avg_emotion_loss, epoch)
        writer.add_scalar('Train/Epoch Arousal Loss', avg_arousal_loss, epoch)
        writer.add_scalar('Train/Epoch Valence Loss', avg_valence_loss, epoch)
        writer.add_scalar('Train/Epoch Accuracy', accuracy, epoch)
        writer.add_scalar('Train/Epoch F1 Macro', f1_macro, epoch)
    
    return global_step, {
        'loss': avg_loss,
        'emotion_loss': avg_emotion_loss,
        'arousal_loss': avg_arousal_loss,
        'valence_loss': avg_valence_loss,
        'accuracy': accuracy,
        'f1_macro': f1_macro,
        'arousal_ccc': arousal_ccc,
        'valence_ccc': valence_ccc,
        'avg_ccc': avg_ccc
    }


def validate(model, loader, criterion, device, writer, epoch, phase='Val', logger=None):
    """Validate model"""
    model.eval()
    
    total_loss = 0.0
    total_emotion_loss = 0.0
    total_arousal_loss = 0.0
    total_valence_loss = 0.0
    
    all_preds = []
    all_labels = []
    all_arousal_pred = []
    all_arousal_true = []
    all_valence_pred = []
    all_valence_true = []
    
    pbar = tqdm(loader, desc=f"Epoch {epoch} [{phase}]", leave=False)
    
    with torch.no_grad():
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
            total_emotion_loss += losses_dict['emotion_loss'].item()
            total_arousal_loss += losses_dict['arousal_loss'].item()
            total_valence_loss += losses_dict['valence_loss'].item()
            
            # Get predictions
            emotion_pred = torch.argmax(outputs['emotion_logits'], dim=1).cpu().numpy()
            emotion_label = emotion_labels.cpu().numpy()
            arousal_pred = outputs['arousal_value'].squeeze(-1).detach().cpu().numpy()
            arousal_true = arousal_targets.squeeze(-1).cpu().numpy()
            valence_pred = outputs['valence_value'].squeeze(-1).detach().cpu().numpy()
            valence_true = valence_targets.squeeze(-1).cpu().numpy()
            
            all_preds.extend(emotion_pred)
            all_labels.extend(emotion_label)
            all_arousal_pred.extend(arousal_pred)
            all_arousal_true.extend(arousal_true)
            all_valence_pred.extend(valence_pred)
            all_valence_true.extend(valence_true)
            
            pbar.update(1)
    
    # Compute metrics
    avg_loss = total_loss / len(loader)
    avg_emotion_loss = total_emotion_loss / len(loader)
    avg_arousal_loss = total_arousal_loss / len(loader)
    avg_valence_loss = total_valence_loss / len(loader)
    
    accuracy = accuracy_score(all_labels, all_preds)
    f1_macro = f1_score(all_labels, all_preds, average='macro')
    f1_weighted = f1_score(all_labels, all_preds, average='weighted')
    
    # Flatten for CCC computation
    all_arousal_pred = np.array(all_arousal_pred).flatten()
    all_arousal_true = np.array(all_arousal_true).flatten()
    all_valence_pred = np.array(all_valence_pred).flatten()
    all_valence_true = np.array(all_valence_true).flatten()
    
    arousal_ccc = compute_ccc(all_arousal_true, all_arousal_pred)
    valence_ccc = compute_ccc(all_valence_true, all_valence_pred)
    avg_ccc = (arousal_ccc + valence_ccc) / 2
    
    # Log metrics
    if writer is not None:
        writer.add_scalar(f'{phase}/Loss', avg_loss, epoch)
        writer.add_scalar(f'{phase}/Emotion Loss', avg_emotion_loss, epoch)
        writer.add_scalar(f'{phase}/Arousal Loss', avg_arousal_loss, epoch)
        writer.add_scalar(f'{phase}/Valence Loss', avg_valence_loss, epoch)
        writer.add_scalar(f'{phase}/Accuracy', accuracy, epoch)
        writer.add_scalar(f'{phase}/F1 Macro', f1_macro, epoch)
        writer.add_scalar(f'{phase}/Arousal CCC', arousal_ccc, epoch)
        writer.add_scalar(f'{phase}/Valence CCC', valence_ccc, epoch)
        writer.add_scalar(f'{phase}/Avg CCC', avg_ccc, epoch)
    
    if logger:
        logger.info(f"Epoch {epoch} {phase} - Loss: {avg_loss:.4f}, Acc: {accuracy:.4f}, "
                   f"F1: {f1_macro:.4f}, Arousal CCC: {arousal_ccc:.4f}, "
                   f"Valence CCC: {valence_ccc:.4f}, Avg CCC: {avg_ccc:.4f}")
    
    return {
        'loss': avg_loss,
        'accuracy': accuracy,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted,
        'arousal_ccc': arousal_ccc,
        'valence_ccc': valence_ccc,
        'avg_ccc': avg_ccc
    }


def train_fold(fold_data, config, fold_idx, output_dir, device, logger, resume_checkpoint=None):
    """Train a single fold"""
    
    # Create dataset
    csv_path = config['data']['train_csv']
    df = pd.read_csv(csv_path)
    
    train_indices = fold_data['train_indices']
    test_indices = fold_data['test_indices']
    
    # Split train into train/val (80/20)
    train_indices, val_indices = train_test_split(
        train_indices, test_size=0.2, random_state=42
    )
    
    train_dataset = IEMOCAPDataset(
        df.iloc[train_indices].reset_index(drop=True),
        sample_rate=config['data']['sample_rate'],
        audio_max_length=config['data']['max_audio_length']
    )
    
    val_dataset = IEMOCAPDataset(
        df.iloc[val_indices].reset_index(drop=True),
        sample_rate=config['data']['sample_rate'],
        audio_max_length=config['data']['max_audio_length']
    )
    
    test_dataset = IEMOCAPDataset(
        df.iloc[test_indices].reset_index(drop=True),
        sample_rate=config['data']['sample_rate'],
        audio_max_length=config['data']['max_audio_length']
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['training']['num_workers']
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['training']['num_workers']
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['training']['num_workers']
    )
    
    # Create model
    model = WProAVKfold(
        num_emotions=config['model']['num_emotions'],
        wpro_model_name=config['model'].get('wpro_model_name', 'Orange/Speaker-wavLM-pro'),
        hidden_dim=config['model']['hidden_dim'],
        av_refinement_layers=config['model'].get('av_refinement_layers', 2),
        freeze_wpro=config['model'].get('freeze_wpro', True)
    )
    
    model.to(device)
    
    # Count parameters
    total_params, trainable_params = count_parameters(model)
    logger.info(f"Fold {fold_idx}: Total params: {total_params:,}, Trainable: {trainable_params:,}")
    
    # Create loss function
    criterion = CombinedEmotionLossCCC(
        emotion_weight=config['loss'].get('emotion_weight', 1.0),
        arousal_weight=config['loss'].get('arousal_weight', 1.0),
        valence_weight=config['loss'].get('valence_weight', 1.0)
    )
    
    # Create optimizer
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config['training']['learning_rate'],
        weight_decay=config['training'].get('weight_decay', 0.00001)
    )
    
    # Create scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode=config['scheduler'].get('mode', 'max'),
        factor=config['scheduler'].get('factor', 0.5),
        patience=config['scheduler'].get('patience', 5),
        min_lr=config['scheduler'].get('min_lr', 1e-6)
    )
    
    # TensorBoard writer
    fold_output_dir = output_dir / f'fold{fold_idx}'
    fold_output_dir.mkdir(parents=True, exist_ok=True)
    tb_dir = Path(config['paths']['tensorboard_dir']) / f'wpro_av_kfold_fold{fold_idx}'
    tb_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(str(tb_dir))
    
    # Training loop
    epochs = config['training']['epochs']
    start_epoch = 0
    best_val_ccc = -1.0
    global_step = 0
    
    # Load checkpoint if resuming
    if resume_checkpoint:
        logger.info(f"Loading checkpoint: {resume_checkpoint}")
        checkpoint = torch.load(resume_checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_ccc = checkpoint.get('best_val_ccc', -1.0)
        global_step = checkpoint.get('global_step', 0)
        logger.info(f"Resumed from epoch {start_epoch}, best val CCC: {best_val_ccc:.4f}")
    
    for epoch in range(start_epoch, epochs):
        logger.info(f"Fold {fold_idx} Epoch {epoch+1}/{epochs}")
        
        # Train
        global_step, train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, device, writer, global_step, epoch+1, logger
        )
        logger.info(f"Fold {fold_idx} Epoch {epoch+1} Train - Loss: {train_metrics['loss']:.4f}, "
                   f"Acc: {train_metrics['accuracy']:.4f}, Avg CCC: {train_metrics['avg_ccc']:.4f}")
        
        # Validate
        val_metrics = validate(model, val_loader, criterion, device, writer, epoch+1, 'Val', logger)
        
        # Test
        test_metrics = validate(model, test_loader, criterion, device, writer, epoch+1, 'Test', logger)
        
        # Update scheduler
        scheduler.step(val_metrics['avg_ccc'])
        
        # Save checkpoint
        is_best = val_metrics['avg_ccc'] > best_val_ccc
        if is_best:
            best_val_ccc = val_metrics['avg_ccc']
            logger.info(f"New best Val CCC: {best_val_ccc:.4f}")
        
        checkpoint = {
            'epoch': epoch,
            'fold': fold_idx,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_val_ccc': best_val_ccc,
            'global_step': global_step
        }
        
        checkpoint_path = fold_output_dir / f'checkpoint_epoch{epoch+1}.pt'
        torch.save(checkpoint, checkpoint_path)
        
        if is_best:
            best_checkpoint_path = fold_output_dir / 'checkpoint_best.pt'
            torch.save(checkpoint, best_checkpoint_path)
    
    writer.close()
    
    return {
        'fold': fold_idx,
        'best_val_ccc': best_val_ccc,
        'final_test_metrics': test_metrics
    }


def main():
    parser = argparse.ArgumentParser(description='Train W-PRO A-V (5-fold CV)')
    parser.add_argument('--config', type=str, default='configs/method2_wpro_config.yaml',
                        help='Path to config file')
    parser.add_argument('--csv', type=str, default='datasets/IEMOCAP_KFOLD/all.csv',
                        help='Path to IEMOCAP CSV')
    parser.add_argument('--output_dir', type=str, default='checkpoints/wpro_av_kfold',
                        help='Output directory for checkpoints')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='Device to use')
    parser.add_argument('--resume_fold', type=int, default=None,
                        help='Resume from specific fold (1-5)')
    parser.add_argument('--resume_checkpoint', type=str, default=None,
                        help='Resume checkpoint path')
    
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    config['data']['train_csv'] = args.csv
    
    # Setup device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    log_file = output_dir / f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
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
    logger.info("W-PRO A-V Model (5-Fold Cross-Validation) on IEMOCAP")
    logger.info("="*80)
    logger.info(f"Device: {device}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Config: {args.config}")
    logger.info(f"CSV: {args.csv}")
    
    # Prepare folds
    logger.info("Preparing folds...")
    folds = prepare_folds(args.csv)
    logger.info(f"Prepared {len(folds)} folds")
    
    # Training loop
    start_fold = args.resume_fold if args.resume_fold else 1
    fold_results = []
    
    for fold_idx in range(start_fold, 6):
        logger.info(f"\n{'='*80}")
        logger.info(f"Training Fold {fold_idx}")
        logger.info(f"{'='*80}")
        
        resume_ckpt = args.resume_checkpoint if fold_idx == start_fold else None
        result = train_fold(folds[fold_idx-1], config, fold_idx, output_dir, device, logger, resume_ckpt)
        fold_results.append(result)
        
        logger.info(f"Fold {fold_idx} - Best Val CCC: {result['best_val_ccc']:.4f}")
    
    # Summary
    logger.info(f"\n{'='*80}")
    logger.info("Training Summary")
    logger.info(f"{'='*80}")
    for result in fold_results:
        logger.info(f"Fold {result['fold']}: Best Val CCC = {result['best_val_ccc']:.4f}")
    
    avg_ccc = np.mean([r['best_val_ccc'] for r in fold_results])
    logger.info(f"Average Val CCC (5-Fold): {avg_ccc:.4f}")
    logger.info(f"{'='*80}")


if __name__ == '__main__':
    main()
