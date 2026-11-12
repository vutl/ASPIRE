"""
5-Fold Cross-Validation Training for Method 2 using precomputed emotion2vec embeddings
Strategy: 3/1/1 Split (Train/Val/Test)
- Test: 1 Session
- Val:  1 Session (next session in rotation)
- Train: 3 Sessions (remaining)

Architecture: Wpro (WavLM) + RoBERTa cross-attention + Emotion2Vec final fusion
"""
import os
import sys
import torch
import torch.nn as nn
import torchaudio
import pandas as pd
import numpy as np
import yaml
import argparse
import logging
import time
from tqdm import tqdm
from pathlib import Path
from sklearn.metrics import accuracy_score, f1_score, classification_report
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from thop import profile, clever_format

# Add parent directory to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from models.method2_crossattn_av_emotion2vec import Method2CrossAttentionAV_Emotion2Vec
from train.losses import CombinedEmotionLossCCC
from data.iemocap_dataset_emotion2vec import get_dataloaders_emotion2vec


def setup_logging(log_dir, exp_name):
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"{exp_name}.log")
    
    # Create logger
    logger = logging.getLogger(exp_name)
    logger.setLevel(logging.INFO)
    
    # Create handlers
    c_handler = logging.StreamHandler()
    f_handler = logging.FileHandler(log_file)
    c_handler.setLevel(logging.INFO)
    f_handler.setLevel(logging.INFO)
    
    # Create formatters and add it to handlers
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    c_handler.setFormatter(formatter)
    f_handler.setFormatter(formatter)
    
    # Add handlers to the logger
    logger.addHandler(c_handler)
    logger.addHandler(f_handler)
    
    return logger


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


def prepare_folds_3_1_1(csv_path):
    """
    Prepare 5-fold split based on sessions using 3/1/1 strategy
    
    Fold | Test Session | Val Session | Train Sessions
    -----|--------------|-------------|---------------
    1    | Ses01        | Ses02       | Ses03, Ses04, Ses05
    2    | Ses02        | Ses03       | Ses04, Ses05, Ses01
    3    | Ses03        | Ses04       | Ses05, Ses01, Ses02
    4    | Ses04        | Ses05       | Ses01, Ses02, Ses03
    5    | Ses05        | Ses01       | Ses02, Ses03, Ses04
    """
    df = pd.read_csv(csv_path)
    
    # Map 10 speaker folds to 5 session folds if needed
    if 'session' not in df.columns:
        fold_to_session = {
            1: 1, 2: 1,  # Ses01F, Ses01M
            3: 2, 4: 2,  # Ses02F, Ses02M
            5: 3, 6: 3,  # Ses03F, Ses03M
            7: 4, 8: 4,  # Ses04F, Ses04M
            9: 5, 10: 5  # Ses05F, Ses05M
        }
        df['session'] = df['fold'].map(fold_to_session)
    
    folds = []
    sessions = [1, 2, 3, 4, 5]
    
    for i in range(5):
        test_session = sessions[i]
        val_session = sessions[(i + 1) % 5]
        train_sessions = [s for s in sessions if s != test_session and s != val_session]
        
        test_df = df[df['session'] == test_session].reset_index(drop=True)
        val_df = df[df['session'] == val_session].reset_index(drop=True)
        train_df = df[df['session'].isin(train_sessions)].reset_index(drop=True)
        
        folds.append({
            'fold_idx': i + 1,
            'test_session': test_session,
            'val_session': val_session,
            'train_sessions': train_sessions,
            'test': test_df,
            'val': val_df,
            'train': train_df
        })
        
    return folds


def train_one_epoch(model, loader, criterion, optimizer, device, writer, global_step, epoch):
    """Train for one epoch"""
    model.train()
    
    total_loss = 0.0
    total_emotion_loss = 0.0
    total_arousal_loss = 0.0
    total_valence_loss = 0.0
    
    all_preds = []
    all_labels = []
    
    pbar = tqdm(loader, desc=f"Epoch {epoch} [Train]", leave=False)
    
    for batch_idx, batch in enumerate(pbar):
        # Move to device
        audio = batch['audio'].to(device)
        audio_emb = batch['audio_embedding'].to(device)
        text_input_ids = batch['text_input_ids'].to(device)
        text_attention_mask = batch['text_attention_mask'].to(device)
        emotion_label = batch['emotion_label'].to(device)
        arousal = batch['arousal_value'].to(device)
        valence = batch['valence_value'].to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(
            audio_input=audio,
            text_input=text_input_ids,
            audio_embedding=audio_emb,
            text_attention_mask=text_attention_mask
        )
        
        # Compute loss
        loss_dict = criterion(
            outputs,
            {
                'emotion_label': emotion_label,
                'arousal_value': arousal,
                'valence_value': valence
            }
        )
        
        loss = loss_dict['total_loss']
        
        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        # Metrics
        total_loss += loss.item()
        total_emotion_loss += loss_dict['emotion_loss'].item()
        total_arousal_loss += loss_dict['arousal_loss'].item()
        total_valence_loss += loss_dict['valence_loss'].item()
        
        preds = torch.argmax(outputs['emotion_logits'], dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(emotion_label.cpu().numpy())
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f"{loss.item():.4f}",
            'acc': f"{accuracy_score(all_labels, all_preds):.3f}"
        })
        
        # Log to tensorboard
        if writer is not None and batch_idx % 10 == 0:
            writer.add_scalar('Train/Batch/Total_Loss', loss.item(), global_step)
            writer.add_scalar('Train/Batch/Emotion_Loss', loss_dict['emotion_loss'].item(), global_step)
            writer.add_scalar('Train/Batch/Arousal_Loss', loss_dict['arousal_loss'].item(), global_step)
            writer.add_scalar('Train/Batch/Valence_Loss', loss_dict['valence_loss'].item(), global_step)
            global_step += 1
    
    # Compute epoch metrics
    avg_loss = total_loss / len(loader)
    avg_emotion_loss = total_emotion_loss / len(loader)
    avg_arousal_loss = total_arousal_loss / len(loader)
    avg_valence_loss = total_valence_loss / len(loader)
    accuracy = accuracy_score(all_labels, all_preds)
    f1_macro = f1_score(all_labels, all_preds, average='macro')
    f1_weighted = f1_score(all_labels, all_preds, average='weighted')
    
    # Log epoch metrics
    if writer is not None:
        writer.add_scalar('Train/Epoch/Total_Loss', avg_loss, epoch)
        writer.add_scalar('Train/Epoch/Emotion_Loss', avg_emotion_loss, epoch)
        writer.add_scalar('Train/Epoch/Arousal_Loss', avg_arousal_loss, epoch)
        writer.add_scalar('Train/Epoch/Valence_Loss', avg_valence_loss, epoch)
        writer.add_scalar('Train/Epoch/Accuracy', accuracy, epoch)
        writer.add_scalar('Train/Epoch/F1_Macro', f1_macro, epoch)
        writer.add_scalar('Train/Epoch/F1_Weighted', f1_weighted, epoch)
    
    return {
        'loss': avg_loss,
        'emotion_loss': avg_emotion_loss,
        'arousal_loss': avg_arousal_loss,
        'valence_loss': avg_valence_loss,
        'accuracy': accuracy,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted
    }, global_step


def validate(model, loader, criterion, device, writer, epoch, phase='Val'):
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
            # Move to device
            audio = batch['audio'].to(device)
            audio_emb = batch['audio_embedding'].to(device)
            text_input_ids = batch['text_input_ids'].to(device)
            text_attention_mask = batch['text_attention_mask'].to(device)
            emotion_label = batch['emotion_label'].to(device)
            arousal = batch['arousal_value'].to(device)
            valence = batch['valence_value'].to(device)
            
            # Forward pass
            outputs = model(
                audio_input=audio,
                text_input=text_input_ids,
                audio_embedding=audio_emb,
                text_attention_mask=text_attention_mask
            )
            
            # Compute loss
            loss_dict = criterion(
                outputs,
                {
                    'emotion_label': emotion_label,
                    'arousal_value': arousal,
                    'valence_value': valence
                }
            )
            
            loss = loss_dict['total_loss']
            
            # Metrics
            total_loss += loss.item()
            total_emotion_loss += loss_dict['emotion_loss'].item()
            total_arousal_loss += loss_dict['arousal_loss'].item()
            total_valence_loss += loss_dict['valence_loss'].item()
            
            preds = torch.argmax(outputs['emotion_logits'], dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(emotion_label.cpu().numpy())
            
            all_arousal_pred.extend(outputs['arousal_value'].cpu().numpy())
            all_arousal_true.extend(arousal.cpu().numpy())
            all_valence_pred.extend(outputs['valence_value'].cpu().numpy())
            all_valence_true.extend(valence.cpu().numpy())
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'acc': f"{accuracy_score(all_labels, all_preds):.3f}"
            })
    
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
        writer.add_scalar(f'{phase}/Epoch/Total_Loss', avg_loss, epoch)
        writer.add_scalar(f'{phase}/Epoch/Emotion_Loss', avg_emotion_loss, epoch)
        writer.add_scalar(f'{phase}/Epoch/Arousal_Loss', avg_arousal_loss, epoch)
        writer.add_scalar(f'{phase}/Epoch/Valence_Loss', avg_valence_loss, epoch)
        writer.add_scalar(f'{phase}/Epoch/Accuracy', accuracy, epoch)
        writer.add_scalar(f'{phase}/Epoch/F1_Macro', f1_macro, epoch)
        writer.add_scalar(f'{phase}/Epoch/F1_Weighted', f1_weighted, epoch)
        writer.add_scalar(f'{phase}/Epoch/Arousal_CCC', arousal_ccc, epoch)
        writer.add_scalar(f'{phase}/Epoch/Valence_CCC', valence_ccc, epoch)
        writer.add_scalar(f'{phase}/Epoch/Avg_CCC', avg_ccc, epoch)
    
    return {
        'loss': avg_loss,
        'emotion_loss': avg_emotion_loss,
        'arousal_loss': avg_arousal_loss,
        'valence_loss': avg_valence_loss,
        'accuracy': accuracy,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted,
        'arousal_ccc': arousal_ccc,
        'valence_ccc': valence_ccc,
        'avg_ccc': avg_ccc
    }


def train_fold(fold_data, config, fold_idx, output_dir, device, emotion2vec_dir, detected_emb_dim, resume_checkpoint=None):
    """Train one fold"""
    print(f"\n{'='*80}")
    print(f"🏁 Training Fold {fold_idx}")
    print(f"   Test Session:  {fold_data['test_session']}")
    print(f"   Val Session:   {fold_data['val_session']}")
    print(f"   Train Session: {fold_data['train_sessions']}")
    print(f"{'='*80}")
    
    # Get DataFrames
    train_df = fold_data['train']
    val_df = fold_data['val']
    test_df = fold_data['test']
    
    print(f"Train: {len(train_df)} samples")
    print(f"Val:   {len(val_df)} samples")
    print(f"Test:  {len(test_df)} samples")
    
    # Determine embedding dimension
    emb_dim_for_dataset = detected_emb_dim if detected_emb_dim is not None else config.get('data', {}).get('emotion2vec_dim', 768)
    num_workers = config['training']['num_workers']
    batch_size = config['training']['batch_size']
    
    # Create dataloaders
    train_loader = get_dataloaders_emotion2vec(
        train_df,
        emotion2vec_dir=emotion2vec_dir,
        batch_size=batch_size,
        num_workers=num_workers,
        emotion2vec_dim=emb_dim_for_dataset
    )
    
    val_loader = get_dataloaders_emotion2vec(
        val_df,
        emotion2vec_dir=emotion2vec_dir,
        batch_size=batch_size,
        num_workers=num_workers,
        emotion2vec_dim=emb_dim_for_dataset
    )
    
    test_loader = get_dataloaders_emotion2vec(
        test_df,
        emotion2vec_dir=emotion2vec_dir,
        batch_size=batch_size,
        num_workers=num_workers,
        emotion2vec_dim=emb_dim_for_dataset
    )
    
    # Create model
    model_cfg = config['model']
    model = Method2CrossAttentionAV_Emotion2Vec(
        num_emotions=model_cfg['num_emotions'],
        wavlm_model_name=model_cfg['wavlm_model_name'],
        wavlm_prosody_model_name=model_cfg.get('wavlm_prosody_model_name', None),
        roberta_model_name=model_cfg.get('roberta_model_name', 'roberta-base'),
        hidden_dim=model_cfg['hidden_dim'],
        av_refinement_layers=model_cfg['av_refinement_layers'],
        freeze_wavlm=model_cfg['freeze_wavlm'],
        freeze_roberta=model_cfg.get('freeze_roberta', False),
        num_cross_attn_heads=model_cfg['num_cross_attn_heads'],
        post_attention_layers=model_cfg.get('post_attention_layers', 2),
        av_prediction_layers=model_cfg.get('av_prediction_layers', 3),
        use_emotion2vec=True,
        emotion2vec_dim=emb_dim_for_dataset
    ).to(device)
    
    # Print model info
    total_params, trainable_params = count_parameters(model)
    print(f"\n{'='*80}")
    print(f"📊 MODEL STATISTICS")
    print(f"{'='*80}")
    print(f"Total parameters:     {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Frozen parameters:    {total_params - trainable_params:,}")
    
    # Architecture summary
    print(f"\n{'='*80}")
    print(f"🏗️  ARCHITECTURE SUMMARY (Emotion2Vec Variant)")
    print(f"{'='*80}")
    print(f"1. Wpro (WavLM):          Extract acoustic/prosody features")
    print(f"2. RoBERTa:               Extract semantic/linguistic features from text")
    print(f"3. Cross-Attention:       Wpro ↔ RoBERTa bidirectional attention")
    print(f"4. Post-Attention:        {model_cfg.get('post_attention_layers', 2)} Transformer layers per branch")
    print(f"5. Branch Fusion:         Gated fusion of Wpro + RoBERTa")
    print(f"6. A-V Prediction:        {model_cfg.get('av_prediction_layers', 3)} MLP layers → Arousal + Valence")
    print(f"7. A-V Refinement:        {model_cfg['av_refinement_layers']} MLP layers with residual")
    print(f"8. Emotion2Vec:           Precomputed embeddings (dim={emb_dim_for_dataset})")
    print(f"9. Final Fusion:          Emotion2Vec + Branch_Fusion + AV_Refined → Emotion")
    print(f"{'='*80}\n")
    
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
        weight_decay=config['training']['weight_decay']
    )
    
    # Create scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',
        factor=0.5,
        patience=5
    )
    
    # Resume from checkpoint if provided
    start_epoch = 1
    best_val_acc = 0.0
    
    if resume_checkpoint and os.path.exists(resume_checkpoint):
        print(f"\n🔄 Resuming from checkpoint: {resume_checkpoint}")
        checkpoint = torch.load(resume_checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_acc = checkpoint.get('val_accuracy', 0.0)
        print(f"   Starting from epoch {start_epoch}")
        print(f"   Previous best val accuracy: {best_val_acc:.4f}\n")
    
    # TensorBoard writer
    log_dir = os.path.join(output_dir, f'logs/fold_{fold_idx}')
    writer = SummaryWriter(log_dir)
    
    # Training loop
    best_epoch = 0
    global_step = 0
    
    checkpoint_dir = os.path.join(output_dir, f'fold_{fold_idx}')
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    for epoch in range(start_epoch, config['training']['epochs'] + 1):
        print(f"\n{'='*60}")
        print(f"Fold {fold_idx} - Epoch {epoch}/{config['training']['epochs']}")
        print(f"{'='*60}")
        
        # Train
        train_metrics, global_step = train_one_epoch(
            model, train_loader, criterion, optimizer, device,
            writer, global_step, epoch
        )
        
        # Validate
        val_metrics = validate(
            model, val_loader, criterion, device, writer, epoch, phase='Val'
        )
        
        # Scheduler step
        scheduler.step(val_metrics['accuracy'])
        
        # Log learning rate
        current_lr = optimizer.param_groups[0]['lr']
        writer.add_scalar('Train/Learning_Rate', current_lr, epoch)
        
        # Print metrics
        print(f"\n[Train] Loss: {train_metrics['loss']:.4f} | "
              f"Acc: {train_metrics['accuracy']:.4f} | "
              f"F1: {train_metrics['f1_macro']:.4f}")
        print(f"[Val]   Loss: {val_metrics['loss']:.4f} | "
              f"Acc: {val_metrics['accuracy']:.4f} | "
              f"F1: {val_metrics['f1_macro']:.4f} | "
              f"CCC: {val_metrics['avg_ccc']:.4f}")
        
        # Save best model
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
    
    # Load best model and test
    print(f"\n{'='*60}")
    print(f"Loading best model from epoch {best_epoch}")
    print(f"{'='*60}")
    
    checkpoint = torch.load(os.path.join(checkpoint_dir, 'best_model.pt'), weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Test
    test_metrics = validate(
        model, test_loader, criterion, device, writer, 0, phase='Test'
    )
    
    print(f"\n{'='*60}")
    print(f"🎯 Fold {fold_idx} Test Results:")
    print(f"{'='*60}")
    print(f"Accuracy:    {test_metrics['accuracy']:.4f}")
    print(f"F1 Macro:    {test_metrics['f1_macro']:.4f}")
    print(f"F1 Weighted: {test_metrics['f1_weighted']:.4f}")
    print(f"Arousal CCC: {test_metrics['arousal_ccc']:.4f}")
    print(f"Valence CCC: {test_metrics['valence_ccc']:.4f}")
    print(f"Avg CCC:     {test_metrics['avg_ccc']:.4f}")
    
    writer.close()
    
    return test_metrics


def main():
    parser = argparse.ArgumentParser(description='5-Fold Cross-Validation Training for Method 2 (Emotion2Vec)')
    parser.add_argument('--config', type=str, default='configs/method2_config.yaml',
                        help='Path to config file')
    parser.add_argument('--csv', type=str, default='datasets/IEMOCAP_KFOLD/tenfolds/all.csv',
                        help='Path to CSV file with fold information')
    parser.add_argument('--output_dir', type=str, default='checkpoints/method2_kfold_emotion2vec',
                        help='Output directory for checkpoints and logs')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='Device to use (cuda:0, cuda:1, cpu)')
    parser.add_argument('--emotion2vec_dir', type=str, required=True,
                        help='Directory containing emotion2vec .npy files')
    parser.add_argument('--resume_fold', type=int, default=None,
                        help='Resume training from specific fold')
    parser.add_argument('--resume_checkpoint', type=str, default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='Override batch size from config')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Override epochs from config')
    
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Override batch size if specified
    if args.batch_size:
        config['training']['batch_size'] = args.batch_size
    
    # Override epochs if specified
    if args.epochs:
        config['training']['epochs'] = args.epochs
    
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*80}")
    print(f"🚀 STARTING 5-FOLD CROSS-VALIDATION TRAINING (Emotion2Vec Variant)")
    print(f"{'='*80}")
    print(f"Device: {device}")
    print(f"Config: {args.config}")
    print(f"CSV: {args.csv}")
    print(f"Emotion2Vec Dir: {args.emotion2vec_dir}")
    print(f"Output: {args.output_dir}")
    if args.resume_fold:
        print(f"Resume from fold: {args.resume_fold}")
    if args.resume_checkpoint:
        print(f"Resume checkpoint: {args.resume_checkpoint}")
    print(f"{'='*80}\n")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'logs'), exist_ok=True)
    
    # Setup logging
    log_name = f"train_method2_emotion2vec_{int(time.time())}"
    logger = setup_logging(os.path.join(args.output_dir, 'logs'), log_name)
    
    # Detect embedding dimension
    detected_emb_dim = None
    try:
        files = [f for f in os.listdir(args.emotion2vec_dir) if f.endswith('.npy')]
        if files:
            sample_path = os.path.join(args.emotion2vec_dir, files[0])
            arr = np.load(sample_path)
            if arr.ndim == 1:
                detected_emb_dim = arr.shape[0]
            else:
                detected_emb_dim = arr.shape[-1]
            print(f"✅ Detected emotion2vec embedding dim = {detected_emb_dim} from {files[0]}")
    except Exception as e:
        print(f"⚠️ Warning: could not detect embedding dim from {args.emotion2vec_dir}: {e}")
    
    # Prepare folds (3/1/1 strategy)
    print(f"Preparing 5-fold cross-validation splits (3/1/1 strategy)...")
    folds = prepare_folds_3_1_1(args.csv)
    
    # Train each fold
    all_results = []
    
    start_fold = args.resume_fold if args.resume_fold else 1
    
    for fold_idx, fold_data in enumerate(folds[start_fold-1:], start_fold):
        resume_ckpt = args.resume_checkpoint if fold_idx == start_fold else None
        fold_results = train_fold(
            fold_data, config, fold_idx, args.output_dir, device, 
            args.emotion2vec_dir, detected_emb_dim, resume_ckpt
        )
        fold_results['fold'] = fold_idx
        fold_results['session'] = fold_data['test_session']
        all_results.append(fold_results)
    
    # Compute average results
    print(f"\n{'='*80}")
    print(f"📊 5-FOLD CROSS-VALIDATION RESULTS")
    print(f"{'='*80}")
    
    results_df = pd.DataFrame(all_results)
    print(results_df.to_string(index=False))
    
    print(f"\n{'='*80}")
    print(f"📈 AVERAGE RESULTS:")
    print(f"{'='*80}")
    mean_results = results_df.mean(numeric_only=True)
    std_results = results_df.std(numeric_only=True)
    
    print(f"Accuracy:    {mean_results['accuracy']:.4f} ± {std_results['accuracy']:.4f}")
    print(f"F1 Macro:    {mean_results['f1_macro']:.4f} ± {std_results['f1_macro']:.4f}")
    print(f"F1 Weighted: {mean_results['f1_weighted']:.4f} ± {std_results['f1_weighted']:.4f}")
    print(f"Arousal CCC: {mean_results['arousal_ccc']:.4f} ± {std_results['arousal_ccc']:.4f}")
    print(f"Valence CCC: {mean_results['valence_ccc']:.4f} ± {std_results['valence_ccc']:.4f}")
    print(f"Avg CCC:     {mean_results['avg_ccc']:.4f} ± {std_results['avg_ccc']:.4f}")
    
    # Save results
    results_path = os.path.join(args.output_dir, 'kfold_results.csv')
    results_df.to_csv(results_path, index=False)
    print(f"\n💾 Results saved to: {results_path}")


if __name__ == '__main__':
    main()
