#!/usr/bin/env python3
"""
5-Fold Cross-Validation Training for OpenSMILE Emotion + A-V Classifier
Speaker-independent: Each fold uses 1 session for test, 4 sessions for train/val

Architecture: OpenSMILE features (88-d) → MLP → 4-class emotion + Arousal + Valence
"""
import os
import sys
import yaml
import json
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

sys.path.append(str(Path(__file__).resolve().parent.parent))

from models.opensmile_emotion_av import OpenSMILEEmotionAVClassifier


def load_config(config_path):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def compute_ccc(y_true, y_pred):
    """Compute Concordance Correlation Coefficient (CCC)"""
    y_true = y_true.cpu().numpy() if isinstance(y_true, torch.Tensor) else y_true
    y_pred = y_pred.cpu().numpy() if isinstance(y_pred, torch.Tensor) else y_pred
    
    y_true_mean = y_true.mean()
    y_pred_mean = y_pred.mean()
    
    numerator = 2 * ((y_true - y_true_mean) * (y_pred - y_pred_mean)).mean()
    denominator = ((y_true - y_true_mean) ** 2).mean() + ((y_pred - y_pred_mean) ** 2).mean()
    
    if denominator == 0:
        return 1.0 if (y_true == y_pred).all() else 0.0
    return numerator / denominator


def compute_accuracy(preds, targets):
    """Compute accuracy."""
    pred_labels = preds.argmax(dim=1)
    return (pred_labels == targets).float().mean().item()


def prepare_folds(csv_path):
    """
    Prepare 5-fold split based on sessions (speaker-independent)
    
    Fold structure (5-fold):
    - Fold 1: Test Session 1, Train Sessions 2,3,4,5
    - Fold 2: Test Session 2, Train Sessions 1,3,4,5
    - Fold 3: Test Session 3, Train Sessions 1,2,4,5
    - Fold 4: Test Session 4, Train Sessions 1,2,3,5
    - Fold 5: Test Session 5, Train Sessions 1,2,3,4
    """
    df = pd.read_csv(csv_path)
    
    folds = {}
    for fold_idx in range(1, 6):
        test_session = fold_idx
        train_val_sessions = [s for s in range(1, 6) if s != test_session]
        
        test_df = df[df['fold'] == test_session]
        train_val_df = df[df['fold'].isin(train_val_sessions)]
        
        folds[fold_idx] = {
            'session': test_session,
            'test': test_df,
            'train_val': train_val_df
        }
    
    return folds


class OpenSMILEDataset(Dataset):
    """Load OpenSMILE features from .npy files"""
    def __init__(self, df, opensmile_dir, mapping_json=None):
        self.df = df.reset_index(drop=True)
        self.opensmile_dir = opensmile_dir
        
        # Load mapping
        self.mapping = {}
        if mapping_json and os.path.exists(mapping_json):
            with open(mapping_json, 'r') as f:
                self.mapping = json.load(f)
        
        # Emotion to label mapping
        self.emotion_to_label = {'hap': 0, 'sad': 1, 'neu': 2, 'ang': 3}
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        sample_name = row['sample_name']
        emotion = row['emotion']
        arousal = float(row.get('arousal', 0.0))
        valence = float(row.get('valence', 0.0))
        
        # Get OpenSMILE filename from mapping or use sample_name directly
        if sample_name in self.mapping:
            npy_file = self.mapping[sample_name]
        else:
            npy_file = os.path.join(self.opensmile_dir, f"{sample_name}.npy")
        
        # Load OpenSMILE features
        try:
            embedding = np.load(npy_file).squeeze()  # (88,)
        except:
            embedding = np.zeros(88, dtype=np.float32)
        
        emotion_label = self.emotion_to_label.get(emotion, 2)
        
        # Normalize A-V to [-1, 1] (IEMOCAP: 1-5 scale → [-1, 1])
        arousal_norm = (arousal - 3.0) / 2.0
        valence_norm = (valence - 3.0) / 2.0
        
        return {
            'embedding': embedding.astype(np.float32),
            'emotion_label': emotion_label,
            'arousal': np.array([arousal_norm], dtype=np.float32),
            'valence': np.array([valence_norm], dtype=np.float32),
            'sample_name': sample_name
        }


def collate_fn(batch):
    """Collate function for DataLoader"""
    embeddings = np.stack([item['embedding'] for item in batch])
    emotion_labels = np.array([item['emotion_label'] for item in batch])
    arousals = np.concatenate([item['arousal'] for item in batch]).squeeze()
    valences = np.concatenate([item['valence'] for item in batch]).squeeze()
    
    return {
        'embedding': embeddings,
        'emotion_label': emotion_labels,
        'arousal': arousals,
        'valence': valences,
    }


def train_one_epoch(model, loader, criterion_emotion, criterion_av, optimizer, device, writer, global_step, epoch):
    """Train for one epoch"""
    model.train()
    
    total_loss = 0.0
    total_emotion_loss = 0.0
    total_av_loss = 0.0
    all_preds = []
    all_labels = []
    all_arousal_preds = []
    all_arousal_targets = []
    all_valence_preds = []
    all_valence_targets = []
    
    pbar = tqdm(loader, desc=f"Epoch {epoch} [Train]", leave=False)
    
    for batch_idx, batch in enumerate(pbar):
        embeddings = torch.from_numpy(batch['embedding']).to(device)
        emotion_labels = torch.from_numpy(batch['emotion_label']).to(device).long()
        arousals = torch.from_numpy(batch['arousal']).to(device).float()
        valences = torch.from_numpy(batch['valence']).to(device).float()
        
        # Forward
        optimizer.zero_grad()
        outputs = model(embeddings)
        emotion_logits = outputs['emotion_logits']
        arousal_pred = outputs['arousal']  # [B, 1]
        valence_pred = outputs['valence']  # [B, 1]
        
        # Expand targets to [B, 1]
        arousals_expanded = arousals.unsqueeze(1) if arousals.dim() == 1 else arousals  # [B, 1]
        valences_expanded = valences.unsqueeze(1) if valences.dim() == 1 else valences  # [B, 1]
        
        # Loss
        emotion_loss = criterion_emotion(emotion_logits, emotion_labels)
        av_loss = criterion_av(arousal_pred, arousals_expanded) + criterion_av(valence_pred, valences_expanded)
        loss = emotion_loss + av_loss
        
        # Backward
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        total_emotion_loss += emotion_loss.item()
        total_av_loss += av_loss.item()
        
        acc = compute_accuracy(emotion_logits, emotion_labels)
        
        # Predictions
        pred_labels = emotion_logits.argmax(dim=1).cpu().numpy()
        all_preds.extend(pred_labels)
        all_labels.extend(emotion_labels.cpu().numpy())
        
        all_arousal_preds.extend(arousal_pred.detach().cpu().numpy())
        all_arousal_targets.extend(arousals.cpu().numpy())
        all_valence_preds.extend(valence_pred.detach().cpu().numpy())
        all_valence_targets.extend(valences.cpu().numpy())
        
        pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{acc:.4f}'})
        
        global_step += 1
    
    avg_loss = total_loss / len(loader)
    avg_emotion_loss = total_emotion_loss / len(loader)
    avg_av_loss = total_av_loss / len(loader)
    accuracy = accuracy_score(all_labels, all_preds)
    f1_macro = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    f1_weighted = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    
    arousal_ccc = compute_ccc(np.array(all_arousal_targets).squeeze(), np.array(all_arousal_preds).squeeze())
    valence_ccc = compute_ccc(np.array(all_valence_targets).squeeze(), np.array(all_valence_preds).squeeze())
    
    if writer is not None:
        writer.add_scalar('train/loss', avg_loss, epoch)
        writer.add_scalar('train/emotion_loss', avg_emotion_loss, epoch)
        writer.add_scalar('train/av_loss', avg_av_loss, epoch)
        writer.add_scalar('train/accuracy', accuracy, epoch)
        writer.add_scalar('train/f1_macro', f1_macro, epoch)
        writer.add_scalar('train/f1_weighted', f1_weighted, epoch)
        writer.add_scalar('train/arousal_ccc', arousal_ccc, epoch)
        writer.add_scalar('train/valence_ccc', valence_ccc, epoch)
    
    return avg_loss, accuracy, f1_macro, f1_weighted, arousal_ccc, valence_ccc, global_step


def validate(model, loader, criterion_emotion, criterion_av, device, writer, epoch, phase='Val'):
    """Validate model"""
    model.eval()
    
    total_loss = 0.0
    total_emotion_loss = 0.0
    total_av_loss = 0.0
    all_preds = []
    all_labels = []
    all_arousal_preds = []
    all_arousal_targets = []
    all_valence_preds = []
    all_valence_targets = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            embeddings = torch.from_numpy(batch['embedding']).to(device)
            emotion_labels = torch.from_numpy(batch['emotion_label']).to(device).long()
            arousals = torch.from_numpy(batch['arousal']).to(device).float()
            valences = torch.from_numpy(batch['valence']).to(device).float()
            
            # Expand targets to [B, 1] if needed
            arousals_expanded = arousals.unsqueeze(1) if arousals.dim() == 1 else arousals
            valences_expanded = valences.unsqueeze(1) if valences.dim() == 1 else valences
            
            outputs = model(embeddings)
            emotion_logits = outputs['emotion_logits']
            arousal_pred = outputs['arousal']
            valence_pred = outputs['valence']
            
            emotion_loss = criterion_emotion(emotion_logits, emotion_labels)
            av_loss = criterion_av(arousal_pred, arousals_expanded) + criterion_av(valence_pred, valences_expanded)
            loss = emotion_loss + av_loss
            total_loss += loss.item()
            total_emotion_loss += emotion_loss.item()
            total_av_loss += av_loss.item()
            
            pred_labels = emotion_logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(pred_labels)
            all_labels.extend(emotion_labels.cpu().numpy())
            
            all_arousal_preds.extend(arousal_pred.cpu().numpy())
            all_arousal_targets.extend(arousals.cpu().numpy())
            all_valence_preds.extend(valence_pred.cpu().numpy())
            all_valence_targets.extend(valences.cpu().numpy())
    
    avg_loss = total_loss / len(loader)
    avg_emotion_loss = total_emotion_loss / len(loader)
    avg_av_loss = total_av_loss / len(loader)
    accuracy = accuracy_score(all_labels, all_preds)
    f1_macro = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    f1_weighted = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    
    arousal_ccc = compute_ccc(np.array(all_arousal_targets).squeeze(), np.array(all_arousal_preds).squeeze())
    valence_ccc = compute_ccc(np.array(all_valence_targets).squeeze(), np.array(all_valence_preds).squeeze())
    
    if writer is not None:
        writer.add_scalar(f'{phase.lower()}/loss', avg_loss, epoch)
        writer.add_scalar(f'{phase.lower()}/emotion_loss', avg_emotion_loss, epoch)
        writer.add_scalar(f'{phase.lower()}/av_loss', avg_av_loss, epoch)
        writer.add_scalar(f'{phase.lower()}/accuracy', accuracy, epoch)
        writer.add_scalar(f'{phase.lower()}/f1_macro', f1_macro, epoch)
        writer.add_scalar(f'{phase.lower()}/f1_weighted', f1_weighted, epoch)
        writer.add_scalar(f'{phase.lower()}/arousal_ccc', arousal_ccc, epoch)
        writer.add_scalar(f'{phase.lower()}/valence_ccc', valence_ccc, epoch)
    
    return {
        'loss': avg_loss,
        'emotion_loss': avg_emotion_loss,
        'av_loss': avg_av_loss,
        'accuracy': accuracy,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted,
        'arousal_ccc': arousal_ccc,
        'valence_ccc': valence_ccc
    }


def train_fold(fold_data, config, fold_idx, output_dir, device):
    """Train one fold"""
    print(f"\n{'='*80}")
    print(f"🏁 Training Fold {fold_idx} (Test Session {fold_data['session']})")
    print(f"{'='*80}")
    
    # Split train/val (80/20)
    train_df, val_df = train_test_split(
        fold_data['train_val'],
        test_size=0.2,
        random_state=42,
        stratify=fold_data['train_val']['emotion']
    )
    
    test_df = fold_data['test']
    
    print(f"Train: {len(train_df)} samples")
    print(f"Val:   {len(val_df)} samples")
    print(f"Test:  {len(test_df)} samples")
    
    # Create datasets
    opensmile_dir = config['opensmile_dir']
    mapping_json = config.get('mapping_json')
    
    train_dataset = OpenSMILEDataset(train_df, opensmile_dir, mapping_json)
    val_dataset = OpenSMILEDataset(val_df, opensmile_dir, mapping_json)
    test_dataset = OpenSMILEDataset(test_df, opensmile_dir, mapping_json)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn
    )
    
    # Create model
    model = OpenSMILEEmotionAVClassifier(
        input_dim=config['input_dim'],
        hidden_dim=config['hidden_dim'],
        num_emotions=config.get('num_emotions', 4),
        num_transformer_layers=config.get('num_transformer_layers', 4),
        num_heads=config.get('num_heads', 4),
        ff_dim=config.get('ff_dim', 512),
        dropout=config.get('dropout', 0.1)
    ).to(device)
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\n{'='*80}")
    print(f"📊 MODEL STATISTICS")
    print(f"{'='*80}")
    print(f"Total parameters:     {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    print(f"\n{'='*80}")
    print(f"🏗️  ARCHITECTURE SUMMARY")
    print(f"{'='*80}")
    print(f"Input:                OpenSMILE features (88-d)")
    print(f"Embedding:            88 → {config['hidden_dim']}")
    print(f"Transformer layers:   {config['num_transformer_layers']} (heads={config['num_heads']}, ff_dim={config['ff_dim']})")
    print(f"Output:               Emotion (4-class) + Arousal + Valence (continuous)")
    print(f"{'='*80}\n")
    
    # Create loss functions
    criterion_emotion = nn.CrossEntropyLoss()
    criterion_av = nn.MSELoss()
    
    # Create optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config['learning_rate']),
        weight_decay=float(config.get('weight_decay', 1e-5))
    )
    
    # Create scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config['epochs']
    )
    
    # Create TensorBoard writer
    fold_log_dir = os.path.join(config.get('log_dir', 'logs/iemocap_opensmile_emotion_av_tensorboard'), f'fold_{fold_idx}')
    os.makedirs(fold_log_dir, exist_ok=True)
    writer = SummaryWriter(fold_log_dir)
    
    # Create fold output directory
    fold_output_dir = os.path.join(output_dir, f'fold_{fold_idx}')
    os.makedirs(fold_output_dir, exist_ok=True)
    
    best_val_acc = 0.0
    global_step = 0
    
    # Training loop
    for epoch in range(config['epochs']):
        train_loss, train_acc, train_f1_macro, train_f1_weighted, train_arousal_ccc, train_valence_ccc, global_step = train_one_epoch(
            model, train_loader, criterion_emotion, criterion_av, optimizer, device, writer, global_step, epoch
        )
        
        val_metrics = validate(model, val_loader, criterion_emotion, criterion_av, device, writer, epoch, phase='Val')
        test_metrics = validate(model, test_loader, criterion_emotion, criterion_av, device, writer, epoch, phase='Test')
        
        scheduler.step()
        
        print(f"Epoch {epoch:3d} | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} (Arous CCC: {train_arousal_ccc:.3f}, Valence CCC: {train_valence_ccc:.3f}) | "
              f"Val Loss: {val_metrics['loss']:.4f} Acc: {val_metrics['accuracy']:.4f} (Arous CCC: {val_metrics['arousal_ccc']:.3f}, Valence CCC: {val_metrics['valence_ccc']:.3f}) | "
              f"Test Acc: {test_metrics['accuracy']:.4f} (Arous CCC: {test_metrics['arousal_ccc']:.3f}, Valence CCC: {test_metrics['valence_ccc']:.3f})")
        
        # Save best model
        if val_metrics['accuracy'] > best_val_acc:
            best_val_acc = val_metrics['accuracy']
            torch.save(model.state_dict(), os.path.join(fold_output_dir, 'best_model.pth'))
            print(f"  ✓ Best model saved (Val Acc: {best_val_acc:.4f})")
    
    # Load best model and evaluate on test set
    model.load_state_dict(torch.load(os.path.join(fold_output_dir, 'best_model.pth')))
    test_metrics = validate(model, test_loader, criterion_emotion, criterion_av, device, None, 0, phase='Test')
    
    writer.close()
    
    return {
        'fold': fold_idx,
        'test_accuracy': test_metrics['accuracy'],
        'test_f1_macro': test_metrics['f1_macro'],
        'test_f1_weighted': test_metrics['f1_weighted'],
        'test_arousal_ccc': test_metrics['arousal_ccc'],
        'test_valence_ccc': test_metrics['valence_ccc'],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Config file path')
    parser.add_argument('--csv', type=str, required=True, help='CSV file path')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device to use')
    
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Set random seed
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Prepare folds
    folds = prepare_folds(args.csv)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Train all folds
    fold_results = []
    for fold_idx in range(1, 6):
        fold_result = train_fold(folds[fold_idx], config, fold_idx, args.output_dir, device)
        fold_results.append(fold_result)
    
    # Print final results
    print(f"\n{'='*80}")
    print(f"🏆 FINAL 5-FOLD CROSS-VALIDATION RESULTS")
    print(f"{'='*80}")
    
    accuracies = [r['test_accuracy'] for r in fold_results]
    f1_macros = [r['test_f1_macro'] for r in fold_results]
    f1_weighteds = [r['test_f1_weighted'] for r in fold_results]
    arousal_cccs = [r['test_arousal_ccc'] for r in fold_results]
    valence_cccs = [r['test_valence_ccc'] for r in fold_results]
    
    for i, result in enumerate(fold_results):
        print(f"Fold {i+1}: Acc={result['test_accuracy']:.4f} F1(macro)={result['test_f1_macro']:.4f} F1(w)={result['test_f1_weighted']:.4f} "
              f"ArousalCCC={result['test_arousal_ccc']:.4f} ValenceCCC={result['test_valence_ccc']:.4f}")
    
    print(f"\nMean Accuracy:       {np.mean(accuracies):.4f} ± {np.std(accuracies):.4f}")
    print(f"Mean F1 (macro):     {np.mean(f1_macros):.4f} ± {np.std(f1_macros):.4f}")
    print(f"Mean F1 (weighted):  {np.mean(f1_weighteds):.4f} ± {np.std(f1_weighteds):.4f}")
    print(f"Mean Arousal CCC:    {np.mean(arousal_cccs):.4f} ± {np.std(arousal_cccs):.4f}")
    print(f"Mean Valence CCC:    {np.mean(valence_cccs):.4f} ± {np.std(valence_cccs):.4f}")
    print(f"{'='*80}")


if __name__ == '__main__':
    main()
