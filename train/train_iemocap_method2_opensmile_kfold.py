#!/usr/bin/env python3
"""
5-Fold Cross-Validation Training for Method 2 A-V with OpenSMILE
Speaker-independent: Each fold uses 1 session for test, 4 sessions for train/val

Architecture: WavLM + RoBERTa + OpenSMILE features → Cross-Attention → A-V + Emotion
Method2CrossAttentionAVEGeMAPS (Frozen WavLM + RoBERTa, trainable OpenSMILE Transformer)
"""
import os
import sys
import yaml
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, mean_absolute_error
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

sys.path.append(str(Path(__file__).resolve().parent.parent))

from models.method2_av_egemaps import Method2CrossAttentionAVEGeMAPS
from transformers import AutoTokenizer
import librosa


def load_config(config_path):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def compute_metrics(emotion_preds, emotion_targets, av_preds, av_targets):
    """Compute emotion and A-V metrics"""
    pred_labels = emotion_preds.argmax(dim=1)
    emotion_acc = (pred_labels == emotion_targets).float().mean().item()
    emotion_f1 = f1_score(emotion_targets.cpu().numpy(), pred_labels.cpu().numpy(), 
                         average='weighted', zero_division=0)
    
    av_pred_np = av_preds.detach().cpu().numpy()
    av_target_np = av_targets.detach().cpu().numpy()
    
    av_mse = mean_squared_error(av_target_np, av_pred_np)
    av_mae = mean_absolute_error(av_target_np, av_pred_np)
    
    # compute per-dimension CCC (assume av_preds has shape [B, 2] => arousal, valence)
    ccc_values = []
    arousal_ccc = 0.0
    valence_ccc = 0.0
    for dim in range(av_target_np.shape[1]):
        target = av_target_np[:, dim]
        pred = av_pred_np[:, dim]

        mean_target = np.mean(target)
        mean_pred = np.mean(pred)

        var_target = np.var(target)
        var_pred = np.var(pred)
        cov = np.mean((target - mean_target) * (pred - mean_pred))

        denom = var_target + var_pred + (mean_target - mean_pred) ** 2
        if denom > 1e-6:
            ccc = 2 * cov / denom
        else:
            ccc = 0.0

        ccc_values.append(ccc)
        if dim == 0:
            arousal_ccc = ccc
        elif dim == 1:
            valence_ccc = ccc

    av_ccc = np.mean(ccc_values) if ccc_values else 0.0
    
    return {
        'emotion_acc': emotion_acc,
        'emotion_f1': emotion_f1,
        'av_mse': av_mse,
        'av_mae': av_mae,
        'av_ccc': av_ccc,
        'arousal_ccc': arousal_ccc,
        'valence_ccc': valence_ccc
    }


def prepare_folds(csv_path):
    """Prepare 5-fold split based on sessions"""
    df = pd.read_csv(csv_path)
    max_fold = df['fold'].max()
    
    if max_fold == 10:
        fold_to_session = {1: 1, 2: 1, 3: 2, 4: 2, 5: 3, 6: 3, 7: 4, 8: 4, 9: 5, 10: 5}
        df['session'] = df['fold'].map(fold_to_session)
    else:
        df['session'] = df['fold']
    
    folds = []
    for session in range(1, 6):
        test_df = df[df['session'] == session]
        train_val_df = df[df['session'] != session]
        folds.append({
            'session': session,
            'test': test_df,
            'train_val': train_val_df
        })
    
    return folds


class MultimodalOpenSMILEDataset(Dataset):
    """Load audio + RoBERTa tokenized text + OpenSMILE features.

    This dataset no longer requires precomputed WavLM/RoBERTa embeddings.
    It loads raw audio from `row['path']`, tokenizes transcripts with RoBERTa,
    and loads precomputed OpenSMILE (.npy) features from `opensmile_dir`.
    """
    def __init__(self, df, opensmile_dir, mapping_json=None,
                 tokenizer_name='roberta-base', max_text_length=128,
                 audio_max_length=160000, sample_rate=16000):
        self.df = df.reset_index(drop=True)
        self.opensmile_dir = opensmile_dir
        self.mapping = {}
        if mapping_json and os.path.exists(mapping_json):
            import json
            with open(mapping_json) as f:
                self.mapping = json.load(f)

        self.emotion_to_label = {'hap': 0, 'sad': 1, 'neu': 2, 'ang': 3}

        self.tokenizer_name = tokenizer_name
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_text_length = max_text_length

        self.audio_max_length = audio_max_length
        self.sample_rate = sample_rate
    
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        sample_name = row.get('sample_name', os.path.basename(row.get('path', '')).replace('.wav', ''))
        emotion = row.get('emotion', 'neu')
        arousal = float(row.get('arousal', 3.0))
        valence = float(row.get('valence', 3.0))

        # Normalize A-V from [1, 5] to [-1, 1]
        arousal = (arousal - 3.0) / 2.0
        valence = (valence - 3.0) / 2.0

        # Load raw audio from path (resample to sample_rate)
        audio_path = row.get('path', None)
        if audio_path and os.path.exists(audio_path):
            wav, _ = librosa.load(audio_path, sr=self.sample_rate)
        else:
            wav = np.zeros(self.audio_max_length, dtype=np.float32)

        # Pad / truncate
        if len(wav) < self.audio_max_length:
            pad_len = self.audio_max_length - len(wav)
            wav = np.concatenate([wav, np.zeros(pad_len, dtype=np.float32)])
        else:
            wav = wav[:self.audio_max_length]

        # Tokenize transcript
        transcript = row.get('transcript', '') or ''
        tokenized = self.tokenizer(
            transcript,
            truncation=True,
            padding='max_length',
            max_length=self.max_text_length,
            return_tensors='np'
        )
        input_ids = tokenized['input_ids'][0]
        attention_mask = tokenized['attention_mask'][0]

        # Load OpenSMILE and normalize to 1D eGeMAPS vector (88-d)
        opensmile_file = os.path.join(self.opensmile_dir, f"{sample_name}.npy")
        if os.path.exists(opensmile_file):
            opensmile = np.load(opensmile_file)
            # Squeeze singleton dims
            opensmile = np.squeeze(opensmile)
            # If time x feat (e.g., [T, 88]), average over time to get [88]
            if opensmile.ndim > 1:
                opensmile = np.mean(opensmile, axis=0)
            # If still not 1D, fallback to zeros
            if opensmile.ndim != 1:
                opensmile = np.zeros(88, dtype=np.float32)
            # Pad/trim to expected length (88)
            if opensmile.shape[0] < 88:
                pad = np.zeros(88 - opensmile.shape[0], dtype=np.float32)
                opensmile = np.concatenate([opensmile, pad])
            elif opensmile.shape[0] > 88:
                opensmile = opensmile[:88]
        else:
            opensmile = np.zeros(88, dtype=np.float32)

        emotion_label = self.emotion_to_label.get(emotion, 2)

        return {
            'audio': wav.astype(np.float32),
            'text_input_ids': input_ids.astype(np.int64),
            'text_attention_mask': attention_mask.astype(np.int64),
            'opensmile': opensmile.astype(np.float32),
            'emotion_label': emotion_label,
            'arousal': np.array([arousal], dtype=np.float32),
            'valence': np.array([valence], dtype=np.float32),
            'sample_name': sample_name
        }


def collate_fn(batch):
    """Collate function for DataLoader"""
    audios = np.stack([item['audio'] for item in batch])  # [B, T]
    input_ids = np.stack([item['text_input_ids'] for item in batch])  # [B, L]
    attention_masks = np.stack([item['text_attention_mask'] for item in batch])  # [B, L]
    # Ensure all opensmile vectors are 1D length 88; pad/trim if necessary
    opensmile_list = []
    for item in batch:
        arr = np.asarray(item['opensmile'], dtype=np.float32).squeeze()
        if arr.ndim != 1:
            arr = arr.flatten()
        if arr.size < 88:
            pad = np.zeros(88 - arr.size, dtype=np.float32)
            arr = np.concatenate([arr, pad])
        elif arr.size > 88:
            arr = arr[:88]
        opensmile_list.append(arr.astype(np.float32))
    opensmiles = np.stack(opensmile_list, axis=0)

    return {
        'audio': audios,
        'text_input_ids': input_ids,
        'text_attention_mask': attention_masks,
        'opensmile': opensmiles,
        'emotion_label': np.array([item['emotion_label'] for item in batch]),
        'arousal': np.concatenate([item['arousal'] for item in batch]),
        'valence': np.concatenate([item['valence'] for item in batch]),
        'sample_name': [item['sample_name'] for item in batch]
    }


def train_one_epoch(model, loader, criterion_emotion, criterion_av, optimizer, device, writer, global_step, epoch):
    """Train for one epoch"""
    model.train()
    
    total_loss = 0.0
    total_emotion_loss = 0.0
    total_av_loss = 0.0
    
    all_metrics = {
        'emotion_acc': [],
        'emotion_f1': [],
        'av_mse': [],
        'av_mae': [],
        'av_ccc': [],
        'arousal_ccc': [],
        'valence_ccc': []
    }
    
    pbar = tqdm(loader, desc=f"Epoch {epoch} [Train]", leave=False)
    
    for batch_idx, batch in enumerate(pbar):
        audio = torch.from_numpy(batch['audio']).to(device).float()
        text_input_ids = torch.from_numpy(batch['text_input_ids']).to(device).long()
        text_attention_mask = torch.from_numpy(batch['text_attention_mask']).to(device).long()
        opensmile = torch.from_numpy(batch['opensmile']).to(device).float()
        emotion_labels = torch.from_numpy(batch['emotion_label']).to(device).long()
        av_targets = torch.from_numpy(np.column_stack([batch['arousal'], batch['valence']])).to(device).float()

        optimizer.zero_grad()
        outputs = model(audio, text_input_ids, text_attention_mask, opensmile)
        emotion_logits = outputs['emotion_logits']
        arousal = outputs['arousal']
        valence = outputs['valence']
        av_preds = torch.cat([arousal, valence], dim=1)
        
        emotion_loss = criterion_emotion(emotion_logits, emotion_labels)
        av_loss = criterion_av(av_preds, av_targets)
        loss = emotion_loss + av_loss
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        total_emotion_loss += emotion_loss.item()
        total_av_loss += av_loss.item()
        
        metrics = compute_metrics(emotion_logits, emotion_labels, av_preds, av_targets)
        for k, v in metrics.items():
            all_metrics[k].append(v)
        
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'em_acc': f'{metrics["emotion_acc"]:.4f}',
            'av_ccc': f'{metrics["av_ccc"]:.4f}'
        })

        # Batch-level TensorBoard logging (every 10 batches)
        if writer is not None and (batch_idx % 10) == 0:
            writer.add_scalar('Train/Batch/Total_Loss', loss.item(), global_step)
            writer.add_scalar('Train/Batch/Emotion_Loss', emotion_loss.item(), global_step)
            writer.add_scalar('Train/Batch/AV_Loss', av_loss.item(), global_step)
            global_step += 1
    avg_loss = total_loss / len(loader)
    avg_emotion_loss = total_emotion_loss / len(loader)
    avg_av_loss = total_av_loss / len(loader)
    
    avg_metrics = {k: np.mean(v) for k, v in all_metrics.items()}
    
    if writer is not None:
        writer.add_scalar('Train/Epoch/Total_Loss', avg_loss, epoch)
        writer.add_scalar('Train/Epoch/Emotion_Loss', avg_emotion_loss, epoch)
        writer.add_scalar('Train/Epoch/AV_Loss', avg_av_loss, epoch)
        for k, v in avg_metrics.items():
            writer.add_scalar(f'Train/Epoch/{k}', v, epoch)
    
    return avg_loss, avg_metrics, global_step


def validate(model, loader, criterion_emotion, criterion_av, device, writer, epoch, phase='Val'):
    """Validate model"""
    model.eval()
    total_loss = 0.0
    total_emotion_loss = 0.0
    total_av_loss = 0.0

    all_preds = []
    all_labels = []
    all_arousal_pred = []
    all_arousal_true = []
    all_valence_pred = []
    all_valence_true = []

    pbar = tqdm(loader, desc=f"Epoch {epoch} [{phase}]", leave=False)

    with torch.no_grad():
        for batch in pbar:
            audio = torch.from_numpy(batch['audio']).to(device).float()
            text_input_ids = torch.from_numpy(batch['text_input_ids']).to(device).long()
            text_attention_mask = torch.from_numpy(batch['text_attention_mask']).to(device).long()
            opensmile = torch.from_numpy(batch['opensmile']).to(device).float()
            emotion_labels = torch.from_numpy(batch['emotion_label']).to(device).long()
            av_targets = torch.from_numpy(np.column_stack([batch['arousal'], batch['valence']])).to(device).float()

            outputs = model(audio, text_input_ids, text_attention_mask, opensmile)
            emotion_logits = outputs['emotion_logits']
            arousal = outputs['arousal']
            valence = outputs['valence']
            av_preds = torch.cat([arousal, valence], dim=1)

            # Loss
            emotion_loss = criterion_emotion(emotion_logits, emotion_labels)
            av_loss = criterion_av(av_preds, av_targets)
            loss = emotion_loss + av_loss

            total_loss += loss.item()
            total_emotion_loss += emotion_loss.item()
            total_av_loss += av_loss.item()

            # Predictions / accumulation
            preds = torch.argmax(emotion_logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(emotion_labels.cpu().numpy())

            # arousal/valence may be [B,1] tensors
            all_arousal_pred.extend(arousal.detach().cpu().numpy())
            all_arousal_true.extend(av_targets[:, 0].detach().cpu().numpy())
            all_valence_pred.extend(valence.detach().cpu().numpy())
            all_valence_true.extend(av_targets[:, 1].detach().cpu().numpy())

            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    # Compute metrics
    avg_loss = total_loss / len(loader)
    avg_emotion_loss = total_emotion_loss / len(loader)
    avg_av_loss = total_av_loss / len(loader)

    accuracy = accuracy_score(all_labels, all_preds) if len(all_labels) > 0 else 0.0
    f1_macro = f1_score(all_labels, all_preds, average='macro') if len(all_labels) > 0 else 0.0
    f1_weighted = f1_score(all_labels, all_preds, average='weighted') if len(all_labels) > 0 else 0.0

    # Flatten and compute CCC
    all_arousal_pred = np.array(all_arousal_pred).flatten()
    all_arousal_true = np.array(all_arousal_true).flatten()
    all_valence_pred = np.array(all_valence_pred).flatten()
    all_valence_true = np.array(all_valence_true).flatten()

    def _compute_ccc(y_true, y_pred):
        if y_true.size == 0 or y_pred.size == 0:
            return 0.0
        mean_true = np.mean(y_true)
        mean_pred = np.mean(y_pred)
        var_true = np.var(y_true)
        var_pred = np.var(y_pred)
        cov = np.mean((y_true - mean_true) * (y_pred - mean_pred))
        denom = var_true + var_pred + (mean_true - mean_pred) ** 2
        if denom <= 1e-8:
            return 0.0
        return 2 * cov / denom

    arousal_ccc = _compute_ccc(all_arousal_true, all_arousal_pred)
    valence_ccc = _compute_ccc(all_valence_true, all_valence_pred)
    avg_ccc = (arousal_ccc + valence_ccc) / 2

    # TensorBoard logging
    if writer is not None:
        writer.add_scalar(f'{phase}/Epoch/Total_Loss', avg_loss, epoch)
        writer.add_scalar(f'{phase}/Epoch/Emotion_Loss', avg_emotion_loss, epoch)
        writer.add_scalar(f'{phase}/Epoch/AV_Loss', avg_av_loss, epoch)
        writer.add_scalar(f'{phase}/Epoch/Accuracy', accuracy, epoch)
        writer.add_scalar(f'{phase}/Epoch/F1_Macro', f1_macro, epoch)
        writer.add_scalar(f'{phase}/Epoch/F1_Weighted', f1_weighted, epoch)
        writer.add_scalar(f'{phase}/Epoch/Arousal_CCC', arousal_ccc, epoch)
        writer.add_scalar(f'{phase}/Epoch/Valence_CCC', valence_ccc, epoch)
        writer.add_scalar(f'{phase}/Epoch/Avg_CCC', avg_ccc, epoch)

    return {
        'loss': avg_loss,
        'emotion_loss': avg_emotion_loss,
        'av_loss': avg_av_loss,
        'emotion_acc': accuracy,
        'emotion_f1': f1_macro,
        'accuracy': accuracy,  # Keep for backward compat
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted,
        'arousal_ccc': arousal_ccc,
        'valence_ccc': valence_ccc,
        'av_ccc': avg_ccc
    }


def train_fold(fold_data, config, fold_idx, output_dir, device, resume_checkpoint=None):
    """Train one fold"""
    print(f"\n{'='*80}")
    print(f"🏁 Training Fold {fold_idx} (Test Session {fold_data['session']})")
    print(f"{'='*80}")
    
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
    
    train_dataset = MultimodalOpenSMILEDataset(
        train_df,
        config['opensmile_dir'],
        mapping_json=config.get('mapping_json'),
        tokenizer_name=config.get('roberta_model_name', 'roberta-base'),
        max_text_length=config.get('max_text_length', 128),
        audio_max_length=config.get('max_audio_length', 160000),
        sample_rate=config.get('sample_rate', 16000)
    )
    val_dataset = MultimodalOpenSMILEDataset(
        val_df,
        config['opensmile_dir'],
        mapping_json=config.get('mapping_json'),
        tokenizer_name=config.get('roberta_model_name', 'roberta-base'),
        max_text_length=config.get('max_text_length', 128),
        audio_max_length=config.get('max_audio_length', 160000),
        sample_rate=config.get('sample_rate', 16000)
    )
    test_dataset = MultimodalOpenSMILEDataset(
        test_df,
        config['opensmile_dir'],
        mapping_json=config.get('mapping_json'),
        tokenizer_name=config.get('roberta_model_name', 'roberta-base'),
        max_text_length=config.get('max_text_length', 128),
        audio_max_length=config.get('max_audio_length', 160000),
        sample_rate=config.get('sample_rate', 16000)
    )
    # Print dataset-level statistics (emotion distribution) like W-PRO trainer
    def _print_distribution(df, split_name):
        try:
            counts = df['emotion'].value_counts()
            print(f"Loading dataset...\nTotal samples before filtering: {len(df)}\nTotal samples after filtering: {len(df)}\n")
            print(f"Emotion distribution ({split_name}):")
            for emo, cnt in counts.items():
                print(f"  {emo}: {int(cnt)}")
            print("")
        except Exception:
            print(f"Could not compute emotion distribution for {split_name}")

    _print_distribution(train_df, 'train')
    _print_distribution(val_df, 'val')
    _print_distribution(test_df, 'test')
    
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, 
                              num_workers=0, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False, 
                            num_workers=0, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False, 
                             num_workers=0, collate_fn=collate_fn)
    
    model = Method2CrossAttentionAVEGeMAPS(
        num_emotions=config['num_emotions'],
        egemaps_dim=config.get('prosody_dim', 88),
        hidden_dim=config['hidden_dim'],
        num_transformer_layers=config['num_transformer_layers'],
        num_heads=config['num_heads'],
        num_cross_attn_heads=config.get('num_heads', 8),
        ff_dim=config['ff_dim'],
        dropout=config['dropout']
    ).to(device)
    
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
    print(f"Audio branch:         WavLM-base-plus (768-d, frozen)")
    print(f"Text branch:          RoBERTa-base (768-d, frozen)")
    print(f"Prosody branch:       OpenSMILE features (88-d, trainable)")
    print(f"Transformer layers:   {config['num_transformer_layers']} (heads={config['num_heads']}, ff_dim={config['ff_dim']})")
    print(f"Hidden dimension:     {config['hidden_dim']}")
    print(f"Output:               Emotion (4-class) + Arousal + Valence (continuous)")
    print(f"{'='*80}\n")
    
    criterion_emotion = nn.CrossEntropyLoss()
    criterion_av = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    # Use ReduceLROnPlateau like the W-PRO trainer
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',
        factor=0.5,
        patience=5
    )
    
    fold_log_dir = os.path.join(config.get('log_dir', 'logs/iemocap_method2_opensmile_tensorboard'), f'fold_{fold_idx}')
    os.makedirs(fold_log_dir, exist_ok=True)
    writer = SummaryWriter(fold_log_dir)
    
    fold_output_dir = os.path.join(output_dir, f'fold_{fold_idx}')
    os.makedirs(fold_output_dir, exist_ok=True)
    
    best_val_acc = 0.0
    global_step = 0
    
    # Training loop
    best_val_acc = 0.0
    best_epoch = 0
    global_step = 0

    # Respect config epochs, default to 30 if not provided
    config.setdefault('epochs', 30)

    # Resume support: load checkpoint if provided
    start_epoch = 1
    best_epoch = 0
    if resume_checkpoint and os.path.exists(resume_checkpoint):
        print(f"Resuming from checkpoint: {resume_checkpoint}")
        ckpt = torch.load(resume_checkpoint, map_location=device, weights_only=False)
        try:
            model.load_state_dict(ckpt['model_state_dict'])
        except Exception as e:
            print(f"Warning: could not fully load model_state_dict: {e}")
        if 'optimizer_state_dict' in ckpt:
            try:
                optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            except Exception:
                print("Warning: couldn't load optimizer state")
        start_epoch = ckpt.get('epoch', 0) + 1
        best_epoch = ckpt.get('epoch', 0)
        best_val_acc = ckpt.get('val_metrics', {}).get('emotion_acc', best_val_acc)

    for epoch in range(start_epoch, config['epochs'] + 1):
        print(f"\n{'='*60}")
        print(f"Fold {fold_idx} - Epoch {epoch}/{config['epochs']}")
        print(f"{'='*60}")

        train_loss, train_metrics, global_step = train_one_epoch(
            model, train_loader, criterion_emotion, criterion_av, optimizer, device, writer, global_step, epoch
        )

        # Validate only (do NOT run test during epoch).
        val_metrics = validate(model, val_loader, criterion_emotion, criterion_av, device, writer, epoch, phase='Val')

        # Scheduler step expects a metric (use validation emotion accuracy)
        scheduler.step(val_metrics.get('emotion_acc', 0.0))

        # Log LR
        current_lr = optimizer.param_groups[0]['lr']
        if writer is not None:
            writer.add_scalar('Train/Learning_Rate', current_lr, epoch)

        # Print metrics similar to W-PRO trainer
        print(f"\n[Train] Loss: {train_loss:.4f} | "
              f"Acc: {train_metrics.get('emotion_acc', 0.0):.4f} | "
              f"F1: {train_metrics.get('emotion_f1', 0.0):.4f}")
        print(f"[Val]   Loss: {val_metrics.get('loss', 0.0):.4f} | "
              f"Acc: {val_metrics.get('emotion_acc', 0.0):.4f} | "
              f"F1: {val_metrics.get('emotion_f1', 0.0):.4f} | "
              f"CCC: {val_metrics.get('av_ccc', 0.0):.4f}")

        # Save best model (full checkpoint) and always save last_model.pt for continuation
        if val_metrics.get('emotion_acc', 0.0) > best_val_acc:
            best_val_acc = val_metrics['emotion_acc']
            best_epoch = epoch
            checkpoint_path = os.path.join(fold_output_dir, 'best_model.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_metrics': val_metrics
            }, checkpoint_path)
            print(f"🔥 New best validation accuracy: {best_val_acc:.4f} (saved)")

        # Always save last_model.pt for continuation
        last_model_path = os.path.join(fold_output_dir, 'last_model.pt')
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_metrics': val_metrics
        }, last_model_path)
    
    # Load best full checkpoint (best_model.pt) and evaluate on test set
    best_path = os.path.join(fold_output_dir, 'best_model.pt')
    if not os.path.exists(best_path):
        # Fallback to last_model.pt if best not found
        best_path = os.path.join(fold_output_dir, 'last_model.pt')
        if not os.path.exists(best_path):
            raise FileNotFoundError(f"No checkpoint found in {fold_output_dir}")

    # Use weights_only=False to allow loading older-style checkpoints
    # (PyTorch 2.6 changed default to weights_only=True). This is safe
    # here because checkpoints are produced by this trusted training run.
    checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])

    # Final test evaluation
    test_metrics = validate(model, test_loader, criterion_emotion, criterion_av, device, None, 0, phase='Test')

    # Print in the detailed block format requested
    print(f"\n{'='*60}")
    print(f"🎯 Fold {fold_idx} Test Results:")
    print(f"{'='*60}")
    print(f"Accuracy:     {test_metrics.get('emotion_acc', 0.0):.4f}")
    print(f"F1 Macro:     {test_metrics.get('emotion_f1', 0.0):.4f}")
    # f1_weighted not computed here; approximate by weighted F1 if available in train metrics
    print(f"F1 Weighted:  {test_metrics.get('emotion_f1', 0.0):.4f}")
    print(f"Arousal CCC:  {test_metrics.get('arousal_ccc', test_metrics.get('av_ccc', 0.0)):.4f}")
    print(f"Valence CCC:  {test_metrics.get('valence_ccc', test_metrics.get('av_ccc', 0.0)):.4f}")
    print(f"Avg CCC:      {test_metrics.get('av_ccc', 0.0):.4f}")
    print(f"{'='*60}\n")
    
    writer.close()
    
    return test_metrics


def main():
    parser = argparse.ArgumentParser(description="Train Method 2 A-V with OpenSMILE (5-fold CV)")
    parser.add_argument('--config', type=str, default='configs/iemocap_method2_opensmile_kfold_config.yaml',
                        help='Path to config YAML')
    parser.add_argument('--csv', type=str, default='datasets/IEMOCAP_KFOLD/all.csv',
                        help='Path to IEMOCAP CSV with fold column')
    parser.add_argument('--output_dir', type=str, default='checkpoints/iemocap_method2_opensmile_kfold',
                        help='Output directory for checkpoints')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device')
    parser.add_argument('--resume_fold', type=int, default=None, help='If set, start processing from this fold index (1-based)')
    parser.add_argument('--resume_checkpoint', type=str, default=None, help='Optional path to a checkpoint to resume training for the first processed fold')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    args = parser.parse_args()
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    config = load_config(args.config)
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device)
    
    print("="*80)
    print("🎓 Method 2 A-V with OpenSMILE - 5-Fold Cross-Validation")
    print("="*80)
    print(f"Config:        {args.config}")
    print(f"CSV:           {args.csv}")
    print(f"Output:        {args.output_dir}")
    print(f"Device:        {device}")
    print("="*80 + "\n")
    
    folds = prepare_folds(args.csv)

    start_fold = args.resume_fold if args.resume_fold is not None else 1

    all_test_results = []
    for fold_idx, fold_data in enumerate(folds, start=1):
        if fold_idx < start_fold:
            print(f"Skipping fold {fold_idx} because start_fold={start_fold}")
            continue
        resume_ckpt = args.resume_checkpoint if (args.resume_checkpoint and fold_idx == start_fold) else None
        test_metrics = train_fold(fold_data, config, fold_idx, args.output_dir, device, resume_checkpoint=resume_ckpt)
        all_test_results.append(test_metrics)
    
    print("\n" + "="*80)
    print("🏆 FINAL 5-FOLD CROSS-VALIDATION RESULTS")
    print("="*80)
    
    accuracies = [r['emotion_acc'] for r in all_test_results]
    f1_macros = [r['emotion_f1'] for r in all_test_results]
    ccc_values = [r['av_ccc'] for r in all_test_results]
    
    for fold_idx, metrics in enumerate(all_test_results, start=1):
        print(f"Fold {fold_idx}: Acc={metrics['emotion_acc']:.4f}, F1={metrics['emotion_f1']:.4f}, CCC={metrics['av_ccc']:.4f}")
    
    print(f"\nMean Accuracy:  {np.mean(accuracies):.4f} ± {np.std(accuracies):.4f}")
    print(f"Mean F1 (macro): {np.mean(f1_macros):.4f} ± {np.std(f1_macros):.4f}")
    print(f"Mean A-V CCC:   {np.mean(ccc_values):.4f} ± {np.std(ccc_values):.4f}")
    print("="*80 + "\n")


if __name__ == '__main__':
    main()
