"""
Method2 W-PRO with ASR Text - 5-Fold Cross-Validation (audio + text only)

This mirrors the standard Method2 W-PRO trainer but swaps transcripts with
ASR text before building the existing IEMOCAPDataset. No video branch is used.
"""
import os
import sys
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import yaml
import argparse
from tqdm import tqdm
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from thop import profile, clever_format

# Add repo root to import path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from models.method2_crossattn_av import Method2CrossAttentionAV
from train.losses import CombinedEmotionLossCCC
from data.iemocap_dataset import IEMOCAPDataset


def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def compute_ccc(y_true, y_pred):
    """Compute Concordance Correlation Coefficient."""
    mean_true = np.mean(y_true)
    mean_pred = np.mean(y_pred)
    var_true = np.var(y_true)
    var_pred = np.var(y_pred)
    covariance = np.mean((y_true - mean_true) * (y_pred - mean_pred))
    ccc = (2 * covariance) / (var_true + var_pred + (mean_true - mean_pred) ** 2 + 1e-8)
    return ccc


def count_parameters(model):
    """Count total and trainable parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def compute_flops(model, audio_length=16000 * 10, text_length=128, device='cpu'):
    """Compute FLOPs for the model."""
    try:
        audio_input = torch.randn(1, audio_length).to(device)
        text_input = torch.randint(0, 50000, (1, text_length)).to(device)
        text_mask = torch.ones(1, text_length).to(device)

        flops, params = profile(
            model,
            inputs=(audio_input, text_input, None, text_mask),
            verbose=False
        )
        flops_str, params_str = clever_format([flops, params], "%.3f")
        return flops, flops_str, params_str
    except Exception as e:
        print(f"Warning: FLOPs computation failed: {e}")
        return None, "N/A", "N/A"


def prepare_folds(csv_path):
    """Prepare 5 folds mapping sessions (matches standard Method2 trainer)."""
    df = pd.read_csv(csv_path)

    unique_folds = df['fold'].unique()
    max_fold = df['fold'].max()

    if max_fold == 10:
        fold_to_session = {
            1: 1, 2: 1,
            3: 2, 4: 2,
            5: 3, 6: 3,
            7: 4, 8: 4,
            9: 5, 10: 5
        }
        df['session'] = df['fold'].map(fold_to_session)
    else:
        df['session'] = df['fold']

    folds = []
    for session in range(1, 6):
        test_df = df[df['session'] == session]
        train_val_df = df[df['session'] != session]
        folds.append({'session': session, 'test': test_df, 'train_val': train_val_df})

    return folds


def apply_asr_transcript(df, asr_column):
    """Replace transcript with ASR column so the standard dataset can be reused."""
    df = df.copy()
    if asr_column not in df.columns:
        raise ValueError(f"ASR column '{asr_column}' not found in CSV")
    df['transcript'] = df[asr_column].fillna("")
    return df


def train_one_epoch(model, loader, criterion, optimizer, device, writer, global_step, epoch):
    """Train for one epoch (identical logging to standard Method2)."""
    model.train()

    total_loss = 0.0
    total_emotion_loss = 0.0
    total_arousal_loss = 0.0
    total_valence_loss = 0.0

    all_preds = []
    all_labels = []

    pbar = tqdm(loader, desc=f"Epoch {epoch} [Train]", leave=False)

    for batch_idx, batch in enumerate(pbar):
        audio = batch['audio'].to(device)
        text_input_ids = batch['text_input_ids'].to(device)
        text_attention_mask = batch['text_attention_mask'].to(device)
        emotion_label = batch['emotion_label'].to(device)
        arousal = batch['arousal_value'].to(device)
        valence = batch['valence_value'].to(device)

        optimizer.zero_grad()
        outputs = model(
            audio_input=audio,
            text_input=text_input_ids,
            text_attention_mask=text_attention_mask
        )

        loss_dict = criterion(
            outputs,
            {
                'emotion_label': emotion_label,
                'arousal_value': arousal,
                'valence_value': valence
            }
        )

        loss = loss_dict['total_loss']
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        total_emotion_loss += loss_dict['emotion_loss'].item()
        total_arousal_loss += loss_dict['arousal_loss'].item()
        total_valence_loss += loss_dict['valence_loss'].item()

        preds = torch.argmax(outputs['emotion_logits'], dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(emotion_label.cpu().numpy())

        pbar.set_postfix({
            'loss': f"{loss.item():.4f}",
            'acc': f"{accuracy_score(all_labels, all_preds):.3f}"
        })

        if writer is not None and batch_idx % 10 == 0:
            writer.add_scalar('Train/Batch/Total_Loss', loss.item(), global_step)
            writer.add_scalar('Train/Batch/Emotion_Loss', loss_dict['emotion_loss'].item(), global_step)
            writer.add_scalar('Train/Batch/Arousal_Loss', loss_dict['arousal_loss'].item(), global_step)
            writer.add_scalar('Train/Batch/Valence_Loss', loss_dict['valence_loss'].item(), global_step)
            global_step += 1

    avg_loss = total_loss / len(loader)
    avg_emotion_loss = total_emotion_loss / len(loader)
    avg_arousal_loss = total_arousal_loss / len(loader)
    avg_valence_loss = total_valence_loss / len(loader)
    accuracy = accuracy_score(all_labels, all_preds)
    f1_macro = f1_score(all_labels, all_preds, average='macro')
    f1_weighted = f1_score(all_labels, all_preds, average='weighted')

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
    """Validate model."""
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
            text_input_ids = batch['text_input_ids'].to(device)
            text_attention_mask = batch['text_attention_mask'].to(device)
            emotion_label = batch['emotion_label'].to(device)
            arousal = batch['arousal_value'].to(device)
            valence = batch['valence_value'].to(device)

            outputs = model(
                audio_input=audio,
                text_input=text_input_ids,
                text_attention_mask=text_attention_mask
            )

            loss_dict = criterion(
                outputs,
                {
                    'emotion_label': emotion_label,
                    'arousal_value': arousal,
                    'valence_value': valence
                }
            )

            loss = loss_dict['total_loss']
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

            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'acc': f"{accuracy_score(all_labels, all_preds):.3f}"
            })

    avg_loss = total_loss / len(loader)
    avg_emotion_loss = total_emotion_loss / len(loader)
    avg_arousal_loss = total_arousal_loss / len(loader)
    avg_valence_loss = total_valence_loss / len(loader)

    accuracy = accuracy_score(all_labels, all_preds)
    f1_macro = f1_score(all_labels, all_preds, average='macro')
    f1_weighted = f1_score(all_labels, all_preds, average='weighted')

    all_arousal_pred = np.array(all_arousal_pred).flatten()
    all_arousal_true = np.array(all_arousal_true).flatten()
    all_valence_pred = np.array(all_valence_pred).flatten()
    all_valence_true = np.array(all_valence_true).flatten()

    arousal_ccc = compute_ccc(all_arousal_true, all_arousal_pred)
    valence_ccc = compute_ccc(all_valence_true, all_valence_pred)
    avg_ccc = (arousal_ccc + valence_ccc) / 2

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


def train_fold(fold_data, config, fold_idx, output_dir, device, asr_column, resume_checkpoint=None):
    """Train one fold with ASR transcripts replacing ground truth text."""
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

    # Swap in ASR transcripts
    train_df = apply_asr_transcript(train_df, asr_column)
    val_df = apply_asr_transcript(val_df, asr_column)
    test_df = apply_asr_transcript(test_df, asr_column)

    print(f"Train: {len(train_df)} samples")
    print(f"Val:   {len(val_df)} samples")
    print(f"Test:  {len(test_df)} samples")

    train_dataset = IEMOCAPDataset(
        train_df,
        audio_max_length=config['data'].get('max_audio_length', 160000),
        max_text_length=config['data'].get('max_text_length', 128)
    )

    val_dataset = IEMOCAPDataset(
        val_df,
        audio_max_length=config['data'].get('max_audio_length', 160000),
        max_text_length=config['data'].get('max_text_length', 128)
    )

    test_dataset = IEMOCAPDataset(
        test_df,
        audio_max_length=config['data'].get('max_audio_length', 160000),
        max_text_length=config['data'].get('max_text_length', 128)
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['training']['num_workers'],
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['training']['num_workers'],
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['training']['num_workers'],
        pin_memory=True
    )

    model = Method2CrossAttentionAV(
        num_emotions=config['model']['num_emotions'],
        wavlm_model_name=config['model']['wavlm_model_name'],
        wavlm_prosody_model_name=config['model'].get('wavlm_prosody_model_name', None),
        roberta_model_name=config['model'].get('roberta_model_name', 'roberta-base'),
        hidden_dim=config['model']['hidden_dim'],
        av_refinement_layers=config['model']['av_refinement_layers'],
        freeze_wavlm=config['model']['freeze_wavlm'],
        freeze_roberta=config['model'].get('freeze_roberta', False),
        num_cross_attn_heads=config['model']['num_cross_attn_heads'],
        post_attention_layers=config['model'].get('post_attention_layers', 2),
        av_prediction_layers=config['model'].get('av_prediction_layers', 3)
    ).to(device)

    total_params, trainable_params = count_parameters(model)
    print(f"\n{'='*80}")
    print(f"📊 MODEL STATISTICS")
    print(f"{'='*80}")
    print(f"Total parameters:     {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Frozen parameters:    {total_params - trainable_params:,}")

    print(f"\n⚡ Computing FLOPs...")
    _, flops_str, params_str = compute_flops(
        model,
        audio_length=config['data'].get('max_audio_length', 160000),
        text_length=config['data'].get('max_text_length', 128),
        device=device
    )
    print(f"FLOPs:  {flops_str}")
    print(f"Params: {params_str}")

    print(f"\n{'='*80}")
    print(f"📉 LOSS FUNCTION")
    print(f"{'='*80}")
    print(f"Type: CombinedEmotionLossCCC")
    print(f"  L_total = {config['loss']['emotion_weight']} * L_emotion")
    print(f"          + {config['loss']['arousal_weight']} * L_arousal_CCC")
    print(f"          + {config['loss']['valence_weight']} * L_valence_CCC")
    print(f"{'='*80}\n")

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
        factor=0.5,
        patience=5
    )

    start_epoch = 1
    best_val_acc = 0.0

    checkpoint_dir = os.path.join(output_dir, f'fold_{fold_idx}')
    os.makedirs(checkpoint_dir, exist_ok=True)

    if resume_checkpoint and os.path.exists(resume_checkpoint):
        print(f"\n🔄 Resuming from checkpoint: {resume_checkpoint}")
        checkpoint = torch.load(resume_checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_acc = checkpoint.get('val_accuracy', 0.0)
        print(f"   Starting from epoch {start_epoch}")
        print(f"   Previous best val accuracy: {best_val_acc:.4f}\n")

    writer = SummaryWriter(os.path.join(output_dir, f'logs/fold_{fold_idx}'))
    best_epoch = 0
    global_step = 0

    for epoch in range(start_epoch, config['training']['epochs'] + 1):
        print(f"\n{'='*60}")
        print(f"Fold {fold_idx} - Epoch {epoch}/{config['training']['epochs']}")
        print(f"{'='*60}")

        train_metrics, global_step = train_one_epoch(
            model, train_loader, criterion, optimizer, device,
            writer, global_step, epoch
        )

        val_metrics = validate(
            model, val_loader, criterion, device, writer, epoch, phase='Val'
        )

        scheduler.step(val_metrics['accuracy'])
        current_lr = optimizer.param_groups[0]['lr']
        writer.add_scalar('Train/Learning_Rate', current_lr, epoch)

        print(f"\n[Train] Loss: {train_metrics['loss']:.4f} | Acc: {train_metrics['accuracy']:.4f} | F1: {train_metrics['f1_macro']:.4f}")
        print(f"[Val]   Loss: {val_metrics['loss']:.4f} | Acc: {val_metrics['accuracy']:.4f} | F1: {val_metrics['f1_macro']:.4f} | CCC: {val_metrics['avg_ccc']:.4f}")

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

    print(f"\n{'='*60}")
    print(f"Loading best model from epoch {best_epoch}")
    print(f"{'='*60}")

    best_path = os.path.join(checkpoint_dir, 'best_model.pt')
    if not os.path.exists(best_path):
        if resume_checkpoint and os.path.exists(resume_checkpoint):
            print(f"No new best model found in {checkpoint_dir}; falling back to resume checkpoint: {resume_checkpoint}")
            best_path = resume_checkpoint
        else:
            raise FileNotFoundError(f"No best_model.pt found in {checkpoint_dir} and no resume checkpoint available.")

    checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])

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
    parser = argparse.ArgumentParser(description='5-Fold Training for Method2 W-PRO with ASR text')
    parser.add_argument('--config', type=str, default='configs/method2_wpro_asr_kfold_config.yaml',
                        help='Path to config file')
    parser.add_argument('--csv', type=str, default='datasets/IEMOCAP_KFOLD/tenfolds/all_asr.csv',
                        help='Path to CSV with ASR transcripts and fold column')
    parser.add_argument('--output_dir', type=str, default='checkpoints/method2_kfold_wpro_asr',
                        help='Output directory for checkpoints and logs')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='Device to use')
    parser.add_argument('--resume_fold', type=int, default=None,
                        help='Resume training from specific fold')
    parser.add_argument('--resume_checkpoint', type=str, default=None,
                        help='Path to checkpoint to resume from')

    args = parser.parse_args()

    config = load_config(args.config)
    asr_column = config['data'].get('asr_column', 'asr_transcript')

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*80}")
    print(f"🚀 STARTING 5-FOLD TRAINING (ASR)")
    print(f"{'='*80}")
    print(f"Device: {device}")
    print(f"Config: {args.config}")
    print(f"CSV: {args.csv}")
    print(f"Output: {args.output_dir}")
    if args.resume_fold:
        print(f"Resume from fold: {args.resume_fold}")
    if args.resume_checkpoint:
        print(f"Resume checkpoint: {args.resume_checkpoint}")
    print(f"ASR column: {asr_column}")
    print(f"{'='*80}\n")

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'logs'), exist_ok=True)

    print(f"Preparing 5-fold cross-validation splits...")
    folds = prepare_folds(args.csv)

    all_results = []
    start_fold = args.resume_fold if args.resume_fold else 1

    for fold_idx, fold_data in enumerate(folds[start_fold-1:], start_fold):
        resume_ckpt = args.resume_checkpoint if fold_idx == start_fold else None
        fold_results = train_fold(fold_data, config, fold_idx, args.output_dir, device, asr_column, resume_ckpt)
        fold_results['fold'] = fold_idx
        fold_results['session'] = fold_data['session']
        all_results.append(fold_results)

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

    results_path = os.path.join(args.output_dir, 'kfold_results.csv')
    results_df.to_csv(results_path, index=False)
    print(f"\n💾 Results saved to: {results_path}")


if __name__ == '__main__':
    main()


def load_config(config_path):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def train_epoch(model, dataloader, criterion, optimizer, device, epoch, num_epochs):
    """Train for one epoch"""
    model.train()
    total_loss = 0.0
    total_acc = 0.0
    num_samples = 0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch} [Train]")
    for batch_idx, batch in enumerate(pbar):
        audio = batch['audio'].to(device)
        video = batch['video'].to(device)
        text_input_ids = batch['text_input_ids'].to(device)
        text_attention_mask = batch['text_attention_mask'].to(device)
        emotion = batch['emotion'].to(device)
        arousal = batch['arousal'].to(device)
        valence = batch['valence'].to(device)
        dominance = batch['dominance'].to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        emotion_logits, arousal_pred, valence_pred, dominance_pred = model(
            audio, video, 
            text_input_ids=text_input_ids,
            text_attention_mask=text_attention_mask
        )
        
        # Compute loss
        loss = criterion(
            emotion_logits, emotion,
            arousal_pred, arousal,
            valence_pred, valence,
            dominance_pred, dominance
        )
        
        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        # Compute accuracy
        emotion_preds = emotion_logits.argmax(dim=1)
        acc = (emotion_preds == emotion).float().mean()
        
        total_loss += loss.item() * audio.size(0)
        total_acc += acc.item() * audio.size(0)
        num_samples += audio.size(0)
        
        pbar.set_postfix({'loss': f'{total_loss/num_samples:.4f}', 'acc': f'{total_acc/num_samples:.3f}'})
    
    avg_loss = total_loss / num_samples
    avg_acc = total_acc / num_samples
    
    return avg_loss, avg_acc


def validate(model, dataloader, criterion, device, epoch):
    """Validate for one epoch"""
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    num_samples = 0
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc=f"Epoch {epoch} [Val]")
        for batch in pbar:
            audio = batch['audio'].to(device)
            video = batch['video'].to(device)
            text_input_ids = batch['text_input_ids'].to(device)
            text_attention_mask = batch['text_attention_mask'].to(device)
            emotion = batch['emotion'].to(device)
            arousal = batch['arousal'].to(device)
            valence = batch['valence'].to(device)
            dominance = batch['dominance'].to(device)
            
            # Forward pass
            emotion_logits, arousal_pred, valence_pred, dominance_pred = model(
                audio, video,
                text_input_ids=text_input_ids,
                text_attention_mask=text_attention_mask
            )
            
            # Compute loss
            loss = criterion(
                emotion_logits, emotion,
                arousal_pred, arousal,
                valence_pred, valence,
                dominance_pred, dominance
            )
            
            # Compute accuracy
            emotion_preds = emotion_logits.argmax(dim=1)
            acc = (emotion_preds == emotion).float().mean()
            
            total_loss += loss.item() * audio.size(0)
            total_acc += acc.item() * audio.size(0)
            num_samples += audio.size(0)
            
            pbar.set_postfix({'loss': f'{total_loss/num_samples:.4f}', 'acc': f'{total_acc/num_samples:.3f}'})
    
    avg_loss = total_loss / num_samples
    avg_acc = total_acc / num_samples
    
    return avg_loss, avg_acc


def train_fold(fold_id, model, train_loader, val_loader, criterion, optimizer, scheduler, 
               device, num_epochs, checkpoint_dir, patience=5):
    """Train a single fold"""
    best_val_acc = 0.0
    patience_count = 0
    
    fold_checkpoint_dir = os.path.join(checkpoint_dir, f'fold_{fold_id}')
    os.makedirs(fold_checkpoint_dir, exist_ok=True)
    
    print(f"\n{'='*50}")
    print(f"Fold {fold_id} - Epoch 1/{num_epochs}")
    print(f"{'='*50}")
    
    for epoch in range(1, num_epochs + 1):
        print(f"\nFold {fold_id} - Epoch {epoch}/{num_epochs}")
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device, epoch, num_epochs)
        
        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device, epoch)
        
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.3f}")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.3f}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_count = 0
            
            best_model_path = os.path.join(fold_checkpoint_dir, 'best_model.pt')
            torch.save(model.state_dict(), best_model_path)
            print(f"✓ Best model saved: {best_model_path}")
        else:
            patience_count += 1
        
        # Save last model
        last_model_path = os.path.join(fold_checkpoint_dir, 'last_model.pt')
        torch.save(model.state_dict(), last_model_path)
        
        scheduler.step()
        
        # Early stopping
        if patience_count >= patience:
            print(f"Early stopping after {epoch} epochs (patience: {patience})")
            break
    
    print(f"\nFold {fold_id} completed. Best Val Acc: {best_val_acc:.3f}")
    return best_val_acc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--csv', type=str, required=True, help='Path to CSV with ASR text')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory for checkpoints')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device to use')
    parser.add_argument('--resume_fold', type=int, default=None, help='Resume from specific fold')
    parser.add_argument('--resume_checkpoint', type=str, default=None, help='Resume from checkpoint')
    
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    device = torch.device(args.device)
    print(f"Using device: {device}")
    
    # Create checkpoint directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Training parameters
    num_epochs = config.get('num_epochs', 30)
    batch_size = config.get('batch_size', 8)
    learning_rate = config.get('learning_rate', 1e-4)
    patience = config.get('patience', 5)
    num_folds = 5
    
    # Fold results
    fold_results = []
    
    # Determine starting fold
    start_fold = args.resume_fold if args.resume_fold else 1
    
    for fold_id in range(start_fold, num_folds + 1):
        print(f"\n{'='*70}")
        print(f"Training Fold {fold_id}/{num_folds}")
        print(f"{'='*70}")
        
        # Create dataset
        train_dataset = IEMOCAPDatasetKFoldASR(
            csv_path=args.csv,
            fold=fold_id,
            split='train',
            audio_root='datasets/IEMOCAP'
        )
        
        val_dataset = IEMOCAPDatasetKFoldASR(
            csv_path=args.csv,
            fold=fold_id,
            split='val',
            audio_root='datasets/IEMOCAP'
        )
        
        # Create dataloaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        
        print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
        
        # Create model
        model = Method2CrossAttentionAV(
            audio_dim=config.get('audio_dim', 768),
            video_dim=config.get('video_dim', 768),
            text_dim=config.get('text_dim', 768),
            hidden_dim=config.get('hidden_dim', 256),
            num_emotions=4,
            dropout=config.get('dropout', 0.1)
        )
        model = model.to(device)
        
        # Resume from checkpoint if provided
        if args.resume_checkpoint and fold_id == start_fold:
            print(f"Loading checkpoint: {args.resume_checkpoint}")
            checkpoint = torch.load(args.resume_checkpoint, map_location=device)
            model.load_state_dict(checkpoint)
            print("Checkpoint loaded!")
        
        # Loss and optimizer
        criterion = CombinedEmotionLossAVDCCC()
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=1e-5
        )
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=10,
            gamma=0.5
        )
        
        # Train fold
        best_val_acc = train_fold(
            fold_id=fold_id,
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            num_epochs=num_epochs,
            checkpoint_dir=args.output_dir,
            patience=patience
        )
        
        fold_results.append({
            'fold': fold_id,
            'best_val_acc': best_val_acc
        })
    
    # Print summary
    print(f"\n{'='*70}")
    print("Training Summary")
    print(f"{'='*70}")
    for result in fold_results:
        print(f"Fold {result['fold']}: Best Val Acc = {result['best_val_acc']:.3f}")
    
    if fold_results:
        mean_acc = np.mean([r['best_val_acc'] for r in fold_results])
        std_acc = np.std([r['best_val_acc'] for r in fold_results])
        print(f"\nMean Val Acc: {mean_acc:.3f} ± {std_acc:.3f}")
    
    print("\nTraining completed!")


if __name__ == '__main__':
    main()
