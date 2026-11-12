"""
5-Fold Cross-Validation Training for Method 2 (W-PRO) - No A/V supervision.

This keeps the same fold preparation, logging style, checkpoint structure, and
summary printing as the original k-fold trainer, but trains a classification-only
variant with the same 3-backbone architecture.
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

# Add parent directory to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from models.method2_crossattn_av_ablation_no_avsup import Method2CrossAttentionAV_AblationNoAVSup
from train.losses import FocalCrossEntropyLoss
from data.iemocap_dataset import IEMOCAPDataset


def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def compute_flops(model, audio_length=16000 * 10, text_length=128, device='cpu'):
    try:
        audio_input = torch.randn(1, audio_length).to(device)
        text_input = torch.randint(0, 50000, (1, text_length)).to(device)
        text_mask = torch.ones(1, text_length).to(device)

        flops, params = profile(
            model,
            inputs=(audio_input, text_input, None, text_mask),
            verbose=False,
        )
        flops_str, params_str = clever_format([flops, params], "%.3f")
        return flops, flops_str, params_str
    except Exception as e:
        print(f"Warning: FLOPs computation failed: {e}")
        return None, "N/A", "N/A"


def prepare_folds(csv_path):
    df = pd.read_csv(csv_path)

    max_fold = df['fold'].max()
    if max_fold == 10:
        fold_to_session = {
            1: 1, 2: 1,
            3: 2, 4: 2,
            5: 3, 6: 3,
            7: 4, 8: 4,
            9: 5, 10: 5,
        }
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
            'train_val': train_val_df,
        })
    return folds


def train_one_epoch(model, loader, criterion, optimizer, device, writer, global_step, epoch):
    model.train()

    total_loss = 0.0
    all_preds = []
    all_labels = []

    pbar = tqdm(loader, desc=f"Epoch {epoch} [Train]", leave=False)

    for batch_idx, batch in enumerate(pbar):
        audio = batch['audio'].to(device)
        text_input_ids = batch['text_input_ids'].to(device)
        text_attention_mask = batch['text_attention_mask'].to(device)
        emotion_label = batch['emotion_label'].to(device)

        optimizer.zero_grad()
        outputs = model(
            audio_input=audio,
            text_input=text_input_ids,
            text_attention_mask=text_attention_mask,
        )

        loss = criterion(outputs['emotion_logits'], emotion_label)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()

        preds = torch.argmax(outputs['emotion_logits'], dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(emotion_label.cpu().numpy())

        pbar.set_postfix({
            'loss': f"{loss.item():.4f}",
            'acc': f"{accuracy_score(all_labels, all_preds):.3f}",
        })

        if writer is not None and batch_idx % 10 == 0:
            writer.add_scalar('Train/Batch/Total_Loss', loss.item(), global_step)
            global_step += 1

    avg_loss = total_loss / len(loader)
    accuracy = accuracy_score(all_labels, all_preds)
    f1_macro = f1_score(all_labels, all_preds, average='macro')
    f1_weighted = f1_score(all_labels, all_preds, average='weighted')

    if writer is not None:
        writer.add_scalar('Train/Epoch/Total_Loss', avg_loss, epoch)
        writer.add_scalar('Train/Epoch/Accuracy', accuracy, epoch)
        writer.add_scalar('Train/Epoch/F1_Macro', f1_macro, epoch)
        writer.add_scalar('Train/Epoch/F1_Weighted', f1_weighted, epoch)

    return {
        'loss': avg_loss,
        'accuracy': accuracy,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted,
    }, global_step


def validate(model, loader, criterion, device, writer, epoch, phase='Val'):
    model.eval()

    total_loss = 0.0
    all_preds = []
    all_labels = []

    pbar = tqdm(loader, desc=f"Epoch {epoch} [{phase}]", leave=False)

    with torch.no_grad():
        for batch in pbar:
            audio = batch['audio'].to(device)
            text_input_ids = batch['text_input_ids'].to(device)
            text_attention_mask = batch['text_attention_mask'].to(device)
            emotion_label = batch['emotion_label'].to(device)

            outputs = model(
                audio_input=audio,
                text_input=text_input_ids,
                text_attention_mask=text_attention_mask,
            )

            loss = criterion(outputs['emotion_logits'], emotion_label)
            total_loss += loss.item()

            preds = torch.argmax(outputs['emotion_logits'], dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(emotion_label.cpu().numpy())

            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'acc': f"{accuracy_score(all_labels, all_preds):.3f}",
            })

    avg_loss = total_loss / len(loader)
    accuracy = accuracy_score(all_labels, all_preds)
    f1_macro = f1_score(all_labels, all_preds, average='macro')
    f1_weighted = f1_score(all_labels, all_preds, average='weighted')

    if writer is not None:
        writer.add_scalar(f'{phase}/Epoch/Total_Loss', avg_loss, epoch)
        writer.add_scalar(f'{phase}/Epoch/Accuracy', accuracy, epoch)
        writer.add_scalar(f'{phase}/Epoch/F1_Macro', f1_macro, epoch)
        writer.add_scalar(f'{phase}/Epoch/F1_Weighted', f1_weighted, epoch)

    return {
        'loss': avg_loss,
        'accuracy': accuracy,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted,
    }


def train_fold(fold_data, config, fold_idx, output_dir, device, resume_checkpoint=None):
    print(f"\n{'='*80}")
    print(f"🏁 Training Fold {fold_idx} (Test Session {fold_data['session']})")
    print(f"{'='*80}")

    train_df, val_df = train_test_split(
        fold_data['train_val'],
        test_size=0.2,
        random_state=42,
        stratify=fold_data['train_val']['emotion'],
    )
    test_df = fold_data['test']

    print(f"Train: {len(train_df)} samples")
    print(f"Val:   {len(val_df)} samples")
    print(f"Test:  {len(test_df)} samples")

    audio_max_length = config['data'].get('max_audio_length', 160000)
    max_text_length = config['data'].get('max_text_length', 128)

    train_dataset = IEMOCAPDataset(
        train_df,
        audio_max_length=audio_max_length,
        max_text_length=max_text_length,
        tokenizer_name=config['model'].get('roberta_model_name', 'roberta-base'),
    )
    val_dataset = IEMOCAPDataset(
        val_df,
        audio_max_length=audio_max_length,
        max_text_length=max_text_length,
        tokenizer_name=config['model'].get('roberta_model_name', 'roberta-base'),
    )
    test_dataset = IEMOCAPDataset(
        test_df,
        audio_max_length=audio_max_length,
        max_text_length=max_text_length,
        tokenizer_name=config['model'].get('roberta_model_name', 'roberta-base'),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['training']['num_workers'],
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['training']['num_workers'],
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['training']['num_workers'],
        pin_memory=True,
    )

    model = Method2CrossAttentionAV_AblationNoAVSup(
        num_emotions=config['model']['num_emotions'],
        wavlm_model_name=config['model']['wavlm_model_name'],
        wavlm_prosody_model_name=config['model'].get('wavlm_prosody_model_name', None),
        roberta_model_name=config['model'].get('roberta_model_name', 'roberta-base'),
        hidden_dim=config['model']['hidden_dim'],
        av_refinement_layers=config['model'].get('av_refinement_layers', 2),
        freeze_wavlm=config['model']['freeze_wavlm'],
        freeze_roberta=config['model'].get('freeze_roberta', False),
        num_cross_attn_heads=config['model']['num_cross_attn_heads'],
        post_attention_layers=config['model'].get('post_attention_layers', 2),
        av_prediction_layers=config['model'].get('av_prediction_layers', 3),
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
        audio_length=audio_max_length,
        text_length=max_text_length,
        device=device,
    )
    print(f"FLOPs:  {flops_str}")
    print(f"Params: {params_str}")

    print(f"\n{'='*80}")
    print(f"🏗️  ARCHITECTURE SUMMARY")
    print(f"{'='*80}")
    print(f"1. Wpro (WavLM):          Extract acoustic/prosody features")
    print(f"2. RoBERTa:               Extract semantic/linguistic features from text")
    print(f"3. Cross-Attention:       Wpro ↔ RoBERTa bidirectional attention")
    print(f"4. Post-Attention:        {config['model'].get('post_attention_layers', 2)} Transformer layers per branch")
    print(f"5. Branch Fusion:         Gated fusion of Wpro + RoBERTa")
    print(f"6. WavLM Final:           Global acoustic features")
    print(f"7. Final Fusion:          WavLM + eA + eV -> Emotion")
    print(f"8. No A/V supervision:    No A/V heads, no CCC loss, no AV refinement")
    print(f"{'='*80}\n")

    use_focal = config['loss'].get('use_focal', False)
    if use_focal:
        criterion = FocalCrossEntropyLoss(
            alpha=config['loss'].get('focal_alpha', 1.0),
            gamma=config['loss'].get('focal_gamma', 2.0),
        )
    else:
        criterion = nn.CrossEntropyLoss()

    print(f"\n{'='*80}")
    print(f"📉 LOSS FUNCTION")
    print(f"{'='*80}")
    if use_focal:
        print(f"Type: FocalCrossEntropyLoss")
        print(f"  L_emotion = FocalCE(alpha={config['loss'].get('focal_alpha', 1.0)}, gamma={config['loss'].get('focal_gamma', 2.0)})")
    else:
        print(f"Type: CrossEntropyLoss")
        print(f"  L_emotion = CrossEntropyLoss(logits, labels)")
    print(f"  L_total = L_emotion")
    print(f"{'='*80}\n")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay'],
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',
        factor=0.5,
        patience=5,
    )

    start_epoch = 1
    best_val_acc = 0.0

    if resume_checkpoint and os.path.exists(resume_checkpoint):
        print(f"\n🔄 Resuming from checkpoint: {resume_checkpoint}")
        checkpoint = torch.load(resume_checkpoint, map_location=device, weights_only=False)

        # Some older checkpoints may contain modules/metadata no longer present
        # in the current ablation model (e.g., branch_fusion, THOP hooks).
        state_dict = checkpoint.get('model_state_dict', {})
        incompatible = model.load_state_dict(state_dict, strict=False)
        if incompatible.unexpected_keys:
            print(f"   Ignored unexpected keys: {len(incompatible.unexpected_keys)}")
        if incompatible.missing_keys:
            print(f"   Missing keys in checkpoint: {len(incompatible.missing_keys)}")

        # Optimizer state can also mismatch when parameter groups changed.
        if 'optimizer_state_dict' in checkpoint:
            try:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            except Exception as e:
                print(f"   Warning: could not load optimizer state, using fresh optimizer: {e}")

        start_epoch = checkpoint['epoch'] + 1
        best_val_acc = checkpoint.get('val_accuracy', 0.0)
        print(f"   Starting from epoch {start_epoch}")
        print(f"   Previous best val accuracy: {best_val_acc:.4f}\n")

    log_dir = os.path.join(output_dir, f'logs/fold_{fold_idx}')
    writer = SummaryWriter(log_dir)

    best_epoch = 0
    global_step = 0
    checkpoint_dir = os.path.join(output_dir, f'fold_{fold_idx}')
    os.makedirs(checkpoint_dir, exist_ok=True)

    for epoch in range(start_epoch, config['training']['epochs'] + 1):
        print(f"\n{'='*60}")
        print(f"Fold {fold_idx} - Epoch {epoch}/{config['training']['epochs']}")
        print(f"{'='*60}")

        train_metrics, global_step = train_one_epoch(
            model, train_loader, criterion, optimizer, device, writer, global_step, epoch
        )
        val_metrics = validate(model, val_loader, criterion, device, writer, epoch, phase='Val')

        scheduler.step(val_metrics['accuracy'])
        current_lr = optimizer.param_groups[0]['lr']
        writer.add_scalar('Train/Learning_Rate', current_lr, epoch)

        print(
            f"\n[Train] Loss: {train_metrics['loss']:.4f} | "
            f"Acc: {train_metrics['accuracy']:.4f} | "
            f"F1: {train_metrics['f1_macro']:.4f}"
        )
        print(
            f"[Val]   Loss: {val_metrics['loss']:.4f} | "
            f"Acc: {val_metrics['accuracy']:.4f} | "
            f"F1: {val_metrics['f1_macro']:.4f}"
        )

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
            }, checkpoint_path)
            print(f"🔥 New best validation accuracy: {best_val_acc:.4f} (saved)")

        last_model_path = os.path.join(checkpoint_dir, 'last_model.pt')
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_accuracy': val_metrics['accuracy'],
            'val_f1_macro': val_metrics['f1_macro'],
        }, last_model_path)

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
    # Allow extra keys in checkpoint (e.g., from older model versions)
    state_dict = checkpoint.get('model_state_dict', {})
    incompatible = model.load_state_dict(state_dict, strict=False)
    if incompatible.unexpected_keys:
        print(f"   Ignored unexpected keys: {len(incompatible.unexpected_keys)}")
    if incompatible.missing_keys:
        print(f"   Missing keys in checkpoint: {len(incompatible.missing_keys)}")

    test_metrics = validate(model, test_loader, criterion, device, writer, 0, phase='Test')

    print(f"\n{'='*60}")
    print(f"🎯 Fold {fold_idx} Test Results:")
    print(f"{'='*60}")
    print(f"Accuracy:    {test_metrics['accuracy']:.4f}")
    print(f"F1 Macro:    {test_metrics['f1_macro']:.4f}")
    print(f"F1 Weighted: {test_metrics['f1_weighted']:.4f}")

    writer.close()
    return test_metrics


def main():
    parser = argparse.ArgumentParser(description='5-Fold Cross-Validation Training for Method 2 (W-PRO) - No A/V supervision')
    parser.add_argument('--config', type=str, default='configs/method2_wpro_ablation_no_avsup_config.yaml',
                        help='Path to config file')
    parser.add_argument('--csv', type=str, default='datasets/IEMOCAP_KFOLD/tenfolds/all.csv',
                        help='Path to CSV file with fold information')
    parser.add_argument('--output_dir', type=str, default='checkpoints/method2_kfold_wpro_ablation_no_avsup',
                        help='Output directory for checkpoints and logs')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='Device to use (cuda:0, cuda:1, cpu)')
    parser.add_argument('--resume_fold', type=int, default=None,
                        help='Resume training from specific fold')
    parser.add_argument('--resume_checkpoint', type=str, default=None,
                        help='Path to checkpoint to resume from')

    args = parser.parse_args()

    config = load_config(args.config)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    print(f"\n{'='*80}")
    print(f"🚀 STARTING 5-FOLD CROSS-VALIDATION TRAINING")
    print(f"{'='*80}")
    print(f"Device: {device}")
    print(f"Config: {args.config}")
    print(f"CSV: {args.csv}")
    print(f"Output: {args.output_dir}")
    if args.resume_fold:
        print(f"Resume from fold: {args.resume_fold}")
    if args.resume_checkpoint:
        print(f"Resume checkpoint: {args.resume_checkpoint}")
    print(f"{'='*80}\n")

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'logs'), exist_ok=True)

    folds = prepare_folds(args.csv)
    all_results = []
    start_fold = args.resume_fold if args.resume_fold else 1

    for fold_idx, fold_data in enumerate(folds[start_fold - 1:], start_fold):
        resume_ckpt = args.resume_checkpoint if fold_idx == start_fold else None
        fold_results = train_fold(fold_data, config, fold_idx, args.output_dir, device, resume_ckpt)
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

    results_path = os.path.join(args.output_dir, 'kfold_results.csv')
    results_df.to_csv(results_path, index=False)
    print(f"\n💾 Results saved to: {results_path}")


if __name__ == '__main__':
    main()
