"""
IEMOCAP Method 2 (W-PRO) using precomputed embeddings (AV only).
Avoids encoder forward passes; consumes wavlm/wpro/roberta embeddings extracted offline.
Logging/prints mirror the raw-audio kfold script.
"""
import os
import argparse
import yaml
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

from models.method2_emb_fusion_av import Method2EmbeddingFusionAV
from train.losses import CombinedEmotionLossCCC
from data.iemocap_embeddings_npz_dataset import get_iemocap_emb_dataloaders_for_fold


def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def compute_ccc(y_true, y_pred):
    mean_true = np.mean(y_true)
    mean_pred = np.mean(y_pred)
    var_true = np.var(y_true)
    var_pred = np.var(y_pred)
    covariance = np.mean((y_true - mean_true) * (y_pred - mean_pred))
    return (2 * covariance) / (var_true + var_pred + (mean_true - mean_pred) ** 2 + 1e-8)


def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def train_one_epoch(model, loader, criterion, optimizer, device, writer, global_step, epoch):
    model.train()
    total_loss = 0.0
    total_emotion_loss = 0.0
    total_arousal_loss = 0.0
    total_valence_loss = 0.0

    all_preds = []
    all_labels = []

    for batch in loader:
        wavlm = batch['wavlm_emb'].to(device)
        wpro = batch['wpro_emb'].to(device)
        rob = batch['roberta_emb'].to(device)
        labels = batch['emotion_label'].to(device)
        arousal = batch['arousal_value'].to(device)
        valence = batch['valence_value'].to(device)

        outputs = model(wavlm, wpro, rob)
        loss_dict = criterion(outputs, {
            'emotion_label': labels,
            'arousal_value': arousal,
            'valence_value': valence,
        })

        loss = loss_dict['total_loss']
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_emotion_loss += loss_dict['emotion_loss'].item()
        total_arousal_loss += loss_dict['arousal_loss'].item()
        total_valence_loss += loss_dict['valence_loss'].item()

        preds = torch.argmax(outputs['emotion_logits'], dim=1)
        all_preds.extend(preds.cpu().numpy().tolist())
        all_labels.extend(labels.cpu().numpy().tolist())

        writer.add_scalar('Train/Batch_Loss', loss.item(), global_step)
        global_step += 1

    avg_loss = total_loss / len(loader)
    avg_emotion_loss = total_emotion_loss / len(loader)
    avg_arousal_loss = total_arousal_loss / len(loader)
    avg_valence_loss = total_valence_loss / len(loader)
    accuracy = accuracy_score(all_labels, all_preds)
    f1_macro = f1_score(all_labels, all_preds, average='macro')
    f1_weighted = f1_score(all_labels, all_preds, average='weighted')

    writer.add_scalar('Train/Loss', avg_loss, epoch)
    writer.add_scalar('Train/Emotion_Loss', avg_emotion_loss, epoch)
    writer.add_scalar('Train/Arousal_Loss', avg_arousal_loss, epoch)
    writer.add_scalar('Train/Valence_Loss', avg_valence_loss, epoch)
    writer.add_scalar('Train/Accuracy', accuracy, epoch)
    writer.add_scalar('Train/F1_Macro', f1_macro, epoch)
    writer.add_scalar('Train/F1_Weighted', f1_weighted, epoch)

    return {
        'loss': avg_loss,
        'emotion_loss': avg_emotion_loss,
        'arousal_loss': avg_arousal_loss,
        'valence_loss': avg_valence_loss,
        'accuracy': accuracy,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted,
    }, global_step


def evaluate(model, loader, criterion, device, writer, epoch, phase='Val'):
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

    with torch.no_grad():
        for batch in loader:
            wavlm = batch['wavlm_emb'].to(device)
            wpro = batch['wpro_emb'].to(device)
            rob = batch['roberta_emb'].to(device)
            labels = batch['emotion_label'].to(device)
            arousal = batch['arousal_value'].to(device)
            valence = batch['valence_value'].to(device)

            outputs = model(wavlm, wpro, rob)
            loss_dict = criterion(outputs, {
                'emotion_label': labels,
                'arousal_value': arousal,
                'valence_value': valence,
            })

            total_loss += loss_dict['total_loss'].item()
            total_emotion_loss += loss_dict['emotion_loss'].item()
            total_arousal_loss += loss_dict['arousal_loss'].item()
            total_valence_loss += loss_dict['valence_loss'].item()

            preds = torch.argmax(outputs['emotion_logits'], dim=1)
            all_preds.extend(preds.cpu().numpy().tolist())
            all_labels.extend(labels.cpu().numpy().tolist())
            all_arousal_pred.extend(outputs['arousal_value'].cpu().numpy().flatten().tolist())
            all_arousal_true.extend(arousal.cpu().numpy().flatten().tolist())
            all_valence_pred.extend(outputs['valence_value'].cpu().numpy().flatten().tolist())
            all_valence_true.extend(valence.cpu().numpy().flatten().tolist())

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

    writer.add_scalar(f'{phase}/Loss', avg_loss, epoch)
    writer.add_scalar(f'{phase}/Emotion_Loss', avg_emotion_loss, epoch)
    writer.add_scalar(f'{phase}/Arousal_Loss', avg_arousal_loss, epoch)
    writer.add_scalar(f'{phase}/Valence_Loss', avg_valence_loss, epoch)
    writer.add_scalar(f'{phase}/Accuracy', accuracy, epoch)
    writer.add_scalar(f'{phase}/F1_Macro', f1_macro, epoch)
    writer.add_scalar(f'{phase}/F1_Weighted', f1_weighted, epoch)
    writer.add_scalar(f'{phase}/Arousal_CCC', arousal_ccc, epoch)
    writer.add_scalar(f'{phase}/Valence_CCC', valence_ccc, epoch)
    writer.add_scalar(f'{phase}/Avg_CCC', avg_ccc, epoch)

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
        'avg_ccc': avg_ccc,
    }


def train_fold(config, embedding_dir, fold_idx, device):
    print(f"\n{'='*80}")
    print(f"🏁 Training Fold {fold_idx} (Test Session {fold_idx})")
    print(f"{'='*80}")

    train_loader, val_loader, test_loader = get_iemocap_emb_dataloaders_for_fold(
        embedding_dir=embedding_dir,
        target_session=fold_idx,
        batch_size=config['data']['batch_size'],
        num_workers=config['data']['num_workers'],
        val_size=config['data'].get('val_size', 0.2),
        random_state=config['data'].get('random_state', 42),
    )

    model = Method2EmbeddingFusionAV(
        num_emotions=config['model']['num_emotions'],
        wavlm_dim=config['model']['wavlm_dim'],
        wpro_dim=config['model']['wpro_dim'],
        roberta_dim=config['model']['roberta_dim'],
        hidden_dim=config['model']['hidden_dim'],
        dropout=config['model']['dropout'],
        av_hidden=config['model']['av_hidden'],
    ).to(device)

    total_params, trainable_params = count_parameters(model)
    print(f"\n{'='*80}")
    print(f"📊 MODEL STATISTICS")
    print(f"{'='*80}")
    print(f"Total parameters:     {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Frozen parameters:    {total_params - trainable_params:,}")

    print(f"\n{'='*80}")
    print(f"📉 LOSS FUNCTION")
    print(f"{'='*80}")
    print(f"Type: CombinedEmotionLossCCC")
    print(f"  L_total = {config['loss']['emotion_weight']} * L_emotion + {config['loss']['arousal_weight']} * L_arousal_CCC + {config['loss']['valence_weight']} * L_valence_CCC")

    criterion = CombinedEmotionLossCCC(
        emotion_weight=config['loss']['emotion_weight'],
        arousal_weight=config['loss']['arousal_weight'],
        valence_weight=config['loss']['valence_weight'],
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay'],
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',
        factor=config['scheduler'].get('factor', 0.5),
        patience=config['scheduler'].get('patience', 3),
        min_lr=config['scheduler'].get('min_lr', 1e-6),
    )

    checkpoint_dir = os.path.join(config['paths']['checkpoint_dir'], f'fold_{fold_idx}')
    os.makedirs(checkpoint_dir, exist_ok=True)
    writer = SummaryWriter(os.path.join(config['paths']['log_dir'], f'fold_{fold_idx}'))

    best_val_acc = 0.0
    best_epoch = 0
    global_step = 0

    for epoch in range(1, config['training']['epochs'] + 1):
        print(f"\n{'='*60}")
        print(f"Fold {fold_idx} - Epoch {epoch}/{config['training']['epochs']}")
        print(f"{'='*60}")

        train_metrics, global_step = train_one_epoch(model, train_loader, criterion, optimizer, device, writer, global_step, epoch)
        val_metrics = evaluate(model, val_loader, criterion, device, writer, epoch, phase='Val')

        scheduler.step(val_metrics['accuracy'])
        current_lr = optimizer.param_groups[0]['lr']
        writer.add_scalar('Train/Learning_Rate', current_lr, epoch)

        print(f"\n[Train] Loss: {train_metrics['loss']:.4f} | Acc: {train_metrics['accuracy']:.4f} | F1: {train_metrics['f1_macro']:.4f}")
        print(f"[Val]   Loss: {val_metrics['loss']:.4f} | Acc: {val_metrics['accuracy']:.4f} | F1: {val_metrics['f1_macro']:.4f} | CCC: {val_metrics['avg_ccc']:.4f}")

        if val_metrics['accuracy'] > best_val_acc:
            best_val_acc = val_metrics['accuracy']
            best_epoch = epoch
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_accuracy': val_metrics['accuracy'],
                'val_f1_macro': val_metrics['f1_macro'],
                'val_avg_ccc': val_metrics['avg_ccc'],
            }, os.path.join(checkpoint_dir, 'best_model.pt'))
            print(f"🔥 New best validation accuracy: {best_val_acc:.4f} (saved)")

        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_accuracy': val_metrics['accuracy'],
            'val_f1_macro': val_metrics['f1_macro'],
            'val_avg_ccc': val_metrics['avg_ccc'],
        }, os.path.join(checkpoint_dir, 'last_model.pt'))

    best_path = os.path.join(checkpoint_dir, 'best_model.pt')
    checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])

    test_metrics = evaluate(model, test_loader, criterion, device, writer, 0, phase='Test')
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
    parser = argparse.ArgumentParser(description='IEMOCAP Method2 W-PRO (embeddings) AV training (5-fold).')
    parser.add_argument('--config', type=str, default='configs/iemocap_wpro_emb_av_config.yaml')
    parser.add_argument('--embedding_dir', type=str, default='datasets/IEMOCAP_KFOLD/embeddings')
    parser.add_argument('--output_dir', type=str, default='checkpoints/iemocap_wpro_emb_av')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--start_fold', type=int, default=1, help='Fold to start from (1-5).')
    args = parser.parse_args()

    config = load_config(args.config)
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(config['paths']['log_dir'], exist_ok=True)
    config['paths']['checkpoint_dir'] = args.output_dir

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*80}")
    print(f"🚀 STARTING 5-FOLD TRAINING (EMBEDDINGS, AV)")
    print(f"{'='*80}")
    print(f"Device: {device}")
    print(f"Config: {args.config}")
    print(f"Embeddings: {args.embedding_dir}")
    print(f"Output: {args.output_dir}")
    print(f"Start fold: {args.start_fold}")
    print(f"{'='*80}\n")

    all_results = []
    for fold_idx in range(args.start_fold, 6):
        fold_results = train_fold(config, args.embedding_dir, fold_idx, device)
        fold_results['fold'] = fold_idx
        all_results.append(fold_results)

    results_path = os.path.join(args.output_dir, 'kfold_results.csv')
    import pandas as pd
    df = pd.DataFrame(all_results)
    df.to_csv(results_path, index=False)

    print(f"\n{'='*80}")
    print(f"📊 5-FOLD RESULTS")
    print(f"{'='*80}")
    print(df.to_string(index=False))
    print(f"\nResults saved to: {results_path}")


if __name__ == '__main__':
    main()
