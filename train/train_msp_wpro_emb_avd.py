"""
Training MSP-Podcast using precomputed embeddings (wavlm, wpro, roberta).
Model: lightweight fusion -> emotion + A/V/D with CCC losses.
"""
import os
import argparse
import yaml
import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score

from data.msp_podcast_embeddings_npz_dataset import get_msp_podcast_emb_dataloaders
from models.method2_emb_fusion_avd import Method2EmbeddingFusionAVD
from train.losses import CombinedEmotionLossAVDCCC


def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def compute_ccc(y_true, y_pred):
    mean_true = np.mean(y_true)
    mean_pred = np.mean(y_pred)
    var_true = np.var(y_true)
    var_pred = np.var(y_pred)
    covariance = np.mean((y_true - mean_true) * (y_pred - mean_pred))
    return (2 * covariance) / (var_true + var_pred + (mean_true - mean_pred) ** 2 + 1e-8)


def train_one_epoch(model, loader, criterion, optimizer, device, epoch, writer=None, global_step=0):
    model.train()
    total_loss = total_em = total_ar = total_va = total_do = 0.0
    all_preds, all_labels = [], []

    pbar = tqdm(loader, desc=f"Epoch {epoch} [Train]", leave=False)
    for batch in pbar:
        wavlm = batch['wavlm_emb'].to(device)
        wpro = batch['wpro_emb'].to(device)
        roberta = batch['roberta_emb'].to(device)
        labels = batch['emotion_label'].to(device)
        aro = batch['arousal_value'].to(device)
        val = batch['valence_value'].to(device)
        dom = batch['dominance_value'].to(device)

        optimizer.zero_grad()
        outputs = model(wavlm, wpro, roberta)
        losses = criterion(outputs, {
            'emotion_label': labels,
            'arousal_value': aro,
            'valence_value': val,
            'dominance_value': dom,
        })
        losses['total_loss'].backward()
        optimizer.step()

        total_loss += losses['total_loss'].item()
        total_em += losses['emotion_loss'].item()
        total_ar += losses['arousal_loss'].item()
        total_va += losses['valence_loss'].item()
        total_do += losses['dominance_loss'].item()

        preds = torch.argmax(outputs['emotion_logits'], dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

        if writer:
            writer.add_scalar('Train/Total_Loss', losses['total_loss'].item(), global_step)
        global_step += 1

    n = len(loader)
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='macro')

    return {
        'loss': total_loss / n,
        'emotion_loss': total_em / n,
        'arousal_loss': total_ar / n,
        'valence_loss': total_va / n,
        'dominance_loss': total_do / n,
        'accuracy': acc,
        'f1_macro': f1,
    }, global_step


def evaluate(model, loader, criterion, device, epoch, phase='Val', writer=None):
    model.eval()
    total_loss = total_em = total_ar = total_va = total_do = 0.0
    all_preds, all_labels = [], []
    all_ar_pred, all_va_pred = [], []
    all_ar_true, all_va_true = [], []

    pbar = tqdm(loader, desc=f"Epoch {epoch} [{phase}]", leave=False)
    with torch.no_grad():
        for batch in pbar:
            wavlm = batch['wavlm_emb'].to(device)
            wpro = batch['wpro_emb'].to(device)
            roberta = batch['roberta_emb'].to(device)
            labels = batch['emotion_label'].to(device)
            aro = batch['arousal_value'].to(device)
            val = batch['valence_value'].to(device)
            dom = batch['dominance_value'].to(device)

            outputs = model(wavlm, wpro, roberta)
            losses = criterion(outputs, {
                'emotion_label': labels,
                'arousal_value': aro,
                'valence_value': val,
                'dominance_value': dom,
            })

            total_loss += losses['total_loss'].item()
            total_em += losses['emotion_loss'].item()
            total_ar += losses['arousal_loss'].item()
            total_va += losses['valence_loss'].item()
            total_do += losses['dominance_loss'].item()

            preds = torch.argmax(outputs['emotion_logits'], dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_ar_pred.extend(outputs['arousal_value'].cpu().numpy().flatten())
            all_va_pred.extend(outputs['valence_value'].cpu().numpy().flatten())
            all_ar_true.extend(aro.cpu().numpy().flatten())
            all_va_true.extend(val.cpu().numpy().flatten())

    n = len(loader)
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='macro')
    all_ar_pred = np.array(all_ar_pred)
    all_va_pred = np.array(all_va_pred)
    ar_true = np.array(all_ar_true)
    va_true = np.array(all_va_true)
    ar_ccc = compute_ccc(ar_true, all_ar_pred)
    va_ccc = compute_ccc(va_true, all_va_pred)

    metrics = {
        'loss': total_loss / n,
        'emotion_loss': total_em / n,
        'arousal_loss': total_ar / n,
        'valence_loss': total_va / n,
        'dominance_loss': total_do / n,
        'accuracy': acc,
        'f1_macro': f1,
        'arousal_ccc': ar_ccc,
        'valence_ccc': va_ccc,
    }

    if writer:
        prefix = f"{phase}"
        for k, v in metrics.items():
            writer.add_scalar(f"{prefix}/{k}", v, epoch)

    return metrics


def save_checkpoint(path, epoch, model, optimizer, val_acc, val_f1, ar_ccc, va_ccc):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_accuracy': val_acc,
        'val_f1': val_f1,
        'arousal_ccc': ar_ccc,
        'valence_ccc': va_ccc,
    }, path)


def main():
    parser = argparse.ArgumentParser(description='MSP-Podcast (precomputed embeddings) AVD training')
    parser.add_argument('--config', type=str, default='configs/msp_wpro_emb_avd_config.yaml')
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device(cfg['training'].get('device', 'cuda:0') if torch.cuda.is_available() else 'cpu')

    # Data
    train_loader, dev_loader, test1_loader, test2_loader = get_msp_podcast_emb_dataloaders(
        embedding_dir=cfg['data']['embedding_dir'],
        batch_size=cfg['data']['batch_size'],
        num_workers=cfg['data']['num_workers'],
        four_class_only=cfg['data']['four_class_only'],
        normalize_av=cfg['data']['normalize_av'],
    )

    # Model
    model = Method2EmbeddingFusionAVD(
        num_emotions=cfg['model']['num_emotions'],
        wavlm_dim=cfg['model']['wavlm_dim'],
        wpro_dim=cfg['model']['wpro_dim'],
        roberta_dim=cfg['model']['roberta_dim'],
        hidden_dim=cfg['model']['hidden_dim'],
        dropout=cfg['model']['dropout'],
        av_hidden=cfg['model']['av_hidden'],
    ).to(device)

    # Loss
    criterion = CombinedEmotionLossAVDCCC(
        emotion_weight=cfg['loss']['emotion_weight'],
        arousal_weight=cfg['loss']['arousal_weight'],
        valence_weight=cfg['loss']['valence_weight'],
        dominance_weight=cfg['loss']['dominance_weight'],
    )

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg['training']['learning_rate'],
        weight_decay=cfg['training']['weight_decay'],
        betas=cfg['optimizer']['betas'],
        eps=cfg['optimizer']['eps'],
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode=cfg['scheduler']['mode'],
        factor=cfg['scheduler']['factor'],
        patience=cfg['scheduler']['patience'],
        min_lr=cfg['scheduler']['min_lr'],
    )

    # Logging
    writer = SummaryWriter(cfg['paths']['log_dir'])
    best_acc = 0.0
    global_step = 0

    for epoch in range(1, cfg['training']['epochs'] + 1):
        train_metrics, global_step = train_one_epoch(model, train_loader, criterion, optimizer, device, epoch, writer, global_step)
        dev_metrics = evaluate(model, dev_loader, criterion, device, epoch, phase='Val', writer=writer)

        scheduler.step(dev_metrics['accuracy'])

        print(f"Epoch {epoch}: Train acc {train_metrics['accuracy']:.4f} | Dev acc {dev_metrics['accuracy']:.4f} | Dev f1 {dev_metrics['f1_macro']:.4f} | Dev CCC (A/V) {dev_metrics['arousal_ccc']:.4f}/{dev_metrics['valence_ccc']:.4f}")

        # Save best
        if dev_metrics['accuracy'] > best_acc:
            best_acc = dev_metrics['accuracy']
            ckpt_path = os.path.join(cfg['paths']['checkpoint_dir'], 'best_model.pt')
            save_checkpoint(ckpt_path, epoch, model, optimizer, dev_metrics['accuracy'], dev_metrics['f1_macro'], dev_metrics['arousal_ccc'], dev_metrics['valence_ccc'])
            print(f"  ✓ Saved best checkpoint at epoch {epoch}")

    # Final eval on test splits with best model
    best_path = os.path.join(cfg['paths']['checkpoint_dir'], 'best_model.pt')
    if os.path.exists(best_path):
        state = torch.load(best_path, map_location=device)
        model.load_state_dict(state['model_state_dict'])
        print("Loaded best model for final evaluation.")

    for name, loader in [('test1', test1_loader), ('test2', test2_loader)]:
        metrics = evaluate(model, loader, criterion, device, epoch=cfg['training']['epochs'], phase=name, writer=writer)
        print(f"Final {name}: Acc {metrics['accuracy']:.4f} | F1 {metrics['f1_macro']:.4f} | CCC (A/V) {metrics['arousal_ccc']:.4f}/{metrics['valence_ccc']:.4f}")

    writer.close()


if __name__ == '__main__':
    main()
