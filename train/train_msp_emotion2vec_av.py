"""
Training script for MSP-Podcast using precomputed emotion2vec embeddings.
"""
import os
import sys
import yaml
import torch
import torch.nn as nn
import numpy as np
import logging
import argparse
from datetime import datetime
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.tensorboard import SummaryWriter
from thop import profile, clever_format

sys.path.append(str(Path(__file__).resolve().parent.parent))

from models.emotion2vec_av import Emotion2VecAV
from train.losses import CombinedEmotionLossCCC
from data.msp_podcast_embedding_dataset import get_msp_podcast_embedding_dataloaders


def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def compute_ccc(y_true, y_pred):
    mean_true = np.mean(y_true)
    mean_pred = np.mean(y_pred)
    var_true = np.var(y_true)
    var_pred = np.var(y_pred)
    covariance = np.mean((y_true - mean_true) * (y_pred - mean_pred))
    return (2 * covariance) / (var_true + var_pred + (mean_true - mean_pred) ** 2 + 1e-8)


def train_one_epoch(model, loader, criterion, optimizer, device, writer, global_step, epoch, logger):
    model.train()
    total_loss = 0.0
    emotion_losses, arousal_losses, valence_losses = [], [], []
    all_emotion_preds, all_emotion_labels = [], []
    all_arousal_preds, all_arousal_targets = [], []
    all_valence_preds, all_valence_targets = [], []

    pbar = tqdm(loader, desc=f"Epoch {epoch} [Train]")
    for batch in pbar:
        embedding = batch['embedding'].to(device)
        emotion_labels = batch['emotion_label'].to(device)
        arousal_targets = batch['arousal_value'].to(device)
        valence_targets = batch['valence_value'].to(device)

        outputs = model(embedding)
        targets = {
            'emotion_label': emotion_labels,
            'arousal_value': arousal_targets,
            'valence_value': valence_targets,
        }
        loss_dict = criterion(outputs, targets)
        loss = loss_dict['total_loss']

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        emotion_losses.append(loss_dict['emotion_loss'].item())
        arousal_losses.append(loss_dict['arousal_loss'].item())
        valence_losses.append(loss_dict['valence_loss'].item())

        emotion_pred = torch.argmax(outputs['emotion_logits'], dim=1).cpu().numpy()
        emotion_label = emotion_labels.cpu().numpy()
        arousal_pred = outputs['arousal_value'].squeeze(-1).detach().cpu().numpy()
        arousal_target = arousal_targets.squeeze(-1).cpu().numpy()
        valence_pred = outputs['valence_value'].squeeze(-1).detach().cpu().numpy()
        valence_target = valence_targets.squeeze(-1).cpu().numpy()

        all_emotion_preds.extend(emotion_pred)
        all_emotion_labels.extend(emotion_label)
        all_arousal_preds.extend(arousal_pred)
        all_arousal_targets.extend(arousal_target)
        all_valence_preds.extend(valence_pred)
        all_valence_targets.extend(valence_target)

        pbar.set_postfix({'loss': loss.item()})
        if writer:
            writer.add_scalar('Train/Loss', loss.item(), global_step)
            writer.add_scalar('Train/Emotion Loss', loss_dict['emotion_loss'].item(), global_step)
            writer.add_scalar('Train/Arousal Loss', loss_dict['arousal_loss'].item(), global_step)
            writer.add_scalar('Train/Valence Loss', loss_dict['valence_loss'].item(), global_step)
            global_step += 1

    avg_loss = total_loss / len(loader)
    accuracy = accuracy_score(all_emotion_labels, all_emotion_preds)
    f1_macro = f1_score(all_emotion_labels, all_emotion_preds, average='macro')
    f1_weighted = f1_score(all_emotion_labels, all_emotion_preds, average='weighted')
    arousal_ccc = compute_ccc(np.array(all_arousal_targets), np.array(all_arousal_preds))
    valence_ccc = compute_ccc(np.array(all_valence_targets), np.array(all_valence_preds))
    avg_ccc = (arousal_ccc + valence_ccc) / 2

    logger.info(
        f"Epoch {epoch} Train - Loss: {avg_loss:.4f}, Acc: {accuracy:.4f}, "
        f"F1 Macro: {f1_macro:.4f}, Arousal CCC: {arousal_ccc:.4f}, "
        f"Valence CCC: {valence_ccc:.4f}, Avg CCC: {avg_ccc:.4f}"
    )

    if writer:
        writer.add_scalar('Train/Epoch Loss', avg_loss, epoch)
        writer.add_scalar('Train/Epoch Accuracy', accuracy, epoch)
        writer.add_scalar('Train/Epoch F1 Macro', f1_macro, epoch)
        writer.add_scalar('Train/Epoch Arousal CCC', arousal_ccc, epoch)
        writer.add_scalar('Train/Epoch Valence CCC', valence_ccc, epoch)
        writer.add_scalar('Train/Epoch Avg CCC', avg_ccc, epoch)

    return global_step


def evaluate(model, loader, criterion, device, writer, epoch, phase, logger):
    model.eval()
    total_loss = 0.0
    emotion_losses, arousal_losses, valence_losses = [], [], []
    all_emotion_preds, all_emotion_labels = [], []
    all_arousal_preds, all_arousal_targets = [], []
    all_valence_preds, all_valence_targets = [], []

    with torch.no_grad():
        pbar = tqdm(loader, desc=f"Epoch {epoch} [{phase}]")
        for batch in pbar:
            embedding = batch['embedding'].to(device)
            emotion_labels = batch['emotion_label'].to(device)
            arousal_targets = batch['arousal_value'].to(device)
            valence_targets = batch['valence_value'].to(device)

            outputs = model(embedding)
            targets = {
                'emotion_label': emotion_labels,
                'arousal_value': arousal_targets,
                'valence_value': valence_targets,
            }
            loss_dict = criterion(outputs, targets)
            loss = loss_dict['total_loss']

            total_loss += loss.item()
            emotion_losses.append(loss_dict['emotion_loss'].item())
            arousal_losses.append(loss_dict['arousal_loss'].item())
            valence_losses.append(loss_dict['valence_loss'].item())

            emotion_pred = torch.argmax(outputs['emotion_logits'], dim=1).cpu().numpy()
            emotion_label = emotion_labels.cpu().numpy()
            arousal_pred = outputs['arousal_value'].squeeze(-1).detach().cpu().numpy()
            arousal_true = arousal_targets.squeeze(-1).cpu().numpy()
            valence_pred = outputs['valence_value'].squeeze(-1).detach().cpu().numpy()
            valence_true = valence_targets.squeeze(-1).cpu().numpy()

            all_emotion_preds.extend(emotion_pred)
            all_emotion_labels.extend(emotion_label)
            all_arousal_preds.extend(arousal_pred)
            all_arousal_targets.extend(arousal_true)
            all_valence_preds.extend(valence_pred)
            all_valence_targets.extend(valence_true)

    avg_loss = total_loss / len(loader)
    accuracy = accuracy_score(all_emotion_labels, all_emotion_preds)
    f1_macro = f1_score(all_emotion_labels, all_emotion_preds, average='macro')
    f1_weighted = f1_score(all_emotion_labels, all_emotion_preds, average='weighted')
    arousal_ccc = compute_ccc(np.array(all_arousal_targets), np.array(all_arousal_preds))
    valence_ccc = compute_ccc(np.array(all_valence_targets), np.array(all_valence_preds))
    avg_ccc = (arousal_ccc + valence_ccc) / 2

    logger.info(
        f"Epoch {epoch} {phase} - Loss: {avg_loss:.4f}, Acc: {accuracy:.4f}, "
        f"F1 Macro: {f1_macro:.4f}, Arousal CCC: {arousal_ccc:.4f}, "
        f"Valence CCC: {valence_ccc:.4f}, Avg CCC: {avg_ccc:.4f}"
    )

    if writer:
        writer.add_scalar(f'{phase}/Loss', avg_loss, epoch)
        writer.add_scalar(f'{phase}/Accuracy', accuracy, epoch)
        writer.add_scalar(f'{phase}/F1 Macro', f1_macro, epoch)
        writer.add_scalar(f'{phase}/Arousal CCC', arousal_ccc, epoch)
        writer.add_scalar(f'{phase}/Valence CCC', valence_ccc, epoch)
        writer.add_scalar(f'{phase}/Avg CCC', avg_ccc, epoch)

    return {
        'loss': avg_loss,
        'accuracy': accuracy,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted,
        'arousal_ccc': arousal_ccc,
        'valence_ccc': valence_ccc,
        'avg_ccc': avg_ccc,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/msp_emotion2vec_av_config.yaml')
    parser.add_argument('--device', type=str, default='cuda:0')
    args = parser.parse_args()

    config = load_config(args.config)

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    output_dir = Path(config['paths']['checkpoint_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)

    log_file = output_dir / f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
    )
    logger = logging.getLogger(__name__)

    logger.info("=" * 70)
    logger.info("Emotion2Vec Embedding A-V Training on MSP-Podcast")
    logger.info("=" * 70)
    logger.info(f"Device: {device}")
    logger.info(f"Config: {args.config}")
    logger.info(f"TSV: {config['data']['tsv_path']}")
    logger.info(f"Embedding dir: {config['data']['embedding_dir']}")

    model = Emotion2VecAV(
        num_emotions=config['model']['num_emotions'],
        input_dim=config['model']['input_dim'],
        hidden_dim=config['model']['hidden_dim'],
        av_refinement_layers=config['model']['av_refinement_layers'],
    ).to(device)

    total_params, trainable_params = count_parameters(model)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    logger.info(f"Frozen parameters: {total_params - trainable_params:,}")

    try:
        dummy = torch.randn(1, config['model']['input_dim'], device=device)
        macs, params = profile(model, inputs=(dummy,), verbose=False)
        flops = macs * 2
        flops_str = clever_format([flops], "%.3f")[0]
        logger.info(f"FLOPs: {flops_str}")
    except Exception:
        logger.info("FLOPs: N/A")

    train_loader, dev_loader, test1_loader, test2_loader = get_msp_podcast_embedding_dataloaders(
        tsv_path=config['data']['tsv_path'],
        embedding_dir=config['data']['embedding_dir'],
        batch_size=config['training']['batch_size'],
        num_workers=config['training']['num_workers'],
        four_class_only=config['data'].get('four_class_only', True),
    )

    criterion = CombinedEmotionLossCCC(
        emotion_weight=config['loss']['emotion_weight'],
        arousal_weight=config['loss']['arousal_weight'],
        valence_weight=config['loss']['valence_weight'],
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay'],
        betas=config['optimizer']['betas'],
        eps=config['optimizer']['eps'],
    )

    scheduler = None
    if config['scheduler']['type'] == 'reduce_on_plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=config['scheduler']['mode'],
            factor=config['scheduler']['factor'],
            patience=config['scheduler']['patience'],
        )

    tb_dir = Path(config['paths']['tensorboard_dir']) / 'msp_emotion2vec_av'
    tb_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(tb_dir))

    best_ccc = -1.0
    best_state = None
    global_step = 0

    for epoch in range(1, config['training']['epochs'] + 1):
        global_step = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            writer,
            global_step,
            epoch,
            logger,
        )

        dev_metrics = evaluate(model, dev_loader, criterion, device, writer, epoch, 'Dev', logger)
        test1_metrics = evaluate(model, test1_loader, criterion, device, writer, epoch, 'Test1', logger)
        test2_metrics = evaluate(model, test2_loader, criterion, device, writer, epoch, 'Test2', logger)

        if scheduler:
            scheduler.step(dev_metrics['avg_ccc'])

        avg_ccc = dev_metrics['avg_ccc']
        if avg_ccc > best_ccc:
            best_ccc = avg_ccc
            best_state = {
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'best_ccc': best_ccc,
                'global_step': global_step,
            }
            ckpt_path = output_dir / 'checkpoint_best.pt'
            torch.save(best_state, ckpt_path)
            logger.info(f"Saved best checkpoint to {ckpt_path} (Dev Avg CCC: {best_ccc:.4f})")

    writer.close()
    logger.info("Training complete")


if __name__ == "__main__":
    main()
