# Quick Start Guide: Simple A-V Models

## What You Have

✅ **3 MSP-Podcast Models** (all with frozen backbones, only task layers trainable)
- WavLM-Base-Plus
- HuBERT  
- W-PRO

✅ **1 IEMOCAP Kfold Model** (frozen backbone, only task layers trainable)
- W-PRO (5-fold cross-validation)

---

## Quick Commands

### Train WavLM-Base on MSP-Podcast
```bash
python train/train_msp_wavlm_base_av.py \
    --config configs/msp_podcast_config.yaml \
    --output_dir checkpoints/msp_wavlm_base_av \
    --device cuda:0
```

### Train HuBERT on MSP-Podcast
```bash
python train/train_msp_hubert_av.py \
    --config configs/msp_podcast_config.yaml \
    --output_dir checkpoints/msp_hubert_av \
    --device cuda:1
```

### Train W-PRO on MSP-Podcast
```bash
python train/train_msp_wpro_av.py \
    --config configs/msp_podcast_config.yaml \
    --output_dir checkpoints/msp_wpro_av \
    --device cuda:2
```

### Train W-PRO on IEMOCAP (5-fold)
```bash
python train/train_wpro_av_kfold.py \
    --config configs/method2_wpro_config.yaml \
    --csv datasets/IEMOCAP_KFOLD/all.csv \
    --output_dir checkpoints/wpro_av_kfold \
    --device cuda:0
```

---

## Model Specifications

All 3 MSP models have identical architecture:
- **Backbone**: Frozen (not trainable)
- **Trainable layers**: ~15K parameters (embedding → classifier only)
- **Output**: 
  - emotion_logits [B, 4]
  - arousal_value [B, 1]
  - valence_value [B, 1]

W-PRO Kfold model:
- Same architecture as MSP W-PRO
- Uses 5-fold cross-validation on IEMOCAP

---

## Key Configuration (MSP-Podcast)

```yaml
batch_size: 4          # OOM prevention
epochs: 10
learning_rate: 0.0001
num_workers: 0         # Use main thread, not workers
optimizer: adamw
scheduler: reduce_on_plateau (patience=3)
```

Same config used for all 3 MSP models (just swap model class).

---

## Logging

### MSP-Podcast Logs
- Location: `checkpoints/msp_<model>/logs/train_*.log`
- Metrics per epoch:
  - Train: Loss, Accuracy, F1 (macro), Arousal CCC, Valence CCC, Avg CCC
  - Dev: Same metrics
  - Test1: Same metrics
  - Test2: Same metrics
- TensorBoard: `logs/tensorboard/msp_<model>_av/`

### IEMOCAP Kfold Logs
- Location: `checkpoints/wpro_av_kfold/train_*.log`
- Per fold: Fold 1-5 trained sequentially
- Metrics per epoch:
  - Train: Loss, Accuracy, F1 (macro), Arousal CCC, Valence CCC, Avg CCC
  - Val: Same metrics
  - Test: Same metrics (session-wise test)
- TensorBoard: `logs/tensorboard/wpro_av_kfold_fold{1-5}/`

---

## Resuming Training

### MSP-Podcast
```bash
python train/train_msp_wavlm_base_av.py \
    --config configs/msp_podcast_config.yaml \
    --output_dir checkpoints/msp_wavlm_base_av \
    --device cuda:0 \
    --resume checkpoints/msp_wavlm_base_av/checkpoint_epoch5.pt
```

### IEMOCAP (Resume from fold 3)
```bash
python train/train_wpro_av_kfold.py \
    --config configs/method2_wpro_config.yaml \
    --csv datasets/IEMOCAP_KFOLD/all.csv \
    --output_dir checkpoints/wpro_av_kfold \
    --device cuda:0 \
    --resume_fold 3 \
    --resume_checkpoint checkpoints/wpro_av_kfold/fold3/checkpoint_epoch10.pt
```

---

## File Summary

### Models
```
models/
├── wavlm_base_av.py       ← MSP: WavLM-Base-Plus
├── hubert_av.py           ← MSP: HuBERT
├── wpro_av.py             ← MSP: W-PRO
└── wpro_av_kfold.py       ← IEMOCAP: W-PRO (kfold)
```

### Trainers
```
train/
├── train_msp_wavlm_base_av.py  ← MSP WavLM-Base trainer
├── train_msp_hubert_av.py      ← MSP HuBERT trainer
├── train_msp_wpro_av.py        ← MSP W-PRO trainer
└── train_wpro_av_kfold.py      ← IEMOCAP W-PRO kfold trainer
```

---

## Architecture (All Models)

```
Audio [B, T]
  ↓
Backbone (frozen) [B, T, 768 or 512]
  ↓
SelfAttentionPooling [B, 768/512]
  ↓
Embedding [B, 256]
  ↓
A-V Prediction [B, 256]
  ├→ Arousal Head [B, 1]
  └→ Valence Head [B, 1]
  ↓
A-V Refinement [B, 256]
  ↓
Final Fusion [B, 256]
  ↓
Emotion Classifier [B, 4]
```

---

## Performance Metrics

Logged for each split (Train/Dev/Val/Test):
- **Emotion**: Accuracy, F1 (macro, weighted)
- **Arousal**: CCC (Concordance Correlation Coefficient)
- **Valence**: CCC
- **Combined**: Avg CCC = (Arousal CCC + Valence CCC) / 2

Loss components:
- Emotion: CrossEntropyLoss
- Arousal: MSELoss
- Valence: MSELoss
- Total: emotion_weight × emotion + arousal_weight × arousal + valence_weight × valence

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| OOM Error | Set `batch_size: 2` in config, ensure `num_workers: 0` |
| Slow Training | Use `num_workers: 4` if memory allows (batch_size must be 2) |
| Poor Loss | Check data loading first batch, verify audio preprocessing |
| Missing Module | Ensure `configs/msp_podcast_config.yaml` exists and is valid |

---

## Next Steps

1. **Train MSP models in parallel**:
   ```bash
   nohup python train/train_msp_wavlm_base_av.py ... > log1.txt 2>&1 &
   nohup python train/train_msp_hubert_av.py ... > log2.txt 2>&1 &
   nohup python train/train_msp_wpro_av.py ... > log3.txt 2>&1 &
   ```

2. **Monitor progress**:
   ```bash
   tail -f checkpoints/msp_wavlm_base_av/logs/train_*.log
   ```

3. **After training**, compare results:
   - Check best dev CCC for each model
   - Compare test set performance
   - Analyze TensorBoard curves

---

**All models use identical loss functions and optimization strategies.**
**Only difference: Backbone (WavLM-Base, HuBERT, or W-PRO)**

See `papers_and_notes/SIMPLE_AV_MODELS.md` for detailed documentation.
