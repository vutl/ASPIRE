# Simple A-V Model Suite Documentation

## Overview

This document describes the simple single-branch A-V emotion recognition models and their training scripts. These models provide lightweight alternatives to Method2 (which uses multiple WavLM instances + RoBERTa).

---

## Models

### 1. **MSP-Podcast Models** (Fixed splits: train/dev/test1/test2)

#### Model Files
- `models/wavlm_base_av.py` - WavLM-Base-Plus backbone
- `models/hubert_av.py` - HuBERT backbone
- `models/wpro_av.py` - W-PRO (WavLM-Prosody) backbone

#### Training Scripts
- `train/train_msp_wavlm_base_av.py` - WavLM-Base trainer
- `train/train_msp_hubert_av.py` - HuBERT trainer
- `train/train_msp_wpro_av.py` - W-PRO trainer

#### Architecture (All Three)
```
Audio Input [B, T]
    ↓
Backbone (frozen) [B, T, hidden_size]
    ↓
SelfAttentionPooling [B, hidden_size]
    ↓
Embedding: hidden_size → 256 [B, 256]
    ↓
A-V Prediction: 256 → 256 [B, 256]
    ├→ Arousal Head: 256 → 1 [B, 1]
    └→ Valence Head: 256 → 1 [B, 1]
    ↓
A-V Refinement: [B, 2] → [B, 256]
    ↓
Final Fusion: [256 + 256] → 256
    ↓
Emotion Classifier: 256 → 4 [B, 4]
```

#### Key Features
- **Frozen Backbone**: All pretrained parameters are frozen
- **Trainable Parameters**: ~15K (only task layers from embedding to classifier)
- **Output**: 
  - `emotion_logits`: [B, 4]
  - `arousal_value`: [B, 1]
  - `valence_value`: [B, 1]

#### Usage (MSP-Podcast)
```bash
# WavLM-Base
python train/train_msp_wavlm_base_av.py \
    --config configs/msp_podcast_config.yaml \
    --output_dir checkpoints/msp_wavlm_base_av \
    --device cuda:0

# HuBERT
python train/train_msp_hubert_av.py \
    --config configs/msp_podcast_config.yaml \
    --output_dir checkpoints/msp_hubert_av \
    --device cuda:0

# W-PRO
python train/train_msp_wpro_av.py \
    --config configs/msp_podcast_config.yaml \
    --output_dir checkpoints/msp_wpro_av \
    --device cuda:0
```

#### Logging (MSP-Podcast)
Same as Method2 MSP training:
- Train metrics: Loss, Accuracy, F1 (macro/weighted), Arousal CCC, Valence CCC, Avg CCC
- Dev metrics: Same as above
- Test1 metrics: Same as above
- Test2 metrics: Same as above
- Log location: `checkpoints/<model>/logs/train_YYYYMMDD_HHMMSS.log`
- TensorBoard: `logs/tensorboard/msp_<backbone>_av/`

---

### 2. **IEMOCAP Kfold Model** (5-Fold CV)

#### Model File
- `models/wpro_av_kfold.py` - W-PRO (only model for kfold)

#### Training Script
- `train/train_wpro_av_kfold.py` - W-PRO kfold trainer

#### Architecture
Same as MSP models (see above)

#### Key Features
- **5-Fold Cross-Validation**: Session-independent splits (1 session = test, 4 sessions = train)
- **Frozen Backbone**: All pretrained parameters are frozen
- **Trainable Parameters**: ~15K
- **Output**: Same as MSP models

#### Usage (IEMOCAP Kfold)
```bash
# Full 5-fold training
python train/train_wpro_av_kfold.py \
    --config configs/method2_wpro_config.yaml \
    --csv datasets/IEMOCAP_KFOLD/all.csv \
    --output_dir checkpoints/wpro_av_kfold \
    --device cuda:0

# Resume from specific fold
python train/train_wpro_av_kfold.py \
    --config configs/method2_wpro_config.yaml \
    --csv datasets/IEMOCAP_KFOLD/all.csv \
    --output_dir checkpoints/wpro_av_kfold \
    --device cuda:0 \
    --resume_fold 3 \
    --resume_checkpoint checkpoints/wpro_av_kfold/fold3/checkpoint_best.pt
```

#### Logging (IEMOCAP Kfold)
Same as standard kfold training:
- Train metrics: Loss, Accuracy, F1 (macro), Arousal CCC, Valence CCC, Avg CCC
- Val metrics: Same as above
- Test metrics: Same as above (per fold)
- Log location: `checkpoints/wpro_av_kfold/train_YYYYMMDD_HHMMSS.log`
- TensorBoard: `logs/tensorboard/wpro_av_kfold_fold{1-5}/`
- Checkpoint structure: `checkpoints/wpro_av_kfold/fold{1-5}/checkpoint_epoch*.pt`

---

## Model Comparison

| Model | Backbone | Params (M) | Trainable | Dataset | Splits |
|-------|----------|-----------|-----------|---------|--------|
| WavLM-Base A-V | Wav2Vec2 768 | ~86 | ~15K | MSP-Podcast | train/dev/test1/test2 |
| HuBERT A-V | HuBERT 768 | ~90 | ~15K | MSP-Podcast | train/dev/test1/test2 |
| W-PRO A-V | WavLM 512 | ~87 | ~15K | MSP-Podcast | train/dev/test1/test2 |
| W-PRO A-V | WavLM 512 | ~87 | ~15K | IEMOCAP | 5-fold CV |

---

## Configuration

All models use the same loss function and optimization strategy:

```yaml
# Loss
loss:
  type: "ccc"  # CCC for continuous A-V values
  emotion_weight: 1.0
  arousal_weight: 1.0
  valence_weight: 1.0

# Training (MSP-Podcast default)
training:
  batch_size: 4  # Reduced from 8 to prevent OOM
  epochs: 10
  learning_rate: 0.0001
  weight_decay: 0.00001
  num_workers: 0  # Use 0 to avoid worker memory bloat

# Optimizer
optimizer:
  type: "adamw"
  betas: [0.9, 0.999]

# Scheduler
scheduler:
  type: "reduce_on_plateau"
  mode: "max"
  factor: 0.5
  patience: 3
  min_lr: 1.0e-06
```

---

## Training Notes

### MSP-Podcast Models
- **Batch Size**: 4 (reduced from 8 to prevent OOM with num_workers > 0)
- **Num Workers**: 0 (use main thread to avoid worker memory duplication)
- **Epochs**: 10
- **Learning Rate**: 0.0001 with AdamW
- **Early Stopping**: Via ReduceLROnPlateau with patience=3

### IEMOCAP Kfold Model
- **Batch Size**: 4 (from config)
- **Num Workers**: Configurable (default from config)
- **Epochs**: 30 (from config)
- **Learning Rate**: 0.0001 with AdamW
- **Early Stopping**: Via ReduceLROnPlateau with patience=5 (from config)

---

## Output Structure

### MSP-Podcast Checkpoints
```
checkpoints/msp_wavlm_base_av/
├── logs/
│   └── train_YYYYMMDD_HHMMSS.log
├── checkpoint_epoch1.pt
├── checkpoint_epoch2.pt
├── ...
└── checkpoint_best.pt  # Best dev CCC
```

### IEMOCAP Kfold Checkpoints
```
checkpoints/wpro_av_kfold/
├── train_YYYYMMDD_HHMMSS.log
├── fold1/
│   ├── checkpoint_epoch1.pt
│   ├── checkpoint_epoch2.pt
│   └── checkpoint_best.pt
├── fold2/
│   └── ...
└── fold5/
    └── ...
```

---

## Performance Metrics

### Logged During Training

**Emotion Recognition** (Classification):
- Accuracy
- F1 Score (macro, weighted)

**Arousal/Valence** (Regression):
- Concordance Correlation Coefficient (CCC)
- Average CCC = (Arousal CCC + Valence CCC) / 2

### Loss Components
- Emotion Loss (CrossEntropy)
- Arousal Loss (MSE)
- Valence Loss (MSE)
- Total Loss (weighted combination)

---

## Backbone Details

### WavLM-Base-Plus
- Model: `microsoft/wavlm-base-plus`
- Hidden Size: 768
- Context: Full audio context
- Frozen: Yes

### HuBERT
- Model: `facebook/hubert-base-multilingual`
- Hidden Size: 768
- Context: Full audio context
- Frozen: Yes

### W-PRO (WavLM-Prosody)
- Model: `Orange/Speaker-wavLM-pro`
- Hidden Size: 512 (smaller than standard WavLM)
- Specialization: Prosody extraction (non-timbral features)
- Training Data: VoxCeleb + VCTK
- Frozen: Yes

---

## Resume Training

### MSP-Podcast
```bash
python train/train_msp_wavlm_base_av.py \
    --config configs/msp_podcast_config.yaml \
    --output_dir checkpoints/msp_wavlm_base_av \
    --device cuda:0 \
    --resume checkpoints/msp_wavlm_base_av/checkpoint_epoch5.pt
```

### IEMOCAP Kfold
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

## Implementation Details

### SelfAttentionPooling
```python
# Temporal aggregation using learned attention weights
attention_weights = softmax(Linear(hidden_size) → Tanh() → Linear(1))
pooled = sum(features * attention_weights)  # [B, T, D] → [B, D]
```

### A-V Refinement
```python
# Refine A-V values using MLP
av_concat = [arousal_value, valence_value]  # [B, 2]
av_refined = MLP(av_concat)  # [B, 2] → [B, 256]
```

### Final Fusion
```python
# Combine embedding and refined A-V for emotion classification
fused = Concat(embedding, av_refined)  # [B, 512]
fused = FusionLayer(fused)  # [B, 512] → [B, 256]
emotion_logits = EmotionClassifier(fused)  # [B, 256] → [B, 4]
```

---

## Troubleshooting

### OOM (Out of Memory)
- Reduce `batch_size` in config
- Set `num_workers: 0` to avoid worker memory duplication

### Slow Training
- Check `num_workers` (should be > 0 if memory allows)
- Verify CUDA is being used (`--device cuda:0`)

### Poor Convergence
- Check learning rate (default 0.0001 is good)
- Verify data loading (check first batch)
- Monitor loss curves in TensorBoard

---

## File Locations Summary

```
models/
├── wavlm_base_av.py           # MSP-Podcast WavLM-Base
├── hubert_av.py               # MSP-Podcast HuBERT
├── wpro_av.py                 # MSP-Podcast W-PRO
└── wpro_av_kfold.py           # IEMOCAP Kfold W-PRO

train/
├── train_msp_wavlm_base_av.py  # MSP-Podcast WavLM-Base trainer
├── train_msp_hubert_av.py      # MSP-Podcast HuBERT trainer
├── train_msp_wpro_av.py        # MSP-Podcast W-PRO trainer
└── train_wpro_av_kfold.py      # IEMOCAP Kfold W-PRO trainer
```

---

## Citation & Attribution

Models use:
- **WavLM-Base-Plus**: Microsoft Speech Language Models
- **HuBERT**: Meta/Facebook Speech Models
- **W-PRO**: Orange - Speaker-wavLM-pro (Gengembre et al., Interspeech 2024)

Loss function:
- **CCC Loss**: Concordance Correlation Coefficient (Lin's CCC)
- **Combined Loss**: Weighted emotion + arousal + valence

---

**Last Updated**: December 7, 2025
