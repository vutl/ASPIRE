# Cấu Hình Training

## 1. METHOD2_WPRO - IEMOCAP K-FOLD CONFIG

**File:** `configs/method2_wpro_config.yaml`

```yaml
# Data
data:
  csv: "datasets/IEMOCAP_KFOLD/tenfolds/all.csv"
  max_audio_length: 160000    # 10 seconds at 16kHz
  max_text_length: 128
  sample_rate: 16000

# Model
model:
  name: "method2_crossattn_av"
  wavlm_model_name: "microsoft/wavlm-base-plus"
  wavlm_prosody_model_name: "Orange/Speaker-wavLM-pro"  # W-PRO
  roberta_model_name: "roberta-base"
  num_emotions: 4
  hidden_dim: 256
  av_refinement_layers: 2
  freeze_wavlm: true           # Freeze all WavLM models
  freeze_roberta: true         # Freeze RoBERTa
  num_cross_attn_heads: 8
  post_attention_layers: 2
  av_prediction_layers: 3

# Training
training:
  batch_size: 8                
  epochs: 30
  learning_rate: 0.0001        # 1e-4
  weight_decay: 0.00001        # 1e-5
  num_workers: 2

# Loss
loss:
  type: "ccc"                  # Concordance Correlation Coefficient
  emotion_weight: 1.0          # Cross-entropy loss weight
  arousal_weight: 1.0          # CCC loss weight
  valence_weight: 1.0          # CCC loss weight

# Optimizer
optimizer:
  type: "adamw"
  betas: [0.9, 0.999]
  eps: 1.0e-08

# Scheduler
scheduler:
  type: "reduce_on_plateau"
  mode: "max"                  # Monitor validation WA (higher is better)
  factor: 0.5                  # LR *= 0.5 when plateau
  patience: 5                  # Wait 5 epochs before reducing
  min_lr: 1.0e-06              # Minimum learning rate

# Paths
paths:
  log_dir: "logs"
  checkpoint_dir: "checkpoints/method2_kfold_wpro"
  tensorboard_dir: "logs/tensorboard"

# Device
device: "cuda:0"

# Misc
seed: 42
```

---

## 2. METHOD2_WPRO - MSP-PODCAST CONFIG

**File:** `configs/msp_podcast_wpro_config.yaml`

```yaml
# Data
data:
  tsv_path: "datasets/MSP-Podcast/msp_podcast.tsv"
  max_audio_length: 160000
  max_text_length: 128
  sample_rate: 16000

# Model
model:
  # Same as IEMOCAP
  name: "method2_crossattn_av"
  wavlm_model_name: "microsoft/wavlm-base-plus"
  wavlm_prosody_model_name: "Orange/Speaker-wavLM-pro"
  roberta_model_name: "roberta-base"
  num_emotions: 4
  hidden_dim: 256
  av_refinement_layers: 2
  freeze_wavlm: true
  freeze_roberta: true
  num_cross_attn_heads: 8
  post_attention_layers: 2
  av_prediction_layers: 3

# Training
training:
  batch_size: 8                # Can increase to 16 if memory allows
  epochs: 10                   # MSP-Podcast is large, fewer epochs
  learning_rate: 0.0001
  weight_decay: 0.00001
  num_workers: 4               # More workers for large dataset

# Loss
loss:
  type: "ccc"
  emotion_weight: 1.0
  arousal_weight: 1.0
  valence_weight: 1.0

# Optimizer
optimizer:
  type: "adamw"
  betas: [0.9, 0.999]
  eps: 1.0e-08

# Scheduler
scheduler:
  type: "reduce_on_plateau"
  mode: "max"
  factor: 0.5
  patience: 3                  # Less patience for large dataset
  min_lr: 1.0e-06

# Paths
paths:
  log_dir: "logs"
  checkpoint_dir: "checkpoints/method2_msp_wpro"
  tensorboard_dir: "logs/tensorboard"

# Device
device: "cuda:0"

# Misc
seed: 42
```

---

## 3. SSL BASELINES CONFIG

**Áp dụng cho:** WavLM, HuBERT, Emotion2Vec, Wav2Vec2

**Config chung cho cả IEMOCAP và MSP-Podcast:**

```yaml
# Model
model:
  ssl_model: "microsoft/wavlm-base-plus"     # Hoặc hubert/emotion2vec/wav2vec2
  freeze_ssl: true
  pooling: "attention"
  num_emotions: 4
  hidden_dim: 256

# Data
data:
  max_audio_length: 160000     # 10 seconds at 16kHz
  sample_rate: 16000

# Training (UNIFIED - Cả IEMOCAP và MSP-Podcast đều dùng config này)
training:
  batch_size: 16               # Cố định cho tất cả SSL baselines
  max_epochs: 50               # Cố định
  learning_rate: 0.0001
  weight_decay: 0.00001
  num_workers: 4
  early_stopping: 10           # Dừng nếu không improve sau 10 epochs

# Loss
loss:
  type: "cross_entropy"        # Chỉ emotion classification

# Optimizer
optimizer:
  type: "adamw"
  betas: [0.9, 0.999]

# Scheduler
scheduler:
  type: "reduce_on_plateau"
  mode: "max"
  factor: 0.5
  patience: 5
  min_lr: 1.0e-06

# Device
device: "cuda:0"
seed: 42
```

**Khác biệt so với Method2_WPRO:**
- **Batch size:** 16 (vs 8) - lớn hơn vì chỉ xử lý audio
- **Max epochs:** 50 (vs 30/10) - nhiều hơn vì single modality
- **Early stopping:** 10 epochs - tránh overfitting
- **Task:** Chỉ emotion classification (không có A-V regression)

---

## 4. SO SÁNH CONFIG

### Method2_WPRO: IEMOCAP vs MSP-Podcast

| Config | IEMOCAP | MSP-Podcast |
|--------|---------|-------------|
| **Batch size** | 8 | 8 |
| **Epochs** | 30 | 10 |
| **Num workers** | 2 | 4 |
| **Scheduler patience** | 5 | 3 |

### SSL Baselines vs Method2_WPRO

| Config | SSL Baselines | Method2_WPRO |
|--------|---------------|--------------|
| **Batch size** | 16 | 8 |
| **Max epochs** | 50 | 30 (IEMOCAP) / 10 (MSP) |
| **Early stopping** | 10 | No |
| **Task** | Emotion only | Emotion + A-V |

---