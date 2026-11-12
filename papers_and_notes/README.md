# Speech Emotion Recognition with A-V Modeling

Multi-task emotion recognition system combining **emotion classification** and **arousal-valence (A-V) regression** using multimodal deep learning.

**Recommended architecture**: Method 2 (Cross-Attention A-V) ⭐

---

## 📋 Overview

This project implements two multimodal architectures for speech emotion recognition:

| Method | Description | Parameters | Status |
|--------|-------------|------------|--------|
| **Method 1** | Triple WavLM with separate A-V branches | 204M | Working |
| **Method 2** ⭐ | Cross-Attention A-V (audio + text fusion) | 319M | **Recommended** |

### Why Method 2?

- ✅ **Multimodal fusion**: Combines acoustic (WavLM) and semantic (RoBERTa) features via cross-attention
- ✅ **Pretrained backbones frozen**: Only ~6.3M task-specific parameters trained (98% of model frozen)
- ✅ **Multi-task learning**: Joint emotion classification + A-V regression with CCC loss
- ✅ **Better generalization**: Cross-attention allows prosody and text to inform each other

---

## 🎯 Method 2: Cross-Attention A-V Architecture

### High-Level Flow

```
Audio (raw waveform) ──> WavLM_prosody (frozen) ──┐
                                                   ├──> Bidirectional Cross-Attention
Text (transcript) ──────> RoBERTa (frozen) ───────┘
                                                   │
                                                   ├──> Post-Attention Layers
                                                   │
                                                   ├──> Branch Fusion ──> A-V Prediction
                                                   │                      ├─> Arousal
                                                   │                      └─> Valence
                                                   │
Audio (raw waveform) ──> WavLM_final (frozen) ────┤
                                                   │
                                                   └──> Final Fusion (with A-V embeddings)
                                                        └──> Emotion Classifier (4 classes)
```

### Architecture Components

**3 Pretrained Models (2 configurations available):**

#### Standard Configuration (`configs/method2_config.yaml`)
1. **WavLM-base-plus (prosody)** - 94M params - Generic acoustic/prosody features
2. **RoBERTa-base** - 125M params - Semantic/linguistic features from text
3. **WavLM-base-plus (final)** - 94M params - Global acoustic features
**Total**: 319M params (98.03% frozen, only 6.3M trainable)

#### W-PRO Configuration (`configs/method2_wpro_config.yaml`) ⭐ **RECOMMENDED**
1. **W-PRO (Orange/Speaker-wavLM-pro)** - ~300M params - **Specialized non-timbral prosody extractor**
2. **RoBERTa-base** - 125M params - Semantic/linguistic features from text
3. **WavLM-base-plus (final)** - 94M params - Global acoustic features
**Total**: ~519M params (98.8% frozen, only 6.3M trainable)

**Why W-PRO?**
- ✅ **Specialized prosody extraction**: Trained to extract pitch patterns, rhythm, speaking rate (NOT speaker identity)
- ✅ **Better arousal prediction**: Prosody strongly correlates with arousal
- ✅ **Interpretability**: Explicitly models non-timbral features → easier to explain predictions
- ✅ **Paper**: Gengembre et al., Interspeech 2024 - "Disentangling prosody and timbre embeddings"

**Trainable Task-Specific Layers:**
- Prosody & text projection (768→256)
- Bidirectional cross-attention (8 heads)
- Post-attention transformer layers (2 layers each branch)
- Self-attention pooling
- Branch fusion (gated fusion)
- A-V prediction heads (3 MLP layers → arousal/valence)
- A-V refinement module (2 layers)
- Final fusion (1536→256)
- Emotion classifier (256→128→4)

### Forward Pass Details

**Input:**
- Audio: `[B, 160000]` - 10 seconds at 16kHz
- Text tokens: `[B, 128]` - RoBERTa tokenized transcript
- Text mask: `[B, 128]` - Attention mask

**Step-by-Step:**

1. **Prosody Branch**: Audio → WavLM_prosody → `[B, 499, 768]` → Projection → `[B, 499, 256]`
2. **Text Branch**: Text → RoBERTa → `[B, 128, 768]` → Projection → `[B, 128, 256]`
3. **Cross-Attention**: Prosody ↔ Text (bidirectional) → attended features `[B, T, 256]`
4. **Post-Attention**: Each branch → 2 transformer layers + self-attention pooling → `[B, 256]`
5. **Branch Fusion**: Gated fusion of prosody_pooled + text_pooled → `[B, 256]`
6. **A-V Prediction**: Branch fusion → 3 MLP layers → 2 heads → arousal `[B, 1]`, valence `[B, 1]`
7. **A-V Refinement**: `[arousal, valence]` → 2-layer MLP → `[B, 256]` (A-V embeddings)
8. **WavLM Final**: Audio → WavLM_final → self-attention pooling → `[B, 768]`
9. **Final Fusion**: Concat `[WavLM_pooled(768), Arousal_emb(256), Valence_emb(256), AV_refined(256)]` → `[B, 1536]` → Fusion layers → `[B, 256]`
10. **Emotion Classifier**: `[B, 256]` → 2-layer MLP → `[B, 4]` (emotion logits)

**Output:**
- `emotion_logits`: `[B, 4]` - Logits for {ang, hap, neu, sad}
- `arousal_value`: `[B, 1]` - Continuous arousal prediction
- `valence_value`: `[B, 1]` - Continuous valence prediction
- Intermediate: arousal_embedding, valence_embedding, av_refined

---

## 📊 Supported Datasets

### 1. IEMOCAP (5-fold cross-validation)

**Setup:**
- **Fold strategy**: Speaker-independent, 5 sessions → 5 folds
- **Total samples**: ~10,000 (4-class filtered: ang, hap, neu, sad)
- **Split per fold**: 80% train (4 sessions), 20% test (1 session)
- **A-V values**: Continuous arousal/valence in range [1, 5]
- **Transcripts**: Manual transcriptions available

**File structure:**
```
datasets/IEMOCAP_KFOLD/
├── all.csv                    # Full dataset with fold assignments
├── tenfolds/                  # 10-fold split (mapped to 5 sessions)
│   ├── fold_1.csv
│   ├── fold_2.csv
│   ...
│   └── fold_10.csv
├── IEMOCAP_train4classes.csv # Legacy train/test split
└── IEMOCAP_test4classes.csv
```

**Training command:**
```bash
# 5-fold cross-validation
bash run_train_kfold.sh cuda:0

# Or manual:
python train/train_method2_kfold.py \
    --config configs/method2_config.yaml \
    --csv datasets/IEMOCAP_KFOLD/tenfolds/all.csv \
    --output_dir checkpoints/method2_kfold \
    --device cuda:0
```

**Key config (IEMOCAP):**
- Batch size: 4
- Epochs: 30 per fold
- Learning rate: 1e-4
- Loss: CombinedEmotionLossCCC (CrossEntropy + CCC)
- Validation: 80/20 split within training folds
- Checkpoint: Save best model per fold based on validation accuracy

### 2. MSP-Podcast (fixed splits)

**Setup:**
- **Split strategy**: Fixed train/dev/test1/test2 (no cross-validation)
- **Total samples**: 197,091 (4-class filtered: ang, hap, neu, sad)
  - Train: 129,064
  - Dev: 23,885
  - Test1: 33,430
  - Test2: 10,712
- **A-V values**: Continuous arousal/valence in range [1, 7], normalized to z-score
  - Arousal: μ=4.245, σ=0.982
  - Valence: μ=3.823, σ=0.954
- **Transcripts**: Available for all samples

**File structure:**
```
datasets/MSP-Podcast/
├── msp_podcast.tsv            # Main metadata file
├── Audios/                    # Symlink to audio files
├── Transcripts/               # Symlink to transcript files
└── Labels/
    └── labels_consensus.csv   # A-V annotations
```

**Training command:**
```bash
# Standard training (batch_size=4)
bash run_train_msp.sh cuda:0

# Higher batch size on more powerful GPU (batch_size=8, device cuda:2)
bash run_train_msp_bs8.sh cuda:2

# Or manual:
python train/train_method2_msp.py \
    --config configs/msp_podcast_config.yaml \
    --tsv datasets/MSP-Podcast/msp_podcast.tsv \
    --output_dir checkpoints/method2_msp \
    --device cuda:0
```

**Key config (MSP-Podcast):**
- Batch size: 4 or 8 (depending on GPU memory)
- Epochs: 10 (dataset is large)
- Learning rate: 1e-4
- Loss: CombinedEmotionLossCCC (CrossEntropy + CCC)
- Validation: Evaluate on dev set after each epoch
- Checkpoint: Save best model based on dev accuracy
- Final evaluation: Test on both test1 and test2 sets

**Key differences from IEMOCAP:**

| Aspect | IEMOCAP | MSP-Podcast |
|--------|---------|-------------|
| **Size** | ~10k samples | ~197k samples |
| **Split** | 5-fold CV | Fixed train/dev/test1/test2 |
| **Epochs** | 30 per fold | 10 total |
| **A-V range** | [1, 5] | [1, 7] (normalized) |
| **Batch size** | 4 | 4-8 |
| **Training time** | ~2-3h per fold | ~15-55h total (depends on batch size) |

---

## 🚀 Quick Start

### 1. Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Verify installation
python test_models.py
```

### 2. Test Architecture

```bash
# Quick test (no training, just forward pass)
python test_models.py
```

### 3. Train on IEMOCAP (5-fold CV)

**Standard WavLM:**
```bash
# Full 5-fold cross-validation
bash scripts/run_train_kfold.sh cuda:0

# Resume from specific fold
bash scripts/run_train_kfold.sh cuda:0 3 checkpoints/method2_kfold/fold_3/best_model.pt
```

**W-PRO (Recommended):**
```bash
# Full 5-fold cross-validation with W-PRO
bash scripts/run_train_kfold_wpro.sh cuda:0

# Or use GPU 2
bash scripts/run_train_kfold_wpro.sh cuda:2

# Resume from specific fold
bash scripts/run_train_kfold_wpro.sh cuda:0 3 checkpoints/method2_kfold_wpro/fold_3/best_model.pt
```

### 4. Train on MSP-Podcast

```bash
# Standard training
bash run_train_msp.sh cuda:0

# Higher batch size (requires more GPU memory)
bash run_train_msp_bs8.sh cuda:2

# Resume training
python train/train_method2_msp.py \
    --config configs/msp_podcast_config.yaml \
    --tsv datasets/MSP-Podcast/msp_podcast.tsv \
    --output_dir checkpoints/method2_msp_bs8_cuda2 \
    --device cuda:2 \
    --resume checkpoints/method2_msp/best_model.pt
```

### 5. Monitor Training

```bash
# Check log file
tail -f logs/train_method2_kfold_YYYYMMDD_HHMMSS.log

# Or for MSP-Podcast
tail -f logs/train_msp_bs8_YYYYMMDD_HHMMSS.log

# Tensorboard (IEMOCAP)
tensorboard --logdir checkpoints/method2_kfold/logs --port 6006

# Tensorboard (MSP-Podcast)
tensorboard --logdir checkpoints/method2_msp/logs/tensorboard --port 6008
```

---

## 📈 Expected Results

### IEMOCAP 5-fold CV (preliminary)

| Metric | Method 2 |
|--------|----------|
| Accuracy | ~64-70% |
| F1 Weighted | ~0.64-0.70 |
| F1 Macro | ~0.60-0.65 |
| Arousal CCC | ~0.55-0.65 |
| Valence CCC | ~0.50-0.60 |
| Avg CCC | ~0.55-0.62 |

*Note: Results may vary by fold. Session 5 typically performs best due to data characteristics.*

### MSP-Podcast (in progress)

Training currently in progress with batch_size=8 on cuda:2.

**Intermediate results (after 10 epochs, batch_size=4):**
- Dev Accuracy: ~64.2%
- Training ongoing with larger batch size for potential improvement

---

## 🔧 Configuration Files

### IEMOCAP Config (`configs/method2_config.yaml`)

```yaml
model:
  wavlm_model_name: "microsoft/wavlm-base-plus"
  roberta_model_name: "roberta-base"
  num_emotions: 4
  hidden_dim: 256
  freeze_wavlm: true
  freeze_roberta: true
  num_cross_attn_heads: 8
  post_attention_layers: 2
  av_prediction_layers: 3

training:
  batch_size: 4
  epochs: 30
  learning_rate: 0.0001
  weight_decay: 0.00001

loss:
  type: "ccc"
  emotion_weight: 1.0
  arousal_weight: 1.0
  valence_weight: 1.0
```

### MSP-Podcast Config (`configs/msp_podcast_config.yaml` or `msp_podcast_config_bs8.yaml`)

```yaml
model:
  # Same as IEMOCAP

training:
  batch_size: 4  # or 8 for bs8 config
  epochs: 10     # Fewer epochs due to large dataset
  learning_rate: 0.0001
  weight_decay: 0.00001

loss:
  type: "ccc"
  emotion_weight: 1.0
  arousal_weight: 1.0
  valence_weight: 1.0
```

---

## 📁 Project Structure

```
AVD_Test/
├── models/                   # Model architectures
│   ├── base_components.py   # Reusable modules (attention, fusion, etc.)
│   ├── method1_triple_wavlm.py
│   └── method2_crossattn_av.py  ⭐ Main architecture
├── train/                    # Training scripts
│   ├── losses.py            # Multi-task loss functions
│   ├── train_method2_kfold.py   # IEMOCAP 5-fold training
│   └── train_method2_msp.py     # MSP-Podcast training
├── data/                     # Dataset loaders
│   ├── iemocap_dataset.py
│   └── msp_podcast_dataset.py
├── configs/                  # YAML config files
│   ├── method2_config.yaml
│   ├── msp_podcast_config.yaml
│   └── msp_podcast_config_bs8.yaml
├── datasets/                 # Dataset files
│   ├── IEMOCAP/
│   ├── IEMOCAP_KFOLD/
│   ├── MSP-Podcast/
│   └── EMOVOME/
├── checkpoints/              # Saved model checkpoints
│   ├── method2_kfold/       # IEMOCAP fold results
│   ├── method2_msp/         # MSP-Podcast batch_size=4
│   └── method2_msp_bs8_cuda2/  # MSP-Podcast batch_size=8
├── logs/                     # Training logs
├── run_train_kfold.sh       # Quick start script for IEMOCAP
├── run_train_msp.sh         # Quick start script for MSP-Podcast
├── run_train_msp_bs8.sh     # MSP-Podcast with batch_size=8
└── README.md                 # This file
```

---

## 🛠️ Advanced Usage

### Hyperparameter Tuning

**Learning rate:**
```bash
python train/train_method2_msp.py \
    --config configs/msp_podcast_config.yaml \
    ... \
    --learning_rate 5e-5  # Lower LR for fine-tuning
```

**Loss weights (prioritize A-V regression):**
```bash
python train/train_method2_msp.py \
    --config configs/msp_podcast_config.yaml \
    ... \
    --emotion_weight 0.5 \
    --arousal_weight 1.5 \
    --valence_weight 1.5
```

**Batch size vs GPU memory:**
- Batch size 2: ~8GB GPU memory
- Batch size 4: ~12-14GB GPU memory
- Batch size 8: ~20-24GB GPU memory (requires V100 or A100)

### Evaluation Only

```bash
# Evaluate specific checkpoint
python test_kfold_results.py  # For IEMOCAP folds

# For MSP-Podcast (modify train script to skip training)
python train/train_method2_msp.py \
    --config configs/msp_podcast_config.yaml \
    --tsv datasets/MSP-Podcast/msp_podcast.tsv \
    --output_dir checkpoints/method2_msp \
    --device cuda:0 \
    --resume checkpoints/method2_msp/best_model.pt \
    --epochs 0  # Skip training, only evaluate
```

---

## 📚 Additional Documentation

- **[WPRO_INTEGRATION.md](WPRO_INTEGRATION.md)** ⭐ - **W-PRO setup, pretrain strategy, and interpretability**
- **[QUICKSTART.md](QUICKSTART.md)** - Quick commands and minimal examples
- **[METHOD2_PIPELINE.md](METHOD2_PIPELINE.md)** - Detailed layer-by-layer architecture walkthrough
- **[FINAL_ARCHITECTURE.md](FINAL_ARCHITECTURE.md)** - Comparison of Method 1 vs Method 2
- **[MSP_PODCAST_SETUP.md](MSP_PODCAST_SETUP.md)** - MSP-Podcast dataset specifics
- **[PROJECT_SUMMARY.md](PROJECT_SUMMARY.md)** - Implementation status and progress

---

## 🔍 Key Implementation Details

### Multi-Task Loss

**CombinedEmotionLossCCC:**
```
L_total = α·L_emotion + β·L_arousal + γ·L_valence

where:
  L_emotion = CrossEntropyLoss (emotion classification)
  L_arousal = 1 - CCC(arousal_pred, arousal_true)
  L_valence = 1 - CCC(valence_pred, valence_true)
  
  CCC = Concordance Correlation Coefficient
      = 2·ρ·σ_pred·σ_true / (σ_pred² + σ_true² + (μ_pred - μ_true)²)
```

Default weights: α=1.0, β=1.0, γ=1.0 (equal importance)

### Why Freeze Pretrained Models?

1. **Prevent overfitting**: IEMOCAP has only ~10k samples; full fine-tuning of 319M params would overfit
2. **Faster training**: Only ~6.3M params updated per batch
3. **Transfer learning**: Pretrained models already learned robust audio/text representations
4. **Resource efficient**: Lower GPU memory usage, faster convergence

### Data Augmentation (IEMOCAP only)

IEMOCAP uses audio augmentation during training:
- Time stretching (rate: 0.9-1.1)
- Pitch shifting (n_steps: -2 to +2)
- Background noise addition (SNR: 15-25 dB)
- SpecAugment on spectrograms

MSP-Podcast currently does not use augmentation due to dataset size.

---

## 🐛 Troubleshooting

### OOM (Out of Memory) Errors

**Solution 1: Reduce batch size**
```yaml
# In config file
training:
  batch_size: 2  # Reduce from 4 or 8
```

**Solution 2: Use gradient accumulation** (not yet implemented)

**Solution 3: Use smaller audio length**
```yaml
data:
  max_audio_length: 96000  # 6 seconds instead of 10
```

### Training Too Slow

**MSP-Podcast specific:**
- Use higher batch size if GPU allows (batch_size=8 vs 4)
- Ensure audio files are on SSD, not network drive
- Check `num_workers` in config (default: 4)

**IEMOCAP specific:**
- Reduce epochs (30 → 20 for faster experimentation)
- Use fewer folds for initial testing (e.g., fold 1 only)

### Low Accuracy

**Checklist:**
- [ ] Verify emotion label mapping is correct (ang:0, hap:1, neu:2, sad:3)
- [ ] Check A-V normalization (should be z-score or [0,1] range)
- [ ] Ensure pretrained models are frozen (`freeze_wavlm: true`)
- [ ] Verify loss weights (try increasing emotion_weight if A-V dominates)
- [ ] Check learning rate (1e-4 is recommended, avoid >5e-4)

---

## 📝 Citation

If you use this code, please cite:

```
[Your paper citation here when published]
```

**Pretrained models:**
- WavLM: https://huggingface.co/microsoft/wavlm-base-plus
- RoBERTa: https://huggingface.co/roberta-base

---

## 📧 Contact

For questions or issues, please open an issue on GitHub or contact [your email].

---

**Last Updated**: November 15, 2025
