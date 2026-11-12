# W-PRO Integration & Pretrain Strategy

## 🎯 Overview

This document explains the integration of **WavLM-Pro (W-PRO)** for prosody extraction and the pretrain strategy for Method 2.

---

## 🔬 What is W-PRO?

**W-PRO** (WavLM-Prosody) is a specialized model from Orange AI for extracting **non-timbral voice embeddings**:

- **Base model**: WavLM-large (fine-tuned for prosody)
- **Purpose**: Extract prosody, accent, speaking style (NOT speaker identity/timbre)
- **Output**: Non-timbral embeddings that represent how someone speaks, not who speaks
- **Training**: VoxCeleb + VCTK datasets (English-focused)
- **Paper**: Gengembre et al., "Disentangling prosody and timbre embeddings via voice conversion", Interspeech 2024

### Model Variants

| Variant | HuggingFace ID | Embedding Size | EER (%) | Notes |
|---------|----------------|----------------|---------|-------|
| **W-PRO** | `Orange/Speaker-wavLM-pro` | 250 | 10.68 | Main variant, baseline in paper |
| WNTA128 | `Orange/Speaker-wavLM-pro` (revision: wnta128) | 128 | 5.00 | Enriched training |
| WNTA64 | `Orange/Speaker-wavLM-pro` (revision: wnta64) | 64 | 5.13 | Compact embeddings |

---

## 🏗️ Architecture Changes

### Before (Standard WavLM)

```
Audio ──> WavLM-base-plus (94M) ──> Prosody features [B, 499, 768]
          └─ Generic acoustic features
```

### After (W-PRO)

```
Audio ──> W-PRO (WavLM-large based, ~300M) ──> Prosody features [B, T, hidden_size]
          └─ Specialized non-timbral features (prosody, accent, style)
```

### Full Method 2 Architecture with W-PRO

```
Audio ──> W-PRO (frozen) ──────────┐
                                   ├──> Cross-Attention ──> A-V Prediction
Text ──> RoBERTa (frozen) ────────┘                         ├─> Arousal
                                                             └─> Valence
                                   │
Audio ──> WavLM-base-plus (frozen) ┤
                                   │
                                   └──> Final Fusion ──> Emotion Classifier
```

**Key Points:**
- **3 frozen pretrained models**: W-PRO, RoBERTa, WavLM-base-plus
- **Only task-specific layers trained**: ~6.3M parameters
- **W-PRO for prosody**: Better arousal representation (prosody ↔ arousal correlation)
- **WavLM-base-plus for final**: Global acoustic features for emotion classification

---

## ⚙️ Configuration

### Old Config (Standard WavLM)

```yaml
# configs/method2_config.yaml
model:
  wavlm_model_name: "microsoft/wavlm-base-plus"
  # Same model used for BOTH prosody and final branches
```

### New Config (W-PRO)

```yaml
# configs/method2_wpro_config.yaml
model:
  wavlm_model_name: "microsoft/wavlm-base-plus"  # For final branch
  wavlm_prosody_model_name: "Orange/Speaker-wavLM-pro"  # For prosody branch
  freeze_wavlm: true  # Freeze both WavLM models
  freeze_roberta: true  # Freeze RoBERTa
```

### Checkpoint Separation

| Config | Checkpoint Directory | Purpose |
|--------|---------------------|---------|
| `method2_config.yaml` | `checkpoints/method2_kfold/` | **Old runs** with standard WavLM (DO NOT OVERWRITE) |
| `method2_wpro_config.yaml` | `checkpoints/method2_kfold_wpro/` | **New runs** with W-PRO |

---

## 🎓 Pretrain Strategy

### ❌ What NOT to Do

**DO NOT pretrain/fine-tune the frozen backbones** (W-PRO, RoBERTa, WavLM):
- Would cause **catastrophic forgetting** of general knowledge
- Would **overfit** on small datasets (3 datasets < 5k samples total)
- Would lose the **pretrained representations** that cost millions to train

### ✅ Recommended Strategy: Multi-task Learning

**Keep backbones frozen, train task-specific layers on multiple datasets:**

```python
# Pseudocode
for epoch in epochs:
    for batch in multi_dataset_loader:  # Mix IEMOCAP, MSP-Podcast, EMOVOME
        # W-PRO, RoBERTa, WavLM-base-plus: FROZEN (requires_grad=False)
        # Only these are trained:
        #   - Cross-attention
        #   - A-V prediction heads
        #   - Emotion classifier
        #   - Fusion layers
        loss = combined_loss(batch)
        loss.backward()  # Only updates task-specific params (~6.3M)
```

**Benefits:**
- ✅ **Diverse data**: Exposure to multiple datasets, accents, recording conditions
- ✅ **Better generalization**: Task layers learn robust A-V and emotion patterns
- ✅ **No overfitting**: Backbones frozen prevents overfitting on small data
- ✅ **Fast training**: Only 6.3M params updated (98% of model frozen)

### Implementation Options

#### Option 1: Sequential Training (Current)

Train separately on each dataset:

```bash
# Train on IEMOCAP (5-fold CV)
bash scripts/run_train_kfold_wpro.sh cuda:0

# Train on MSP-Podcast
bash scripts/run_train_msp.sh cuda:0
```

**Pros**: Simple, easy to debug, dataset-specific checkpoints
**Cons**: Each model only sees one dataset

#### Option 2: Multi-task Learning (Recommended for pretrain)

Create mixed dataset loader:

```bash
# Future implementation: train_method2_multitask.py
python train/train_method2_multitask.py \
    --datasets iemocap,msp_podcast,emovome \
    --config configs/method2_wpro_config.yaml \
    --device cuda:0
```

**Pros**: Better generalization, diverse data exposure
**Cons**: Requires dataset loader implementation

#### Option 3: Transfer Learning (Large → Small)

Pretrain on large dataset, fine-tune on small:

```bash
# Step 1: Train on MSP-Podcast (197k samples)
python train/train_method2_msp.py \
    --config configs/method2_wpro_config.yaml \
    --device cuda:0

# Step 2: Fine-tune on EMOVOME (999 samples)
python train/train_method2_emovome.py \
    --config configs/method2_wpro_config.yaml \
    --resume checkpoints/method2_msp_wpro/best_model.pt \
    --learning_rate 5e-5 \
    --device cuda:0
```

**Pros**: Leverage large dataset knowledge for small datasets
**Cons**: Requires 2-stage training

---

## 🚀 Usage

### Train IEMOCAP with W-PRO

```bash
# GPU 0 (default)
bash scripts/run_train_kfold_wpro.sh cuda:0

# GPU 2
bash scripts/run_train_kfold_wpro.sh cuda:2

# Resume from fold 3
bash scripts/run_train_kfold_wpro.sh cuda:0 3 checkpoints/method2_kfold_wpro/fold_3/best_model.pt
```

### Compare W-PRO vs Standard WavLM

After training both:

```bash
# Old results (standard WavLM)
python scripts/test_kfold_results.py --checkpoint_dir checkpoints/method2_kfold

# New results (W-PRO)
python scripts/test_kfold_results.py --checkpoint_dir checkpoints/method2_kfold_wpro
```

---

## 📊 Expected Improvements

### Hypothesis

W-PRO should improve **arousal prediction** because:
- Prosody (pitch, rhythm, speaking rate) strongly correlates with arousal
- W-PRO is specifically trained to extract non-timbral prosody features
- Standard WavLM extracts generic acoustic features (may include timbre)

### Metrics to Watch

| Metric | Standard WavLM | W-PRO (Expected) |
|--------|----------------|------------------|
| **Arousal CCC** | ~0.55-0.65 | **0.60-0.70** ⬆️ |
| Valence CCC | ~0.50-0.60 | ~0.50-0.60 (similar) |
| Emotion Acc | ~64-70% | ~65-72% (slight ⬆️) |

**Why valence may not improve:**
- Valence depends more on semantic content (words) than prosody
- RoBERTa branch already captures semantic info for valence
- W-PRO's prosody features may not add much for valence

---

## 🔍 Interpretability

### Why W-PRO Improves Interpretability

**Old model (standard WavLM):**
```
"Prosody branch uses WavLM-base-plus"
↓
What features does it extract? Generic acoustic? Timbre? Prosody?
↓
Hard to explain why it predicts high/low arousal
```

**New model (W-PRO):**
```
"Prosody branch uses W-PRO (specialized non-timbral extractor)"
↓
Explicitly extracts: pitch patterns, rhythm, speaking rate, stress
↓
Easy to explain: High arousal = fast speech + high pitch variation
```

### Ablation Study Opportunities

With W-PRO, you can now analyze:
1. **Prosody-only prediction**: Disable text branch, use only W-PRO → arousal
2. **Text-only prediction**: Disable prosody branch, use only RoBERTa → valence
3. **Cross-attention importance**: Visualize attention weights between prosody ↔ text

---

## 🐛 Potential Issues & Solutions

### Issue 1: W-PRO Model Size

**Problem**: W-PRO is based on WavLM-large (~300M params) → larger memory footprint

**Solution**:
```yaml
training:
  batch_size: 2  # Reduce if OOM (was 4 for standard WavLM)
```

### Issue 2: Different Embedding Dimensions

**Problem**: W-PRO outputs different hidden size than WavLM-base-plus

**Solution**: Projection layer automatically adapts:
```python
self.prosody_projection = nn.Linear(self.wavlm_dim, hidden_dim)
# wavlm_dim = W-PRO's hidden_size (e.g., 1024 for WavLM-large)
# hidden_dim = 256 (consistent across all branches)
```

### Issue 3: Language Limitation

**Problem**: W-PRO trained on English-focused datasets (VoxCeleb, VCTK)

**Impact**:
- ✅ Good for IEMOCAP (English)
- ✅ Good for MSP-Podcast (English)
- ⚠️ May degrade for non-English datasets

**Solution**: Evaluate on English-only datasets first

---

## 📚 References

1. **W-PRO Paper**: Gengembre, N., Le Blouch, O., Gendrot, C. (2024). "Disentangling prosody and timbre embeddings via voice conversion". *Interspeech 2024*, 2765-2769.

2. **HuggingFace Model**: https://huggingface.co/Orange/Speaker-wavLM-pro

3. **Related Models**:
   - W-TBR: Timbre-related embeddings
   - W-SPK: Speaker embeddings (ASV)

---

## ✅ Checklist Before Training

- [ ] Install W-PRO model: `pip install transformers>=4.30.0`
- [ ] Verify GPU memory: W-PRO needs ~20-24GB for batch_size=4
- [ ] Check checkpoint directory: `checkpoints/method2_kfold_wpro/` (separate from old)
- [ ] Backup old results: `checkpoints/method2_kfold/` should NOT be modified
- [ ] Choose GPU: cuda:0 or cuda:2 based on availability
- [ ] Test forward pass: `python scripts/test_models.py` (verify W-PRO loads)

---

**Ready to train with W-PRO! 🚀**
