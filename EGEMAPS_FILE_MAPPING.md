# eGeMAPS Models - File Mapping Guide

This document shows the relationship between configuration files, model files, and training scripts.

## Model Definitions

### 1. EGeMAPSEmotionClassifier
**File**: `models/egemaps_emotion.py`

```
Input: eGeMAPS (88-d)
  ↓
Linear(88 → 256) + LayerNorm + ReLU
  ↓
Transformer Layers (4 layers, 4 heads)
  ↓
Emotion Head: 256 → 128 → 4 classes
  ↓
Output: emotion_logits [B, 4]
```

**Usage**:
- Config: `iemocap_egemaps_emotion_kfold_config.yaml`
- Training: `train_iemocap_egemaps_emotion_kfold.py --fold 1-5`
- Dataset: eGeMAPS .npy files from `outputs/iemocap/egemaps_npy/`

---

### 2. OpenSMILEEmotionAVClassifier
**File**: `models/opensmile_emotion_av.py`

```
Input: eGeMAPS (88-d)
  ↓
Linear(88 → 256) + LayerNorm + GELU (embedding)
  ↓
Transformer Layers (4 layers, 4 heads)
  ↓
├─ Emotion Head: 256 → 128 → 4 classes → emotion_logits [B, 4]
├─ Arousal Head: 256 → 128 → 1 → Tanh → arousal [B, 1]
└─ Valence Head: 256 → 128 → 1 → Tanh → valence [B, 1]
  ↓
Output: {emotion_logits, arousal, valence}
```

**Usage - Config 1**:
- Config: `iemocap_opensmile_emotion_kfold_config.yaml`
- Training: `train_iemocap_opensmile_emotion_kfold.py --fold 1-5`
- Type: Emotion + A-V training

**Usage - Config 2**:
- Config: `iemocap_opensmile_emotion_kfold_config.yaml` (same config)
- Training: `train_iemocap_opensmile_emotion_av_kfold.py --fold 1-5`
- Type: Emotion + A-V training (same model, slight training differences)

**Dataset**: eGeMAPS .npy files from `outputs/iemocap/egemaps_npy/`

---

### 3. Method2CrossAttentionAVOpenSMILE
**File**: `models/method2_crossattn_av_opensmile.py`

```
Inputs:
  - WavLM embeddings (precomputed): [B, T, 768]
  - RoBERTa embeddings (precomputed): [B, S, 768]
  - eGeMAPS features: [B, 88]
  
Audio Branch:
  WavLM [B, T, 768] → Linear(768 → 256) → SelfAttentionPooling → [B, 256]
  
Text Branch:
  RoBERTa [B, S, 768] → Linear(768 → 256) → SelfAttentionPooling → [B, 256]
  
Prosody Branch:
  eGeMAPS [B, 88] → Linear(88 → 256) → [B, 256]
  
Fusion:
  Concat(audio, text, prosody) [B, 768] → MLP → [B, 256]
  
A-V Prediction:
  [B, 256] → Linear(256 → 128) → [B, 2]
  ↓
  arousal [B, 1], valence [B, 1]
  
Emotion Prediction:
  [B, 256] → Linear(256 → 128) → [B, 4]
  ↓
  emotion_logits [B, 4]

Output: {emotion_logits, arousal, valence}
```

**Usage**:
- Config: `iemocap_method2_opensmile_kfold_config.yaml`
- Training: `train_iemocap_method2_opensmile_kfold.py --fold 1-5`
- Dataset: Precomputed embeddings (WavLM + RoBERTa) + eGeMAPS .npy files
- Location:
  - WavLM: `datasets/IEMOCAP/wavlm_embeddings/`
  - RoBERTa: `datasets/IEMOCAP/roberta_embeddings/`
  - eGeMAPS: `outputs/iemocap/egemaps_npy/`

---

### 4. Method2CrossAttentionAVEGeMAPS
**File**: `models/method2_av_egemaps.py`

```
Same architecture as Method2CrossAttentionAVOpenSMILE
(Can be used for any dataset with WavLM + RoBERTa + eGeMAPS features)

Inputs:
  - Raw audio: [B, audio_length]
  - Text IDs: [B, max_text_length]
  - Text attention mask: [B, max_text_length]
  - eGeMAPS features: [B, 88]
  
Audio Branch:
  Raw audio → WavLM (frozen) → [B, T, 768]
           → Linear(768 → 256) → SelfAttentionPooling → [B, 256]
  
Text Branch:
  Text IDs → RoBERTa (frozen) → [B, S, 768]
         → Linear(768 → 256) → SelfAttentionPooling → [B, 256]
         
Prosody Branch:
  eGeMAPS [B, 88] → Linear(88 → 256) → Transformer → [B, 256]
  
Cross-Attention (bidirectional):
  Audio ←→ Text
  
Post-Attention + Fusion:
  Concat → [B, 512] → MLP → [B, 256]
  
A-V Refinement:
  [B, 2] → MLP (2 layers) → [B, 2]
  
Final Emotion:
  WavLM final [B, 256] + fusion [B, 256] + AV_refined [B, 2]
           → [B, 512+2] → MLP → [B, 256] → [B, 4]

Output: {emotion_logits, arousal, valence}
```

**Usage**:
- Config: `method2_av_egemaps_config.yaml` (for MSP-Podcast)
- Training: Custom script required (use as template)
- Dataset: 
  - WavLM: `datasets/MSP-Podcast/wavlm_embeddings/` (need to create)
  - RoBERTa: `datasets/MSP-Podcast/roberta_embeddings/` (need to create)
  - eGeMAPS: `outputs/msp_podcast/egemaps_npy/`

---

## Configuration Files

### IEMOCAP - 5 Fold CV

| Config File | Model | Class | Training Script | Dataset |
|---|---|---|---|---|
| `iemocap_egemaps_emotion_kfold_config.yaml` | eGeMAPS emotion-only | EGeMAPSEmotionClassifier | `train_iemocap_egemaps_emotion_kfold.py` | eGeMAPS only |
| `iemocap_opensmile_emotion_kfold_config.yaml` | eGeMAPS emotion+AV | OpenSMILEEmotionAVClassifier | `train_iemocap_opensmile_emotion_kfold.py` | eGeMAPS only |
| `iemocap_method2_opensmile_kfold_config.yaml` | Method2 + eGeMAPS | Method2CrossAttentionAVOpenSMILE | `train_iemocap_method2_opensmile_kfold.py` | WavLM + RoBERTa + eGeMAPS |

### MSP-Podcast

| Config File | Model | Class | Training Script | Dataset |
|---|---|---|---|---|
| `method2_av_egemaps_config.yaml` | Method2 + eGeMAPS | Method2CrossAttentionAVEGeMAPS | Custom needed | WavLM + RoBERTa + eGeMAPS |

---

## Training Command Examples

### IEMOCAP - eGeMAPS Emotion Only
```bash
python train/train_iemocap_egemaps_emotion_kfold.py \
  --config configs/iemocap_egemaps_emotion_kfold_config.yaml \
  --fold 1 \
  --device cuda:0
```

### IEMOCAP - eGeMAPS Emotion + A-V
```bash
python train/train_iemocap_opensmile_emotion_kfold.py \
  --config configs/iemocap_opensmile_emotion_kfold_config.yaml \
  --fold 1 \
  --device cuda:0
```

### IEMOCAP - Method2 + eGeMAPS
```bash
python train/train_iemocap_method2_opensmile_kfold.py \
  --config configs/iemocap_method2_opensmile_kfold_config.yaml \
  --fold 1 \
  --device cuda:0
```

---

## Dataset Preparation

### eGeMAPS Features
Extract once and reuse across all training:
```bash
# Extract eGeMAPS from audio files
# Output: outputs/iemocap/egemaps_npy/ (88-d .npy files)
# Named by: sample_name.npy
```

### WavLM Embeddings (for Method2)
Extract frozen WavLM features:
```bash
# From: datasets/IEMOCAP/Audios/
# To: datasets/IEMOCAP/wavlm_embeddings/
# One file per sample: sample_name.npy [T, 768]
```

### RoBERTa Embeddings (for Method2)
Extract frozen RoBERTa features from text:
```bash
# From: datasets/IEMOCAP_KFOLD/fold_{fold}/transcripts.txt
# To: datasets/IEMOCAP/roberta_embeddings/
# One file per sample: sample_name.npy [S, 768]
```

---

## Key Differences

### Models by Complexity
1. **EGeMAPSEmotionClassifier** - Simplest
   - Single input: eGeMAPS
   - Single task: Emotion classification
   - ~0.5M parameters

2. **OpenSMILEEmotionAVClassifier** - Single-branch multi-task
   - Single input: eGeMAPS
   - Multi-task: Emotion + A-V
   - ~1M parameters

3. **Method2CrossAttentionAVOpenSMILE** - Multi-branch, simplified
   - Three inputs: WavLM + RoBERTa + eGeMAPS
   - Multi-task: Emotion + A-V
   - Simple fusion (concatenate)
   - ~1M parameters (only head trainable)

4. **Method2CrossAttentionAVEGeMAPS** - Multi-branch, full
   - Three inputs: Raw audio + Text IDs + eGeMAPS
   - Multi-task: Emotion + A-V
   - Complex fusion (cross-attention + refinement)
   - ~6M parameters (WavLM final is trainable)

---

## Quick Reference

**Want to train emotion-only?**
→ Use `EGeMAPSEmotionClassifier`

**Want to train emotion + A-V from single acoustic features?**
→ Use `OpenSMILEEmotionAVClassifier`

**Want to train full multimodal method2 with text?**
→ Use `Method2CrossAttentionAVOpenSMILE` (IEMOCAP)
→ Use `Method2CrossAttentionAVEGeMAPS` (Custom dataset)

