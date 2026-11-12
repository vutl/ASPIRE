# eGeMAPS Models - Summary of Changes

## Overview
This document summarizes the cleanup and fixes made to eGeMAPS-based emotion recognition models and their training scripts.

## Models

### 1. `models/method2_av_egemaps.py` - **REWRITTEN**
**Status**: ✅ FIXED

**Architecture**: Method 2 Cross-Attention with eGeMAPS Prosody
- **Class**: `Method2CrossAttentionAVEGeMAPS`
- **Input branches**:
  - Audio: WavLM (frozen) → prosody features
  - Text: RoBERTa (frozen) → semantic features
  - Prosody: eGeMAPS (88-d) → Transformer layers
- **Processing**:
  - WavLM and RoBERTa cross-attention
  - Post-attention fusion
  - A-V prediction + refinement
  - Final emotion prediction
- **Output**:
  - `emotion_logits`: (batch_size, 4) - Emotion classification
  - `arousal`: (batch_size, 1) - Continuous arousal
  - `valence`: (batch_size, 1) - Continuous valence

**Previous Issue**: Model was incomplete (A-V only, missing emotion + RoBERTa)
**Fix**: Complete rewrite to match full Method2 architecture

---

### 2. `models/opensmile_emotion_av.py` - **CLEANED UP**
**Status**: ✅ FIXED

**Architecture**: Single-Branch eGeMAPS → Emotion + A-V
- **Class**: `OpenSMILEEmotionAVClassifier`
- **Input**: eGeMAPS features (88-d)
- **Processing**:
  - Embedding projection
  - Transformer layers for self-attention
  - Separate heads for emotion and A-V
- **Output**:
  - `emotion_logits`: (batch_size, 4) - Emotion classification
  - `arousal`: (batch_size, 1) - Continuous arousal
  - `valence`: (batch_size, 1) - Continuous valence

**Previous Issue**: Redundant `av_preds` in return dict, unclear docstring
**Fixes**:
- Removed redundant `av_preds` (use arousal/valence directly)
- Updated docstring to clarify it's eGeMAPS-based, not "OpenSMILE-only"

---

### 3. `models/egemaps_emotion.py` - **UNCHANGED**
**Status**: ✅ OK

**Architecture**: eGeMAPS → Emotion Only
- **Class**: `EGeMAPSEmotionClassifier`
- **Input**: eGeMAPS features (88-d)
- **Output**: `emotion_logits` (batch_size, 4)
- No A-V prediction

---

## Training Scripts

### IEMOCAP 5-Fold Cross-Validation

| Script | Model | Architecture | Status |
|--------|-------|--------------|--------|
| `train_iemocap_egemaps_emotion_kfold.py` | EGeMAPSEmotionClassifier | eGeMAPS → Emotion | ✅ OK |
| `train_iemocap_method2_opensmile_kfold.py` | Method2CrossAttentionAVOpenSMILE | WavLM + RoBERTa + eGeMAPS → Cross-Attn → A-V + Emotion | ✅ OK |
| `train_iemocap_opensmile_emotion_av_kfold.py` | OpenSMILEEmotionAVClassifier | eGeMAPS → Emotion + A-V | ✅ OK |
| `train_iemocap_opensmile_emotion_kfold.py` | OpenSMILEEmotionAVClassifier | eGeMAPS → Emotion + A-V | ⚠️ FIXED |

**Note**: `train_iemocap_egemaps_method2_av_kfold.py` was **DELETED** (incorrect architecture)

---

## Fixes Applied

### 1. `train/train_iemocap_opensmile_emotion_kfold.py`
**Issue**: Collate function didn't include arousal/valence data, but training loop expected them
**Fix**: Updated collate_fn to include arousal and valence in batch
```python
# OLD: Only returned embedding, emotion_label, sample_name
# NEW: Also returns arousal, valence
```

**Issue**: Docstring said "emotion-only" but model outputs A-V
**Fix**: Updated docstring to "Emotion + A-V Classifier"

---

### 2. `models/method2_av_egemaps.py`
**Issue**: Was incomplete (A-V only, no emotion, no RoBERTa)
**Fix**: Complete rewrite as full Method2 architecture
- Added RoBERTa text branch
- Added emotion prediction head
- Implemented cross-attention between WavLM and RoBERTa
- Added A-V refinement module
- Follows same pattern as `method2_crossattn_av.py` but with eGeMAPS instead of W-PRO

---

### 3. `models/opensmile_emotion_av.py`
**Issue**: Docstring unclear, redundant return value
**Fixes**:
- Updated docstring to clarify it's eGeMAPS-based (not just "OpenSMILE")
- Removed redundant `av_preds` from return dict (use arousal/valence directly)

---

### 4. `configs/method2_av_egemaps_config.yaml` (MSP-Podcast)
**Issue**: Config was for A-V only model (old architecture)
**Fix**: Updated to new Method2 architecture
- Added `wavlm_dir`, `roberta_dir` for precomputed embeddings
- Updated model parameters to match `Method2CrossAttentionAVEGeMAPS`
- Aligned with `iemocap_method2_opensmile_kfold_config.yaml` structure

---

## Model Lineup

### eGeMAPS-based Models (Final)

1. **EGeMAPSEmotionClassifier** - Simple baseline
   - Input: eGeMAPS (88-d)
   - Output: Emotion (4-class)
   - Use: `train_iemocap_egemaps_emotion_kfold.py`

2. **OpenSMILEEmotionAVClassifier** - Single-branch with A-V
   - Input: eGeMAPS (88-d)
   - Output: Emotion (4-class) + Arousal + Valence
   - Use: 
     - `train_iemocap_opensmile_emotion_av_kfold.py`
     - `train_iemocap_opensmile_emotion_kfold.py`

3. **Method2CrossAttentionAVOpenSMILE** - Multi-branch (IEMOCAP)
   - Inputs: WavLM (frozen) + RoBERTa (frozen) + eGeMAPS
   - Output: Emotion (4-class) + Arousal + Valence
   - Use: `train_iemocap_method2_opensmile_kfold.py`

4. **Method2CrossAttentionAVEGeMAPS** - Multi-branch (General)
   - Inputs: WavLM (frozen) + RoBERTa (frozen) + eGeMAPS
   - Output: Emotion (4-class) + Arousal + Valence
   - Use: For custom training (e.g., MSP-Podcast if needed)

---

## Files Deleted

- `train/train_iemocap_egemaps_method2_av_kfold.py` 
  - Reason: Incompatible with new model architecture, duplicate functionality
  - Replacement: Use `train_iemocap_method2_opensmile_kfold.py` (handles eGeMAPS the same way)

---

## Config Files Reference

### IEMOCAP
| Config | Purpose | Model | Dataset |
|--------|---------|-------|---------|
| `iemocap_egemaps_emotion_kfold_config.yaml` | eGeMAPS emotion-only | EGeMAPSEmotionClassifier | IEMOCAP 5-fold |
| `iemocap_opensmile_emotion_kfold_config.yaml` | eGeMAPS emotion+A-V | OpenSMILEEmotionAVClassifier | IEMOCAP 5-fold |
| `iemocap_method2_opensmile_kfold_config.yaml` | Method2 with eGeMAPS | Method2CrossAttentionAVOpenSMILE | IEMOCAP 5-fold |

### MSP-Podcast
| Config | Purpose | Model | Dataset |
|--------|---------|-------|---------|
| `method2_av_egemaps_config.yaml` | Method2 with eGeMAPS | Method2CrossAttentionAVEGeMAPS | MSP-Podcast |

---

## Verification

All models verified importable and initialized without errors:
```
✓ Method2CrossAttentionAVEGeMAPS imported
✓ OpenSMILEEmotionAVClassifier imported
✓ EGeMAPSEmotionClassifier imported
```

---

## Notes

- eGeMAPS is the traditional approach using hand-crafted acoustic features (88 dimensions)
- All models use frozen pre-trained backbones (WavLM, RoBERTa) for efficiency
- eGeMAPS features should be precomputed in `.npy` files organized by sample name
- The naming "OpenSMILE" in some scripts is used interchangeably with "eGeMAPS" (eGeMAPS is extracted via OpenSmile tool)
