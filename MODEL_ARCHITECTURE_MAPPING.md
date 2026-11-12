# Model Architecture & Trainer Mapping

## Summary of All Models

| # | Model Name | Trainer File | Model Class | Config File | Dataset | Notes |
|---|---|---|---|---|---|---|
| 1 | **OpenSMILE Only (Emotion)** | `train_iemocap_opensmile_emotion_kfold.py` | `OpenSMILEEmotionClassifier` | `iemocap_opensmile_emotion_kfold_config.yaml` | IEMOCAP (5-fold) | Baseline: OpenSMILE 88-d → emotion only |
| 2 | **OpenSMILE Only (Emotion+AV)** | `train_iemocap_opensmile_emotion_av_kfold.py` | `OpenSMILEEmotionAVClassifier` | *(custom)* | IEMOCAP (5-fold) | **[NEW]** OpenSMILE → emotion + arousal + valence (continuous) |
| 3 | **EGeMAPS Only (Emotion)** | `train_iemocap_egemaps_emotion_kfold.py` | `EGeMAPSEmotionClassifier` | `iemocap_egemaps_emotion_kfold_config.yaml` | IEMOCAP (5-fold) | Baseline: EGeMAPS features → emotion only |
| 4 | **Method1 Triple WavLM** | `train_method1.py` | `Method1TripleWavLM` | `method1_config.yaml` | ? | Original method 1 architecture |
| **IEMOCAP 5-Fold Models** | | | | | | |
| 5 | **Method2 (WavLM+RoBERTa+WavLM)** | `train_method2_kfold.py` | `Method2CrossAttentionAV` | `method2_config.yaml` | IEMOCAP (5-fold) | WavLM + RoBERTa cross-attention + WavLM final fusion → emotion + arousal + valence |
| 6 | **Method2 + OpenSMILE** | `train_iemocap_method2_opensmile_kfold.py` | `Method2CrossAttentionAVOpenSMILE` | `iemocap_method2_opensmile_kfold_config.yaml` | IEMOCAP (5-fold) | **[NEW]** WavLM + RoBERTa + OpenSMILE → emotion + A-V |
| 7 | **Method2 + EGeMAPS (Emotion+AV)** | `train_iemocap_egemaps_method2_av_kfold.py` | `Method2CrossAttentionAV` | `method2_av_egemaps_config.yaml` | IEMOCAP (5-fold) | WavLM + RoBERTa + EGeMAPS → emotion + arousal + valence |
| 8 | **Method2 + WPro (Emotion+AV)** | `train_method2_kfold_wpro.py` | `Method2CrossAttentionAV` | `method2_wpro_config.yaml` | IEMOCAP (5-fold) | WavLM + RoBERTa + W-PRO (prosody) → emotion + arousal + valence |
| 9 | **Method2 + WPro + Emotion2Vec** | `train_method2_kfold_emotion2vec.py` | `Method2CrossAttentionAV_Emotion2Vec` | `method2_emotion2vec_config.yaml` | IEMOCAP (5-fold) | WavLM + RoBERTa + Emotion2Vec → emotion + A-V |
| 10 | **Method2 + WPro + AVD** | `train_method2_kfold_wpro_avd.py` | `Method2CrossAttentionAVD` | `method2_wpro_avd_kfold_config.yaml` | IEMOCAP (5-fold) | WavLM + RoBERTa + W-PRO → emotion + arousal + valence + dominance |
| 11 | **Method2 + WPro + AVD + ASR** | `train_method2_kfold_wpro_asr.py` | `Method2CrossAttentionAV` | `method2_wpro_asr_kfold_config.yaml` | IEMOCAP (5-fold) | WavLM + RoBERTa + W-PRO + ASR text → emotion + A-V |
| 12 | **Method2 + WPro + Fusion Concat** | `train_method2_kfold_fusion_concat.py` | `Method2CrossAttentionAV_FusionConcat` | `method2_config.yaml` | IEMOCAP (5-fold) | WavLM + RoBERTa + Fusion (final concat uses branch_fused) → emotion + A-V |
| 13 | **Method2 + WPro Simplified** | `train_wpro_simplified_kfold.py` | `Method2WPROSimplified` | `wpro_simplified_kfold_config.yaml` | IEMOCAP (5-fold) | Simplified W-PRO variant → emotion + A-V |
| 14 | **WPro Only** | `train_wpro_av_kfold.py` | `WProAVKfold` | `method2_wpro_config.yaml` | IEMOCAP (5-fold) | W-PRO prosody only → emotion + A-V |
| 15 | **SSL Baseline** | `train_ssl_baseline_kfold.py` | `SSLBaselineAVD` | `ssl_baseline_kfold_config.yaml` | IEMOCAP (5-fold) | HuBERT/WavLM only (SSL pre-trained) → emotion + A-V + dominance |
| **MSP-Podcast Models** | | | | | | |
| 16 | **Method2 (MSP)** | `train_method2_msp.py` | `Method2CrossAttentionAV` | `msp_podcast_config.yaml` | MSP-Podcast | WavLM + RoBERTa + WavLM final fusion → emotion + A-V (test1/test2) |
| 17 | **Method2 + WPro (MSP)** | `train_method2_msp_wpro.py` | `Method2CrossAttentionAV` | `msp_podcast_config.yaml` | MSP-Podcast | WavLM + RoBERTa + W-PRO → emotion + A-V (test1/test2) |
| 18 | **Method2 + WPro + AVD (MSP)** | `train_method2_msp_wpro_avd.py` | `Method2CrossAttentionAVD` | `msp_podcast_avd_wpro_baseline_config.yaml` | MSP-Podcast | WavLM + RoBERTa + W-PRO → emotion + A-V + dominance (test1/test2) |
| 19 | **HuBERT (MSP)** | `train_msp_hubert_av.py` | `HuBERTAV` | `msp_podcast_config.yaml` | MSP-Podcast | HuBERT pre-trained → emotion + A-V (test1/test2) |
| 20 | **WavLM Base (MSP)** | `train_msp_wavlm_base_av.py` | `WavLMBaseAV` | `msp_hubert_av_resume_config.yaml` | MSP-Podcast | WavLM pre-trained → emotion + A-V (test1/test2) |
| 21 | **WPro Only (MSP)** | `train_msp_wpro_av.py` | `WProAV` | `msp_podcast_config.yaml` | MSP-Podcast | W-PRO prosody only → emotion + A-V (test1/test2) |
| 22 | **Emotion2Vec (MSP)** | `train_msp_emotion2vec_av.py` | `Emotion2VecAV` | `msp_emotion2vec_av_config.yaml` | MSP-Podcast | Emotion2Vec embeddings → emotion + A-V (test1/test2) |
| 23 | **WPro Simplified (MSP)** | `train_wpro_simplified_msp.py` | `Method2WPROSimplified` | `wpro_simplified_msp_config.yaml` | MSP-Podcast | Simplified W-PRO variant → emotion + A-V (test1/test2) |
| 24 | **SSL Baseline (MSP)** | `train_ssl_baseline_msp.py` | `SSLBaselineAVD` | `ssl_baseline_msp_config.yaml` | MSP-Podcast | HuBERT/WavLM SSL → emotion + A-V + dominance (test1/test2) |

---

## Detailed Architecture Breakdown

### 1. **OpenSMILE-Only Models**
- **Input:** OpenSMILE 88-dimensional acoustic features (`.npy` files)
- **Processing:** MLP with shared hidden layer
- **Outputs:**
  - Emotion: 4-class logits (anger, happiness, sadness, neutral)
  - Arousal: continuous [0-5] or [-1, 1]
  - Valence: continuous [0-5] or [-1, 1]
- **Loss:** CrossEntropyLoss (emotion) + MSELoss (A-V)
- **Metrics:** Emotion accuracy, F1; Arousal/Valence CCC
- **Checkpoints:** `iemocap_opensmile_emotion_av_kfold/fold_{1-5}/`

### 2. **Method2 Core Architecture (WavLM + RoBERTa + Fusion)**
```
Audio Input (raw waveform)
    ↓
WavLM-Base [Wav2Vec2Model]
    ↓ [B, T, 768] embeddings
    
Text Input (speaker + emotion context)
    ↓
RoBERTa
    ↓ [B, T, 768] embeddings

Cross-Attention Fusion
    ↓ [B, hidden_dim]
    
Final Fusion Branch
    ↓
Output Heads:
  - Emotion: [B, 4]
  - Arousal: [B, 1]
  - Valence: [B, 1]
  - (Optional Dominance: [B, 1])
```

- **Base Config:** `method2_config.yaml`
- **Variations:**
  - **+OpenSMILE:** Replace final fusion with OpenSMILE 88-d concatenation
  - **+EGeMAPS:** Replace final fusion with EGeMAPS features
  - **+WPro:** Replace/add W-PRO prosody features to final fusion
  - **+Emotion2Vec:** Use Emotion2Vec embeddings instead of RoBERTa
  - **+ASR:** Add ASR text tokens to RoBERTa input
  - **+Fusion Concat:** Modify final concatenation strategy (use `branch_fused` variable)
  - **+AVD:** Output dominance as 4th dimension
  - **+Simplified:** Simplified architecture (fewer layers/projections)

### 3. **Single-Modality Models (MSP-Podcast)**
These models use only one modality for A-V prediction:

- **HuBERT:** Pre-trained HuBERT-Base [facebook/hubert-base-ls960]
- **WavLM:** Pre-trained WavLM-Base [microsoft/wavlm-base]
- **WPro:** W-PRO prosody features (extracted offline)
- **Emotion2Vec:** Emotion2Vec downstream embeddings

### 4. **SSL Baseline**
- **Architecture:** Simple SSL-based model (HuBERT/WavLM only)
- **Outputs:** Emotion + Arousal + Valence + Dominance
- **Config:** `ssl_baseline_kfold_config.yaml` (IEMOCAP), `ssl_baseline_msp_config.yaml` (MSP)

---

## Dataset Handling

### IEMOCAP (5-Fold Cross-Validation)
- **Files:** `datasets/IEMOCAP_KFOLD/all.csv`
- **Fold Structure:** 
  - Fold 1: Test Session 1, Train Sessions 2-5
  - Fold 2: Test Session 2, Train Sessions 1,3-5
  - Fold 3: Test Session 3, Train Sessions 1-2,4-5
  - Fold 4: Test Session 4, Train Sessions 1-3,5
  - Fold 5: Test Session 5, Train Sessions 1-4
- **Split Ratio:** Train:Val:Test = 3:1:1 per fold (3 sessions train, 1 val, 1 test)
- **Features Needed:**
  - Raw audio `.wav` files: `datasets/IEMOCAP/*/sentences/wav/`
  - OpenSMILE: `checkpoints/*/opensmile_features.npy` or computed on-the-fly
  - EGeMAPS: `checkpoints/*/egemaps_features.npy` or computed on-the-fly
  - WPro: Pre-extracted prosody features
  - Emotion2Vec: `emotion2vec/iemocap_downstream/`

### MSP-Podcast
- **Files:** `datasets/MSP-Podcast/{audio,transcriptions}`
- **Split Structure:** 
  - test1: Speaker-independent test set 1
  - test2: Speaker-independent test set 2
  - Train: Remaining samples
- **Features Needed:**
  - Raw audio `.wav` files
  - WPro: Pre-extracted prosody features
  - Emotion2Vec: Pre-extracted embeddings
  - HuBERT/WavLM: Computed on-the-fly from pre-trained models

---

## Config File Mapping

| Config File | Used By | Dataset | Notes |
|---|---|---|---|
| `iemocap_opensmile_emotion_kfold_config.yaml` | train_iemocap_opensmile_emotion_kfold.py | IEMOCAP | OpenSMILE emotion-only baseline |
| `iemocap_method2_opensmile_kfold_config.yaml` | train_iemocap_method2_opensmile_kfold.py | IEMOCAP | **[NEW]** OpenSMILE + Method2 → A-V |
| `iemocap_egemaps_emotion_kfold_config.yaml` | train_iemocap_egemaps_emotion_kfold.py | IEMOCAP | EGeMAPS emotion-only baseline |
| `method2_config.yaml` | Multiple trainers | IEMOCAP | Default Method2 (WavLM+RoBERTa+WavLM) |
| `method2_av_egemaps_config.yaml` | train_iemocap_egemaps_method2_av_kfold.py | IEMOCAP | Method2 + EGeMAPS |
| `method2_wpro_config.yaml` | train_method2_kfold_wpro.py, train_wpro_av_kfold.py | IEMOCAP | Method2 + WPro variant |
| `method2_emotion2vec_config.yaml` | train_method2_kfold_emotion2vec.py | IEMOCAP | Method2 + Emotion2Vec |
| `method2_wpro_asr_kfold_config.yaml` | train_method2_kfold_wpro_asr.py | IEMOCAP | Method2 + WPro + ASR |
| `method2_wpro_avd_kfold_config.yaml` | train_method2_kfold_wpro_avd.py | IEMOCAP | Method2 + WPro → A-V-D |
| `method2_wpro_continue_config.yaml` | train_method2_msp_wpro_continue.py (if exists) | IEMOCAP | Method2 resume training config |
| `ssl_baseline_kfold_config.yaml` | train_ssl_baseline_kfold.py | IEMOCAP | SSL baseline (HuBERT/WavLM) |
| `wpro_simplified_kfold_config.yaml` | train_wpro_simplified_kfold.py | IEMOCAP | WPro simplified |
| `msp_podcast_config.yaml` | Multiple MSP trainers | MSP-Podcast | Default MSP config |
| `msp_podcast_config_bs8.yaml` | - | MSP-Podcast | Batch size 8 variant |
| `msp_podcast_wpro_config.yaml` | - | MSP-Podcast | MSP + WPro variant |
| `msp_emotion2vec_av_config.yaml` | train_msp_emotion2vec_av.py | MSP-Podcast | MSP + Emotion2Vec → A-V |
| `msp_hubert_av_resume_config.yaml` | train_msp_wavlm_base_av.py | MSP-Podcast | WavLM resume/fine-tune |
| `ssl_baseline_msp_config.yaml` | train_ssl_baseline_msp.py | MSP-Podcast | SSL baseline (MSP) |
| `wpro_simplified_msp_config.yaml` | train_wpro_simplified_msp.py | MSP-Podcast | WPro simplified (MSP) |
| `msp_podcast_avd_wpro_baseline_config.yaml` | train_method2_msp_wpro_avd.py, train_ssl_baseline_msp.py, train_wpro_simplified_msp.py | MSP-Podcast | Baseline with A-V-D |
| `method1_config.yaml` | train_method1.py | ? | Method1 original architecture |

---

## Checkpoint Directory Status

✅ **Existing Checkpoints (23 directories):**

```
IEMOCAP (5-fold):
✅ iemocap_method2_opensmile_kfold/        (fold_1-5)   [NEW: Method2 + OpenSMILE → A-V]
✅ iemocap_opensmile_emotion_av_kfold/     (fold_1-5)   [NEW: OpenSMILE → emotion + A-V]
✅ iemocap_opensmile_emotion_kfold/        (fold_1-5)   [OLD: OpenSMILE → emotion only]
✅ method2_kfold/                          (fold_1-5)   [Method2: WavLM+RoBERTa+WavLM]
✅ method2_kfold_emotion2vec/              (fold_1-5)   [Method2 + Emotion2Vec]
✅ method2_kfold_fusion_concat/            (fold_1-5)   [Method2 + Fusion Concat variant]
✅ method2_kfold_wpro/                     (fold_1-5)   [Method2 + WPro]
✅ method2_kfold_wpro_asr/                 (fold_1-5)   [Method2 + WPro + ASR]
✅ method2_kfold_wpro_avd/                 (fold_1-5)   [Method2 + WPro → A-V-D]
✅ ssl_baseline_kfold/                     (fold_1-5)   [SSL baseline]
✅ wpro_av_kfold/                          (fold_1-5)   [WPro only]
✅ wpro_simplified_kfold/                  (fold_1-5)   [WPro simplified]

MSP-Podcast (test1 + test2):
✅ method2_msp/                            (no folds)   [Method2: WavLM+RoBERTa+WavLM]
✅ method2_msp_avd/                        (no folds)   [Method2 → A-V-D]
✅ method2_msp_bs8_cuda2/                  (no folds)   [Method2 batch size 8]
✅ method2_msp_wpro/                       (no folds)   [Method2 + WPro]
✅ method2_msp_wpro_bs4_cuda0/             (no folds)   [Method2 + WPro batch size 4]
✅ method2_msp_wpro_continue/              (no folds)   [Method2 + WPro resumed training]
✅ msp_egemaps_emotion/                    (no folds)   [EGeMAPS emotion only]
✅ msp_emotion2vec_av/                     (no folds)   [Emotion2Vec → A-V]
✅ msp_hubert_av/                          (no folds)   [HuBERT → A-V]
✅ msp_wavlm_base_av/                      (no folds)   [WavLM Base → A-V]
✅ msp_wpro_av/                            (no folds)   [WPro → A-V]
✅ ssl_baseline_kfold/ (MSP)               (no folds)   [SSL baseline]
✅ wpro_simplified_kfold/ (MSP)            (no folds)   [WPro simplified]
```

---

## Evaluation Requirements

For user's requested evaluation:

### New Trainers to Create (if needed):
These are the **missing trainers** you requested:

1. **Method2 WPro AV (IEMOCAP)** ✅ Already exists: `train_method2_kfold_wpro.py`
2. **Method2 WPro AV (MSP)** ✅ Already exists: `train_method2_msp_wpro.py`
3. **Method2 without WPro (WavLM for prosody too) - IEMOCAP** ⚠️ Use `train_method2_kfold.py`
4. **Method2 without WPro (WavLM for prosody too) - MSP** ⚠️ Use `train_method2_msp.py`
5. **Method2 WPro AVD (IEMOCAP)** ✅ Already exists: `train_method2_kfold_wpro_avd.py`
6. **Method2 WPro AV Simplified (IEMOCAP)** ✅ Already exists: `train_wpro_simplified_kfold.py`
7. **Method2 WPro AV (Emotion2Vec instead of WavLM+RoBERTa) - IEMOCAP** ✅ Partially: `train_method2_kfold_emotion2vec.py`
8. **HuBERT/WavLM/WPro single modalities (MSP)** ✅ Exist: `train_msp_hubert_av.py`, `train_msp_wavlm_base_av.py`, `train_msp_wpro_av.py`
9. **Method2 WPro AV ASR (IEMOCAP)** ✅ Already exists: `train_method2_kfold_wpro_asr.py`
10. **Method2 WPro AV Fusion (IEMOCAP)** ✅ Already exists: `train_method2_kfold_fusion_concat.py`
11. **Baseline (IEMOCAP)** ✅ Already exists: `train_ssl_baseline_kfold.py`

### Key Points:
- ✅ **Most trainers already exist!**
- ⚠️ For "Method2 without WPro", use the base `train_method2_kfold.py` (uses WavLM for both text and prosody fusion)
- The `train_iemocap_opensmile_emotion_av_kfold.py` is the **new one created** for OpenSMILE emotion + A-V prediction with CCC metric

---

## Next Steps for User

To update evaluation script (`eval_all_models.py`) to handle all 24 models:

1. **Add config loading per model type**
2. **Auto-detect architecture from checkpoint name or config**
3. **Handle different output dimensions:**
   - Some models: [emotion_logits, arousal, valence] (3 outputs)
   - Some models: [emotion_logits, arousal, valence, dominance] (4 outputs)
   - Some models: [emotion_logits] (emotion only - skip A-V evaluation)
4. **Ensure IEMOCAP models test on 5 folds separately**
5. **Ensure MSP models test on test1 and test2 separately**
6. **Generate separate A-V scatter plots per fold/test set**
