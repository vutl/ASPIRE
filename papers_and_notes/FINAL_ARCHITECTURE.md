# Final Architecture Summary - Updated November 4, 2025

## Overview

Cả hai methods đều được cập nhật với:
1. **Deep classification layers** sau final fusion (theo Standard Deep Classifier pattern)
2. **GELU activation** thay vì ReLU (tốt hơn cho transformer-based models)
3. **Progressive dropout** (0.3 → 0.2 → 0.1) cho regularization
4. **Method 2 fixed**: Include arousal/valence embeddings trong final fusion

---

## Method 1: Triple WavLM

### Architecture Flow
```
Audio ──> WavLM ─┬──> Prosody Encoder ──> Fusion(W+Prosody) ──> Arousal branch
                 │                                          └──> arousal_value [B, 1]
                 │
                 └──> Text Encoder ────> Fusion(W+Text) ──> Valence branch
                                                         └──> valence_value [B, 1]
                                      
                 └──> [Arousal, Valence] ──> Refinement ──> AV_refined [B, 256]
                 
                 └──> Final Fusion ──> Deep Classifier ──> Emotion [B, 4]
```

### Final Fusion Details
```python
Input: [B, 1536] = [WavLM:768 + Arousal_emb:256 + Valence_emb:256 + AV_refined:256]

self.final_fusion = nn.Sequential(
    nn.Linear(1536, 512),      # Reduce dimensionality
    nn.LayerNorm(512),
    nn.GELU(),
    nn.Dropout(0.3),
    
    nn.Linear(512, 256),
    nn.LayerNorm(256),
    nn.GELU(),
    nn.Dropout(0.2)
)
# Output: [B, 256]
```

### Emotion Classifier (Deep)
```python
self.emotion_classifier = nn.Sequential(
    nn.Linear(256, 128),       # Further reduction
    nn.LayerNorm(128),
    nn.GELU(),
    nn.Dropout(0.1),
    nn.Linear(128, 4)          # Final output
)
```

### Parameters
- **Total**: 204,088,441 (~204M)
- **Components**:
  - WavLM (used once): ~94M
  - Prosody/Text encoders: Small
  - Final fusion + classifier: ~1-2M

---

## Method 2: Cross-Attention A-V

### Architecture Flow
```
Audio ──> WavLM_prosody ──┐
                          ├──> Bidirectional Cross-Attention
Text ──> RoBERTa ─────────┘     ├──> Prosody attended [B, T, 256]
                                 └──> Text attended [B, T, 256]
                                        │
                                        ├──> Pool ──> Arousal_embedding [B, 256]
                                        │         └──> arousal_value [B, 1]
                                        │
                                        └──> Pool ──> Valence_embedding [B, 256]
                                                  └──> valence_value [B, 1]
                                                  
                                        └──> Refinement ──> AV_refined [B, 256]

Audio ──> WavLM_final ──> Pool ──> WavLM_pooled [B, 768]

Final Fusion:  [WavLM, Arousal_emb, Valence_emb, AV_refined] ──> Deep Classifier ──> Emotion
```

### Final Fusion Details (FIXED)
```python
# TRƯỚC (thiếu information):
Input: [B, 1280] = [WavLM:768 + branch_fused:256 + AV_refined:256]

# SAU (đầy đủ information):
Input: [B, 1536] = [WavLM:768 + Arousal_emb:256 + Valence_emb:256 + AV_refined:256]

self.final_fusion = nn.Sequential(
    nn.Linear(1536, 512),      # Same as Method 1
    nn.LayerNorm(512),
    nn.GELU(),
    nn.Dropout(0.3),
    
    nn.Linear(512, 256),
    nn.LayerNorm(256),
    nn.GELU(),
    nn.Dropout(0.2)
)
# Output: [B, 256]
```

### Emotion Classifier (Deep) - UPDATED
```python
# TRƯỚC (quá đơn giản):
self.emotion_classifier = nn.Linear(256, 4)  # Chỉ 1 layer!

# SAU (deep architecture):
self.emotion_classifier = nn.Sequential(
    nn.Linear(256, 128),       # Thêm hidden layer
    nn.LayerNorm(128),
    nn.GELU(),
    nn.Dropout(0.1),
    nn.Linear(128, 4)          # Final output
)
```

### Parameters
- **Total**: 319,676,809 (~320M)
- **Components**:
  - WavLM_prosody: ~94M
  - RoBERTa: ~125M
  - WavLM_final: ~94M
  - Cross-attention + fusion + classifier: ~7M

---

## Key Improvements

### 1. Deep Classification Layers
**Before**: 1 linear layer (Method 2) hoặc 2 layers (Method 1)
```python
# Old Method 2
emotion_logits = nn.Linear(256, 4)(fused_features)
```

**After**: Standard Deep Classifier cho cả 2 methods
```python
# New: Both methods
fused_features [B, 256]
    ↓
Linear(256 → 128)
LayerNorm(128)
GELU()
Dropout(0.1)
    ↓
Linear(128 → 4)
```

**Benefits**:
- Thêm capacity để học complex decision boundaries
- LayerNorm stabilize training
- Progressive dropout (0.3 → 0.2 → 0.1) prevent overfitting
- GELU activation tốt hơn ReLU cho transformer-based models

### 2. Method 2: Fixed Final Fusion
**Problem**: Mất information từ cross-attention process
```python
# Old: Chỉ dùng branch_fused (sau khi merge 2 branches)
final_representation = torch.cat([
    wavlm_pooled,    # [B, 768]
    branch_fused,    # [B, 256] - đã merge arousal + valence
    av_refined       # [B, 256]
], dim=-1)  # [B, 1280]
```

**Solution**: Giữ lại arousal/valence embeddings riêng biệt
```python
# New: Include cả arousal và valence embeddings
final_representation = torch.cat([
    wavlm_pooled,        # [B, 768] - global acoustic
    arousal_embedding,   # [B, 256] - prosody-attended features
    valence_embedding,   # [B, 256] - text-attended features
    av_refined           # [B, 256] - refined A-V values
], dim=-1)  # [B, 1536]
```

**Benefits**:
- Giữ được rich information từ cả prosody và text branches
- Không bị information loss từ early fusion
- Consistent với Method 1 architecture (cả 2 đều dùng 4 components)

### 3. Activation Function: GELU vs ReLU
**GELU** (Gaussian Error Linear Unit) thường perform tốt hơn ReLU trong:
- Transformer-based architectures (BERT, GPT, etc.)
- Multimodal learning tasks
- Deep networks với many layers

**Comparison**:
```python
# ReLU: f(x) = max(0, x)  - Hard cutoff at 0
# GELU: f(x) = x * Φ(x)   - Smooth, probabilistic gating
```

### 4. Progressive Dropout Strategy
```python
Dropout(0.3)  # Early layers: Strong regularization
    ↓
Dropout(0.2)  # Middle layers: Moderate regularization
    ↓
Dropout(0.1)  # Near output: Light regularization
```

**Rationale**:
- Early layers need strong regularization (prevent overfitting to low-level features)
- Output layers need less dropout (preserve learned representations)

---

## Architecture Comparison

| Aspect | Method 1 | Method 2 |
|--------|----------|----------|
| **Parameters** | 204M | 320M |
| **Text Processing** | Text Encoder (simulated) | RoBERTa (real) |
| **Cross-Attention** | No | Yes (Bidirectional) |
| **WavLM Usage** | 1x | 2x (prosody + final) |
| **Final Fusion Input** | 1536 dim | 1536 dim |
| **Classifier Depth** | 2 layers | 2 layers |
| **Activation** | GELU | GELU |
| **Dropout** | Progressive | Progressive |

---

## Training Configuration

### Loss Function (Both Methods)
```python
CombinedEmotionLossCCC:
    L_total = 1.0 * L_emotion (CrossEntropy)
            + 1.0 * L_arousal_CCC
            + 1.0 * L_valence_CCC
```

### Hyperparameters
```yaml
batch_size: 2  # Due to GPU memory constraints
learning_rate: 1e-4
weight_decay: 1e-5
optimizer: AdamW
scheduler: ReduceLROnPlateau (patience=5, factor=0.5)
epochs: 50
gradient_clipping: 1.0
```

### Data Augmentation (Recommended)
- SpecAugment for audio
- Random text masking (for Method 2)
- Mixup/CutMix for regularization

---

## Expected Improvements

### From Deep Classifier
- **+2-3% accuracy** from additional learning capacity
- **Better generalization** from progressive dropout
- **Stable training** from LayerNorm

### From Method 2 Fix
- **+1-2% accuracy** from preserving cross-attention information
- **Better A-V prediction** from richer features
- **Improved emotion recognition** from multimodal fusion

---

## Next Steps

1. **Train cả 2 methods** với updated architectures
2. **Compare performance**:
   - Accuracy, F1-score (emotion classification)
   - CCC scores (arousal/valence regression)
   - Inference time, memory usage
3. **Ablation studies**:
   - Deep classifier vs simple classifier
   - Method 2: với/không có arousal_embedding trong final fusion
   - GELU vs ReLU activation
4. **Ensemble methods**:
   - Average predictions từ cả 2 methods
   - Stacking với meta-learner

---

## GPU Status

- **GPU 0**: Method 2 đang train (32GB used)
- **GPU 1**: 29GB used
- **GPU 2**: 30GB used  
- **GPU 3**: Method 1 sẽ train (1.3GB used - AVAILABLE)

---

## References

1. **Standard Deep Classifier**: Proven architecture in SER literature
2. **GELU Activation**: Hendrycks & Gimpel, 2016
3. **Progressive Dropout**: Srivastava et al., 2014
4. **Cross-Attention**: Vaswani et al., 2017 (Attention Is All You Need)
5. **CCC Loss**: Lin, 1989 (Concordance Correlation Coefficient)
