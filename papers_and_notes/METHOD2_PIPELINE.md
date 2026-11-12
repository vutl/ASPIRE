# Method 2: Cross-Attention A-V Architecture - Detailed Pipeline

## Architecture Overview

**3 Models (2 configurations available):**

### Standard Configuration (method2_config.yaml)
1. **WavLM_prosody** - 94M params - WavLM-base-plus for acoustic/prosody features
2. **RoBERTa** - 125M params - Extract semantic/linguistic features from text transcripts
3. **WavLM_final** - 94M params - Extract global acoustic features for final fusion
**Total Parameters: ~319M**

### W-PRO Configuration (method2_wpro_config.yaml) ⭐ **RECOMMENDED**
1. **W-PRO (WavLM-Pro)** - ~300M params - **Specialized non-timbral prosody extractor** (Orange/Speaker-wavLM-pro)
2. **RoBERTa** - 125M params - Extract semantic/linguistic features from text transcripts
3. **WavLM_final** - 94M params - Extract global acoustic features for final fusion
**Total Parameters: ~519M** (backbones frozen, only ~6.3M trainable)

**Why W-PRO?** Explicitly trained to extract non-timbral features (prosody, accent, rhythm) → better arousal prediction & interpretability

---

## Detailed Forward Pass

### Input Shape
- **Audio**: `[B, 160000]` - Raw waveform (10 seconds at 16kHz)
- **Text tokens**: `[B, 128]` - Tokenized transcript (RoBERTa tokens)
- **Text mask**: `[B, 128]` - Attention mask for text

---

### Layer-by-Layer Flow

#### **Step 1: Prosody Branch (Wpro)**

**Standard WavLM:**
```
Input: Audio [B, 160000]
  ↓ self.wavlm_prosody (WavLM-base-plus CNN + Transformer)
  │   - CNN feature extractor: [B, 160000] → [B, 499, 768]
  │   - 12 Transformer layers
Wpro features [B, 499, 768]
  ↓ self.prosody_projection (Linear)
Wpro projected [B, 499, 256]
```

**W-PRO (Recommended):**
```
Input: Audio [B, 160000]
  ↓ self.wavlm_prosody (W-PRO - WavLM-large based + fine-tuned)
  │   - Specialized non-timbral feature extraction
  │   - Outputs prosody, accent, rhythm (NOT speaker identity)
Wpro features [B, T, hidden_size]  # hidden_size depends on W-PRO config
  ↓ self.prosody_projection (Linear)
Wpro projected [B, T, 256]
```

#### **Step 2: Text Branch (RoBERTa)**
```
Input: Text tokens [B, 128]
  ↓ self.roberta (RoBERTa embedding + Transformer)
  │   - Token embedding + positional encoding
  │   - 12 Transformer layers
Text features [B, 128, 768]
  ↓ self.text_projection (Linear)
Text projected [B, 128, 256]
```

#### **Step 3: Bidirectional Cross-Attention**
```
Wpro projected [B, 499, 256] ─┐
                               ├─→ BiDirectionalCrossAttention
Text projected [B, 128, 256] ─┘
  │
  ├─→ Wpro attends to Text: MultiHeadAttention(Q=Wpro, K=Text, V=Text)
  │   Output: Wpro attended [B, 499, 256]
  │
  └─→ Text attends to Wpro: MultiHeadAttention(Q=Text, K=Wpro, V=Wpro)
      Output: Text attended [B, 128, 256]
```

#### **Step 4: Post-Attention Processing**

**Prosody Branch:**
```
Wpro attended [B, 499, 256]
  ↓ self.prosody_post_attention (2 Transformer layers)
  │   - Layer 1: Self-Attention + FFN + Residual + LayerNorm
  │   - Layer 2: Self-Attention + FFN + Residual + LayerNorm
Wpro refined [B, 499, 256]
  ↓ self.prosody_pooling (Self-Attention Pooling)
  │   - Attention weights: softmax(Linear(x))
  │   - Weighted sum over time dimension
Wpro pooled [B, 256]
```

**Text Branch:**
```
Text attended [B, 128, 256]
  ↓ self.text_post_attention (2 Transformer layers)
  │   - Layer 1: Self-Attention + FFN + Residual + LayerNorm
  │   - Layer 2: Self-Attention + FFN + Residual + LayerNorm
Text refined [B, 128, 256]
  ↓ self.text_pooling (Self-Attention Pooling)
Text pooled [B, 256]
```

#### **Step 5: Branch Fusion**
```
Wpro pooled [B, 256] ─┐
                      ├─→ Gated Fusion Layer
Text pooled [B, 256] ─┘
  │
  ├─→ Gate = sigmoid(Linear([Wpro; Text]))  # [B, 512] → [B, 256]
  └─→ Branch fused = Gate ⊙ Wpro + (1 - Gate) ⊙ Text
  
Branch fused [B, 256]
```

#### **Step 6: A-V Prediction**
```
Branch fused [B, 256]
  ↓ self.av_prediction_layers (3 MLP layers)
  │   - Layer 1: Linear(256→256) + LayerNorm + ReLU + Dropout(0.1)
  │   - Layer 2: Linear(256→256) + LayerNorm + ReLU + Dropout(0.1)
  │   - Layer 3: Linear(256→256) + LayerNorm + ReLU + Dropout(0.1)
AV features [B, 256]
  ├─→ self.arousal_head (Linear 256→1)
  │   Arousal value [B, 1]
  │
  └─→ self.valence_head (Linear 256→1)
      Valence value [B, 1]
```

#### **Step 7: A-V Refinement**
```
[Arousal, Valence] [B, 2]
  ↓ self.av_refinement (AVRefinementModule - 2 MLP layers)
  │   Input projection: Linear(2→256)
  │   - Layer 1: Linear(256→256) + ReLU + Dropout
  │   - Layer 2: Linear(256→256) + ReLU + Dropout
  │   + Residual connection
AV refined [B, 256]
```

#### **Step 8: Global WavLM Branch**
```
Input: Audio [B, 160000]
  ↓ self.wavlm_final (WavLM CNN + Transformer)
WavLM features [B, 499, 768]
  ↓ self.wavlm_pooling (Self-Attention Pooling)
WavLM pooled [B, 768]
```

#### **Step 9: Final Fusion → Emotion**
```
Concatenate:
  WavLM pooled [B, 768] +
  Branch fused [B, 256] +
  AV refined   [B, 256]
  ↓
Final concat [B, 1280]
  ↓ self.final_fusion (2 MLP layers)
  │   - Layer 1: Linear(1280→512) + LayerNorm + ReLU + Dropout(0.2)
  │   - Layer 2: Linear(512→256) + LayerNorm + ReLU + Dropout(0.2)
Final features [B, 256]
  ↓ self.emotion_classifier (Linear 256→4)
Emotion logits [B, 4]
```

---

## Loss Function: CombinedEmotionLossCCC

### Formula
```
L_total = α * L_emotion + β * L_arousal_CCC + γ * L_valence_CCC

where α = β = γ = 1.0 (default)
```

### Component Losses

#### 1. Emotion Classification Loss
```python
L_emotion = CrossEntropyLoss(emotion_logits, emotion_label)

Input:
  emotion_logits: [B, 4]  # Logits for 4 classes (hap, sad, neu, ang)
  emotion_label:  [B]     # Integer labels

Output: scalar loss
```

#### 2. Arousal CCC Loss
```python
L_arousal_CCC = 1 - CCC(arousal_pred, arousal_true)

CCC = (2 * covariance) / (var_pred + var_true + (mean_pred - mean_true)²)

where:
  mean_pred = mean(arousal_pred)
  mean_true = mean(arousal_true)
  var_pred  = variance(arousal_pred)
  var_true  = variance(arousal_true)
  covariance = mean((arousal_pred - mean_pred) * (arousal_true - mean_true))

Input:
  arousal_pred: [B, 1]  # Predicted arousal values [-1, 1]
  arousal_true: [B, 1]  # Ground truth arousal values [-1, 1]

Output: scalar loss (range: [0, 2], lower is better)
```

#### 3. Valence CCC Loss
```python
L_valence_CCC = 1 - CCC(valence_pred, valence_true)

Same formula as arousal CCC loss

Input:
  valence_pred: [B, 1]  # Predicted valence values [-1, 1]
  valence_true: [B, 1]  # Ground truth valence values [-1, 1]

Output: scalar loss (range: [0, 2], lower is better)
```

### CCC (Concordance Correlation Coefficient)
- **Range**: [-1, 1]
- **Perfect agreement**: CCC = 1
- **No agreement**: CCC = 0
- **Properties**:
  - Measures both precision (correlation) and accuracy (bias)
  - More suitable than Pearson correlation for continuous emotion
  - Penalizes both variance difference and mean difference

---

## Output Dictionary
```python
{
  'emotion_logits':   [B, 4],      # Classification logits
  'arousal_value':    [B, 1],      # Predicted arousal [-1, 1]
  'valence_value':    [B, 1],      # Predicted valence [-1, 1]
  'branch_fused':     [B, 256],    # Fusion of Wpro + Text
  'av_refined':       [B, 256],    # Refined A-V features
  'av_features':      [B, 256],    # Features before A-V prediction
  'wpro_attended':    [B, 499, 256], # For analysis
  'text_attended':    [B, 128, 256]  # For analysis
}
```

---

## Training Configuration

### 5-Fold Cross-Validation
```
Fold 1: Test Session 1, Train Sessions 2,3,4,5 (80% train, 20% val)
Fold 2: Test Session 2, Train Sessions 1,3,4,5 (80% train, 20% val)
Fold 3: Test Session 3, Train Sessions 1,2,4,5 (80% train, 20% val)
Fold 4: Test Session 4, Train Sessions 1,2,3,5 (80% train, 20% val)
Fold 5: Test Session 5, Train Sessions 1,2,3,4 (80% train, 20% val)

Final accuracy = Average of 5 test accuracies
```

### Hyperparameters
```yaml
model:
  num_emotions: 4
  wavlm_model_name: microsoft/wavlm-base-plus
  roberta_model_name: roberta-base
  hidden_dim: 256
  av_refinement_layers: 2
  freeze_wavlm: false
  freeze_roberta: false
  num_cross_attn_heads: 8
  post_attention_layers: 2
  av_prediction_layers: 3

training:
  batch_size: 8
  epochs: 50
  learning_rate: 1e-4
  weight_decay: 1e-5
  num_workers: 4

loss:
  emotion_weight: 1.0
  arousal_weight: 1.0
  valence_weight: 1.0

data:
  max_audio_length: 160000  # 10 seconds
  max_text_length: 128
  sample_rate: 16000
```

### Metrics Logged
- **Training**: Loss, Emotion Loss, Arousal Loss, Valence Loss, Accuracy, F1 Macro, F1 Weighted
- **Validation**: Same as training + Arousal CCC, Valence CCC, Avg CCC
- **Test**: Same as validation

---

## Computational Requirements

- **Parameters**: ~319M
- **FLOPs**: ~XXX GFLOPs (computed during training)
- **Memory**: ~8-12 GB GPU memory for batch_size=8
- **Training time**: ~X hours per fold on V100

---

## Usage

### Train from scratch (Standard WavLM):
```bash
bash scripts/run_train_kfold.sh cuda:0
```

### Train from scratch (W-PRO - Recommended):
```bash
bash scripts/run_train_kfold_wpro.sh cuda:0
```

### Resume from checkpoint:
```bash
# Standard WavLM
bash scripts/run_train_kfold.sh cuda:0 3 checkpoints/method2_kfold/fold_3/best_model.pt

# W-PRO
bash scripts/run_train_kfold_wpro.sh cuda:0 3 checkpoints/method2_kfold_wpro/fold_3/best_model.pt
```

### Monitor training:
```bash
# View logs
tail -f logs/train_method2_kfold_TIMESTAMP.log

# TensorBoard
tensorboard --logdir checkpoints/method2_kfold/logs --port 6006
```

## AVD Variant + W-PRO (Method2 AVD with WavLM-Pro)

**Status / confirmation:**
- Training entry point: `train/train_method2_kfold_wpro.py` (wrapper: `scripts/run_train_kfold_wpro.sh`).
- Model: `models/method2_crossattn_av.py` is used; when `wavlm_prosody_model_name` is set (e.g. `Orange/Speaker-wavLM-pro`) the code loads W-PRO and sets `self.use_wpro = True`.
- The trainer prints the same key diagnostics as the standard Method2 trainer: model statistics (total/trainable params), FLOPs, architecture summary, loss configuration and per-epoch logs.
- Checkpoint logic: per-fold checkpoints are written to `checkpoints/method2_kfold_wpro/fold_<i>/best_model.pt`. If resuming, the trainer will load a provided resume checkpoint and fall back to it for the final test if no new best model was saved during the resumed run.

**Behavioral notes:**
- W-PRO is loaded via Hugging Face (`WavLMModel.from_pretrained`) and projected into the model pipeline; `freeze_wavlm` in the config controls whether W-PRO (and final WavLM) weights are trainable.
- The training script uses the same fold loop semantics as other Method2 trainers: `--resume_fold` controls which fold index to start from; the loop runs from that fold to the end (3→5 if `--resume_fold 3`).
- The script saves `best_model.pt` when validation accuracy improves, and later loads the best file for test or falls back to the resume checkpoint if none was saved.

**Recommended run command (maps a physical GPU to the process-visible `cuda:0`):**
```bash
# Example: run on physical GPU 2
CUDA_VISIBLE_DEVICES=2 bash scripts/run_train_kfold_wpro.sh cuda:0
```

This will start training with W-PRO enabled (if `configs/method2_wpro_config.yaml` sets `wavlm_prosody_model_name`) and will create logs in `logs/` and checkpoints under `checkpoints/method2_kfold_wpro/`.

---
