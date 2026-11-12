# Quy Trình Xử Lý Dữ Liệu IEMOCAP và MSP-Podcast cho Method2_WPRO

## Tổng Quan

Method2_WPRO là một kiến trúc multimodal cho nhận dạng cảm xúc, kết hợp:
- **Audio**: WavLM-base-plus (acoustic features) + W-PRO/WavLM-pro (prosody features)
- **Text**: RoBERTa (semantic features từ transcript)
- **Cross-attention**: Fusion giữa prosody và text
- **Multi-task**: Phân loại cảm xúc (4 lớp) + dự đoán Arousal-Valence (A-V regression)

---

## 1. QUY TRÌNH XỬ LÝ IEMOCAP

### 1.1. Chuẩn Bị Dữ Liệu Gốc

**Input:** Dataset IEMOCAP gốc với cấu trúc:
```
IEMOCAP_full_release/
├── Session1/
│   ├── sentences/wav/  # Audio files
│   ├── dialog/EmoEvaluation/  # Annotation files
├── Session2/
├── Session3/
├── Session4/
└── Session5/
```

**Metadata gốc:** File CSV chứa:
- `sample_name`: Tên file (ví dụ: Ses01F_script02_1_F000)
- `path`: Đường dẫn đầy đủ tới file audio .wav
- `start_time`, `end_time`: Timestamp
- `emotion`: Nhãn cảm xúc (hap, sad, neu, ang, exc, fru, ...)
- `arousal`, `valence`, `dominance`: Giá trị A-V-D (scale 1-5)
- `transcript`: Transcript text ground-truth

### 1.2. Xử Lý và Gộp Nhãn Cảm Xúc

**Script:** `datasets/IEMOCAP_KFOLD/prepare_csv.py`

**Các bước:**

1. **Đọc file CSV gốc:**
   ```python
   df = pd.read_csv("/home4/quanpn/IEMOCAP/IEMOCAP.csv")
   ```

2. **Gộp nhãn cảm xúc:**
   - Gộp `exc` (excited) → `hap` (happy) vì có tính chất tương đồng
   - Giữ lại **4 lớp cảm xúc chính:**
     - `hap` (happy/excited) → label 0
     - `sad` (sadness) → label 1
     - `neu` (neutral) → label 2
     - `ang` (anger) → label 3
   
   ```python
   df['emotion'] = df['emotion'].replace('exc', 'hap')
   keep_emotions = ['neu', 'ang', 'sad', 'hap']
   df = df[df['emotion'].isin(keep_emotions)].reset_index(drop=True)
   ```

3. **Xác định fold theo speaker:**
   - IEMOCAP có 10 speakers (5 sessions × 2 genders):
     - Session 1: Ses01F, Ses01M → fold 1, 2
     - Session 2: Ses02F, Ses02M → fold 3, 4
     - Session 3: Ses03F, Ses03M → fold 5, 6
     - Session 4: Ses04F, Ses04M → fold 7, 8
     - Session 5: Ses05F, Ses05M → fold 9, 10
   
   ```python
   def extract_speaker(x):
       m = re.match(r"(Ses\d+[MF])", x)
       return m.group(1) if m else None
   
   df['fold'] = df['sample_name'].apply(extract_speaker)
   unique_folds = sorted(df['fold'].unique())
   fold_map = {spk: i + 1 for i, spk in enumerate(unique_folds)}
   df['fold'] = df['fold'].map(fold_map)
   ```

**Output:** `datasets/IEMOCAP_KFOLD/tenfolds/all.csv` (5,533 samples)

**Cấu trúc CSV:**
```csv
sample_name,start_time,end_time,emotion,emotion_id,arousal,valence,dominance,transcript,path,fold
Ses01F_script02_1_F000,15.14,17.21,neu,1,2.0,2.5,2.0,Fine.,/path/to/audio.wav,1
```

### 1.3. Chia K-Fold Cross-Validation (5 folds)

**Logic:** Speaker-independent splits
- Mỗi fold test sử dụng 1 session (2 speakers)
- 4 sessions còn lại cho train/val

**Mapping 10 folds → 5 session folds:**
```python
fold_to_session = {
    1: 1, 2: 1,  # Ses01F, Ses01M → Session 1
    3: 2, 4: 2,  # Ses02F, Ses02M → Session 2
    5: 3, 6: 3,  # Ses03F, Ses03M → Session 3
    7: 4, 8: 4,  # Ses04F, Ses04M → Session 4
    9: 5, 10: 5  # Ses05F, Ses05M → Session 5
}
```

**Chia train/val trong mỗi fold:**
- Test: 1 session (~20%)
- Train: 4 sessions remaining (~80%)
- Val: Random split từ train (ví dụ: 10-20% của train)

**Thực hiện trong code:** `train/train_method2_kfold.py` → function `prepare_folds()`

### 1.4. Trích Xuất ASR Text (Optional - cho thí nghiệm ASR)

**Script:** `scripts/extract_asr_text_whisper.py`

**Sử dụng:** Whisper model (base) để trích xuất transcript tự động

**Các bước:**
1. Load Whisper model:
   ```python
   model = whisper.load_model("base", device="cuda")
   ```

2. Transcribe từng audio file:
   ```python
   result = model.transcribe(audio_path, language="en", fp16=True)
   text = result["text"].strip()
   ```

3. Lưu vào CSV mới với cột `asr_transcript`

**Output:** `datasets/IEMOCAP_KFOLD/tenfolds/all_asr.csv`

**Mục đích:** So sánh hiệu suất giữa ground-truth transcript vs ASR transcript

---

## 2. QUY TRÌNH XỬ LÝ MSP-PODCAST

### 2.1. Cấu Trúc Dataset Gốc

**Thư mục:** `datasets/MSP-Podcast/`

```
MSP-Podcast/
├── Audios/              # Audio files (.wav)
│   └── MSP-PODCAST_0001_0008.wav
├── Transcripts/         # Text transcripts (.txt)
│   └── MSP-PODCAST_0001_0008.txt
├── Labels/
│   └── labels_consensus.csv  # A-V-D values
├── msp_podcast.tsv      # Metadata chính
├── Partitions.txt       # Train/dev/test splits
└── Speaker_id.txt       # Speaker information
```

### 2.2. File Metadata Chính: `msp_podcast.tsv`

**Cấu trúc:**
```tsv
utt_id	path	label	split	speaker	lang	duration	fold
MSP-PODCAST_0001_0008	/path/to/audio.wav	neu	test1	30	en	10.488	3
```

**Các trường:**
- `utt_id`: Unique ID (dùng để match với Transcripts/)
- `path`: Đường dẫn audio file
- `label`: Nhãn cảm xúc (ang, sad, hap, neu, ...)
- `split`: train/dev/test1/test2 (partition sẵn của dataset)
- `speaker`: Speaker ID
- `lang`: Ngôn ngữ (en = English)
- `duration`: Độ dài audio (giây)
- `fold`: Fold number (cho cross-validation nếu cần)

**Tổng số samples:** ~212,810 utterances

### 2.3. Lọc Dữ Liệu 4 Lớp Cảm Xúc

**Trong code:** `data/msp_podcast_dataset.py`

**Lọc:**
```python
if four_class_only:
    df = df[df['label'].isin(['ang', 'sad', 'hap', 'neu'])].reset_index(drop=True)
```

**Mapping nhãn (UNIFIED với IEMOCAP):**
```python
emotion_map = {
    'hap': 0,  # Happiness
    'sad': 1,  # Sadness
    'neu': 2,  # Neutral
    'ang': 3   # Anger
}
```

### 2.4. Load Arousal-Valence Values

**File:** `Labels/labels_consensus.csv`

**Cấu trúc:**
```csv
FileName,EmoAct,EmoVal,EmoDom
MSP-PODCAST_0001_0008.wav,3.5,2.8,3.2
```

**Trong code:**
```python
labels_csv = os.path.join(base_dir, 'Labels', 'labels_consensus.csv')
labels_df = pd.read_csv(labels_csv)

# Create lookup dict
av_dict = {}
for _, row in labels_df.iterrows():
    filename = row['FileName']
    av_dict[filename] = {
        'arousal': row['EmoAct'],
        'valence': row['EmoVal'],
        'dominance': row['EmoDom']
    }
```

### 2.5. Load Transcript Text

**Thư mục:** `Transcripts/`

**Format:** Mỗi file .txt tương ứng với `utt_id`

**Ví dụ:**
- Audio: `MSP-PODCAST_0001_0008.wav`
- Transcript: `Transcripts/MSP-PODCAST_0001_0008.txt`

**Trong code:**
```python
transcript_file = os.path.join(transcript_dir, f'{utt_id}.txt')
if os.path.exists(transcript_file):
    with open(transcript_file, 'r', encoding='utf-8') as f:
        transcript = f.read().strip()
else:
    transcript = ""  # Fallback nếu không có transcript
```

### 2.6. Normalize A-V Values

MSP-Podcast sử dụng **z-score normalization** (khác IEMOCAP):

**Tính mean/std từ toàn bộ dataset:**
```python
arousal_values = labels_df['EmoAct'].values
valence_values = labels_df['EmoVal'].values

arousal_mean = np.mean(arousal_values)
arousal_std = np.std(arousal_values)
valence_mean = np.mean(valence_values)
valence_std = np.std(valence_values)
```

**Normalize:**
```python
arousal_value = (arousal_value - arousal_mean) / (arousal_std + 1e-8)
valence_value = (valence_value - valence_mean) / (valence_std + 1e-8)
```

**Note:** IEMOCAP sử dụng linear scaling từ [1,5] → [-1,1]:
```python
# IEMOCAP normalization
arousal = (arousal - 3.0) / 2.0  # [1, 5] → [-1, 1]
```

### 2.7. Trích Xuất ASR Text (Optional)

**Script:** `scripts/extract_asr_msp_podcast.py`

Tương tự IEMOCAP, sử dụng Whisper để tạo ASR transcripts:

```python
model = whisper.load_model("base", device="cuda")
result = model.transcribe(audio_path, language="en", fp16=True)
```

**Output:** `msp_podcast_asr.tsv` (với cột `asr_transcript` thêm vào)

---

## 3. DATASET CLASS IMPLEMENTATION

### 3.1. IEMOCAP Dataset Class

**File:** `data/iemocap_dataset.py`

**Class:** `IEMOCAPDataset`

**Key features:**

1. **Load audio:**
   - Resample to 16kHz
   - Convert to mono
   - Pad/truncate to `audio_max_length` (160,000 samples = 10s)

2. **Process text:**
   - Tokenize bằng RoBERTa tokenizer
   - Max length: 128 tokens
   - Padding: 'max_length'

3. **Normalize A-V:**
   - Linear scaling: [1, 5] → [-1, 1]
   ```python
   arousal = (arousal - 3.0) / 2.0
   valence = (valence - 3.0) / 2.0
   ```

4. **Return dict:**
   ```python
   {
       'audio': waveform,              # [T] = 160000
       'emotion_label': emotion_label,  # int (0-3)
       'arousal_value': arousal,        # float [-1, 1]
       'valence_value': valence,        # float [-1, 1]
       'dominance_value': dominance,    # float [-1, 1]
       'text_input_ids': text_input_ids,        # [128]
       'text_attention_mask': text_attention_mask,  # [128]
       'transcript': transcript,        # string
       'sample_name': sample_name,
       'audio_path': audio_path
   }
   ```

### 3.2. MSP-Podcast Dataset Class

**File:** `data/msp_podcast_dataset.py`

**Class:** `MSPPodcastDataset`

**Key features:**

1. **Load from TSV:**
   ```python
   df = pd.read_csv(tsv_path, sep='\t')
   df = df[df['split'] == split]  # train/dev/test1/test2
   ```

2. **Filter 4 emotions:**
   ```python
   df = df[df['label'].isin(['ang', 'sad', 'hap', 'neu'])]
   ```

3. **Load A-V từ labels_consensus.csv:**
   ```python
   audio_filename = os.path.basename(audio_path)
   arousal_value = av_dict[audio_filename]['arousal']
   valence_value = av_dict[audio_filename]['valence']
   ```

4. **Load transcript từ Transcripts/:**
   ```python
   transcript_file = f'Transcripts/{utt_id}.txt'
   ```

5. **Normalize A-V (z-score):**
   ```python
   arousal = (arousal - arousal_mean) / arousal_std
   valence = (valence - valence_mean) / valence_std
   ```

6. **Return dict:** Tương tự IEMOCAP

---

## 4. MODEL ARCHITECTURE: METHOD2_WPRO

**File:** `models/method2_crossattn_av.py`

**Class:** `Method2CrossAttentionAV`

### 4.1. Architecture Flow

```
Audio ──────────────────────────────────────────┐
    │                                          │
    ├─> WavLM-Pro (Prosody) ─┐                │
    │   [freeze]             │                │
    │                        │                │
    └─> WavLM-base-plus ─────┼────────────────┤
        [freeze]             │                │
                             │                │
Text ──> RoBERTa ────────────┤                │
         [freeze]            │                │
                             │                │
                             ↓                │
                    Cross-Attention           │
                        (BiDir)               │
                             │                │
                             ↓                │
                    Post-Attention Layers     │
                    (Transformer Encoder)     │
                             │                │
                   ┌─────────┴─────────┐      │
                   │                   │      │
            Prosody Branch      Text Branch   │
                   │                   │      │
                   └─────────┬─────────┘      │
                             ↓                │
                      Fusion Layer            │
                             │                │
                             ↓                │
                    A-V Prediction Layers     │
                             │                │
                             ↓                │
                    A-V Values (continuous)   │
                             │                │
                             ↓                │
                   A-V Refinement Module      │
                             │                │
                             ↓                │
                      AV_refined_emb          │
                             │                │
                             └────────┬───────┘
                                      │
                                      ↓
                              Final Fusion
                         (WavLM + Fusion + AV)
                                      │
                                      ↓
                            Emotion Classification
                                (4 classes)
```

### 4.2. Key Components

1. **WavLM-Pro (Prosody Branch):**
   - Model: `Orange/Speaker-wavLM-pro`
   - Purpose: Extract prosody features (non-timbral)
   - Frozen: Yes
   - Output dim: 768 → project to `hidden_dim` (256)

2. **RoBERTa (Text Branch):**
   - Model: `roberta-base`
   - Purpose: Extract semantic features from text
   - Frozen: Yes
   - Output dim: 768 → project to `hidden_dim` (256)

3. **Cross-Attention:**
   - BiDirectional: Prosody ↔ Text
   - Heads: 8
   - Dim: 256

4. **Post-Attention Layers:**
   - Transformer Encoder layers
   - Num layers: 2
   - FFN dim: 256 × 4 = 1024

5. **A-V Prediction:**
   - Input: Fused prosody + text features
   - Output: Arousal, Valence (continuous)
   - Layers: 3 MLP layers

6. **A-V Refinement:**
   - Purpose: Refine A-V predictions
   - Layers: 2 (configurable)
   - Output: `AV_refined_emb` (embedded A-V)

7. **WavLM Final Branch:**
   - Model: `microsoft/wavlm-base-plus`
   - Purpose: Global acoustic features
   - Frozen: Yes

8. **Final Fusion:**
   - Concatenate: WavLM + branch_fusion + AV_refined
   - Output: Emotion classification (4 classes)

### 4.3. Trainable vs Frozen

**Frozen (Feature Extraction Only):**
- WavLM-Pro prosody encoder (~94M params)
- WavLM-base-plus acoustic encoder (~94M params)
- RoBERTa text encoder (~125M params)
- **Total frozen:** ~313M params

**Trainable (Task-Specific):**
- Projection layers
- Cross-attention modules
- Post-attention layers
- Fusion layers
- A-V prediction heads
- A-V refinement module
- Emotion classification head
- **Total trainable:** ~6.3M params

---

## 5. TRAINING CONFIGURATION

### 5.1. IEMOCAP K-Fold Config

**File:** `configs/method2_wpro_config.yaml`

```yaml
# Data
data:
  csv: "datasets/IEMOCAP_KFOLD/tenfolds/all.csv"
  max_audio_length: 160000  # 10s at 16kHz
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
  freeze_wavlm: true
  freeze_roberta: true
  num_cross_attn_heads: 8
  post_attention_layers: 2
  av_prediction_layers: 3

# Training
training:
  batch_size: 8
  epochs: 30
  learning_rate: 0.0001
  weight_decay: 0.00001
  num_workers: 2

# Loss
loss:
  type: "ccc"  # Concordance Correlation Coefficient
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
  patience: 5
  min_lr: 1.0e-06

# Paths
paths:
  checkpoint_dir: "checkpoints/method2_kfold_wpro"
```

### 5.2. MSP-Podcast Config

**File:** `configs/msp_podcast_wpro_config.yaml`

```yaml
# Data
data:
  tsv_path: "datasets/MSP-Podcast/msp_podcast.tsv"
  max_audio_length: 160000
  max_text_length: 128
  sample_rate: 16000

# Model: Same as IEMOCAP

# Training
training:
  batch_size: 8
  epochs: 10  # MSP-Podcast lớn hơn, ít epoch hơn
  learning_rate: 0.0001
  weight_decay: 0.00001
  num_workers: 4

# Paths
paths:
  checkpoint_dir: "checkpoints/method2_msp_wpro"
```

---

## 6. LOSS FUNCTION

**File:** `train/losses.py`

**Class:** `CombinedEmotionLossCCC`

### Multi-Task Loss

**Công thức:**
```
Total Loss = λ_emotion * L_emotion + λ_arousal * L_arousal + λ_valence * L_valence
```

**Components:**

1. **Emotion Loss (Cross-Entropy):**
   ```python
   L_emotion = CrossEntropyLoss(logits, labels)
   ```

2. **Arousal Loss (CCC):**
   - Concordance Correlation Coefficient (CCC) loss
   ```python
   CCC = (2 * ρ * σ_x * σ_y) / (σ_x² + σ_y² + (μ_x - μ_y)²)
   L_arousal = 1 - CCC
   ```

3. **Valence Loss (CCC):**
   ```python
   L_valence = 1 - CCC
   ```

**Weights:**
- `emotion_weight = 1.0`
- `arousal_weight = 1.0`
- `valence_weight = 1.0`

**Why CCC?**
- CCC đo lường agreement giữa predicted và ground-truth
- Tốt hơn MSE cho regression vì xét cả bias và scale
- Range: [-1, 1], higher is better

---

## 7. TRAINING PIPELINE

### 7.1. K-Fold Training (IEMOCAP)

**Script:** `train/train_method2_kfold.py`

**Flow:**

1. **Load config:**
   ```python
   config = load_config('configs/method2_wpro_config.yaml')
   ```

2. **Prepare 5 folds:**
   ```python
   folds = prepare_folds(csv_path)
   # Fold 1: Test Session1, Train Sessions 2,3,4,5
   # ...
   ```

3. **For each fold:**
   
   a. **Create datasets:**
   ```python
   train_df = fold['train_val']
   test_df = fold['test']
   
   # Split train_df into train/val (80/20)
   train_df, val_df = train_test_split(train_df, test_size=0.2, random_state=42)
   
   train_dataset = IEMOCAPDataset(train_df)
   val_dataset = IEMOCAPDataset(val_df)
   test_dataset = IEMOCAPDataset(test_df)
   ```
   
   b. **Create model:**
   ```python
   model = Method2CrossAttentionAV(**config['model'])
   ```
   
   c. **Train epochs:**
   ```python
   for epoch in range(epochs):
       train_one_epoch(model, train_loader, criterion, optimizer, device, epoch)
       val_metrics = validate(model, val_loader, criterion, device, epoch)
       
       # Save best model based on val_wa (weighted accuracy)
       if val_metrics['wa'] > best_wa:
           torch.save(model.state_dict(), f'best_fold{fold_idx}.pt')
   ```
   
   d. **Test:**
   ```python
   test_metrics = validate(model, test_loader, criterion, device, epoch, phase='Test')
   ```

4. **Aggregate results:**
   ```python
   avg_wa = np.mean([fold_results[i]['wa'] for i in range(5)])
   avg_uar = np.mean([fold_results[i]['uar'] for i in range(5)])
   ```

### 7.2. MSP-Podcast Training

**Script:** `train/train_method2_msp_wpro.py`

**Flow:**

1. **Load config:**
   ```python
   config = load_config('configs/msp_podcast_wpro_config.yaml')
   ```

2. **Create datasets:**
   ```python
   train_dataset = MSPPodcastDataset(tsv_path, split='train')
   dev_dataset = MSPPodcastDataset(tsv_path, split='dev')
   test1_dataset = MSPPodcastDataset(tsv_path, split='test1')
   test2_dataset = MSPPodcastDataset(tsv_path, split='test2')
   ```

3. **Train:**
   ```python
   for epoch in range(epochs):
       train_one_epoch(model, train_loader, criterion, optimizer, device, epoch)
       dev_metrics = validate(model, dev_loader, criterion, device, epoch, phase='Dev')
       
       # Save best model based on dev performance
       if dev_metrics['wa'] > best_wa:
           torch.save(model.state_dict(), 'best_msp.pt')
   ```

4. **Test:**
   ```python
   test1_metrics = validate(model, test1_loader, criterion, device, phase='Test1')
   test2_metrics = validate(model, test2_loader, criterion, device, phase='Test2')
   ```

---

## 8. EVALUATION METRICS

### 8.1. Emotion Classification Metrics

1. **Weighted Accuracy (WA):**
   ```python
   wa = accuracy_score(y_true, y_pred)
   ```

2. **Unweighted Average Recall (UAR):**
   ```python
   uar = recall_score(y_true, y_pred, average='macro')
   ```

3. **F1-Score (per class):**
   ```python
   f1 = f1_score(y_true, y_pred, average=None)
   ```

4. **Confusion Matrix**

### 8.2. A-V Regression Metrics

1. **CCC (Concordance Correlation Coefficient):**
   ```python
   ccc_arousal = compute_ccc(arousal_true, arousal_pred)
   ccc_valence = compute_ccc(valence_true, valence_pred)
   ```

2. **Pearson Correlation:**
   ```python
   r_arousal, _ = pearsonr(arousal_true, arousal_pred)
   r_valence, _ = pearsonr(valence_true, valence_pred)
   ```

3. **MSE (Mean Squared Error):**
   ```python
   mse_arousal = np.mean((arousal_true - arousal_pred) ** 2)
   ```

---

## 9. CHECKPOINTS VÀ MODELS

### 9.1. Cấu Trúc Checkpoint

```
checkpoints/
├── method2_kfold_wpro/          # IEMOCAP K-fold
│   ├── fold_1/
│   │   ├── best_model.pt
│   │   ├── last_model.pt
│   │   └── training_log.json
│   ├── fold_2/
│   ├── ...
│   └── fold_5/
│
├── method2_msp_wpro/            # MSP-Podcast
│   ├── best_model.pt
│   ├── last_model.pt
│   └── training_log.json
│
└── method2_kfold_wpro_asr/      # IEMOCAP with ASR text
    └── ...
```

### 9.2. Checkpoint Contents

**Model state dict:**
```python
checkpoint = {
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'best_wa': best_wa,
    'config': config
}
torch.save(checkpoint, 'best_model.pt')
```

---

## 10. TÓM TẮT PIPELINE

### IEMOCAP Pipeline

```
1. Raw IEMOCAP → prepare_csv.py
   ↓
2. all.csv (5,533 samples, 4 emotions)
   ↓
3. K-fold split (5 folds, speaker-independent)
   ↓
4. Load IEMOCAPDataset:
   - Audio: resample, pad/truncate to 10s
   - Text: RoBERTa tokenize to 128 tokens
   - A-V: normalize [1,5] → [-1,1]
   ↓
5. Train Method2_WPRO:
   - Prosody: WavLM-Pro (frozen)
   - Text: RoBERTa (frozen)
   - Cross-attention + Fusion
   - Multi-task: Emotion + A-V
   ↓
6. Evaluate:
   - WA, UAR, F1 (emotion)
   - CCC, Pearson (A-V)
   ↓
7. Aggregate 5-fold results
```

### MSP-Podcast Pipeline

```
1. Raw MSP-Podcast (msp_podcast.tsv)
   ↓
2. Filter 4 emotions (ang, sad, hap, neu)
   ↓
3. Load MSPPodcastDataset:
   - Audio: from Audios/
   - Text: from Transcripts/
   - A-V: from Labels/labels_consensus.csv
   - Normalize: z-score
   ↓
4. Train Method2_WPRO:
   - Same architecture as IEMOCAP
   - Train/Dev/Test1/Test2 splits
   ↓
5. Evaluate:
   - Dev: for model selection
   - Test1, Test2: final results
   ↓
6. Report metrics
```

---

## 11. SO SÁNH IEMOCAP vs MSP-PODCAST

| Aspect | IEMOCAP | MSP-Podcast |
|--------|---------|-------------|
| **Dataset size** | 5,533 samples | ~212,810 samples |
| **Emotions** | 4 classes (hap, sad, neu, ang) | 4 classes (same) |
| **A-V scale** | 1-5 (discrete) | Continuous |
| **A-V normalization** | Linear [1,5] → [-1,1] | Z-score normalization |
| **Transcripts** | Ground-truth available | Ground-truth available |
| **Splits** | 5-fold cross-validation (speaker-independent) | Fixed train/dev/test1/test2 |
| **Audio format** | .wav (16kHz) | .wav (16kHz) |
| **Domain** | Acted emotions (scripts + improv) | Natural emotions (podcast) |

---

## 12. CHÚ Ý QUAN TRỌNG

### 12.1. Frozen vs Fine-tuning

**Current setup:** Tất cả pretrained models đều frozen
- **Pros:** Fast training, fewer params, less overfitting
- **Cons:** Có thể chưa tối ưu cho task-specific features

**Alternative:** Fine-tuning with small learning rate
```yaml
freeze_wavlm: false
freeze_roberta: false
learning_rate: 1e-5  # Smaller LR for fine-tuning
```

### 12.2. A-V Normalization Differences

**IEMOCAP:** Linear scaling
```python
# [1, 5] → [-1, 1]
arousal = (arousal - 3.0) / 2.0
```

**MSP-Podcast:** Z-score
```python
# μ ≈ 3.5, σ ≈ 0.8 (example)
arousal = (arousal - arousal_mean) / arousal_std
```

**Implication:** Không thể dùng trực tiếp model trained trên IEMOCAP cho MSP-Podcast và ngược lại (cần re-normalize hoặc transfer learning)

### 12.3. Batch Size và Memory

**WavLM-Pro + WavLM + RoBERTa rất tốn GPU memory**

**Recommendations:**
- IEMOCAP: batch_size = 4-8 (depends on GPU)
- MSP-Podcast: batch_size = 4-8
- Use gradient accumulation nếu cần batch size lớn hơn

### 12.4. Text Quality

**Ground-truth transcript:**
- IEMOCAP: High quality, manual transcription
- MSP-Podcast: High quality, manual transcription

**ASR transcript (Whisper):**
- Accuracy: ~95% WER for IEMOCAP (clean audio)
- Trade-off: Realistic scenario nhưng có noise
- Usage: Thí nghiệm robustness

---

## 13. FUTURE IMPROVEMENTS

1. **Data Augmentation:**
   - SpecAugment cho audio
   - Paraphrasing cho text

2. **Advanced Fusion:**
   - Attention-based fusion thay vì concat
   - Late fusion với separate heads

3. **Focal Loss:**
   - Xử lý class imbalance (nếu có)

4. **Multi-task Weighting:**
   - Dynamic weight adjustment
   - Uncertainty-based weighting

5. **Ensemble:**
   - Ensemble multiple folds
   - Ensemble IEMOCAP + MSP-Podcast

---

## 14. REFERENCES

- **IEMOCAP:** Busso et al., LREC 2008
- **MSP-Podcast:** Lotfian & Busso, IEEE TASLP 2019
- **WavLM:** Chen et al., ICASSP 2022
- **W-PRO:** Gengembre et al., Interspeech 2024
- **RoBERTa:** Liu et al., arXiv 2019
- **CCC Loss:** Lin, Biometrics 1989

---

**Tài liệu này cung cấp toàn bộ quy trình xử lý dữ liệu từ raw data đến trained model cho method2_wpro trên IEMOCAP và MSP-Podcast.**
