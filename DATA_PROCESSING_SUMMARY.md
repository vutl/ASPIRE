# Quy Trình Xử Lý Dữ Liệu cho Method2_WPRO K-Fold

## 1. XỬ LÝ IEMOCAP

### Bước 1: Chuẩn bị CSV gốc
**Script:** `datasets/IEMOCAP_KFOLD/prepare_csv.py`

```python
# Input: IEMOCAP.csv (raw dataset)
df = pd.read_csv("/home4/quanpn/IEMOCAP/IEMOCAP.csv")

# Gộp emotion: exc → hap
df['emotion'] = df['emotion'].replace('exc', 'hap')

# Lọc 4 emotions chính
keep_emotions = ['neu', 'ang', 'sad', 'hap']
df = df[df['emotion'].isin(keep_emotions)].reset_index(drop=True)

# Tạo fold theo speaker (10 speakers → 10 folds)
# Ses01F, Ses01M → fold 1, 2
# Ses02F, Ses02M → fold 3, 4
# ...
# Ses05F, Ses05M → fold 9, 10
df['fold'] = df['sample_name'].apply(extract_speaker).map(fold_map)

# Output: all.csv (5,533 samples)
df.to_csv("datasets/IEMOCAP_KFOLD/tenfolds/all.csv", index=False)
```

**Cấu trúc all.csv:**
```csv
sample_name,emotion,arousal,valence,dominance,transcript,path,fold
Ses01F_script02_1_F000,neu,2.0,2.5,2.0,Fine.,/path/to/audio.wav,1
```

### Bước 2: Chia K-Fold (5 folds)
**Trong:** `train/train_method2_kfold.py` → `prepare_folds()`

```python
# Map 10 speaker folds → 5 session folds
fold_to_session = {
    1: 1, 2: 1,  # Session 1
    3: 2, 4: 2,  # Session 2
    5: 3, 6: 3,  # Session 3
    7: 4, 8: 4,  # Session 4
    9: 5, 10: 5  # Session 5
}

# Mỗi fold:
# - Test: 1 session (~20%)
# - Train: 4 sessions còn lại (~80%)
# - Val: split từ train (20%)
```

### Bước 3: Dataset Loading
**Class:** `data/iemocap_dataset.py` → `IEMOCAPDataset`

**Audio processing:**
```python
# 1. Load audio
waveform, sr = torchaudio.load(audio_path)

# 2. Resample to 16kHz
if sr != 16000:
    waveform = resampler(waveform)

# 3. Convert to mono
if waveform.shape[0] > 1:
    waveform = torch.mean(waveform, dim=0, keepdim=True)

# 4. Pad/truncate to 160,000 samples (10 seconds)
if len(waveform) > 160000:
    waveform = waveform[:160000]
else:
    waveform = F.pad(waveform, (0, 160000 - len(waveform)))
```

**Text processing:**
```python
# Tokenize với RoBERTa
encoding = tokenizer(
    transcript,
    max_length=128,
    padding='max_length',
    truncation=True,
    return_tensors='pt'
)
text_input_ids = encoding['input_ids'].squeeze(0)
text_attention_mask = encoding['attention_mask'].squeeze(0)
```

**A-V normalization:**
```python
# IEMOCAP: [1, 5] → [-1, 1]
arousal = (arousal - 3.0) / 2.0
valence = (valence - 3.0) / 2.0
dominance = (dominance - 3.0) / 2.0
```

**Emotion mapping:**
```python
emotion_map = {
    'hap': 0,  # Happy
    'sad': 1,  # Sad
    'neu': 2,  # Neutral
    'ang': 3   # Angry
}
```

---

## 2. XỬ LÝ MSP-PODCAST

### Bước 1: Load TSV Metadata
**File chính:** `datasets/MSP-Podcast/msp_podcast.tsv`

```tsv
utt_id	path	label	split	speaker	lang	duration	fold
MSP-PODCAST_0001_0008	/path/to/audio.wav	neu	train	30	en	10.488	1
```

### Bước 2: Dataset Loading
**Class:** `data/msp_podcast_dataset.py` → `MSPPodcastDataset`

```python
# 1. Load TSV và filter
df = pd.read_csv(tsv_path, sep='\t')
df = df[df['split'] == split]  # train/dev/test1/test2

# 2. Filter 4 emotions
df = df[df['label'].isin(['ang', 'sad', 'hap', 'neu'])]

# 3. Load A-V values từ labels_consensus.csv
labels_df = pd.read_csv('Labels/labels_consensus.csv')
av_dict = {
    row['FileName']: {
        'arousal': row['EmoAct'],
        'valence': row['EmoVal'],
        'dominance': row['EmoDom']
    }
    for _, row in labels_df.iterrows()
}

# 4. Load transcript từ Transcripts/{utt_id}.txt
transcript_file = f'Transcripts/{utt_id}.txt'
with open(transcript_file, 'r') as f:
    transcript = f.read().strip()
```

**Audio & Text processing:** Giống IEMOCAP (resample, pad, tokenize)

**A-V normalization:**
```python
# MSP-Podcast: Z-score normalization
arousal_mean = np.mean(all_arousal_values)  # ≈ 3.5
arousal_std = np.std(all_arousal_values)    # ≈ 0.8

arousal = (arousal - arousal_mean) / (arousal_std + 1e-8)
valence = (valence - valence_mean) / (valence_std + 1e-8)
```

**Emotion mapping:** Giống IEMOCAP (hap=0, sad=1, neu=2, ang=3)

---

## 3. TRÍCH XUẤT ASR (OPTIONAL)

### IEMOCAP ASR
**Script:** `scripts/extract_asr_text_whisper.py`

```python
# Load Whisper
model = whisper.load_model("base", device="cuda")

# Transcribe audio
result = model.transcribe(audio_path, language="en", fp16=True)
asr_text = result["text"].strip()

# Save to all_asr.csv
```

**Output:** `datasets/IEMOCAP_KFOLD/tenfolds/all_asr.csv`

### MSP-Podcast ASR
**Script:** `scripts/extract_asr_msp_podcast.py`

Tương tự IEMOCAP, output: `msp_podcast_asr.tsv`

---

## 4. TÓM TẮT PIPELINE

### IEMOCAP K-Fold
```
Raw IEMOCAP
    ↓
prepare_csv.py (gộp exc→hap, filter 4 emotions, tạo fold)
    ↓
all.csv (5,533 samples, 10 folds)
    ↓
prepare_folds() (10 folds → 5 session folds)
    ↓
IEMOCAPDataset
  - Audio: resample 16kHz, pad/truncate 10s
  - Text: RoBERTa tokenize, max_length=128
  - A-V: normalize [1,5] → [-1,1]
  - Emotion: map to 0-3
    ↓
DataLoader (batch_size=8)
    ↓
Training (5 folds, speaker-independent)
```

### MSP-Podcast
```
msp_podcast.tsv + labels_consensus.csv + Transcripts/
    ↓
MSPPodcastDataset
  - Filter split (train/dev/test1/test2)
  - Filter 4 emotions
  - Load A-V từ labels_consensus.csv
  - Load transcript từ Transcripts/
  - Audio: resample 16kHz, pad/truncate 10s
  - Text: RoBERTa tokenize, max_length=128
  - A-V: z-score normalize
  - Emotion: map to 0-3
    ↓
DataLoader (batch_size=8)
    ↓
Training (fixed train/dev/test splits)
```

---

## 5. SO SÁNH KEY DIFFERENCES

| Aspect | IEMOCAP | MSP-Podcast |
|--------|---------|-------------|
| **Size** | 5,533 | ~212,810 |
| **Splits** | 5-fold CV (speaker-independent) | Fixed train/dev/test1/test2 |
| **A-V norm** | Linear [1,5]→[-1,1] | Z-score |
| **A-V scale** | Discrete (1,2,3,4,5) | Continuous |
| **Domain** | Acted (scripts+improv) | Natural (podcasts) |
| **Transcript source** | Ground-truth | Ground-truth |
| **Metadata file** | CSV | TSV |

---

## 6. CHÚ Ý QUAN TRỌNG

1. **Emotion mapping UNIFIED:** Cả 2 datasets đều map về (hap=0, sad=1, neu=2, ang=3)

2. **A-V normalization khác nhau:**
   - IEMOCAP: Linear scaling
   - MSP-Podcast: Z-score
   - ⚠️ Không thể transfer model trực tiếp giữa 2 datasets

3. **Audio format:** Cả 2 đều xử lý về 16kHz mono, 10 seconds

4. **Text tokenization:** Cả 2 đều dùng RoBERTa, max_length=128

5. **K-fold cho IEMOCAP:** Speaker-independent (test trên session chưa thấy)

6. **MSP-Podcast splits:** Sử dụng partition sẵn của dataset

7. **Missing data handling:**
   - IEMOCAP: Tất cả samples đều có đầy đủ data
   - MSP-Podcast: Một số transcript có thể missing → fallback to ""
