# MSP-Podcast Training Setup - Summary

## Dataset Information

### MSP-Podcast Dataset (4-class filtered)
- **Total samples**: 197,091 (92.6% of full dataset)
- **Train**: 129,064 samples
  - ang: 22,609 | hap: 37,048 | neu: 51,151 | sad: 18,256
- **Dev**: 23,885 samples
  - ang: 5,728 | hap: 7,487 | neu: 8,318 | sad: 2,352
- **Test1**: 33,430 samples
  - ang: 6,985 | hap: 10,947 | neu: 12,457 | sad: 3,041
- **Test2**: 10,712 samples
  - ang: 538 | hap: 2,801 | neu: 6,792 | sad: 581

### Emotion Label Mapping
```python
{
  'ang': 0,  # Anger
  'hap': 1,  # Happiness
  'neu': 2,  # Neutral
  'sad': 3   # Sadness
}
```

### A-V Value Statistics (normalized)
- **Arousal**: μ=4.245, σ=0.982 (original range: [1, 7])
- **Valence**: μ=3.823, σ=0.954 (original range: [1, 7])

---

## Training Configuration

### Model: Method 2 (Cross-Attention A-V)
- **Architecture**: WavLM_prosody + RoBERTa + WavLM_final
- **Parameters**: ~319M (frozen pretrained models)
- **Batch size**: 2 (due to large model)
- **Epochs**: 10 (dataset is large, 10 should be enough)
- **Learning rate**: 0.0001
- **Loss**: CombinedEmotionLossCCC (CrossEntropy + CCC for A-V)

### Training Strategy
1. Train on `train` split
2. Validate on `dev` split after each epoch
3. Save checkpoint when dev accuracy improves
4. After training, test on both `test1` and `test2` splits

### Key Differences from IEMOCAP
| Aspect | IEMOCAP | MSP-Podcast |
|--------|---------|-------------|
| Split Strategy | 5-fold CV | Fixed train/dev/test1/test2 |
| Dataset Size | ~10k samples | ~197k samples (4-class) |
| Epochs | 30 | 10 |
| Validation | 80/20 per fold | Fixed dev set |
| Test Sets | 1 per fold | 2 separate test sets |

---

## Files Created

### 1. Dataset Loader
**File**: `data/msp_podcast_dataset.py`
- Loads TSV with metadata
- Filters to 4 emotions (ang, sad, hap, neu)
- Loads audio from `Audios/` (symlinked)
- Loads transcripts from `Transcripts/`
- Loads A-V values from `Labels/labels_consensus.csv`
- Normalizes A-V values (z-score normalization)
- Returns batch dict compatible with Method 2 model

### 2. Training Script
**File**: `train/train_method2_msp.py`
- Trains Method 2 on MSP-Podcast
- No k-fold, uses fixed splits
- Validates on dev set after each epoch
- Saves best checkpoint based on dev accuracy
- After training, tests on test1 and test2
- Logs to file and tensorboard

### 3. Configuration
**File**: `configs/msp_podcast_config.yaml`
- Model: Method 2 with frozen WavLM/RoBERTa
- Batch size: 2
- Epochs: 10
- Loss: CCC for A-V, CrossEntropy for emotion
- Optimizer: AdamW with ReduceLROnPlateau scheduler

### 4. Run Script
**File**: `run_train_msp.sh`
- Simple wrapper to launch training
- Runs in background with nohup
- Logs to `logs/train_msp_{timestamp}.log`

---

## How to Run

### Quick Start
```bash
# Make sure you're in ser_env conda environment
conda activate ser_env

# Launch training (runs in background)
bash run_train_msp.sh cuda:0

# Monitor training
tail -f logs/train_msp_*.log

# View tensorboard
tensorboard --logdir checkpoints/method2_msp/logs/tensorboard --port 6008
```

### Manual Run
```bash
python train/train_method2_msp.py \
    --config configs/msp_podcast_config.yaml \
    --tsv datasets/MSP-Podcast/msp_podcast.tsv \
    --output_dir checkpoints/method2_msp \
    --device cuda:0
```

### Resume from Checkpoint
```bash
python train/train_method2_msp.py \
    --config configs/msp_podcast_config.yaml \
    --tsv datasets/MSP-Podcast/msp_podcast.tsv \
    --output_dir checkpoints/method2_msp \
    --device cuda:0 \
    --resume checkpoints/method2_msp/last_model.pt
```

---

## Expected Output

### During Training
- Train metrics per epoch (loss, acc, F1, CCC)
- Dev metrics per epoch (loss, acc, F1, CCC)
- Best checkpoint saved when dev accuracy improves

### After Training
- **Test1 Results**: Accuracy, F1 Macro/Weighted, Arousal CCC, Valence CCC
- **Test2 Results**: Accuracy, F1 Macro/Weighted, Arousal CCC, Valence CCC

### Checkpoints
- `checkpoints/method2_msp/best_model.pt` - Best model based on dev accuracy
- `checkpoints/method2_msp/last_model.pt` - Last epoch checkpoint

---

## Monitoring

### Tensorboard Metrics
- Train/Dev/Test1/Test2: loss, accuracy, f1_macro, f1_weighted
- Train/Dev/Test1/Test2: arousal_ccc, valence_ccc, avg_ccc

### Log File
- Full training log with timestamps
- Epoch-by-epoch metrics
- Final test results on test1 and test2

---

## Notes

1. **Symlinks Created**:
   - `datasets/MSP-Podcast/Audios` → `/home4/quanpn/ICME2026/ser_ssl_prosody/data/MSP-Podcast/Audios`
   - `datasets/MSP-Podcast/Transcripts` → `/home4/quanpn/ICME2026/ser_ssl_prosody/data/MSP-Podcast/Transcripts`

2. **A-V Values**: Loaded from `Labels/labels_consensus.csv` and normalized with μ/σ from full dataset

3. **4-Class Filter**: Only samples with labels `ang`, `sad`, `hap`, `neu` are used (197k out of 213k total)

4. **Seed**: 42 (for reproducibility)

5. **Model Freezing**: WavLM and RoBERTa pretrained weights are frozen (only train task-specific layers)

---

## Comparison with IEMOCAP Training

| Setting | IEMOCAP | MSP-Podcast |
|---------|---------|-------------|
| Dataset | `IEMOCAP_KFOLD/tenfolds/all.csv` | `MSP-Podcast/msp_podcast.tsv` |
| Splits | 5-fold CV (10→5 sessions) | Fixed train/dev/test1/test2 |
| Train samples | ~8k per fold | 129k |
| Val samples | ~2k per fold | 23.9k (dev) |
| Test samples | ~2k per fold | 33.4k (test1), 10.7k (test2) |
| Epochs | 30 | 10 |
| Training time | ~6-8 hours per fold | TBD (~12-15 hours estimated) |
| Checkpoints | Per-fold best model | Single best model |
| Final metrics | Average across 5 folds | test1 + test2 |

---

## Ready to Train!

All files are created and tested. You can now start training with:

```bash
bash run_train_msp.sh cuda:0
```

The training will:
1. Load 129k train samples
2. Train for 10 epochs with dev validation
3. Save best checkpoint based on dev accuracy
4. Test on test1 (33k samples) and test2 (10k samples)
5. Report final metrics for both test sets

Good luck! 🚀
