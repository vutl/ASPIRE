# Quick Start Guide

## 📋 Tổng quan

Project này implement 2 phương pháp nhận dạng cảm xúc từ speech với A-V modeling:
- **Method 1**: Triple WavLM (204M parameters)
- **Method 2**: Cross-Attention A-V (119M parameters, recommended ⭐)

---

## 🚀 Quick Commands

### 1. Test models (kiểm tra code)
```bash
python test_models.py
```

### 2. Train Method 1
```bash
# Full command
python train/train_method1.py \
    --train_csv datasets/IEMOCAP_KFOLD/IEMOCAP_train4classes.csv \
    --test_csv datasets/IEMOCAP_KFOLD/IEMOCAP_test4classes.csv \
    --batch_size 8 \
    --num_epochs 50 \
    --device cuda:0

# Or use script
./run_training.sh method1 cuda:0 8 50
```

### 3. Train Method 2 (Recommended ⭐)
```bash
# Full command
python train/train_method2.py \
    --train_csv datasets/IEMOCAP_KFOLD/IEMOCAP_train4classes.csv \
    --test_csv datasets/IEMOCAP_KFOLD/IEMOCAP_test4classes.csv \
    --batch_size 8 \
    --num_epochs 50 \
    --device cuda:0

# Or use script
./run_training.sh method2 cuda:0 8 50
```

### 4. Monitor training with Tensorboard
```bash
tensorboard --logdir logs/tensorboard --port 6006
```

### 5. Compare trained models
```bash
python compare_models.py \
    --method1_checkpoint checkpoints/method1_XXX/best_model.pt \
    --method2_checkpoint checkpoints/method2_XXX/best_model.pt \
    --test_csv datasets/IEMOCAP_KFOLD/IEMOCAP_test4classes.csv
```

---

## 📊 Expected Results

### Baseline Performance (IEMOCAP 4-class)

| Method | Accuracy | F1-weighted | A-CCC | V-CCC | Parameters |
|--------|----------|-------------|-------|-------|------------|
| Method 1 | ~65-70% | ~0.65-0.70 | ~0.55 | ~0.50 | 204M |
| Method 2 | ~65-70% | ~0.65-0.70 | ~0.55 | ~0.50 | 119M |

*Note: Kết quả thực tế phụ thuộc vào hyperparameters và training setup*

---

## 🔧 Hyperparameter Tuning

### Learning Rate
```bash
# Try different learning rates
python train/train_method2.py --learning_rate 1e-5  # Lower
python train/train_method2.py --learning_rate 5e-4  # Higher
```

### Loss Weights
```bash
# Prioritize emotion classification
python train/train_method2.py --emotion_weight 2.0 --arousal_weight 0.3 --valence_weight 0.3

# Prioritize A-V regression
python train/train_method2.py --emotion_weight 0.5 --arousal_weight 1.0 --valence_weight 1.0
```

### Use CCC loss instead of MSE
```bash
python train/train_method2.py --loss_type ccc
```

### Freeze WavLM (faster training, less VRAM)
```bash
python train/train_method2.py --freeze_wavlm
```

---

## 🗂️ Project Structure

```
AVD_Test/
├── models/                    # Model architectures
│   ├── base_components.py     # Attention, fusion, encoders
│   ├── method1_triple_wavlm.py
│   └── method2_crossattn_av.py
├── train/                     # Training scripts
│   ├── losses.py
│   ├── train_method1.py
│   └── train_method2.py
├── data/                      # Data processing
│   └── iemocap_dataset.py
├── configs/                   # Config files
│   ├── method1_config.yaml
│   └── method2_config.yaml
├── datasets/                  # Data
│   ├── IEMOCAP/
│   └── IEMOCAP_KFOLD/
├── logs/                      # Training logs
├── checkpoints/               # Model checkpoints
├── test_models.py             # Quick test
├── compare_models.py          # Model comparison
├── run_training.sh            # Training script
└── README.md
```

---

## 💡 Tips

### 1. GPU Memory Issues
```bash
# Reduce batch size
python train/train_method2.py --batch_size 4

# Or freeze WavLM
python train/train_method2.py --freeze_wavlm
```

### 2. Speed up training
```bash
# Use smaller audio length
python train/train_method2.py --audio_max_length 80000  # 5 seconds

# More workers
python train/train_method2.py --num_workers 8
```

### 3. Debug mode (quick test)
```bash
# Train for 2 epochs only
python train/train_method2.py --num_epochs 2 --batch_size 2
```

---

## 📝 Notes

### Why Method 2 is recommended?
1. **41.7% fewer parameters** → less memory, faster training
2. **Cross-attention between prosody and text** → better feature interaction
3. **WavLM used 2 times instead of 3** → less redundancy

### A-V Refinement
- Thay vì dùng GAT/GCN phức tạp, chúng ta dùng MLP với residual connections
- Đơn giản nhưng hiệu quả cho task này

### Multi-task Learning
- Model được train đồng thời 3 tasks: emotion classification, arousal regression, valence regression
- Loss weights có thể điều chỉnh để ưu tiên task nào

---

## 🐛 Troubleshooting

### Error: CUDA out of memory
```bash
# Solution 1: Reduce batch size
--batch_size 4

# Solution 2: Freeze WavLM
--freeze_wavlm

# Solution 3: Use smaller audio
--audio_max_length 80000
```

### Error: Dataset not found
```bash
# Check paths
ls datasets/IEMOCAP_KFOLD/

# Update paths in training script
--train_csv <your_path>
--test_csv <your_path>
```

### Error: Checkpoint loading failed
```bash
# Make sure model architecture matches checkpoint
# Check num_emotions, hidden_dim, etc.
```

---

## 📧 Contact

Nếu có vấn đề, tạo issue hoặc liên hệ.

**Good luck! 🚀**
