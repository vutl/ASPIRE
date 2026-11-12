"""Wrapper trainer: use the WavLM+RoBERTa emotion-only model for k-fold training.

Monkeypatches `Method2CrossAttentionAV` in the base trainer so the training
flow is unchanged.
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import train.train_method2_kfold as base_train
from models.method2_wavlm_roberta_emotion_only import Method2WavLMRobertaEmotionOnly

# Monkeypatch model class used inside the base trainer
base_train.Method2CrossAttentionAV = Method2WavLMRobertaEmotionOnly


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/method2_config.yaml')
    parser.add_argument('--csv', default='datasets/IEMOCAP_KFOLD/tenfolds/all.csv')
    parser.add_argument('--output_dir', default='checkpoints/method2_kfold_wavlm_roberta_emotion_only')
    parser.add_argument('--device', default=None)
    args, remaining = parser.parse_known_args()

    new_argv = [sys.argv[0], '--config', args.config, '--csv', args.csv, '--output_dir', args.output_dir]
    if args.device:
        new_argv += ['--device', args.device]
    new_argv += remaining
    sys.argv = new_argv
    base_train.main()
