
"""Wrapper trainer: run original k-fold W-PRO trainer but monkeypatch the model class
and (optionally) the dataset tokenizer to match the configured lightweight text encoder.
"""
import sys
from pathlib import Path
import yaml
import torch
from thop import profile, clever_format

sys.path.append(str(Path(__file__).resolve().parent.parent))

import train.train_method2_kfold_wpro as base_train
from data.iemocap_dataset import IEMOCAPDataset as _BaseIEMOCAPDataset
from models.method2_crossattn_av_lite_albert_egemaps import Method2CrossAttentionAVLiteAlbertEGeMAPS


class _TokenizerPatchedDataset(_BaseIEMOCAPDataset):
    tokenizer_name_override = 'roberta-base'

    def __init__(self, *args, **kwargs):
        kwargs.setdefault('tokenizer_name', self.tokenizer_name_override)
        super().__init__(*args, **kwargs)


# Monkeypatch model class used inside the base trainer
base_train.Method2CrossAttentionAV = Method2CrossAttentionAVLiteAlbertEGeMAPS


def _safe_compute_flops(model, audio_length=16000 * 10, text_length=128, device='cpu'):
    """Compute FLOPs with a vocab-safe dummy text input for lightweight text encoders."""
    try:
        orig_device = next(model.parameters()).device
    except StopIteration:
        orig_device = torch.device('cpu')

    try:
        flops_device = torch.device('cpu')
        model = model.to(flops_device)
        audio_input = torch.randn(1, audio_length, device=flops_device)

        vocab_size = 30522
        text_model = getattr(model, 'text_model', None)
        if text_model is not None:
            vocab_size = int(getattr(text_model.config, 'vocab_size', vocab_size))
        vocab_size = max(vocab_size, 2)

        text_input = torch.randint(0, vocab_size, (1, text_length), device=flops_device)
        text_mask = torch.ones(1, text_length, device=flops_device, dtype=torch.long)

        flops, params = profile(
            model,
            inputs=(audio_input, text_input, None, text_mask),
            verbose=False,
        )
        flops_str, params_str = clever_format([flops, params], "%.3f")
        return flops, flops_str, params_str
    except Exception as e:
        print(f"Warning: FLOPs computation failed: {e}")
        return None, "N/A", "N/A"
    finally:
        model.to(orig_device)


base_train.compute_flops = _safe_compute_flops


if __name__ == '__main__':
    import argparse
    import os

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/method2_lite_albert_egemaps_config.yaml')
    parser.add_argument('--csv', default='datasets/IEMOCAP_KFOLD/tenfolds/all.csv')
    parser.add_argument('--output_dir', default='checkpoints/method2_kfold_lite_albert_egemaps')
    parser.add_argument('--device', default=None)
    parser.add_argument('--auto_resume_fold5', action='store_true')
    args, remaining = parser.parse_known_args()

    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)
    _TokenizerPatchedDataset.tokenizer_name_override = cfg['model'].get('roberta_model_name', 'roberta-base')
    base_train.IEMOCAPDataset = _TokenizerPatchedDataset

    output_dir = args.output_dir
    resume_ckpt = None
    if args.auto_resume_fold5:
        fold5_dir = os.path.join(output_dir, 'fold_5')
        if os.path.exists(fold5_dir):
            last_path = os.path.join(fold5_dir, 'last_model.pt')
            best_path = os.path.join(fold5_dir, 'best_model.pt')
            if os.path.exists(last_path):
                resume_ckpt = last_path
            elif os.path.exists(best_path):
                resume_ckpt = best_path

    new_argv = [sys.argv[0], '--config', args.config, '--csv', args.csv, '--output_dir', output_dir]
    if args.device:
        new_argv += ['--device', args.device]
    if resume_ckpt is not None:
        new_argv += ['--resume_fold', '5', '--resume_checkpoint', resume_ckpt]
    new_argv += remaining
    sys.argv = new_argv
    base_train.main()
