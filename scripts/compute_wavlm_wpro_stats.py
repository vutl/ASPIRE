#!/usr/bin/env python3
"""Compute parameter counts and FLOPs for WavLM+WPRO AV model.

This script uses `use_pretrained=False` by default to avoid downloading HF
weights; set `--use_pretrained` to load real WavLM/W-PRO (will download).
"""
import argparse
import torch
from pathlib import Path
import numpy as np

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))

from models.wavlm_wpro_av import WavLMWProAV

def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = total - trainable
    return total, trainable, frozen

def fmt(n):
    if n >= 1e9:
        return f"{n/1e9:.3f}B"
    if n >= 1e6:
        return f"{n/1e6:.3f}M"
    if n >= 1e3:
        return f"{n/1e3:.3f}K"
    return str(n)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_pretrained', action='store_true')
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--audio_seconds', type=int, default=3)
    parser.add_argument('--batch', type=int, default=1)
    args = parser.parse_args()

    device = torch.device(args.device)
    model = WavLMWProAV(use_pretrained=args.use_pretrained)
    model.to(device)

    total, trainable, frozen = count_params(model)
    print('='*110)
    print('📊 MODEL STATISTICS')
    print('='*110)
    print('Total parameters:     ', fmt(total))
    print('Trainable parameters: ', fmt(trainable))
    print('Frozen parameters:    ', fmt(frozen))

    # Compute FLOPs using thop if available
    try:
        from thop import profile, clever_format
        audio_len = 16000 * args.audio_seconds
        dummy = torch.randn(args.batch, audio_len).to(device)
        # thop expects call signature model(dummy)
        macs, params = profile(model, inputs=(dummy,), verbose=False)
        macs_str, params_str = clever_format([macs, params], '%.3f')
        print('\n⚡ Computing FLOPs...')
        print('FLOPs: ', macs_str)
        print('Params:', params_str)
    except Exception as e:
        print('\n⚡ FLOPs computation skipped (thop missing or failed):', e)

if __name__ == '__main__':
    main()
