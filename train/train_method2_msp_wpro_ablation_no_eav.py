"""Wrapper trainer: run MSP W-PRO trainer with no-eAV ablation model.

This keeps all MSP training logic/logging/metrics identical to the base trainer and
only swaps the model class.
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import train.train_method2_msp_wpro as base_train
from models.method2_crossattn_av_ablation_no_eav import Method2CrossAttentionAV_AblationNoEAV

# Monkeypatch model class used inside the base trainer
base_train.Method2CrossAttentionAV = Method2CrossAttentionAV_AblationNoEAV


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/msp_podcast_wpro_ablation_no_eav_config.yaml")
    parser.add_argument("--tsv", default="datasets/MSP-Podcast/msp_podcast.tsv")
    parser.add_argument("--output_dir", default="checkpoints/method2_msp_wpro_ablation_no_eav")
    parser.add_argument("--device", default=None)
    parser.add_argument("--resume", default=None)
    args, remaining = parser.parse_known_args()

    new_argv = [
        sys.argv[0],
        "--config", args.config,
        "--tsv", args.tsv,
        "--output_dir", args.output_dir,
    ]
    if args.device:
        new_argv += ["--device", args.device]
    if args.resume:
        new_argv += ["--resume", args.resume]

    new_argv += remaining
    sys.argv = new_argv
    base_train.main()
