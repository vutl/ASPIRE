"""Wrapper trainer: run original k-fold W-PRO trainer with no-eAV ablation model.

This keeps all training logic/logging/metrics identical to the base trainer and
only swaps the model class.
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import train.train_method2_kfold_wpro as base_train
from models.method2_crossattn_av_ablation_no_eav import Method2CrossAttentionAV_AblationNoEAV

# Monkeypatch model class used inside the base trainer
base_train.Method2CrossAttentionAV = Method2CrossAttentionAV_AblationNoEAV


if __name__ == "__main__":
    import argparse
    import os

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/method2_wpro_ablation_no_eav_config.yaml")
    parser.add_argument("--csv", default="datasets/IEMOCAP_KFOLD/tenfolds/all.csv")
    parser.add_argument("--output_dir", default="checkpoints/method2_kfold_wpro_ablation_no_eav")
    parser.add_argument("--device", default=None)
    args, remaining = parser.parse_known_args()

    # Discover resume checkpoint for fold_5 if exists (prefer last_model.pt then best_model.pt)
    output_dir = args.output_dir
    fold5_dir = os.path.join(output_dir, "fold_5")
    resume_ckpt = None
    if os.path.exists(fold5_dir):
        last_path = os.path.join(fold5_dir, "last_model.pt")
        best_path = os.path.join(fold5_dir, "best_model.pt")
        if os.path.exists(last_path):
            resume_ckpt = last_path
        elif os.path.exists(best_path):
            resume_ckpt = best_path

    new_argv = [sys.argv[0], "--config", args.config, "--csv", args.csv, "--output_dir", output_dir]
    if args.device:
        new_argv += ["--device", args.device]
    if resume_ckpt is not None:
        new_argv += ["--resume_fold", "5", "--resume_checkpoint", resume_ckpt]

    new_argv += remaining
    sys.argv = new_argv
    base_train.main()