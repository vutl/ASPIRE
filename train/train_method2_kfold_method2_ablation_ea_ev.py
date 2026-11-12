"""Wrapper trainer to run the Method2 ablation (AV_refined + eA + eV) under k-fold.

Monkeypatches `Method2CrossAttentionAV` symbol in the base k-fold trainer module
to point to the ablation class so the training loop remains unchanged.
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import train.train_method2_kfold as base_train
from models.method2_crossattn_av_ablation_ea_ev import Method2CrossAttentionAV_Ablation_eA_eV

# Monkeypatch
base_train.Method2CrossAttentionAV = Method2CrossAttentionAV_Ablation_eA_eV


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/method2_wpro_ablation_roberta_config.yaml')
    parser.add_argument('--csv', default='datasets/IEMOCAP_KFOLD/tenfolds/all.csv')
    parser.add_argument('--output_dir', default='checkpoints/method2_kfold_method2_ablation_ea_ev')
    parser.add_argument('--device', default=None)
    # Model/backbone overrides (forwarded to the underlying trainer)
    parser.add_argument('--wavlm_model_name', default=None, help='WavLM backbone name for final wavlm branch')
    parser.add_argument('--wavlm_prosody_model_name', default=None, help='Prosody WavLM (WPRO) model name; if set, use WPRO')
    parser.add_argument('--roberta_model_name', default=None, help='RoBERTa model name')
    parser.add_argument('--hidden_dim', type=int, default=None, help='hidden dim override')
    parser.add_argument('--freeze_wavlm', action='store_true', help='freeze wavlm weights')
    parser.add_argument('--freeze_roberta', action='store_true', help='freeze roberta weights')
    parser.add_argument('--av_refinement_layers', type=int, default=None, help='override av_refinement_layers')
    parser.add_argument('--av_prediction_layers', type=int, default=None, help='override av_prediction_layers')
    args, remaining = parser.parse_known_args()
    # If user passed backbone/model overrides, apply them into a copied config
    # and pass that config to the base trainer (avoids unknown-arg errors).
    import yaml
    import os

    config_path = args.config
    try:
        with open(config_path, 'r') as f:
            cfg = yaml.safe_load(f) or {}
    except Exception:
        cfg = {}

    cfg.setdefault('model', {})
    updated = False
    if args.wavlm_model_name:
        cfg['model']['wavlm_model_name'] = args.wavlm_model_name
        updated = True
    if args.wavlm_prosody_model_name:
        cfg['model']['wavlm_prosody_model_name'] = args.wavlm_prosody_model_name
        updated = True
    if args.roberta_model_name:
        cfg['model']['roberta_model_name'] = args.roberta_model_name
        updated = True
    if args.hidden_dim is not None:
        cfg['model']['hidden_dim'] = args.hidden_dim
        updated = True
    if args.freeze_wavlm:
        cfg['model']['freeze_wavlm'] = True
        updated = True
    if args.freeze_roberta:
        cfg['model']['freeze_roberta'] = True
        updated = True
    if args.av_refinement_layers is not None:
        cfg['model']['av_refinement_layers'] = args.av_refinement_layers
        updated = True
    if args.av_prediction_layers is not None:
        cfg['model']['av_prediction_layers'] = args.av_prediction_layers
        updated = True

    if updated:
        # write override config into the output dir so it's saved with the run
        os.makedirs(args.output_dir, exist_ok=True)
        override_path = os.path.join(args.output_dir, 'config_override.yaml')
        with open(override_path, 'w') as f:
            yaml.safe_dump(cfg, f)
        config_path = override_path

    new_argv = [sys.argv[0], '--config', config_path, '--csv', args.csv, '--output_dir', args.output_dir]
    if args.device:
        new_argv += ['--device', args.device]
    new_argv += remaining
    sys.argv = new_argv
    base_train.main()
