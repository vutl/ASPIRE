#!/usr/bin/env python3
"""
Resume 5-fold run: run only fold 1 test (using saved checkpoint) then continue folds 2-5.
This avoids retraining fold 1 while allowing the overall process to continue.
"""
import os
import sys
import argparse
import logging
import torch
import numpy as np
from torch.utils.data import DataLoader

from sklearn.model_selection import train_test_split

# Import of training module is deferred until we ensure project root is on sys.path



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/iemocap_method2_opensmile_kfold_config.yaml')
    parser.add_argument('--csv', type=str, default='datasets/IEMOCAP_KFOLD/all.csv')
    parser.add_argument('--output_dir', type=str, default='checkpoints/iemocap_method2_opensmile_kfold')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)

    # Redirect stdout/stderr to a log file in the output directory so the resume
    # run matches existing logging behavior for other trainers.
    log_path = os.path.join(args.output_dir, 'iemocap_method2_opensmile_kfold_resume.log')
    log_file = open(log_path, 'a', buffering=1)
    sys.stdout = log_file
    sys.stderr = log_file

    # Basic logging as well
    logging.basicConfig(level=logging.INFO, filename=log_path, filemode='a',
                        format='%(asctime)s %(levelname)s: %(message)s')

    # Ensure project root is on sys.path so `import train.*` works when launched
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    # Prefer writing resume logs to the repository `logs/` folder (consistent with other trainers)
    logs_dir = os.path.join(project_root, 'logs')
    os.makedirs(logs_dir, exist_ok=True)
    log_path = os.path.join(logs_dir, 'iemocap_method2_opensmile_kfold_resume.log')
    # Keep a backup copy also in the output_dir for convenience
    os.makedirs(args.output_dir, exist_ok=True)
    chk_log_path = os.path.join(args.output_dir, 'iemocap_method2_opensmile_kfold_resume.log')

    # Redirect stdout/stderr to the logs file
    log_file = open(log_path, 'a', buffering=1)
    sys.stdout = log_file
    sys.stderr = log_file

    # Basic logging as well
    logging.basicConfig(level=logging.INFO, filename=log_path, filemode='a',
                        format='%(asctime)s %(levelname)s: %(message)s')

    # Now import the runner module
    import importlib
    runner = importlib.import_module('train.train_iemocap_method2_opensmile_kfold')

    config = runner.load_config(args.config)
    folds = runner.prepare_folds(args.csv)
    device = torch.device(args.device)

    all_test_results = []
    for fold_idx, fold_data in enumerate(folds, start=1):
        fold_output_dir = os.path.join(args.output_dir, f'fold_{fold_idx}')
        os.makedirs(fold_output_dir, exist_ok=True)

        if fold_idx == 1:
            # Build test dataloader for fold 1
            test_df = fold_data['test']
            test_dataset = runner.MultimodalOpenSMILEDataset(
                test_df,
                config['opensmile_dir'],
                mapping_json=config.get('mapping_json'),
                tokenizer_name=config.get('roberta_model_name', 'roberta-base'),
                max_text_length=config.get('max_text_length', 128),
                audio_max_length=config.get('max_audio_length', 160000),
                sample_rate=config.get('sample_rate', 16000)
            )

            test_loader = DataLoader(
                test_dataset,
                batch_size=config['batch_size'],
                shuffle=False,
                num_workers=0,
                collate_fn=runner.collate_fn,
                pin_memory=True
            )

            # Create model and load checkpoint (weights_only=False to support older saved objects)
            model = runner.Method2CrossAttentionAVEGeMAPS(
                num_emotions=config['num_emotions'],
                egemaps_dim=config.get('prosody_dim', 88),
                hidden_dim=config['hidden_dim'],
                num_transformer_layers=config.get('num_transformer_layers', 4),
                num_heads=config.get('num_heads', 4),
                num_cross_attn_heads=config.get('num_cross_attn_heads', 8),
                ff_dim=config.get('ff_dim', 512),
                dropout=config.get('dropout', 0.1),
            ).to(device)

            best_path = os.path.join(fold_output_dir, 'best_model.pt')
            if not os.path.exists(best_path):
                best_path = os.path.join(fold_output_dir, 'last_model.pt')
                if not os.path.exists(best_path):
                    raise FileNotFoundError(f"No checkpoint found for fold 1 in {fold_output_dir}")

            checkpoint = torch.load(best_path, map_location=device, weights_only=False)
            model.load_state_dict(checkpoint['model_state_dict'])

            criterion_emotion = torch.nn.CrossEntropyLoss()
            criterion_av = torch.nn.MSELoss()

            test_metrics = runner.validate(model, test_loader, criterion_emotion, criterion_av, device, None, 0, phase='Test')

            print(f"Fold 1 test results: {test_metrics}")
            all_test_results.append(test_metrics)

        else:
            # For folds 2..5, call the original train_fold to run training+eval
            test_metrics = runner.train_fold(fold_data, config, fold_idx, args.output_dir, device)
            all_test_results.append(test_metrics)

    # Print summary
    accuracies = [r['emotion_acc'] for r in all_test_results]
    print(f"Mean Accuracy: {np.mean(accuracies):.4f} ± {np.std(accuracies):.4f}")


if __name__ == '__main__':
    main()
