#!/usr/bin/env python3
"""
Resume ASR Method2 W-PRO K-Fold Training from a specific fold
Usage: python scripts/resume_asr_training.py --fold 4 --gpu 0
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path

def check_fold_completed(output_dir, fold_id):
    """Check if a fold has been completed"""
    fold_dir = Path(output_dir) / f"fold_{fold_id}"
    best_model = fold_dir / "best_model.pt"
    return best_model.exists()

def get_completed_folds(output_dir, num_folds=5):
    """Get list of completed folds"""
    completed = []
    for fold_id in range(1, num_folds + 1):
        if check_fold_completed(output_dir, fold_id):
            completed.append(fold_id)
    return completed

def main():
    parser = argparse.ArgumentParser(description='Resume ASR Method2 W-PRO K-Fold Training')
    parser.add_argument('--fold', type=int, required=True, 
                        help='Fold number to resume from (1-5)')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU ID to use (default: 0)')
    parser.add_argument('--config', type=str, 
                        default='configs/method2_wpro_asr_kfold_config.yaml',
                        help='Config file path')
    parser.add_argument('--csv', type=str,
                        default='datasets/IEMOCAP_KFOLD/tenfolds/all_asr.csv',
                        help='CSV file path')
    parser.add_argument('--output_dir', type=str,
                        default='checkpoints/method2_kfold_wpro_asr',
                        help='Output directory')
    parser.add_argument('--nice', type=int, default=10,
                        help='Nice level (default: 10)')
    parser.add_argument('--dry_run', action='store_true',
                        help='Print command without executing')
    
    args = parser.parse_args()
    
    # Validate fold number
    if args.fold < 1 or args.fold > 5:
        print(f"Error: Fold number must be between 1 and 5, got {args.fold}")
        sys.exit(1)
    
    # Check workspace
    workspace = Path(__file__).parent.parent
    if not (workspace / 'train' / 'train_method2_kfold_wpro_asr.py').exists():
        print(f"Error: Training script not found in {workspace}")
        sys.exit(1)
    
    # Check completed folds
    completed_folds = get_completed_folds(args.output_dir)
    print(f"Completed folds: {completed_folds if completed_folds else 'None'}")
    
    if args.fold <= len(completed_folds):
        print(f"Warning: Fold {args.fold} appears to be completed already")
        print(f"Checkpoint exists: {workspace / args.output_dir / f'fold_{args.fold}' / 'best_model.pt'}")
        response = input(f"Continue anyway? (y/n): ")
        if response.lower() != 'y':
            print("Aborted")
            sys.exit(0)
    
    # Prepare command
    log_file = f"logs/train_method2_kfold_wpro_asr_fold{args.fold}_cuda{args.gpu}.log"
    pid_file = f"logs/train_method2_kfold_wpro_asr_fold{args.fold}_cuda{args.gpu}.pid"
    
    cmd = [
        'nohup', 'nice', f'-n', str(args.nice),
        'python', '-u', 'train/train_method2_kfold_wpro_asr.py',
        '--config', args.config,
        '--csv', args.csv,
        '--output_dir', args.output_dir,
        '--device', f'cuda:{args.gpu}',
        '--resume_fold', str(args.fold)
    ]
    
    print("\n" + "="*70)
    print("Resume ASR Method2 W-PRO K-Fold Training")
    print("="*70)
    print(f"Resume from fold: {args.fold}")
    print(f"GPU: cuda:{args.gpu}")
    print(f"Config: {args.config}")
    print(f"CSV: {args.csv}")
    print(f"Output: {args.output_dir}")
    print(f"Log: {log_file}")
    print(f"PID: {pid_file}")
    print("\nCommand:")
    print(" ".join(cmd) + f" > {log_file} 2>&1 &")
    print("="*70)
    
    if args.dry_run:
        print("\nDry run mode - command not executed")
        return
    
    # Change to workspace directory
    os.chdir(workspace)
    
    # Execute command
    with open(log_file, 'w') as log_f:
        process = subprocess.Popen(
            cmd,
            stdout=log_f,
            stderr=subprocess.STDOUT,
            start_new_session=True
        )
    
    # Save PID
    with open(pid_file, 'w') as pid_f:
        pid_f.write(str(process.pid))
    
    print(f"\nTraining started successfully!")
    print(f"PID: {process.pid}")
    print(f"\nMonitor with:")
    print(f"  tail -f {log_file}")
    print(f"\nCheck status with:")
    print(f"  ps -p {process.pid}")
    print(f"\nStop with:")
    print(f"  kill $(cat {pid_file})")

if __name__ == '__main__':
    main()
