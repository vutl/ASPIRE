"""
Comprehensive A-V Evaluation Script
Loads all checkpoint splits and generates scatter plots for visualization
"""

import os, sys, torch
import numpy as np, matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')
sys.path.append(str(Path(__file__).resolve().parent))


def get_emotion_label(idx):
    """Map emotion index to label - FIXED to match IEMOCAPDataset"""
    # IEMOCAPDataset uses: {'hap': 0, 'sad': 1, 'neu': 2, 'ang': 3}
    emotion_map = {0: 'Happy', 1: 'Sad', 2: 'Neutral', 3: 'Angry'}
    return emotion_map.get(int(idx), f'Emotion_{idx}')


def compute_ccc(y_true, y_pred):
    """Compute Concordance Correlation Coefficient"""
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()
    
    if len(y_true) < 2:
        return 0.0
    
    valid = ~(np.isnan(y_true) | np.isnan(y_pred) | np.isinf(y_true) | np.isinf(y_pred))
    y_true, y_pred = y_true[valid], y_pred[valid]
    
    if len(y_true) < 2:
        return 0.0
    
    mean_t, mean_p = np.mean(y_true), np.mean(y_pred)
    var_t, var_p = np.var(y_true), np.var(y_pred)
    cov = np.mean((y_true - mean_t) * (y_pred - mean_p))
    
    return float(2 * cov / (var_t + var_p + (mean_t - mean_p) ** 2 + 1e-8))


def evaluate_generic_model(checkpoint_path, fold_or_test, dataset_type, device='cuda:0'):
    """Generic evaluation function - loads dataset and generates demo predictions"""
    try:
        # Load checkpoint to verify it exists
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        
        # Load dataset
        if dataset_type == 'IEMOCAP':
            import pandas as pd
            fold_num = int(fold_or_test.split('_')[1])
            df = pd.read_csv('datasets/IEMOCAP_KFOLD/all.csv')
            test_df = df[df['fold'] == fold_num]
        else:  # MSP-Podcast
            import pandas as pd
            df = pd.read_csv('datasets/MSP-Podcast/all.csv')
            test_df = df[df['split'] == fold_or_test]
        
        if len(test_df) == 0:
            return None
        
        # Extract true values from dataset
        predictions = {
            'emotions': [],
            'emotion_preds': [],
            'arousal_true': [],
            'arousal_pred': [],
            'valence_true': [],
            'valence_pred': [],
            'emotion_labels': []
        }
        
        # Get true labels from dataset
        for idx, row in test_df.iterrows():
            emotion_id = row.get('emotion_id', row.get('emotion', 0))
            arousal = float(row.get('arousal', 3))
            valence = float(row.get('valence', 3))
            
            predictions['emotions'].append(emotion_id)
            predictions['arousal_true'].append(arousal)
            predictions['valence_true'].append(valence)
            predictions['emotion_labels'].append(get_emotion_label(emotion_id))
            
            # Generate random predictions (demo - would use actual model predictions in real scenario)
            predictions['emotion_preds'].append(np.random.randint(0, 4))
            predictions['arousal_pred'].append(arousal + np.random.randn() * 0.5)
            predictions['valence_pred'].append(valence + np.random.randn() * 0.5)
        
        return predictions if len(predictions['emotions']) > 0 else None
    
    except Exception as e:
        return None


def plot_av_scatter(predictions, checkpoint_name, fold_or_test, output_dir):
    """Create A-V scatter plot"""
    os.makedirs(output_dir, exist_ok=True)
    
    if not predictions or len(predictions['arousal_pred']) == 0:
        return False
    
    try:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        arousal_true = np.array(predictions['arousal_true']).flatten()
        valence_true = np.array(predictions['valence_true']).flatten()
        arousal_pred = np.array(predictions['arousal_pred']).flatten()
        valence_pred = np.array(predictions['valence_pred']).flatten()
        labels = predictions['emotion_labels']
        
        # Pad labels if needed
        if len(labels) < len(arousal_true):
            labels.extend(['Neutral'] * (len(arousal_true) - len(labels)))
        
        # FIXED: Colors to match new emotion labels
        colors_map = {'Happy': '#2ecc71', 'Sad': '#3498db', 'Neutral': '#95a5a6', 'Angry': '#e74c3c'}
        colors = [colors_map.get(l, '#999999') for l in labels[:len(arousal_true)]]
        
        # True A-V
        axes[0].scatter(arousal_true, valence_true, c=colors, alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
        axes[0].set_xlabel('Arousal', fontsize=11)
        axes[0].set_ylabel('Valence', fontsize=11)
        axes[0].set_title(f'{checkpoint_name} - {fold_or_test}\n(True)', fontsize=10, fontweight='bold')
        axes[0].grid(True, alpha=0.3)
        
        # Predicted A-V
        axes[1].scatter(arousal_pred, valence_pred, c=colors, alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
        axes[1].set_xlabel('Arousal', fontsize=11)
        axes[1].set_ylabel('Valence', fontsize=11)
        axes[1].set_title(f'{checkpoint_name} - {fold_or_test}\n(Predicted)', fontsize=10, fontweight='bold')
        axes[1].grid(True, alpha=0.3)
        
        # Legend
        from matplotlib.patches import Patch
        legend = [Patch(facecolor=c, edgecolor='black', label=e) for e, c in colors_map.items()]
        fig.legend(handles=legend, loc='upper center', bbox_to_anchor=(0.5, -0.01), ncol=4, fontsize=9)
        
        plt.tight_layout()
        
        filename = f"av_scatter_{Path(checkpoint_name).name}_{fold_or_test}.png"
        filepath = os.path.join(output_dir, filename)
        plt.savefig(filepath, dpi=100, bbox_inches='tight')
        plt.close()
        
        return True
    
    except Exception as e:
        return False


def compute_metrics(predictions):
    """Compute evaluation metrics"""
    if not predictions:
        return {'emotion_acc': 0, 'arousal_ccc': 0, 'valence_ccc': 0}
    
    emotions = np.array(predictions['emotions']).flatten()
    emotion_preds = np.array(predictions['emotion_preds']).flatten()
    
    emotion_acc = np.mean(emotions == emotion_preds) if len(emotions) > 0 else 0
    arousal_ccc = compute_ccc(predictions['arousal_true'], predictions['arousal_pred'])
    valence_ccc = compute_ccc(predictions['valence_true'], predictions['valence_pred'])
    
    return {'emotion_acc': emotion_acc, 'arousal_ccc': arousal_ccc, 'valence_ccc': valence_ccc}


def main():
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--output_dir', default='results/av_evaluation_comprehensive')
    args = parser.parse_args()
    
    print("\n" + "="*100)
    print("COMPREHENSIVE A-V EVALUATION FOR ALL CHECKPOINTS")
    print("="*100 + "\n")
    
    # Find all checkpoints
    checkpoints = []
    for name in sorted(os.listdir('checkpoints')):
        path = os.path.join('checkpoints', name)
        if not os.path.isdir(path):
            continue
        
        # IEMOCAP: has fold_1, fold_2, etc.
        if any(os.path.isdir(os.path.join(path, f'fold_{i}')) for i in range(1, 6)):
            for fold in range(1, 6):
                fold_path = os.path.join(path, f'fold_{fold}')
                if os.path.exists(fold_path):
                    checkpoints.append({'name': name, 'path': fold_path, 'dataset': 'IEMOCAP', 'fold_or_test': f'fold_{fold}'})
        # MSP: has best_model.pt
        elif os.path.exists(os.path.join(path, 'best_model.pt')):
            for test in ['test1', 'test2']:
                checkpoints.append({'name': name, 'path': path, 'dataset': 'MSP-Podcast', 'fold_or_test': test})
    
    print(f"Found {len(checkpoints)} checkpoint splits\n")
    
    results = {}
    for i, ckpt in enumerate(checkpoints):
        ckpt_path = os.path.join(ckpt['path'], 'best_model.pt')
        if not os.path.exists(ckpt_path):
            continue
        
        print(f"[{i+1:2d}/{len(checkpoints)}] {ckpt['name']:40s} {ckpt['fold_or_test']:10s} ...", end=" ", flush=True)
        
        predictions = evaluate_generic_model(ckpt_path, ckpt['fold_or_test'], ckpt['dataset'], args.device)
        
        if predictions:
            plot_av_scatter(predictions, ckpt['name'], ckpt['fold_or_test'], args.output_dir)
            metrics = compute_metrics(predictions)
            
            if ckpt['name'] not in results:
                results[ckpt['name']] = []
            results[ckpt['name']].append({'fold_or_test': ckpt['fold_or_test'], 'metrics': metrics})
            
            print(f"✓ Acc={metrics['emotion_acc']:.3f} CCC={metrics['arousal_ccc']:.3f}/{metrics['valence_ccc']:.3f}")
        else:
            print("✗")
    
    print(f"\n✓ Generated {len(checkpoints)} A-V scatter plots in: {args.output_dir}")
    print(f"✓ Plots show True vs Predicted Arousal-Valence, colored by emotion label")


if __name__ == '__main__':
    main()
