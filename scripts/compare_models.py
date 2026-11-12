"""
Comparison and evaluation script for trained models
"""
import os
import sys
import argparse
import torch
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.method1_triple_wavlm import Method1TripleWavLM
from models.method2_crossattn_av import Method2CrossAttentionAV
from data.iemocap_dataset import get_iemocap_dataloaders


def load_checkpoint(checkpoint_path, model):
    """Load model from checkpoint"""
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    return model, checkpoint


def evaluate_model(model, test_loader, device):
    """Evaluate model on test set"""
    model.eval()
    
    all_preds = []
    all_labels = []
    all_arousal_preds = []
    all_arousal_targets = []
    all_valence_preds = []
    all_valence_targets = []
    
    with torch.no_grad():
        for batch in test_loader:
            audio = batch['audio'].to(device)
            emotion_labels = batch['emotion_label']
            arousal_values = batch['arousal_value']
            valence_values = batch['valence_value']
            
            outputs = model(audio)
            
            emotion_preds = torch.argmax(outputs['emotion_logits'], dim=1)
            all_preds.extend(emotion_preds.cpu().numpy())
            all_labels.extend(emotion_labels.numpy())
            all_arousal_preds.extend(outputs['arousal_value'].cpu().numpy().flatten())
            all_arousal_targets.extend(arousal_values.numpy().flatten())
            all_valence_preds.extend(outputs['valence_value'].cpu().numpy().flatten())
            all_valence_targets.extend(valence_values.numpy().flatten())
    
    # Compute metrics
    acc = accuracy_score(all_labels, all_preds)
    f1_weighted = f1_score(all_labels, all_preds, average='weighted')
    f1_macro = f1_score(all_labels, all_preds, average='macro')
    
    # A-V CCC
    def compute_ccc(pred, target):
        pred_mean = np.mean(pred)
        target_mean = np.mean(target)
        pred_var = np.var(pred)
        target_var = np.var(target)
        covariance = np.mean((np.array(pred) - pred_mean) * (np.array(target) - target_mean))
        ccc = (2 * covariance) / (pred_var + target_var + (pred_mean - target_mean) ** 2 + 1e-8)
        return ccc
    
    arousal_ccc = compute_ccc(all_arousal_preds, all_arousal_targets)
    valence_ccc = compute_ccc(all_valence_preds, all_valence_targets)
    
    # MSE
    arousal_mse = np.mean((np.array(all_arousal_preds) - np.array(all_arousal_targets)) ** 2)
    valence_mse = np.mean((np.array(all_valence_preds) - np.array(all_valence_targets)) ** 2)
    
    return {
        'accuracy': acc,
        'f1_weighted': f1_weighted,
        'f1_macro': f1_macro,
        'arousal_ccc': arousal_ccc,
        'valence_ccc': valence_ccc,
        'arousal_mse': arousal_mse,
        'valence_mse': valence_mse,
        'predictions': all_preds,
        'labels': all_labels
    }


def compare_models(method1_checkpoint, method2_checkpoint, test_csv, device='cuda:0'):
    """Compare two methods"""
    print("="*60)
    print("Comparing Method 1 vs Method 2")
    print("="*60)
    
    # Load test data
    _, test_loader = get_iemocap_dataloaders(
        train_csv=test_csv,  # Dummy
        test_csv=test_csv,
        batch_size=8,
        num_workers=4
    )
    
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    
    results = {}
    
    # Evaluate Method 1
    if os.path.exists(method1_checkpoint):
        print("\nEvaluating Method 1...")
        model1 = Method1TripleWavLM(num_emotions=4, hidden_dim=256)
        model1, ckpt1 = load_checkpoint(method1_checkpoint, model1)
        model1 = model1.to(device)
        results['method1'] = evaluate_model(model1, test_loader, device)
        results['method1']['params'] = sum(p.numel() for p in model1.parameters())
        print("✅ Method 1 evaluated")
    else:
        print(f"⚠️ Method 1 checkpoint not found: {method1_checkpoint}")
    
    # Evaluate Method 2
    if os.path.exists(method2_checkpoint):
        print("\nEvaluating Method 2...")
        model2 = Method2CrossAttentionAV(num_emotions=4, hidden_dim=256)
        model2, ckpt2 = load_checkpoint(method2_checkpoint, model2)
        model2 = model2.to(device)
        results['method2'] = evaluate_model(model2, test_loader, device)
        results['method2']['params'] = sum(p.numel() for p in model2.parameters())
        print("✅ Method 2 evaluated")
    else:
        print(f"⚠️ Method 2 checkpoint not found: {method2_checkpoint}")
    
    # Print comparison
    print("\n" + "="*60)
    print("Results Comparison")
    print("="*60)
    
    if 'method1' in results and 'method2' in results:
        df = pd.DataFrame({
            'Metric': ['Accuracy', 'F1 Weighted', 'F1 Macro', 
                      'Arousal CCC', 'Valence CCC', 
                      'Arousal MSE', 'Valence MSE',
                      'Parameters'],
            'Method 1': [
                f"{results['method1']['accuracy']:.4f}",
                f"{results['method1']['f1_weighted']:.4f}",
                f"{results['method1']['f1_macro']:.4f}",
                f"{results['method1']['arousal_ccc']:.4f}",
                f"{results['method1']['valence_ccc']:.4f}",
                f"{results['method1']['arousal_mse']:.4f}",
                f"{results['method1']['valence_mse']:.4f}",
                f"{results['method1']['params']:,}"
            ],
            'Method 2': [
                f"{results['method2']['accuracy']:.4f}",
                f"{results['method2']['f1_weighted']:.4f}",
                f"{results['method2']['f1_macro']:.4f}",
                f"{results['method2']['arousal_ccc']:.4f}",
                f"{results['method2']['valence_ccc']:.4f}",
                f"{results['method2']['arousal_mse']:.4f}",
                f"{results['method2']['valence_mse']:.4f}",
                f"{results['method2']['params']:,}"
            ]
        })
        
        print(df.to_string(index=False))
        
        # Determine winner
        print("\n" + "="*60)
        print("Analysis")
        print("="*60)
        
        if results['method2']['f1_weighted'] > results['method1']['f1_weighted']:
            print("🏆 Method 2 has better F1-score")
        else:
            print("🏆 Method 1 has better F1-score")
        
        param_reduction = (results['method1']['params'] - results['method2']['params']) / results['method1']['params'] * 100
        print(f"📊 Method 2 has {param_reduction:.1f}% fewer parameters")
        
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Compare trained models')
    parser.add_argument('--method1_checkpoint', type=str, required=True)
    parser.add_argument('--method2_checkpoint', type=str, required=True)
    parser.add_argument('--test_csv', type=str,
                       default='datasets/IEMOCAP_KFOLD/IEMOCAP_test4classes.csv')
    parser.add_argument('--device', type=str, default='cuda:0')
    
    args = parser.parse_args()
    
    results = compare_models(
        args.method1_checkpoint,
        args.method2_checkpoint,
        args.test_csv,
        args.device
    )
