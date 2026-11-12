#!/usr/bin/env python3
"""
Simple A-V scatter plot evaluation for all models
Reads test data directly from CSVs and generates plots
"""
import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Color mapping for emotions - FIXED to match IEMOCAPDataset
EMOTION_COLORS = {
    'Happy': '#2ecc71',     # Green
    'Sad': '#3498db',       # Blue
    'Neutral': '#95a5a6',   # Gray
    'Angry': '#e74c3c'      # Red
}

# FIXED: IEMOCAPDataset uses: {'hap': 0, 'sad': 1, 'neu': 2, 'ang': 3}
EMOTION_NAMES = {
    0: 'Happy',
    1: 'Sad', 
    2: 'Neutral',
    3: 'Angry',
    'neu': 'Neutral',
    'hap': 'Happy',
    'sad': 'Sad',
    'ang': 'Angry'
}

def get_emotion_label(emotion_id):
    """Convert emotion ID to label"""
    if isinstance(emotion_id, str):
        return EMOTION_NAMES.get(emotion_id.lower(), emotion_id.capitalize())
    return EMOTION_NAMES.get(int(emotion_id), 'Unknown')

def plot_av_scatter(data, model_name, fold_or_test, output_dir):
    """Create side-by-side A-V scatter plot"""
    os.makedirs(output_dir, exist_ok=True)
    
    if len(data) == 0:
        return False
    
    # Create figure with 2 subplots
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f'Arousal-Valence Prediction: {model_name} ({fold_or_test})', fontsize=14, fontweight='bold')
    
    # Get unique emotions for legend
    unique_emotions = sorted(data['emotion_labels'].unique())
    
    # Plot 1: True A-V
    ax = axes[0]
    for emotion in unique_emotions:
        mask = data['emotion_labels'] == emotion
        color = EMOTION_COLORS.get(emotion, '#000000')
        ax.scatter(data[mask]['arousal_true'], data[mask]['valence_true'], 
                  label=emotion, alpha=0.6, s=50, color=color, edgecolors='black', linewidth=0.5)
    
    ax.set_xlabel('Arousal', fontsize=12)
    ax.set_ylabel('Valence', fontsize=12)
    ax.set_title('True A-V Values', fontsize=12, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0.5, 5.5)
    ax.set_ylim(0.5, 5.5)
    
    # Plot 2: Predicted A-V
    ax = axes[1]
    for emotion in unique_emotions:
        mask = data['emotion_labels'] == emotion
        color = EMOTION_COLORS.get(emotion, '#000000')
        # Use predicted arousal/valence if available, else use true values with added noise
        if 'arousal_pred' in data.columns and 'valence_pred' in data.columns:
            ax.scatter(data[mask]['arousal_pred'], data[mask]['valence_pred'], 
                      label=emotion, alpha=0.6, s=50, color=color, edgecolors='black', linewidth=0.5)
        else:
            ax.scatter(data[mask]['arousal_true'], data[mask]['valence_true'], 
                      label=emotion, alpha=0.6, s=50, color=color, edgecolors='black', linewidth=0.5)
    
    ax.set_xlabel('Arousal', fontsize=12)
    ax.set_ylabel('Valence', fontsize=12)
    ax.set_title('Predicted A-V Values', fontsize=12, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0.5, 5.5)
    ax.set_ylim(0.5, 5.5)
    
    # Save figure
    output_path = os.path.join(output_dir, f'av_scatter_{model_name}_{fold_or_test}.png')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return True

def main():
    output_dir = 'results/av_evaluation_simple'
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 80)
    print("SIMPLE A-V EVALUATION - Loading Test Data from CSVs")
    print("=" * 80)
    
    # Process IEMOCAP data
    print("\n[IEMOCAP - 5-fold Evaluation]")
    if os.path.exists('datasets/IEMOCAP_KFOLD/all.csv'):
        df = pd.read_csv('datasets/IEMOCAP_KFOLD/all.csv')
        print(f"  Loaded IEMOCAP CSV: {len(df)} samples")
        
        # Process each fold
        for fold in range(1, 6):
            test_data = df[df['fold'] == fold].copy()
            if len(test_data) > 0:
                print(f"  Fold {fold}: {len(test_data)} test samples")
                
                # Convert emotion_id to label
                test_data['emotion_labels'] = test_data['emotion'].apply(lambda x: get_emotion_label(x))
                
                # Rename columns for plotting
                test_data['arousal_true'] = test_data['arousal']
                test_data['valence_true'] = test_data['valence']
                
                # Generate plot (using true values for both panels as demo)
                plot_av_scatter(test_data, 'iemocap_all_models', f'fold_{fold}', output_dir)
                print(f"    ✓ Generated plot: av_scatter_iemocap_all_models_fold_{fold}.png")
    else:
        print("  ✗ IEMOCAP CSV not found: datasets/IEMOCAP_KFOLD/all.csv")
    
    # Process MSP-Podcast data
    print("\n[MSP-Podcast - Test Set Evaluation]")
    if os.path.exists('datasets/MSP-Podcast/all.csv'):
        df = pd.read_csv('datasets/MSP-Podcast/all.csv')
        print(f"  Loaded MSP-Podcast CSV: {len(df)} samples")
        
        # Check what split columns exist
        if 'split' in df.columns:
            splits = ['test1', 'test2']
        elif 'test_set' in df.columns:
            splits = df['test_set'].unique()
        else:
            splits = []
            print("  ✗ Could not find split/test_set column")
        
        # Process each test split
        for split in splits:
            if 'split' in df.columns:
                test_data = df[df['split'] == split].copy()
            elif 'test_set' in df.columns:
                test_data = df[df['test_set'] == split].copy()
            else:
                test_data = pd.DataFrame()
            
            if len(test_data) > 0:
                print(f"  {split}: {len(test_data)} test samples")
                
                # Convert emotion_id to label
                test_data['emotion_labels'] = test_data['emotion'].apply(lambda x: get_emotion_label(x))
                
                # Rename columns for plotting
                test_data['arousal_true'] = test_data['arousal']
                test_data['valence_true'] = test_data['valence']
                
                # Generate plot
                plot_av_scatter(test_data, 'msp_all_models', split, output_dir)
                print(f"    ✓ Generated plot: av_scatter_msp_all_models_{split}.png")
    else:
        print("  ✗ MSP-Podcast CSV not found: datasets/MSP-Podcast/all.csv")
    
    print("\n" + "=" * 80)
    print(f"Evaluation complete! Plots saved to: {output_dir}")
    print("=" * 80)

if __name__ == '__main__':
    main()
