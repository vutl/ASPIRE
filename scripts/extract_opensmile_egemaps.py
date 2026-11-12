#!/usr/bin/env python3
"""
Extract eGeMAPS v2 features using OpenSMILE for MSP-Podcast dataset
eGeMAPS v2: 88-dimensional feature set for emotion recognition
Outputs: /home4/quanpn/ICME2026/only_gay/outputs/msp_podcast/egemaps_npy
"""
import os
import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import argparse

try:
    import opensmile
except ImportError:
    print("Error: opensmile not installed!")
    print("Install with: pip install opensmile")
    sys.exit(1)


def extract_egemaps_features(
    csv_path: str,
    output_dir: str,
    feature_set: str = 'eGeMAPSv02',
    feature_level: str = 'Functionals'
):
    """
    Extract OpenSMILE features for all samples in CSV
    
    Args:
        csv_path: Path to CSV with audio file paths
        output_dir: Directory to save extracted features
        feature_set: OpenSMILE feature set ('eGeMAPSv02', 'GeMAPSv01b', etc.)
        feature_level: 'Functionals' (utterance-level) or 'LowLevelDescriptors' (frame-level)
    """
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize OpenSMILE
    print(f"Initializing OpenSMILE with {feature_set} ({feature_level})...")
    smile = opensmile.Smile(
        feature_set=feature_set,
        feature_level=feature_level,
    )
    
    # Load CSV
    print(f"Loading CSV from {csv_path}...")
    df = pd.read_csv(csv_path)
    print(f"Total samples: {len(df)}")
    
    # Check feature dimension
    test_audio = df.iloc[0]['path']
    if not os.path.exists(test_audio):
        print(f"Warning: Test audio not found: {test_audio}")
        print("Checking if path needs adjustment...")
        # Try to find audio in common locations
        possible_paths = [
            test_audio,
            os.path.join('/home4/quanpn/ICME2026/ser_ssl_prosody/data/MSP-Podcast/Audios/', os.path.basename(test_audio)),
        ]
        for p in possible_paths:
            if os.path.exists(p):
                test_audio = p
                break
    
    test_feats = smile.process_file(test_audio)
    feature_dim = test_feats.shape[1]
    print(f"Feature dimension: {feature_dim}")
    print(f"Feature names: {list(test_feats.columns)[:5]}... (showing first 5)")
    print()
    
    # Create mapping: audio_path -> feature_filename
    mapping = {}
    missing = []
    
    print("Extracting features...")
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing"):
        audio_path = row['path']
        sample_name = row['sample_name']
        
        # Check if audio exists
        if not os.path.exists(audio_path):
            missing.append(audio_path)
            continue
        
        try:
            # Extract features
            features = smile.process_file(audio_path)
            
            # Convert to numpy array (1, feature_dim)
            feat_array = features.values  # (1, feature_dim)
            
            # Save as .npy
            feat_filename = f"{sample_name}.npy"
            feat_path = os.path.join(output_dir, feat_filename)
            np.save(feat_path, feat_array)
            
            # Store mapping
            abs_path = os.path.abspath(audio_path)
            mapping[abs_path] = feat_filename
            
        except Exception as e:
            print(f"\nError processing {audio_path}: {e}")
            missing.append(audio_path)
            continue
    
    # Save mapping
    mapping_path = os.path.join(output_dir, 'mapping.json')
    with open(mapping_path, 'w') as f:
        json.dump(mapping, f, indent=2)
    
    print(f"\n{'='*80}")
    print("Extraction completed!")
    print(f"  Total samples: {len(df)}")
    print(f"  Extracted: {len(mapping)}")
    print(f"  Missing: {len(missing)}")
    print(f"  Feature dimension: {feature_dim}")
    print(f"  Output directory: {output_dir}")
    print(f"  Mapping saved to: {mapping_path}")
    print(f"{'='*80}")
    
    if missing:
        print(f"\nMissing files ({len(missing)}):")
        for m in missing[:10]:
            print(f"  {m}")
        if len(missing) > 10:
            print(f"  ... and {len(missing) - 10} more")
    
    return mapping


def main():
    parser = argparse.ArgumentParser(description="Extract OpenSMILE eGeMAPS features")
    
    parser.add_argument('--data_csv', type=str, required=True, help='Path to CSV with audio paths')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory for features')
    parser.add_argument('--feature_set', type=str, default='eGeMAPSv02', 
                        choices=['eGeMAPSv02', 'GeMAPSv01b', 'ComParE_2016'],
                        help='OpenSMILE feature set')
    parser.add_argument('--feature_level', type=str, default='Functionals',
                        choices=['Functionals', 'LowLevelDescriptors'],
                        help='Feature level (Functionals for utterance-level)')
    
    args = parser.parse_args()
    
    print("="*80)
    print("OpenSMILE Feature Extraction")
    print(f"Feature set: {args.feature_set}")
    print(f"Feature level: {args.feature_level}")
    print(f"Input CSV: {args.data_csv}")
    print(f"Output: {args.output_dir}")
    print("="*80)
    print()
    
    extract_egemaps_features(
        csv_path=args.data_csv,
        output_dir=args.output_dir,
        feature_set=args.feature_set,
        feature_level=args.feature_level
    )


if __name__ == "__main__":
    main()
