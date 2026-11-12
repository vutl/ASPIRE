import pandas as pd

def analyze_mismatch():
    tenfold_path = 'datasets/IEMOCAP_KFOLD/tenfolds/all.csv'
    fivefold_path = 'datasets/IEMOCAP_KFOLD/all.csv'
    
    print(f"Loading {tenfold_path}...")
    df_10 = pd.read_csv(tenfold_path)
    print(f"10-fold shape: {df_10.shape}")
    
    print(f"Loading {fivefold_path}...")
    df_5 = pd.read_csv(fivefold_path)
    print(f"5-fold shape: {df_5.shape}")
    
    # Find samples in 10-fold that are NOT in 5-fold
    # Using 'path' as key
    missing_mask = ~df_10['path'].isin(df_5['path'])
    missing_df = df_10[missing_mask]
    
    print(f"\nFound {len(missing_df)} samples present in 10-fold but missing in 5-fold.")
    
    if len(missing_df) > 0:
        print("\n--- Analysis of Missing Samples ---")
        
        # Check Emotion Distribution
        print("\nEmotion Distribution of Missing Samples:")
        print(missing_df['emotion'].value_counts())
        
        # Check Session Distribution (using fold mapping)
        fold_to_session = {
            1: 1, 2: 1, 3: 2, 4: 2, 5: 3, 6: 3, 7: 4, 8: 4, 9: 5, 10: 5
        }
        missing_df['mapped_session'] = missing_df['fold'].map(fold_to_session)
        print("\nSession Distribution of Missing Samples:")
        print(missing_df['mapped_session'].value_counts().sort_index())
        
        # Check if specific patterns in sample names
        print("\nSample Name Examples:")
        print(missing_df['sample_name'].head())
        
        # Compare with kept samples
        print("\n--- Comparison with Kept Samples (5-fold) ---")
        print("Emotion Distribution of Kept Samples:")
        print(df_5['emotion'].value_counts())

if __name__ == "__main__":
    analyze_mismatch()
