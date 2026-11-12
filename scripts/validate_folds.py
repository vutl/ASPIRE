import pandas as pd
import os

def validate_folds():
    # Paths
    tenfold_path = 'datasets/IEMOCAP_KFOLD/tenfolds/all.csv'
    fivefold_path = 'datasets/IEMOCAP_KFOLD/all.csv'
    
    print(f"Loading {tenfold_path}...")
    df_10 = pd.read_csv(tenfold_path)
    print(f"10-fold shape: {df_10.shape}")
    print(f"10-fold unique folds: {sorted(df_10['fold'].unique())}")
    
    print(f"\nLoading {fivefold_path}...")
    df_5 = pd.read_csv(fivefold_path)
    print(f"5-fold shape: {df_5.shape}")
    print(f"5-fold unique folds: {sorted(df_5['fold'].unique())}")
    
    # Mapping logic from train_method2_kfold_emotion2vec.py
    fold_to_session = {
        1: 1, 2: 1,  # Ses01
        3: 2, 4: 2,  # Ses02
        5: 3, 6: 3,  # Ses03
        7: 4, 8: 4,  # Ses04
        9: 5, 10: 5  # Ses05
    }
    
    print("\nApplying mapping to 10-fold data...")
    df_10['mapped_session'] = df_10['fold'].map(fold_to_session)
    
    # Rename 'fold' in 5-fold to 'target_session' for comparison
    # Assuming 'fold' in all.csv corresponds to session ID (1-5)
    df_5 = df_5.rename(columns={'fold': 'target_session'})
    
    # Merge on path to compare
    # We use 'path' as the unique identifier
    print("\nMerging dataframes on 'path'...")
    merged = pd.merge(
        df_10[['path', 'mapped_session', 'fold']], 
        df_5[['path', 'target_session']], 
        on='path', 
        how='inner',
        suffixes=('_10', '_5')
    )
    
    print(f"Merged shape: {merged.shape}")
    
    # Check for mismatches
    mismatches = merged[merged['mapped_session'] != merged['target_session']]
    
    if len(mismatches) == 0:
        print("\n✅ SUCCESS: Mapping logic perfectly matches the 5-fold CSV!")
    else:
        print(f"\n❌ FAILURE: Found {len(mismatches)} mismatches!")
        print(mismatches.head())
        
        # Analyze mismatches
        print("\nMismatch examples:")
        for idx, row in mismatches.head().iterrows():
            print(f"Path: {row['path']}")
            print(f"  10-fold original: {row['fold']}")
            print(f"  Mapped session: {row['mapped_session']}")
            print(f"  Target session: {row['target_session']}")

if __name__ == "__main__":
    validate_folds()
