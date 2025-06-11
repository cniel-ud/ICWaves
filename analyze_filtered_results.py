#!/usr/bin/env python3
"""
Script to analyze specific CSV evaluation results from a manual list.
"""

import pandas as pd
from pathlib import Path
import numpy as np

# MANUAL LIST OF FILES TO ANALYZE - Edit this list as needed
# target_files = [
#     #'eval_brain_f1_random_forest_bowav_300_cmmn-original-resampled.csv',
#     'eval_brain_f1_random_forest_bowav_300_cmmn-subj_to_subj-resampled.csv',
#     'eval_brain_f1_random_forest_bowav_300_cmmn-subj_to_subj-resampled_clf-trained-on-filtered-data.csv',
#     'eval_brain_f1_random_forest_bowav_300_cmmn-subj_to_subj.csv'
# ]

# all files
target_files = [f for f in Path("results/cue/evaluation").glob("eval_brain_f1*.csv")]

def compute_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Compute mean and standard deviation across subjects for each prediction window."""
    stats = df.groupby("Prediction window [minutes]")["Brain F1 score"].agg(['mean', 'std', 'count']).reset_index()
    stats.columns = ["Prediction window [minutes]", "Mean F1", "Std F1", "N subjects"]
    return stats

def main():
    results_dir = Path("results/cue/evaluation")
    
    if not results_dir.exists():
        print(f"Results directory {results_dir} does not exist!")
        return
    
    if not target_files:
        print("No files specified in target_files list. Please add filenames to analyze.")
        return
    
    print(f"Analyzing {len(target_files)} specified files:")
    
    # Dictionary to store results
    results = {}
    
    for filename in target_files:
        # Handle both string filenames and Path objects
        if isinstance(filename, Path):
            csv_file = filename
            display_name = filename.name
        else:
            csv_file = results_dir / filename
            display_name = filename
        
        if not csv_file.exists():
            print(f"❌ File not found: {display_name}")
            continue
            
        print(f"✓ Processing: {display_name}")
        
        try:
            # Read CSV file
            df = pd.read_csv(csv_file)
            
            # Check if it has the expected columns
            required_cols = ["Prediction window [minutes]", "Subject ID", "Brain F1 score"]
            if not all(col in df.columns for col in required_cols):
                print(f"  WARNING: Missing required columns in {display_name}")
                continue
            
            # Remove duplicates (some files have duplicate entries)
            original_len = len(df)
            df = df.drop_duplicates(subset=["Prediction window [minutes]", "Subject ID"])
            if len(df) < original_len:
                print(f"  INFO: Removed {original_len - len(df)} duplicate rows")
            
            # Verify we have exactly 12 subjects per prediction window
            subjects_per_window = df.groupby("Prediction window [minutes]").size()
            if not all(subjects_per_window == 12):
                print(f"  WARNING: Inconsistent subject counts: {subjects_per_window.unique()}")
            
            # Compute statistics
            stats = compute_stats(df)
            
            # Store results using display name as key
            results[display_name] = {
                'stats': stats,
                'total_rows': len(df),
                'prediction_windows': len(stats)
            }
            
        except Exception as e:
            print(f"  ERROR processing {display_name}: {e}")
    
    print(f"\n{'='*80}")
    print(f"RESULTS FOR {len(results)} FILES")
    print(f"{'='*80}")
    
    for filename, result in results.items():
        stats = result['stats']
        
        print(f"\n{filename}")
        print(f"  Total rows: {result['total_rows']}")
        print(f"  Prediction windows: {result['prediction_windows']}")
        
        # Show key statistics
        overall_mean = stats['Mean F1'].mean()
        overall_std = stats['Mean F1'].std()
        max_f1 = stats['Mean F1'].max()
        max_f1_window = stats.loc[stats['Mean F1'].idxmax(), 'Prediction window [minutes]']
        
        print(f"  Overall Mean F1: {overall_mean:.3f} ± {overall_std:.3f}")
        print(f"  Best F1: {max_f1:.3f} at {max_f1_window:.1f} minutes")
        
        # Filter for specific prediction windows
        target_windows = [1.0, 2.0, 5.0, 15.0, 30.0, 50.0]
        filtered_stats = stats[stats['Prediction window [minutes]'].isin(target_windows)]
        
        print(f"  Results for target prediction windows (1, 2, 5, 15, 30, 50 min):")
        if len(filtered_stats) == 0:
            print("    No matching prediction windows found")
        else:
            for _, row in filtered_stats.iterrows():
                print(f"    {row['Prediction window [minutes]']:6.1f} min: {row['Mean F1']:.3f} ± {row['Std F1']:.3f} (n={row['N subjects']:.0f})")

if __name__ == "__main__":
    main() 