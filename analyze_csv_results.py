#!/usr/bin/env python3
"""
Script to analyze all CSV evaluation results and compute averages across subjects.
"""

import pandas as pd
from pathlib import Path
import numpy as np
from typing import Dict, List
import re

def parse_filename(filename: str) -> Dict[str, str]:
    """Parse CSV filename to extract configuration parameters."""
    # Remove .csv extension
    name = filename.replace('.csv', '')
    
    # Pattern: eval_brain_f1_{classifier}_{feature}_{validation_len}_{cmmn_filter}.csv
    parts = name.split('_')
    
    config = {}
    
    # Extract classifier type
    if 'random_forest' in name:
        config['classifier'] = 'random_forest'
        start_idx = 4
    elif 'ensembled_logistic' in name:
        config['classifier'] = 'ensembled_logistic'
        start_idx = 4
    elif 'logistic' in name:
        config['classifier'] = 'logistic'
        start_idx = 4
    else:
        config['classifier'] = 'unknown'
        start_idx = 4
    
    # Extract feature type
    if 'bowav_psd_autocorr' in name:
        config['feature'] = 'bowav_psd_autocorr'
    elif 'psd_autocorr' in name:
        config['feature'] = 'psd_autocorr'
    elif 'bowav' in name:
        config['feature'] = 'bowav'
    else:
        config['feature'] = 'unknown'
    
    # Extract validation segment length
    if '_300_' in name or name.endswith('_300'):
        config['validation_length'] = '300'
    elif '_None_' in name or '_none_' in name or name.endswith('_none') or name.endswith('_None'):
        config['validation_length'] = 'None'
    else:
        config['validation_length'] = 'unknown'
    
    # Extract CMMN filter
    if 'cmmn-subj_to_subj-resampled' in name:
        config['cmmn_filter'] = 'subj_to_subj-resampled'
    elif 'cmmn-subj_to_subj' in name:
        config['cmmn_filter'] = 'subj_to_subj'
    elif 'cmmn-original-resampled' in name:
        config['cmmn_filter'] = 'original-resampled'
    elif 'cmmn-original' in name:
        config['cmmn_filter'] = 'original'
    else:
        config['cmmn_filter'] = 'none'
    
    # Check if classifier trained on filtered data
    config['trained_on_filtered'] = 'clf-trained-on-filtered-data' in name
    
    return config

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
    
    # Find all CSV files
    csv_files = list(results_dir.glob("*.csv"))
    print(f"Found {len(csv_files)} CSV files")
    
    # Dictionary to store all results
    all_results = {}
    
    for csv_file in csv_files:
        print(f"\nProcessing: {csv_file.name}")
        
        # Parse filename to get configuration
        config = parse_filename(csv_file.name)
        print(f"  Config: {config}")
        
        try:
            # Read CSV file
            df = pd.read_csv(csv_file)
            
            # Check if it has the expected columns
            required_cols = ["Prediction window [minutes]", "Subject ID", "Brain F1 score"]
            if not all(col in df.columns for col in required_cols):
                print(f"  WARNING: Missing required columns in {csv_file.name}")
                continue
            
            # Compute statistics
            stats = compute_stats(df)
            
            # Store results
            key = f"{config['classifier']}_{config['feature']}_{config['validation_length']}_{config['cmmn_filter']}"
            if config['trained_on_filtered']:
                key += "_filtered"
            
            all_results[key] = {
                'config': config,
                'stats': stats,
                'filename': csv_file.name
            }
            
            print(f"  Processed {len(df)} rows, {len(stats)} prediction windows")
            print(f"  Mean F1 range: {stats['Mean F1'].min():.3f} - {stats['Mean F1'].max():.3f}")
            
        except Exception as e:
            print(f"  ERROR processing {csv_file.name}: {e}")
    
    print(f"\n{'='*80}")
    print("SUMMARY OF ALL RESULTS")
    print(f"{'='*80}")
    
    # Sort results for better display
    sorted_keys = sorted(all_results.keys())
    
    for key in sorted_keys:
        result = all_results[key]
        config = result['config']
        stats = result['stats']
        
        print(f"\nConfiguration: {key}")
        print(f"  Classifier: {config['classifier']}")
        print(f"  Feature: {config['feature']}")
        print(f"  Validation Length: {config['validation_length']}")
        print(f"  CMMN Filter: {config['cmmn_filter']}")
        print(f"  Trained on Filtered: {config['trained_on_filtered']}")
        print(f"  File: {result['filename']}")
        
        # Show key statistics
        overall_mean = stats['Mean F1'].mean()
        overall_std = stats['Mean F1'].std()
        max_f1 = stats['Mean F1'].max()
        max_f1_window = stats.loc[stats['Mean F1'].idxmax(), 'Prediction window [minutes]']
        
        print(f"  Overall Mean F1: {overall_mean:.3f} ± {overall_std:.3f}")
        print(f"  Best F1: {max_f1:.3f} at {max_f1_window:.1f} minutes")
        print(f"  Prediction windows: {len(stats)} ({stats['Prediction window [minutes]'].min():.1f} - {stats['Prediction window [minutes]'].max():.1f} min)")
        
        # Show detailed results for each prediction window
        print(f"  Detailed results:")
        for _, row in stats.iterrows():
            print(f"    {row['Prediction window [minutes]']:6.1f} min: {row['Mean F1']:.3f} ± {row['Std F1']:.3f} (n={row['N subjects']:.0f})")
    
    print(f"\n{'='*80}")
    print("HOW STANDARD DEVIATION WAS CALCULATED")
    print(f"{'='*80}")
    print("""
Based on the code analysis, here's how standard deviation was calculated:

1. **Individual Subject F1 Scores**: For each prediction window and each subject,
   the Brain F1 score was computed using sklearn.metrics.f1_score with:
   - labels=[0] (brain class only)
   - average=None (returns F1 for each label)
   - Only expert-labeled ICs were considered (expert_label_mask)

2. **Aggregation Across Subjects**: For each prediction window, the standard 
   deviation was computed using pandas groupby:
   
   std_df = df.groupby("Prediction window [minutes]")["Brain F1 score"].std()
   
   This computes the sample standard deviation (ddof=1 by default) across all 
   12 subjects for each prediction window.

3. **Error Bounds in Plots**: The plots used "error areas" showing mean ± std,
   which represents the spread of F1 scores across subjects for each time window.

Key points:
- Standard deviation is calculated ACROSS SUBJECTS (not across ICs)
- Each subject contributes one F1 score per prediction window
- The std reflects inter-subject variability in classifier performance
- Error bounds show the typical range of performance across the 12 subjects
    """)

if __name__ == "__main__":
    main() 