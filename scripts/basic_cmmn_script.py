#!/usr/bin/env python
"""
Basic CMMN Script - Minimal example for applying CMMN filters
"""

from pathlib import Path
from icwaves.cmmn import CMMNProcessor

# Define your data directories here
SOURCE_DATA_DIR = "/path/to/source/data"  # Directory with source domain .npy/.npz files
TARGET_DATA_DIR = "/path/to/target/data"  # Directory with target domain .npy/.npz files
OUTPUT_DIR = "./cmmn_filters"             # Where to save the filters

def main():
    # 1. Create processor
    processor = CMMNProcessor(method='normed-barycenter')

    # 2. Fit filters from source to target
    processor.fit(SOURCE_DATA_DIR, TARGET_DATA_DIR)

    # 3. Transform target data
    filtered_data = processor.transform(processor.target_data)

    # 4. Save filters for later use
    processor.save_filters(OUTPUT_DIR)

    print(f"Done! Filters saved to {OUTPUT_DIR}")
    print(f"Filtered {len(filtered_data)} subjects")

if __name__ == "__main__":
    main()