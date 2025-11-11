#!/usr/bin/env python
"""
Basic CMMN Script - Works with .mat, .npy, and .npz files

Dataset-specific caching: PSDs are cached separately for each dataset name,
so you can switch between datasets without clearing the cache.
Just change SOURCE_DATASET_NAME and TARGET_DATASET_NAME below.
"""

import sys
from pathlib import Path
# Add parent directory to path to import icwaves
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from icwaves.cmmn import CMMNProcessor

# Define your data directories and dataset names here
SOURCE_DATA_DIR = "/work/cniel/data/emotion_study/raw_data_and_IC_labels"  # Directory with source EEG files (.mat, .npy, or .npz)
TARGET_DATA_DIR = "/work/cniel/data/epic/raw_data_and_IC_labels"  # Directory with target EEG files (.mat, .npy, or .npz)
SOURCE_DATASET_NAME = "emotion"  # Name for source dataset cache ('emotion', 'cue', 'epic', etc.)
TARGET_DATASET_NAME = "epic"     # Name for target dataset cache ('emotion', 'cue', 'epic', etc.)
METHOD = "normed-barycenter"     # Method: 'normed-barycenter', 'unnormed-barycenter', or 'subj-to-subj'
OUTPUT_BASE_DIR = "./cmmn_filters"  # Base directory for filters
SAMPLING_RATE = 256               # Sampling rate in Hz (change if different)

def load_data_from_directory(directory):
    """Load EEG data from directory, automatically handling .mat, .npy, and .npz files.

    Args:
        directory: Path to directory containing data files

    Returns:
        Dictionary mapping subject IDs to data arrays (channels x samples)
    """
    directory = Path(directory)
    data_dict = {}

    # Check for different file types
    mat_files = list(directory.glob("*.mat"))
    npy_files = list(directory.glob("*.npy"))
    npz_files = list(directory.glob("*.npz"))

    # Load .mat files if present
    if mat_files:
        import scipy.io
        print(f"  Found {len(mat_files)} .mat files")
        for mat_file in sorted(mat_files):
            subj_id = mat_file.stem  # e.g., 'subj-000'
            mat_data = scipy.io.loadmat(mat_file)

            # Extract the data array (channels x samples)
            if 'data' in mat_data:
                data_dict[subj_id] = mat_data['data']

                # Try to get sampling rate from first file
                if 'srate' in mat_data and 'sampling_rate' not in locals():
                    global SAMPLING_RATE
                    SAMPLING_RATE = int(mat_data['srate'][0, 0])
                    print(f"  Detected sampling rate: {SAMPLING_RATE} Hz")

    # Load .npy files if present
    elif npy_files:
        print(f"  Found {len(npy_files)} .npy files")
        for npy_file in sorted(npy_files):
            subj_id = npy_file.stem
            data = np.load(npy_file)
            data_dict[subj_id] = data

    # Load .npz files if present
    elif npz_files:
        print(f"  Found {len(npz_files)} .npz files")
        for npz_file in sorted(npz_files):
            subj_id = npz_file.stem
            npz_data = np.load(npz_file)

            # Try to find the data array
            if 'data' in npz_data:
                data_dict[subj_id] = npz_data['data']
            else:
                # Use the first array in the file
                keys = list(npz_data.keys())
                if keys:
                    data_dict[subj_id] = npz_data[keys[0]]

    else:
        raise FileNotFoundError(f"No .mat, .npy, or .npz files found in {directory}")

    # Validate data shapes
    for subj_id, data in data_dict.items():
        if data.ndim != 2:
            raise ValueError(f"{subj_id}: Expected 2D array (channels x samples), got shape {data.shape}")
        print(f"    {subj_id}: {data.shape[0]} channels, {data.shape[1]} samples")

    return data_dict


def main():
    print("="*60)
    print("CMMN Filter Generation")
    print(f"Source: {SOURCE_DATASET_NAME} -> Target: {TARGET_DATASET_NAME}")
    print(f"Method: {METHOD}")
    print("="*60)

    # Create output directory based on method and dataset pair
    method_suffix = METHOD.replace('-', '_')  # Convert normed-barycenter to normed_barycenter
    if METHOD == 'normed-barycenter':
        method_suffix = 'barycenter'  # Simplify for normed-barycenter
    elif METHOD == 'unnormed-barycenter':
        method_suffix = 'unnormed_barycenter'
    elif METHOD == 'subj-to-subj':
        method_suffix = 'subj_to_subj'

    OUTPUT_DIR = Path(OUTPUT_BASE_DIR) / f"{SOURCE_DATASET_NAME}_to_{TARGET_DATASET_NAME}_{method_suffix}"
    print(f"Output directory: {OUTPUT_DIR}")

    # Check for cached PSDs using dataset names
    psd_cache_dir = Path(OUTPUT_BASE_DIR) / "psd_cache"
    source_psd_file = psd_cache_dir / SOURCE_DATASET_NAME / "all_psds.npz"
    target_psd_file = psd_cache_dir / TARGET_DATASET_NAME / "all_psds.npz"

    # Try to load cached PSDs
    source_data = None
    target_data = None

    if source_psd_file.exists():
        print(f"\nFound cached PSDs for {SOURCE_DATASET_NAME} dataset")
        source_data = 'cached'
    else:
        print(f"\nNo cached PSDs for {SOURCE_DATASET_NAME} dataset")

    if target_psd_file.exists():
        print(f"Found cached PSDs for {TARGET_DATASET_NAME} dataset")
        target_data = 'cached'
    else:
        print(f"No cached PSDs for {TARGET_DATASET_NAME} dataset")

    # Load data if not using cache
    if source_data != 'cached':
        # 1. Load source data
        print(f"\nLoading source data from: {SOURCE_DATA_DIR}")
        source_data = load_data_from_directory(SOURCE_DATA_DIR)
        print(f"  Loaded {len(source_data)} subjects")

    if target_data != 'cached':
        # 2. Load target data
        print(f"\nLoading target data from: {TARGET_DATA_DIR}")
        target_data = load_data_from_directory(TARGET_DATA_DIR)
        print(f"  Loaded {len(target_data)} subjects")

    # 3. Create processor with detected or specified sampling rate
    print(f"\nCreating CMMN processor:")
    print(f"  Method: {METHOD}")
    print(f"  Sampling rate: {SAMPLING_RATE} Hz")

    processor = CMMNProcessor(
        method=METHOD,
        sampling_rate=SAMPLING_RATE,
        nperseg=min(SAMPLING_RATE, 256)  # Use appropriate segment length
    )

    # 4. Fit filters from source to target
    print("\nFitting CMMN filters...")

    # Load cached PSDs if available
    if source_data == 'cached':
        print(f"Loading cached {SOURCE_DATASET_NAME} PSDs...")
        source_psd_data = np.load(source_psd_file, allow_pickle=True)
        processor.source_psds = {k: v for k, v in source_psd_data.items()}
        print(f"  Loaded {len(processor.source_psds)} {SOURCE_DATASET_NAME} PSDs from cache")
    else:
        # Compute source PSDs
        processor.source_data = source_data
        processor.source_psds = processor._compute_psds(source_data)
        print(f"  Computed {len(processor.source_psds)} {SOURCE_DATASET_NAME} PSDs")

    if target_data == 'cached':
        print(f"Loading cached {TARGET_DATASET_NAME} PSDs...")
        target_psd_data = np.load(target_psd_file, allow_pickle=True)
        processor.target_psds = {k: v for k, v in target_psd_data.items()}
        print(f"  Loaded {len(processor.target_psds)} {TARGET_DATASET_NAME} PSDs from cache")
    else:
        # Compute target PSDs
        processor.target_data = target_data
        processor.target_psds = processor._compute_psds(target_data)
        print(f"  Computed {len(processor.target_psds)} {TARGET_DATASET_NAME} PSDs")

    # Generate filters using the PSDs (cached or computed)
    print(f"\nGenerating {METHOD} filters...")
    if METHOD == 'normed-barycenter':
        processor._generate_normed_barycenter_filters()
    elif METHOD == 'unnormed-barycenter':
        processor._generate_unnormed_barycenter_filters()
    elif METHOD == 'subj-to-subj':
        processor._generate_subj_to_subj_filters()
        if processor.subj_matches is not None:
            print(f"  Subject matching completed: {len(processor.subj_matches)} matches")

    # Save PSDs that weren't cached
    if source_data != 'cached' or target_data != 'cached':
        print("\nSaving newly computed PSDs to cache...")

        if source_data != 'cached':
            # Save source PSDs
            source_psd_file.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(source_psd_file, **processor.source_psds)
            print(f"  Saved {len(processor.source_psds)} {SOURCE_DATASET_NAME} PSDs to cache")

        if target_data != 'cached':
            # Save target PSDs
            target_psd_file.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(target_psd_file, **processor.target_psds)
            print(f"  Saved {len(processor.target_psds)} {TARGET_DATASET_NAME} PSDs to cache")

    # 5. Transform target data
    print("\nApplying filters to target data...")
    if target_data == 'cached':
        # Need to load actual target data for transformation
        print("Loading target data for transformation...")
        target_data = load_data_from_directory(TARGET_DATA_DIR)
    filtered_data = processor.transform(target_data)

    # 6. Save filters for later use
    print(f"\nSaving filters to: {OUTPUT_DIR}")
    processor.save_filters(OUTPUT_DIR)

    print("\n" + "="*60)
    print("COMPLETE!")
    print("="*60)
    print(f"Processed {len(filtered_data)} subjects")
    print(f"Method: {METHOD}")
    print(f"Filters saved to: {OUTPUT_DIR}/")

    if METHOD == 'subj-to-subj' and processor.subj_matches is not None:
        # Save subject matches for reference
        matches_file = OUTPUT_DIR / "subject_matches.txt"
        with open(matches_file, 'w') as f:
            f.write(f"Subject-to-Subject Matching Results\n")
            f.write(f"Source: {SOURCE_DATASET_NAME}, Target: {TARGET_DATASET_NAME}\n")
            f.write("="*40 + "\n")
            target_ids = list(processor.target_psds.keys())
            source_ids = list(processor.source_psds.keys())
            for i, match_idx in enumerate(processor.subj_matches):
                if i < len(target_ids):
                    f.write(f"Target {target_ids[i]} -> Source {source_ids[match_idx]}\n")
        print(f"Subject matches saved to: {matches_file}")

    print(f"\nDataset-specific PSDs cached in:")
    print(f"  {psd_cache_dir}/{SOURCE_DATASET_NAME}/all_psds.npz")
    print(f"  {psd_cache_dir}/{TARGET_DATASET_NAME}/all_psds.npz")
    print(f"\nTo apply these filters later:")
    print(f"  processor = CMMNProcessor(method='{METHOD}')")
    print(f"  processor.load_filters('{OUTPUT_DIR}')")
    print(f"  filtered = processor.transform(new_data)")

if __name__ == "__main__":
    main()
