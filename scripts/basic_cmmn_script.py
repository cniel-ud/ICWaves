#!/usr/bin/env python
"""
Basic CMMN Script - Works with .mat, .npy, and .npz files
"""

import sys
from pathlib import Path
# Add parent directory to path to import icwaves
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from icwaves.cmmn import CMMNProcessor

# Define your data directories here
SOURCE_DATA_DIR = "/work/cniel/data/emotion_study/raw_data_and_IC_labels"  # Directory with source EEG files (.mat, .npy, or .npz)
TARGET_DATA_DIR = "/work/cniel/data/epic/raw_data_and_IC_labels"  # Directory with target EEG files (.mat, .npy, or .npz)
OUTPUT_DIR = "./cmmn_filters"             # Where to save the filters
SAMPLING_RATE = 256                       # Sampling rate in Hz (change if different)

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
    print("="*60)

    # 1. Load source data
    print(f"\nLoading source data from: {SOURCE_DATA_DIR}")
    source_data = load_data_from_directory(SOURCE_DATA_DIR)
    print(f"  Loaded {len(source_data)} subjects")

    # 2. Load target data
    print(f"\nLoading target data from: {TARGET_DATA_DIR}")
    target_data = load_data_from_directory(TARGET_DATA_DIR)
    print(f"  Loaded {len(target_data)} subjects")

    # 3. Create processor with detected or specified sampling rate
    print(f"\nCreating CMMN processor:")
    print(f"  Method: normed-barycenter")
    print(f"  Sampling rate: {SAMPLING_RATE} Hz")

    processor = CMMNProcessor(
        method='normed-barycenter',
        sampling_rate=SAMPLING_RATE,
        nperseg=min(SAMPLING_RATE, 256)  # Use appropriate segment length
    )

    # 4. Fit filters from source to target
    print("\nFitting CMMN filters...")
    processor.fit(source_data, target_data)

    # 5. Transform target data
    print("\nApplying filters to target data...")
    filtered_data = processor.transform(target_data)

    # 6. Save filters for later use
    print(f"\nSaving filters to: {OUTPUT_DIR}")
    processor.save_filters(OUTPUT_DIR)

    print("\n" + "="*60)
    print("COMPLETE!")
    print("="*60)
    print(f"Processed {len(filtered_data)} subjects")
    print(f"Filters saved to {OUTPUT_DIR}/")
    print(f"\nTo apply these filters later:")
    print(f"  processor = CMMNProcessor()")
    print(f"  processor.load_filters('{OUTPUT_DIR}')")
    print(f"  filtered = processor.transform(new_data)")

if __name__ == "__main__":
    main()