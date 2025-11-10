#!/usr/bin/env python
"""
Simple test to verify CMMN single-file implementation works
"""

import numpy as np
from icwaves.cmmn import CMMNProcessor, apply_filter_to_ic

def test_basic():
    """Test basic CMMN functionality with synthetic data."""
    print("Testing CMMN single-file implementation...")

    # Create synthetic data
    print("\n1. Creating synthetic data...")
    source_data = {
        'subj_00': np.random.randn(32, 1000),  # 32 channels, 1000 samples
        'subj_01': np.random.randn(32, 1000),
        'subj_02': np.random.randn(32, 1000),
    }

    target_data = {
        'subj_00': np.random.randn(32, 1000) + 0.5,  # Shifted data
        'subj_01': np.random.randn(32, 1000) + 0.5,
    }

    # Test normed-barycenter method
    print("\n2. Testing normed-barycenter method...")
    processor = CMMNProcessor(method='normed-barycenter', verbose=False)
    processor.fit(source_data, target_data)
    filtered = processor.transform(target_data)
    print(f"   Generated filters: {len(processor.filters['time'])} subjects")
    print(f"   Barycenter shape: {processor.barycenter.shape}")

    # Test subject-to-subject method
    print("\n3. Testing subject-to-subject method...")
    processor2 = CMMNProcessor(method='subj-to-subj', verbose=False)
    processor2.fit(source_data, target_data)
    filtered2 = processor2.transform(target_data)
    print(f"   Subject matches: {processor2.subj_matches}")

    # Test apply_filter_to_ic utility
    print("\n4. Testing apply_filter_to_ic utility...")
    ic_data = np.random.randn(1000)  # Single IC
    filter_array = processor.filters['time']['subj_00']
    filtered_ic = apply_filter_to_ic(ic_data, filter_array)
    print(f"   Input shape: {ic_data.shape}, Output shape: {filtered_ic.shape}")

    # Test save/load
    print("\n5. Testing save/load functionality...")
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        processor.save_filters(tmpdir)

        new_processor = CMMNProcessor(verbose=False)
        new_processor.load_filters(tmpdir)
        print(f"   Loaded {len(new_processor.filters['time'])} filters")

    print("\n✅ All tests passed! CMMN single-file implementation is working.")

if __name__ == "__main__":
    test_basic()