#!/usr/bin/env python3
"""
Label Verification Script
=========================

Verify that labels are correctly extracted and stored in preprocessed segments.
"""

import pickle
import glob
from pathlib import Path
from collections import Counter, defaultdict
import sys


def check_labels(preprocessed_dir):
    """
    Verify labels in preprocessed data.
    
    Args:
        preprocessed_dir: Path to preprocessed/ directory
    """
    print("="*80)
    print("LABEL VERIFICATION")
    print("="*80)
    
    # Find all segment files
    segment_files = sorted(glob.glob(str(Path(preprocessed_dir) / '*.pkl')))
    
    if not segment_files:
        print(f"\n✗ No preprocessed segments found in {preprocessed_dir}")
        return False
    
    print(f"\n✓ Found {len(segment_files)} preprocessed segments")
    
    # Analysis containers
    labels = []
    label_by_file = defaultdict(set)
    segments_without_labels = []
    
    # Check each segment
    print("\n" + "-"*80)
    print("Checking segments...")
    print("-"*80)
    
    for seg_file in segment_files:
        try:
            with open(seg_file, 'rb') as f:
                data = pickle.load(f)
            
            # Extract label
            if 'label' in data['metadata']:
                label = data['metadata']['label']
                labels.append(label)
                
                # Track which file this came from
                original_file = Path(data['metadata']['file_path']).name
                label_by_file[original_file].add(label)
            else:
                segments_without_labels.append(Path(seg_file).name)
        
        except Exception as e:
            print(f"  ✗ Error loading {Path(seg_file).name}: {e}")
    
    # Results
    print("\n" + "="*80)
    print("RESULTS")
    print("="*80)
    
    # 1. Missing labels
    if segments_without_labels:
        print(f"\n✗ WARNING: {len(segments_without_labels)} segments missing labels!")
        print(f"  First 5: {segments_without_labels[:5]}")
        return False
    else:
        print(f"\n✓ All segments have labels")
    
    # 2. Label distribution
    print("\n" + "-"*80)
    print("Label Distribution")
    print("-"*80)
    
    label_counts = Counter(labels)
    
    for label, count in sorted(label_counts.items()):
        percentage = (count / len(labels)) * 100
        print(f"  {label:15s}: {count:6d} segments ({percentage:5.1f}%)")
    
    print(f"\n  Total: {len(labels)} segments")
    print(f"  Unique labels: {len(label_counts)}")
    
    # 3. Consistency check (same file should have same label)
    print("\n" + "-"*80)
    print("Consistency Check")
    print("-"*80)
    
    inconsistent_files = []
    for filename, file_labels in label_by_file.items():
        if len(file_labels) > 1:
            inconsistent_files.append((filename, file_labels))
    
    if inconsistent_files:
        print(f"\n✗ WARNING: {len(inconsistent_files)} files have inconsistent labels!")
        for filename, file_labels in inconsistent_files[:5]:
            print(f"  {filename}: {file_labels}")
        return False
    else:
        print(f"\n✓ All segments from same file have consistent labels")
    
    # 4. Sample verification
    print("\n" + "-"*80)
    print("Sample Verification (First 5 Segments)")
    print("-"*80)
    
    for i, seg_file in enumerate(segment_files[:5]):
        with open(seg_file, 'rb') as f:
            data = pickle.load(f)
        
        file_path = data['metadata']['file_path']
        label = data['metadata']['label']
        
        # Verify label matches path
        expected_label = None
        if 'abnormal' in file_path.lower():
            expected_label = 'abnormal'
        elif 'normal' in file_path.lower():
            expected_label = 'normal'
        
        match = "✓" if expected_label == label else "✗"
        
        print(f"\n{i+1}. {Path(seg_file).name}")
        print(f"   Source: ...{file_path[-60:]}")
        print(f"   Label: {label}")
        if expected_label:
            print(f"   Expected: {expected_label} {match}")
    
    # Final verdict
    print("\n" + "="*80)
    if len(label_counts) == 0 or 'unknown' in label_counts:
        print("⚠ WARNING: Labels may not be extracted correctly")
        print("  - Check that file paths contain label keywords")
        print("  - Verify dataset configuration in config_tuh.yaml")
        return False
    else:
        print("✓ LABEL VERIFICATION PASSED")
        return True


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Verify labels in preprocessed data')
    parser.add_argument('--preprocessed-dir', type=str, 
                       default='neurovault_data/neurovault_tuab/preprocessed',
                       help='Path to preprocessed directory')
    
    args = parser.parse_args()
    
    success = check_labels(args.preprocessed_dir)
    
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
