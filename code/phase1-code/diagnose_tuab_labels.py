#!/usr/bin/env python3
"""
TUAB Label Diagnostic Script
=============================

Check what labels are being extracted from file paths.

Usage:
    python diagnose_tuab_labels.py
"""

import pickle
from pathlib import Path
from collections import Counter

# Configuration
PREPROCESSED_DIR = r"neurovault_data\neurovault_tusl\preprocessed"
SAMPLE_SIZE = 100  # Check first 100 segments

def main():
    preprocessed_dir = Path(PREPROCESSED_DIR)
    
    if not preprocessed_dir.exists():
        print(f"❌ Directory not found: {preprocessed_dir}")
        return
    
    segment_files = sorted(list(preprocessed_dir.glob("*.pkl")))[:SAMPLE_SIZE]
    
    print(f"Checking {len(segment_files)} segments...")
    print("=" * 80)
    
    labels = []
    file_paths = []
    
    for seg_file in segment_files:
        with open(seg_file, 'rb') as f:
            data = pickle.load(f)
        
        metadata = data['metadata']
        label = metadata.get('label', 'unknown')
        file_path = metadata.get('file_path', '')
        
        labels.append(label)
        file_paths.append(file_path)
    
    # Count labels
    label_counts = Counter(labels)
    
    print("\n📊 Label Distribution (Sample):")
    print("-" * 80)
    for label, count in label_counts.most_common():
        pct = (count / len(labels)) * 100
        print(f"  {label:15s}: {count:4d} ({pct:5.1f}%)")
    print("-" * 80)
    
    # Show example file paths for each label
    print("\n📁 Example File Paths:")
    print("-" * 80)
    
    for label in label_counts.keys():
        print(f"\n{label.upper()}:")
        examples = [fp for fp, lbl in zip(file_paths, labels) if lbl == label][:3]
        for example in examples:
            print(f"  {example}")
    
    print("\n" + "=" * 80)
    
    # Test label extraction on example paths
    print("\n🧪 Testing Label Extraction Logic:")
    print("-" * 80)
    
    test_paths = [
        r"C:\Users\Pc\Desktop\tuh_data\tuab\edf\eval\abnormal\01_tcp_ar\aaaaaajp\s004_2013\aaaaaajp_s004_t000.edf",
        r"C:\Users\Pc\Desktop\tuh_data\tuab\edf\eval\normal\01_tcp_ar\aaaaaabq\s001_2013\aaaaaabq_s001_t000.edf",
        r"C:\Users\Pc\Desktop\tuh_data\tuab\edf\train\abnormal\01_tcp_ar\aaaaaalj\s001_2012\aaaaaalj_s001_t000.edf",
        r"C:\Users\Pc\Desktop\tuh_data\tuab\edf\train\normal\01_tcp_ar\aaaaaabo\s002_2012\aaaaaabo_s002_t000.edf"
    ]
    
    for test_path in test_paths:
        file_path_str = test_path.lower()
        
        if 'abnormal' in file_path_str:
            extracted_label = 'abnormal'
        elif 'normal' in file_path_str:
            extracted_label = 'normal'
        else:
            extracted_label = 'unknown'
        
        # Show which part matched
        if 'abnormal' in file_path_str:
            highlight = '...\\abnormal\\...'
        elif 'normal' in file_path_str:
            highlight = '...\\normal\\...'
        else:
            highlight = 'NO MATCH'
        
        print(f"\nPath: {Path(test_path).parts[-5:]}")
        print(f"  Match: {highlight}")
        print(f"  Label: {extracted_label}")
    
    print("\n" + "=" * 80)
    
    # Check if 'normal' paths exist in your data
    print("\n🔍 Searching for 'normal' in actual file paths:")
    print("-" * 80)
    
    normal_count = sum(1 for fp in file_paths if 'normal' in fp.lower())
    abnormal_count = sum(1 for fp in file_paths if 'abnormal' in fp.lower())
    
    print(f"File paths containing 'normal': {normal_count}")
    print(f"File paths containing 'abnormal': {abnormal_count}")
    
    if normal_count == 0:
        print("\n⚠️  WARNING: No 'normal' found in file paths!")
        print("This means you may have only processed abnormal data.")
        print("\nFirst 5 file paths:")
        for fp in file_paths[:5]:
            print(f"  {fp}")
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
