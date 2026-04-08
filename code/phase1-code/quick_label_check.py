#!/usr/bin/env python3
"""
Quick Label Check - No Arguments Needed
========================================
Fast check of label extraction from preprocessed segments.

Just run: python quick_label_check.py
"""

import pickle
from pathlib import Path
from collections import Counter
import random

# ============================================================================
# CONFIGURATION - Adjust if needed
# ============================================================================

PREPROCESSED_DIR = r"neurovault_data\neurovault_tuar\preprocessed"  # Change this
SAMPLE_SIZE = 1000

# ============================================================================

def main():
    print("=" * 70)
    print("QUICK LABEL VERIFICATION")
    print("=" * 70)
    
    preprocessed_dir = Path(PREPROCESSED_DIR)
    
    if not preprocessed_dir.exists():
        print(f"❌ Directory not found: {preprocessed_dir}")
        print(f"   Please update PREPROCESSED_DIR in this script.")
        return
    
    # Find segment files
    segment_files = list(preprocessed_dir.glob("*.pkl"))
    
    if not segment_files:
        print(f"❌ No .pkl files found in {preprocessed_dir}")
        return
    
    total = len(segment_files)
    
    # Sample
    if SAMPLE_SIZE < total:
        segment_files = random.sample(segment_files, SAMPLE_SIZE)
        print(f"\n📊 Checking {SAMPLE_SIZE:,} / {total:,} segments (random sample)")
    else:
        print(f"\n📊 Checking all {total:,} segments")
    
    # Check labels
    labels = Counter()
    missing = 0
    
    print("🔍 Verifying labels...\n")
    
    for i, seg_file in enumerate(segment_files, 1):
        try:
            with open(seg_file, 'rb') as f:
                data = pickle.load(f)
            
            label = data['metadata'].get('label', None)
            
            if label:
                labels[label] += 1
            else:
                missing += 1
                if missing <= 3:  # Show first 3 missing
                    print(f"  ⚠️  Missing: {seg_file.name}")
        
        except Exception as e:
            print(f"  ❌ Error: {seg_file.name} - {e}")
        
        # Progress
        if i % 200 == 0:
            print(f"  Checked {i:,} / {len(segment_files):,}...", end='\r')
    
    print(" " * 60, end='\r')  # Clear progress line
    
    # Results
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    
    print(f"\n✅ Segments with labels: {sum(labels.values()):,}")
    print(f"❌ Segments missing labels: {missing}")
    
    if labels:
        print(f"\n📊 Label Distribution:")
        print("-" * 50)
        total_labeled = sum(labels.values())
        for label, count in labels.most_common():
            pct = (count / total_labeled) * 100
            print(f"  {label:15s}: {count:7,d} ({pct:5.1f}%)")
        print("-" * 50)
        print(f"  {'TOTAL':15s}: {total_labeled:7,d} (100.0%)")
    
    # Status
    print("\n" + "=" * 70)
    if missing == 0:
        print("✅ SUCCESS: All labels present!")
    else:
        print(f"⚠️  WARNING: {missing} segments missing labels")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
