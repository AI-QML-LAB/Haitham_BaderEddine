#!/usr/bin/env python3
"""
TUEV Directory Structure Diagnostic
====================================

Checks what annotation files actually exist in TUEV dataset.

Usage:
    python diagnose_tuev_structure.py
"""

import pickle
from pathlib import Path
from collections import Counter

# Paths
PREPROCESSED_DIR = r"neurovault_data\neurovault_tuev\preprocessed"
TUEV_DATA_DIR = r"C:\Users\Pc\Desktop\tuh_data\tuev\edf"

def check_preprocessed_paths():
    """Check what file paths are stored in preprocessed segments."""
    print("=" * 80)
    print("PART 1: Checking Preprocessed Segment Metadata")
    print("=" * 80)
    
    preprocessed_dir = Path(PREPROCESSED_DIR)
    
    if not preprocessed_dir.exists():
        print(f"❌ Preprocessed directory not found: {preprocessed_dir}")
        return None
    
    segment_files = list(preprocessed_dir.glob("*.pkl"))[:10]  # Check first 10
    
    print(f"\nChecking first 10 segments...\n")
    
    original_paths = []
    
    for seg_file in segment_files:
        with open(seg_file, 'rb') as f:
            data = pickle.load(f)
        
        original_path = data['metadata']['file_path']
        original_paths.append(original_path)
        
        print(f"Segment: {seg_file.name}")
        print(f"  Original path: {original_path}")
        
        # Check what files exist at that location
        orig_path = Path(original_path)
        if orig_path.exists():
            parent_dir = orig_path.parent
            print(f"  Parent dir exists: ✅")
            
            # List files in parent directory
            related_files = list(parent_dir.glob(orig_path.stem + ".*"))
            print(f"  Related files:")
            for rf in related_files:
                print(f"    - {rf.name}")
        else:
            print(f"  Original file exists: ❌")
        print()
    
    return original_paths


def check_tuev_directory():
    """Check TUEV data directory structure."""
    print("=" * 80)
    print("PART 2: Checking TUEV Data Directory")
    print("=" * 80)
    
    tuev_dir = Path(TUEV_DATA_DIR)
    
    if not tuev_dir.exists():
        print(f"❌ TUEV directory not found: {tuev_dir}")
        return
    
    print(f"\n📁 TUEV Directory: {tuev_dir}\n")
    
    # Find all files
    all_files = list(tuev_dir.rglob("*.*"))
    
    # Count by extension
    extensions = Counter([f.suffix.lower() for f in all_files if f.is_file()])
    
    print("📊 Files by Extension:")
    print("-" * 80)
    for ext, count in extensions.most_common():
        print(f"  {ext:15s}: {count:6,d} files")
    
    # Show example directory structure
    print("\n📂 Example Directory Structure:")
    print("-" * 80)
    
    # Find first EDF file and show its directory
    edf_files = list(tuev_dir.rglob("*.edf"))[:3]
    
    for edf_file in edf_files:
        print(f"\nEDF: {edf_file.name}")
        print(f"  Path: {edf_file}")
        
        # Show all files in same directory
        related = list(edf_file.parent.glob("*"))
        print(f"  Files in same directory:")
        for f in related:
            if f.is_file():
                print(f"    - {f.name} ({f.stat().st_size} bytes)")
    
    # Look for annotation files
    print("\n🔍 Searching for Annotation Files:")
    print("-" * 80)
    
    annotation_patterns = ['*.lab', '*.lbl', '*.csv', '*.txt', '*.tse', '*.ann']
    
    for pattern in annotation_patterns:
        files = list(tuev_dir.rglob(pattern))
        if files:
            print(f"\n{pattern}: {len(files)} files found")
            print(f"  Example: {files[0]}")
            
            # Show first few lines
            if files[0].stat().st_size < 10000:  # Only if small file
                try:
                    with open(files[0], 'r') as f:
                        lines = f.readlines()[:10]
                    print(f"  First lines:")
                    for line in lines:
                        print(f"    {line.rstrip()}")
                except:
                    print(f"  (binary or unreadable)")
        else:
            print(f"{pattern}: Not found")


def check_sample_paths():
    """Check a specific example path."""
    print("\n" + "=" * 80)
    print("PART 3: Testing Annotation File Detection")
    print("=" * 80)
    
    # Sample path from TUEV
    sample_paths = [
        r"C:\Users\Pc\Desktop\tuh_data\tuev\edf\eval\000\aaaaaaaj_s005_t000.edf",
        r"C:\Users\Pc\Desktop\tuh_data\tuev\edf\eval\000\aaaaaaaj_s005_t001.edf",
        r"C:\Users\Pc\Desktop\tuh_data\tuev\edf\eval\001\aaaaaalh_s007_t000.edf",
    ]
    
    print("\nTesting annotation file detection:\n")
    
    for sample_path in sample_paths:
        path = Path(sample_path)
        print(f"EDF file: {path.name}")
        
        if path.exists():
            print(f"  ✅ EDF exists")
            
            # Try different extensions
            for ext in ['.lab', '.lbl', '.csv', '.txt', '.tse', '.ann']:
                ann_file = path.with_suffix(ext)
                if ann_file.exists():
                    print(f"  ✅ Found: {ann_file.name} ({ann_file.stat().st_size} bytes)")
                else:
                    print(f"  ❌ Not found: {ann_file.name}")
        else:
            print(f"  ❌ EDF doesn't exist")
            
            # Maybe files are in a different location?
            # Try to find by filename
            tuev_dir = Path(TUEV_DATA_DIR)
            matches = list(tuev_dir.rglob(path.name))
            if matches:
                print(f"  ℹ️  Found at: {matches[0]}")
        
        print()


def main():
    print("\n" + "=" * 80)
    print("TUEV STRUCTURE DIAGNOSTIC")
    print("=" * 80)
    print()
    
    # Part 1: Check preprocessed metadata
    original_paths = check_preprocessed_paths()
    
    # Part 2: Check TUEV directory
    check_tuev_directory()
    
    # Part 3: Test specific paths
    check_sample_paths()
    
    print("\n" + "=" * 80)
    print("DIAGNOSTIC COMPLETE")
    print("=" * 80)
    print()


if __name__ == "__main__":
    main()
