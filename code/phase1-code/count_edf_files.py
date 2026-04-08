#!/usr/bin/env python3
"""
Count EDF Files in TUH Datasets
================================

This script counts the number of EDF files in each TUH sub-corpus
and provides statistics about the dataset structure.

Author: BADEREDDINE Haitham
Date: March 2026
"""

import argparse
from pathlib import Path
from collections import defaultdict
import json


def count_edf_files(data_dir: Path, dataset_name: str = None) -> dict:
    """
    Count EDF files in a directory (recursively).
    
    Args:
        data_dir: Root directory to search
        dataset_name: Optional dataset name for labeling
    
    Returns:
        Dictionary with file counts and statistics
    """
    # Find all EDF files recursively
    edf_files = list(data_dir.rglob('*.edf'))
    
    # Count total files
    total_files = len(edf_files)
    
    # Count by subdirectory (for datasets with train/dev/eval splits)
    subdir_counts = defaultdict(int)
    for edf_file in edf_files:
        # Get relative path from data_dir
        rel_path = edf_file.relative_to(data_dir)
        parts = rel_path.parts
        
        # First directory level (e.g., 'train', 'dev', 'eval', or subject dirs)
        if len(parts) > 0:
            subdir = parts[0]
            subdir_counts[subdir] += 1
    
    # Get unique subjects (assuming pattern: patient_id/session/...)
    subjects = set()
    for edf_file in edf_files:
        rel_path = edf_file.relative_to(data_dir)
        parts = rel_path.parts
        
        # Try to extract subject ID (usually first or second level)
        if len(parts) >= 2:
            # Skip 'train'/'dev'/'eval' directories
            if parts[0] in ['train', 'dev', 'eval', 'normal', 'abnormal', 
                           '00_epilepsy', '01_no_epilepsy']:
                if len(parts) >= 3:
                    subjects.add(parts[1])  # Subject is after split
            else:
                subjects.add(parts[0])  # Subject is first level
    
    return {
        'dataset': dataset_name or data_dir.name,
        'total_files': total_files,
        'unique_subjects': len(subjects),
        'subdirectory_counts': dict(subdir_counts),
        'files': [str(f.relative_to(data_dir)) for f in edf_files[:5]]  # First 5 as examples
    }


def print_dataset_summary(stats: dict):
    """Print formatted summary of dataset statistics."""
    print(f"\n{'='*70}")
    print(f"Dataset: {stats['dataset'].upper()}")
    print(f"{'='*70}")
    print(f"Total EDF files:    {stats['total_files']:,}")
    print(f"Unique subjects:    {stats['unique_subjects']:,}")
    
    if stats['subdirectory_counts']:
        print(f"\nBreakdown by subdirectory:")
        for subdir, count in sorted(stats['subdirectory_counts'].items()):
            print(f"  {subdir:20s}: {count:6,} files")
    
    if stats['files']:
        print(f"\nExample files:")
        for example_file in stats['files'][:3]:
            print(f"  {example_file}")
    
    print(f"{'='*70}")


def main():
    """Main function to count EDF files."""
    parser = argparse.ArgumentParser(
        description='Count EDF files in TUH datasets'
    )
    parser.add_argument(
        '--data-dir',
        type=str,
        required=True,
        help='Path to TUH data directory'
    )
    parser.add_argument(
        '--dataset',
        type=str,
        choices=['TUAB', 'TUAR', 'TUEP', 'TUEV', 'TUSL', 'TUSZ', 'all'],
        default='all',
        help='Dataset to count (default: all)'
    )
    parser.add_argument(
        '--export',
        type=str,
        help='Export results to JSON file'
    )
    
    args = parser.parse_args()
    
    data_root = Path(args.data_dir)
    
    if not data_root.exists():
        print(f"❌ Error: Directory not found: {data_root}")
        return
    
    print("\n" + "="*70)
    print("TUH EEG DATASET FILE COUNTER")
    print("="*70)
    
    all_stats = {}
    
    if args.dataset.upper() == 'ALL':
        # Count all datasets
        datasets = ['tuab', 'tuar', 'tuep', 'tuev', 'tusl', 'tusz']
        
        for dataset in datasets:
            dataset_dir = data_root / dataset / 'edf'
            
            if dataset_dir.exists():
                stats = count_edf_files(dataset_dir, dataset)
                all_stats[dataset.upper()] = stats
                print_dataset_summary(stats)
            else:
                print(f"\n⚠️  Directory not found: {dataset_dir}")
        
        # Print overall summary
        total_all = sum(s['total_files'] for s in all_stats.values())
        total_subjects = sum(s['unique_subjects'] for s in all_stats.values())
        
        print(f"\n{'='*70}")
        print(f"OVERALL SUMMARY")
        print(f"{'='*70}")
        print(f"Total datasets:     {len(all_stats)}")
        print(f"Total EDF files:    {total_all:,}")
        print(f"Total subjects:     {total_subjects:,}")
        print(f"\nBreakdown by dataset:")
        for dataset, stats in sorted(all_stats.items()):
            pct = (stats['total_files'] / total_all * 100) if total_all > 0 else 0
            print(f"  {dataset:6s}: {stats['total_files']:7,} files ({pct:5.1f}%) - {stats['unique_subjects']:5,} subjects")
        print(f"{'='*70}\n")
        
    else:
        # Count single dataset
        dataset = args.dataset.lower()
        dataset_dir = data_root / dataset / 'edf'
        
        if not dataset_dir.exists():
            # Try without 'edf' subdirectory
            dataset_dir = data_root / dataset
            
            if not dataset_dir.exists():
                # Try as direct path
                dataset_dir = data_root
        
        if dataset_dir.exists():
            stats = count_edf_files(dataset_dir, dataset)
            all_stats[args.dataset.upper()] = stats
            print_dataset_summary(stats)
        else:
            print(f"❌ Error: Directory not found: {dataset_dir}")
            return
    
    # Export to JSON if requested
    if args.export:
        export_path = Path(args.export)
        with open(export_path, 'w') as f:
            json.dump(all_stats, f, indent=2)
        print(f"\n✅ Results exported to: {export_path}\n")


if __name__ == "__main__":
    main()
