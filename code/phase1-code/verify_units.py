#!/usr/bin/env python3
"""
Unit Verification Script for TUH EEG Datasets
==============================================

This script checks the actual units of EEG data across all TUH datasets
by examining raw EDF files and comparing amplitude statistics.

Author: BADEREDDINE Haitham
Date: March 2026

Purpose:
- Verify if TUSL/TUSZ data is truly in microvolts (µV)
- Compare amplitude characteristics across all datasets
- Determine if unit scaling transformation is needed

Usage:
    python verify_units.py
"""

import numpy as np
import mne
from pathlib import Path
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
from mne.io import read_raw_edf

# ============================================================================
# CONFIGURATION
# ============================================================================

DATASETS = {
    'TUAB': r'C:\Users\Pc\Desktop\tuh_data\tuab\edf',
    'TUAR': r'C:\Users\Pc\Desktop\tuh_data\tuar\edf',
    'TUEP': r'C:\Users\Pc\Desktop\tuh_data\tuep\edf',
    'TUEV': r'C:\Users\Pc\Desktop\tuh_data\tuev\edf',
    'TUSL': r'C:\Users\Pc\Desktop\tuh_data\tusl\edf',
    'TUSZ': r'C:\Users\Pc\Desktop\tuh_data\tusz\edf',
}

# Number of files to sample per dataset
SAMPLE_SIZE = 5  # Increase to 10-20 for more confidence


# ============================================================================
# UNIT DETECTION FUNCTIONS
# ============================================================================

def analyze_raw_edf(file_path: Path) -> Dict:
    """
    Analyze raw EDF file to determine actual units.
    
    Returns statistics about the raw signal amplitudes.
    """
    try:
        # Load raw EDF (don't apply any scaling)
        raw = read_raw_edf(str(file_path), preload=True, verbose=False)
        
        # Get raw data
        data = raw.get_data()  # This is what MNE returns after reading EDF
        
        # Get header information
        ch_names = raw.ch_names
        physical_dims = [raw.info['chs'][i]['unit_mul'] for i in range(len(ch_names))]
        
        # Calculate statistics
        stats = {
            'file': file_path.name,
            'n_channels': len(ch_names),
            'sfreq': raw.info['sfreq'],
            
            # Raw data statistics (what MNE gives us)
            'data_mean': float(np.mean(data)),
            'data_std': float(np.std(data)),
            'data_min': float(np.min(data)),
            'data_max': float(np.max(data)),
            'data_range': float(np.ptp(data)),
            'data_median_std': float(np.median([data[i, :].std() for i in range(data.shape[0])])),
            
            # Per-channel statistics
            'channel_stds': [float(data[i, :].std()) for i in range(min(5, data.shape[0]))],
            
            # Check what the header claims
            'header_unit_mul': physical_dims[0] if physical_dims else None,
        }
        
        return stats
        
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None


def interpret_units(stats: Dict) -> Tuple[str, float]:
    """
    Interpret what units the data is actually in based on statistics.
    
    Returns:
        (interpreted_unit, suggested_scale_factor) tuple
    """
    std = stats['data_std']
    max_val = stats['data_max']
    
    # Expected ranges for clinical EEG
    # Typical clinical EEG: 10-100 µV STD, max ~200-500 µV
    
    if std < 0.0001:  # e.g., 0.00005
        # Data appears to be in Volts
        return "Volts (V)", 1e6
    
    elif 0.0001 <= std < 0.1:  # e.g., 0.05
        # Data appears to be in millivolts
        return "Millivolts (mV)", 1e3
    
    elif 0.1 <= std < 1.0:  # e.g., 0.5
        # Ambiguous - could be:
        # - mV with low amplitude
        # - µV with very low amplitude (TUSL case)
        if max_val > 10:
            return "Millivolts (mV) - low amplitude", 1e3
        else:
            return "Microvolts (µV) - VERY low amplitude", 1.0
    
    elif 1.0 <= std < 10.0:  # e.g., 5
        # Could be µV with low amplitude or mV with very low amplitude
        if max_val > 100:
            return "Millivolts (mV) - possible", 1e3
        else:
            return "Microvolts (µV) - low amplitude", 1.0
    
    elif 10.0 <= std < 200.0:  # e.g., 50
        # Typical clinical EEG in µV
        return "Microvolts (µV) - NORMAL clinical EEG", 1.0
    
    else:  # std >= 200
        # Unusually high - might be already scaled or artifact
        return "Microvolts (µV) - high amplitude", 1.0


# ============================================================================
# DATASET ANALYSIS
# ============================================================================

def analyze_dataset(dataset_name: str, data_dir: str, sample_size: int = 5) -> List[Dict]:
    """
    Analyze a sample of files from a dataset.
    """
    print(f"\n{'='*70}")
    print(f"Analyzing {dataset_name}")
    print(f"{'='*70}")
    
    # Find EDF files
    data_path = Path(data_dir)
    if not data_path.exists():
        print(f"⚠️  Directory not found: {data_dir}")
        return []
    
    edf_files = list(data_path.rglob('*.edf'))
    
    if len(edf_files) == 0:
        print(f"⚠️  No EDF files found in {data_dir}")
        return []
    
    print(f"Found {len(edf_files)} EDF files")
    
    # Sample random files
    import random
    random.seed(42)  # Reproducible sampling
    sample_files = random.sample(edf_files, min(sample_size, len(edf_files)))
    
    print(f"Analyzing {len(sample_files)} sample files...\n")
    
    # Analyze each file
    results = []
    for i, file_path in enumerate(sample_files, 1):
        print(f"[{i}/{len(sample_files)}] {file_path.name}...", end=' ')
        stats = analyze_raw_edf(file_path)
        
        if stats:
            unit, scale = interpret_units(stats)
            stats['interpreted_unit'] = unit
            stats['suggested_scale'] = scale
            results.append(stats)
            print(f"✓ STD={stats['data_std']:.6f}")
        else:
            print("✗ Failed")
    
    return results


def print_dataset_summary(dataset_name: str, results: List[Dict]):
    """
    Print summary statistics for a dataset.
    """
    if not results:
        return
    
    print(f"\n{'='*70}")
    print(f"SUMMARY: {dataset_name}")
    print(f"{'='*70}")
    
    # Aggregate statistics
    stds = [r['data_std'] for r in results]
    median_stds = [r['data_median_std'] for r in results]
    maxs = [r['data_max'] for r in results]
    
    print(f"\nAmplitude Statistics (from MNE-loaded data):")
    print(f"  Mean STD:        {np.mean(stds):10.6f}")
    print(f"  Median STD:      {np.median(stds):10.6f}")
    print(f"  STD range:       {np.min(stds):10.6f} to {np.max(stds):10.6f}")
    print(f"  Mean of median channel STDs: {np.mean(median_stds):10.6f}")
    print(f"  Mean Max:        {np.mean(maxs):10.6f}")
    
    # Interpretation
    print(f"\nInterpretation:")
    units = [r['interpreted_unit'] for r in results]
    scales = [r['suggested_scale'] for r in results]
    
    # Most common interpretation
    from collections import Counter
    unit_counts = Counter(units)
    most_common_unit, count = unit_counts.most_common(1)[0]
    
    print(f"  Most common: {most_common_unit} ({count}/{len(results)} files)")
    
    if len(set(scales)) == 1:
        scale = scales[0]
        if scale == 1.0:
            print(f"  ✅ Data appears to be in MICROVOLTS (µV)")
            print(f"  ✅ NO scaling needed")
        else:
            print(f"  ⚠️  Data appears to need scaling by {scale:.0f}×")
            print(f"  ⚠️  Suggested: Multiply by {scale:.0f} to convert to µV")
    else:
        print(f"  ⚠️  INCONSISTENT units detected across files!")
        print(f"  Files suggest scales: {set(scales)}")
    
    print(f"\nPer-file details:")
    print(f"{'File':<30} {'STD':<12} {'Median STD':<12} {'Interpretation'}")
    print(f"{'-'*30} {'-'*12} {'-'*12} {'-'*40}")
    for r in results:
        print(f"{r['file']:<30} {r['data_std']:<12.6f} {r['data_median_std']:<12.6f} {r['interpreted_unit']}")


# ============================================================================
# COMPARISON ACROSS DATASETS
# ============================================================================

def compare_datasets(all_results: Dict[str, List[Dict]]):
    """
    Compare amplitude characteristics across all datasets.
    """
    print(f"\n{'='*70}")
    print(f"CROSS-DATASET COMPARISON")
    print(f"{'='*70}")
    
    print(f"\n{'Dataset':<10} {'Mean STD':<15} {'Median STD':<15} {'Interpretation'}")
    print(f"{'-'*10} {'-'*15} {'-'*15} {'-'*40}")
    
    for dataset_name, results in all_results.items():
        if results:
            stds = [r['data_std'] for r in results]
            median_stds = [r['data_median_std'] for r in results]
            
            # Get most common interpretation
            from collections import Counter
            units = [r['interpreted_unit'] for r in results]
            most_common = Counter(units).most_common(1)[0][0]
            
            print(f"{dataset_name:<10} {np.mean(stds):<15.6f} {np.mean(median_stds):<15.6f} {most_common}")


def plot_comparison(all_results: Dict[str, List[Dict]]):
    """
    Create visual comparison of amplitude distributions.
    """
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    
    datasets = []
    mean_stds = []
    median_stds = []
    colors = []
    
    color_map = {
        'TUAB': 'blue',
        'TUAR': 'green', 
        'TUEP': 'purple',
        'TUEV': 'orange',
        'TUSL': 'red',
        'TUSZ': 'brown',
    }
    
    for dataset_name, results in all_results.items():
        if results:
            datasets.append(dataset_name)
            stds = [r['data_std'] for r in results]
            median_stds = [r['data_median_std'] for r in results]
            mean_stds.append(np.mean(stds))
            median_stds.append(np.mean(median_stds))
            colors.append(color_map.get(dataset_name, 'gray'))
    
    # Plot 1: Mean STD comparison
    axes[0].bar(datasets, mean_stds, color=colors, alpha=0.7)
    axes[0].axhline(y=0.1, color='red', linestyle='--', linewidth=2, label='TUSL QC Threshold (0.1 µV)')
    axes[0].axhline(y=0.5, color='orange', linestyle='--', linewidth=2, label='TUAB QC Threshold (0.5 µV)')
    axes[0].set_ylabel('Mean STD (units from EDF)')
    axes[0].set_title('Comparison of Signal Variability Across TUH Datasets')
    axes[0].legend()
    axes[0].grid(axis='y', alpha=0.3)
    axes[0].set_yscale('log')
    
    # Plot 2: Distribution
    for dataset_name, results in all_results.items():
        if results:
            stds = [r['data_std'] for r in results]
            axes[1].scatter([dataset_name]*len(stds), stds, 
                          color=color_map.get(dataset_name, 'gray'),
                          alpha=0.6, s=100)
    
    axes[1].axhline(y=0.1, color='red', linestyle='--', linewidth=2, alpha=0.5)
    axes[1].axhline(y=0.5, color='orange', linestyle='--', linewidth=2, alpha=0.5)
    axes[1].set_ylabel('STD (units from EDF)')
    axes[1].set_title('Distribution of STD Values per Dataset')
    axes[1].grid(axis='y', alpha=0.3)
    axes[1].set_yscale('log')
    
    plt.tight_layout()
    plt.savefig('unit_verification_comparison.png', dpi=300, bbox_inches='tight')
    print(f"\n📊 Saved comparison plot: unit_verification_comparison.png")
    plt.close()


# ============================================================================
# MAIN ANALYSIS
# ============================================================================

def main():
    """
    Main analysis routine.
    """
    print(f"\n{'='*70}")
    print(f"TUH EEG UNIT VERIFICATION TOOL")
    print(f"{'='*70}")
    print(f"\nThis script will analyze raw EDF files to determine actual units.")
    print(f"Sampling {SAMPLE_SIZE} files per dataset...\n")
    
    all_results = {}
    
    # Analyze each dataset
    for dataset_name, data_dir in DATASETS.items():
        results = analyze_dataset(dataset_name, data_dir, SAMPLE_SIZE)
        if results:
            all_results[dataset_name] = results
            print_dataset_summary(dataset_name, results)
    
    # Cross-dataset comparison
    if all_results:
        compare_datasets(all_results)
        plot_comparison(all_results)
    
    # Final recommendations
    print(f"\n{'='*70}")
    print(f"RECOMMENDATIONS")
    print(f"{'='*70}\n")
    
    for dataset_name, results in all_results.items():
        if results:
            scales = set([r['suggested_scale'] for r in results])
            if len(scales) == 1:
                scale = list(scales)[0]
                if scale == 1.0:
                    print(f"✅ {dataset_name}: Data is in µV - NO transformation needed")
                else:
                    print(f"⚠️  {dataset_name}: Multiply data by {scale:.0f}× to convert to µV")
            else:
                print(f"❌ {dataset_name}: INCONSISTENT - manual inspection needed")
    
    print(f"\n{'='*70}\n")


if __name__ == "__main__":
    main()
