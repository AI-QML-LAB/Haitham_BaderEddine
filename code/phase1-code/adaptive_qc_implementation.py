#!/usr/bin/env python3
"""
Dataset-Adaptive Quality Control Implementation
================================================

This module implements dataset-specific QC thresholds for TUH EEG preprocessing.

Key Insight:
Different TUH sub-corpora have different amplitude characteristics due to
acquisition equipment settings (gain control). TUSL uses higher gain settings,
resulting in ~100× lower amplitudes than TUAB/TUEP/TUEV.

Solution:
Use dataset-adaptive thresholds instead of uniform thresholds across all datasets.

Author: BADEREDDINE Haitham
Date: March 2026
"""

import numpy as np
from typing import Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


# ============================================================================
# DATASET-SPECIFIC QC THRESHOLDS
# ============================================================================

# Based on empirical analysis of each TUH sub-corpus
QC_THRESHOLDS = {
    'TUAB': {
        'min_median_std': 0.5,           # µV - typical clinical EEG
        'max_amplitude_percentile': 95,  # Use 95th percentile
        'max_amplitude_threshold': 500.0,  # µV
        'min_signal_range': 1.0,         # µV
        'max_flat_channels_ratio': 0.5,  # 50% max flat channels
        'flat_threshold': 0.1,           # µV
        'description': 'Normal/Abnormal classification - standard clinical EEG'
    },
    
    'TUAR': {
        'min_median_std': 0.5,
        'max_amplitude_percentile': 95,
        'max_amplitude_threshold': 500.0,
        'min_signal_range': 1.0,
        'max_flat_channels_ratio': 0.5,
        'flat_threshold': 0.1,
        'description': 'Artifact classification - similar to TUAB'
    },
    
    'TUEP': {
        'min_median_std': 0.5,
        'max_amplitude_percentile': 95,
        'max_amplitude_threshold': 500.0,
        'min_signal_range': 1.0,
        'max_flat_channels_ratio': 0.5,
        'flat_threshold': 0.1,
        'description': 'Epilepsy detection - standard clinical EEG'
    },
    
    'TUEV': {
        'min_median_std': 0.5,
        'max_amplitude_percentile': 95,
        'max_amplitude_threshold': 500.0,
        'min_signal_range': 1.0,
        'max_flat_channels_ratio': 0.5,
        'flat_threshold': 0.1,
        'description': 'Event classification - clean annotated data'
    },
    
    'TUSL': {
        # ADJUSTED FOR LOW AMPLITUDE (higher gain settings)
        'min_median_std': 0.1,           # µV - RELAXED (was 0.5)
        'max_amplitude_percentile': 95,
        'max_amplitude_threshold': 100.0,  # µV - RELAXED (was 500)
        'min_signal_range': 0.2,         # µV - RELAXED (was 1.0)
        'max_flat_channels_ratio': 0.5,
        'flat_threshold': 0.05,          # µV - RELAXED (was 0.1)
        'description': 'Slowing detection - LOW AMPLITUDE due to high gain'
    },
    
    'TUSZ': {
        # INTERMEDIATE (mixed amplitude characteristics)
        'min_median_std': 0.3,           # µV - INTERMEDIATE
        'max_amplitude_percentile': 95,
        'max_amplitude_threshold': 300.0,  # µV - INTERMEDIATE
        'min_signal_range': 0.5,         # µV - INTERMEDIATE
        'max_flat_channels_ratio': 0.5,
        'flat_threshold': 0.08,          # µV - INTERMEDIATE
        'description': 'Seizure detection - mixed amplitude range'
    },
}


# ============================================================================
# QUALITY CONTROL FUNCTIONS
# ============================================================================

def get_dataset_thresholds(dataset_name: str) -> Dict:
    """
    Get QC thresholds for a specific dataset.
    
    Args:
        dataset_name: Name of dataset (TUAB, TUAR, TUEP, TUEV, TUSL, TUSZ)
    
    Returns:
        Dictionary of QC thresholds for the dataset
    
    Raises:
        ValueError: If dataset_name is not recognized
    """
    dataset_name = dataset_name.upper()
    
    if dataset_name not in QC_THRESHOLDS:
        available = ', '.join(QC_THRESHOLDS.keys())
        raise ValueError(
            f"Unknown dataset: {dataset_name}. "
            f"Available datasets: {available}"
        )
    
    thresholds = QC_THRESHOLDS[dataset_name].copy()
    logger.info(
        f"Using adaptive QC thresholds for {dataset_name}: "
        f"{thresholds['description']}"
    )
    
    return thresholds


def check_segment_quality(
    segment: np.ndarray,
    dataset_name: str,
    epsilon: float = 1e-8
) -> Tuple[bool, str]:
    """
    Check if segment passes quality control using dataset-adaptive thresholds.
    
    Args:
        segment: EEG segment, shape (n_channels, n_samples)
        dataset_name: Name of dataset (TUAB, TUAR, TUEP, TUEV, TUSL, TUSZ)
        epsilon: Small value for numerical stability
    
    Returns:
        (pass_qc, rejection_reason) tuple
        - pass_qc: True if segment passes all QC checks
        - rejection_reason: Reason for rejection, or 'pass' if accepted
    
    Example:
        >>> segment = np.random.randn(20, 2000) * 50  # 20 channels, 2000 samples
        >>> passed, reason = check_segment_quality(segment, 'TUAB')
        >>> if passed:
        >>>     print("Segment accepted")
        >>> else:
        >>>     print(f"Segment rejected: {reason}")
    """
    # Get dataset-specific thresholds
    thresh = get_dataset_thresholds(dataset_name)
    
    n_channels, n_samples = segment.shape
    
    # -------------------------------------------------------------------------
    # Check 1: NaN or Inf values
    # -------------------------------------------------------------------------
    if np.any(np.isnan(segment)) or np.any(np.isinf(segment)):
        return False, 'invalid_values'
    
    # -------------------------------------------------------------------------
    # Check 2: Maximum amplitude (using percentile for robustness)
    # -------------------------------------------------------------------------
    max_amplitude = np.percentile(
        np.abs(segment), 
        thresh['max_amplitude_percentile']
    )
    
    if max_amplitude > thresh['max_amplitude_threshold']:
        logger.debug(
            f"Amplitude too high: {max_amplitude:.2f} µV > "
            f"{thresh['max_amplitude_threshold']} µV"
        )
        return False, 'high_amplitude'
    
    # -------------------------------------------------------------------------
    # Check 3: Median channel variance (dataset-adaptive)
    # -------------------------------------------------------------------------
    channel_stds = np.array([segment[ch, :].std() for ch in range(n_channels)])
    median_std = np.median(channel_stds)
    
    if median_std < thresh['min_median_std']:
        logger.debug(
            f"Variance too low: {median_std:.4f} µV < "
            f"{thresh['min_median_std']} µV"
        )
        return False, 'low_variance'
    
    # -------------------------------------------------------------------------
    # Check 4: Flat channels (dataset-adaptive threshold)
    # -------------------------------------------------------------------------
    flat_channels = np.sum(channel_stds < thresh['flat_threshold'])
    flat_ratio = flat_channels / n_channels
    
    if flat_ratio > thresh['max_flat_channels_ratio']:
        logger.debug(
            f"Too many flat channels: {flat_channels}/{n_channels} "
            f"({flat_ratio:.1%}) > {thresh['max_flat_channels_ratio']:.1%}"
        )
        return False, 'flat_channels'
    
    # -------------------------------------------------------------------------
    # Check 5: Signal range (dataset-adaptive)
    # -------------------------------------------------------------------------
    signal_range = np.ptp(segment)  # peak-to-peak
    
    if signal_range < thresh['min_signal_range']:
        logger.debug(
            f"Signal range too small: {signal_range:.4f} µV < "
            f"{thresh['min_signal_range']} µV"
        )
        return False, 'low_range'
    
    # -------------------------------------------------------------------------
    # All checks passed
    # -------------------------------------------------------------------------
    return True, 'pass'


def apply_quality_control_batch(
    segments: list,
    dataset_name: str,
    verbose: bool = True
) -> Tuple[list, Dict]:
    """
    Apply quality control to a batch of segments.
    
    Args:
        segments: List of EEG segments, each shape (n_channels, n_samples)
        dataset_name: Name of dataset
        verbose: Print summary statistics
    
    Returns:
        (valid_segments, stats) tuple
        - valid_segments: List of segments that passed QC
        - stats: Dictionary with QC statistics
    
    Example:
        >>> segments = [np.random.randn(20, 2000) for _ in range(100)]
        >>> valid_segs, stats = apply_quality_control_batch(segments, 'TUAB')
        >>> print(f"Accepted: {stats['accepted']}/{stats['total']}")
    """
    valid_segments = []
    rejection_counts = {}
    
    total = len(segments)
    
    for seg_idx, segment in enumerate(segments):
        passed, reason = check_segment_quality(segment, dataset_name)
        
        if passed:
            valid_segments.append(segment)
        else:
            rejection_counts[reason] = rejection_counts.get(reason, 0) + 1
    
    # Compute statistics
    accepted = len(valid_segments)
    rejected = total - accepted
    acceptance_rate = (accepted / total * 100) if total > 0 else 0
    
    stats = {
        'total': total,
        'accepted': accepted,
        'rejected': rejected,
        'acceptance_rate': acceptance_rate,
        'rejection_reasons': rejection_counts,
        'dataset': dataset_name
    }
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"Quality Control Results - {dataset_name}")
        print(f"{'='*70}")
        print(f"Total segments:     {total:,}")
        print(f"Accepted:           {accepted:,} ({acceptance_rate:.1f}%)")
        print(f"Rejected:           {rejected:,} ({100-acceptance_rate:.1f}%)")
        
        if rejection_counts:
            print(f"\nRejection breakdown:")
            for reason, count in sorted(
                rejection_counts.items(), 
                key=lambda x: x[1], 
                reverse=True
            ):
                pct = count / total * 100
                print(f"  {reason:20s}: {count:6,} ({pct:5.1f}%)")
        print(f"{'='*70}\n")
    
    return valid_segments, stats


# ============================================================================
# COMPARISON: UNIFORM vs ADAPTIVE THRESHOLDS
# ============================================================================

def compare_uniform_vs_adaptive(
    segments: list,
    dataset_name: str
) -> Dict:
    """
    Compare QC results using uniform vs. adaptive thresholds.
    
    Demonstrates the benefit of dataset-adaptive thresholds.
    
    Args:
        segments: List of EEG segments
        dataset_name: Name of dataset
    
    Returns:
        Dictionary with comparison statistics
    """
    print(f"\n{'='*70}")
    print(f"COMPARISON: Uniform vs. Adaptive QC Thresholds")
    print(f"Dataset: {dataset_name}")
    print(f"{'='*70}\n")
    
    # Apply uniform thresholds (TUAB-calibrated)
    print("1. UNIFORM THRESHOLDS (TUAB-calibrated):")
    uniform_valid, uniform_stats = apply_quality_control_batch(
        segments, 
        'TUAB',  # Force TUAB thresholds
        verbose=True
    )
    
    # Apply adaptive thresholds
    print("2. ADAPTIVE THRESHOLDS (Dataset-specific):")
    adaptive_valid, adaptive_stats = apply_quality_control_batch(
        segments, 
        dataset_name,  # Use actual dataset
        verbose=True
    )
    
    # Compute improvement
    improvement = adaptive_stats['acceptance_rate'] - uniform_stats['acceptance_rate']
    recovered = adaptive_stats['accepted'] - uniform_stats['accepted']
    
    print(f"{'='*70}")
    print(f"IMPROVEMENT SUMMARY")
    print(f"{'='*70}")
    print(f"Uniform acceptance:  {uniform_stats['acceptance_rate']:6.1f}%")
    print(f"Adaptive acceptance: {adaptive_stats['acceptance_rate']:6.1f}%")
    print(f"Improvement:         {improvement:+6.1f}% ({recovered:+,} segments)")
    print(f"{'='*70}\n")
    
    return {
        'uniform': uniform_stats,
        'adaptive': adaptive_stats,
        'improvement_pct': improvement,
        'segments_recovered': recovered
    }


# ============================================================================
# INTEGRATION WITH EXISTING PREPROCESSING CODE
# ============================================================================

def integrate_adaptive_qc_example():
    """
    Example of how to integrate adaptive QC into existing preprocessing pipeline.
    
    This shows how to modify your current preprocessing code.
    """
    
    print("""
    ========================================================================
    INTEGRATION EXAMPLE: How to Update Your Preprocessing Code
    ========================================================================
    
    BEFORE (Uniform thresholds):
    ----------------------------
    
    def process_segment(segment, config):
        # Hard-coded thresholds
        if np.median(channel_stds) < 0.5:  # Same for all datasets
            return None, 'low_variance'
        # ... rest of QC checks
        return segment, 'pass'
    
    
    AFTER (Adaptive thresholds):
    ---------------------------
    
    def process_segment(segment, dataset_name, config):
        # Use adaptive QC
        passed, reason = check_segment_quality(segment, dataset_name)
        
        if not passed:
            return None, reason
        
        return segment, 'pass'
    
    
    USAGE IN PREPROCESSING LOOP:
    -----------------------------
    
    # At the top of your preprocessing script:
    from adaptive_qc_implementation import check_segment_quality
    
    # In your main processing loop:
    for file_path in edf_files:
        # ... load and preprocess raw data ...
        
        # Create segments
        segments = create_segments(preprocessed_data, duration=10.0)
        
        # Apply adaptive QC (pass dataset_name!)
        valid_segments = []
        stats = {'accepted': 0, 'rejected': 0, 'reasons': {}}
        
        for seg_idx, segment in enumerate(segments):
            passed, reason = check_segment_quality(
                segment, 
                dataset_name='TUSL'  # Or TUAB, TUEP, etc.
            )
            
            if passed:
                valid_segments.append(segment)
                stats['accepted'] += 1
            else:
                stats['rejected'] += 1
                stats['reasons'][reason] = stats['reasons'].get(reason, 0) + 1
        
        # Save valid segments
        save_segments(valid_segments, output_dir)
    
    ========================================================================
    """)


# ============================================================================
# DEMONSTRATION & TESTING
# ============================================================================

def demo_adaptive_qc():
    """
    Demonstrate adaptive QC with synthetic data mimicking TUAB vs TUSL.
    """
    print("\n" + "="*70)
    print("DEMONSTRATION: Adaptive QC with Synthetic Data")
    print("="*70 + "\n")
    
    np.random.seed(42)
    
    # -------------------------------------------------------------------------
    # Generate synthetic TUAB-like data (normal amplitude)
    # -------------------------------------------------------------------------
    print("Generating TUAB-like segments (normal amplitude ~50 µV)...")
    tuab_segments = [
        np.random.randn(20, 2000) * 50  # STD ~50 µV
        for _ in range(100)
    ]
    
    print("Testing TUAB segments with TUAB thresholds:")
    tuab_valid, tuab_stats = apply_quality_control_batch(
        tuab_segments, 
        'TUAB',
        verbose=True
    )
    
    # -------------------------------------------------------------------------
    # Generate synthetic TUSL-like data (low amplitude)
    # -------------------------------------------------------------------------
    print("\nGenerating TUSL-like segments (low amplitude ~0.5 µV)...")
    tusl_segments = [
        np.random.randn(20, 2000) * 0.5  # STD ~0.5 µV (100× smaller!)
        for _ in range(100)
    ]
    
    # Test with UNIFORM thresholds (will fail)
    print("\n" + "-"*70)
    print("PROBLEM: Testing TUSL segments with UNIFORM (TUAB) thresholds:")
    print("-"*70)
    tusl_uniform_valid, tusl_uniform_stats = apply_quality_control_batch(
        tusl_segments,
        'TUAB',  # Wrong thresholds!
        verbose=True
    )
    
    # Test with ADAPTIVE thresholds (will succeed)
    print("-"*70)
    print("SOLUTION: Testing TUSL segments with ADAPTIVE (TUSL) thresholds:")
    print("-"*70)
    tusl_adaptive_valid, tusl_adaptive_stats = apply_quality_control_batch(
        tusl_segments,
        'TUSL',  # Correct thresholds!
        verbose=True
    )
    
    # Show improvement
    improvement = (
        tusl_adaptive_stats['acceptance_rate'] - 
        tusl_uniform_stats['acceptance_rate']
    )
    
    print("\n" + "="*70)
    print("RESULT: Adaptive QC Recovers Low-Amplitude Data")
    print("="*70)
    print(f"TUSL with UNIFORM thresholds: {tusl_uniform_stats['acceptance_rate']:.1f}%")
    print(f"TUSL with ADAPTIVE thresholds: {tusl_adaptive_stats['acceptance_rate']:.1f}%")
    print(f"Improvement: +{improvement:.1f}%")
    print("="*70 + "\n")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("\n" + "="*70)
    print("DATASET-ADAPTIVE QUALITY CONTROL FOR TUH EEG PREPROCESSING")
    print("="*70)
    
    # Show threshold table
    print("\nDataset-Specific QC Thresholds:")
    print("-" * 70)
    print(f"{'Dataset':<10} {'Min STD':<12} {'Max Amp':<12} {'Description'}")
    print("-" * 70)
    for dataset, thresh in QC_THRESHOLDS.items():
        print(
            f"{dataset:<10} "
            f"{thresh['min_median_std']:<12.2f} "
            f"{thresh['max_amplitude_threshold']:<12.1f} "
            f"{thresh['description']}"
        )
    print("-" * 70)
    
    # Run demonstration
    demo_adaptive_qc()
    
    # Show integration example
    integrate_adaptive_qc_example()
    
    print("\n✅ Demonstration complete!\n")
