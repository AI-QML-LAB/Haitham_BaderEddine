#!/usr/bin/env python3
"""
TUH EEG Preprocessing Pipeline - FINAL CORRECTED VERSION
=========================================================

CRITICAL CORRECTIONS APPLIED:
1. V→µV conversion (×1e6) added after loading EDF files
2. Uniform QC thresholds for all datasets (amplitude verification showed
   ALL datasets have normal clinical EEG ranges, not low amplitude)

Complete preprocessing pipeline for Temple University Hospital EEG datasets
with uniform quality control thresholds.

Author: BADEREDDINE Haitham
Project: Decoding Biomedical Time Series into Natural Language 
         Using Hybrid Quantum-Classical Transformers
Date: March 2026
Version: 4.0 (FINAL CORRECTED - Uniform thresholds + V→µV conversion)

Usage:
    python preprocess_tuh_final.py --dataset TUSL --data-dir /path/to/tusl/edf
    python preprocess_tuh_final.py --dataset TUSZ --data-dir /path/to/tusz/edf
    python preprocess_tuh_final.py --dataset TUAB --data-dir /path/to/tuab/edf
    python preprocess_tuh_final.py --dataset all --data-dir /path/to/tuh_data
"""

import os
import sys
import argparse
import yaml
import numpy as np
import pickle
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging
from datetime import datetime
from tqdm import tqdm

# MNE imports
import mne
from mne.io import read_raw_edf
from scipy import signal

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# UNIFORM QC THRESHOLDS (IN MICROVOLTS) - ALL DATASETS
# ============================================================================
# Based on amplitude verification showing ALL TUH datasets have normal
# clinical EEG amplitude ranges (20-200 µV typical).
# TUSL and TUSZ do NOT have low amplitude - this was a false assumption.
# ============================================================================

# Uniform thresholds applied to all datasets
UNIFORM_THRESHOLDS = {
    'min_median_std': 0.5,           # µV
    'max_amplitude_percentile': 95,
    'max_amplitude_threshold': 500.0,  # µV
    'min_signal_range': 1.0,         # µV
    'max_flat_channels_ratio': 0.5,
    'flat_threshold': 0.1,           # µV
    'description': 'Uniform thresholds for all TUH datasets (normal clinical EEG)'
}

# For backward compatibility, map all datasets to uniform thresholds
QC_THRESHOLDS = {
    'TUAB': UNIFORM_THRESHOLDS.copy(),
    'TUAR': UNIFORM_THRESHOLDS.copy(),
    'TUEP': UNIFORM_THRESHOLDS.copy(),
    'TUEV': UNIFORM_THRESHOLDS.copy(),
    'TUSL': UNIFORM_THRESHOLDS.copy(),  # CORRECTED: Same as others
    'TUSZ': UNIFORM_THRESHOLDS.copy(),  # CORRECTED: Same as others
}


# ============================================================================
# TUH TCP MONTAGE (20 CHANNELS)
# ============================================================================

TUH_TCP_CHANNELS = [
    'FP1-F7', 'F7-T3', 'T3-T5', 'T5-O1',  # Left temporal
    'FP2-F8', 'F8-T4', 'T4-T6', 'T6-O2',  # Right temporal
    'T3-C3', 'C3-CZ', 'CZ-C4', 'C4-T4',   # Central
    'FP1-F3', 'F3-C3', 'C3-P3', 'P3-O1',  # Left parasagittal
    'FP2-F4', 'F4-C4', 'C4-P4', 'P4-O2',  # Right parasagittal
]


# ============================================================================
# LABEL EXTRACTION FUNCTIONS
# ============================================================================

def extract_tuab_labels(file_path: str) -> str:
    """Extract labels for TUAB (abnormal/normal from directory)."""
    path_str = str(file_path).replace('\\', '/').lower()
    
    # Use directory separators to avoid substring matching
    if '/abnormal/' in path_str:
        return 'abnormal'
    elif '/normal/' in path_str:
        return 'normal'
    else:
        return 'unknown'


def extract_tuep_labels(file_path: str) -> str:
    """Extract labels for TUEP (epilepsy/no_epilepsy from directory)."""
    path_str = str(file_path).replace('\\', '/').lower()
    
    # Check more specific label FIRST to avoid substring matching
    if '/01_no_epilepsy/' in path_str or '/no_epilepsy/' in path_str:
        return 'no_epilepsy'
    elif '/00_epilepsy/' in path_str or '/epilepsy/' in path_str:
        return 'epilepsy'
    else:
        return 'unknown'


def extract_tusl_labels(file_path: str) -> str:
    """
    Extract labels for TUSL (slowing annotations from CSV).
    
    TUSL has per-channel annotations in CSV format:
    - slow: Slowing detected
    - bckg: Background (normal)
    - seiz: Seizure (less common)
    
    For file-level label, we return the most common NON-BACKGROUND label.
    If only background, return 'bckg'.
    """
    # TUSL uses a special naming convention with segment numbers
    # Try multiple CSV file patterns
    csv_paths = [
        Path(str(file_path).replace('.edf', '.csv')),
        Path(str(file_path).replace('.edf', '_00.csv')),  # Segment suffix
    ]
    
    # Also try wildcard search for any CSV with this base name
    base_name = Path(file_path).stem
    parent_dir = Path(file_path).parent
    
    csv_files = list(parent_dir.glob(f"{base_name}*.csv"))
    
    if not csv_files:
        # Try the standard paths
        csv_files = [p for p in csv_paths if p.exists()]
    
    if not csv_files:
        return 'bckg'  # Default to background
    
    # Count labels across all CSV files
    label_counts = {}
    
    for csv_file in csv_files:
        try:
            with open(csv_file, 'r') as f:
                for line in f:
                    # Skip comments and header
                    if line.startswith('#') or line.startswith('channel'):
                        continue
                    
                    # Parse: channel,start_time,stop_time,label,confidence
                    parts = line.strip().split(',')
                    if len(parts) >= 4:
                        label = parts[3].strip()
                        label_counts[label] = label_counts.get(label, 0) + 1
        
        except Exception as e:
            logger.warning(f"Could not read TUSL CSV {csv_file.name}: {e}")
            continue
    
    # Return most common non-background label
    if label_counts:
        # Remove background from consideration
        label_counts_no_bckg = {k: v for k, v in label_counts.items() 
                                if k not in ['bckg', 'background']}
        
        if label_counts_no_bckg:
            # Return most common event (slow or seiz)
            most_common = max(label_counts_no_bckg.items(), key=lambda x: x[1])[0]
            return most_common
        else:
            # Only background present
            return 'bckg'
    else:
        return 'bckg'  # Fallback


def extract_tusz_labels(file_path: str) -> str:
    """
    Extract labels for TUSZ (seizure annotations from CSV).
    
    TUSZ has per-channel seizure annotations in CSV format:
    - Seizure types: seiz, fnsz, gnsz, spsz, cpsz, absz, tnsz, cnsz, tcsz, etc.
    - bckg: Background (no seizure)
    
    For file-level label, we return the most common seizure type.
    If no seizures, return 'bckg'.
    """
    # TUSZ typically uses *_bi.csv for bipolar montage annotations
    csv_path = Path(str(file_path).replace('.edf', '_bi.csv'))
    
    if not csv_path.exists():
        csv_path = Path(str(file_path).replace('.edf', '.csv'))
    
    if not csv_path.exists():
        return 'bckg'  # No annotations = background
    
    # Count labels
    label_counts = {}
    
    try:
        with open(csv_path, 'r') as f:
            for line in f:
                # Skip comments and header
                if line.startswith('#') or line.startswith('channel'):
                    continue
                
                # Parse: channel,start_time,stop_time,label,confidence
                parts = line.strip().split(',')
                if len(parts) >= 4:
                    label = parts[3].strip()
                    label_counts[label] = label_counts.get(label, 0) + 1
    
    except Exception as e:
        logger.warning(f"Could not read TUSZ CSV {csv_path.name}: {e}")
        return 'bckg'
    
    # Return most common non-background label
    if label_counts:
        # Remove background from consideration
        label_counts_no_bckg = {k: v for k, v in label_counts.items() 
                                if k not in ['bckg', 'background']}
        
        if label_counts_no_bckg:
            # Return most common seizure type
            most_common = max(label_counts_no_bckg.items(), key=lambda x: x[1])[0]
            return most_common
        else:
            # Only background present
            return 'bckg'
    else:
        return 'bckg'  # Fallback


def extract_tuar_labels(file_path: str) -> str:
    """
    Extract labels for TUAR (artifact types from CSV).
    
    TUAR has detailed per-channel, per-segment artifact annotations in CSV files.
    For file-level label, we return the most common artifact type in the file.
    """
    # Look for corresponding CSV file
    csv_path = Path(str(file_path).replace('.edf', '.csv'))
    
    if not csv_path.exists():
        # Try alternative naming (*_ar.csv)
        csv_path = Path(str(file_path).replace('.edf', '_ar.csv'))
    
    if csv_path.exists():
        try:
            # Read CSV and count artifact types
            artifact_counts = {}
            
            with open(csv_path, 'r') as f:
                for line in f:
                    # Skip comments and header
                    if line.startswith('#') or line.startswith('channel'):
                        continue
                    
                    # Parse CSV line: channel,start_time,stop_time,label,confidence
                    parts = line.strip().split(',')
                    if len(parts) >= 4:
                        label = parts[3].strip()
                        artifact_counts[label] = artifact_counts.get(label, 0) + 1
            
            # Return most common artifact type
            if artifact_counts:
                most_common = max(artifact_counts.items(), key=lambda x: x[1])[0]
                return most_common
            else:
                return 'artifact'  # Fallback
        
        except Exception as e:
            logger.warning(f"Could not read TUAR CSV for {file_path.name}: {e}")
            return 'artifact'
    else:
        # No CSV found, return generic label
        return 'artifact'


def extract_tuev_labels(file_path: str) -> str:
    """
    Extract labels for TUEV (event annotations from .lab or .rec files).
    
    TUEV has detailed event-type annotations in .lab files.
    .lab format: start_time stop_time label (times in 10s of microseconds)
    
    Possible labels:
    - spsw: spike and slow wave
    - gped: generalized periodic epileptiform discharge
    - pled: periodic lateralized epileptiform discharge
    - eyem: eye movement
    - artf: artifact
    - bckg: background
    
    For file-level label, we return the most common NON-BACKGROUND event type.
    """
    # Look for corresponding .lab file (per-channel annotations)
    # TUEV has multiple .lab files per EDF (one per channel)
    # Try to find any .lab file associated with this EDF
    
    base_path = Path(str(file_path).replace('.edf', ''))
    edf_dir = file_path.parent
    edf_name = file_path.stem  # filename without extension
    
    # Find all .lab files for this recording
    lab_files = list(edf_dir.glob(f"{edf_name}*.lab"))
    
    if not lab_files:
        # Try .rec file (alternative format)
        rec_file = Path(str(file_path).replace('.edf', '.rec'))
        if rec_file.exists():
            return extract_tuev_from_rec(rec_file)
        else:
            return 'bckg'  # Default to background
    
    # Count event types across all channels
    event_counts = {}
    
    for lab_file in lab_files:
        try:
            with open(lab_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    
                    # Parse: start_time stop_time label
                    # Example: 15760000 15860000 artf
                    parts = line.split()
                    if len(parts) >= 3:
                        label = parts[2].strip()
                        event_counts[label] = event_counts.get(label, 0) + 1
        
        except Exception as e:
            logger.warning(f"Could not read TUEV .lab file {lab_file.name}: {e}")
            continue
    
    # Return most common NON-BACKGROUND event type
    if event_counts:
        # Remove background from consideration
        event_counts_no_bckg = {k: v for k, v in event_counts.items() if k != 'bckg'}
        
        if event_counts_no_bckg:
            # Return most common event (excluding background)
            most_common = max(event_counts_no_bckg.items(), key=lambda x: x[1])[0]
            return most_common
        else:
            # Only background present
            return 'bckg'
    else:
        return 'bckg'  # Fallback


def extract_tuev_from_rec(rec_file: Path) -> str:
    """
    Extract TUEV labels from .rec file format.
    
    .rec format: channel,start_time,stop_time,label_code
    Label codes: 1=spsw, 2=gped, 3=pled, 4=eyem, 5=artf, 6=bckg
    """
    label_map = {
        '1': 'spsw',
        '2': 'gped',
        '3': 'pled',
        '4': 'eyem',
        '5': 'artf',
        '6': 'bckg'
    }
    
    event_counts = {}
    
    try:
        with open(rec_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                # Parse: channel,start,stop,label_code
                # Example: 13,90.4,91.4,6
                parts = line.split(',')
                if len(parts) >= 4:
                    label_code = parts[3].strip()
                    label = label_map.get(label_code, 'bckg')
                    event_counts[label] = event_counts.get(label, 0) + 1
    
    except Exception as e:
        logger.warning(f"Could not read TUEV .rec file {rec_file.name}: {e}")
        return 'bckg'
    
    # Return most common non-background event
    if event_counts:
        event_counts_no_bckg = {k: v for k, v in event_counts.items() if k != 'bckg'}
        
        if event_counts_no_bckg:
            most_common = max(event_counts_no_bckg.items(), key=lambda x: x[1])[0]
            return most_common
        else:
            return 'bckg'
    else:
        return 'bckg'


LABEL_EXTRACTORS = {
    'TUAB': extract_tuab_labels,
    'TUEP': extract_tuep_labels,
    'TUSL': extract_tusl_labels,
    'TUSZ': extract_tusz_labels,
    'TUAR': extract_tuar_labels,
    'TUEV': extract_tuev_labels,
}


# ============================================================================
# CHANNEL STANDARDIZATION
# ============================================================================

def parse_channel_name(ch_name: str) -> str:
    """
    Parse TUH channel name to standard format.
    
    Examples:
        "EEG FP1-REF" -> "FP1"
        "EEG C3-LE"   -> "C3"
        "EEG CZ-A1"   -> "CZ"
    """
    # Remove "EEG " prefix
    ch = ch_name.upper().replace('EEG ', '').strip()
    
    # Take only first part before hyphen
    ch = ch.split('-')[0].strip()
    
    # Standardize case
    if ch in ['FP1', 'FP2']:
        return ch
    elif ch == 'CZ':
        return ch
    else:
        # Capitalize first letter, lowercase rest
        return ch.capitalize()


def get_bipolar_channel_data(raw: mne.io.Raw, bipolar_name: str) -> Optional[np.ndarray]:
    """
    Get data for a bipolar channel by selecting the first electrode.
    
    For "FP1-F7", select "FP1-REF" monopolar channel.
    
    Args:
        raw: MNE Raw object
        bipolar_name: Target bipolar channel (e.g., "FP1-F7")
    
    Returns:
        Channel data or None if not found
    """
    # Extract first electrode name
    first_electrode = bipolar_name.split('-')[0].upper()
    
    # Get all channel names
    ch_names = raw.ch_names
    
    # Find matching channel
    for ch_name in ch_names:
        parsed = parse_channel_name(ch_name)
        if parsed.upper() == first_electrode:
            # Get data for this channel
            idx = raw.ch_names.index(ch_name)
            data, _ = raw[idx, :]
            return data[0]  # Return 1D array
    
    return None


def standardize_channels(raw: mne.io.Raw, target_channels: List[str]) -> Optional[np.ndarray]:
    """
    Standardize to 20-channel TUH TCP montage.
    
    Args:
        raw: MNE Raw object with monopolar channels
        target_channels: List of target bipolar channel names
    
    Returns:
        Standardized data (20, n_samples) or None if insufficient channels
    """
    n_samples = raw.n_times
    standardized_data = []
    
    for bipolar_ch in target_channels:
        ch_data = get_bipolar_channel_data(raw, bipolar_ch)
        
        if ch_data is not None:
            standardized_data.append(ch_data)
        else:
            # Missing channel - cannot proceed
            logger.warning(f"Missing channel: {bipolar_ch}")
            return None
    
    if len(standardized_data) < len(target_channels):
        logger.warning(f"Only found {len(standardized_data)}/{len(target_channels)} channels")
        return None
    
    return np.array(standardized_data)


# ============================================================================
# PREPROCESSING FUNCTIONS
# ============================================================================

def apply_bandpass_filter(data: np.ndarray, sfreq: float, 
                         low: float = 0.5, high: float = 75.0) -> np.ndarray:
    """Apply FIR bandpass filter."""
    nyquist = sfreq / 2
    
    # Design FIR filter
    numtaps = min(int(sfreq * 3), data.shape[1] - 1)  # 3 seconds or less
    if numtaps % 2 == 0:
        numtaps += 1  # Must be odd for zero-phase
    
    fir_coeff = signal.firwin(
        numtaps,
        [low, high],
        pass_zero=False,
        fs=sfreq,
        window='hamming'
    )
    
    # Apply filter (zero-phase)
    filtered = signal.filtfilt(fir_coeff, 1.0, data, axis=1)
    
    return filtered


def apply_notch_filter(data: np.ndarray, sfreq: float, 
                       notch_freq: float = 60.0) -> np.ndarray:
    """Apply notch filter for powerline noise."""
    # Design notch filter
    quality_factor = 30.0
    b_notch, a_notch = signal.iirnotch(notch_freq, quality_factor, sfreq)
    
    # Apply filter
    filtered = signal.filtfilt(b_notch, a_notch, data, axis=1)
    
    return filtered


def apply_global_zscore(data: np.ndarray, epsilon: float = 1e-8) -> np.ndarray:
    """Apply global z-score normalization."""
    mean = np.mean(data)
    std = np.std(data)
    
    normalized = (data - mean) / (std + epsilon)
    
    return normalized


# ============================================================================
# QUALITY CONTROL
# ============================================================================

def check_segment_quality(segment: np.ndarray, dataset_name: str) -> Tuple[bool, str]:
    """
    Check segment quality using dataset-adaptive thresholds.
    
    Args:
        segment: EEG segment (n_channels, n_samples) IN MICROVOLTS
        dataset_name: Dataset name (TUAB, TUSL, etc.)
    
    Returns:
        (passed, rejection_reason) tuple
    """
    # Get dataset-specific thresholds
    thresh = QC_THRESHOLDS[dataset_name.upper()]
    
    n_channels, n_samples = segment.shape
    
    # Check 1: NaN or Inf
    if np.any(np.isnan(segment)) or np.any(np.isinf(segment)):
        return False, 'invalid_values'
    
    # Check 2: Maximum amplitude (95th percentile)
    max_amplitude = np.percentile(np.abs(segment), thresh['max_amplitude_percentile'])
    if max_amplitude > thresh['max_amplitude_threshold']:
        return False, 'high_amplitude'
    
    # Check 3: Median variance (DATASET-ADAPTIVE)
    channel_stds = np.array([segment[ch, :].std() for ch in range(n_channels)])
    median_std = np.median(channel_stds)
    
    if median_std < thresh['min_median_std']:
        return False, 'low_variance'
    
    # Check 4: Flat channels
    flat_channels = np.sum(channel_stds < thresh['flat_threshold'])
    flat_ratio = flat_channels / n_channels
    
    if flat_ratio > thresh['max_flat_channels_ratio']:
        return False, 'flat_channels'
    
    # Check 5: Signal range
    signal_range = np.ptp(segment)
    
    if signal_range < thresh['min_signal_range']:
        return False, 'low_range'
    
    return True, 'pass'


# ============================================================================
# SEGMENTATION
# ============================================================================

def create_segments(data: np.ndarray, sfreq: float, 
                   duration: float = 10.0, overlap: float = 0.0) -> List[np.ndarray]:
    """
    Create fixed-duration segments from continuous data.
    
    Args:
        data: Continuous EEG (n_channels, n_samples)
        sfreq: Sampling frequency
        duration: Segment duration in seconds
        overlap: Overlap fraction (0.0 = non-overlapping)
    
    Returns:
        List of segments
    """
    n_channels, n_samples = data.shape
    samples_per_segment = int(duration * sfreq)
    step = int(samples_per_segment * (1 - overlap))
    
    segments = []
    
    for start in range(0, n_samples - samples_per_segment + 1, step):
        end = start + samples_per_segment
        segment = data[:, start:end]
        segments.append(segment)
    
    return segments


# ============================================================================
# FEATURE EXTRACTION (BASIC)
# ============================================================================

def extract_basic_features(segment: np.ndarray, sfreq: float) -> Dict:
    """Extract basic features from segment."""
    features = {
        'mean': float(np.mean(segment)),
        'std': float(np.std(segment)),
        'min': float(np.min(segment)),
        'max': float(np.max(segment)),
        'range': float(np.ptp(segment)),
    }
    
    return features


# ============================================================================
# SAVE/LOAD FUNCTIONS
# ============================================================================

def save_segment(segment_norm: np.ndarray, segment_raw: np.ndarray, 
                metadata: Dict, output_dir: Path, file_idx: int, seg_idx: int):
    """
    Save segment as pickle file.
    
    Args:
        segment_norm: Normalized segment (for saving)
        segment_raw: Raw segment before normalization (for feature extraction)
        metadata: Segment metadata
        output_dir: Output directory
        file_idx: File index
        seg_idx: Segment index
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create filename
    filename = f"{metadata['dataset']}_{file_idx:06d}_seg{seg_idx:04d}.pkl"
    filepath = output_dir / filename
    
    # Create data structure
    # CRITICAL: Extract features from RAW segment (before normalization)
    # This preserves the actual µV statistics for verification
    data = {
        'segment': segment_norm,  # Save normalized segment
        'metadata': metadata,
        'features': extract_basic_features(segment_raw, metadata['sfreq'])  # Features from raw
    }
    
    # Save
    with open(filepath, 'wb') as f:
        pickle.dump(data, f)


def save_statistics(stats: Dict, output_dir: Path):
    """Save preprocessing statistics."""
    stats_dir = output_dir / 'statistics'
    stats_dir.mkdir(parents=True, exist_ok=True)
    
    filepath = stats_dir / 'preprocessing_stats.json'
    
    with open(filepath, 'w') as f:
        json.dump(stats, f, indent=2)
    
    logger.info(f"Statistics saved to {filepath}")


# ============================================================================
# MAIN PREPROCESSING FUNCTION
# ============================================================================

def preprocess_file(file_path: Path, dataset_name: str, 
                   file_idx: int, config: Dict, output_base: Path) -> Dict:
    """
    Preprocess a single EDF file.
    
    Args:
        file_path: Path to EDF file
        dataset_name: Dataset name
        file_idx: File index
        config: Configuration dictionary
        output_base: Base output directory
    
    Returns:
        Statistics dictionary
    """
    stats = {
        'total_segments': 0,
        'valid_segments': 0,
        'rejected_segments': 0,
        'rejection_reasons': {}
    }
    
    try:
        # Load raw EDF
        raw = read_raw_edf(str(file_path), preload=True, verbose=False)
        
        # Standardize channels to 20 TUH TCP
        data = standardize_channels(raw, TUH_TCP_CHANNELS)
        
        if data is None:
            logger.warning(f"Insufficient channels in {file_path.name}")
            return stats
        
        # ====================================================================
        # CRITICAL FIX: Convert from VOLTS to MICROVOLTS
        # ====================================================================
        # MNE returns data in volts (SI units)
        # QC thresholds are in microvolts (clinical convention)
        # Without this conversion, ALL segments will fail QC!
        data = data * 1e6  # V → µV
        # ====================================================================
        
        # Get sampling frequency
        original_sfreq = raw.info['sfreq']
        target_sfreq = config['preprocessing']['target_sfreq']
        
        # Resample if needed
        if original_sfreq != target_sfreq:
            n_channels, n_samples = data.shape
            n_samples_new = int(n_samples * target_sfreq / original_sfreq)
            data_resampled = np.zeros((n_channels, n_samples_new))
            
            for ch in range(n_channels):
                data_resampled[ch, :] = signal.resample(data[ch, :], n_samples_new)
            
            data = data_resampled
            sfreq = target_sfreq
        else:
            sfreq = original_sfreq
        
        # Apply filters (NOW OPERATES ON µV)
        if config['preprocessing']['bandpass_filter']['apply']:
            low = config['preprocessing']['bandpass_filter']['low']
            high = config['preprocessing']['bandpass_filter']['high']
            data = apply_bandpass_filter(data, sfreq, low, high)
        
        if config['preprocessing']['notch_filter']['apply']:
            notch_freq = config['preprocessing']['notch_filter']['frequency']
            data = apply_notch_filter(data, sfreq, notch_freq)
        
        # Create segments (NOW IN µV)
        duration = config['preprocessing']['segment_duration']
        overlap = config['preprocessing']['segment_overlap']
        segments = create_segments(data, sfreq, duration, overlap)
        
        stats['total_segments'] = len(segments)
        
        # Extract file-level label
        label_extractor = LABEL_EXTRACTORS.get(dataset_name.upper())
        file_label = label_extractor(file_path) if label_extractor else 'unknown'
        
        # Process each segment
        output_dir = output_base / f"neurovault_{dataset_name.lower()}" / "preprocessed"
        
        for seg_idx, segment in enumerate(segments):
            # ============================================================
            # CRITICAL: Quality control on RAW µV values BEFORE normalization
            # ============================================================
            passed, reason = check_segment_quality(segment, dataset_name)
            
            if passed:
                # AFTER QC passes: Apply normalization to valid segments
                segment_norm = apply_global_zscore(
                    segment, 
                    epsilon=config['preprocessing']['normalization']['epsilon']
                )
                
                # Create metadata
                metadata = {
                    'file_path': str(file_path),
                    'file_idx': file_idx,
                    'segment_idx': seg_idx,
                    'dataset': dataset_name,
                    'label': file_label,
                    'sfreq': sfreq,
                    'n_channels': segment_norm.shape[0],
                    'n_samples': segment_norm.shape[1],
                    'duration': duration,
                }
                
                # Save segment (pass both normalized and raw)
                save_segment(segment_norm, segment, metadata, output_dir, file_idx, seg_idx)
                stats['valid_segments'] += 1
            else:
                stats['rejected_segments'] += 1
                stats['rejection_reasons'][reason] = \
                    stats['rejection_reasons'].get(reason, 0) + 1
        
        return stats
        
    except Exception as e:
        logger.error(f"Error processing {file_path}: {e}")
        return stats


# ============================================================================
# DATASET PROCESSING
# ============================================================================

def preprocess_dataset(dataset_name: str, data_dir: Path, 
                      config: Dict, output_base: Path) -> Dict:
    """
    Preprocess entire dataset.
    
    Args:
        dataset_name: Dataset name (TUAB, TUSL, etc.)
        data_dir: Directory containing EDF files
        config: Configuration dictionary
        output_base: Base output directory
    
    Returns:
        Overall statistics
    """
    logger.info("=" * 70)
    logger.info(f"Processing {dataset_name} with UNIFORM QC Thresholds")
    logger.info("=" * 70)
    
    # Get thresholds (uniform for all datasets)
    thresh = QC_THRESHOLDS[dataset_name.upper()]
    logger.info(f"Quality control thresholds:")
    logger.info(f"  min_median_std: {thresh['min_median_std']} µV")
    logger.info(f"  max_amplitude: {thresh['max_amplitude_threshold']} µV")
    logger.info(f"  Description: {thresh['description']}")
    logger.info("")
    
    # Find EDF files
    edf_files = sorted(list(Path(data_dir).rglob('*.edf')))
    logger.info(f"Found {len(edf_files)} EDF files")
    
    if len(edf_files) == 0:
        logger.error(f"No EDF files found in {data_dir}")
        return {}
    
    # Initialize overall statistics
    overall_stats = {
        'dataset': dataset_name,
        'start_time': datetime.now().isoformat(),
        'total_files': len(edf_files),
        'processed_files': 0,
        'failed_files': 0,
        'total_segments': 0,
        'valid_segments': 0,
        'rejected_segments': 0,
        'rejection_reasons': {}
    }
    
    # Process files with progress bar
    for file_idx, file_path in enumerate(tqdm(edf_files, desc=f"Processing {dataset_name}")):
        file_stats = preprocess_file(
            file_path, 
            dataset_name, 
            file_idx, 
            config, 
            output_base
        )
        
        # Update overall statistics
        if file_stats['total_segments'] > 0:
            overall_stats['processed_files'] += 1
        else:
            overall_stats['failed_files'] += 1
        
        overall_stats['total_segments'] += file_stats['total_segments']
        overall_stats['valid_segments'] += file_stats['valid_segments']
        overall_stats['rejected_segments'] += file_stats['rejected_segments']
        
        for reason, count in file_stats['rejection_reasons'].items():
            overall_stats['rejection_reasons'][reason] = \
                overall_stats['rejection_reasons'].get(reason, 0) + count
    
    overall_stats['end_time'] = datetime.now().isoformat()
    
    # Print summary
    print_summary(overall_stats)
    
    # Save statistics
    save_statistics(overall_stats, output_base / f"neurovault_{dataset_name.lower()}")
    
    return overall_stats


def print_summary(stats: Dict):
    """Print preprocessing summary."""
    acceptance_rate = (stats['valid_segments'] / stats['total_segments'] * 100 
                      if stats['total_segments'] > 0 else 0)
    
    print(f"\n{'='*70}")
    print(f"PREPROCESSING SUMMARY - {stats['dataset']}")
    print(f"{'='*70}")
    print(f"Files processed:    {stats['processed_files']}/{stats['total_files']}")
    print(f"Files failed:       {stats['failed_files']}")
    print(f"Total segments:     {stats['total_segments']:,}")
    print(f"Valid segments:     {stats['valid_segments']:,} ({acceptance_rate:.1f}%)")
    print(f"Rejected segments:  {stats['rejected_segments']:,} ({100-acceptance_rate:.1f}%)")
    
    if stats['rejection_reasons']:
        print(f"\nRejection breakdown:")
        for reason, count in sorted(stats['rejection_reasons'].items(), 
                                   key=lambda x: x[1], reverse=True):
            pct = count / stats['total_segments'] * 100
            print(f"  {reason:20s}: {count:6,} ({pct:5.1f}%)")
    print(f"{'='*70}\n")


# ============================================================================
# CONFIGURATION LOADING
# ============================================================================

def load_config(config_path: str) -> Dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def create_default_config() -> Dict:
    """Create default configuration if file not found."""
    return {
        'preprocessing': {
            'target_sfreq': 200,
            'segment_duration': 10.0,
            'segment_overlap': 0.0,
            'bandpass_filter': {
                'apply': True,
                'low': 0.5,
                'high': 75.0
            },
            'notch_filter': {
                'apply': True,
                'frequency': 60
            },
            'normalization': {
                'method': 'global_zscore',
                'epsilon': 1e-8
            }
        }
    }


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def main():
    """Main preprocessing pipeline."""
    parser = argparse.ArgumentParser(
        description='TUH EEG Preprocessing - FINAL CORRECTED (Uniform QC + V→µV conversion)'
    )
    parser.add_argument(
        '--dataset',
        type=str,
        required=True,
        choices=['TUAB', 'TUAR', 'TUEP', 'TUEV', 'TUSL', 'TUSZ', 'all'],
        help='Dataset to process'
    )
    parser.add_argument(
        '--data-dir',
        type=str,
        required=True,
        help='Path to raw EDF files'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='neurovault_data',
        help='Base output directory'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='config_tuh_adaptive.yaml',
        help='Configuration file path'
    )
    
    args = parser.parse_args()
    
    # Load configuration
    if os.path.exists(args.config):
        config = load_config(args.config)
        logger.info(f"Loaded configuration from {args.config}")
    else:
        logger.warning(f"Config file not found: {args.config}")
        logger.info("Using default configuration")
        config = create_default_config()
    
    # Create output directory
    output_base = Path(args.output_dir)
    output_base.mkdir(parents=True, exist_ok=True)
    
    # Process dataset(s)
    if args.dataset.upper() == 'ALL':
        # Process all datasets
        datasets = ['TUAB', 'TUAR', 'TUEP', 'TUEV', 'TUSL', 'TUSZ']
        for dataset in datasets:
            dataset_dir = Path(args.data_dir) / dataset.lower() / 'edf'
            if dataset_dir.exists():
                preprocess_dataset(dataset, dataset_dir, config, output_base)
            else:
                logger.warning(f"Directory not found: {dataset_dir}")
    else:
        # Process single dataset
        data_dir = Path(args.data_dir)
        preprocess_dataset(args.dataset.upper(), data_dir, config, output_base)
    
    logger.info("\n✅ Preprocessing complete!")


if __name__ == "__main__":
    main()
