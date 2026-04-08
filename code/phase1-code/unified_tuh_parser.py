"""
Unified TUH Annotation Parser - All Datasets
=============================================

Handles label extraction for all 6 TUH datasets:
- TUAB: Directory-based (abnormal/normal)
- TUAR: CSV format (artifacts)
- TUEP: CSV + Directory (epilepsy detection)
- TUEV: LAB format (events, time in microseconds!)
- TUSL: CSV + LBL format (slowing)
- TUSZ: CSV format (seizures)

Author: Haitham
Date: March 17, 2026
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
from collections import Counter


class UnifiedTUHParser:
    """Unified parser for all TUH annotation formats."""
    
    def __init__(self, dataset: str):
        """
        Initialize parser for specific dataset.
        
        Args:
            dataset: Dataset name (TUAB, TUAR, TUEP, TUEV, TUSL, TUSZ)
        """
        self.dataset = dataset.upper()
    
    def extract_label(self, file_path: Path, segment_idx: int = 0, 
                     segment_duration: float = 10.0) -> str:
        """
        Extract label for a specific segment.
        
        Args:
            file_path: Path to the EDF file
            segment_idx: Segment index (0, 1, 2, ...)
            segment_duration: Segment duration in seconds
            
        Returns:
            Label string
        """
        if self.dataset == 'TUAB':
            return self._extract_tuab(file_path)
        elif self.dataset == 'TUAR':
            return self._extract_csv(file_path, segment_idx, segment_duration, 'duration_weighted')
        elif self.dataset == 'TUEP':
            return self._extract_tuep(file_path, segment_idx, segment_duration)
        elif self.dataset == 'TUEV':
            return self._extract_tuev(file_path, segment_idx, segment_duration)
        elif self.dataset == 'TUSL':
            return self._extract_tusl(file_path, segment_idx, segment_duration)
        elif self.dataset == 'TUSZ':
            return self._extract_csv(file_path, segment_idx, segment_duration, 'duration_weighted')
        else:
            return 'unknown'
    
    # ========================================================================
    # TUAB - Directory-based labels
    # ========================================================================
    
    def _extract_tuab(self, file_path: Path) -> str:
        """Extract label from directory structure for TUAB."""
        file_path_str = str(file_path).lower()
        
        # Use directory separator to avoid substring confusion
        # ('abnormal' contains 'normal', so check with path separators)
        if '\\abnormal\\' in file_path_str or '/abnormal/' in file_path_str:
            return 'abnormal'
        elif '\\normal\\' in file_path_str or '/normal/' in file_path_str:
            return 'normal'
        
        return 'unknown'
    
    # ========================================================================
    # CSV-based parsing (TUAR, TUEP, TUSL, TUSZ)
    # ========================================================================
    
    def _extract_csv(self, file_path: Path, segment_idx: int, 
                    segment_duration: float, strategy: str = 'duration_weighted') -> str:
        """Extract label from CSV annotation file."""
        csv_file = file_path.with_suffix('.csv')
        
        if not csv_file.exists():
            return 'bckg'  # Background/clean if no annotation
        
        try:
            # Read CSV, skipping comment lines
            df = pd.read_csv(csv_file, comment='#', skipinitialspace=True)
            df.columns = df.columns.str.strip()
            
            # Calculate segment time window
            segment_start = segment_idx * segment_duration
            segment_end = segment_start + segment_duration
            
            # Find overlapping annotations
            mask = (df['start_time'] < segment_end) & (df['stop_time'] > segment_start)
            window_annotations = df[mask]
            
            if len(window_annotations) == 0:
                return 'bckg'  # No annotation = background
            
            # Aggregate labels based on strategy
            if strategy == 'duration_weighted':
                return self._aggregate_duration_weighted(
                    window_annotations, segment_start, segment_end
                )
            elif strategy == 'majority_vote':
                return window_annotations['label'].mode()[0]
            else:
                return window_annotations['label'].mode()[0]
        
        except Exception as e:
            print(f"Warning: CSV parsing failed for {csv_file}: {e}")
            return 'bckg'
    
    def _aggregate_duration_weighted(self, annotations: pd.DataFrame, 
                                    start_time: float, end_time: float) -> str:
        """Aggregate labels by duration."""
        label_durations = {}
        
        for _, ann in annotations.iterrows():
            label = ann['label']
            overlap_start = max(ann['start_time'], start_time)
            overlap_end = min(ann['stop_time'], end_time)
            overlap_duration = max(0, overlap_end - overlap_start)
            
            if label not in label_durations:
                label_durations[label] = 0
            label_durations[label] += overlap_duration
        
        if not label_durations:
            return 'bckg'
        
        return max(label_durations, key=label_durations.get)
    
    # ========================================================================
    # TUEP - CSV + Directory
    # ========================================================================
    
    def _extract_tuep(self, file_path: Path, segment_idx: int, 
                     segment_duration: float) -> str:
        """
        Extract label for TUEP (Epilepsy).
        
        Strategy: Use directory for main label (epilepsy/normal),
                 CSV provides segment-level background annotations.
        """
        file_path_str = str(file_path).lower()
        
        # Use directory separator to avoid substring confusion
        # ('no_epilepsy' contains 'epilepsy', so check with separators)
        # Check no_epilepsy FIRST (more specific)
        if '\\01_no_epilepsy\\' in file_path_str or '/01_no_epilepsy/' in file_path_str:
            return 'normal'
        elif '\\no_epilepsy\\' in file_path_str or '/no_epilepsy/' in file_path_str:
            return 'normal'
        elif '\\00_epilepsy\\' in file_path_str or '/00_epilepsy/' in file_path_str:
            return 'epilepsy'
        elif '\\epilepsy\\' in file_path_str or '/epilepsy/' in file_path_str:
            return 'epilepsy'
        
        # Fallback: check CSV for background annotation
        csv_label = self._extract_csv(file_path, segment_idx, segment_duration)
        if csv_label != 'bckg':
            return csv_label
        
        return 'unknown'
    
    # ========================================================================
    # TUEV - LAB format (time in MICROSECONDS!)
    # ========================================================================
    
    def _extract_tuev(self, file_path: Path, segment_idx: int, 
                     segment_duration: float) -> str:
        """
        Extract label from TUEV .lab file.
        
        CRITICAL: Time is in MICROSECONDS, not seconds!
        """
        lab_file = file_path.with_suffix('.lab')
        
        if not lab_file.exists():
            return 'bckg'
        
        try:
            # Read tab-separated file
            df = pd.read_csv(lab_file, sep='\t', header=None, 
                           names=['start_us', 'stop_us', 'label'])
            
            # Convert microseconds to seconds
            df['start_time'] = df['start_us'] / 1_000_000
            df['stop_time'] = df['stop_us'] / 1_000_000
            
            # Calculate segment time window
            segment_start = segment_idx * segment_duration
            segment_end = segment_start + segment_duration
            
            # Find overlapping annotations
            mask = (df['start_time'] < segment_end) & (df['stop_time'] > segment_start)
            window_annotations = df[mask]
            
            if len(window_annotations) == 0:
                return 'bckg'
            
            # Duration-weighted aggregation
            return self._aggregate_duration_weighted(
                window_annotations, segment_start, segment_end
            )
        
        except Exception as e:
            print(f"Warning: LAB parsing failed for {lab_file}: {e}")
            return 'bckg'
    
    # ========================================================================
    # TUSL - CSV (preferred) or LBL format
    # ========================================================================
    
    def _extract_tusl(self, file_path: Path, segment_idx: int, 
                     segment_duration: float) -> str:
        """
        Extract label for TUSL (Slowing).
        
        Strategy: Try CSV first (simpler), fallback to LBL if needed.
        """
        # Try CSV first
        csv_file = file_path.with_suffix('.csv')
        if csv_file.exists():
            return self._extract_csv(file_path, segment_idx, segment_duration)
        
        # Fallback to LBL (complex format - simplified parsing)
        lbl_file = file_path.with_suffix('.lbl')
        if lbl_file.exists():
            return self._extract_lbl_simple(lbl_file, segment_idx, segment_duration)
        
        return 'bckg'
    
    def _extract_lbl_simple(self, lbl_file: Path, segment_idx: int, 
                           segment_duration: float) -> str:
        """
        Simplified LBL parser.
        
        LBL format is complex - this extracts basic labels.
        Full parser would need to decode the symbol mapping.
        """
        try:
            with open(lbl_file, 'r') as f:
                content = f.read()
            
            # Extract symbol definitions
            # symbols[0] = {0: '(null)', 1: 'bckg', 2: 'seiz', 3: 'slow'}
            symbols = {}
            for line in content.split('\n'):
                if 'symbols[0]' in line:
                    # Simple extraction - would need proper parsing for production
                    if 'bckg' in line:
                        return 'bckg'
                    elif 'slow' in line:
                        return 'slow'
                    elif 'seiz' in line:
                        return 'seiz'
            
            return 'bckg'
        
        except Exception as e:
            print(f"Warning: LBL parsing failed for {lbl_file}: {e}")
            return 'bckg'


# ============================================================================
# Convenience function for easy integration
# ============================================================================

def extract_label_unified(dataset: str, file_path: Path, segment_idx: int = 0, 
                         segment_duration: float = 10.0) -> str:
    """
    Extract label using unified parser.
    
    Args:
        dataset: Dataset name (TUAB, TUAR, TUEP, TUEV, TUSL, TUSZ)
        file_path: Path to EDF file
        segment_idx: Segment index
        segment_duration: Segment duration in seconds
        
    Returns:
        Label string
    """
    parser = UnifiedTUHParser(dataset)
    return parser.extract_label(file_path, segment_idx, segment_duration)


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    # Test each dataset
    examples = {
        'TUAB': Path("C:/path/to/tuab/edf/eval/abnormal/file.edf"),
        'TUAR': Path("C:/path/to/tuar/edf/01_tcp_ar/file.edf"),
        'TUEP': Path("C:/path/to/tuep/00_epilepsy/patient/session/montage/file.edf"),
        'TUEV': Path("C:/path/to/tuev/edf/eval/000/file.edf"),
        'TUSL': Path("C:/path/to/tusl/edf/patient/session/montage/file.edf"),
        'TUSZ': Path("C:/path/to/tusz/edf/dev/patient/session/montage/file.edf")
    }
    
    for dataset, file_path in examples.items():
        label = extract_label_unified(dataset, file_path, segment_idx=0, segment_duration=10.0)
        print(f"{dataset}: {label}")
