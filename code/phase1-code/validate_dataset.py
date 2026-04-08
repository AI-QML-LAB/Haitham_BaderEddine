#!/usr/bin/env python3
"""
TUH Dataset Validator
=====================

Validates TUH EEG dataset structure before preprocessing.
Checks for:
- Correct directory structure
- Required files and formats
- Annotation files
- Dataset-specific requirements
"""

import sys
from pathlib import Path
import argparse
import yaml
import mne
from collections import defaultdict


class TUHDatasetValidator:
    """Validate TUH dataset structure and files."""
    
    def __init__(self, config_path):
        """Initialize with config."""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.dataset_specs = {
            'TUAB': {
                'expected_subdirs': ['eval', 'train'],
                'expected_sub_subdirs': ['abnormal', 'normal'],
                'annotation_files': [],
                'metadata_files': []
            },
            'TUAR': {
                'expected_subdirs': ['01_tcp_ar', '02_tcp_le', '03_tcp_ar_a'],
                'annotation_files': ['.csv'],
                'metadata_files': []
            },
            'TUEP': {
                'expected_subdirs': ['00_epilepsy', '01_no_epilepsy'],
                'annotation_files': ['.csv', '.csv_bi'],
                'metadata_files': ['DOCS/metadata_v00r.xlsx']
            },
            'TUEV': {
                'expected_subdirs': ['eval', 'train'],
                'annotation_files': ['.rec', '.lab', '.htk'],
                'metadata_files': []
            },
            'TUSL': {
                'expected_subdirs': ['eval', 'train'],
                'annotation_files': ['.tse', '.tse_agg', '.lbl', '.lbl_agg'],
                'metadata_files': []
            },
            'TUSZ': {
                'expected_subdirs': ['dev', 'eval', 'train'],
                'annotation_files': ['.csv', '.csv_bi'],
                'metadata_files': ['DOCS']
            }
        }
    
    def validate_directory_structure(self, dataset_path, dataset_name):
        """Validate dataset directory structure."""
        print(f"\n{'='*80}")
        print(f"Validating {dataset_name} Dataset Structure")
        print(f"{'='*80}")
        print(f"Path: {dataset_path}\n")
        
        if not dataset_path.exists():
            print(f"✗ ERROR: Dataset path does not exist!")
            return False
        
        specs = self.dataset_specs.get(dataset_name)
        if not specs:
            print(f"✗ ERROR: Unknown dataset: {dataset_name}")
            return False
        
        # Check expected subdirectories
        print("Checking directory structure...")
        expected_subdirs = specs['expected_subdirs']
        
        all_found = True
        for subdir in expected_subdirs:
            subdir_path = dataset_path / subdir
            if subdir_path.exists():
                print(f"  ✓ Found: {subdir}/")
            else:
                print(f"  ✗ Missing: {subdir}/")
                all_found = False
        
        return all_found
    
    def validate_edf_files(self, dataset_path, dataset_name, max_check=10):
        """Validate EDF files."""
        print(f"\nChecking EDF files...")
        
        edf_files = sorted(list(dataset_path.rglob('*.edf')))
        
        if len(edf_files) == 0:
            print(f"  ✗ No EDF files found!")
            return False
        
        print(f"  ✓ Found {len(edf_files)} EDF files")
        
        # Check sample of files
        print(f"\nValidating sample of {min(max_check, len(edf_files))} EDF files...")
        
        valid_count = 0
        invalid_files = []
        
        for i, edf_file in enumerate(edf_files[:max_check]):
            try:
                raw = mne.io.read_raw_edf(edf_file, preload=False, verbose=False)
                
                # Basic checks
                sfreq = raw.info['sfreq']
                n_channels = len(raw.ch_names)
                duration = raw.times[-1]
                
                if i == 0:
                    print(f"\n  Sample file: {edf_file.name}")
                    print(f"    - Sampling rate: {sfreq} Hz")
                    print(f"    - Channels: {n_channels}")
                    print(f"    - Duration: {duration:.2f} seconds")
                    print(f"    - Channel names: {raw.ch_names[:5]}..." if n_channels > 5 else f"    - Channel names: {raw.ch_names}")
                
                valid_count += 1
                
            except Exception as e:
                invalid_files.append((edf_file.name, str(e)))
        
        print(f"\n  ✓ {valid_count}/{min(max_check, len(edf_files))} files validated successfully")
        
        if invalid_files:
            print(f"  ✗ {len(invalid_files)} files failed validation:")
            for fname, error in invalid_files[:3]:
                print(f"    - {fname}: {error}")
        
        return len(invalid_files) == 0
    
    def validate_annotations(self, dataset_path, dataset_name):
        """Validate annotation files."""
        specs = self.dataset_specs.get(dataset_name)
        if not specs or not specs['annotation_files']:
            print(f"\n  • No annotation files expected for {dataset_name}")
            return True
        
        print(f"\nChecking annotation files...")
        
        # Count annotation files by type
        annotation_counts = defaultdict(int)
        for ext in specs['annotation_files']:
            files = list(dataset_path.rglob(f'*{ext}'))
            annotation_counts[ext] = len(files)
        
        all_found = True
        for ext in specs['annotation_files']:
            count = annotation_counts[ext]
            if count > 0:
                print(f"  ✓ Found {count} {ext} files")
            else:
                print(f"  ✗ No {ext} files found")
                all_found = False
        
        return all_found
    
    def validate_metadata(self, dataset_path, dataset_name):
        """Validate metadata files."""
        specs = self.dataset_specs.get(dataset_name)
        if not specs or not specs['metadata_files']:
            return True
        
        print(f"\nChecking metadata files...")
        
        all_found = True
        for metadata_file in specs['metadata_files']:
            metadata_path = dataset_path / metadata_file
            if metadata_path.exists():
                print(f"  ✓ Found: {metadata_file}")
            else:
                print(f"  ✗ Missing: {metadata_file}")
                all_found = False
        
        return all_found
    
    def estimate_dataset_size(self, dataset_path):
        """Estimate total dataset size."""
        print(f"\nEstimating dataset size...")
        
        edf_files = list(dataset_path.rglob('*.edf'))
        
        # Calculate total size
        total_size_bytes = sum(f.stat().st_size for f in edf_files)
        total_size_gb = total_size_bytes / (1024**3)
        
        # Estimate duration
        sample_duration = 0
        if len(edf_files) > 0:
            try:
                raw = mne.io.read_raw_edf(edf_files[0], preload=False, verbose=False)
                sample_duration = raw.times[-1]
            except:
                pass
        
        if sample_duration > 0:
            estimated_hours = (sample_duration * len(edf_files)) / 3600
            print(f"  • Total files: {len(edf_files)}")
            print(f"  • Total size: {total_size_gb:.2f} GB")
            print(f"  • Estimated duration: {estimated_hours:.1f} hours")
        else:
            print(f"  • Total files: {len(edf_files)}")
            print(f"  • Total size: {total_size_gb:.2f} GB")
    
    def check_config_compatibility(self, dataset_name):
        """Check if config matches dataset requirements."""
        print(f"\nChecking configuration compatibility...")
        
        dataset_config = self.config['datasets'].get(dataset_name)
        if not dataset_config:
            print(f"  ✗ No configuration found for {dataset_name}")
            return False
        
        # Check annotation format
        if 'annotation_format' in dataset_config:
            ann_format = dataset_config['annotation_format']
            print(f"  ✓ Annotation format: {ann_format}")
        
        # Check if reports are used
        use_reports = dataset_config.get('use_reports', False)
        use_annotations = dataset_config.get('use_annotations', False)
        
        if use_reports:
            print(f"  ✓ Will use text reports")
        elif use_annotations:
            print(f"  ✓ Will use annotation files")
        else:
            print(f"  ✓ Will use path-based labels")
        
        return True
    
    def validate_dataset(self, dataset_path, dataset_name, max_check=10):
        """Run complete validation on dataset."""
        success = True
        
        # 1. Directory structure
        if not self.validate_directory_structure(dataset_path, dataset_name):
            success = False
        
        # 2. EDF files
        if not self.validate_edf_files(dataset_path, dataset_name, max_check):
            success = False
        
        # 3. Annotations
        if not self.validate_annotations(dataset_path, dataset_name):
            print(f"  ⚠ Warning: Some annotation files missing (may affect labeling)")
        
        # 4. Metadata
        if not self.validate_metadata(dataset_path, dataset_name):
            print(f"  ⚠ Warning: Some metadata files missing")
        
        # 5. Size estimation
        self.estimate_dataset_size(dataset_path)
        
        # 6. Config compatibility
        if not self.check_config_compatibility(dataset_name):
            success = False
        
        # Summary
        print(f"\n{'='*80}")
        if success:
            print(f"✓ {dataset_name} dataset validation PASSED")
            print(f"  Ready for preprocessing!")
        else:
            print(f"✗ {dataset_name} dataset validation FAILED")
            print(f"  Please fix errors before preprocessing.")
        print(f"{'='*80}\n")
        
        return success


def main():
    parser = argparse.ArgumentParser(description='Validate TUH dataset structure')
    parser.add_argument('--config', type=str, default='config_tuh.yaml',
                       help='Path to config YAML')
    parser.add_argument('--dataset', type=str, required=True,
                       choices=['TUAB', 'TUAR', 'TUEP', 'TUEV', 'TUSL', 'TUSZ'],
                       help='Dataset name')
    parser.add_argument('--data-dir', type=str, required=True,
                       help='Path to dataset directory')
    parser.add_argument('--max-check', type=int, default=10,
                       help='Maximum files to validate')
    
    args = parser.parse_args()
    
    validator = TUHDatasetValidator(args.config)
    success = validator.validate_dataset(
        Path(args.data_dir),
        args.dataset,
        max_check=args.max_check
    )
    
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
