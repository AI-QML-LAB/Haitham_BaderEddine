#!/usr/bin/env python3
"""
NeuroVault Batch Processor
===========================

Process all 6 TUH datasets with integrated preprocessing and visualization.
"""

import sys
import subprocess
from pathlib import Path
import yaml
import json
from datetime import datetime
import argparse


class BatchProcessor:
    """Batch process multiple TUH datasets."""
    
    def __init__(self, config_path, data_root, max_files_per_dataset=None):
        """
        Initialize batch processor.
        
        Args:
            config_path: Path to YAML config
            data_root: Root directory containing all TUH datasets
            max_files_per_dataset: Optional limit for testing
        """
        self.config_path = Path(config_path)
        self.data_root = Path(data_root)
        self.max_files = max_files_per_dataset
        
        # Load config
        with open(self.config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Define datasets
        self.datasets = {
            'TUAB': 'tuh_abnormal/v3.0.1/edf',
            'TUAR': 'tuh_artifact/v2.0.0/edf',
            'TUEP': 'tuh_epilepsy/v3.0.0/edf',
            'TUEV': 'tuh_events/v2.0.0/edf',
            'TUSL': 'tuh_slowing/v2.0.1/edf',
            'TUSZ': 'tuh_seizure/v2.0.5/edf'
        }
        
        self.results = {}
    
    def check_dataset_exists(self, dataset_name):
        """Check if dataset directory exists."""
        dataset_path = self.data_root / self.datasets[dataset_name]
        
        if not dataset_path.exists():
            print(f"  ✗ Dataset directory not found: {dataset_path}")
            return False
        
        # Count EDF files
        edf_files = list(dataset_path.rglob('*.edf'))
        print(f"  ✓ Found {len(edf_files)} EDF files")
        
        return True
    
    def run_preprocessing(self, dataset_name):
        """
        Run preprocessing for a single dataset.
        
        Returns:
            bool: Success status
        """
        print(f"\n{'='*80}")
        print(f"Processing {dataset_name}")
        print(f"{'='*80}")
        
        # Check if dataset exists
        if not self.check_dataset_exists(dataset_name):
            self.results[dataset_name] = {'status': 'skipped', 'reason': 'dataset not found'}
            return False
        
        # Construct command
        cmd = [
            sys.executable,  # Use same Python interpreter
            'preprocess_tuh.py',
            '--config', str(self.config_path),
            '--dataset', dataset_name,
            '--data-dir', str(self.data_root / self.datasets[dataset_name])
        ]
        
        if self.max_files:
            cmd.extend(['--max-files', str(self.max_files)])
        
        # Run preprocessing
        start_time = datetime.now()
        
        try:
            print(f"\nStarting preprocessing at {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"Command: {' '.join(cmd)}\n")
            
            result = subprocess.run(cmd, check=True, capture_output=False, text=True)
            
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            print(f"\n✓ Preprocessing completed in {duration:.0f} seconds ({duration/60:.1f} minutes)")
            
            self.results[dataset_name] = {
                'status': 'success',
                'duration_seconds': duration,
                'start_time': start_time.isoformat(),
                'end_time': end_time.isoformat()
            }
            
            return True
        
        except subprocess.CalledProcessError as e:
            print(f"\n✗ Preprocessing failed with error code {e.returncode}")
            self.results[dataset_name] = {
                'status': 'failed',
                'error': str(e)
            }
            return False
        
        except Exception as e:
            print(f"\n✗ Unexpected error: {e}")
            self.results[dataset_name] = {
                'status': 'error',
                'error': str(e)
            }
            return False
    
    def run_visualization(self, dataset_name):
        """
        Run visualization for a preprocessed dataset.
        
        Returns:
            bool: Success status
        """
        print(f"\n{'-'*80}")
        print(f"Generating visualizations for {dataset_name}")
        print(f"{'-'*80}")
        
        # Define paths
        base_dir = Path(self.config['output']['base_dir']) / f"neurovault_{dataset_name.lower()}"
        preprocessed_dir = base_dir / self.config['output']['preprocessed_dir']
        viz_dir = base_dir / self.config['output']['visualizations_dir']
        stats_dir = base_dir / self.config['output']['statistics_dir']
        
        # Check if preprocessed data exists
        if not preprocessed_dir.exists() or len(list(preprocessed_dir.glob('*.pkl'))) == 0:
            print(f"  ✗ No preprocessed data found, skipping visualization")
            return False
        
        # Find a raw EDF file for before/after comparison
        data_dir = self.data_root / self.datasets[dataset_name]
        edf_files = list(data_dir.rglob('*.edf'))
        raw_edf = str(edf_files[0]) if len(edf_files) > 0 else None
        
        # Construct command
        cmd = [
            sys.executable,
            'visualizer.py',
            '--preprocessed-dir', str(preprocessed_dir),
            '--viz-dir', str(viz_dir),
            '--stats-dir', str(stats_dir),
            '--config', str(self.config_path)
        ]
        
        if raw_edf:
            cmd.extend(['--raw-edf', raw_edf])
        
        try:
            result = subprocess.run(cmd, check=True, capture_output=False, text=True)
            print(f"✓ Visualizations generated")
            return True
        
        except subprocess.CalledProcessError as e:
            print(f"✗ Visualization failed: {e}")
            return False
    
    def save_batch_summary(self):
        """Save batch processing summary."""
        summary_file = Path(self.config['output']['base_dir']) / 'batch_processing_summary.json'
        
        summary = {
            'processing_date': datetime.now().isoformat(),
            'config_file': str(self.config_path),
            'data_root': str(self.data_root),
            'max_files_per_dataset': self.max_files,
            'results': self.results
        }
        
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\n{'='*80}")
        print(f"Batch Summary saved to: {summary_file}")
        print(f"{'='*80}\n")
    
    def print_summary(self):
        """Print final summary."""
        print(f"\n{'='*80}")
        print("BATCH PROCESSING SUMMARY")
        print(f"{'='*80}\n")
        
        successful = sum(1 for r in self.results.values() if r['status'] == 'success')
        failed = sum(1 for r in self.results.values() if r['status'] in ['failed', 'error'])
        skipped = sum(1 for r in self.results.values() if r['status'] == 'skipped')
        
        print(f"Total datasets: {len(self.results)}")
        print(f"  ✓ Successful: {successful}")
        print(f"  ✗ Failed: {failed}")
        print(f"  ⊘ Skipped: {skipped}")
        
        print(f"\nDataset Details:")
        for dataset_name, result in self.results.items():
            status_symbol = {
                'success': '✓',
                'failed': '✗',
                'error': '✗',
                'skipped': '⊘'
            }.get(result['status'], '?')
            
            print(f"  {status_symbol} {dataset_name}: {result['status']}", end='')
            
            if result['status'] == 'success':
                duration_min = result['duration_seconds'] / 60
                print(f" ({duration_min:.1f} min)")
            elif result['status'] in ['failed', 'error']:
                print(f" - {result.get('error', 'unknown error')}")
            else:
                print(f" - {result.get('reason', 'unknown reason')}")
        
        print(f"\n{'='*80}\n")
    
    def run_all(self, datasets=None, skip_visualization=False):
        """
        Run batch processing on all or selected datasets.
        
        Args:
            datasets: List of dataset names, or None for all
            skip_visualization: If True, skip visualization step
        """
        if datasets is None:
            datasets = list(self.datasets.keys())
        
        print(f"\n{'='*80}")
        print(f"NeuroVault Batch Preprocessing")
        print(f"{'='*80}")
        print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Datasets to process: {', '.join(datasets)}")
        print(f"Max files per dataset: {self.max_files if self.max_files else 'unlimited'}")
        print(f"{'='*80}\n")
        
        for dataset_name in datasets:
            # Run preprocessing
            success = self.run_preprocessing(dataset_name)
            
            # Run visualization if preprocessing succeeded
            if success and not skip_visualization:
                self.run_visualization(dataset_name)
        
        # Save summary
        self.save_batch_summary()
        
        # Print summary
        self.print_summary()


def main():
    parser = argparse.ArgumentParser(description='NeuroVault Batch Processor')
    parser.add_argument('--config', type=str, default='config_tuh.yaml',
                       help='Path to config YAML')
    parser.add_argument('--data-root', type=str, required=True,
                       help='Root directory containing all TUH datasets')
    parser.add_argument('--datasets', type=str, nargs='+', 
                       choices=['TUAB', 'TUAR', 'TUEP', 'TUEV', 'TUSL', 'TUSZ'],
                       help='Specific datasets to process (default: all)')
    parser.add_argument('--max-files', type=int, default=None,
                       help='Maximum files per dataset (for testing)')
    parser.add_argument('--skip-viz', action='store_true',
                       help='Skip visualization generation')
    
    args = parser.parse_args()
    
    # Create batch processor
    processor = BatchProcessor(
        config_path=args.config,
        data_root=args.data_root,
        max_files_per_dataset=args.max_files
    )
    
    # Run
    processor.run_all(datasets=args.datasets, skip_visualization=args.skip_viz)


if __name__ == '__main__':
    main()
