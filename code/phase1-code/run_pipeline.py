#!/usr/bin/env python3
"""
NeuroVault Master Pipeline
===========================

Complete preprocessing pipeline with all steps integrated.
This is the recommended way to run the full preprocessing workflow.
"""

import sys
import argparse
import subprocess
from pathlib import Path
import yaml
from datetime import datetime


class MasterPipeline:
    """Master pipeline orchestrator."""
    
    def __init__(self, config_path, data_root):
        """
        Initialize master pipeline.
        
        Args:
            config_path: Path to YAML config
            data_root: Root directory with all TUH datasets
        """
        self.config_path = Path(config_path)
        self.data_root = Path(data_root)
        
        with open(self.config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.datasets = {
            'TUAB': 'tuh_abnormal/v3.0.1/edf',
            'TUAR': 'tuh_artifact/v2.0.0/edf',
            'TUEP': 'tuh_epilepsy/v3.0.0/edf',
            'TUEV': 'tuh_events/v2.0.0/edf',
            'TUSL': 'tuh_slowing/v2.0.1/edf',
            'TUSZ': 'tuh_seizure/v2.0.5/edf'
        }
    
    def print_header(self):
        """Print pipeline header."""
        print("\n" + "="*80)
        print("NeuroVault TUH EEG Preprocessing - Master Pipeline")
        print("="*80)
        print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Config: {self.config_path}")
        print(f"Data root: {self.data_root}")
        print("="*80 + "\n")
    
    def step1_validate_config(self):
        """Step 1: Validate configuration."""
        print("\n" + "─"*80)
        print("STEP 1: Validate Configuration")
        print("─"*80 + "\n")
        
        # Check critical settings
        issues = []
        
        # Check channels
        n_channels = self.config['preprocessing']['n_channels']
        if n_channels != 22:
            issues.append(f"n_channels is {n_channels}, should be 22 for TUH TCP montage")
        
        # Check bandpass
        low_freq = self.config['preprocessing']['bandpass_filter']['low_freq']
        if low_freq == 0.1:
            issues.append(f"low_freq is 0.1 Hz, recommend 0.5 Hz for clinical EEG")
        
        # Check normalization
        if 'normalization' not in self.config['preprocessing']:
            issues.append("normalization not configured (should be global_zscore)")
        
        if issues:
            print("⚠ Configuration issues found:")
            for issue in issues:
                print(f"  • {issue}")
            print("\nReview config_tuh.yaml before proceeding.")
            response = input("\nContinue anyway? (y/n): ")
            if response.lower() != 'y':
                print("Exiting. Please fix configuration.")
                sys.exit(1)
        else:
            print("✓ Configuration validated")
    
    def step2_validate_datasets(self, datasets_to_process):
        """Step 2: Validate dataset structure."""
        print("\n" + "─"*80)
        print("STEP 2: Validate Datasets")
        print("─"*80 + "\n")
        
        valid_datasets = []
        
        for dataset_name in datasets_to_process:
            dataset_path = self.data_root / self.datasets[dataset_name]
            
            print(f"\nValidating {dataset_name}...")
            
            cmd = [
                sys.executable,
                'validate_dataset.py',
                '--config', str(self.config_path),
                '--dataset', dataset_name,
                '--data-dir', str(dataset_path),
                '--max-check', '5'
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                print(f"✓ {dataset_name} validated")
                valid_datasets.append(dataset_name)
            else:
                print(f"✗ {dataset_name} validation failed")
                print(result.stdout)
        
        if not valid_datasets:
            print("\n✗ No valid datasets found. Exiting.")
            sys.exit(1)
        
        print(f"\n✓ {len(valid_datasets)}/{len(datasets_to_process)} datasets validated")
        
        return valid_datasets
    
    def step3_run_diagnostics(self, datasets_to_process):
        """Step 3: Run diagnostic on sample files."""
        print("\n" + "─"*80)
        print("STEP 3: Run Diagnostics")
        print("─"*80 + "\n")
        
        for dataset_name in datasets_to_process:
            dataset_path = self.data_root / self.datasets[dataset_name]
            
            # Find first EDF file
            edf_files = list(dataset_path.rglob('*.edf'))
            if not edf_files:
                continue
            
            sample_file = edf_files[0]
            
            print(f"\nDiagnostic for {dataset_name}...")
            print(f"Sample file: {sample_file.name}")
            
            cmd = [
                sys.executable,
                'diagnostic.py',
                '--config', str(self.config_path),
                '--edf-file', str(sample_file)
            ]
            
            subprocess.run(cmd)
    
    def step4_test_preprocessing(self, datasets_to_process, test_size=5):
        """Step 4: Test preprocessing on small sample."""
        print("\n" + "─"*80)
        print(f"STEP 4: Test Preprocessing ({test_size} files per dataset)")
        print("─"*80 + "\n")
        
        for dataset_name in datasets_to_process:
            dataset_path = self.data_root / self.datasets[dataset_name]
            
            print(f"\nTesting {dataset_name}...")
            
            cmd = [
                sys.executable,
                'preprocess_tuh.py',
                '--config', str(self.config_path),
                '--dataset', dataset_name,
                '--data-dir', str(dataset_path),
                '--max-files', str(test_size)
            ]
            
            result = subprocess.run(cmd)
            
            if result.returncode != 0:
                print(f"✗ Test failed for {dataset_name}")
                return False
        
        print("\n✓ All test runs completed successfully")
        return True
    
    def step5_review_test_results(self, datasets_to_process):
        """Step 5: Review test results."""
        print("\n" + "─"*80)
        print("STEP 5: Review Test Results")
        print("─"*80 + "\n")
        
        for dataset_name in datasets_to_process:
            base_dir = Path(self.config['output']['base_dir']) / f"neurovault_{dataset_name.lower()}"
            stats_file = base_dir / 'statistics' / 'preprocessing_stats.json'
            
            if stats_file.exists():
                import json
                with open(stats_file, 'r') as f:
                    stats = json.load(f)
                
                total_segs = stats['total_segments']
                rejected = stats['rejected_segments']
                accepted = total_segs - rejected
                yield_rate = (accepted / max(total_segs, 1)) * 100
                
                print(f"\n{dataset_name}:")
                print(f"  Files: {stats['processed_files']}/{stats['total_files']}")
                print(f"  Segments: {accepted}/{total_segs} accepted ({yield_rate:.1f}% yield)")
                
                if yield_rate < 20:
                    print(f"  ⚠ WARNING: Very low yield rate!")
                elif yield_rate < 40:
                    print(f"  ⚠ NOTE: Moderate yield rate")
                else:
                    print(f"  ✓ Good yield rate")
        
        print("\nReview the statistics and visualizations in neurovault_data/")
        response = input("\nProceed with full preprocessing? (y/n): ")
        
        return response.lower() == 'y'
    
    def step6_full_preprocessing(self, datasets_to_process):
        """Step 6: Full preprocessing."""
        print("\n" + "─"*80)
        print("STEP 6: Full Preprocessing")
        print("─"*80 + "\n")
        
        print(f"Processing {len(datasets_to_process)} datasets: {', '.join(datasets_to_process)}")
        print("\nThis may take several hours...")
        
        cmd = [
            sys.executable,
            'batch_process.py',
            '--config', str(self.config_path),
            '--data-root', str(self.data_root)
        ]
        
        if len(datasets_to_process) < 6:
            cmd.extend(['--datasets'] + datasets_to_process)
        
        result = subprocess.run(cmd)
        
        return result.returncode == 0
    
    def step7_generate_summary(self):
        """Step 7: Generate final summary."""
        print("\n" + "─"*80)
        print("STEP 7: Generate Summary Report")
        print("─"*80 + "\n")
        
        summary_file = Path(self.config['output']['base_dir']) / 'batch_processing_summary.json'
        
        if summary_file.exists():
            import json
            with open(summary_file, 'r') as f:
                summary = json.load(f)
            
            print("Processing Results:")
            for dataset_name, result in summary['results'].items():
                status = result['status']
                if status == 'success':
                    duration_min = result['duration_seconds'] / 60
                    print(f"  ✓ {dataset_name}: Success ({duration_min:.1f} min)")
                elif status == 'failed':
                    print(f"  ✗ {dataset_name}: Failed")
                else:
                    print(f"  ⊘ {dataset_name}: {status}")
        
        print(f"\n✓ Preprocessing complete!")
        print(f"\nOutput location: {Path(self.config['output']['base_dir']).absolute()}")
    
    def run_interactive(self, datasets=None, skip_test=False):
        """Run interactive pipeline."""
        self.print_header()
        
        # Determine datasets to process
        if datasets:
            datasets_to_process = datasets
        else:
            datasets_to_process = list(self.datasets.keys())
        
        print(f"Datasets to process: {', '.join(datasets_to_process)}\n")
        
        # Step 1: Validate config
        self.step1_validate_config()
        
        # Step 2: Validate datasets
        valid_datasets = self.step2_validate_datasets(datasets_to_process)
        
        if not skip_test:
            # Step 3: Diagnostics
            self.step3_run_diagnostics(valid_datasets)
            
            response = input("\nProceed to test preprocessing? (y/n): ")
            if response.lower() != 'y':
                print("Exiting.")
                return
            
            # Step 4: Test preprocessing
            if not self.step4_test_preprocessing(valid_datasets):
                print("\n✗ Test preprocessing failed. Fix errors before proceeding.")
                return
            
            # Step 5: Review test results
            if not self.step5_review_test_results(valid_datasets):
                print("Exiting.")
                return
        
        # Step 6: Full preprocessing
        success = self.step6_full_preprocessing(valid_datasets)
        
        # Step 7: Summary
        if success:
            self.step7_generate_summary()
        else:
            print("\n✗ Preprocessing failed. Check logs for details.")
    
    def run_auto(self, datasets=None):
        """Run automatic pipeline (no interaction)."""
        self.print_header()
        
        if datasets:
            datasets_to_process = datasets
        else:
            datasets_to_process = list(self.datasets.keys())
        
        print(f"Auto mode: Processing {', '.join(datasets_to_process)}")
        print("No test run - proceeding directly to full preprocessing\n")
        
        # Validate and run
        valid_datasets = self.step2_validate_datasets(datasets_to_process)
        success = self.step6_full_preprocessing(valid_datasets)
        
        if success:
            self.step7_generate_summary()


def main():
    parser = argparse.ArgumentParser(description='NeuroVault Master Pipeline')
    parser.add_argument('--config', type=str, default='config_tuh.yaml',
                       help='Path to config YAML')
    parser.add_argument('--data-root', type=str, required=True,
                       help='Root directory with all TUH datasets')
    parser.add_argument('--datasets', type=str, nargs='+',
                       choices=['TUAB', 'TUAR', 'TUEP', 'TUEV', 'TUSL', 'TUSZ'],
                       help='Specific datasets to process (default: all)')
    parser.add_argument('--auto', action='store_true',
                       help='Run automatically without test or confirmation')
    parser.add_argument('--skip-test', action='store_true',
                       help='Skip test preprocessing step')
    
    args = parser.parse_args()
    
    pipeline = MasterPipeline(args.config, args.data_root)
    
    if args.auto:
        pipeline.run_auto(datasets=args.datasets)
    else:
        pipeline.run_interactive(datasets=args.datasets, skip_test=args.skip_test)


if __name__ == '__main__':
    main()
