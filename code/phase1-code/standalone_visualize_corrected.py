#!/usr/bin/env python3
"""
Standalone EEG Visualization Generator - CORRECTED
===================================================

Generates visualizations from already-preprocessed EEG datasets.
Use this AFTER preprocessing to generate clear visualizations.

FIXES:
- Fixed frequency_bands extraction from features
- Removed preprocessing_pipeline visualization
- Improved label diversity sampling
- Better error handling

Usage:
    python standalone_visualize_corrected.py --dataset TUAB
    python standalone_visualize_corrected.py --dataset TUAR
    python standalone_visualize_corrected.py --dataset all

Author: Haitham
Date: March 2026
Version: 2.0 (CORRECTED)
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import signal
import pickle
import argparse
from typing import List, Dict
from collections import Counter
from tqdm import tqdm
import random

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


class StandaloneEEGVisualizer:
    """Generate visualizations from preprocessed EEG data."""
    
    def __init__(self, dataset_name, max_samples=5, dpi=300):
        """
        Initialize visualizer.
        
        Args:
            dataset_name: Name of dataset (TUAB, TUAR, TUEP, etc.)
            max_samples: Number of sample segments to visualize
            dpi: Resolution for saved figures
        """
        self.dataset_name = dataset_name
        self.max_samples = max_samples
        self.dpi = dpi
        
        # Plot types (removed preprocessing_pipeline)
        self.plot_types = [
            'raw_eeg_traces',
            'psd_analysis', 
            'frequency_bands',
            'spatial_distribution',
            'dataset_statistics',
            'quality_metrics'
        ]
    
    def generate_all(self, preprocessed_dir: Path, output_dir: Path):
        """Generate all visualizations."""
        preprocessed_dir = Path(preprocessed_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\n{'=' * 80}")
        print(f"GENERATING VISUALIZATIONS: {self.dataset_name}")
        print(f"{'=' * 80}")
        print(f"📁 Preprocessed dir: {preprocessed_dir}")
        print(f"📂 Output dir: {output_dir}")
        print(f"🎨 Plot types: {len(self.plot_types)}")
        print(f"📊 Max samples: {self.max_samples}")
        print()
        
        # Load sample segments
        segment_files = sorted(list(preprocessed_dir.glob("*.pkl")))
        
        if not segment_files:
            print(f"❌ No segments found in {preprocessed_dir}")
            return
        
        print(f"Found {len(segment_files):,} segments")
        
        # Sample segments with DIVERSE LABELS
        print(f"Sampling segments to ensure label diversity...")
        
        # Random sample to find label distribution
        check_size = min(1000, len(segment_files))
        random_sample = random.sample(segment_files, check_size)
        
        segments_by_label = {}
        
        for f in random_sample:
            try:
                with open(f, 'rb') as file:
                    data = pickle.load(file)
                    label = data['metadata'].get('label', 'unknown')
                    
                    if label not in segments_by_label:
                        segments_by_label[label] = []
                    segments_by_label[label].append(f)
            except:
                continue
        
        print(f"Found {len(segments_by_label)} unique labels: {', '.join(sorted(segments_by_label.keys()))}")
        
        # Sample from each label to ensure diversity
        sample_files = []
        samples_per_label = max(1, self.max_samples // len(segments_by_label))
        
        for label, files in segments_by_label.items():
            n_samples = min(samples_per_label, len(files))
            sample_files.extend(random.sample(files, n_samples) if len(files) > n_samples else files)
        
        # If we need more samples, add random ones
        if len(sample_files) < self.max_samples:
            remaining = self.max_samples - len(sample_files)
            additional = random.sample(segment_files, min(remaining, len(segment_files)))
            sample_files.extend(additional)
        
        # Limit to max_samples
        sample_files = sample_files[:self.max_samples]
        
        print(f"Visualizing {len(sample_files)} samples with diverse labels\n")
        
        # Load sample data
        sample_data = []
        for f in sample_files:
            try:
                with open(f, 'rb') as file:
                    sample_data.append(pickle.load(file))
            except Exception as e:
                print(f"⚠️  Failed to load {f.name}: {e}")
                continue
        
        if not sample_data:
            print("❌ No valid samples loaded")
            return
        
        # DEBUG: Print labels in sample_data
        print("🔍 Labels in loaded samples:")
        for i, data in enumerate(sample_data):
            label = data['metadata'].get('label', 'unknown')
            print(f"  Sample {i+1}: {label}")
        print()
        
        # Compute statistics
        stats = self._compute_statistics(preprocessed_dir, segment_files)
        
        # Generate each plot type
        for plot_type in self.plot_types:
            try:
                print(f"  Generating {plot_type}...", end=' ')
                
                if plot_type == 'raw_eeg_traces':
                    self._plot_raw_eeg_traces(sample_data, output_dir)
                elif plot_type == 'psd_analysis':
                    self._plot_psd_analysis(sample_data, output_dir)
                elif plot_type == 'frequency_bands':
                    self._plot_frequency_bands(sample_data, output_dir)
                elif plot_type == 'spatial_distribution':
                    self._plot_spatial_distribution(sample_data, output_dir)
                elif plot_type == 'dataset_statistics':
                    self._plot_dataset_statistics(stats, output_dir)
                elif plot_type == 'quality_metrics':
                    self._plot_quality_metrics(stats, output_dir)
                
                print("✓")
                
            except Exception as e:
                print(f"✗ ({e})")
                import traceback
                traceback.print_exc()
        
        print(f"\n{'=' * 80}")
        print(f"✅ VISUALIZATION COMPLETE")
        print(f"{'=' * 80}")
        print(f"📂 Saved to: {output_dir}")
        print(f"{'=' * 80}\n")
    
    def _compute_statistics(self, preprocessed_dir, segment_files):
        """Compute dataset statistics."""
        print("Computing statistics...")
        
        # Random sample for label distribution
        sample_size = min(1000, len(segment_files))
        sample_files = random.sample(segment_files, sample_size)
        
        labels = []
        for f in tqdm(sample_files, desc="  Sampling labels", ncols=80):
            try:
                with open(f, 'rb') as file:
                    data = pickle.load(file)
                labels.append(data['metadata'].get('label', 'unknown'))
            except:
                pass
        
        label_counts = Counter(labels)
        
        # Extrapolate to full dataset
        total_segments = len(segment_files)
        extrapolated_labels = {}
        for label, count in label_counts.items():
            extrapolated_labels[label] = int(count * total_segments / sample_size)
        
        stats = {
            'total_segments': total_segments,
            'label_distribution': extrapolated_labels,
            'sample_size': sample_size
        }
        
        print()
        return stats
    
    # ========================================================================
    # 1. RAW EEG TRACES
    # ========================================================================
    
    def _plot_raw_eeg_traces(self, sample_data, output_dir):
        """Plot raw EEG time series."""
        for idx, data in enumerate(sample_data):
            segment = data['segment']
            metadata = data['metadata']
            
            n_channels, n_samples = segment.shape
            sfreq = metadata['sfreq']
            time = np.arange(n_samples) / sfreq
            
            fig, axes = plt.subplots(n_channels, 1, figsize=(15, n_channels * 1.2), 
                                    sharex=True)
            
            if n_channels == 1:
                axes = [axes]
            
            for ch_idx in range(n_channels):
                axes[ch_idx].plot(time, segment[ch_idx], linewidth=0.6, 
                                color='#2E86AB', alpha=0.8)
                axes[ch_idx].set_ylabel(f'Ch {ch_idx+1}\n(z-score)', fontsize=9, 
                                       rotation=0, ha='right', va='center')
                axes[ch_idx].grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
                axes[ch_idx].set_xlim([0, time[-1]])
                
                if ch_idx < n_channels - 1:
                    axes[ch_idx].set_xticklabels([])
            
            axes[-1].set_xlabel('Time (seconds)', fontsize=11, fontweight='bold')
            
            label = metadata.get('label', 'unknown')
            file_name = Path(metadata['file_path']).name
            
            fig.suptitle(
                f"{self.dataset_name} | File: {file_name} | Segment: {metadata['segment_idx']} | "
                f"Label: {label.upper()} | {n_channels} channels",
                fontsize=12, fontweight='bold', y=0.995
            )
            
            plt.tight_layout()
            output_file = output_dir / f"raw_eeg_traces_{idx:03d}.png"
            plt.savefig(output_file, dpi=self.dpi, bbox_inches='tight', facecolor='white')
            plt.close()
    
    # ========================================================================
    # 2. PSD ANALYSIS
    # ========================================================================
    
    def _plot_psd_analysis(self, sample_data, output_dir):
        """Plot power spectral density."""
        for idx, data in enumerate(sample_data):
            segment = data['segment']
            metadata = data['metadata']
            
            n_channels = segment.shape[0]
            sfreq = metadata['sfreq']
            
            fig, axes = plt.subplots(n_channels, 1, figsize=(12, n_channels * 1.5), 
                                    sharex=True)
            
            if n_channels == 1:
                axes = [axes]
            
            bands = {
                'Delta': (0.5, 4),
                'Theta': (4, 8),
                'Alpha': (8, 13),
                'Beta': (13, 30),
                'Gamma': (30, 50)
            }
            band_colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8']
            
            for ch_idx in range(n_channels):
                freqs, psd = signal.welch(segment[ch_idx], fs=sfreq, 
                                        nperseg=256, noverlap=128)
                
                axes[ch_idx].semilogy(freqs, psd, linewidth=1.5, 
                                     color='#2E86AB', alpha=0.8)
                axes[ch_idx].set_ylabel(f'Ch {ch_idx+1}\nPSD', fontsize=9, 
                                       rotation=0, ha='right', va='center')
                axes[ch_idx].set_xlim([0, 50])
                axes[ch_idx].grid(True, alpha=0.3, which='both')
                
                for (band_name, (low, high)), color in zip(bands.items(), band_colors):
                    axes[ch_idx].axvspan(low, high, alpha=0.15, color=color, 
                                        label=band_name if ch_idx == 0 else '')
                
                if ch_idx < n_channels - 1:
                    axes[ch_idx].set_xticklabels([])
            
            axes[-1].set_xlabel('Frequency (Hz)', fontsize=11, fontweight='bold')
            axes[0].legend(loc='upper right', fontsize=8, ncol=5)
            
            label = metadata.get('label', 'unknown')
            fig.suptitle(
                f"Power Spectral Density | {self.dataset_name} | Label: {label.upper()}",
                fontsize=12, fontweight='bold'
            )
            
            plt.tight_layout()
            output_file = output_dir / f"psd_analysis_{idx:03d}.png"
            plt.savefig(output_file, dpi=self.dpi, bbox_inches='tight', facecolor='white')
            plt.close()
    
    # ========================================================================
    # 3. FREQUENCY BANDS (FIXED)
    # ========================================================================
    
    def _plot_frequency_bands(self, sample_data, output_dir):
        """Plot frequency band power distribution."""
        # FIXED: Extract band powers from features properly
        band_names = ['delta', 'theta', 'alpha', 'beta', 'gamma']
        band_powers_by_label = {}
        
        # Collect band powers for each sample
        for data in sample_data:
            label = data['metadata'].get('label', 'unknown')
            segment = data['segment']
            sfreq = data['metadata']['sfreq']
            
            # FIXED: Compute band powers from segment directly
            # (features may not exist in older preprocessed files)
            band_powers = self._compute_band_powers(segment, sfreq)
            
            if label not in band_powers_by_label:
                band_powers_by_label[label] = {bn: [] for bn in band_names}
            
            for band_name in band_names:
                if band_name in band_powers:
                    band_powers_by_label[label][band_name].append(band_powers[band_name])
        
        # Check if we have data to plot
        if not band_powers_by_label or not any(band_powers_by_label.values()):
            print("  ⚠️  No band power data available, skipping frequency_bands")
            return
        
        # Create subplots for each label
        n_labels = len(band_powers_by_label)
        fig, axes = plt.subplots(1, n_labels, figsize=(5 * n_labels, 6))
        
        if n_labels == 1:
            axes = [axes]
        
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8']
        
        for ax_idx, (label, powers) in enumerate(band_powers_by_label.items()):
            # Compute means and stds
            means = []
            stds = []
            
            for bn in band_names:
                if powers[bn]:  # Check if we have data for this band
                    means.append(np.mean(powers[bn]))
                    stds.append(np.std(powers[bn]))
                else:
                    means.append(0)
                    stds.append(0)
            
            x_pos = np.arange(len(band_names))
            axes[ax_idx].bar(x_pos, means, yerr=stds, capsize=5, 
                            color=colors, alpha=0.7, edgecolor='black')
            axes[ax_idx].set_xticks(x_pos)
            axes[ax_idx].set_xticklabels([bn.capitalize() for bn in band_names], 
                                        rotation=45)
            axes[ax_idx].set_ylabel('Relative Power', fontsize=10)
            axes[ax_idx].set_title(f'Label: {label.upper()}', fontsize=11, 
                                  fontweight='bold')
            axes[ax_idx].grid(True, alpha=0.3, axis='y')
        
        fig.suptitle(f'Frequency Band Power Distribution | {self.dataset_name}', 
                    fontsize=13, fontweight='bold')
        
        plt.tight_layout()
        output_file = output_dir / "frequency_bands.png"
        plt.savefig(output_file, dpi=self.dpi, bbox_inches='tight', facecolor='white')
        plt.close()
    
    def _compute_band_powers(self, segment: np.ndarray, sfreq: float) -> Dict[str, float]:
        """
        Compute relative power in each frequency band.
        
        Args:
            segment: EEG data (n_channels, n_samples)
            sfreq: Sampling frequency
        
        Returns:
            Dictionary with band powers
        """
        bands = {
            'delta': (0.5, 4),
            'theta': (4, 8),
            'alpha': (8, 13),
            'beta': (13, 30),
            'gamma': (30, 50)
        }
        
        # Average across channels
        avg_signal = np.mean(segment, axis=0)
        
        # Compute PSD
        freqs, psd = signal.welch(avg_signal, fs=sfreq, nperseg=256, noverlap=128)
        
        # Compute band powers
        band_powers = {}
        total_power = np.sum(psd)
        
        for band_name, (low, high) in bands.items():
            # Find frequency indices
            idx_band = np.logical_and(freqs >= low, freqs <= high)
            
            # Compute power in band
            band_power = np.sum(psd[idx_band])
            
            # Relative power (normalized by total power)
            band_powers[band_name] = band_power / (total_power + 1e-10)
        
        return band_powers
    
    # ========================================================================
    # 4. SPATIAL DISTRIBUTION
    # ========================================================================
    
    def _plot_spatial_distribution(self, sample_data, output_dir):
        """Plot spatial correlation heatmaps."""
        for idx, data in enumerate(sample_data):
            segment = data['segment']
            metadata = data['metadata']
            
            corr_matrix = np.corrcoef(segment)
            
            fig, ax = plt.subplots(figsize=(10, 8))
            
            im = ax.imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1, 
                          aspect='auto', interpolation='nearest')
            
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label('Correlation Coefficient', rotation=270, labelpad=20, 
                          fontsize=10)
            
            n_channels = segment.shape[0]
            ax.set_xticks(np.arange(n_channels))
            ax.set_yticks(np.arange(n_channels))
            ax.set_xticklabels([f'Ch {i+1}' for i in range(n_channels)], 
                              rotation=45, ha='right')
            ax.set_yticklabels([f'Ch {i+1}' for i in range(n_channels)])
            
            ax.set_xticks(np.arange(n_channels) - 0.5, minor=True)
            ax.set_yticks(np.arange(n_channels) - 0.5, minor=True)
            ax.grid(which='minor', color='white', linestyle='-', linewidth=1)
            
            label = metadata.get('label', 'unknown')
            ax.set_title(
                f'Channel Correlation Matrix | {self.dataset_name} | Label: {label.upper()}',
                fontsize=12, fontweight='bold', pad=20
            )
            
            plt.tight_layout()
            output_file = output_dir / f"spatial_distribution_{idx:03d}.png"
            plt.savefig(output_file, dpi=self.dpi, bbox_inches='tight', facecolor='white')
            plt.close()
    
    # ========================================================================
    # 5. DATASET STATISTICS
    # ========================================================================
    
    def _plot_dataset_statistics(self, stats, output_dir):
        """Plot dataset-level statistics."""
        label_counts = stats['label_distribution']
        total_segments = stats['total_segments']
        
        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
        
        # 1. Label distribution (pie chart)
        ax1 = fig.add_subplot(gs[0, 0])
        if label_counts:
            labels_list = list(label_counts.keys())
            counts = list(label_counts.values())
            colors = plt.cm.Set3(np.linspace(0, 1, len(labels_list)))
            
            wedges, texts, autotexts = ax1.pie(counts, labels=labels_list, autopct='%1.1f%%',
                                               colors=colors, startangle=90)
            for autotext in autotexts:
                autotext.set_color('white')
                autotext.set_fontweight('bold')
            
            ax1.set_title('Label Distribution', fontsize=12, fontweight='bold')
        
        # 2. Label counts (bar chart)
        ax2 = fig.add_subplot(gs[0, 1])
        if label_counts:
            sorted_labels = sorted(label_counts.items(), key=lambda x: -x[1])
            labels_list = [l[0] for l in sorted_labels]
            counts = [l[1] for l in sorted_labels]
            
            bars = ax2.barh(labels_list, counts, color='#4CAF50', alpha=0.7, edgecolor='black')
            ax2.set_xlabel('Count', fontsize=10)
            ax2.set_title('Label Counts', fontsize=12, fontweight='bold')
            ax2.grid(True, alpha=0.3, axis='x')
            
            for i, bar in enumerate(bars):
                width = bar.get_width()
                ax2.text(width, bar.get_y() + bar.get_height()/2, 
                        f'{int(width):,}', ha='left', va='center', fontsize=9)
        
        # 3. Percentage breakdown
        ax3 = fig.add_subplot(gs[1, 0])
        ax3.axis('off')
        
        if label_counts:
            breakdown_text = f"""
LABEL BREAKDOWN
{'=' * 40}

Total Segments: {total_segments:,}

"""
            for label, count in sorted(label_counts.items(), key=lambda x: -x[1]):
                pct = (count / sum(label_counts.values())) * 100
                breakdown_text += f"{label:15s}: {count:7,d} ({pct:5.1f}%)\n"
            
            ax3.text(0.1, 0.5, breakdown_text, fontsize=10, family='monospace',
                    verticalalignment='center', bbox=dict(boxstyle='round', 
                    facecolor='wheat', alpha=0.3))
        
        # 4. Dataset summary
        ax4 = fig.add_subplot(gs[1, 1])
        ax4.axis('off')
        
        summary_text = f"""
DATASET SUMMARY
{'=' * 40}

Dataset: {self.dataset_name}
Total Segments: {total_segments:,}
Unique Labels: {len(label_counts)}

Most Common: {max(label_counts, key=label_counts.get) if label_counts else 'N/A'}
Least Common: {min(label_counts, key=label_counts.get) if label_counts else 'N/A'}

Sample Size: {stats['sample_size']:,}
(Extrapolated to full dataset)
        """
        
        ax4.text(0.1, 0.5, summary_text, fontsize=10, family='monospace',
                verticalalignment='center', bbox=dict(boxstyle='round', 
                facecolor='lightblue', alpha=0.3))
        
        fig.suptitle(f'{self.dataset_name} Dataset Statistics', 
                    fontsize=14, fontweight='bold')
        
        output_file = output_dir / "dataset_statistics.png"
        plt.savefig(output_file, dpi=self.dpi, bbox_inches='tight', facecolor='white')
        plt.close()
    
    # ========================================================================
    # 6. QUALITY METRICS
    # ========================================================================
    
    def _plot_quality_metrics(self, stats, output_dir):
        """Plot quality control metrics."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # 1. Label distribution (pie chart)
        label_counts = stats['label_distribution']
        
        if label_counts:
            labels = list(label_counts.keys())
            counts = list(label_counts.values())
            colors = plt.cm.Set3(np.linspace(0, 1, len(labels)))
            
            wedges, texts, autotexts = axes[0].pie(counts, labels=labels, 
                                                   autopct='%1.1f%%', colors=colors,
                                                   startangle=90)
            for autotext in autotexts:
                autotext.set_color('white')
                autotext.set_fontweight('bold')
            
            axes[0].set_title('Label Distribution', fontsize=12, fontweight='bold')
        
        # 2. Summary text
        axes[1].axis('off')
        
        total = sum(label_counts.values())
        
        summary_text = f"""
QUALITY SUMMARY
{'=' * 40}

Total Segments: {total:,}

LABEL DISTRIBUTION:
{'-' * 40}
"""
        
        for label, count in sorted(label_counts.items(), key=lambda x: -x[1]):
            pct = (count / total * 100) if total > 0 else 0
            summary_text += f"{label:20s}: {count:7,d} ({pct:5.1f}%)\n"
        
        summary_text += f"\n{'=' * 40}\n"
        summary_text += f"\nDataset: {self.dataset_name}\n"
        summary_text += f"Sample Size: {stats['sample_size']:,}\n"
        
        axes[1].text(0.1, 0.5, summary_text, fontsize=9, family='monospace',
                    verticalalignment='center', bbox=dict(boxstyle='round',
                    facecolor='wheat', alpha=0.3))
        
        fig.suptitle(f'{self.dataset_name} Quality Metrics', 
                    fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        output_file = output_dir / "quality_metrics.png"
        plt.savefig(output_file, dpi=self.dpi, bbox_inches='tight', facecolor='white')
        plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='Generate visualizations from preprocessed EEG datasets (CORRECTED)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Visualize TUAB
  python standalone_visualize_corrected.py --dataset TUAB

  # Visualize TUAR with 10 samples
  python standalone_visualize_corrected.py --dataset TUAR --max-samples 10

  # Visualize all datasets
  python standalone_visualize_corrected.py --dataset all

  # Custom paths
  python standalone_visualize_corrected.py --dataset TUAB --preprocessed-dir custom/path --output-dir custom/output
        """
    )
    
    parser.add_argument('--dataset', type=str, required=True,
                       help='Dataset to visualize (TUAB, TUAR, TUEP, TUEV, TUSL, TUSZ, or "all")')
    parser.add_argument('--preprocessed-dir', type=str, default=None,
                       help='Preprocessed directory (default: neurovault_data/neurovault_DATASET/preprocessed)')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Output directory (default: neurovault_data/neurovault_DATASET/visualizations)')
    parser.add_argument('--max-samples', type=int, default=5,
                       help='Number of sample segments to visualize (default: 5)')
    parser.add_argument('--dpi', type=int, default=300,
                       help='Resolution for saved figures (default: 300)')
    
    args = parser.parse_args()
    
    # Determine datasets to process
    if args.dataset.lower() == 'all':
        datasets = ['TUAB', 'TUAR', 'TUEP', 'TUEV', 'TUSL', 'TUSZ']
    else:
        datasets = [args.dataset.upper()]
    
    # Process each dataset
    for dataset in datasets:
        # Determine paths
        if args.preprocessed_dir:
            preprocessed_dir = Path(args.preprocessed_dir)
        else:
            preprocessed_dir = Path(f'neurovault_data/neurovault_{dataset.lower()}/preprocessed')
        
        if args.output_dir:
            output_dir = Path(args.output_dir)
        else:
            output_dir = Path(f'neurovault_data/neurovault_{dataset.lower()}/visualizations')
        
        # Check if preprocessed directory exists
        if not preprocessed_dir.exists():
            print(f"\n⏭️  Skipping {dataset}: Directory not found: {preprocessed_dir}")
            continue
        
        # Generate visualizations
        visualizer = StandaloneEEGVisualizer(
            dataset_name=dataset,
            max_samples=args.max_samples,
            dpi=args.dpi
        )
        
        visualizer.generate_all(preprocessed_dir, output_dir)


if __name__ == "__main__":
    main()
