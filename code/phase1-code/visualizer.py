#!/usr/bin/env python3
"""
NeuroVault Visualization Module
================================

Generates quality control plots and statistics visualizations.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import pickle
from scipy import signal
import mne


class NeuroVaultVisualizer:
    """Generate visualizations for preprocessing quality control."""
    
    def __init__(self, preprocessed_dir, viz_dir, stats_dir, config):
        """
        Initialize visualizer.
        
        Args:
            preprocessed_dir: Directory with preprocessed segments
            viz_dir: Output directory for plots
            stats_dir: Directory with statistics JSON
            config: Configuration dict
        """
        self.preprocessed_dir = Path(preprocessed_dir)
        self.viz_dir = Path(viz_dir)
        self.stats_dir = Path(stats_dir)
        self.config = config
        
        # Set style
        sns.set_style('whitegrid')
        plt.rcParams['figure.dpi'] = config['visualization']['dpi']
        plt.rcParams['figure.figsize'] = config['visualization']['figure_size']
    
    def load_sample_segments(self, n_samples=50):
        """Load random sample of preprocessed segments."""
        segment_files = sorted(list(self.preprocessed_dir.glob('*.pkl')))
        
        if len(segment_files) == 0:
            raise ValueError(f"No preprocessed segments found in {self.preprocessed_dir}")
        
        # Random sample
        n_samples = min(n_samples, len(segment_files))
        sample_indices = np.random.choice(len(segment_files), n_samples, replace=False)
        
        samples = []
        for idx in sample_indices:
            with open(segment_files[idx], 'rb') as f:
                samples.append(pickle.load(f))
        
        return samples
    
    def plot_psd_analysis(self, samples):
        """
        Plot power spectral density analysis.
        
        Shows:
        - Mean PSD across all samples
        - PSD variability (std bands)
        - Frequency band boundaries
        """
        fig, axes = plt.subplots(2, 1, figsize=(14, 10))
        
        # Collect PSDs
        psds = []
        for sample in samples:
            if 'psd_mean' in sample['features']:
                psds.append(sample['features']['psd_mean'])
        
        if len(psds) == 0:
            print("Warning: No PSD features found in samples")
            plt.close(fig)
            return
        
        psds = np.array(psds)
        
        # Compute frequency axis (from config)
        sfreq = self.config['preprocessing']['target_sfreq']
        nperseg = self.config['feature_extraction']['psd_params']['nperseg']
        freqs = np.fft.rfftfreq(nperseg, 1/sfreq)[:psds.shape[1]]
        
        # Plot 1: Mean PSD with std bands
        ax = axes[0]
        mean_psd = np.mean(psds, axis=0)
        std_psd = np.std(psds, axis=0)
        
        ax.plot(freqs, 10 * np.log10(mean_psd), 'b-', linewidth=2, label='Mean PSD')
        ax.fill_between(freqs, 
                        10 * np.log10(mean_psd - std_psd),
                        10 * np.log10(mean_psd + std_psd),
                        alpha=0.3, label='±1 STD')
        
        # Mark frequency bands
        bands = self.config['feature_extraction']['frequency_bands']
        colors = ['red', 'orange', 'green', 'blue', 'purple']
        for (band_name, (low, high)), color in zip(bands.items(), colors):
            ax.axvspan(low, high, alpha=0.2, color=color, label=f'{band_name} ({low}-{high} Hz)')
        
        ax.set_xlabel('Frequency (Hz)', fontsize=12)
        ax.set_ylabel('Power Spectral Density (dB)', fontsize=12)
        ax.set_title('Mean PSD Across All Segments', fontsize=14, fontweight='bold')
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, 50])
        
        # Plot 2: Band power distribution
        ax = axes[1]
        
        band_powers_all = {band: [] for band in bands.keys()}
        for sample in samples:
            if 'band_powers' in sample['features']:
                for band_name in bands.keys():
                    if band_name in sample['features']['band_powers']:
                        band_powers_all[band_name].append(
                            np.mean(sample['features']['band_powers'][band_name])
                        )
        
        # Box plot
        data_to_plot = [band_powers_all[band] for band in bands.keys() if len(band_powers_all[band]) > 0]
        labels = [band for band in bands.keys() if len(band_powers_all[band]) > 0]
        
        bp = ax.boxplot(data_to_plot, labels=labels, patch_artist=True)
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
        
        ax.set_ylabel('Band Power (µV²)', fontsize=12)
        ax.set_title('Frequency Band Power Distribution', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        output_path = self.viz_dir / 'psd_analysis.png'
        plt.savefig(output_path, dpi=self.config['visualization']['dpi'], bbox_inches='tight')
        plt.close()
        
        print(f"Saved PSD analysis to {output_path}")
    
    def plot_quality_metrics(self):
        """
        Plot quality control statistics.
        
        Shows:
        - Rejection rate pie chart
        - Rejection reasons breakdown
        - Processing success rate
        """
        # Load statistics
        stats_file = self.stats_dir / 'preprocessing_stats.json'
        if not stats_file.exists():
            print(f"Warning: Statistics file not found at {stats_file}")
            return
        
        with open(stats_file, 'r') as f:
            stats = json.load(f)
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # Plot 1: File processing success
        ax = axes[0]
        processed = stats['processed_files']
        failed = stats['failed_files']
        
        sizes = [processed, failed]
        labels = [f'Processed\n({processed})', f'Failed\n({failed})']
        colors = ['#2ecc71', '#e74c3c']
        explode = (0.05, 0)
        
        ax.pie(sizes, explode=explode, labels=labels, colors=colors,
               autopct='%1.1f%%', shadow=True, startangle=90)
        ax.set_title('File Processing Success Rate', fontsize=14, fontweight='bold')
        
        # Plot 2: Segment acceptance
        ax = axes[1]
        accepted = stats['total_segments'] - stats['rejected_segments']
        rejected = stats['rejected_segments']
        
        sizes = [accepted, rejected]
        labels = [f'Accepted\n({accepted})', f'Rejected\n({rejected})']
        colors = ['#3498db', '#e67e22']
        explode = (0.05, 0)
        
        ax.pie(sizes, explode=explode, labels=labels, colors=colors,
               autopct='%1.1f%%', shadow=True, startangle=90)
        ax.set_title('Segment Quality Control', fontsize=14, fontweight='bold')
        
        # Plot 3: Rejection reasons breakdown
        ax = axes[2]
        reasons = stats['rejection_reasons']
        
        # Filter out zero counts
        reasons_filtered = {k: v for k, v in reasons.items() if v > 0}
        
        if len(reasons_filtered) > 0:
            labels = list(reasons_filtered.keys())
            sizes = list(reasons_filtered.values())
            colors_palette = sns.color_palette('Set2', len(labels))
            
            ax.pie(sizes, labels=labels, colors=colors_palette,
                   autopct='%1.1f%%', shadow=True, startangle=90)
            ax.set_title('Rejection Reasons', fontsize=14, fontweight='bold')
        else:
            ax.text(0.5, 0.5, 'No rejections', ha='center', va='center',
                   fontsize=16, transform=ax.transAxes)
            ax.axis('off')
        
        plt.tight_layout()
        output_path = self.viz_dir / 'quality_metrics.png'
        plt.savefig(output_path, dpi=self.config['visualization']['dpi'], bbox_inches='tight')
        plt.close()
        
        print(f"Saved quality metrics to {output_path}")
    
    def plot_raw_vs_preprocessed(self, raw_file_path, processed_sample):
        """
        Plot comparison of raw vs preprocessed EEG.
        
        Args:
            raw_file_path: Path to original EDF file
            processed_sample: Dict with preprocessed segment
        """
        try:
            # Load raw EEG
            raw = mne.io.read_raw_edf(raw_file_path, preload=True, verbose=False)
            
            # Get first 10 seconds
            duration = 10.0
            raw_data = raw.get_data(stop=int(duration * raw.info['sfreq']))
            
            # Get preprocessed segment
            preprocessed_data = processed_sample['segment']
            
            # Create time axes
            time_raw = np.arange(raw_data.shape[1]) / raw.info['sfreq']
            time_prep = np.arange(preprocessed_data.shape[1]) / processed_sample['metadata']['sfreq']
            
            # Plot
            fig, axes = plt.subplots(2, 1, figsize=(16, 10), sharex=True)
            
            # Plot raw (first 4 channels)
            ax = axes[0]
            n_channels_to_plot = min(4, raw_data.shape[0])
            for i in range(n_channels_to_plot):
                ax.plot(time_raw, raw_data[i] + i * 500, linewidth=0.5, label=f'Ch{i+1}')
            
            ax.set_ylabel('Amplitude (µV)', fontsize=12)
            ax.set_title('Raw EEG (Before Preprocessing)', fontsize=14, fontweight='bold')
            ax.legend(loc='upper right')
            ax.grid(True, alpha=0.3)
            
            # Plot preprocessed (first 4 channels)
            ax = axes[1]
            n_channels_to_plot = min(4, preprocessed_data.shape[0])
            for i in range(n_channels_to_plot):
                ax.plot(time_prep, preprocessed_data[i] + i * 5, linewidth=0.5, label=f'Ch{i+1}')
            
            ax.set_xlabel('Time (seconds)', fontsize=12)
            ax.set_ylabel('Normalized Amplitude', fontsize=12)
            ax.set_title('Preprocessed EEG (After Filtering & Normalization)', fontsize=14, fontweight='bold')
            ax.legend(loc='upper right')
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            output_path = self.viz_dir / 'raw_vs_preprocessed_example.png'
            plt.savefig(output_path, dpi=self.config['visualization']['dpi'], bbox_inches='tight')
            plt.close()
            
            print(f"Saved raw vs preprocessed comparison to {output_path}")
        
        except Exception as e:
            print(f"Warning: Could not generate raw vs preprocessed plot: {e}")
    
    def plot_channel_montage(self, sample):
        """
        Plot spatial distribution of signal energy across channels.
        
        Args:
            sample: Preprocessed sample dict
        """
        segment = sample['segment']
        
        # Compute channel powers
        channel_powers = np.var(segment, axis=1)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Bar plot
        channels = [f'Ch{i+1}' for i in range(len(channel_powers))]
        colors = plt.cm.viridis(channel_powers / channel_powers.max())
        
        bars = ax.barh(channels, channel_powers, color=colors)
        
        ax.set_xlabel('Channel Power (Variance)', fontsize=12)
        ax.set_ylabel('Channel', fontsize=12)
        ax.set_title('Spatial Energy Distribution Across Channels', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
        
        # Add colorbar
        sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, 
                                   norm=plt.Normalize(vmin=channel_powers.min(), 
                                                     vmax=channel_powers.max()))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax)
        cbar.set_label('Normalized Power', fontsize=10)
        
        plt.tight_layout()
        output_path = self.viz_dir / 'channel_montage_example.png'
        plt.savefig(output_path, dpi=self.config['visualization']['dpi'], bbox_inches='tight')
        plt.close()
        
        print(f"Saved channel montage to {output_path}")
    
    def plot_segment_distribution(self):
        """Plot distribution of segment statistics."""
        # Load statistics
        stats_file = self.stats_dir / 'preprocessing_stats.json'
        if not stats_file.exists():
            return
        
        with open(stats_file, 'r') as f:
            stats = json.load(f)
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Plot 1: Processing summary
        ax = axes[0]
        categories = ['Total Files', 'Processed', 'Failed', 'Total Segments', 'Valid Segments', 'Rejected']
        values = [
            stats['total_files'],
            stats['processed_files'],
            stats['failed_files'],
            stats['total_segments'],
            stats['total_segments'] - stats['rejected_segments'],
            stats['rejected_segments']
        ]
        
        colors = ['#3498db', '#2ecc71', '#e74c3c', '#9b59b6', '#f39c12', '#e67e22']
        bars = ax.bar(categories, values, color=colors, alpha=0.7, edgecolor='black')
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(height)}',
                   ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        ax.set_ylabel('Count', fontsize=12)
        ax.set_title('Processing Summary', fontsize=14, fontweight='bold')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Plot 2: Segment yield per file
        ax = axes[1]
        
        if stats['processed_files'] > 0:
            avg_segments = (stats['total_segments'] - stats['rejected_segments']) / stats['processed_files']
            acceptance_rate = (stats['total_segments'] - stats['rejected_segments']) / max(stats['total_segments'], 1) * 100
            
            metrics = ['Avg Segments\nper File', 'Acceptance\nRate (%)']
            values = [avg_segments, acceptance_rate]
            colors = ['#16a085', '#27ae60']
            
            bars = ax.bar(metrics, values, color=colors, alpha=0.7, edgecolor='black', width=0.6)
            
            # Add value labels
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.1f}',
                       ha='center', va='bottom', fontsize=12, fontweight='bold')
            
            ax.set_ylabel('Value', fontsize=12)
            ax.set_title('Segment Yield Metrics', fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        output_path = self.viz_dir / 'segment_distribution.png'
        plt.savefig(output_path, dpi=self.config['visualization']['dpi'], bbox_inches='tight')
        plt.close()
        
        print(f"Saved segment distribution to {output_path}")
    
    def generate_all_plots(self, raw_edf_file=None):
        """
        Generate all visualization plots.
        
        Args:
            raw_edf_file: Optional path to raw EDF for before/after comparison
        """
        print("\n" + "="*60)
        print("Generating Visualizations")
        print("="*60)
        
        # Load samples
        max_samples = self.config['visualization']['max_samples_to_plot']
        samples = self.load_sample_segments(n_samples=max_samples)
        print(f"Loaded {len(samples)} sample segments")
        
        # Generate plots
        plot_types = self.config['visualization']['plot_types']
        
        if 'psd_analysis' in plot_types:
            self.plot_psd_analysis(samples)
        
        if 'quality_metrics' in plot_types:
            self.plot_quality_metrics()
        
        if 'segment_distribution' in plot_types:
            self.plot_segment_distribution()
        
        if 'channel_montage' in plot_types and len(samples) > 0:
            self.plot_channel_montage(samples[0])
        
        if 'raw_vs_preprocessed' in plot_types and raw_edf_file is not None:
            if len(samples) > 0:
                self.plot_raw_vs_preprocessed(raw_edf_file, samples[0])
        
        print("="*60)
        print("Visualization Complete!")
        print(f"Plots saved to: {self.viz_dir}")
        print("="*60 + "\n")


def main():
    """Standalone visualization script."""
    import argparse
    import yaml
    
    parser = argparse.ArgumentParser(description='Generate NeuroVault visualizations')
    parser.add_argument('--preprocessed-dir', type=str, required=True,
                       help='Directory with preprocessed segments')
    parser.add_argument('--viz-dir', type=str, required=True,
                       help='Output directory for plots')
    parser.add_argument('--stats-dir', type=str, required=True,
                       help='Directory with statistics JSON')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to config YAML')
    parser.add_argument('--raw-edf', type=str, default=None,
                       help='Optional raw EDF file for before/after comparison')
    
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Create visualizer
    visualizer = NeuroVaultVisualizer(
        preprocessed_dir=args.preprocessed_dir,
        viz_dir=args.viz_dir,
        stats_dir=args.stats_dir,
        config=config
    )
    
    # Generate plots
    visualizer.generate_all_plots(raw_edf_file=args.raw_edf)


if __name__ == '__main__':
    main()
