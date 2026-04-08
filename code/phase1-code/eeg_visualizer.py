#!/usr/bin/env python3
"""
TUH EEG Visualization Module
============================

Automatic visualization generation after preprocessing.

Plot types:
- raw_eeg_traces: Time-series plots of EEG channels
- psd_analysis: Power spectral density plots
- frequency_bands: Band power visualization
- spatial_distribution: Channel correlation heatmaps
- dataset_statistics: Label distribution, segment counts
- preprocessing_pipeline: Pipeline steps visualization
- quality_metrics: QC statistics and rejection reasons

Author: Haitham
Date: March 17, 2026
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-GUI backend
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import signal
import pickle
import json
from typing import List, Dict
import logging

logger = logging.getLogger(__name__)

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


class EEGVisualizer:
    """Generate comprehensive visualizations for preprocessed EEG data."""
    
    def __init__(self, config, dataset_name):
        """
        Initialize visualizer.
        
        Args:
            config: Configuration dictionary
            dataset_name: Name of dataset (TUAB, TUAR, etc.)
        """
        self.config = config
        self.dataset_name = dataset_name
        self.viz_config = config.get('visualization', {})
        self.plot_types = self.viz_config.get('plot_types', ['raw_eeg_traces'])
        self.max_samples = self.viz_config.get('max_samples_to_plot', 5)
        self.dpi = self.viz_config.get('dpi', 300)
    
    def generate_all_visualizations(self, preprocessed_dir: Path, 
                                   output_dir: Path, stats: Dict):
        """
        Generate all requested visualizations.
        
        Args:
            preprocessed_dir: Directory with preprocessed segments
            output_dir: Directory to save visualizations
            stats: Preprocessing statistics dictionary
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Generating visualizations in {output_dir}")
        logger.info(f"Plot types: {', '.join(self.plot_types)}")
        
        # Load sample segments
        segment_files = sorted(list(preprocessed_dir.glob("*.pkl")))
        
        if not segment_files:
            logger.warning("No segments found for visualization")
            return
        
        # Sample segments for visualization
        sample_files = segment_files[:self.max_samples]
        sample_data = [self._load_segment(f) for f in sample_files]
        
        # Generate each plot type
        for plot_type in self.plot_types:
            try:
                if plot_type == 'raw_eeg_traces':
                    self._plot_raw_eeg_traces(sample_data, output_dir)
                
                elif plot_type == 'psd_analysis':
                    self._plot_psd_analysis(sample_data, output_dir)
                
                elif plot_type == 'frequency_bands':
                    self._plot_frequency_bands(sample_data, output_dir)
                
                elif plot_type == 'spatial_distribution':
                    self._plot_spatial_distribution(sample_data, output_dir)
                
                elif plot_type == 'dataset_statistics':
                    self._plot_dataset_statistics(preprocessed_dir, stats, output_dir)
                
                elif plot_type == 'preprocessing_pipeline':
                    self._plot_preprocessing_pipeline(output_dir)
                
                elif plot_type == 'quality_metrics':
                    self._plot_quality_metrics(stats, output_dir)
                
                logger.info(f"✓ Generated: {plot_type}")
                
            except Exception as e:
                logger.error(f"Failed to generate {plot_type}: {e}")
        
        logger.info(f"Visualizations saved to: {output_dir}")
    
    def _load_segment(self, segment_file: Path) -> Dict:
        """Load a preprocessed segment."""
        with open(segment_file, 'rb') as f:
            return pickle.load(f)
    
    # ========================================================================
    # 1. RAW EEG TRACES
    # ========================================================================
    
    def _plot_raw_eeg_traces(self, sample_data: List[Dict], output_dir: Path):
        """Plot raw EEG time series."""
        for idx, data in enumerate(sample_data):
            segment = data['segment']
            metadata = data['metadata']
            
            n_channels, n_samples = segment.shape
            sfreq = metadata['sfreq']
            time = np.arange(n_samples) / sfreq
            
            # Create figure
            fig, axes = plt.subplots(n_channels, 1, figsize=(15, n_channels * 1.2), 
                                    sharex=True)
            
            if n_channels == 1:
                axes = [axes]
            
            # Plot each channel
            for ch_idx in range(n_channels):
                axes[ch_idx].plot(time, segment[ch_idx], linewidth=0.6, 
                                color='#2E86AB', alpha=0.8)
                axes[ch_idx].set_ylabel(f'Ch {ch_idx+1}\n(µV)', fontsize=9, 
                                       rotation=0, ha='right', va='center')
                axes[ch_idx].grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
                axes[ch_idx].set_xlim([0, time[-1]])
                
                if ch_idx < n_channels - 1:
                    axes[ch_idx].set_xticklabels([])
            
            axes[-1].set_xlabel('Time (seconds)', fontsize=11, fontweight='bold')
            
            # Title
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
    
    def _plot_psd_analysis(self, sample_data: List[Dict], output_dir: Path):
        """Plot power spectral density analysis."""
        for idx, data in enumerate(sample_data):
            segment = data['segment']
            metadata = data['metadata']
            
            n_channels = segment.shape[0]
            sfreq = metadata['sfreq']
            
            # Create figure
            fig, axes = plt.subplots(n_channels, 1, figsize=(12, n_channels * 1.5), 
                                    sharex=True)
            
            if n_channels == 1:
                axes = [axes]
            
            # Frequency bands
            bands = {
                'Delta': (0.5, 4),
                'Theta': (4, 8),
                'Alpha': (8, 13),
                'Beta': (13, 30),
                'Gamma': (30, 50)
            }
            band_colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8']
            
            for ch_idx in range(n_channels):
                # Compute PSD
                freqs, psd = signal.welch(segment[ch_idx], fs=sfreq, 
                                        nperseg=256, noverlap=128)
                
                # Plot PSD
                axes[ch_idx].semilogy(freqs, psd, linewidth=1.5, 
                                     color='#2E86AB', alpha=0.8)
                axes[ch_idx].set_ylabel(f'Ch {ch_idx+1}\nPSD', fontsize=9, 
                                       rotation=0, ha='right', va='center')
                axes[ch_idx].set_xlim([0, 50])
                axes[ch_idx].grid(True, alpha=0.3, which='both')
                
                # Shade frequency bands
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
    # 3. FREQUENCY BANDS
    # ========================================================================
    
    def _plot_frequency_bands(self, sample_data: List[Dict], output_dir: Path):
        """Plot frequency band power distribution."""
        # Aggregate band powers across samples
        band_names = ['delta', 'theta', 'alpha', 'beta', 'gamma']
        band_powers_by_label = {}
        
        for data in sample_data:
            label = data['metadata'].get('label', 'unknown')
            features = data['features']
            band_powers = features['band_powers']
            
            if label not in band_powers_by_label:
                band_powers_by_label[label] = {bn: [] for bn in band_names}
            
            for band_name in band_names:
                band_powers_by_label[label][band_name].append(band_powers[band_name])
        
        # Create figure
        fig, axes = plt.subplots(1, len(band_powers_by_label), 
                                figsize=(5 * len(band_powers_by_label), 6))
        
        if len(band_powers_by_label) == 1:
            axes = [axes]
        
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8']
        
        for ax_idx, (label, powers) in enumerate(band_powers_by_label.items()):
            # Compute means
            means = [np.mean(powers[bn]) for bn in band_names]
            stds = [np.std(powers[bn]) for bn in band_names]
            
            # Bar plot
            x_pos = np.arange(len(band_names))
            axes[ax_idx].bar(x_pos, means, yerr=stds, capsize=5, 
                            color=colors, alpha=0.7, edgecolor='black')
            axes[ax_idx].set_xticks(x_pos)
            axes[ax_idx].set_xticklabels([bn.capitalize() for bn in band_names], 
                                        rotation=45)
            axes[ax_idx].set_ylabel('Power (µV²/Hz)', fontsize=10)
            axes[ax_idx].set_title(f'Label: {label.upper()}', fontsize=11, 
                                  fontweight='bold')
            axes[ax_idx].grid(True, alpha=0.3, axis='y')
        
        fig.suptitle(f'Frequency Band Power Distribution | {self.dataset_name}', 
                    fontsize=13, fontweight='bold')
        
        plt.tight_layout()
        output_file = output_dir / "frequency_bands.png"
        plt.savefig(output_file, dpi=self.dpi, bbox_inches='tight', facecolor='white')
        plt.close()
    
    # ========================================================================
    # 4. SPATIAL DISTRIBUTION
    # ========================================================================
    
    def _plot_spatial_distribution(self, sample_data: List[Dict], output_dir: Path):
        """Plot spatial correlation heatmaps."""
        for idx, data in enumerate(sample_data):
            segment = data['segment']
            metadata = data['metadata']
            
            # Compute correlation matrix
            corr_matrix = np.corrcoef(segment)
            
            # Create figure
            fig, ax = plt.subplots(figsize=(10, 8))
            
            # Heatmap
            im = ax.imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1, 
                          aspect='auto', interpolation='nearest')
            
            # Colorbar
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label('Correlation Coefficient', rotation=270, labelpad=20, 
                          fontsize=10)
            
            # Labels
            n_channels = segment.shape[0]
            ax.set_xticks(np.arange(n_channels))
            ax.set_yticks(np.arange(n_channels))
            ax.set_xticklabels([f'Ch {i+1}' for i in range(n_channels)], 
                              rotation=45, ha='right')
            ax.set_yticklabels([f'Ch {i+1}' for i in range(n_channels)])
            
            # Grid
            ax.set_xticks(np.arange(n_channels) - 0.5, minor=True)
            ax.set_yticks(np.arange(n_channels) - 0.5, minor=True)
            ax.grid(which='minor', color='white', linestyle='-', linewidth=1)
            
            # Title
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
    
    def _plot_dataset_statistics(self, preprocessed_dir: Path, stats: Dict, 
                                 output_dir: Path):
        """Plot dataset-level statistics."""
        # Load all segments to get label distribution
        segment_files = list(preprocessed_dir.glob("*.pkl"))
        labels = []
        
        # Sample up to 1000 segments for label distribution
        sample_files = segment_files[:min(1000, len(segment_files))]
        
        for seg_file in sample_files:
            try:
                with open(seg_file, 'rb') as f:
                    data = pickle.load(f)
                labels.append(data['metadata'].get('label', 'unknown'))
            except:
                pass
        
        # Count labels
        from collections import Counter
        label_counts = Counter(labels)
        
        # Create figure with subplots
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
        
        # 2. Processing statistics (bar chart)
        ax2 = fig.add_subplot(gs[0, 1])
        stat_names = ['Processed', 'Failed', 'Valid Segs', 'Rejected']
        stat_values = [
            stats['processed_files'],
            stats['failed_files'],
            stats['total_segments'],
            stats['rejected_segments']
        ]
        colors_bar = ['#4CAF50', '#F44336', '#2196F3', '#FF9800']
        
        bars = ax2.bar(stat_names, stat_values, color=colors_bar, alpha=0.7, 
                      edgecolor='black')
        ax2.set_ylabel('Count', fontsize=10)
        ax2.set_title('Processing Statistics', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height):,}', ha='center', va='bottom', fontsize=9)
        
        # 3. Segment yield
        ax3 = fig.add_subplot(gs[1, 0])
        total_attempted = stats['total_segments'] + stats['rejected_segments']
        if total_attempted > 0:
            yield_pct = (stats['total_segments'] / total_attempted) * 100
            reject_pct = 100 - yield_pct
            
            ax3.pie([yield_pct, reject_pct], labels=['Valid', 'Rejected'],
                   autopct='%1.1f%%', colors=['#4CAF50', '#F44336'],
                   startangle=90)
            ax3.set_title(f'Segment Yield: {yield_pct:.2f}%', 
                         fontsize=12, fontweight='bold')
        
        # 4. Summary text
        ax4 = fig.add_subplot(gs[1, 1])
        ax4.axis('off')
        
        summary_text = f"""
DATASET SUMMARY
{'=' * 40}

Dataset: {self.dataset_name}
Total Files: {stats['total_files']:,}
Processed Files: {stats['processed_files']:,}
Failed Files: {stats['failed_files']:,}

Valid Segments: {stats['total_segments']:,}
Rejected Segments: {stats['rejected_segments']:,}

Unique Labels: {len(label_counts)}
Most Common: {label_counts.most_common(1)[0][0] if label_counts else 'N/A'}

Channels per Segment: {self.config['preprocessing']['n_channels']}
Segment Duration: {self.config['preprocessing']['segment_duration']}s
Sampling Rate: {self.config['preprocessing']['target_sfreq']} Hz
        """
        
        ax4.text(0.1, 0.5, summary_text, fontsize=10, family='monospace',
                verticalalignment='center', bbox=dict(boxstyle='round', 
                facecolor='wheat', alpha=0.3))
        
        fig.suptitle(f'{self.dataset_name} Dataset Statistics', 
                    fontsize=14, fontweight='bold')
        
        output_file = output_dir / "dataset_statistics.png"
        plt.savefig(output_file, dpi=self.dpi, bbox_inches='tight', facecolor='white')
        plt.close()
    
    # ========================================================================
    # 6. PREPROCESSING PIPELINE
    # ========================================================================
    
    def _plot_preprocessing_pipeline(self, output_dir: Path):
        """Visualize preprocessing pipeline steps."""
        fig, ax = plt.subplots(figsize=(14, 10))
        ax.axis('off')
        
        # Pipeline steps
        steps = [
            ("1. Load EDF", "Read raw EEG data\nAuto-detect units (V/mV/µV)\nScale to µV"),
            ("2. Channel Selection", f"Target: {self.config['preprocessing']['n_channels']} TCP channels\n"
                                    f"Handle 15-22 available\nFirst-electrode selection"),
            ("3. Resample", f"Target: {self.config['preprocessing']['target_sfreq']} Hz\n"
                           f"Method: Polyphase FIR"),
            ("4. Bandpass Filter", f"Range: {self.config['preprocessing']['bandpass_filter']['low']}-"
                                  f"{self.config['preprocessing']['bandpass_filter']['high']} Hz\n"
                                  f"Method: {self.config['preprocessing']['bandpass_filter']['method']}\n"
                                  f"Phase: {self.config['preprocessing']['bandpass_filter']['phase']}"),
            ("5. Notch Filter", f"Frequency: {self.config['preprocessing']['notch_filter']['frequency']} Hz\n"
                               f"Remove powerline noise"),
            ("6. Segmentation", f"Duration: {self.config['preprocessing']['segment_duration']}s\n"
                               f"Overlap: 0s (non-overlapping)"),
            ("7. Normalization", f"Method: Global z-score\n"
                                f"Epsilon: {self.config['preprocessing']['normalization']['epsilon']}"),
            ("8. Quality Control", f"Max amplitude: {self.config['preprocessing']['quality_control']['max_amplitude_threshold']} µV\n"
                                  f"Min variance: {self.config['preprocessing']['quality_control']['min_median_std']}\n"
                                  f"Flat channel ratio: {self.config['preprocessing']['quality_control']['max_flat_channels_ratio']}")
        ]
        
        # Draw pipeline
        n_steps = len(steps)
        y_positions = np.linspace(0.9, 0.1, n_steps)
        
        for i, ((title, description), y_pos) in enumerate(zip(steps, y_positions)):
            # Draw box
            box = plt.Rectangle((0.1, y_pos - 0.04), 0.8, 0.08, 
                               facecolor='lightblue', edgecolor='black', 
                               linewidth=2, alpha=0.7)
            ax.add_patch(box)
            
            # Add text
            ax.text(0.15, y_pos + 0.01, title, fontsize=11, fontweight='bold',
                   verticalalignment='center')
            ax.text(0.15, y_pos - 0.02, description, fontsize=8,
                   verticalalignment='center')
            
            # Draw arrow
            if i < n_steps - 1:
                ax.arrow(0.5, y_pos - 0.04, 0, -0.04, head_width=0.03, 
                        head_length=0.01, fc='black', ec='black')
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_title(f'{self.dataset_name} Preprocessing Pipeline', 
                    fontsize=14, fontweight='bold', pad=20)
        
        output_file = output_dir / "preprocessing_pipeline.png"
        plt.savefig(output_file, dpi=self.dpi, bbox_inches='tight', facecolor='white')
        plt.close()
    
    # ========================================================================
    # 7. QUALITY METRICS
    # ========================================================================
    
    def _plot_quality_metrics(self, stats: Dict, output_dir: Path):
        """Plot quality control metrics."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # 1. Rejection reasons (pie chart)
        rejection_reasons = stats['rejection_reasons']
        reasons_with_counts = {k: v for k, v in rejection_reasons.items() if v > 0}
        
        if reasons_with_counts:
            labels = list(reasons_with_counts.keys())
            counts = list(reasons_with_counts.values())
            colors = plt.cm.Set3(np.linspace(0, 1, len(labels)))
            
            wedges, texts, autotexts = axes[0].pie(counts, labels=labels, 
                                                   autopct='%1.1f%%', colors=colors,
                                                   startangle=90)
            for autotext in autotexts:
                autotext.set_color('white')
                autotext.set_fontweight('bold')
            
            axes[0].set_title('Rejection Reasons', fontsize=12, fontweight='bold')
        else:
            axes[0].text(0.5, 0.5, 'No Rejections', ha='center', va='center',
                        fontsize=14, transform=axes[0].transAxes)
            axes[0].set_title('Rejection Reasons', fontsize=12, fontweight='bold')
        
        # 2. Quality metrics summary
        axes[1].axis('off')
        
        total_attempted = stats['total_segments'] + stats['rejected_segments']
        yield_pct = (stats['total_segments'] / total_attempted * 100) if total_attempted > 0 else 0
        
        qc_text = f"""
QUALITY CONTROL SUMMARY
{'=' * 40}

Total Segments Attempted: {total_attempted:,}
Valid Segments: {stats['total_segments']:,}
Rejected Segments: {stats['rejected_segments']:,}

Segment Yield: {yield_pct:.2f}%

REJECTION BREAKDOWN:
{'-' * 40}
"""
        
        for reason, count in rejection_reasons.items():
            if count > 0:
                pct = (count / stats['rejected_segments'] * 100) if stats['rejected_segments'] > 0 else 0
                qc_text += f"{reason:20s}: {count:6,d} ({pct:5.1f}%)\n"
        
        qc_text += f"\n{'=' * 40}\n"
        qc_text += f"\nQC THRESHOLDS:\n{'-' * 40}\n"
        qc_params = self.config['preprocessing']['quality_control']
        qc_text += f"Max Amplitude: {qc_params['max_amplitude_threshold']} µV\n"
        qc_text += f"Min Median STD: {qc_params['min_median_std']}\n"
        qc_text += f"Max Flat Ratio: {qc_params['max_flat_channels_ratio']}\n"
        qc_text += f"Flat Threshold: {qc_params['flat_threshold']}\n"
        qc_text += f"Min Signal Range: {qc_params['min_signal_range']}\n"
        
        axes[1].text(0.1, 0.5, qc_text, fontsize=9, family='monospace',
                    verticalalignment='center', bbox=dict(boxstyle='round',
                    facecolor='wheat', alpha=0.3))
        
        fig.suptitle(f'{self.dataset_name} Quality Metrics', 
                    fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        output_file = output_dir / "quality_metrics.png"
        plt.savefig(output_file, dpi=self.dpi, bbox_inches='tight', facecolor='white')
        plt.close()
