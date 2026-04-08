#!/usr/bin/env python3
"""
Quick Visualization Script - No Arguments Needed
=================================================
This script generates visualizations using default paths.
Just run: python quick_visualize.py

Adjust the paths below if your directories are different.
"""

import pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm

# ============================================================================
# CONFIGURATION - ADJUST THESE PATHS IF NEEDED
# ============================================================================

PREPROCESSED_DIR = r"neurovault_data\neurovault_tuab\preprocessed"  # Windows path
OUTPUT_DIR = r"neurovault_data\neurovault_tuab\visualizations"
MAX_PLOTS = 50  # Number of plots to generate

# ============================================================================

def create_plot(segment, metadata, output_path):
    """Create a clean EEG visualization."""
    n_channels, n_samples = segment.shape
    sfreq = metadata.get('sfreq', 200)
    
    # Create figure
    fig, axes = plt.subplots(n_channels, 1, figsize=(15, n_channels * 1.2), sharex=True)
    
    if n_channels == 1:
        axes = [axes]
    
    # Time vector
    time = np.arange(n_samples) / sfreq
    
    # Plot each channel
    for ch_idx in range(n_channels):
        axes[ch_idx].plot(time, segment[ch_idx], linewidth=0.6, color='blue')
        axes[ch_idx].set_ylabel(f'Ch {ch_idx+1}\n(µV)', fontsize=9, rotation=0, ha='right')
        axes[ch_idx].grid(True, alpha=0.3)
        axes[ch_idx].set_xlim([0, time[-1]])
        
        if ch_idx < n_channels - 1:
            axes[ch_idx].set_xticklabels([])
    
    axes[-1].set_xlabel('Time (seconds)', fontsize=10, fontweight='bold')
    
    # Title
    file_name = Path(metadata.get('file_path', 'unknown')).name
    label = metadata.get('label', 'unknown')
    seg_idx = metadata.get('segment_idx', 'N/A')
    
    fig.suptitle(
        f"File: {file_name} | Segment: {seg_idx} | Label: {label.upper()} | {n_channels} channels",
        fontsize=11, fontweight='bold'
    )
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def main():
    print("=" * 70)
    print("QUICK EEG VISUALIZATION")
    print("=" * 70)
    
    # Setup paths
    preprocessed_dir = Path(PREPROCESSED_DIR)
    output_dir = Path(OUTPUT_DIR)
    
    # Check if preprocessed directory exists
    if not preprocessed_dir.exists():
        print(f"❌ ERROR: Directory not found: {preprocessed_dir}")
        print(f"\nPlease update the PREPROCESSED_DIR path in this script.")
        return
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find segment files
    segment_files = sorted(list(preprocessed_dir.glob("*.pkl")))
    
    if not segment_files:
        print(f"❌ No .pkl files found in {preprocessed_dir}")
        return
    
    print(f"\n📁 Found {len(segment_files)} preprocessed segments")
    print(f"🎨 Generating {min(MAX_PLOTS, len(segment_files))} visualizations...")
    print(f"📂 Output: {output_dir.absolute()}\n")
    
    # Generate plots
    created = 0
    failed = 0
    
    for segment_file in tqdm(segment_files[:MAX_PLOTS], desc="Creating plots"):
        try:
            # Load segment
            with open(segment_file, 'rb') as f:
                data = pickle.load(f)
            
            segment = data['segment']
            metadata = data['metadata']
            
            # Create plot
            output_path = output_dir / f"{segment_file.stem}.png"
            create_plot(segment, metadata, output_path)
            created += 1
            
        except Exception as e:
            print(f"\n❌ Failed: {segment_file.name} - {e}")
            failed += 1
    
    print(f"\n{'=' * 70}")
    print(f"✅ Created: {created} plots")
    if failed > 0:
        print(f"❌ Failed: {failed} plots")
    print(f"📂 Location: {output_dir.absolute()}")
    print(f"{'=' * 70}\n")


if __name__ == "__main__":
    main()
