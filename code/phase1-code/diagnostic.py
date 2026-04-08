#!/usr/bin/env python3
"""
NeuroVault Diagnostic Tool
==========================

Validates preprocessing pipeline before full run.
Tests unit detection, channel mapping, and quality thresholds.
"""

import numpy as np
import mne
from pathlib import Path
import yaml


class PreprocessingDiagnostic:
    """Diagnostic tool to validate preprocessing pipeline."""
    
    def __init__(self, config_path):
        """Initialize with config file."""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
    
    def detect_units(self, raw):
        """
        Detect actual units in EDF file.
        
        Returns:
            dict with unit info
        """
        data = raw.get_data()
        data_std = np.std(data)
        data_max = np.max(np.abs(data))
        data_mean = np.mean(np.abs(data))
        
        # Determine likely unit
        if data_std < 1.0:
            if data_max > 1.0:
                detected_unit = 'mV (millivolts)'
                scale_factor = 1e3
            elif data_max < 0.1:
                detected_unit = 'V (volts)'
                scale_factor = 1e6
            else:
                detected_unit = 'µV (microvolts) - but suspiciously low amplitude'
                scale_factor = 1.0
        else:
            detected_unit = 'µV (microvolts)'
            scale_factor = 1.0
        
        return {
            'detected_unit': detected_unit,
            'scale_factor': scale_factor,
            'std_before': data_std,
            'max_before': data_max,
            'mean_before': data_mean,
            'std_after': data_std * scale_factor,
            'max_after': data_max * scale_factor,
            'mean_after': data_mean * scale_factor
        }
    
    def check_channels(self, raw):
        """
        Check channel configuration.
        
        Returns:
            dict with channel info
        """
        current_channels = raw.ch_names
        target_channels = self.config['preprocessing']['target_channels']
        
        # Check how many target channels exist in data
        channels_found = []
        channels_missing = []
        channel_mapping = {}
        
        for target_ch in target_channels:
            found = False
            # Parse target channel (e.g., "FP1-F7" -> "FP1", "F7")
            parts = target_ch.split('-')
            if len(parts) == 2:
                ch1, ch2 = parts
                
                # Try multiple matching patterns for TUH data
                patterns = [
                    target_ch,  # Exact: "FP1-F7"
                    f"EEG {target_ch}",  # With prefix: "EEG FP1-F7"
                    f"EEG {ch1}-REF",  # Monopolar reference: "EEG FP1-REF"
                    f"EEG {ch1.upper()}-REF",
                    f"EEG {ch1.lower()}-REF",
                    target_ch.upper(),
                    target_ch.lower(),
                ]
                
                for curr_ch in current_channels:
                    curr_ch_clean = curr_ch.strip()
                    for pattern in patterns:
                        if pattern.lower() in curr_ch_clean.lower() or curr_ch_clean.lower() in pattern.lower():
                            channels_found.append(target_ch)
                            channel_mapping[target_ch] = curr_ch
                            found = True
                            break
                    if found:
                        break
            
            if not found:
                channels_missing.append(target_ch)
        
        return {
            'n_current_channels': len(current_channels),
            'n_target_channels': len(target_channels),
            'n_found': len(channels_found),
            'n_missing': len(channels_missing),
            'channels_found': channels_found,
            'channels_missing': channels_missing,
            'channel_mapping': channel_mapping,
            'current_channel_names': current_channels
        }
    
    def predict_segment_yield(self, raw):
        """
        Predict how many segments will pass quality control.
        
        Returns:
            dict with predictions
        """
        # Get config
        segment_duration = self.config['preprocessing']['segment_duration']
        qc_config = self.config['preprocessing']['quality_control']
        
        # Create segments
        sfreq = raw.info['sfreq']
        n_samples_per_segment = int(segment_duration * sfreq)
        
        data = raw.get_data()
        n_channels, n_samples = data.shape
        
        total_possible = int(n_samples / n_samples_per_segment)
        
        # Test each segment
        passed = 0
        failed_reasons = {
            'excessive_amplitude': 0,
            'low_variance': 0,
            'flat_channels': 0,
            'invalid_data': 0
        }
        
        for i in range(total_possible):
            start = i * n_samples_per_segment
            end = start + n_samples_per_segment
            segment = data[:, start:end]
            
            # Apply normalization (as pipeline would)
            if self.config['preprocessing']['normalization']['method'] == 'global_zscore':
                mu = np.mean(segment)
                sigma = np.std(segment)
                epsilon = float(self.config['preprocessing']['normalization']['epsilon'])
                segment = (segment - mu) / max(sigma, epsilon)
            
            # Check quality
            reason = self._check_segment_quality(segment, qc_config)
            
            if reason == 'passed':
                passed += 1
            else:
                failed_reasons[reason] += 1
        
        return {
            'total_possible_segments': total_possible,
            'predicted_passed': passed,
            'predicted_rejected': total_possible - passed,
            'predicted_yield_rate': passed / max(total_possible, 1) * 100,
            'rejection_reasons': failed_reasons
        }
    
    def _check_segment_quality(self, segment, qc_config):
        """Internal quality check (matches main pipeline)."""
        # Check for NaN/Inf
        if np.any(np.isnan(segment)) or np.any(np.isinf(segment)):
            return 'invalid_data'
        
        # Amplitude check
        amplitude_95 = np.percentile(np.abs(segment), qc_config['max_amplitude_percentile'])
        if amplitude_95 > qc_config['max_amplitude_threshold']:
            return 'excessive_amplitude'
        
        # Variance check
        channel_stds = np.std(segment, axis=1)
        median_std = np.median(channel_stds)
        if median_std < qc_config['min_median_std']:
            return 'low_variance'
        
        # Flat channel check
        n_flat = np.sum(channel_stds < qc_config['flat_threshold'])
        if n_flat / len(channel_stds) > qc_config['max_flat_channels_ratio']:
            return 'flat_channels'
        
        # Signal range check
        signal_range = np.max(segment) - np.min(segment)
        if signal_range < qc_config['min_signal_range']:
            return 'low_variance'
        
        return 'passed'
    
    def run_diagnostic(self, edf_file_path):
        """
        Run complete diagnostic on an EDF file.
        
        Args:
            edf_file_path: Path to EDF file
        """
        print("\n" + "="*80)
        print(f"NeuroVault Preprocessing Diagnostic")
        print("="*80)
        print(f"File: {Path(edf_file_path).name}")
        print()
        
        # Load file
        try:
            raw = mne.io.read_raw_edf(edf_file_path, preload=True, verbose=False)
            print(f"✓ Successfully loaded EDF file")
            print(f"  - Sampling rate: {raw.info['sfreq']} Hz")
            print(f"  - Duration: {raw.times[-1]:.2f} seconds")
            print(f"  - Number of channels: {len(raw.ch_names)}")
        except Exception as e:
            print(f"✗ Failed to load EDF file: {e}")
            return
        
        print("\n" + "-"*80)
        print("1. UNIT DETECTION")
        print("-"*80)
        
        unit_info = self.detect_units(raw)
        print(f"Detected unit: {unit_info['detected_unit']}")
        print(f"Scale factor: {unit_info['scale_factor']}")
        print(f"\nBefore scaling:")
        print(f"  - Mean: {unit_info['mean_before']:.6f}")
        print(f"  - Std:  {unit_info['std_before']:.6f}")
        print(f"  - Max:  {unit_info['max_before']:.6f}")
        print(f"\nAfter scaling to µV:")
        print(f"  - Mean: {unit_info['mean_after']:.2f} µV")
        print(f"  - Std:  {unit_info['std_after']:.2f} µV")
        print(f"  - Max:  {unit_info['max_after']:.2f} µV")
        
        # Scale for remaining checks
        raw._data *= unit_info['scale_factor']
        
        print("\n" + "-"*80)
        print("2. CHANNEL CONFIGURATION")
        print("-"*80)
        
        channel_info = self.check_channels(raw)
        print(f"Current channels in file: {channel_info['n_current_channels']}")
        print(f"Target channels in config: {channel_info['n_target_channels']}")
        print(f"Channels found: {channel_info['n_found']}")
        print(f"Channels missing: {channel_info['n_missing']}")
        
        # Show actual channel names from file
        print(f"\nActual channel names in this file:")
        for i, ch in enumerate(channel_info['current_channel_names'][:10]):
            print(f"  {i+1}. {ch}")
        if channel_info['n_current_channels'] > 10:
            print(f"  ... and {channel_info['n_current_channels'] - 10} more")
        
        if channel_info['n_missing'] > 0:
            print(f"\n⚠ WARNING: {channel_info['n_missing']} channels will need interpolation:")
            for ch in channel_info['channels_missing'][:5]:  # Show first 5
                print(f"  - {ch}")
            if channel_info['n_missing'] > 5:
                print(f"  ... and {channel_info['n_missing'] - 5} more")
        
        if channel_info['n_found'] > 0:
            print(f"\nChannel mapping (found {channel_info['n_found']}):")
            for target, actual in list(channel_info['channel_mapping'].items())[:5]:
                print(f"  {target} → {actual}")
            if len(channel_info['channel_mapping']) > 5:
                print(f"  ... and {len(channel_info['channel_mapping']) - 5} more")
        
        print(f"\n✓ {channel_info['n_found']}/{channel_info['n_target_channels']} channels available")
        
        print("\n" + "-"*80)
        print("3. RESAMPLING")
        print("-"*80)
        
        target_sfreq = self.config['preprocessing']['target_sfreq']
        current_sfreq = raw.info['sfreq']
        
        if current_sfreq != target_sfreq:
            print(f"Will resample from {current_sfreq} Hz to {target_sfreq} Hz")
            raw.resample(target_sfreq, verbose=False)
            print(f"✓ Resampled successfully")
        else:
            print(f"✓ Already at target frequency ({target_sfreq} Hz)")
        
        print("\n" + "-"*80)
        print("4. FILTERING")
        print("-"*80)
        
        bp_config = self.config['preprocessing']['bandpass_filter']
        print(f"Bandpass filter: {bp_config['low_freq']}-{bp_config['high_freq']} Hz")
        
        raw.filter(
            l_freq=bp_config['low_freq'],
            h_freq=bp_config['high_freq'],
            method=bp_config['method'],
            phase=bp_config['phase'],
            verbose=False
        )
        print(f"✓ Bandpass filter applied")
        
        notch_freq = self.config['preprocessing']['notch_filter']['freq_60hz']
        print(f"Notch filter: {notch_freq} Hz")
        raw.notch_filter(freqs=notch_freq, method='fir', phase='zero', verbose=False)
        print(f"✓ Notch filter applied")
        
        print("\n" + "-"*80)
        print("5. SEGMENT YIELD PREDICTION")
        print("-"*80)
        
        prediction = self.predict_segment_yield(raw)
        print(f"Total possible segments: {prediction['total_possible_segments']}")
        print(f"Predicted to pass QC: {prediction['predicted_passed']}")
        print(f"Predicted to be rejected: {prediction['predicted_rejected']}")
        print(f"Predicted yield rate: {prediction['predicted_yield_rate']:.1f}%")
        
        if prediction['predicted_rejected'] > 0:
            print(f"\nRejection reasons:")
            for reason, count in prediction['rejection_reasons'].items():
                if count > 0:
                    pct = count / prediction['total_possible_segments'] * 100
                    print(f"  - {reason}: {count} ({pct:.1f}%)")
        
        print("\n" + "="*80)
        print("DIAGNOSTIC COMPLETE")
        print("="*80)
        
        # Summary
        if prediction['predicted_yield_rate'] < 20:
            print("\n⚠ WARNING: Very low segment yield (<20%)!")
            print("   Consider adjusting quality control thresholds.")
        elif prediction['predicted_yield_rate'] < 50:
            print("\n⚠ NOTE: Moderate segment yield (<50%).")
            print("   This may be acceptable for clinical data with artifacts.")
        else:
            print("\n✓ Good segment yield rate!")
        
        if channel_info['n_missing'] > 5:
            print("\n⚠ WARNING: Many channels missing!")
            print("   Verify channel configuration matches your data.")
        
        print()


def main():
    """Run diagnostic on sample files."""
    import argparse
    
    parser = argparse.ArgumentParser(description='NeuroVault Preprocessing Diagnostic')
    parser.add_argument('--config', type=str, required=True, help='Path to config YAML')
    parser.add_argument('--edf-file', type=str, required=True, help='Path to EDF file')
    
    args = parser.parse_args()
    
    diagnostic = PreprocessingDiagnostic(args.config)
    diagnostic.run_diagnostic(args.edf_file)


if __name__ == '__main__':
    main()
