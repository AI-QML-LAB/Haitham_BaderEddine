import mne

file_path = r'C:\Users\Pc\Desktop\tuh_data\tusl\edf\aaaaaaju\s005_2010\01_tcp_ar\aaaaaaju_s005_t000.edf'
raw = mne.io.read_raw_edf(file_path, preload=False, verbose=True)

# Check what MNE thinks the units are
print("\nChannel info:")
for i, ch in enumerate(raw.info['chs'][:5]):
    print(f"Channel {i}: {raw.ch_names[i]}")
    print(f"  unit: {ch['unit']}")
    print(f"  unit_mul: {ch['unit_mul']}")
    print(f"  cal: {ch['cal']}")