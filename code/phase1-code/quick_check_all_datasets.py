# quick_check_all_datasets.py
import pickle
from pathlib import Path
from collections import Counter

datasets = {
    'TUAB': r'neurovault_data\neurovault_tuab\preprocessed',
    'TUEP': r'neurovault_data\neurovault_tuep\preprocessed',
    'TUEV': r'neurovault_data\neurovault_tuev\preprocessed'
}

for dataset_name, preprocessed_dir in datasets.items():
    print(f"\n{'='*60}")
    print(f"{dataset_name}")
    print('='*60)
    
    path = Path(preprocessed_dir)
    if not path.exists():
        print(f"❌ Directory not found: {preprocessed_dir}")
        continue
    
    # Sample 100 random files
    all_files = list(path.glob("*.pkl"))
    sample_files = all_files[:100] if len(all_files) > 100 else all_files
    
    labels = Counter()
    for pkl_file in sample_files:
        with open(pkl_file, 'rb') as f:
            data = pickle.load(f)
            label = data['metadata'].get('label', 'unknown')
            labels[label] += 1
    
    print(f"📊 Sample: {len(sample_files)} segments")
    for label, count in labels.most_common():
        pct = (count / len(sample_files)) * 100
        print(f"  {label:15s}: {count:4d} ({pct:5.1f}%)")