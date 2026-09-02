import sys
import os
import pandas as pd

sys.path.append("D:/Mimamori")
from data.loaders.loaders import SisFallLoader, KFallLoader
from preprocess.gait import GaitFeatureExtractor
import data.splits as splits

def process_dataset(ds_name: str, loader, loader_method, source_name: str, original_hz: int, split_dict: dict):
    print(f"\n--- Processing {ds_name} ---")
    extractor = GaitFeatureExtractor()
    
    total_windows = 0
    total_cadence = 0.0
    
    # We will just process 1 subject from each split to verify end-to-end without taking hours
    for split_type in ['train', 'val', 'test']:
        subjects = split_dict.get(split_type, [])
        if not subjects:
            continue
            
        subject = subjects[0] # Just the first one for validation
        print(f"  Validating subject {subject} ({split_type} split)...")
        
        try:
            if ds_name == "SisFall":
                trials = loader.list_trials(subject)
                if not trials: continue
                df = loader.load_trial(trials[0])
            else: # KFall
                # Assume sensor csv exists
                import glob
                sensor_path = os.path.join(loader.sensor_dir, subject)
                if not os.path.exists(sensor_path):
                    sensor_path = os.path.join(loader.elder_dir, subject)
                csvs = glob.glob(os.path.join(sensor_path, "*.csv"))
                if not csvs: continue
                df = loader.load_sensor_csv(csvs[0])
                
            df = extractor.standardize_columns(df, source_name)
            df = extractor.resample(df, original_hz=original_hz)
            windows = extractor.window_signal(df)
            
            for w in windows:
                feats = extractor.extract_features(w)
                total_cadence += feats['cadence']
                total_windows += 1
        except Exception as e:
            print(f"  Error processing subject {subject}: {e}")
            
    if total_windows > 0:
        avg_cadence = total_cadence / total_windows
        print(f"  {ds_name} extraction successful. Sampled {total_windows} windows. Avg Cadence: {avg_cadence:.2f}")
    else:
        print(f"  {ds_name} extraction yielded no windows.")

if __name__ == "__main__":
    frozen_splits = splits.get_frozen_splits()
    
    # Process SisFall
    process_dataset("SisFall", SisFallLoader(), "load_trial", "sisfall", 200, frozen_splits.get("SisFall", {}))
    
    # Process KFall
    process_dataset("KFall", KFallLoader(), "load_sensor_csv", "kfall", 100, frozen_splits.get("KFall", {}))
    
    print("\nEpic 3 Generalization Validation Complete.")