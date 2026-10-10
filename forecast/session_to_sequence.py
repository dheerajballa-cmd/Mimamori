"""
forecast/session_to_sequence.py
epic 8.2 & 8.3

Workaround for granularity mismatch:
Treats each SisFall/KFall lab trial window as a 'synthetic day'.
Concatenates a subject's windows to form short synthetic 'weeks'.
Constructs proxy supervised labels (fall-vs-ADL and gait-severity).
"""
import os
import sys
import warnings
import pandas as pd
import numpy as np

sys.path.insert(0, "D:/Mimamori")
from data.loaders.loaders import SisFallLoader, KFallLoader
from preprocess.gait import GaitFeatureExtractor
import data.splits as splits

class SessionToSequenceTransform:
    def __init__(self, context_length: int = 4, target_horizon: int = 3, max_subjects_per_dataset: int = 15):
        """
        context_length: number of days/windows in the input sequence (e.g. 4)
        target_horizon: how many days/windows ahead to predict (short: 3-7)
        """
        self.context_length = context_length
        self.target_horizon = target_horizon
        self.max_subjects = max_subjects_per_dataset
        self.extractor = GaitFeatureExtractor()
        
    def _is_sisfall_fall(self, trial_path: str) -> int:
        filename = os.path.basename(trial_path)
        return 1 if filename.startswith("F") else 0
        
    def build_dataset(self) -> pd.DataFrame:
        sf_loader = SisFallLoader()
        frozen = splits.get_frozen_splits()
        
        all_records = []
        
        # 1. Process SisFall
        sf_subs = (frozen.get("SisFall", {}).get("train", []) + 
                   frozen.get("SisFall", {}).get("val", []) + 
                   frozen.get("SisFall", {}).get("test", []))
                   
        print(f"Processing {min(self.max_subjects, len(sf_subs))} SisFall subjects...")
        for subject in sf_subs[:self.max_subjects]:
            trials = sf_loader.list_trials(subject)
            subject_windows = []
            for trial in trials:
                is_fall = self._is_sisfall_fall(trial)
                try:
                    df_raw = sf_loader.load_trial(trial)
                    df_std = self.extractor.standardize_columns(df_raw, "sisfall")
                    df_res = self.extractor.resample(df_std, original_hz=200)
                    windows = self.extractor.window_signal(df_res)
                    for w in windows:
                        feats = self.extractor.extract_features(w)
                        subject_windows.append({
                            "subject_id": f"SisFall_{subject}",
                            "is_fall": is_fall,
                            "cadence": feats.get("cadence", 0),
                            "stride_variability": feats.get("stride_variability", 0),
                            "sway_index": feats.get("sway_index", 0)
                        })
                except Exception as e:
                    pass
            
            # Create sequences
            all_records.extend(self._create_sequences(subject_windows))
            
        return pd.DataFrame(all_records)
        
    def _create_sequences(self, subject_windows: list) -> list:
        records = []
        n = len(subject_windows)
        for i in range(n - self.context_length - self.target_horizon + 1):
            seq = subject_windows[i : i + self.context_length]
            target = subject_windows[i + self.context_length + self.target_horizon - 1]
            
            records.append({
                "subject_id": seq[0]["subject_id"],
                # Input features (flattened or list)
                "seq_cadence": [w["cadence"] for w in seq],
                "seq_stride_var": [w["stride_variability"] for w in seq],
                "seq_sway": [w["sway_index"] for w in seq],
                # Target Proxy Labels
                "target_is_fall": target["is_fall"],
                "target_gait_severity": target["sway_index"]  # using sway as proxy for severity
            })
        return records

if __name__ == "__main__":
    print("Running session_to_sequence transform...")
    transform = SessionToSequenceTransform(context_length=4, target_horizon=3, max_subjects_per_dataset=15)
    df_seq = transform.build_dataset()
    os.makedirs("D:/Mimamori/data/processed", exist_ok=True)
    out_path = "D:/Mimamori/data/processed/forecast_proxy_dataset.pkl"
    df_seq.to_pickle(out_path)
    print(f"Generated {len(df_seq)} synthetic proxy sequences.")
    print(f"Saved to {out_path}")
