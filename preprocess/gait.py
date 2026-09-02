import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, find_peaks
from typing import Dict

class GaitFeatureExtractor:
    def __init__(self, window_sec: float = 5.0, overlap: float = 0.5):
        self.window_sec = window_sec
        self.overlap = overlap
        self.target_hz = 100

    def standardize_columns(self, df: pd.DataFrame, dataset_source: str) -> pd.DataFrame:
        df = df.copy()
        if dataset_source == "sisfall":
            df = df.rename(columns={
                "acc1_x": "acc_x", "acc1_y": "acc_y", "acc1_z": "acc_z",
                "gyr_x": "gyr_x", "gyr_y": "gyr_y", "gyr_z": "gyr_z"
            })
        elif dataset_source == "kfall":
            df = df.rename(columns={
                "AccX": "acc_x", "AccY": "acc_y", "AccZ": "acc_z",
                "GyrX": "gyr_x", "GyrY": "gyr_y", "GyrZ": "gyr_z"
            })
            
        required_cols = ["acc_x", "acc_y", "acc_z", "gyr_x", "gyr_y", "gyr_z"]
        for col in required_cols:
            if col not in df.columns:
                df[col] = 0.0 # Default if missing
        
        # Keep only required to save memory
        return df[required_cols]

    def resample(self, df: pd.DataFrame, original_hz: int) -> pd.DataFrame:
        if original_hz == self.target_hz:
            return df
            
        # Basic linear interpolation for resampling
        duration = len(df) / original_hz
        new_len = int(duration * self.target_hz)
        
        new_idx = np.linspace(0, len(df)-1, new_len)
        old_idx = np.arange(len(df))
        
        resampled = {}
        for col in df.columns:
            resampled[col] = np.interp(new_idx, old_idx, df[col].values)
            
        return pd.DataFrame(resampled)

    def window_signal(self, df: pd.DataFrame):
        window_len = int(self.window_sec * self.target_hz)
        step = int(window_len * (1 - self.overlap))
        
        windows = []
        for start in range(0, len(df) - window_len + 1, step):
            windows.append(df.iloc[start:start+window_len])
        return windows

    def extract_features(self, window_df: pd.DataFrame) -> Dict[str, float]:
        if window_df.isnull().values.any():
            window_df = window_df.interpolate(method='linear', limit_direction='both').fillna(0)
            
        # Calculate magnitudes
        acc_mag = np.sqrt(window_df['acc_x']**2 + window_df['acc_y']**2 + window_df['acc_z']**2)
        gyr_mag = np.sqrt(window_df['gyr_x']**2 + window_df['gyr_y']**2 + window_df['gyr_z']**2)
        
        # 1. Cadence (Steps per minute)
        # Lowpass filter the acc magnitude (cutoff roughly at 3Hz)
        nyq = 0.5 * self.target_hz
        b, a = butter(4, 3.0 / nyq, btype='low')
        if len(acc_mag) > 15:
            filtered_acc = filtfilt(b, a, acc_mag)
        else:
            filtered_acc = acc_mag.values
            
        peaks, _ = find_peaks(filtered_acc, distance=int(self.target_hz * 0.4)) # at least 0.4s between steps
        num_steps = len(peaks)
        
        actual_window_sec = len(window_df) / self.target_hz
        if actual_window_sec > 0:
            cadence = (num_steps / actual_window_sec) * 60.0
        else:
            cadence = 0.0
            
        # 2. Stride Variability (Standard deviation of step intervals)
        if len(peaks) >= 2:
            intervals = np.diff(peaks) / self.target_hz
            stride_variability = np.std(intervals)
        else:
            stride_variability = 0.0
            
        # 3. Sway Index (RMS of gyroscope magnitude as proxy for angular sway)
        sway_index = np.sqrt(np.mean(gyr_mag**2))
        
        return {
            "cadence": float(cadence),
            "stride_variability": float(stride_variability),
            "sway_index": float(sway_index)
        }