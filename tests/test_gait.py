import pytest
import pandas as pd
import numpy as np
import sys

sys.path.append("D:/Mimamori")
from preprocess.gait import GaitFeatureExtractor

def test_standardize_columns():
    extractor = GaitFeatureExtractor()
    df = pd.DataFrame({'acc1_x': [1,2], 'acc1_y': [3,4], 'acc1_z': [5,6], 'gyr_x': [7,8], 'gyr_y': [9,10], 'gyr_z': [11,12]})
    std_df = extractor.standardize_columns(df, "sisfall")
    assert list(std_df.columns) == ["acc_x", "acc_y", "acc_z", "gyr_x", "gyr_y", "gyr_z"]

def test_resampling():
    extractor = GaitFeatureExtractor()
    df = pd.DataFrame({'acc_x': np.linspace(0, 10, 200)}) # 1 second at 200Hz
    res = extractor.resample(df, original_hz=200)
    assert len(res) == 100 # Should be 100Hz

def test_windowing():
    extractor = GaitFeatureExtractor(window_sec=5.0, overlap=0.5)
    # 10 seconds of data at 100Hz = 1000 samples
    df = pd.DataFrame({'acc_x': np.zeros(1000)})
    windows = extractor.window_signal(df)
    
    # Windows:
    # 0-5s (0-500)
    # 2.5-7.5s (250-750)
    # 5-10s (500-1000)
    assert len(windows) == 3
    assert len(windows[0]) == 500

def test_sensor_dropout():
    extractor = GaitFeatureExtractor()
    df = pd.DataFrame({
        'acc_x': [1.0, np.nan, 3.0, 4.0],
        'acc_y': [0]*4, 'acc_z': [0]*4, 'gyr_x': [0]*4, 'gyr_y': [0]*4, 'gyr_z': [0]*4
    })
    
    # Should not throw exception
    feats = extractor.extract_features(df)
    assert 'cadence' in feats

def test_cadence_calculation():
    extractor = GaitFeatureExtractor()
    
    # Simulate 500 samples (5s at 100Hz)
    t = np.linspace(0, 5, 500)
    
    # Simulate a walking pattern (2 steps per second = 120 cadence)
    # using a sine wave on acc_z
    acc_z = 1.0 + np.sin(2 * np.pi * 2.0 * t) 
    
    df = pd.DataFrame({
        'acc_x': np.zeros(500),
        'acc_y': np.zeros(500),
        'acc_z': acc_z,
        'gyr_x': np.zeros(500),
        'gyr_y': np.zeros(500),
        'gyr_z': np.zeros(500)
    })
    
    feats = extractor.extract_features(df)
    # Cadence should be close to 120
    assert abs(feats['cadence'] - 120.0) < 20.0 # Some tolerance for filtering/peak finding