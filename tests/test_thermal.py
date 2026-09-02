"""
test_thermal.py — Unit tests for preprocess/thermal.py and preprocess/tf66_labels.py
"""

import sys
import pytest
import numpy as np
import pandas as pd

sys.path.insert(0, "D:/Mimamori")
from preprocess.thermal import (
    ThermalFrameParser,
    ThermalFeatureExtractor,
    ThermalFallClassifier,
    aggregate_window_features,
    GRID_ROWS, GRID_COLS
)
from preprocess.tf66_labels import TF66LabelParser


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_blank_frame(base_temp: float = 20.0) -> np.ndarray:
    """Returns a uniform 8×8 frame (no person)."""
    return np.full((GRID_ROWS, GRID_COLS), base_temp, dtype=np.float32)


def make_standing_frame(base_temp: float = 20.0, person_temp: float = 35.0) -> np.ndarray:
    """Simulates a standing person — tall, narrow warm region (centre columns, all rows)."""
    frame = make_blank_frame(base_temp)
    frame[:, 3:5] = person_temp  # 2 columns, all 8 rows → aspect ~4:1 → standing
    return frame


def make_lying_frame(base_temp: float = 20.0, person_temp: float = 35.0) -> np.ndarray:
    """Simulates a lying person — wide, short warm region."""
    frame = make_blank_frame(base_temp)
    frame[6:8, :] = person_temp  # 2 rows, all 8 cols → aspect ~0.25 → lying
    return frame


# ---------------------------------------------------------------------------
# TF-66 Label Parser
# ---------------------------------------------------------------------------

class TestTF66Labels:
    def test_load_returns_dataframe(self):
        parser = TF66LabelParser()
        df = parser.load()
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0

    def test_columns_present(self):
        parser = TF66LabelParser()
        df = parser.load()
        required = ["volunteer_id", "recording_name", "is_fall", "action"]
        for col in required:
            assert col in df.columns, f"Missing column: {col}"

    def test_binary_labels(self):
        parser = TF66LabelParser()
        df = parser.load()
        assert set(df["is_fall"].unique()).issubset({0, 1})

    def test_summary_keys(self):
        parser = TF66LabelParser()
        s = parser.summary()
        assert "total_recordings" in s
        assert "fall_recordings" in s
        assert s["fall_recordings"] > 0

    def test_fall_nonfalls_partition(self):
        parser = TF66LabelParser()
        df = parser.load()
        assert len(parser.fall_recordings()) + len(parser.non_fall_recordings()) == len(df)


# ---------------------------------------------------------------------------
# ThermalFrameParser
# ---------------------------------------------------------------------------

class TestThermalFrameParser:
    def test_parse_real_file(self):
        import glob
        csvs = glob.glob("D:/Mimamori/eHomeSeniors dataset/melexis sensor/*.csv")
        assert csvs, "No Melexis CSVs found — check dataset path."
        parser = ThermalFrameParser()
        frames = parser.parse_file(csvs[0], max_frames=20)
        assert len(frames) > 0

    def test_frame_shape(self):
        import glob
        csvs = glob.glob("D:/Mimamori/eHomeSeniors dataset/melexis sensor/*.csv")
        parser = ThermalFrameParser()
        frames = parser.parse_file(csvs[0], max_frames=5)
        for ts, grid in frames:
            assert grid.shape == (GRID_ROWS, GRID_COLS), f"Expected 8×8, got {grid.shape}"

    def test_temperature_sanity(self):
        import glob
        csvs = glob.glob("D:/Mimamori/eHomeSeniors dataset/melexis sensor/*.csv")
        parser = ThermalFrameParser()
        frames = parser.parse_file(csvs[0], max_frames=10)
        for _, grid in frames:
            assert grid.min() > -50.0
            assert grid.max() < 200.0


# ---------------------------------------------------------------------------
# ThermalFeatureExtractor
# ---------------------------------------------------------------------------

class TestThermalFeatureExtractor:
    def test_no_person_returns_zeros(self):
        extractor = ThermalFeatureExtractor()
        frame = make_blank_frame(20.0)
        # First call initialises background = frame, so delta = 0 and no foreground
        feats = extractor.extract_frame(frame)
        feats2 = extractor.extract_frame(frame)  # second frame against same background
        assert feats2["hot_pixel_count"] == 0

    def test_standing_person_posture(self):
        extractor = ThermalFeatureExtractor()
        bg = make_blank_frame(20.0)
        extractor.extract_frame(bg)   # seed background
        standing = make_standing_frame(20.0, 35.0)
        feats = extractor.extract_frame(standing)
        # aspect_ratio should be > 1.4 → posture_label == 1 (standing)
        assert feats["posture_label"] == 1.0, f"Expected standing(1), got {feats['posture_label']}"

    def test_lying_person_posture(self):
        extractor = ThermalFeatureExtractor()
        bg = make_blank_frame(20.0)
        extractor.extract_frame(bg)   # seed background
        lying = make_lying_frame(20.0, 35.0)
        feats = extractor.extract_frame(lying)
        # aspect_ratio should be ≤ 0.7 → posture_label == 3 (lying)
        assert feats["posture_label"] == 3.0, f"Expected lying(3), got {feats['posture_label']}"

    def test_temporal_delta_spike_on_change(self):
        extractor = ThermalFeatureExtractor()
        frame_a = make_blank_frame(20.0)
        frame_b = make_standing_frame(20.0, 35.0)
        extractor.extract_frame(frame_a)   # seed background + prev
        feats = extractor.extract_frame(frame_a)   # zero delta
        feats_spike = extractor.extract_frame(frame_b)   # big change
        assert feats_spike["temporal_delta"] > feats["temporal_delta"]

    def test_reset_clears_state(self):
        extractor = ThermalFeatureExtractor()
        extractor.extract_frame(make_blank_frame())
        extractor.reset()
        assert extractor._background is None
        assert extractor._prev_frame is None

    def test_extract_sequence_returns_dataframe(self):
        extractor = ThermalFeatureExtractor()
        frames = [make_blank_frame(20.0 + i * 0.5) for i in range(10)]
        df = extractor.extract_sequence(frames)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 10
        assert "posture_label" in df.columns


# ---------------------------------------------------------------------------
# Aggregate Window Features
# ---------------------------------------------------------------------------

class TestAggregateWindowFeatures:
    def test_produces_dict(self):
        extractor = ThermalFeatureExtractor()
        frames = [make_blank_frame(20.0 + i) for i in range(30)]
        df = extractor.extract_sequence(frames)
        agg = aggregate_window_features(df)
        assert isinstance(agg, dict)
        assert len(agg) > 0

    def test_spike_key_present(self):
        extractor = ThermalFeatureExtractor()
        frames = [make_blank_frame(20.0) for _ in range(10)]
        df = extractor.extract_sequence(frames)
        agg = aggregate_window_features(df)
        assert "temporal_delta_spike" in agg


# ---------------------------------------------------------------------------
# ThermalFallClassifier
# ---------------------------------------------------------------------------

class TestThermalFallClassifier:
    def _make_dummy_data(self, n: int = 60):
        """Build synthetic feature matrix and binary labels."""
        np.random.seed(42)
        n_feats = 20
        X = pd.DataFrame(np.random.randn(n, n_feats), columns=[f"feat_{i}" for i in range(n_feats)])
        y = np.random.randint(0, 2, size=n)
        return X, y

    def test_fit_and_predict(self):
        clf = ThermalFallClassifier(n_estimators=10)
        X, y = self._make_dummy_data(60)
        clf.fit(X[:50], y[:50])
        preds = clf.predict(X[50:])
        assert len(preds) == 10
        assert set(preds).issubset({0, 1})

    def test_predict_proba_shape(self):
        clf = ThermalFallClassifier(n_estimators=10)
        X, y = self._make_dummy_data(60)
        clf.fit(X[:50], y[:50])
        proba = clf.predict_proba(X[50:])
        assert proba.shape == (10, 2)

    def test_evaluate_returns_auc(self):
        clf = ThermalFallClassifier(n_estimators=10)
        X, y = self._make_dummy_data(100)
        clf.fit(X[:80], y[:80])
        results = clf.evaluate(X[80:], y[80:])
        assert "roc_auc" in results

    def test_cirdo_stub_no_dir(self):
        clf = ThermalFallClassifier(n_estimators=10)
        msg = clf.check_cirdo(cirdo_dir=None)
        assert "SKIPPED" in msg or "not yet available" in msg
