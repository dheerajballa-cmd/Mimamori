"""
test_audio.py — Unit tests for preprocess/audio.py
"""

import sys
import pytest
import numpy as np
import pandas as pd

sys.path.insert(0, "D:/Mimamori")
from preprocess.audio import (
    AudioEventExtractor,
    AudioEventClassifier,
    EVENT_CATEGORIES,
    N_CATEGORIES,
    map_class_to_category,
    assert_no_raw_audio,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_predictions(class_prob_pairs: list) -> list:
    """Build a predictions list like the JSON dataset format."""
    return [{"class": cls, "prob": prob} for cls, prob in class_prob_pairs]


def make_records(n: int, predictions: list) -> list:
    """Build n identical frame records."""
    return [{"frame_index": i, "time": float(i) * 0.1, "predictions": predictions} for i in range(n)]


# ---------------------------------------------------------------------------
# Category mapping
# ---------------------------------------------------------------------------

class TestCategoryMapping:
    def test_silence_classes(self):
        for cls in ["Silence", "White noise", "Static", "Pink noise", "Hum"]:
            assert map_class_to_category(cls) == 0, f"{cls} should be silence"

    def test_routine_classes(self):
        for cls in ["Music", "Television", "Tick-tock", "Clock", "Clicking"]:
            assert map_class_to_category(cls) == 1, f"{cls} should be routine"

    def test_footsteps_classes(self):
        for cls in ["Rustle", "Writing", "Hands", "Sliding door", "Door"]:
            assert map_class_to_category(cls) == 2, f"{cls} should be footsteps"

    def test_distress_classes(self):
        for cls in ["Breathing", "Snoring", "Wheeze", "Sigh", "Sneeze", "Siren"]:
            assert map_class_to_category(cls) == 3, f"{cls} should be distress"

    def test_fall_classes(self):
        for cls in ["Vibration", "Cacophony", "Gush"]:
            assert map_class_to_category(cls) == 4, f"{cls} should be fall"

    def test_unknown_class_defaults_to_silence(self):
        assert map_class_to_category("XYZ unknown sound") == 0

    def test_case_insensitive(self):
        assert map_class_to_category("SILENCE") == map_class_to_category("silence")
        assert map_class_to_category("Television") == map_class_to_category("television")


# ---------------------------------------------------------------------------
# Frame vector extraction
# ---------------------------------------------------------------------------

class TestFrameVector:
    def test_output_shape(self):
        extractor = AudioEventExtractor()
        preds = make_predictions([("Silence", 0.5), ("White noise", 0.3)])
        vec = extractor.frame_to_vector(preds)
        assert vec.shape == (N_CATEGORIES,), f"Expected ({N_CATEGORIES},), got {vec.shape}"

    def test_sums_to_one(self):
        extractor = AudioEventExtractor()
        preds = make_predictions([("Silence", 0.5), ("Music", 0.3), ("Breathing", 0.2)])
        vec = extractor.frame_to_vector(preds)
        assert abs(vec.sum() - 1.0) < 1e-5, f"Vector should sum to 1.0, got {vec.sum()}"

    def test_empty_predictions_defaults_silence(self):
        extractor = AudioEventExtractor()
        vec = extractor.frame_to_vector([])
        assert vec[0] == 1.0  # silence
        assert vec.sum() == pytest.approx(1.0)

    def test_dominant_category_respected(self):
        extractor = AudioEventExtractor()
        # High probability breathing should land mostly in distress (3)
        preds = make_predictions([("Breathing", 0.9), ("Silence", 0.1)])
        vec = extractor.frame_to_vector(preds)
        assert vec.argmax() == 3, "Expected distress to dominate"

    def test_fall_category_detected(self):
        extractor = AudioEventExtractor()
        preds = make_predictions([("Vibration", 0.8), ("Cacophony", 0.7)])
        vec = extractor.frame_to_vector(preds)
        assert vec.argmax() == 4, "Expected fall to dominate"


# ---------------------------------------------------------------------------
# Session extraction
# ---------------------------------------------------------------------------

class TestSessionExtraction:
    def test_extract_session_shape(self):
        extractor = AudioEventExtractor()
        preds = make_predictions([("Silence", 0.8), ("Music", 0.5)])
        records = make_records(50, preds)
        df = extractor.extract_session(records)
        assert len(df) == 50
        assert list(df.columns) == ["time"] + EVENT_CATEGORIES

    def test_extract_session_max_frames(self):
        extractor = AudioEventExtractor()
        preds = make_predictions([("Silence", 0.5)])
        records = make_records(200, preds)
        df = extractor.extract_session(records, max_frames=100)
        assert len(df) == 100

    def test_category_columns_sum_to_one(self):
        extractor = AudioEventExtractor()
        preds = make_predictions([("Music", 0.6), ("Breathing", 0.4)])
        records = make_records(10, preds)
        df = extractor.extract_session(records)
        row_sums = df[EVENT_CATEGORIES].sum(axis=1)
        assert (row_sums - 1.0).abs().max() < 1e-5


# ---------------------------------------------------------------------------
# Window aggregation
# ---------------------------------------------------------------------------

class TestWindowFeatures:
    def test_windowing_output_columns(self):
        extractor = AudioEventExtractor()
        preds = make_predictions([("Silence", 0.5), ("Music", 0.5)])
        records = make_records(300, preds)
        df = extractor.extract_session(records)
        windowed = extractor.window_features(df, window_size=100, step=50)
        expected_cols = [f"{cat}_{stat}" for cat in EVENT_CATEGORIES for stat in ["mean","max","std"]]
        for col in expected_cols:
            assert col in windowed.columns, f"Missing column: {col}"

    def test_windowing_count(self):
        extractor = AudioEventExtractor()
        preds = make_predictions([("Silence", 0.5)])
        records = make_records(300, preds)
        df = extractor.extract_session(records)
        # 300 frames, window=100, step=50 → (300-100)/50 + 1 = 5 windows
        windowed = extractor.window_features(df, window_size=100, step=50)
        assert len(windowed) == 5

    def test_dominant_category(self):
        extractor = AudioEventExtractor()
        preds = make_predictions([("Vibration", 0.9)])
        records = make_records(50, preds)
        df = extractor.extract_session(records)
        dom = extractor.dominant_category(df)
        assert dom == "fall"


# ---------------------------------------------------------------------------
# Real data smoke test
# ---------------------------------------------------------------------------

class TestRealData:
    def test_load_real_json(self):
        import glob
        jsons = glob.glob("D:/Mimamori/Sounds_of_Home/01/*.json")
        assert jsons, "No Sounds of Home JSON files found."
        extractor = AudioEventExtractor()
        df = extractor.load_and_extract_json(jsons[0], max_frames=200)
        assert len(df) == 200
        assert "footsteps" in df.columns
        row_sums = df[EVENT_CATEGORIES].sum(axis=1)
        assert (row_sums - 1.0).abs().max() < 1e-5

    def test_privacy_check_passes(self):
        msg = assert_no_raw_audio("D:/Mimamori/Sounds_of_Home")
        assert "PRIVACY OK" in msg


# ---------------------------------------------------------------------------
# Classifier
# ---------------------------------------------------------------------------

class TestAudioEventClassifier:
    def _make_data(self, n: int = 120):
        np.random.seed(0)
        n_feats = N_CATEGORIES * 3  # mean, max, std per category = 15
        X = pd.DataFrame(
            np.random.rand(n, n_feats),
            columns=[f"f_{i}" for i in range(n_feats)]
        )
        y = np.random.randint(0, N_CATEGORIES, size=n)
        return X, y

    def test_fit_and_predict(self):
        clf = AudioEventClassifier(n_estimators=10)
        X, y = self._make_data(120)
        clf.fit(X[:90], y[:90])
        preds = clf.predict(X[90:])
        assert len(preds) == 30
        assert set(preds).issubset(set(range(N_CATEGORIES)))

    def test_predict_proba_shape(self):
        clf = AudioEventClassifier(n_estimators=10)
        X, y = self._make_data(100)
        clf.fit(X[:80], y[:80])
        proba = clf.predict_proba(X[80:])
        # Shape: (n_test, n_classes_seen)
        assert proba.shape[0] == 20

    def test_evaluate_returns_string(self):
        clf = AudioEventClassifier(n_estimators=10)
        X, y = self._make_data(100)
        clf.fit(X[:80], y[:80])
        report = clf.evaluate(X[80:], y[80:])
        assert isinstance(report, str)
        assert "precision" in report.lower()

    def test_unfitted_raises(self):
        clf = AudioEventClassifier(n_estimators=10)
        X, _ = self._make_data(10)
        with pytest.raises(RuntimeError):
            clf.predict(X)
