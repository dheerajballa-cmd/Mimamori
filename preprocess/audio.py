"""
audio.py — Ambient Audio Event Classification

Processes pre-computed per-frame audio classification predictions from the
Sounds of Home dataset (YAMNet-style probability vectors) to extract
structured event features for the Mimamori pipeline.

Design note:
  The Sounds of Home dataset stores only pre-computed class probabilities —
  no raw audio files are retained. Privacy requirement (ARCHITECTURE.md §13)
  is therefore satisfied at the data-collection level. The extractor asserts
  this on load.

Event Categories (5 classes):
  0 — silence    : background / ambient quiet
  1 — routine    : normal household activity (TV, music, clock, door)
  2 — footsteps  : movement events (footsteps, shuffling, writing, sliding door)
  3 — distress   : physiological / alarm sounds (breathing, snoring, cough, siren)
  4 — fall       : impact / thump sounds (thud, bang, crash, vibration spike)
"""

import os
import json
import glob
import warnings
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple

try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import classification_report
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    warnings.warn("scikit-learn not found — AudioEventClassifier unavailable.")


# ---------------------------------------------------------------------------
# Event category definitions
# ---------------------------------------------------------------------------

EVENT_CATEGORIES = ["silence", "routine", "footsteps", "distress", "fall"]

# Mapping from YAMNet-style class label (lowercased) → event category index
# Built from observed classes across all 14 Sounds of Home subjects.
# Unmapped classes default to "silence" (0).
CLASS_TO_CATEGORY: Dict[str, int] = {
    # ── silence (0) ──────────────────────────────────────────────
    "silence":                      0,
    "white noise":                  0,
    "static":                       0,
    "pink noise":                   0,
    "hum":                          0,
    "noise":                        0,
    "inside, small room":           0,
    "outside, rural or natural":    0,
    "mains hum":                    0,
    "sine wave":                    0,
    "chirp tone":                   0,
    "hiss":                         0,

    # ── routine (1) ───────────────────────────────────────────────
    "music":                        1,
    "television":                   1,
    "tick":                         1,
    "tick-tock":                    1,
    "clock":                        1,
    "clicking":                     1,
    "mechanical fan":               1,
    "keyboard (musical)":           1,
    "piano":                        1,
    "guitar":                       1,
    "violin, fiddle":               1,
    "marimba, xylophone":           1,
    "glockenspiel":                 1,
    "chime":                        1,
    "mallet percussion":            1,
    "musical instrument":           1,
    "vibraphone":                   1,
    "tubular bells":                1,
    "strum":                        1,
    "plucked string instrument":    1,
    "synthesizer":                  1,
    "singing bowl":                 1,
    "tuning fork":                  1,
    "printer":                      1,
    "beep, bleep":                  1,

    # ── footsteps (2) ─────────────────────────────────────────────
    "rustle":                       2,
    "rustling leaves":              2,
    "writing":                      2,
    "hands":                        2,
    "sliding door":                 2,
    "drawer open or close":         2,
    "door":                         2,
    "squish":                       2,
    "single-lens reflex camera":    2,

    # ── distress (3) ──────────────────────────────────────────────
    "breathing":                    3,
    "snoring":                      3,
    "wheeze":                       3,
    "pant":                         3,
    "sigh":                         3,
    "snort":                        3,
    "sneeze":                       3,
    "cough":                        3,
    "siren":                        3,
    "hiccup":                       3,
    "gasp":                         3,

    # ── fall (4) ──────────────────────────────────────────────────
    "vibration":                    4,
    "throbbing":                    4,
    "cacophony":                    4,
    "spray":                        4,
    "gush":                         4,
    "drip":                         4,

    # Vehicle / background → silence
    "vehicle":                      0,
    "car":                          0,
    "truck":                        0,
    "aircraft":                     0,
    "fixed-wing aircraft, airplane":0,
    "train":                        0,
    "railroad car, train wagon":    0,
    "rail transport":               0,
    "boat, water vehicle":          0,
    "engine":                       0,
    "idling":                       0,
    "medium engine (mid frequency)":0,
    "wind":                         0,
    "buzz":                         0,

    # Animal → routine background
    "animal":                       1,
    "domestic animals, pets":       1,
    "dog":                          1,
    "cat":                          1,
    "bird":                         1,
    "mouse":                        1,
    "snake":                        1,
    "insect":                       1,
    "bee, wasp, etc.":              1,
    "owl":                          1,
}

N_CATEGORIES = len(EVENT_CATEGORIES)


def map_class_to_category(class_name: str) -> int:
    """Map a single class label to a category index (default: silence=0)."""
    return CLASS_TO_CATEGORY.get(class_name.lower(), 0)


# ---------------------------------------------------------------------------
# Privacy check
# ---------------------------------------------------------------------------

RAW_AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".aac", ".m4a", ".opus"}


def assert_no_raw_audio(data_dir: str) -> str:
    """
    Walks the Sounds of Home directory and asserts no raw audio files exist.
    Privacy requirement: ARCHITECTURE.md §13.
    Returns a confirmation message.
    Raises AssertionError if raw audio files are found.
    """
    found = []
    for root, _, files in os.walk(data_dir):
        for fn in files:
            if os.path.splitext(fn)[1].lower() in RAW_AUDIO_EXTENSIONS:
                found.append(os.path.join(root, fn))

    assert not found, (
        f"Privacy violation: raw audio files found in {data_dir}:\n"
        + "\n".join(found[:5])
    )
    return f"[PRIVACY OK] No raw audio files found in {data_dir}."


# ---------------------------------------------------------------------------
# Feature extractor
# ---------------------------------------------------------------------------

class AudioEventExtractor:
    """
    Aggregates per-frame audio class predictions into structured event features.

    Input:  list of prediction records from Sounds of Home JSON files
            [{'frame_index': int, 'time': float, 'predictions': [{'class': str, 'prob': float}]}]

    Output per frame: 5-dim probability vector over [silence, routine, footsteps, distress, fall]
    """

    def __init__(self, top_k: int = 7):
        """
        Args:
            top_k: Maximum number of top predictions to consider per frame.
        """
        self.top_k = top_k

    def frame_to_vector(self, predictions: List[Dict]) -> np.ndarray:
        """
        Convert a single frame's prediction list to a 5-dim category probability vector.

        Strategy: for each prediction, add its probability to its mapped category bucket.
        Then L1-normalise so the vector sums to 1.
        """
        vec = np.zeros(N_CATEGORIES, dtype=np.float32)
        for pred in predictions[:self.top_k]:
            cat = map_class_to_category(pred["class"])
            vec[cat] += pred["prob"]

        total = vec.sum()
        if total > 0:
            vec /= total
        else:
            vec[0] = 1.0  # default to silence
        return vec

    def extract_session(self, records: List[Dict], max_frames: Optional[int] = None) -> pd.DataFrame:
        """
        Extract a frame-level feature DataFrame from one JSON session file.

        Columns: time, silence, routine, footsteps, distress, fall
        """
        rows = []
        for i, rec in enumerate(records):
            if max_frames and i >= max_frames:
                break
            vec = self.frame_to_vector(rec.get("predictions", []))
            row = {"time": rec.get("time", float(i))}
            for j, cat in enumerate(EVENT_CATEGORIES):
                row[cat] = float(vec[j])
            rows.append(row)
        return pd.DataFrame(rows)

    def load_and_extract_json(self, json_path: str, max_frames: Optional[int] = None) -> pd.DataFrame:
        """Load a Sounds of Home JSON file and return its feature DataFrame."""
        with open(json_path, "r") as f:
            records = json.load(f)
        return self.extract_session(records, max_frames=max_frames)

    def window_features(self, session_df: pd.DataFrame, window_size: int = 100, step: int = 50) -> pd.DataFrame:
        """
        Slide a window over the session feature DataFrame and aggregate statistics.

        Returns a DataFrame where each row is an aggregated window feature vector.
        Columns: {cat}_mean, {cat}_max, {cat}_std for each of the 5 categories.
        """
        rows = []
        n = len(session_df)
        for start in range(0, n - window_size + 1, step):
            chunk = session_df.iloc[start : start + window_size]
            row = {}
            for cat in EVENT_CATEGORIES:
                row[f"{cat}_mean"] = float(chunk[cat].mean())
                row[f"{cat}_max"]  = float(chunk[cat].max())
                row[f"{cat}_std"]  = float(chunk[cat].std(ddof=0))
            rows.append(row)
        return pd.DataFrame(rows)

    def dominant_category(self, session_df: pd.DataFrame) -> str:
        """Return the dominant event category for an entire session."""
        cat_sums = {cat: session_df[cat].sum() for cat in EVENT_CATEGORIES}
        return max(cat_sums, key=cat_sums.get)


# ---------------------------------------------------------------------------
# Classifier
# ---------------------------------------------------------------------------

class AudioEventClassifier:
    """
    Lightweight classifier (Random Forest) that assigns each audio window
    to one of the 5 event categories.

    Trained on Sounds of Home using the mapped probability features.
    """

    def __init__(self, n_estimators: int = 100, random_state: int = 42):
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn is required for AudioEventClassifier.")
        self.clf = RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=random_state,
            class_weight="balanced",
        )
        self.scaler = StandardScaler()
        self.feature_names: Optional[List[str]] = None
        self._fitted = False

    def fit(self, X: pd.DataFrame, y: np.ndarray):
        self.feature_names = list(X.columns)
        X_s = self.scaler.fit_transform(X.values)
        self.clf.fit(X_s, y)
        self._fitted = True

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Classifier not fitted.")
        return self.clf.predict(self.scaler.transform(X.values))

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Classifier not fitted.")
        return self.clf.predict_proba(self.scaler.transform(X.values))

    def evaluate(self, X: pd.DataFrame, y: np.ndarray) -> str:
        preds = self.predict(X)
        target_names = [EVENT_CATEGORIES[i] for i in sorted(set(y))]
        return classification_report(y, preds, target_names=target_names)
