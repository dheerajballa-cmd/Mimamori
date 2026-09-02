"""
thermal.py — Thermal Posture / Fall Feature Extraction

Processes low-resolution thermal frame streams from the Melexis Grid-Eye
sensor (eHomeSeniors dataset) to extract posture and fall-risk features.

Architecture:
  ThermalFrameParser     → parses raw pipe-delimited Melexis CSV files
  ThermalFeatureExtractor → per-frame features (centroid, bbox, delta, posture)
  ThermalFallClassifier   → RandomForest trained on extracted feature sequences

Design note:
  TF-66 provides the fall/non-fall label taxonomy.
  eHomeSeniors Melexis CSVs provide the actual thermal pixel streams for
  training. This proxy arrangement is documented in data/README_TF-66.md.
  CIRDO cross-validation is stubbed pending dataset availability.
"""

import os
import glob
import warnings
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple

# sklearn is used for the classifier
try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import classification_report, roc_auc_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    warnings.warn("scikit-learn not found — ThermalFallClassifier will be unavailable.")


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

GRID_ROWS = 8
GRID_COLS = 8
GRID_SIZE = GRID_ROWS * GRID_COLS  # 64 pixels for Melexis MLX90640-compatible


# ---------------------------------------------------------------------------
# Frame Parser
# ---------------------------------------------------------------------------

class ThermalFrameParser:
    """
    Parses the pipe-delimited Melexis Grid-Eye CSV format used in eHomeSeniors.

    Each line:  timestamp | pixel_0 | pixel_1 | ... | pixel_63 | (header junk)
    Returns a list of (timestamp_str, 8×8 np.ndarray) tuples.
    """

    def __init__(self, grid_size: int = GRID_SIZE):
        self.grid_size = grid_size

    def parse_file(self, csv_path: str, max_frames: Optional[int] = None) -> List[Tuple[str, np.ndarray]]:
        """Parse a single Melexis CSV file into a list of (timestamp, grid) tuples."""
        frames = []

        with open(csv_path, "r", errors="replace") as f:
            for idx, line in enumerate(f):
                if max_frames and idx >= max_frames:
                    break
                line = line.strip()
                if not line:
                    continue

                parts = line.split("|")
                timestamp = parts[0].strip()

                temps = []
                for p in parts[1:]:
                    p = p.strip()
                    try:
                        val = float(p)
                        if -50.0 < val < 200.0:   # sanity: plausible thermal range (°C)
                            temps.append(val)
                    except ValueError:
                        continue

                if len(temps) >= self.grid_size:
                    grid = np.array(temps[:self.grid_size], dtype=np.float32).reshape(GRID_ROWS, GRID_COLS)
                    frames.append((timestamp, grid))

        return frames

    def parse_directory(self, dir_path: str, max_files: Optional[int] = None, max_frames_per_file: int = 500) -> List[Tuple[str, np.ndarray]]:
        """Parse all Melexis CSVs in a directory."""
        csvs = sorted(glob.glob(os.path.join(dir_path, "*.csv")))
        if max_files:
            csvs = csvs[:max_files]

        all_frames = []
        for csv_path in csvs:
            all_frames.extend(self.parse_file(csv_path, max_frames=max_frames_per_file))

        return all_frames


# ---------------------------------------------------------------------------
# Feature Extractor
# ---------------------------------------------------------------------------

class ThermalFeatureExtractor:
    """
    Extracts posture and motion features from a sequence of 8×8 thermal frames.

    Features per frame (relative to a rolling background estimate):
      - hot_pixel_count   : number of pixels above background threshold
      - centroid_row      : center of mass row (0–7), position proxy
      - centroid_col      : center of mass col (0–7)
      - bbox_height       : bounding-box height of the warm region
      - bbox_width        : bounding-box width of the warm region
      - aspect_ratio      : height / (width + ε)  → standing > 1, lying < 1
      - mean_temp         : mean temperature of hot pixels
      - max_temp          : max pixel temperature
      - temporal_delta    : frame-to-frame L2 change (motion spike proxy)
      - posture_label     : rule-based class: 0=unknown, 1=standing, 2=sitting, 3=lying
    """

    # Temperature threshold above the rolling background to count as "person"
    PERSON_THRESHOLD_C = 2.0
    # Min hot pixels to declare a person is present
    MIN_HOT_PIXELS = 3

    def __init__(self, bg_alpha: float = 0.05):
        """
        Args:
            bg_alpha: Learning rate for exponential moving-average background model.
                      Lower = slower adaptation (more stable).
        """
        self.bg_alpha = bg_alpha
        self._background: Optional[np.ndarray] = None
        self._prev_frame: Optional[np.ndarray] = None

    def reset(self):
        """Reset background and previous frame (call between independent recordings)."""
        self._background = None
        self._prev_frame = None

    def _update_background(self, frame: np.ndarray):
        if self._background is None:
            self._background = frame.copy()
        else:
            self._background = (1 - self.bg_alpha) * self._background + self.bg_alpha * frame

    def extract_frame(self, frame: np.ndarray) -> Dict[str, float]:
        """Extract features from a single 8×8 thermal frame."""
        self._update_background(frame)

        # Foreground mask: pixels significantly warmer than background
        fg = (frame - self._background) > self.PERSON_THRESHOLD_C

        hot_pixel_count = int(fg.sum())

        # Temporal delta
        if self._prev_frame is not None:
            temporal_delta = float(np.linalg.norm(frame.astype(float) - self._prev_frame.astype(float)))
        else:
            temporal_delta = 0.0
        self._prev_frame = frame.copy()

        if hot_pixel_count < self.MIN_HOT_PIXELS:
            # No clear person silhouette — return zeros
            return {
                "hot_pixel_count": hot_pixel_count,
                "centroid_row": -1.0, "centroid_col": -1.0,
                "bbox_height": 0.0, "bbox_width": 0.0,
                "aspect_ratio": 0.0,
                "mean_temp": 0.0, "max_temp": 0.0,
                "temporal_delta": temporal_delta,
                "posture_label": 0.0,   # unknown
            }

        # Centroid
        rows, cols = np.where(fg)
        centroid_row = float(rows.mean())
        centroid_col = float(cols.mean())

        # Bounding box
        bbox_height = float(rows.max() - rows.min() + 1)
        bbox_width  = float(cols.max() - cols.min() + 1)
        aspect_ratio = bbox_height / (bbox_width + 1e-6)

        # Pixel temperatures
        hot_temps = frame[fg]
        mean_temp = float(hot_temps.mean())
        max_temp  = float(hot_temps.max())

        # Rule-based posture
        posture_label = self._classify_posture(aspect_ratio, centroid_row)

        return {
            "hot_pixel_count": hot_pixel_count,
            "centroid_row": centroid_row,
            "centroid_col": centroid_col,
            "bbox_height": bbox_height,
            "bbox_width": bbox_width,
            "aspect_ratio": aspect_ratio,
            "mean_temp": mean_temp,
            "max_temp": max_temp,
            "temporal_delta": temporal_delta,
            "posture_label": float(posture_label),
        }

    def extract_sequence(self, frames: List[np.ndarray]) -> pd.DataFrame:
        """Extract features from a sequence of frames."""
        self.reset()
        records = [self.extract_frame(f) for f in frames]
        return pd.DataFrame(records)

    @staticmethod
    def _classify_posture(aspect_ratio: float, centroid_row: float) -> int:
        """
        Rule-based posture classification.
          Standing : aspect_ratio > 1.4  (tall silhouette)
          Sitting  : 0.7 < aspect_ratio <= 1.4
          Lying    : aspect_ratio <= 0.7  (wide silhouette)
          Unknown  : 0
        """
        if aspect_ratio > 1.4:
            return 1   # standing
        elif aspect_ratio > 0.7:
            return 2   # sitting
        else:
            return 3   # lying


# ---------------------------------------------------------------------------
# Aggregate window features
# ---------------------------------------------------------------------------

def aggregate_window_features(feature_df: pd.DataFrame) -> Dict[str, float]:
    """
    Reduce a window's frame-level features into a single feature vector
    suitable for the classifier.
    """
    if feature_df.empty:
        return {}

    numeric = feature_df.select_dtypes(include=[np.number])

    agg = {}
    for col in numeric.columns:
        agg[f"{col}_mean"] = float(numeric[col].mean())
        agg[f"{col}_std"]  = float(numeric[col].std(ddof=0))
        agg[f"{col}_max"]  = float(numeric[col].max())

    # Extra: max temporal delta spike (the most fall-indicative signal)
    agg["temporal_delta_spike"] = float(numeric["temporal_delta"].max()) if "temporal_delta" in numeric else 0.0

    return agg


# ---------------------------------------------------------------------------
# Classifier
# ---------------------------------------------------------------------------

class ThermalFallClassifier:
    """
    Random Forest classifier for fall vs. non-fall based on extracted
    thermal feature windows.

    Training proxy:
      Labels come from TF-66 metadata (label taxonomy).
      Pixel data comes from eHomeSeniors Melexis thermal stream.
      See docs/README_TF-66.md for the documented limitation.
    """

    def __init__(self, n_estimators: int = 100, random_state: int = 42):
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn is required for ThermalFallClassifier.")
        self.clf = RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=random_state,
            class_weight="balanced",
        )
        self.scaler = StandardScaler()
        self.feature_names: Optional[List[str]] = None
        self._fitted = False

    def fit(self, X: pd.DataFrame, y: np.ndarray):
        """Fit the classifier. X should be aggregated window feature vectors."""
        self.feature_names = list(X.columns)
        X_scaled = self.scaler.fit_transform(X.values)
        self.clf.fit(X_scaled, y)
        self._fitted = True

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Classifier not fitted yet.")
        X_scaled = self.scaler.transform(X.values)
        return self.clf.predict(X_scaled)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Classifier not fitted yet.")
        X_scaled = self.scaler.transform(X.values)
        return self.clf.predict_proba(X_scaled)

    def evaluate(self, X: pd.DataFrame, y: np.ndarray) -> Dict[str, object]:
        preds = self.predict(X)
        proba = self.predict_proba(X)[:, 1]
        report = classification_report(y, preds, target_names=["non_fall", "fall"], output_dict=True)
        try:
            auc = roc_auc_score(y, proba)
        except Exception:
            auc = float("nan")
        return {"classification_report": report, "roc_auc": auc}

    def check_cirdo(self, cirdo_dir: Optional[str] = None) -> str:
        """
        Stub for CIRDO cross-validation (Epic 4.3).
        Returns a status message. Will run evaluation when CIRDO is provided.
        """
        if cirdo_dir is None or not os.path.exists(str(cirdo_dir)):
            return (
                "[CIRDO] Cross-validation SKIPPED — dataset not yet available. "
                "Provide cirdo_dir path once CIRDO files are acquired. "
                "This is a named risk in ARCHITECTURE.md §9."
            )
        # TODO: implement when CIRDO is available
        return "[CIRDO] Directory found but CIRDO loader not yet implemented."
