"""
aggregate.py — Daily/Weekly Multi-Modal Feature Aggregation

Implements the DailyFeatureVector schema and adapters for all data sources:
  - NHATSAggregator         : NHATS survey (SP) + accelerometry summary → daily vectors
  - EHomeSeniorsAggregator  : Melexis thermal stream → daily aggregated vectors
  - SisFallKFallBridge      : Session-level IMU windows → synthetic weekly sequences
  - SleepProxyEstimator     : NHATS self-report sleep + Sounds of Home nighttime inactivity
  - RoutineDeviationEstimator: Rolling 7-day hourly baseline deviation

Schema specification:
  D:/Mimamori/data/feature_schema.json

Design notes:
  - NHATS/eHomeSeniors → real longitudinal daily sequences (primary pretraining corpus)
  - SisFall/KFall       → session-level; bridged into synthetic weeks (explicit workaround,
                          documented limitation per ARCHITECTURE.md §9)
  - sleep_proxy         → LOWER CONFIDENCE feature; no direct sleep sensor in any dataset.
                          Derived from NHATS hc11aslep30mn + nighttime Sounds-of-Home audio
                          inactivity. Flagged as such in PREPROCESSING_REPORT.md.
"""

import os
import re
import glob
import json
import warnings
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# DailyFeatureVector dataclass
# ---------------------------------------------------------------------------

@dataclass
class DailyFeatureVector:
    """
    Canonical daily feature record consumed by the pretraining and forecasting layers.
    Fields marked LOW_CONF are derived proxies — do not treat as ground truth.
    """
    subject_id:              str
    date:                    str              # YYYY-MM-DD (or synthetic placeholder)
    source:                  str              # nhats | ehome | sisfall_kfall | synthetic
    round:                   Optional[float] = None  # NHATS round number

    # Gait / biomechanical
    gait_cadence:            Optional[float] = None   # steps/min
    stride_variability:      Optional[float] = None   # stride time std dev (s)
    sway_index:              Optional[float] = None   # gyro RMS (°/s)
    steps_per_day:           Optional[float] = None   # total daily steps
    active_minutes_per_day:  Optional[float] = None   # active minutes
    total_activity_counts:   Optional[float] = None   # total accelerometer counts

    # Thermal / posture
    posture_fall_events:     Optional[int]   = None   # lying/fall posture detections/day
    posture_instability_score: Optional[float] = None # thermal delta spike RMS

    # Audio / ambient
    audio_routine_score:     Optional[float] = None   # mean routine event probability
    audio_distress_events:   Optional[int]   = None   # distress windows/day

    # Derived (lower-confidence proxies)
    sleep_proxy:             Optional[float] = None   # [LOW_CONF] 0-1 normalised
    routine_deviation:       Optional[float] = None   # rolling 7-day baseline deviation

    # Survey / self-report
    worries_about_falling:   Optional[float] = None   # NHATS hc_worryfall
    had_multiple_falls:      Optional[float] = None   # NHATS hc_multifall

    # Quality tracking
    data_quality:            float = 0.0              # fraction of non-null fields

    def compute_data_quality(self) -> float:
        """Compute and set the data_quality field (fraction of non-null schema fields)."""
        schema_fields = [
            "gait_cadence", "stride_variability", "sway_index",
            "steps_per_day", "active_minutes_per_day", "total_activity_counts",
            "posture_fall_events", "posture_instability_score",
            "audio_routine_score", "audio_distress_events",
            "sleep_proxy", "routine_deviation",
        ]
        n_present = sum(1 for f in schema_fields if getattr(self, f) is not None)
        self.data_quality = n_present / len(schema_fields)
        return self.data_quality

    def to_dict(self) -> Dict:
        self.compute_data_quality()
        return asdict(self)


# ---------------------------------------------------------------------------
# Schema validation
# ---------------------------------------------------------------------------

REQUIRED_COLUMNS = {"subject_id", "date", "source", "data_quality"}
NUMERIC_COLUMNS  = {
    "gait_cadence", "stride_variability", "sway_index",
    "steps_per_day", "active_minutes_per_day", "total_activity_counts",
    "posture_fall_events", "posture_instability_score",
    "audio_routine_score", "audio_distress_events",
    "sleep_proxy", "routine_deviation", "data_quality",
}


def validate_schema(df: pd.DataFrame, nan_threshold: float = 0.05) -> Tuple[bool, List[str]]:
    """
    Validate a DataFrame of DailyFeatureVectors.

    Returns (passed: bool, issues: List[str]).
    """
    issues = []

    # 1. Required columns
    for col in REQUIRED_COLUMNS:
        if col not in df.columns:
            issues.append(f"Missing required column: {col}")

    # 2. NaN rates per numeric column
    for col in NUMERIC_COLUMNS:
        if col in df.columns:
            nan_rate = df[col].isna().mean()
            if nan_rate > nan_threshold:
                issues.append(f"NaN rate too high for '{col}': {nan_rate:.1%} > {nan_threshold:.1%}")

    # 3. data_quality not null
    if "data_quality" in df.columns and df["data_quality"].isna().any():
        issues.append("data_quality field contains NaN values.")

    return len(issues) == 0, issues


# ---------------------------------------------------------------------------
# NHATS Aggregator
# ---------------------------------------------------------------------------

class NHATSAggregator:
    """
    Adapts NHATS SP survey + accelerometry summary files into DailyFeatureVectors.

    Rounds used:
      - Survey (SP) fields: Rounds 11-14 (most complete accelerometry overlap)
      - Accelerometry summary: Rounds 11-14 (Accel_Summ_File)
    """

    ROUNDS = [11, 12, 13, 14]

    def __init__(self, nhats_dir: str = "D:/Mimamori/NHATS"):
        self.nhats_dir = nhats_dir

    def _find_file(self, pattern: str) -> Optional[str]:
        matches = glob.glob(os.path.join(self.nhats_dir, pattern))
        return matches[0] if matches else None

    def _load_sp(self, round_num: int) -> Optional[pd.DataFrame]:
        path = self._find_file(f"NHATS_Round_{round_num}_SP_File*.dta")
        if not path:
            return None
        sp = pd.read_stata(path, convert_categoricals=False,
                           columns=["spid"] + self._sp_columns(round_num))
        sp = sp.rename(columns={"spid": "subject_id"})
        sp["subject_id"] = sp["subject_id"].astype(str)
        return sp

    def _sp_columns(self, r: int) -> List[str]:
        """Return available SP columns for a given round (with fallback)."""
        candidates = {
            "worryfall":   [f"hc{r}worryfall"],
            "multifall":   [f"hc{r}multifall"],
            "sleep30mn":   [f"hc{r}aslep30mn"],
            "sleepmed":    [f"hc{r}sleepmed"],
        }
        available = []
        sp_path = self._find_file(f"NHATS_Round_{r}_SP_File*.dta")
        if not sp_path:
            return available
        # Only read column list without loading full file
        try:
            with pd.io.stata.StataReader(sp_path) as reader:
                all_cols = set(reader.variable_labels().keys())
            for group, cols in candidates.items():
                for c in cols:
                    if c in all_cols:
                        available.append(c)
                        break
        except Exception as e:
            warnings.warn(f"Failed to inspect columns in {sp_path}: {e}")
        return available

    def _load_accel_summary(self, round_num: int) -> Optional[pd.DataFrame]:
        # Try Accel_Summ first, then Det fallback
        path = self._find_file(f"NHATS_R{round_num}_Accel_Summ_File*.dta")
        if not path:
            return None
        try:
            accel = pd.read_stata(path, convert_categoricals=False)
        except Exception:
            return None

        accel = accel.rename(columns={"spid": "subject_id"})
        accel["subject_id"] = accel["subject_id"].astype(str)

        # Standardise column names (NHATS naming convention: ag{rr}d{metric})
        rr = str(round_num)
        rename_map = {}
        for col in accel.columns:
            if col == f"ag{rr}dastp":  rename_map[col] = "steps_per_day"
            elif col == f"ag{rr}dminact": rename_map[col] = "active_minutes_per_day"
            elif col == f"ag{rr}dtac":    rename_map[col] = "total_activity_counts"
        accel = accel.rename(columns=rename_map)
        return accel

    def build_daily_vectors(self) -> pd.DataFrame:
        """
        Build a DataFrame of DailyFeatureVectors for all NHATS rounds.
        Each row = one participant-round observation (treated as a daily summary).
        """
        records = []
        for r in self.ROUNDS:
            sp = self._load_sp(r)
            accel = self._load_accel_summary(r)

            if sp is None and accel is None:
                warnings.warn(f"No data found for NHATS Round {r}.")
                continue

            # Merge on subject_id
            if sp is not None and accel is not None:
                merged = sp.merge(accel[["subject_id"] +
                                        [c for c in ["steps_per_day","active_minutes_per_day","total_activity_counts"]
                                         if c in accel.columns]],
                                  on="subject_id", how="left")
            elif sp is not None:
                merged = sp
            else:
                merged = accel

            for _, row in merged.iterrows():
                subj_id = str(row.get("subject_id", "unknown"))
                # Build DFV
                dfv = DailyFeatureVector(
                    subject_id = subj_id,
                    date       = f"round_{r:02d}",   # synthetic date placeholder per round
                    source     = "nhats",
                    round      = float(r),
                    steps_per_day         = _safe_float(row.get("steps_per_day")),
                    active_minutes_per_day= _safe_float(row.get("active_minutes_per_day")),
                    total_activity_counts = _safe_float(row.get("total_activity_counts")),
                )

                # Sleep proxy from self-reported sleep quality
                # hc{r}aslep30mn: problems sleeping ≥30min (1=yes, 2=no)
                # Lower value (more sleep problems) → lower proxy score
                sleep_col = f"hc{r}aslep30mn"
                if sleep_col in row and not pd.isna(row[sleep_col]):
                    raw = float(row[sleep_col])
                    # 1 = has problems (bad sleep) → proxy 0.3, 2 = no problems → 0.8, else NaN
                    dfv.sleep_proxy = 0.3 if raw == 1.0 else (0.8 if raw == 2.0 else None)

                # Survey fall risk
                worryfall_col = f"hc{r}worryfall"
                multifall_col = f"hc{r}multifall"
                if worryfall_col in row:
                    dfv.worries_about_falling = _safe_float(row.get(worryfall_col))
                if multifall_col in row:
                    dfv.had_multiple_falls = _safe_float(row.get(multifall_col))

                dfv.compute_data_quality()
                records.append(dfv.to_dict())

        return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# eHomeSeniors Aggregator
# ---------------------------------------------------------------------------

class EHomeSeniorsAggregator:
    """
    Adapts eHomeSeniors Melexis Grid-Eye 8×8 thermal stream into daily DailyFeatureVectors.

    File naming: melexis-G{group}-{session}-f{file}.csv
      Group = household subject
      Session = recording session (treated as a separate day)
      File = sub-recording within a session

    Aggregation: session-level → one DailyFeatureVector per (group, session).
    """

    def __init__(self, ehome_dir: str = "D:/Mimamori/eHomeSeniors dataset/melexis sensor"):
        self.ehome_dir = ehome_dir

    def _parse_filename(self, path: str) -> Optional[Tuple[int, int, int]]:
        m = re.search(r"G(\d+)-(\d+)-f(\d+)", os.path.basename(path))
        return (int(m.group(1)), int(m.group(2)), int(m.group(3))) if m else None

    def _load_session_frames(self, group: int, session: int, max_frames: int = 1000) -> List[np.ndarray]:
        """Load all files for a (group, session) and return thermal frame arrays."""
        files = sorted(glob.glob(os.path.join(self.ehome_dir, f"melexis-G{group}-{session}-f*.csv")))
        all_frames = []
        for fp in files:
            with open(fp, "r", errors="replace") as f:
                for line in f:
                    if len(all_frames) >= max_frames:
                        break
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split("|")
                    temps = []
                    for p in parts[1:]:
                        try:
                            v = float(p.strip())
                            if -50.0 < v < 200.0:
                                temps.append(v)
                        except ValueError:
                            continue
                    if len(temps) >= 64:
                        all_frames.append(np.array(temps[:64], dtype=np.float32).reshape(8, 8))
            if len(all_frames) >= max_frames:
                break
        return all_frames

    def _extract_session_features(self, frames: List[np.ndarray]) -> Dict:
        """Compute daily thermal features from a session's frames."""
        if not frames:
            return {}

        bg = frames[0].copy()
        alpha = 0.05
        fall_events = 0
        delta_spikes = []
        prev_frame = None

        for f in frames:
            bg = (1 - alpha) * bg + alpha * f
            fg = (f - bg) > 2.0
            # Detect lying posture (aspect ratio ≤ 0.7)
            rows_idx, cols_idx = np.where(fg)
            if len(rows_idx) >= 3:
                h = rows_idx.max() - rows_idx.min() + 1
                w = cols_idx.max() - cols_idx.min() + 1
                if h / (w + 1e-6) <= 0.7:
                    fall_events += 1
            # Temporal delta
            if prev_frame is not None:
                delta_spikes.append(float(np.linalg.norm(f.astype(float) - prev_frame.astype(float))))
            prev_frame = f.copy()

        return {
            "posture_fall_events":      fall_events,
            "posture_instability_score": float(np.mean(delta_spikes)) if delta_spikes else None,
        }

    def build_daily_vectors(self) -> pd.DataFrame:
        """Build a DataFrame of DailyFeatureVectors, one per (group, session)."""
        files = glob.glob(os.path.join(self.ehome_dir, "melexis-G*.csv"))
        # Unique (group, session) pairs
        pairs = set()
        for fp in files:
            parsed = self._parse_filename(fp)
            if parsed:
                pairs.add((parsed[0], parsed[1]))

        records = []
        for (group, session) in sorted(pairs):
            frames = self._load_session_frames(group, session, max_frames=500)
            feats  = self._extract_session_features(frames)

            dfv = DailyFeatureVector(
                subject_id = f"ehome_G{group}",
                date       = f"session_{session:02d}",
                source     = "ehome",
                posture_fall_events      = feats.get("posture_fall_events"),
                posture_instability_score= feats.get("posture_instability_score"),
            )
            dfv.compute_data_quality()
            records.append(dfv.to_dict())

        return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# SisFall / KFall Bridge
# ---------------------------------------------------------------------------

class SisFallKFallBridge:
    """
    Bridges SisFall and KFall session-level window features into synthetic weekly
    sequences suitable for longitudinal pretraining.

    ARCHITECTURE.md §9 note:
    This is an explicit, documented workaround. SisFall/KFall subjects have no
    multi-day longitudinal structure. We concatenate their per-session windowed
    features into synthetic 7-day sequences by treating each subject's N sessions
    as N consecutive days. This limitation is stated plainly in PREPROCESSING_REPORT.md.
    """

    def __init__(self):
        pass

    def build_from_extracted_features(self, features: List[Dict], dataset_name: str) -> pd.DataFrame:
        """
        Given a list of extracted gait feature dicts (from run_epic3.py-style extraction),
        group by subject_id and build synthetic day-sequences.
        """
        records = []
        # Group features by subject
        by_subject: Dict[str, List[Dict]] = {}
        for feat in features:
            sid = feat.get("subject_id", "unknown")
            by_subject.setdefault(sid, []).append(feat)

        for sid, subject_feats in by_subject.items():
            for day_idx, window in enumerate(subject_feats):
                dfv = DailyFeatureVector(
                    subject_id       = sid,
                    date             = f"synthetic_day_{day_idx:03d}",
                    source           = "sisfall_kfall",
                    gait_cadence     = _safe_float(window.get("cadence")),
                    stride_variability= _safe_float(window.get("stride_variability")),
                    sway_index       = _safe_float(window.get("sway_index")),
                )
                dfv.compute_data_quality()
                records.append(dfv.to_dict())

        return pd.DataFrame(records)

    def build_from_loaders(self, n_subjects_per_dataset: int = 5) -> pd.DataFrame:
        """
        Build a bridge dataset by running the gait extractor on a sample
        of SisFall and KFall subjects and creating synthetic day sequences.
        """
        import sys
        sys.path.insert(0, "D:/Mimamori")
        from data.loaders.loaders import SisFallLoader, KFallLoader
        from preprocess.gait import GaitFeatureExtractor
        import data.splits as splits_module

        extractor = GaitFeatureExtractor()
        frozen = splits_module.get_frozen_splits()
        records = []

        datasets = [
            ("SisFall",  SisFallLoader(), "sisfall", 200, frozen.get("SisFall", {})),
            ("KFall",    KFallLoader(),   "kfall",   100, frozen.get("KFall", {})),
        ]

        for ds_name, loader, src_name, hz, splits in datasets:
            all_subjects = list(splits.get("train", []) + splits.get("val", []) + splits.get("test", []))
            for subject in all_subjects[:n_subjects_per_dataset]:
                try:
                    if ds_name == "SisFall":
                        trials = loader.list_trials(subject)
                        if not trials:
                            continue
                        df_raw = loader.load_trial(trials[0])
                    else:
                        import glob as glob_mod
                        sensor_path = os.path.join(loader.sensor_dir, subject)
                        if not os.path.exists(sensor_path):
                            sensor_path = os.path.join(loader.elder_dir, subject)
                        csvs = glob_mod.glob(os.path.join(sensor_path, "*.csv"))
                        if not csvs:
                            continue
                        df_raw = loader.load_sensor_csv(csvs[0])

                    df_std = extractor.standardize_columns(df_raw, src_name)
                    df_res = extractor.resample(df_std, original_hz=hz)
                    windows = extractor.window_signal(df_res)

                    for day_idx, w in enumerate(windows):
                        feats = extractor.extract_features(w)
                        dfv = DailyFeatureVector(
                            subject_id        = f"{ds_name}_{subject}",
                            date              = f"synthetic_day_{day_idx:03d}",
                            source            = "sisfall_kfall",
                            gait_cadence      = feats.get("cadence"),
                            stride_variability = feats.get("stride_variability"),
                            sway_index        = feats.get("sway_index"),
                        )
                        dfv.compute_data_quality()
                        records.append(dfv.to_dict())

                except Exception as e:
                    warnings.warn(f"[{ds_name}] Subject {subject} failed: {e}")

        return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Sleep Proxy Estimator
# ---------------------------------------------------------------------------

class SleepProxyEstimator:
    """
    Derives a normalised sleep proxy score from two sources:
      1. NHATS self-reported sleep items (hc{r}aslep30mn, hc{r}sleepmed)
      2. Sounds of Home nighttime audio inactivity (silence score during 22:00-06:00)

    LOWER CONFIDENCE: No direct sleep sensor. Documented in PREPROCESSING_REPORT.md
    and feature_schema.json.
    """

    # Hours classified as nighttime
    NIGHTTIME_HOURS = set(range(22, 24)) | set(range(0, 7))

    def estimate_from_nhats_row(self, row: pd.Series, round_num: int) -> Optional[float]:
        """Derive a 0-1 sleep proxy from a single NHATS SP row."""
        r = round_num
        scores = []

        sleep_col = f"hc{r}aslep30mn"
        if sleep_col in row.index and not pd.isna(row[sleep_col]):
            # 1 = problems sleeping → bad (0.2), 2 = no problems → good (0.8)
            val = float(row[sleep_col])
            scores.append(0.2 if val == 1.0 else (0.8 if val == 2.0 else np.nan))

        med_col = f"hc{r}sleepmed"
        if med_col in row.index and not pd.isna(row[med_col]):
            # Taking sleep medication (1=yes) is a mild negative signal → 0.4
            val = float(row[med_col])
            scores.append(0.4 if val == 1.0 else (0.7 if val == 2.0 else np.nan))

        valid = [s for s in scores if not np.isnan(s)]
        return float(np.mean(valid)) if valid else None

    def estimate_from_audio_night_silence(self, session_df: pd.DataFrame) -> Optional[float]:
        """
        Estimate nighttime quietness from a Sounds of Home session feature DataFrame.
        Uses the silence column's mean during annotated nighttime hours.
        Since SoH files don't carry hour-of-day metadata, we use the first/last
        20% of frames as a nighttime proxy for files beginning/ending at night.
        """
        if "silence" not in session_df.columns or len(session_df) < 10:
            return None
        # Use tails of the session as nighttime proxy
        n_tail = max(1, len(session_df) // 5)
        nighttime_frames = pd.concat([
            session_df.head(n_tail),
            session_df.tail(n_tail),
        ])
        return float(nighttime_frames["silence"].mean())


# ---------------------------------------------------------------------------
# Routine Deviation Estimator
# ---------------------------------------------------------------------------

class RoutineDeviationEstimator:
    """
    Computes routine_deviation as the rolling 7-day standard deviation of
    a simple daily activity metric (steps_per_day or total_activity_counts),
    normalised by the rolling mean to make it household-scale-invariant.
    """

    def compute(self, df: pd.DataFrame, activity_col: str = "steps_per_day",
                window: int = 7) -> pd.Series:
        """
        Returns a Series of routine_deviation values aligned to df's index.
        Higher = more irregular daily routine.
        """
        if activity_col not in df.columns:
            return pd.Series(np.nan, index=df.index)

        acts = df[activity_col].copy()
        rolling_mean = acts.rolling(window, min_periods=1).mean()
        rolling_std  = acts.rolling(window, min_periods=1).std(ddof=0)
        # Coefficient of variation (std / mean), clipped to [0, 10]
        deviation = (rolling_std / (rolling_mean + 1e-6)).clip(0, 10)
        return deviation


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_float(val) -> Optional[float]:
    """Convert a value to float, returning None for invalid/negative codes."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return None
    try:
        f = float(val)
        # NHATS uses negative codes for missing (-1, -7, -8, -9)
        return None if f < 0 else f
    except (ValueError, TypeError):
        return None
