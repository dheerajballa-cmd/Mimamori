"""
Unified Data Loaders for Mimamori Pipeline.

Provides clean programmatic access to:
- SisFall (9-axis IMU)
- KFall & KFall-elderly (IMU + impact labels)
- NHATS (STATA .dta files: SP, OP, Tracker, Actigraphy)
- eHomeSeniors (Melexis thermal & Omron environmental sensors)
- Sounds of Home (ambient audio event probabilities)
- Mimamori Synthetic Longitudinal (trajectories, causal graphs, outcomes)
- Thermal Fall 66 (metadata & protocol definitions)
"""

import os
import glob
import json
from typing import Dict, List, Optional, Union, Tuple
import pandas as pd
import numpy as np


class SisFallLoader:
    """Loader for SisFall dataset (waist-mounted accelerometer & gyroscope @ 200Hz)."""

    def __init__(self, data_dir: str = "D:/Mimamori/data/SisFall_dataset"):
        self.data_dir = data_dir

    def list_subjects(self) -> List[str]:
        sub_dirs = glob.glob(os.path.join(self.data_dir, "S*"))
        return sorted([os.path.basename(p) for p in sub_dirs if os.path.isdir(p)])

    def list_trials(self, subject_id: str) -> List[str]:
        sub_path = os.path.join(self.data_dir, subject_id)
        return sorted(glob.glob(os.path.join(sub_path, "*.txt")))

    def load_trial(self, file_path: str) -> pd.DataFrame:
        raw_lines = []
        with open(file_path, "r") as f:
            for line in f:
                cleaned = line.strip().rstrip(";").strip()
                if cleaned:
                    parts = [float(x.strip()) for x in cleaned.split(",") if x.strip()]
                    if len(parts) >= 9:
                        raw_lines.append(parts[:9])
        
        columns = [
            "acc1_x", "acc1_y", "acc1_z",
            "gyr_x", "gyr_y", "gyr_z",
            "acc2_x", "acc2_y", "acc2_z"
        ]
        return pd.DataFrame(raw_lines, columns=columns)


class KFallLoader:
    """Loader for KFall and KFall elderly add-in datasets."""

    def __init__(self, data_dir: str = "D:/Mimamori/KFall+elderly_add-in"):
        self.data_dir = data_dir
        self.sensor_dir = os.path.join(data_dir, "sensor_data")
        self.elder_dir = os.path.join(data_dir, "sensor_data_elder")
        self.label_dir = os.path.join(data_dir, "label_data")

    def list_subjects(self, elderly: bool = False) -> List[str]:
        target = self.elder_dir if elderly else self.sensor_dir
        sub_dirs = glob.glob(os.path.join(target, "*"))
        return sorted([os.path.basename(p) for p in sub_dirs if os.path.isdir(p)])

    def load_labels(self, subject_id: str) -> Optional[pd.DataFrame]:
        label_file = os.path.join(self.label_dir, f"{subject_id}_label.xlsx")
        if os.path.exists(label_file):
            return pd.read_excel(label_file)
        return None

    def load_sensor_csv(self, file_path: str) -> pd.DataFrame:
        return pd.read_csv(file_path)


class NHATSLoader:
    """Loader for NHATS longitudinal survey panel and accelerometry actigraphy."""

    def __init__(self, data_dir: str = "D:/Mimamori/NHATS"):
        self.data_dir = data_dir

    def list_available_files(self) -> List[str]:
        files = glob.glob(os.path.join(self.data_dir, "*.dta"))
        return sorted([os.path.basename(f) for f in files])

    def load_round_sp(self, round_num: int) -> pd.DataFrame:
        pattern = os.path.join(self.data_dir, f"NHATS_Round_{round_num}_SP_File*.dta")
        matches = glob.glob(pattern)
        if not matches:
            raise FileNotFoundError(f"No SP file found for NHATS Round {round_num}")
        return pd.read_stata(matches[0], convert_categoricals=False)

    def load_accelerometry(self, round_num: int = 12) -> pd.DataFrame:
        pattern = os.path.join(self.data_dir, f"NHATS_Round_{round_num}_Accel_Det_File*.dta")
        matches = glob.glob(pattern)
        if not matches:
            raise FileNotFoundError(f"No Accelerometry Detail file found for NHATS Round {round_num}")
        return pd.read_stata(matches[0], convert_categoricals=False)

    def load_tracker(self, round_num: int = 14) -> pd.DataFrame:
        pattern = os.path.join(self.data_dir, f"NHATS_Round_{round_num}_Tracker_File*.dta")
        matches = glob.glob(pattern)
        if not matches:
            raise FileNotFoundError(f"No Tracker file found for NHATS Round {round_num}")
        return pd.read_stata(matches[0], convert_categoricals=False)


class EHomeSeniorsLoader:
    """Loader for eHomeSeniors smart home sensor dataset (Melexis Grid-Eye & Omron)."""

    def __init__(self, data_dir: str = "D:/Mimamori/eHomeSeniors dataset"):
        self.data_dir = data_dir
        self.melexis_dir = os.path.join(data_dir, "melexis sensor")
        self.omron_dir = os.path.join(data_dir, "omron sensor")

    def list_melexis_files(self) -> List[str]:
        return sorted(glob.glob(os.path.join(self.melexis_dir, "*.csv")))

    def load_melexis_thermal_stream(self, file_path: str, max_rows: Optional[int] = None) -> List[Dict]:
        frames = []
        with open(file_path, "r") as f:
            for idx, line in enumerate(f):
                if max_rows and idx >= max_rows:
                    break
                line = line.strip()
                if not line:
                    continue
                parts = line.split("|")
                if len(parts) > 10:
                    try:
                        timestamp = parts[0]
                        temps = []
                        for p in parts[1:]:
                            try:
                                val = float(p)
                                temps.append(val)
                            except ValueError:
                                continue
                        if temps:
                            frames.append({"timestamp": timestamp, "thermal_pixels": np.array(temps)})
                    except Exception:
                        continue
        return frames


class SoundsOfHomeLoader:
    """Loader for Sounds of Home ambient audio event inference records."""

    def __init__(self, data_dir: str = "D:/Mimamori/Sounds_of_Home"):
        self.data_dir = data_dir

    def list_subject_dirs(self) -> List[str]:
        sub_dirs = glob.glob(os.path.join(self.data_dir, "*"))
        return sorted([p for p in sub_dirs if os.path.isdir(p)])

    def load_event_json(self, json_path: str) -> List[Dict]:
        with open(json_path, "r") as f:
            return json.load(f)


class SyntheticLongitudinalLoader:
    """Loader for Mimamori Synthetic Longitudinal Causal dataset v0.1."""

    def __init__(self, data_dir: str = "D:/Mimamori/Mimamori_Synthetic_Longitudinal_v0.1"):
        self.data_dir = data_dir

    def load_trajectories(self) -> pd.DataFrame:
        return pd.read_csv(os.path.join(self.data_dir, "daily_trajectories.csv"))

    def load_participants(self) -> pd.DataFrame:
        return pd.read_csv(os.path.join(self.data_dir, "participants.csv"))

    def load_outcomes(self) -> pd.DataFrame:
        return pd.read_csv(os.path.join(self.data_dir, "outcomes.csv"))

    def load_counterfactuals(self) -> pd.DataFrame:
        return pd.read_csv(os.path.join(self.data_dir, "counterfactuals.csv"))

    def load_metadata(self) -> Dict:
        with open(os.path.join(self.data_dir, "metadata.json"), "r") as f:
            return json.load(f)
