"""
tf66_labels.py — TF-66 Thermal Fall Dataset Metadata Parser

Parses TF-66.xlsx into a clean, structured DataFrame for use as the
fall/non-fall label taxonomy across the thermal preprocessing pipeline.
"""

import pandas as pd
import os
from typing import Optional


class TF66LabelParser:
    """
    Parses the TF-66 metadata spreadsheet into a clean label table.

    TF-66 contains 66 volunteer recordings annotated with:
    - Sample ID, Volunteer ID, Recording Name
    - Fall / NonFall label
    - Start frame, first fall frame, last frame
    - Action description, Room ID, Room Height

    NOTE: TF-66.xlsx is a metadata file. The actual thermal video files
    are not bundled with this spreadsheet. This parser provides the label
    schema used for training the thermal classifier.
    """

    LABEL_COL = "Type of Recording (Fall/Non Fall)"
    ACTION_COL = "Action taken in video (ex. walking through frame. falling off chair. slipping on carpet and falling. sleeping)"

    def __init__(self, xlsx_path: str = "D:/Mimamori/TF-66.xlsx"):
        self.xlsx_path = xlsx_path
        self._df: Optional[pd.DataFrame] = None

    def load(self) -> pd.DataFrame:
        """Load and clean the TF-66 spreadsheet. Returns a normalized DataFrame."""
        if not os.path.exists(self.xlsx_path):
            raise FileNotFoundError(f"TF-66 spreadsheet not found at: {self.xlsx_path}")

        raw = pd.read_excel(self.xlsx_path, skiprows=2)

        # Keep only rows that have a valid label (Fall or NonFall)
        raw = raw[raw[self.LABEL_COL].isin(["Fall", "NonFall"])].copy()
        raw = raw.reset_index(drop=True)

        df = pd.DataFrame({
            "sample_id":       pd.to_numeric(raw["Sample ID"], errors="coerce"),
            "volunteer_id":    raw["Volunteer ID"].astype(str).str.strip().str.zfill(2),
            "recording_name":  raw["Recording Name"].astype(str).str.strip(),
            "duration_sec":    pd.to_numeric(raw["Length of Video (sec)"], errors="coerce"),
            "is_fall":         (raw[self.LABEL_COL] == "Fall").astype(int),
            "action":          raw[self.ACTION_COL].astype(str).str.strip(),
            "start_frame":     pd.to_numeric(raw["Start Frame of Video"], errors="coerce"),
            "fall_frame":      pd.to_numeric(raw["First Fall Frame of Video"], errors="coerce"),
            "last_frame":      pd.to_numeric(raw["Last Frame of Video"], errors="coerce"),
            "frames_before_fall": pd.to_numeric(raw["framesBeforeFall"], errors="coerce"),
            "frames_after_fall":  pd.to_numeric(raw["framesAfterFall"], errors="coerce"),
            "room_id":         raw["Room ID Video was recorded in"].astype(str).str.strip(),
            "room_height_ft":  pd.to_numeric(raw["Room Height"], errors="coerce"),
        })

        self._df = df
        return df

    @property
    def labels(self) -> pd.DataFrame:
        if self._df is None:
            self.load()
        return self._df

    def fall_recordings(self) -> pd.DataFrame:
        return self.labels[self.labels["is_fall"] == 1].reset_index(drop=True)

    def non_fall_recordings(self) -> pd.DataFrame:
        return self.labels[self.labels["is_fall"] == 0].reset_index(drop=True)

    def summary(self) -> dict:
        df = self.labels
        return {
            "total_recordings":     len(df),
            "fall_recordings":      int(df["is_fall"].sum()),
            "non_fall_recordings":  int((df["is_fall"] == 0).sum()),
            "unique_volunteers":    df["volunteer_id"].nunique(),
            "fall_ratio":           round(df["is_fall"].mean(), 4),
        }
