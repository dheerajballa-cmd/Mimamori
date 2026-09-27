"""
pretrain/dataset.py — Epic 7.3

Dataset loader for NHATS + eHomeSeniors daily-feature sequences.
Applies the subject-level splits from data/processed/splits.json.

Primary corpus:  NHATS  (real longitudinal aging data — multi-round sequences)
Secondary corpus: eHomeSeniors (multi-day smart-home routine structure)

Design notes:
- Each subject becomes one or more fixed-length sequences of daily feature vectors.
- Missing (NaN) feature values are imputed with the per-feature training-set mean
  computed before any validation data is touched (to avoid leakage).
- eHomeSeniors sequences are up-sampled using a weighted sampler so they contribute
  ~20% of each training batch (configurable via config["data"]["ehome_sample_weight"]).
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler


# ── Feature columns ───────────────────────────────────────────────────────────

DEFAULT_FEATURE_COLS = [
    "steps_per_day",
    "active_minutes_per_day",
    "total_activity_counts",
    "routine_deviation",
    "data_quality",
]

# ── Dataset class ─────────────────────────────────────────────────────────────

class DailySequenceDataset(Dataset):
    """Sequence dataset for self-supervised pretraining.

    Each item is a (T × C) tensor of normalised daily feature values, where T
    is the sequence length and C is the number of feature columns.

    Sequences shorter than max_seq_len are zero-padded on the right with a
    boolean padding mask returned alongside the data.
    """

    def __init__(
        self,
        sequences: list[np.ndarray],          # each: (T, C)  float32
        status_labels: list[Optional[float]],  # per-sequence functional status (may be NaN)
        max_seq_len: int = 4,
        source_tag: str = "nhats",
    ) -> None:
        self.sequences = sequences
        self.status_labels = status_labels
        self.max_seq_len = max_seq_len
        self.source_tag = source_tag

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> dict:
        seq = self.sequences[idx]        # (T, C) float32
        T, C = seq.shape
        label = self.status_labels[idx]

        # Pad to max_seq_len
        pad_len = self.max_seq_len - T
        if pad_len > 0:
            pad = np.zeros((pad_len, C), dtype=np.float32)
            seq = np.concatenate([seq, pad], axis=0)

        # Padding mask: True = valid time-step
        pad_mask = torch.zeros(self.max_seq_len, dtype=torch.bool)
        pad_mask[:T] = True

        return {
            "x": torch.tensor(seq, dtype=torch.float32),  # (T, C)
            "pad_mask": pad_mask,                          # (T,) bool
            "label": torch.tensor(label if not np.isnan(label) else -1.0, dtype=torch.float32),
            "source": self.source_tag,
        }


# ── Preprocessing helpers ─────────────────────────────────────────────────────

def _extract_sequences(
    df: pd.DataFrame,
    feature_cols: list[str],
    subject_col: str,
    time_col: str,
    status_col: str,
    max_seq_len: int,
    min_seq_len: int,
    fill_means: Optional[dict[str, float]] = None,
    source_tag: str = "nhats",
) -> tuple[list[np.ndarray], list[float], dict[str, float]]:
    """Convert a flat DataFrame into per-subject sequences.

    Args:
        df: flat daily-feature DataFrame
        feature_cols: columns to include in the feature vector
        subject_col: column used to group by subject
        time_col: column used to sort within subject
        status_col: column containing functional-status label
        max_seq_len: truncate sequences longer than this
        min_seq_len: discard sequences shorter than this
        fill_means: if provided, use these means for NaN imputation (test/val);
                    if None, compute from df (train split)
        source_tag: tag attached to each sequence item

    Returns:
        (sequences, status_labels, fill_means)
        - sequences: list of (T, C) float32 arrays
        - status_labels: list of per-subject floats (NaN if unavailable)
        - fill_means: dict of {col: mean} used for imputation
    """
    # Ensure all feature cols exist (fill with NaN if missing in this source)
    for col in feature_cols:
        if col not in df.columns:
            df[col] = np.nan

    # Convert feature cols to numeric
    for col in feature_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Compute or reuse fill means
    if fill_means is None:
        fill_means = {}
        for col in feature_cols:
            m = df[col].mean()
            fill_means[col] = float(m) if not np.isnan(m) else 0.0

    sequences = []
    status_labels = []

    for subject_id, grp in df.groupby(subject_col, sort=False):
        grp = grp.sort_values(time_col, ascending=True)
        T = len(grp)

        if T < min_seq_len:
            continue

        # Truncate
        grp = grp.head(max_seq_len)
        T = len(grp)

        # Build feature matrix
        feat = grp[feature_cols].copy()
        for col in feature_cols:
            feat[col] = feat[col].fillna(fill_means[col])

        seq = feat.values.astype(np.float32)  # (T, C)
        sequences.append(seq)

        # Status label: use first non-NaN value for the subject
        if status_col in grp.columns:
            label_vals = pd.to_numeric(grp[status_col], errors="coerce").dropna()
            status_labels.append(float(label_vals.iloc[0]) if len(label_vals) > 0 else float("nan"))
        else:
            status_labels.append(float("nan"))

    return sequences, status_labels, fill_means


def _normalise(
    sequences: list[np.ndarray],
    mean: Optional[np.ndarray] = None,
    std: Optional[np.ndarray] = None,
) -> tuple[list[np.ndarray], np.ndarray, np.ndarray]:
    """Z-score normalise sequences.

    Args:
        sequences: list of (T, C) arrays
        mean / std: if provided, use these (for val/test); otherwise compute from data

    Returns:
        (normalised_sequences, mean, std)
    """
    if mean is None or std is None:
        all_data = np.concatenate(sequences, axis=0)  # (N*T, C)
        mean = all_data.mean(axis=0)
        std = all_data.std(axis=0)
        std[std < 1e-8] = 1.0  # avoid div-by-zero for constant columns

    normalised = [(s - mean) / std for s in sequences]
    return normalised, mean, std


# ── Public data loading interface ─────────────────────────────────────────────

def build_dataloaders(
    cfg: dict,
    batch_size: int = 256,
    num_workers: int = 0,
    seed: int = 42,
) -> tuple[DataLoader, DataLoader, dict]:
    """Build train and val DataLoaders from config.

    Returns:
        (train_loader, val_loader, data_stats)
        where data_stats contains {'fill_means', 'norm_mean', 'norm_std',
        'n_train', 'n_val', 'n_nhats', 'n_ehome'}
    """
    dcfg = cfg["data"]
    mcfg = cfg["model"]
    tcfg = cfg["training"]

    feature_cols = dcfg["feature_cols"]
    max_seq_len = dcfg.get("max_seq_len", 4)
    min_seq_len = dcfg.get("min_seq_len", 2)
    status_col = dcfg.get("status_col", "worries_about_falling")
    seq_col = dcfg.get("sequence_col", "subject_id")
    time_col = dcfg.get("time_col", "round")

    rng = np.random.default_rng(seed)

    # ── Load NHATS ────────────────────────────────────────────────────────────
    nhats_df = pd.read_pickle(dcfg["nhats_path"])
    splits_path = Path(dcfg["splits_path"])
    splits = json.loads(splits_path.read_text())

    # splits.json uses SisFall / KFall / Synthetic keys — no NHATS key exists.
    # Perform a deterministic subject-level 85/15 split for NHATS using the
    # frozen seed from config (matches the philosophy of splits.py: subject-level,
    # frozen before feature extraction).
    all_subjects = sorted(nhats_df[seq_col].unique().tolist())  # sort for determinism
    rng_split = np.random.default_rng(seed)
    rng_split.shuffle(all_subjects)
    n_val = max(1, int(len(all_subjects) * tcfg.get("val_fraction", 0.15)))
    nhats_val_subjects = set(all_subjects[:n_val])
    nhats_train_subjects = set(all_subjects[n_val:])

    nhats_train_df = nhats_df[nhats_df[seq_col].isin(nhats_train_subjects)].copy()
    nhats_val_df = nhats_df[nhats_df[seq_col].isin(nhats_val_subjects)].copy()

    # ── Extract NHATS sequences ───────────────────────────────────────────────
    nhats_train_seqs, nhats_train_labels, fill_means = _extract_sequences(
        nhats_train_df, feature_cols, seq_col, time_col, status_col,
        max_seq_len, min_seq_len, source_tag="nhats",
    )
    nhats_val_seqs, nhats_val_labels, _ = _extract_sequences(
        nhats_val_df, feature_cols, seq_col, time_col, status_col,
        max_seq_len, min_seq_len, fill_means=fill_means, source_tag="nhats",
    )

    # ── Load eHomeSeniors (secondary corpus) ──────────────────────────────────
    ehome_train_seqs, ehome_train_labels = [], []
    try:
        ehome_df = pd.read_pickle(dcfg["ehome_path"])
        # eHomeSeniors has very few subjects — use all for training
        ehome_seqs, ehome_labels, _ = _extract_sequences(
            ehome_df.copy(), feature_cols, seq_col, time_col, status_col,
            max_seq_len, min_seq_len, fill_means=fill_means, source_tag="ehome",
        )
        ehome_train_seqs = ehome_seqs
        ehome_train_labels = ehome_labels
    except (FileNotFoundError, KeyError):
        pass  # eHomeSeniors is secondary; continue if missing

    # ── Combine train sequences ───────────────────────────────────────────────
    all_train_seqs = nhats_train_seqs + ehome_train_seqs
    all_train_labels = nhats_train_labels + ehome_train_labels

    # ── Normalise ─────────────────────────────────────────────────────────────
    all_train_seqs, norm_mean, norm_std = _normalise(all_train_seqs)
    nhats_val_seqs, _, _ = _normalise(nhats_val_seqs, norm_mean, norm_std)

    # ── Build datasets ────────────────────────────────────────────────────────
    train_dataset = DailySequenceDataset(all_train_seqs, all_train_labels, max_seq_len)
    val_dataset = DailySequenceDataset(nhats_val_seqs, nhats_val_labels, max_seq_len)

    # ── Weighted sampler: up-weight eHomeSeniors to ~20% of each batch ───────
    ehome_weight = dcfg.get("ehome_sample_weight", 0.2)
    n_nhats = len(nhats_train_seqs)
    n_ehome = len(ehome_train_seqs)
    if n_ehome > 0:
        per_nhats_w = (1.0 - ehome_weight) / n_nhats if n_nhats > 0 else 0.0
        per_ehome_w = ehome_weight / n_ehome
        weights = [per_nhats_w] * n_nhats + [per_ehome_w] * n_ehome
        sampler = WeightedRandomSampler(
            weights=weights,
            num_samples=len(train_dataset),
            replacement=True,
            generator=torch.Generator().manual_seed(seed),
        )
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, sampler=sampler,
            num_workers=num_workers, drop_last=True,
        )
    else:
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, drop_last=True,
        )

    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers,
    )

    data_stats = {
        "fill_means": fill_means,
        "norm_mean": norm_mean.tolist(),
        "norm_std": norm_std.tolist(),
        "n_train": len(train_dataset),
        "n_val": len(val_dataset),
        "n_nhats_train": n_nhats,
        "n_ehome_train": n_ehome,
        "feature_cols": feature_cols,
    }

    return train_loader, val_loader, data_stats
