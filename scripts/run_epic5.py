"""
run_epic5.py — Epic 5 End-to-End Validation Script

1. Privacy check — confirms no raw audio in Sounds of Home
2. Loads per-subject JSON files using subject-level splits from data/splits.py
3. Extracts windowed audio event features for all subjects
4. Trains AudioEventClassifier on train split, evaluates on val + test splits
5. Prints per-category classification report
"""

import sys
import os
import json
import glob
import numpy as np
import pandas as pd

sys.path.insert(0, "D:/Mimamori")

from preprocess.audio import (
    AudioEventExtractor,
    AudioEventClassifier,
    EVENT_CATEGORIES,
    assert_no_raw_audio,
)
import data.splits as splits_module

SOH_DIR = "D:/Mimamori/Sounds_of_Home"
WINDOW_SIZE = 100     # frames per window (~10s at typical frame rate)
STEP        = 50      # 50% overlap
MAX_FRAMES_PER_FILE = 500   # cap per session file for speed


def load_subject_features(subject_id: str, extractor: AudioEventExtractor) -> pd.DataFrame:
    """
    Load all JSON session files for a subject, extract windowed features.
    Returns a DataFrame of aggregated feature windows.
    """
    subj_dir = os.path.join(SOH_DIR, subject_id.lstrip("0").zfill(2))
    if not os.path.isdir(subj_dir):
        # Try without zero-padding
        subj_dir = os.path.join(SOH_DIR, subject_id)
    if not os.path.isdir(subj_dir):
        return pd.DataFrame()

    json_files = sorted(glob.glob(os.path.join(subj_dir, "*.json")))

    all_windows = []
    for jf in json_files[:5]:   # cap at 5 files per subject for speed
        try:
            session_df = extractor.load_and_extract_json(jf, max_frames=MAX_FRAMES_PER_FILE)
            windowed = extractor.window_features(session_df, window_size=WINDOW_SIZE, step=STEP)
            all_windows.append(windowed)
        except Exception as e:
            print(f"    Warning: could not load {os.path.basename(jf)}: {e}")

    if not all_windows:
        return pd.DataFrame()
    return pd.concat(all_windows, ignore_index=True)


def derive_labels(feature_df: pd.DataFrame) -> np.ndarray:
    """
    Derive proxy labels for each window by finding the category with the
    highest mean probability. This is the self-supervised proxy — ground-truth
    event annotations are not provided in the dataset.
    """
    cat_means = [f"{cat}_mean" for cat in EVENT_CATEGORIES]
    # argmax across the mean columns → category index
    return feature_df[cat_means].values.argmax(axis=1)


def build_split_dataset(subject_ids: list, extractor: AudioEventExtractor, split_name: str):
    """Build (X, y) for a set of subjects."""
    X_parts, y_parts = [], []
    for subj in subject_ids:
        feat_df = load_subject_features(subj, extractor)
        if feat_df.empty:
            print(f"    [{split_name}] No data for subject {subj}, skipping.")
            continue
        y_subj = derive_labels(feat_df)
        X_parts.append(feat_df)
        y_parts.append(y_subj)
        print(f"    [{split_name}] Subject {subj}: {len(feat_df)} windows")

    if not X_parts:
        return pd.DataFrame(), np.array([])
    return pd.concat(X_parts, ignore_index=True), np.concatenate(y_parts)


def main():
    print("=" * 65)
    print("EPIC 5: AMBIENT AUDIO EVENT CLASSIFICATION")
    print("=" * 65)

    # -----------------------------------------------------------------------
    # 5.3 Privacy check
    # -----------------------------------------------------------------------
    print("\n[1/4] Privacy check (5.3)...")
    msg = assert_no_raw_audio(SOH_DIR)
    print(f"  {msg}")

    # -----------------------------------------------------------------------
    # Load frozen splits
    # -----------------------------------------------------------------------
    print("\n[2/4] Loading subject-level splits from data/splits.py...")
    frozen = splits_module.get_frozen_splits()
    soh_splits = frozen.get("Sounds_of_Home", {})

    if not soh_splits:
        print("  ERROR: Sounds_of_Home splits not found. Re-run data/splits.py.")
        sys.exit(1)

    print(f"  Train: {len(soh_splits['train'])} subjects | "
          f"Val: {len(soh_splits['val'])} subjects | "
          f"Test: {len(soh_splits['test'])} subjects")

    # -----------------------------------------------------------------------
    # Feature extraction
    # -----------------------------------------------------------------------
    print("\n[3/4] Extracting audio event features...")
    extractor = AudioEventExtractor()

    print("  Building TRAIN set...")
    X_train, y_train = build_split_dataset(soh_splits["train"], extractor, "TRAIN")
    print("  Building VAL set...")
    X_val, y_val = build_split_dataset(soh_splits["val"], extractor, "VAL")
    print("  Building TEST set...")
    X_test, y_test = build_split_dataset(soh_splits["test"], extractor, "TEST")

    print(f"\n  Train windows: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

    if len(X_train) < 5:
        print("  ERROR: Insufficient training data.")
        sys.exit(1)

    # -----------------------------------------------------------------------
    # Train & evaluate
    # -----------------------------------------------------------------------
    print("\n[4/4] Training AudioEventClassifier (Random Forest)...")
    clf = AudioEventClassifier(n_estimators=100)
    clf.fit(X_train, y_train)

    # Evaluate on val + test combined
    X_eval = pd.concat([X_val, X_test], ignore_index=True)
    y_eval = np.concatenate([y_val, y_test])

    print("\n  Classification Report (Val + Test):")
    report = clf.evaluate(X_eval, y_eval)
    for line in report.split("\n"):
        print(f"    {line}")

    # Dominant category distribution across train set
    unique, counts = np.unique(y_train, return_counts=True)
    print("\n  Train label distribution:")
    for u, c in zip(unique, counts):
        pct = 100 * c / len(y_train)
        print(f"    {EVENT_CATEGORIES[u]:<12}: {c:4d} windows ({pct:.1f}%)")

    print("\n" + "=" * 65)
    print("Epic 5 validation complete. All steps passed.")
    print("=" * 65)


if __name__ == "__main__":
    main()
