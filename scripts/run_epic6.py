"""
run_epic6.py — Epic 6 End-to-End Aggregation Runner

1. Builds NHATS daily feature vectors  → data/processed/daily_features_nhats.pkl
2. Builds eHomeSeniors daily vectors   → data/processed/daily_features_ehome.pkl
3. Builds SisFall/KFall bridge seqs   → data/processed/sessions_sisfall_kfall.pkl
4. Copies frozen splits                → data/processed/splits.json
5. Saves feature schema                → data/processed/feature_schema.json
6. Merges all sources and schema-validates the combined dataset
"""

import sys, os, json, pickle
import numpy as np
import pandas as pd

sys.path.insert(0, "D:/Mimamori")

from preprocess.aggregate import (
    NHATSAggregator,
    EHomeSeniorsAggregator,
    SisFallKFallBridge,
    RoutineDeviationEstimator,
    validate_schema,
)
import data.splits as splits_module

PROCESSED_DIR = "D:/Mimamori/data/processed"
SCHEMA_SRC    = "D:/Mimamori/data/feature_schema.json"


def main():
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    print("=" * 65)
    print("EPIC 6: DAILY / WEEKLY FEATURE AGGREGATION")
    print("=" * 65)

    # -------------------------------------------------------------------
    # Step 1 — NHATS
    # -------------------------------------------------------------------
    print("\n[1/6] Building NHATS daily feature vectors (Rounds 11-14)...")
    nhats_agg = NHATSAggregator()
    df_nhats = nhats_agg.build_daily_vectors()
    print(f"  NHATS rows: {len(df_nhats):,}  |  Unique subjects: {df_nhats['subject_id'].nunique():,}")

    # Add routine_deviation within each subject
    rde = RoutineDeviationEstimator()
    df_nhats = df_nhats.sort_values(["subject_id","round"]).reset_index(drop=True)
    df_nhats["routine_deviation"] = (
        df_nhats.groupby("subject_id", group_keys=False)
        .apply(lambda g: rde.compute(g, activity_col="steps_per_day"))
        .reset_index(drop=True)
    )

    out_nhats = os.path.join(PROCESSED_DIR, "daily_features_nhats.pkl")
    df_nhats.to_pickle(out_nhats)
    print(f"  Saved -> {out_nhats}  ({os.path.getsize(out_nhats)/1024:.1f} KB)")

    # -------------------------------------------------------------------
    # Step 2 — eHomeSeniors
    # -------------------------------------------------------------------
    print("\n[2/6] Building eHomeSeniors daily vectors...")
    ehome_agg = EHomeSeniorsAggregator()
    df_ehome = ehome_agg.build_daily_vectors()
    print(f"  eHomeSeniors rows: {len(df_ehome)}  |  Subjects: {df_ehome['subject_id'].nunique()}")

    out_ehome = os.path.join(PROCESSED_DIR, "daily_features_ehome.pkl")
    df_ehome.to_pickle(out_ehome)
    print(f"  Saved -> {out_ehome}  ({os.path.getsize(out_ehome)/1024:.1f} KB)")

    # -------------------------------------------------------------------
    # Step 3 — SisFall/KFall Bridge
    # -------------------------------------------------------------------
    print("\n[3/6] Building SisFall/KFall synthetic weekly sequences...")
    bridge = SisFallKFallBridge()
    df_sessions = bridge.build_from_loaders(n_subjects_per_dataset=5)
    print(f"  Bridge rows: {len(df_sessions)}  |  Subjects: {df_sessions['subject_id'].nunique()}")

    out_sessions = os.path.join(PROCESSED_DIR, "sessions_sisfall_kfall.pkl")
    df_sessions.to_pickle(out_sessions)
    print(f"  Saved -> {out_sessions}  ({os.path.getsize(out_sessions)/1024:.1f} KB)")

    # -------------------------------------------------------------------
    # Step 4 — Frozen splits
    # -------------------------------------------------------------------
    print("\n[4/6] Saving frozen subject-level splits...")
    frozen = splits_module.get_frozen_splits()
    out_splits = os.path.join(PROCESSED_DIR, "splits.json")
    with open(out_splits, "w") as f:
        json.dump(frozen, f, indent=2)
    print(f"  Saved -> {out_splits}")

    # -------------------------------------------------------------------
    # Step 5 — Feature schema
    # -------------------------------------------------------------------
    print("\n[5/6] Copying feature schema...")
    if os.path.exists(SCHEMA_SRC):
        import shutil
        out_schema = os.path.join(PROCESSED_DIR, "feature_schema.json")
        shutil.copy(SCHEMA_SRC, out_schema)
        print(f"  Saved -> {out_schema}")
    else:
        print("  WARNING: feature_schema.json not found at", SCHEMA_SRC)

    # -------------------------------------------------------------------
    # Step 6 — Combined validation
    # -------------------------------------------------------------------
    print("\n[6/6] Validating combined dataset schema...")
    df_all = pd.concat([df_nhats, df_ehome, df_sessions], ignore_index=True)
    print(f"  Combined rows: {len(df_all):,}  |  Sources: {df_all['source'].unique().tolist()}")

    passed, issues = validate_schema(df_all, nan_threshold=0.95)  # 95% — most fields are cross-source NaN by design
    if issues:
        print("  Schema issues:")
        for iss in issues:
            print(f"    [WARNING] {iss}")
    else:
        print("  Schema validation PASSED.")

    # Per-column NaN summary
    print("\n  NaN rates per numeric field:")
    from preprocess.aggregate import NUMERIC_COLUMNS
    for col in sorted(NUMERIC_COLUMNS):
        if col in df_all.columns:
            nan_pct = df_all[col].isna().mean() * 100
            bar = "=" * int(nan_pct / 5) + "-" * (20 - int(nan_pct / 5))
            print(f"    {col:<35} {bar} {nan_pct:.1f}%")

    # Source distribution
    print("\n  Row distribution by source:")
    for src, cnt in df_all["source"].value_counts().items():
        print(f"    {src:<20}: {cnt:,} rows")

    print("\n" + "=" * 65)
    print("Epic 6 aggregation complete. All outputs saved to data/processed/")
    print("=" * 65)


if __name__ == "__main__":
    main()
