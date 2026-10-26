"""
causal/simulate.py — Epic 9.2

Structural Causal Model (SCM) for Mimamori fall-risk attribution.

SCM Node Definitions (Epic 9.1):
  sleep_quality             — nightly sleep quality (0–1 scale)
  routine_deviation         — deviation from established daily activity routine (0–1)
  gait_stability            — composite gait health (higher = better, 0–1)
  medication_adherence_proxy — self-care regularity proxy (higher = more adherent, 0–1)
  fall_risk                 — cumulative fall probability (0–1)

Causal graph edges (hand-specified per ARCHITECTURE.md Section 6):
  sleep_quality       -> gait_stability
  routine_deviation   -> gait_stability
  routine_deviation   -> sleep_quality
  medication_adherence_proxy -> gait_stability
  medication_adherence_proxy -> routine_deviation
  gait_stability      -> fall_risk
  sleep_quality       -> fall_risk       (direct path, weaker)

Root causes (exogenous / interventional handles):
  sleep_quality:              driven by u_sleep (Gaussian noise + age effect)
  routine_deviation:          driven by u_routine (Gaussian noise)
  medication_adherence_proxy: driven by u_med (Gaussian noise + frailty effect)

This SCM intentionally has 5 observable nodes (matching tasks.md 9.1) and
produces multi-week household trajectories that can be fed to DoWhy and
causal-learn in Epic 10.

Usage:
    python -m causal.simulate --config configs/synthetic_scm.yaml
    python causal/simulate.py          # also works standalone
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# SCM_NODES defines the 5 observable causal nodes (matches tasks.md 9.1)
SCM_NODES = [
    "sleep_quality",
    "routine_deviation",
    "gait_stability",
    "medication_adherence_proxy",
    "fall_risk",
]

SCM_EDGES: List[Tuple[str, str]] = [
    ("sleep_quality",              "gait_stability"),
    ("routine_deviation",          "gait_stability"),
    ("routine_deviation",          "sleep_quality"),
    ("medication_adherence_proxy", "gait_stability"),
    ("medication_adherence_proxy", "routine_deviation"),
    ("gait_stability",             "fall_risk"),
    ("sleep_quality",              "fall_risk"),
]

# Structural equation coefficients (hand-specified, documented in causal/README.md)
COEFF = {
    # sleep_quality = f(routine_deviation, u_sleep)
    "sleep_quality": {
        "routine_deviation": -0.25,
        "intercept":          0.75,
        "noise_std":          0.08,
    },
    # routine_deviation = f(medication_adherence_proxy, u_routine)
    "routine_deviation": {
        "medication_adherence_proxy": -0.30,
        "intercept":                   0.25,
        "noise_std":                   0.07,
    },
    # gait_stability = f(sleep_quality, routine_deviation, medication_adherence_proxy, u_gait)
    "gait_stability": {
        "sleep_quality":               0.35,
        "routine_deviation":          -0.20,
        "medication_adherence_proxy":  0.25,
        "intercept":                   0.30,
        "noise_std":                   0.06,
    },
    # medication_adherence_proxy = f(age_factor, u_med)
    "medication_adherence_proxy": {
        "age_factor":  -0.05,
        "intercept":    0.80,
        "noise_std":    0.09,
    },
    # fall_risk = f(gait_stability, sleep_quality, age_factor, u_fall)
    "fall_risk": {
        "gait_stability": -0.50,
        "sleep_quality":  -0.15,
        "age_factor":      0.10,
        "intercept":       0.50,
        "noise_std":       0.04,
    },
}

# Scenario-specific perturbations applied to exogenous noise
SCENARIO_PERTURBATIONS = {
    "stable":            {},
    "sleep_driven":      {"u_sleep_shift":   0.25},
    "routine_driven":    {"u_routine_shift":  0.30},
    "medication_driven": {"u_med_shift":     -0.35},
    "mixed": {
        "u_sleep_shift":   0.15,
        "u_routine_shift":  0.15,
        "u_med_shift":     -0.20,
    },
    "activity_driven":   {"u_gait_shift": -0.20},
}

# Ground-truth root cause per scenario (used for top-k evaluation in Epic 10)
GROUND_TRUTH_ROOT_CAUSE = {
    "stable":            None,
    "sleep_driven":      "sleep_quality",
    "routine_driven":    "routine_deviation",
    "medication_driven": "medication_adherence_proxy",
    "mixed":             "sleep_quality",
    "activity_driven":   "gait_stability",
}


def _clip01(x: float) -> float:
    return float(np.clip(x, 0.0, 1.0))


def simulate_one_day(
    prev: Dict[str, float],
    age_factor: float,
    scenario: str,
    rng: np.random.Generator,
    day: int,
    n_days: int,
) -> Dict[str, float]:
    """Simulate one day of SCM observations with AR(1) temporal smoothing."""
    perturbation_ramp = min(1.0, day / (n_days * 0.1))
    pert = SCENARIO_PERTURBATIONS.get(scenario, {})
    AR = 0.70  # auto-correlation mixing

    # medication_adherence_proxy
    c = COEFF["medication_adherence_proxy"]
    med_raw = (
        c["intercept"]
        + c["age_factor"] * age_factor
        + rng.normal(0, c["noise_std"])
        + pert.get("u_med_shift", 0.0) * perturbation_ramp
    )
    med_val = _clip01(AR * prev["medication_adherence_proxy"] + (1 - AR) * med_raw)

    # routine_deviation
    c = COEFF["routine_deviation"]
    rou_raw = (
        c["intercept"]
        + c["medication_adherence_proxy"] * med_val
        + rng.normal(0, c["noise_std"])
        + pert.get("u_routine_shift", 0.0) * perturbation_ramp
    )
    rou_val = _clip01(AR * prev["routine_deviation"] + (1 - AR) * rou_raw)

    # sleep_quality
    c = COEFF["sleep_quality"]
    slp_raw = (
        c["intercept"]
        + c["routine_deviation"] * rou_val
        + rng.normal(0, c["noise_std"])
        + pert.get("u_sleep_shift", 0.0) * perturbation_ramp
    )
    slp_val = _clip01(AR * prev["sleep_quality"] + (1 - AR) * slp_raw)

    # gait_stability
    c = COEFF["gait_stability"]
    gait_raw = (
        c["intercept"]
        + c["sleep_quality"] * slp_val
        + c["routine_deviation"] * rou_val
        + c["medication_adherence_proxy"] * med_val
        + rng.normal(0, c["noise_std"])
        + pert.get("u_gait_shift", 0.0) * perturbation_ramp
    )
    gait_val = _clip01(AR * prev["gait_stability"] + (1 - AR) * gait_raw)

    # fall_risk
    c = COEFF["fall_risk"]
    risk_raw = (
        c["intercept"]
        + c["gait_stability"] * gait_val
        + c["sleep_quality"] * slp_val
        + c["age_factor"] * age_factor
        + rng.normal(0, c["noise_std"])
    )
    risk_val = _clip01(risk_raw)

    return {
        "sleep_quality":              slp_val,
        "routine_deviation":          rou_val,
        "gait_stability":             gait_val,
        "medication_adherence_proxy": med_val,
        "fall_risk":                  risk_val,
    }


def simulate_participant(
    participant_id: str,
    age: int,
    sex: str,
    scenario: str,
    n_days: int,
    seed: int,
) -> pd.DataFrame:
    """Simulate a full multi-week trajectory for one participant."""
    rng = np.random.default_rng(seed)
    age_factor = float(np.clip((age - 60) / 30.0, 0.0, 1.0))

    prev = {
        "sleep_quality":              _clip01(0.75 + rng.normal(0, 0.05)),
        "routine_deviation":          _clip01(0.15 + rng.normal(0, 0.03)),
        "gait_stability":             _clip01(0.70 + rng.normal(0, 0.05)),
        "medication_adherence_proxy": _clip01(0.85 + rng.normal(0, 0.04)),
        "fall_risk":                  _clip01(0.10 + rng.normal(0, 0.02)),
    }

    rows = []
    for day in range(1, n_days + 1):
        obs = simulate_one_day(prev, age_factor, scenario, rng, day, n_days)
        rows.append({
            "participant_id":  participant_id,
            "day":             day,
            "age":             age,
            "sex":             sex,
            "causal_scenario": scenario,
            **obs,
        })
        prev = obs

    return pd.DataFrame(rows)


def generate_dataset(
    n_participants: int = 200,
    n_days: int = 90,
    seed: int = 42,
    scenario_weights: Dict[str, float] = None,
) -> Tuple[pd.DataFrame, Dict]:
    """Generate a full synthetic dataset of household trajectories."""
    if scenario_weights is None:
        scenario_weights = {
            "stable":            0.20,
            "sleep_driven":      0.20,
            "routine_driven":    0.15,
            "medication_driven": 0.20,
            "mixed":             0.15,
            "activity_driven":   0.10,
        }

    rng_main = np.random.default_rng(seed)
    scenarios = list(scenario_weights.keys())
    probs = np.array([scenario_weights[s] for s in scenarios])
    probs /= probs.sum()

    participant_scenarios = rng_main.choice(scenarios, size=n_participants, p=probs)
    ages = rng_main.integers(65, 90, size=n_participants)
    sexes = rng_main.choice(["M", "F"], size=n_participants)

    frames = []
    for i in range(n_participants):
        pid = f"SYN{i+1:04d}"
        df_p = simulate_participant(
            participant_id=pid,
            age=int(ages[i]),
            sex=sexes[i],
            scenario=participant_scenarios[i],
            n_days=n_days,
            seed=seed + i * 17,
        )
        frames.append(df_p)

    df_all = pd.concat(frames, ignore_index=True)
    df_all["ground_truth_root_cause"] = df_all["causal_scenario"].map(GROUND_TRUTH_ROOT_CAUSE)

    metadata = {
        "n_participants": n_participants,
        "n_days_per_participant": n_days,
        "n_observations": len(df_all),
        "seed": seed,
        "scm_nodes": SCM_NODES,
        "scm_edges": SCM_EDGES,
        "scm_coefficients": COEFF,
        "scenario_distribution": {
            s: int((participant_scenarios == s).sum()) for s in scenarios
        },
        "ground_truth_root_cause_map": GROUND_TRUTH_ROOT_CAUSE,
        "important_note": "Fully synthetic SCM-generated data. Not clinical measurements.",
    }

    return df_all, metadata


def split_by_participant(
    df: pd.DataFrame,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Subject-level split (no data leakage)."""
    pids = df["participant_id"].unique()
    rng = np.random.default_rng(seed)
    rng.shuffle(pids)

    n_val  = max(1, int(len(pids) * val_frac))
    n_test = max(1, int(len(pids) * test_frac))

    val_pids   = set(pids[:n_val])
    test_pids  = set(pids[n_val:n_val + n_test])
    train_pids = set(pids[n_val + n_test:])

    return (
        df[df["participant_id"].isin(train_pids)].reset_index(drop=True),
        df[df["participant_id"].isin(val_pids)].reset_index(drop=True),
        df[df["participant_id"].isin(test_pids)].reset_index(drop=True),
    )


def main(out_dir: str = "causal/checkpoints", n_participants: int = 200,
         n_days: int = 90, seed: int = 42):
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Epic 9.2 — Generating Synthetic SCM Trajectories")
    print("=" * 60)
    print(f"  Participants : {n_participants}")
    print(f"  Days/subject : {n_days}")
    print(f"  Seed         : {seed}")

    df_all, metadata = generate_dataset(
        n_participants=n_participants, n_days=n_days, seed=seed
    )
    df_train, df_val, df_test = split_by_participant(df_all, seed=seed)

    print(f"\nTotal observations : {len(df_all):,}")
    print(f"Train participants : {df_train['participant_id'].nunique()}")
    print(f"Val   participants : {df_val['participant_id'].nunique()}")
    print(f"Test  participants : {df_test['participant_id'].nunique()}")
    print("\nScenario distribution:")
    for sc, cnt in metadata["scenario_distribution"].items():
        gt = GROUND_TRUTH_ROOT_CAUSE.get(sc) or "None"
        print(f"  {sc:<22} {cnt:>3} participants  [root cause: {gt}]")

    # Save splits
    for split_name, split_df in [("train", df_train), ("val", df_val), ("test", df_test)]:
        path = out_path / f"synthetic_trajectories_{split_name}.pkl"
        split_df.to_pickle(str(path))
        print(f"Saved {split_name}: {path}")

    # Save SCM ground truth JSON
    scm_gt = {
        "nodes": SCM_NODES,
        "edges": SCM_EDGES,
        "structural_equations": COEFF,
        "ground_truth_root_cause_per_scenario": GROUND_TRUTH_ROOT_CAUSE,
        "scenario_perturbations": {k: v for k, v in SCENARIO_PERTURBATIONS.items()},
        "description": (
            "Hand-specified SCM for Mimamori fall-risk attribution. "
            "Edges encode causal mechanisms; coefficients encode effect strengths. "
            "See causal/README.md for full design rationale."
        ),
    }
    gt_path = out_path / "scm_ground_truth.json"
    with open(gt_path, "w", encoding="utf-8") as f:
        json.dump(scm_gt, f, indent=2, default=str)
    print(f"Saved SCM ground truth: {gt_path}")

    # Save metadata
    meta_path = out_path / "causal_metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, default=str)
    print(f"Saved metadata: {meta_path}")

    print("\n[DONE] Epic 9.2 complete.")
    return df_train, df_val, df_test, metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Mimamori Epic 9.2 SCM Simulator")
    parser.add_argument("--out_dir",        default="causal/checkpoints")
    parser.add_argument("--n_participants", type=int, default=200)
    parser.add_argument("--n_days",         type=int, default=90)
    parser.add_argument("--seed",           type=int, default=42)
    args = parser.parse_args()
    main(args.out_dir, args.n_participants, args.n_days, args.seed)
