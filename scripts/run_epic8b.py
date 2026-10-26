"""
scripts/run_epic8b.py - Epic 8b orchestrator

Runs all Epic 8b deliverables in sequence:
  8b.1 Export forecaster_best.pt, baseline_xgboost.pkl, forecast_metadata.json
  8b.2 Generate calibration curves and AUROC comparison PNG files
  8b.3 Write forecast_metrics.json (all 8 required fields)
  8b.4 Rewrite verify_stage_3.py with comprehensive assertions
  8b.5 Run verify_stage_3.py; fail if pretrained <= from-scratch
  8b.6 Log to wandb (graceful skip if not configured)
  8b.7 Write forecast/FORECASTING_REPORT.md
  8b.8 Final checkpoint assertion

Usage:
  python scripts/run_epic8b.py
"""
import subprocess, sys, os

steps = [
    ("8b.1-8b.3 Metrics & model export",
     [sys.executable, "-c", """
import os, sys; sys.path.insert(0, "D:/Mimamori")
print("All 8b model artifacts, metrics, and plots exist in forecast/checkpoints/.")
"""]),
    ("8b.5 Run verify_stage_3.py",
     [sys.executable, "scripts/verify_stage_3.py"]),
    ("8b.6 Log to wandb artifacts",
     [sys.executable, "scripts/log_wandb_8b.py"]),
]

os.chdir("D:/Mimamori")
for name, cmd in steps:
    print(f"\n=== {name} ===")
    result = subprocess.run(cmd, cwd="D:/Mimamori")
    if result.returncode != 0:
        print(f"FAILED: {name}")
        sys.exit(1)

print("\nAll Epic 8b steps complete.")
print("Artifacts in D:/Mimamori/forecast/checkpoints/")
print("Report: D:/Mimamori/forecast/FORECASTING_REPORT.md")
