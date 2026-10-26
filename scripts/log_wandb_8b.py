"""
scripts/log_wandb_8b.py - Epic 8b.6
Log Epic 8b artifacts to Weights & Biases:
  - forecaster_best.pt (PyTorch checkpoint)
  - baseline_xgboost.pkl (XGBoost baseline)
  - All PNG plots (calibration_curve_short_horizon.png,
                   calibration_curve_long_horizon.png,
                   auroc_comparison.png)
  - forecast_metrics.json & forecast_metadata.json
"""
import os
import sys
import json
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

os.environ["USE_TF"] = "0"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

CKPT_DIR = Path("D:/Mimamori/forecast/checkpoints")

def log_to_wandb():
    import wandb

    print("=" * 60)
    print("Epic 8b.6 - Logging artifacts to Weights & Biases")
    print("=" * 60)

    metrics_path = CKPT_DIR / "forecast_metrics.json"
    metadata_path = CKPT_DIR / "forecast_metadata.json"

    metrics = {}
    if metrics_path.exists():
        with open(metrics_path, "r", encoding="utf-8") as f:
            metrics = json.load(f)

    metadata = {}
    if metadata_path.exists():
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)

    run = wandb.init(
        project="mimamori",
        name="epic8b-stage3-forecasting",
        tags=["epic8b", "stage3", "forecasting", "fall_risk"],
        notes="Stage 3 Forecasting Head verification artifacts: forecaster_best.pt, baseline_xgboost.pkl, calibration curves, AUROC comparison, and forecast_metrics.json",
        reinit="finish_previous",
        config={
            **metadata,
            "stage": "Stage 3 - Forecasting Head",
            "task": "8b.6",
        }
    )

    if metrics:
        scalar_metrics = {k: v for k, v in metrics.items() if isinstance(v, (int, float, bool))}
        run.log(scalar_metrics)
        print("Logged scalar metrics to W&B:")
        for k, v in scalar_metrics.items():
            print(f"  {k}: {v}")

    plot_names = [
        "calibration_curve_short_horizon.png",
        "calibration_curve_long_horizon.png",
        "auroc_comparison.png"
    ]
    for plot_name in plot_names:
        p = CKPT_DIR / plot_name
        if p.exists():
            image_key = plot_name.replace(".png", "")
            run.log({image_key: wandb.Image(str(p), caption=plot_name)})
            print(f"Logged image: {plot_name}")
        else:
            print(f"[warn] Image not found: {p}")

    artifact = wandb.Artifact(
        name="forecaster_stage3",
        type="model",
        description="Epic 8b Stage 3 Forecaster models, evaluation plots, and metrics",
        metadata={
            "architecture": metadata.get("architecture", "PatchTST Forecaster"),
            "task": "8b.6",
        }
    )

    files_to_log = [
        CKPT_DIR / "forecaster_best.pt",
        CKPT_DIR / "baseline_xgboost.pkl",
        CKPT_DIR / "forecast_metrics.json",
        CKPT_DIR / "forecast_metadata.json",
        CKPT_DIR / "calibration_curve_short_horizon.png",
        CKPT_DIR / "calibration_curve_long_horizon.png",
        CKPT_DIR / "auroc_comparison.png"
    ]

    for file_path in files_to_log:
        if file_path.exists():
            artifact.add_file(str(file_path))
            print(f"Added to artifact: {file_path.name}")
        else:
            print(f"[warn] File not found to add: {file_path}")

    run.log_artifact(artifact)
    print("W&B Artifact logged successfully.")

    run_url = getattr(run, "url", "https://wandb.ai")
    run.finish()
    print("=" * 60)
    print(f"Run completed successfully: {run_url}")
    print("=" * 60)
    return run_url

if __name__ == "__main__":
    log_to_wandb()
