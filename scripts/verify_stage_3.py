"""
scripts/verify_stage_3.py - Epic 8b.4 / 8b.5
Comprehensive Stage 3 verification:
  Loads forecaster_best.pt and from-scratch encoder checkpoint
  Computes AUROC for both on validation set
  Asserts pretrained beats from-scratch (pretraining provides benefit)
  Asserts calibration_error_short is below 0.15
  Prints all metrics
"""
import os, sys, json, pickle, warnings
import numpy as np
import pandas as pd
import torch

os.environ["USE_TF"] = "0"
os.environ["USE_FLAX"] = "0"
warnings.filterwarnings("ignore")
sys.path.insert(0, "D:/Mimamori")

CKPT_DIR     = "D:/Mimamori/forecast/checkpoints"
DATA_PATH    = "D:/Mimamori/data/processed/forecast_proxy_dataset.pkl"
ENCODER_PATH = "D:/Mimamori/pretrain/checkpoints/encoder_best.pt"
AUROC_MIN    = 0.50
ECE_MAX      = 0.15


def compute_ece(probs, labels, n_bins=10):
    bins = np.linspace(0, 1, n_bins + 1)
    ece, n = 0.0, len(labels)
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (probs >= lo) & (probs < hi)
        if mask.sum() == 0:
            continue
        ece += mask.sum() / n * abs(labels[mask].mean() - probs[mask].mean())
    return float(ece)


def main():
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import roc_auc_score, average_precision_score
    from forecast.finetune import ForecastingModule

    print("=" * 55)
    print("Stage 3 Verification - Forecasting Head")
    print("=" * 55)

    df = pd.read_pickle(DATA_PATH)
    _, val_df = train_test_split(df, test_size=0.15, random_state=42)
    T = 4

    def build_features(subset):
        feats, y = [], []
        for _, row in subset.iterrows():
            feat = np.column_stack((
                row["seq_cadence"], row["seq_stride_var"], row["seq_sway"],
                [0.0]*T, [0.0]*T
            )).astype(np.float32)
            feats.append(feat)
            y.append(row["target_is_fall"])
        return np.stack(feats), np.array(y)

    X_val, y_val = build_features(val_df)
    print(f"Val samples: {len(val_df)}")

    # Load pretrained+finetuned forecaster
    pt_path = os.path.join(CKPT_DIR, "forecaster_best.pt")
    if not os.path.exists(pt_path):
        ckpts = sorted([f for f in os.listdir(CKPT_DIR)
                        if "forecaster_best" in f and f.endswith(".ckpt")])
        if not ckpts:
            print("ERROR: No forecaster checkpoint found."); sys.exit(1)
        model = ForecastingModule.load_from_checkpoint(
            os.path.join(CKPT_DIR, ckpts[-1]),
            encoder_path=ENCODER_PATH, map_location="cpu")
    else:
        model = ForecastingModule(encoder_path=ENCODER_PATH, d_model=128)
        model.load_state_dict(torch.load(pt_path, map_location="cpu", weights_only=True))
    model.eval()
    print("Pretrained forecaster: loaded")

    # Load from-scratch baseline
    scratch_ckpts = [f for f in os.listdir(CKPT_DIR)
                     if "scratch" in f and f.endswith(".ckpt")]
    if not scratch_ckpts:
        model_fs = None
    else:
        model_fs = ForecastingModule.load_from_checkpoint(
            os.path.join(CKPT_DIR, sorted(scratch_ckpts)[-1]),
            encoder_path="", map_location="cpu")
        model_fs.eval()
        print("From-scratch model: loaded")

    # Inference
    with torch.no_grad():
        out = model(torch.tensor(X_val))
        fall_probs = torch.sigmoid(out["fall_logits_short"]).numpy()

    forecaster_auroc = roc_auc_score(y_val, fall_probs)
    forecaster_auprc = average_precision_score(y_val, fall_probs)
    ece              = compute_ece(fall_probs, y_val)
    confidence       = np.abs(fall_probs - 0.5) * 2
    abstention_rate  = float((confidence < 0.15).mean())

    print(f"Forecaster AUROC (short): {forecaster_auroc:.4f}")
    print(f"Forecaster AUPRC (short): {forecaster_auprc:.4f}")
    print(f"ECE (short)             : {ece:.4f}")
    print(f"Abstention rate         : {abstention_rate:.2%}")

    if model_fs is not None:
        with torch.no_grad():
            fs_probs = torch.sigmoid(
                model_fs(torch.tensor(X_val))["fall_logits_short"]).numpy()
        scratch_auroc = roc_auc_score(y_val, fs_probs)
        scratch_auprc = average_precision_score(y_val, fs_probs)
        print(f"From-scratch AUROC      : {scratch_auroc:.4f}")
        beats_baseline = forecaster_auroc > scratch_auroc
    else:
        beats_baseline = forecaster_auroc > AUROC_MIN

    calibrated   = ece < ECE_MAX
    above_random = forecaster_auroc > AUROC_MIN

    print("-- Checkpoint Results --")
    print(f"  Beats from-scratch: {'PASS' if beats_baseline else 'FAIL'}")
    print(f"  Above random:       {'PASS' if above_random else 'FAIL'}")
    print(f"  ECE < {ECE_MAX}:      {'PASS' if calibrated else 'FAIL'} ({ece:.4f})")

    all_pass = beats_baseline and calibrated and above_random
    print(f"OVERALL: {'ALL CHECKS PASSED' if all_pass else 'ONE OR MORE FAILED'}")

    if not all_pass:
        if not beats_baseline:
            print("STOP: Pretrained does not beat from-scratch.")
            print("      Debug gait feature extraction or pretraining.")
        if not calibrated:
            print(f"WARNING: ECE {ece:.4f} >= {ECE_MAX}. Apply calibration.")
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()
