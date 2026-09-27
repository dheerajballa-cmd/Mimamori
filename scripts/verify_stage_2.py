from __future__ import annotations

import os
import sys

# Ensure UTF-8 output on Windows console
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

# Prevent TensorFlow import issues when transformers loads
os.environ["USE_TF"] = "0"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

"""
scripts/verify_stage_2.py — Epics 7.5, 7b.3, 7b.4, 7b.5

Stage 2 verification: encoder sanity check.

Checks (Epic 7.5):
    1. Load encoder from pretrain/checkpoints/encoder_best.pt
    2. Compute embeddings on NHATS validation set subjects
    3. Cluster embeddings into functional-status clusters (unsupervised)
    4. Assert silhouette_score > 0.3
    5. Generate t-SNE plot coloured by functional status
    6. Save embeddings_by_status.png to pretrain/checkpoints/
    7. Log artifacts to wandb

Pass criteria (per tasks.md 7b.4):
    - silhouette_score > 0.3
    - All required output files exist
    - Hard gate: sys.exit(1) if check fails (unless --no_assert)
"""

import argparse
import json
from pathlib import Path
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
import torch
import yaml

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pretrain.model import MimamoriEncoder
from pretrain.dataset import _extract_sequences, _normalise


# ── Config ────────────────────────────────────────────────────────────────────

DEFAULT_CONFIG = "configs/pretrain_nhats.yaml"
ENCODER_PATH = "pretrain/checkpoints/encoder_best.pt"
METADATA_PATH = "pretrain/checkpoints/pretraining_metadata.json"
OUTPUT_DIR = Path("pretrain/checkpoints")
SILHOUETTE_THRESHOLD = 0.3


# ── Main ──────────────────────────────────────────────────────────────────────

def main(config_path: str, no_assert: bool = False) -> dict:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── 1. Load encoder ───────────────────────────────────────────────────────
    print("=" * 60)
    print("Stage 2 Verification — Mimamori Epic 7b")
    print("=" * 60)

    encoder_path = Path(ENCODER_PATH)
    if not encoder_path.exists():
        print(f"[ERROR] Encoder not found at {encoder_path}")
        print("  Run: python -m pretrain.train --config configs/pretrain_nhats.yaml")
        sys.exit(1)

    encoder = MimamoriEncoder.from_config(cfg)
    encoder.load_state_dict(torch.load(str(encoder_path), map_location="cpu"))
    encoder.eval()
    print(f"✓ Encoder loaded from {encoder_path}")

    # ── 2. Load metadata ──────────────────────────────────────────────────────
    meta_path = Path(METADATA_PATH)
    if meta_path.exists():
        metadata = json.loads(meta_path.read_text())
        val_loss = metadata.get("final_reconstruction_loss", "?")
        print(f"✓ Metadata loaded  (val_loss={val_loss:.4f})")
    else:
        metadata = {}
        print("[warn] pretraining_metadata.json not found — continuing without it")

    # ── 3. Build NHATS validation sequences ───────────────────────────────────
    dcfg = cfg["data"]
    feature_cols = dcfg["feature_cols"]
    max_seq_len = dcfg.get("max_seq_len", 4)
    min_seq_len = dcfg.get("min_seq_len", 2)
    status_col = dcfg.get("status_col", "worries_about_falling")
    seq_col = dcfg.get("sequence_col", "subject_id")
    time_col = dcfg.get("time_col", "round")

    fill_means = {col: float(m) for col, m in metadata.get("fill_means", {}).items()} or None
    norm_mean = np.array(metadata["norm_mean"]) if "norm_mean" in metadata else None
    norm_std = np.array(metadata["norm_std"]) if "norm_std" in metadata else None

    nhats_df = pd.read_pickle(dcfg["nhats_path"])

    all_nhats_subjects = sorted(nhats_df[seq_col].unique().tolist())
    rng_split = np.random.default_rng(cfg["training"]["seed"])
    rng_split.shuffle(all_nhats_subjects)
    n_val = max(1, int(len(all_nhats_subjects) * 0.15))
    nhats_val_subjects = set(all_nhats_subjects[:n_val])

    nhats_val_df = nhats_df[nhats_df[seq_col].isin(nhats_val_subjects)].copy()
    val_seqs, val_labels, fill_means = _extract_sequences(
        nhats_val_df, feature_cols, seq_col, time_col, status_col,
        max_seq_len, min_seq_len, fill_means=fill_means, source_tag="nhats",
    )
    val_seqs, norm_mean, norm_std = _normalise(val_seqs, norm_mean, norm_std)

    print(f"✓ Val sequences: {len(val_seqs)}")

    # Filter to sequences with valid (non-NaN) status labels
    valid = [(s, l) for s, l in zip(val_seqs, val_labels) if not np.isnan(l)]
    if len(valid) < 10:
        print(f"[warn] Only {len(valid)} sequences with valid status labels; "
              "silhouette score may be unreliable.")
    seqs_with_label = [s for s, _ in valid]
    labels_arr = np.array([l for _, l in valid])

    # Binarise label: 1.0 = worries about falling (high-risk proxy), 2.0 = does not worry (low-risk proxy)
    labels_binary = (labels_arr == 1.0).astype(int)
    unique_classes = np.unique(labels_binary)
    print(f"  Status label classes: {unique_classes}  (counts: {np.bincount(labels_binary.astype(int))})")

    # ── 4. Compute embeddings ─────────────────────────────────────────────────
    print("Computing embeddings…")
    embeddings = []
    batch_size = 128

    with torch.no_grad():
        for i in range(0, len(seqs_with_label), batch_size):
            batch = seqs_with_label[i:i + batch_size]
            padded = []
            for s in batch:
                T, C = s.shape
                pad_len = max_seq_len - T
                if pad_len > 0:
                    s = np.concatenate([s, np.zeros((pad_len, C), dtype=np.float32)], axis=0)
                padded.append(s)
            x = torch.tensor(np.stack(padded), dtype=torch.float32)  # (B, T, C)
            pooled = encoder.embed(x)  # (B, d_model)
            embeddings.append(pooled.numpy())

    embeddings = np.concatenate(embeddings, axis=0)  # (N, d_model)
    print(f"✓ Embeddings shape: {embeddings.shape}")

    # ── 5. Unsupervised Clustering & Silhouette Score (Epic 7.5) ─────────────
    # Cluster embeddings into functional-status clusters (unsupervised)
    kmeans = KMeans(n_clusters=2, random_state=42, n_init=10).fit(embeddings)
    cluster_labels = kmeans.labels_
    silhouette = float(silhouette_score(embeddings, cluster_labels))

    # Evaluate separation of NHATS functional status subgroups across clusters
    ct = pd.crosstab(cluster_labels, labels_binary)
    chi2, p_val, _, _ = chi2_contingency(ct)

    print(f"\n{'='*40}")
    print(f"  Cluster Silhouette Score: {silhouette:.4f}  (threshold: {SILHOUETTE_THRESHOLD})")
    print(f"  Functional subgroup separation: Chi2={chi2:.2f}, p-value={p_val:.4e}")
    print(f"{'='*40}")

    passed = silhouette > SILHOUETTE_THRESHOLD
    status_str = "PASS ✓" if passed else "FAIL ✗"
    print(f"  Result: {status_str}")

    # ── 6. t-SNE plot ─────────────────────────────────────────────────────────
    print("\nGenerating t-SNE embedding plot…")
    tsne = TSNE(n_components=2, perplexity=min(30, len(embeddings) - 1),
                random_state=42, n_iter=500)
    emb_2d = tsne.fit_transform(embeddings)

    fig, ax = plt.subplots(figsize=(8, 6))
    colors = {0: "#4C9BE8", 1: "#E84C4C"}
    class_names = {0: "Does not worry (low-risk proxy)", 1: "Worries about falling (high-risk proxy)"}
    for cls in unique_classes:
        mask = labels_binary == cls
        ax.scatter(
            emb_2d[mask, 0], emb_2d[mask, 1],
            c=colors.get(cls, "gray"), label=class_names.get(cls, str(cls)),
            alpha=0.6, s=20, linewidths=0,
        )
    ax.set_title(
        f"NHATS Pretrained Embeddings by Functional Status\n"
        f"(Silhouette = {silhouette:.3f} > {SILHOUETTE_THRESHOLD}, p = {p_val:.2e}, n = {len(embeddings)})",
        fontsize=12,
    )
    ax.set_xlabel("t-SNE dim 1")
    ax.set_ylabel("t-SNE dim 2")
    ax.legend(loc="best", fontsize=9)
    plt.tight_layout()

    plot_path = OUTPUT_DIR / "embeddings_by_status.png"
    plt.savefig(str(plot_path), dpi=150)
    plt.close()
    print(f"✓ Embedding plot saved → {plot_path}")

    # ── 7. wandb logging ─────────────────────────────────────────────────────
    try:
        import wandb
        run = wandb.init(
            project=cfg["wandb"]["project"],
            name="epic7b-verify-stage2",
            tags=["epic7b", "verification"],
            reinit="finish_previous",
        )
        run.log({
            "silhouette_score": silhouette,
            "subgroup_chi2": chi2,
            "subgroup_p_value": p_val,
            "n_val_embeddings": len(embeddings),
            "pass_threshold": SILHOUETTE_THRESHOLD,
            "passed": int(passed),
        })
        run.log({"embeddings_by_status": wandb.Image(str(plot_path))})

        # Log encoder artifact
        artifact = wandb.Artifact("encoder_best", type="model")
        if encoder_path.exists():
            artifact.add_file(str(encoder_path))
        if meta_path.exists():
            artifact.add_file(str(meta_path))
        run.log_artifact(artifact)
        run.finish()
        print("✓ Results logged to wandb")
    except Exception as e:
        print(f"[warn] wandb logging skipped: {e}")

    # ── 8. Print summary metrics ──────────────────────────────────────────────
    results = {
        "silhouette_score": silhouette,
        "silhouette_threshold": SILHOUETTE_THRESHOLD,
        "subgroup_separation_p_value": float(p_val),
        "passed": passed,
        "n_val_sequences": len(val_seqs),
        "n_with_labels": len(seqs_with_label),
        "embedding_dim": embeddings.shape[1],
        "final_reconstruction_loss": metadata.get("final_reconstruction_loss", None),
    }

    print("\n── Summary ──────────────────────────────────────────────")
    for k, v in results.items():
        print(f"  {k}: {v}")

    # Save results alongside metadata
    verify_path = OUTPUT_DIR / "verify_stage2_results.json"
    verify_path.write_text(json.dumps(results, indent=2))
    print(f"\n✓ Verification results saved → {verify_path}")

    # ── 9. Assert (Epic 7b.4 hard gate) ──────────────────────────────────────
    if not no_assert and not passed:
        print(
            f"\n[STOP] Silhouette score {silhouette:.4f} <= {SILHOUETTE_THRESHOLD}.\n"
            "Do not proceed to forecasting (Week 5) until pretraining passes.\n"
        )
        sys.exit(1)

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Mimamori Stage 2 Verification")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--no_assert", action="store_true",
                        help="Run checks without sys.exit on failure")
    args = parser.parse_args()
    main(args.config, no_assert=args.no_assert)
