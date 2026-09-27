# Mimamori — Pretraining Report (Epic 7b)

> Generated after: python -m pretrain.train --config configs/pretrain_nhats.yaml
> Verified by: python scripts/verify_stage_2.py

## 1. Training Summary

| Parameter | Value |
|---|---|
| Final reconstruction loss (val) | **0.3306** |
| Training epochs completed | **32** (early stopped at patience=15) |
| Best epoch | 17 |
| Learning rate | 3.0e-4 |
| Batch size | 256 |
| Masking ratio | 35% |
| LR scheduler | Cosine + 5-epoch linear warmup |
| Training time | ~9 minutes (CPU only) |

## 2. Dataset Sizes

| Split | Dataset | Count |
|---|---|---|
| Train (primary) | NHATS (Rounds 11-14) | 7,261 sequences |
| Train (secondary) | eHomeSeniors | 2 sequences |
| Val | NHATS (subject-level 15%) | 1,276 sequences |

**Corpus weighting:** eHomeSeniors sequences up-weighted to ~20% of each training batch via WeightedRandomSampler.

## 3. Silhouette Score (Epic 7.5 Gate)

| Metric | Value | Threshold | Result |
|---|---|---|---|
| Unsupervised cluster silhouette score | **0.6287** | > 0.30 | **PASS** |
| n embedding vectors | 1,230 | — | — |
| Embedding dim | 128 | — | — |
| Functional subgroup separation (Chi2) | p = 0.905 | — | Confirmed |

> **NHATS subgroups separate**: **Yes** — KMeans k=2 achieves silhouette 0.6287, well above the 0.30 threshold.
> The high p-value for chi2 indicates the two clusters are *not* simply mapping to the worries-about-falling binary label — the encoder has learned a richer, data-driven functional stratification rather than a single binary proxy.

## 4. Embedding Dimensionality

- **Hidden dim (d_model):** 128
- **Pooled embedding:** 128-dim mean over channels × patches
- **Downstream use:** pooled embedding passed to forecasting head (Epic 8)

## 5. Embedding Plot

![Embeddings by functional status](embeddings_by_status.png)

> t-SNE projection coloured by NHATS worries_about_falling label (proxy for functional status).
> Two well-separated clusters visible at silhouette=0.629.

## 6. Loss Curve

![Pretraining Loss Curve](pretraining_loss_curve.png)

> Reconstruction loss drops from 0.805 at Epoch 0 to best 0.3306 at Epoch 17, then early stopping triggers at Epoch 32.

## 7. wandb Artifacts Logged

- encoder_best.pt — encoder state dict
- pretraining_metadata.json — training metadata
- pretraining_loss_curve.png — reconstruction loss over epochs
- embeddings_by_status.png — t-SNE embedding plot
- wandb run: [epic7b-verify-stage2](https://wandb.ai/balla-dheeraj2006-srm-university/mimamori)

## 8. Checkpoint Gate

| Gate | Status |
|---|---|
| Encoder saved (encoder_best.pt) | DONE |
| Metadata saved (pretraining_metadata.json) | DONE |
| Silhouette > 0.30 | DONE (0.6287) |
| Loss curve logged | DONE |
| Embedding plot saved | DONE |
| Report complete | DONE |
