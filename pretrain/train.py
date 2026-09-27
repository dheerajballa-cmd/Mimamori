"""
pretrain/train.py — Epics 7.2, 7.3, 7.4

Masked-patch reconstruction pretraining using PyTorch Lightning.

Objective (Epic 7.2):
    Mask 30–40% of patches at random, reconstruct them with MSE loss.
    Only masked positions contribute to the loss (as in BERT/MAE).

Corpus (Epic 7.3):
    Primary:   NHATS + NHATS accelerometry (real longitudinal aging data)
    Secondary: eHomeSeniors (~20% weighted sampling)

Logging (Epic 7.4):
    Training curves and reconstruction loss logged to wandb.

Usage:
    python -m pretrain.train --config configs/pretrain_nhats.yaml
    python -m pretrain.train --config configs/pretrain_nhats.yaml --fast_dev_run
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import WandbLogger
import yaml

from pretrain.model import MimamoriEncoder
from pretrain.dataset import build_dataloaders


# ── Lightning Module ──────────────────────────────────────────────────────────

class PretrainingModule(pl.LightningModule):
    """Masked-patch self-supervised pretraining module.

    Masking strategy:
        For each sequence in the batch, randomly select `mask_ratio` fraction
        of non-padding time-steps and zero them out before passing to the
        encoder. Only those positions contribute to the reconstruction loss.
    """

    def __init__(
        self,
        encoder: MimamoriEncoder,
        mask_ratio: float = 0.35,
        learning_rate: float = 3e-4,
        weight_decay: float = 1e-5,
        lr_scheduler: str = "cosine",
        warmup_epochs: int = 5,
        max_epochs: int = 60,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["encoder"])
        self.encoder = encoder
        self.mask_ratio = mask_ratio
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.lr_scheduler = lr_scheduler
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.loss_fn = nn.MSELoss(reduction="none")

    # ── Masking ───────────────────────────────────────────────────────────────

    def _make_mask(
        self, pad_mask: torch.Tensor
    ) -> torch.Tensor:
        """Generate a random patch mask.

        Args:
            pad_mask: (B, T) bool — True = valid (not padded)

        Returns:
            mask: (B, T) bool — True = masked (to reconstruct)
        """
        B, T = pad_mask.shape
        mask = torch.zeros(B, T, dtype=torch.bool, device=pad_mask.device)

        for i in range(B):
            valid_indices = pad_mask[i].nonzero(as_tuple=False).squeeze(1)
            n_valid = len(valid_indices)
            n_mask = max(1, int(round(n_valid * self.mask_ratio)))
            perm = torch.randperm(n_valid, device=pad_mask.device)[:n_mask]
            mask[i, valid_indices[perm]] = True

        return mask

    # ── Forward + Loss ────────────────────────────────────────────────────────

    def _step(self, batch: dict, split: str) -> torch.Tensor:
        x = batch["x"]                  # (B, T, C)
        pad_mask = batch["pad_mask"]    # (B, T) bool

        # Generate mask only for training; at val, mask same ratio for consistency
        mask = self._make_mask(pad_mask)

        # Forward pass (masked positions zeroed internally)
        out = self.encoder(x, mask=mask)
        recon = out["reconstructions"]  # (B, T, C)

        # Compute MSE only at masked positions
        loss_per_elem = self.loss_fn(recon, x)          # (B, T, C)
        mask_expanded = mask.unsqueeze(-1).expand_as(loss_per_elem)
        masked_loss = (loss_per_elem * mask_expanded).sum()
        n_masked = mask_expanded.sum().clamp(min=1)
        loss = masked_loss / n_masked

        self.log(f"{split}/reconstruction_loss", loss, prog_bar=True,
                 on_step=(split == "train"), on_epoch=True, sync_dist=False)
        return loss

    def training_step(self, batch: dict, _: int) -> torch.Tensor:
        return self._step(batch, "train")

    def validation_step(self, batch: dict, _: int) -> None:
        self._step(batch, "val")

    # ── Optimiser ─────────────────────────────────────────────────────────────

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        if self.lr_scheduler == "cosine":
            # Linear warmup then cosine annealing
            def lr_lambda(epoch: int) -> float:
                if epoch < self.warmup_epochs:
                    return (epoch + 1) / max(1, self.warmup_epochs)
                progress = (epoch - self.warmup_epochs) / max(
                    1, self.max_epochs - self.warmup_epochs
                )
                return 0.5 * (1.0 + np.cos(np.pi * progress))

            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"},
            }

        return optimizer


# ── Training entry-point ──────────────────────────────────────────────────────

def train(cfg: dict, fast_dev_run: bool = False) -> dict:
    """Run pretraining and save artifacts.

    Returns:
        metadata dict saved to pretrain/checkpoints/pretraining_metadata.json
    """
    pl.seed_everything(cfg["training"]["seed"], workers=True)

    tcfg = cfg["training"]
    ckpt_cfg = cfg["checkpoint"]
    wcfg = cfg["wandb"]

    # ── Data ─────────────────────────────────────────────────────────────────
    print("Loading data…")
    train_loader, val_loader, data_stats = build_dataloaders(
        cfg,
        batch_size=tcfg["batch_size"],
        num_workers=tcfg["num_workers"],
        seed=tcfg["seed"],
    )
    print(
        f"  NHATS train sequences:    {data_stats['n_nhats_train']}\n"
        f"  eHomeSeniors train seqs:  {data_stats['n_ehome_train']}\n"
        f"  Total train:              {data_stats['n_train']}\n"
        f"  Val:                      {data_stats['n_val']}"
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    print("Building encoder…")
    encoder = MimamoriEncoder.from_config(cfg)
    n_params = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
    print(f"  Encoder parameters: {n_params:,}")

    module = PretrainingModule(
        encoder=encoder,
        mask_ratio=cfg["model"]["mask_ratio"],
        learning_rate=tcfg["learning_rate"],
        weight_decay=tcfg["weight_decay"],
        lr_scheduler=tcfg["lr_scheduler"],
        warmup_epochs=tcfg["warmup_epochs"],
        max_epochs=tcfg["max_epochs"],
    )

    # ── Logger (wandb) ────────────────────────────────────────────────────────
    wandb_logger = WandbLogger(
        project=wcfg["project"],
        name=wcfg["name"],
        tags=wcfg.get("tags", []),
        log_model=False,
    )
    wandb_logger.log_hyperparams({
        "mask_ratio": cfg["model"]["mask_ratio"],
        "d_model": cfg["model"]["d_model"],
        "num_hidden_layers": cfg["model"]["num_hidden_layers"],
        "batch_size": tcfg["batch_size"],
        "learning_rate": tcfg["learning_rate"],
        "max_epochs": tcfg["max_epochs"],
        "n_nhats_train": data_stats["n_nhats_train"],
        "n_ehome_train": data_stats["n_ehome_train"],
    })

    # ── Callbacks ─────────────────────────────────────────────────────────────
    ckpt_dir = Path(ckpt_cfg["dirpath"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_cb = ModelCheckpoint(
        dirpath=str(ckpt_dir),
        filename=ckpt_cfg["filename"],
        monitor=ckpt_cfg["monitor"],
        mode=ckpt_cfg["mode"],
        save_top_k=ckpt_cfg["save_top_k"],
        save_weights_only=False,
        verbose=True,
    )
    early_stop_cb = EarlyStopping(
        monitor=ckpt_cfg["monitor"],
        patience=15,
        mode=ckpt_cfg["mode"],
        verbose=True,
    )

    # ── Trainer ───────────────────────────────────────────────────────────────
    trainer = pl.Trainer(
        max_epochs=tcfg["max_epochs"],
        precision=tcfg.get("precision", "32-true"),
        logger=wandb_logger,
        callbacks=[checkpoint_cb, early_stop_cb],
        log_every_n_steps=wcfg.get("log_every_n_steps", 10),
        fast_dev_run=fast_dev_run,
        enable_progress_bar=True,
        deterministic=False,  # PatchTST uses non-deterministic ops
    )

    # ── Train ─────────────────────────────────────────────────────────────────
    print("Starting pretraining…")
    t0 = time.time()
    trainer.fit(module, train_loader, val_loader)
    elapsed = time.time() - t0

    if fast_dev_run:
        print("fast_dev_run complete — skipping checkpoint and metadata save.")
        return {}

    # ── Extract best model's encoder and save ─────────────────────────────────
    best_ckpt_path = checkpoint_cb.best_model_path
    best_val_loss = float(checkpoint_cb.best_model_score or float("nan"))

    print(f"\nBest checkpoint: {best_ckpt_path}  (val_loss={best_val_loss:.4f})")

    # Load best weights back
    best_module = PretrainingModule.load_from_checkpoint(
        best_ckpt_path,
        encoder=MimamoriEncoder.from_config(cfg),
        mask_ratio=cfg["model"]["mask_ratio"],
        learning_rate=tcfg["learning_rate"],
        weight_decay=tcfg["weight_decay"],
        lr_scheduler=tcfg["lr_scheduler"],
        warmup_epochs=tcfg["warmup_epochs"],
        max_epochs=tcfg["max_epochs"],
    )

    # Save encoder-only weights for downstream use
    encoder_path = ckpt_dir / "encoder_best.pt"
    torch.save(best_module.encoder.state_dict(), encoder_path)
    print(f"Encoder weights saved → {encoder_path}")

    # ── Save training metadata (7b.2) ─────────────────────────────────────────
    metadata = {
        "final_reconstruction_loss": best_val_loss,
        "num_training_epochs": trainer.current_epoch,
        "learning_rate": tcfg["learning_rate"],
        "batch_size": tcfg["batch_size"],
        "masking_ratio": cfg["model"]["mask_ratio"],
        "dataset_sizes": {
            "nhats_train": data_stats["n_nhats_train"],
            "ehome_train": data_stats["n_ehome_train"],
            "total_train": data_stats["n_train"],
            "val": data_stats["n_val"],
        },
        "model": {
            "d_model": cfg["model"]["d_model"],
            "num_hidden_layers": cfg["model"]["num_hidden_layers"],
            "num_attention_heads": cfg["model"]["num_attention_heads"],
            "n_parameters": n_params,
        },
        "feature_cols": data_stats["feature_cols"],
        "norm_mean": data_stats["norm_mean"],
        "norm_std": data_stats["norm_std"],
        "training_time_seconds": elapsed,
        "best_checkpoint": str(best_ckpt_path),
        "wandb_run": wcfg["name"],
    }

    meta_path = ckpt_dir / "pretraining_metadata.json"
    meta_path.write_text(json.dumps(metadata, indent=2))
    print(f"Metadata saved → {meta_path}")

    # ── Log encoder artifact to wandb ─────────────────────────────────────────
    try:
        import wandb
        artifact = wandb.Artifact("encoder_best", type="model")
        artifact.add_file(str(encoder_path))
        artifact.add_file(str(meta_path))
        wandb.log_artifact(artifact)
        print("Encoder artifact logged to wandb.")
    except Exception as e:
        print(f"[warn] wandb artifact upload skipped: {e}")

    return metadata


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Mimamori Epic 7 pretraining")
    parser.add_argument("--config", default="configs/pretrain_nhats.yaml")
    parser.add_argument("--fast_dev_run", action="store_true",
                        help="Run 1 batch for smoke-testing")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    train(cfg, fast_dev_run=args.fast_dev_run)


if __name__ == "__main__":
    main()
