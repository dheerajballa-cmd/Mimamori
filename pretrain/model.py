"""
pretrain/model.py — Epic 7.1

PatchTST-style encoder for Mimamori masked-sequence pretraining.

Architecture:
- Input: sequences of DailyFeatureVectors (T time-steps × C features)
- Each time-step is treated as a single "patch" (patch_length=1)
- Uses HuggingFace transformers.PatchTSTConfig / PatchTSTModel as the backbone
- A lightweight linear reconstruction head maps encoder outputs back to the
  original feature space for the masked-patch reconstruction objective

Interface consumed downstream:
- pretrain/train.py: wraps this model in a LightningModule
- forecast/finetune.py: loads encoder_best.pt and attaches a forecasting head
- scripts/verify_stage_2.py: loads encoder and calls forward() / embed()
"""

from __future__ import annotations

import torch
import torch.nn as nn
from transformers import PatchTSTConfig, PatchTSTModel


class MimamoriEncoder(nn.Module):
    """PatchTST-style encoder for multivariate daily-feature sequences.

    Args:
        num_input_channels: number of feature dimensions (C)
        context_length: sequence length in time-steps / patches (T)
        patch_length: tokens per patch (1 = each daily vector is one patch)
        patch_stride: stride between patches
        d_model: encoder hidden dim
        num_attention_heads: number of self-attention heads
        num_hidden_layers: transformer depth
        ffn_dim: feed-forward inner dim
        dropout: attention + residual dropout
        head_dropout: reconstruction head dropout
    """

    def __init__(
        self,
        num_input_channels: int = 5,
        context_length: int = 4,
        patch_length: int = 1,
        patch_stride: int = 1,
        d_model: int = 128,
        num_attention_heads: int = 4,
        num_hidden_layers: int = 3,
        ffn_dim: int = 256,
        dropout: float = 0.1,
        head_dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.num_input_channels = num_input_channels
        self.context_length = context_length
        self.patch_length = patch_length
        self.patch_stride = patch_stride
        self.d_model = d_model

        # ── PatchTST backbone ────────────────────────────────────────────────
        # do_mask_input=False because we apply our own masking externally
        patchtst_cfg = PatchTSTConfig(
            num_input_channels=num_input_channels,
            context_length=context_length,
            patch_length=patch_length,
            patch_stride=patch_stride,
            d_model=d_model,
            num_attention_heads=num_attention_heads,
            num_hidden_layers=num_hidden_layers,
            ffn_dim=ffn_dim,
            dropout=dropout,
            head_dropout=head_dropout,
            # prediction_length is required by config but unused in encoder-only mode
            prediction_length=1,
            do_mask_input=False,
        )
        self.backbone = PatchTSTModel(patchtst_cfg)

        # ── Reconstruction head (one per channel, shared weights) ────────────
        # Maps from d_model → patch_length per channel
        self.reconstruction_head = nn.Sequential(
            nn.Dropout(head_dropout),
            nn.Linear(d_model, patch_length),
        )

    # ── Forward pass ─────────────────────────────────────────────────────────

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Run the encoder and produce reconstructions.

        Args:
            x: input tensor of shape (batch, time, channels)  [B, T, C]
            mask: boolean tensor of shape (batch, time) where True = masked
                  If None, no masking is applied (inference mode).

        Returns:
            dict with keys:
                - "embeddings":      (B, C, num_patches, d_model)
                - "pooled":          (B, d_model)  — mean over patches & channels
                - "reconstructions": (B, T, C)     — per-patch reconstructions
        """
        # HuggingFace PatchTSTModel expects past_values: (B, T, C)
        x_in = x.clone()  # (B, T, C)

        # Apply mask: zero out masked time-steps before feeding to backbone
        if mask is not None:
            # mask: (B, T) → expand to (B, T, C)
            mask_btc = mask.unsqueeze(-1).expand_as(x_in)
            x_in = x_in.masked_fill(mask_btc, 0.0)

        backbone_out = self.backbone(past_values=x_in)
        # last_hidden_state: (B, C, num_patches, d_model)
        hidden = backbone_out.last_hidden_state

        # ── Reconstruction ───────────────────────────────────────────────────
        # hidden: (B, C, num_patches, d_model)
        recon_patches = self.reconstruction_head(hidden)
        # recon_patches: (B, C, num_patches, patch_length) → (B, T, C)
        B, C, num_patches, pl = recon_patches.shape
        recon = recon_patches.permute(0, 2, 3, 1)  # (B, num_patches, pl, C)
        recon = recon.reshape(B, num_patches * pl, C)  # (B, T, C)

        # ── Pooled embedding (for downstream tasks / silhouette check) ───────
        # Mean over patches and channels
        pooled = hidden.mean(dim=(1, 2))  # (B, d_model)

        return {
            "embeddings": hidden,   # (B, C, num_patches, d_model)
            "pooled": pooled,       # (B, d_model)
            "reconstructions": recon,  # (B, T, C)
        }

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        """Convenience method: return only the pooled embedding vector.

        Args:
            x: (B, T, C) input feature sequence

        Returns:
            (B, d_model) embedding
        """
        with torch.no_grad():
            out = self.forward(x, mask=None)
        return out["pooled"]

    @classmethod
    def from_config(cls, cfg: dict) -> "MimamoriEncoder":
        """Instantiate from the model sub-dict of configs/pretrain_nhats.yaml."""
        m = cfg["model"]
        return cls(
            num_input_channels=m["num_input_channels"],
            context_length=m["context_length"],
            patch_length=m["patch_length"],
            patch_stride=m["patch_stride"],
            d_model=m["d_model"],
            num_attention_heads=m["num_attention_heads"],
            num_hidden_layers=m["num_hidden_layers"],
            ffn_dim=m["ffn_dim"],
            dropout=m["dropout"],
            head_dropout=m["head_dropout"],
        )
