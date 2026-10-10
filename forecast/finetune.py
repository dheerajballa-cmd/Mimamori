"""
forecast/finetune.py - Epic 8.1, 8.4, 8.6, 8.7

Fine-tunes the pretrained encoder with a forecasting head.
"""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from pretrain.model import MimamoriEncoder
import torchmetrics

class QuantileLoss(nn.Module):
    def __init__(self, quantiles=(0.1, 0.5, 0.9)):
        super().__init__()
        self.quantiles = quantiles
        
    def forward(self, preds, target):
        # preds: (B, len(quantiles)), target: (B,)
        assert preds.shape[1] == len(self.quantiles)
        loss = 0
        for i, q in enumerate(self.quantiles):
            err = target - preds[:, i]
            loss += torch.max((q - 1) * err, q * err).mean()
        return loss

class ForecastingHead(nn.Module):
    def __init__(self, d_model=128, quantiles=(0.1, 0.5, 0.9)):
        super().__init__()
        self.quantiles = quantiles
        # Quantile head for gait severity
        self.quantile_head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, len(quantiles))
        )
        # Binary head for fall probability
        self.binary_head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1)
        )
        
    def forward(self, x):
        # x is pooled embedding (B, d_model)
        severity_quantiles = self.quantile_head(x)
        fall_logits = self.binary_head(x).squeeze(-1)
        return severity_quantiles, fall_logits

class ForecastingModule(pl.LightningModule):
    def __init__(self, encoder_path: str, d_model: int = 128, learning_rate: float = 1e-4):
        super().__init__()
        self.save_hyperparameters()
        self.encoder = MimamoriEncoder(d_model=d_model)
        
        if encoder_path and os.path.exists(encoder_path):
            state = torch.load(encoder_path, map_location="cpu", weights_only=False)
            missing, unexpected = self.encoder.load_state_dict(state, strict=False)
            if missing:
                print(f"[ForecastingModule] Missing encoder keys: {len(missing)}")
            else:
                print(f"[ForecastingModule] Encoder loaded successfully from {encoder_path}")
            
        self.head_short = ForecastingHead(d_model=d_model)
        self.head_long = ForecastingHead(d_model=d_model)
        
        self.quantile_loss = QuantileLoss()
        self.bce_loss = nn.BCEWithLogitsLoss()
        
        self.train_auprc_short = torchmetrics.classification.BinaryAveragePrecision()
        self.val_auprc_short = torchmetrics.classification.BinaryAveragePrecision()
        
        # Previous risk score state for risk_delta
        self.last_risk = None

    def forward(self, x):
        # x is (B, T, C)
        emb = self.encoder.embed(x)
        sev_short, fall_short = self.head_short(emb)
        sev_long, fall_long = self.head_long(emb)
        return {
            "severity_quantiles_short": sev_short,
            "fall_logits_short": fall_short,
            "severity_quantiles_long": sev_long,
            "fall_logits_long": fall_long
        }
        
    def compute_risk_delta(self, current_risk):
        if self.last_risk is None:
            self.last_risk = current_risk
            return torch.zeros_like(current_risk)
        delta = current_risk - self.last_risk
        self.last_risk = current_risk
        return delta

    def get_confidence_and_abstain(self, logits, threshold=0.15):
        # Epics 8.6: Confidence-gated abstention
        probs = torch.sigmoid(logits)
        # Confidence is how far from 0.5 we are
        confidence = torch.abs(probs - 0.5) * 2
        abstain_mask = confidence < threshold
        return probs, abstain_mask

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, "val")

    def _shared_step(self, batch, phase):
        x, target_fall, target_sev = batch["features"], batch["is_fall"], batch["severity"]
        out = self(x)
        
        loss_fall = self.bce_loss(out["fall_logits_short"], target_fall.float())
        loss_sev = self.quantile_loss(out["severity_quantiles_short"], target_sev.float())
        
        loss = loss_fall + loss_sev
        
        self.log(f"{phase}_loss", loss, prog_bar=True)
        if phase == "train":
            self.train_auprc_short.update(out["fall_logits_short"], target_fall.long())
            self.log("train_auprc", self.train_auprc_short, on_step=False, on_epoch=True)
        else:
            self.val_auprc_short.update(out["fall_logits_short"], target_fall.long())
            self.log("val_auprc", self.val_auprc_short, on_step=False, on_epoch=True, prog_bar=True)
            
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
