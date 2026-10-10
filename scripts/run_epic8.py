import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
import xgboost as xgb
from sklearn.metrics import average_precision_score
from forecast.finetune import ForecastingModule
from data.splits import get_frozen_splits

# Set TF vars to prevent hanging
os.environ["USE_TF"] = "0"
os.environ["USE_FLAX"] = "0"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

class ProxyDataset(Dataset):
    def __init__(self, df):
        self.df = df
        
    def __len__(self):
        return len(self.df)
        
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Build (T, C) tensor
        # Features order expected by encoder: gait_cadence, stride_variability, sway_index, step_length, walking_speed
        T = len(row["seq_cadence"])
        c0 = row["seq_cadence"]
        c1 = row["seq_stride_var"]
        c2 = row["seq_sway"]
        c3 = [0.0] * T
        c4 = [0.0] * T
        
        feat = np.column_stack((c0, c1, c2, c3, c4)).astype(np.float32)
        
        return {
            "features": torch.tensor(feat),
            "is_fall": torch.tensor(row["target_is_fall"], dtype=torch.float32),
            "severity": torch.tensor(row["target_gait_severity"] / 1000.0, dtype=torch.float32)  # normalize sway_index to ~[0,8]
        }

def get_baseline_data(df):
    # Same-day features (day 4, index -1)
    # 3 features
    X = []
    y = []
    for _, row in df.iterrows():
        x_day = [
            row["seq_cadence"][-1],
            row["seq_stride_var"][-1],
            row["seq_sway"][-1]
        ]
        X.append(x_day)
        y.append(row["target_is_fall"])
    return np.array(X), np.array(y)

def run_baseline(train_df, val_df):
    print("--- Running XGBoost Baseline ---")
    X_train, y_train = get_baseline_data(train_df)
    X_val, y_val = get_baseline_data(val_df)
    
    clf = xgb.XGBClassifier(n_estimators=50, max_depth=3, learning_rate=0.1)
    clf.fit(X_train, y_train)
    
    preds_val = clf.predict_proba(X_val)[:, 1]
    auprc = average_precision_score(y_val, preds_val)
    print(f"Baseline Validation AUPRC: {auprc:.4f}")
    return auprc

def run_lightning(train_df, val_df):
    print("--- Running PyTorch Lightning Finetuning ---")
    train_loader = DataLoader(ProxyDataset(train_df), batch_size=64, shuffle=True)
    val_loader = DataLoader(ProxyDataset(val_df), batch_size=64, shuffle=False)
    
    encoder_path = "D:/Mimamori/pretrain/checkpoints/encoder_best.pt"
    model = ForecastingModule(encoder_path=encoder_path, learning_rate=2e-4)
    
    checkpoint_callback = ModelCheckpoint(
        dirpath="D:/Mimamori/forecast/checkpoints",
        filename="forecaster_best",
        monitor="val_auprc",
        mode="max"
    )
    
    trainer = pl.Trainer(
        max_epochs=15,
        callbacks=[checkpoint_callback],
        logger=False,
        enable_progress_bar=True
    )
    
    trainer.fit(model, train_loader, val_loader)
    return checkpoint_callback.best_model_score.item() if checkpoint_callback.best_model_score else 0.0

if __name__ == "__main__":
    df = pd.read_pickle("D:/Mimamori/data/processed/forecast_proxy_dataset.pkl")
    
    # Simple split since subject_ids are "SisFall_SA01" etc.
    frozen = get_frozen_splits()
    sf_test = frozen.get("SisFall", {}).get("test", [])
    sf_val = frozen.get("SisFall", {}).get("val", [])
    
    def get_split(sub_id):
        raw_id = sub_id.split("_")[1]
        if raw_id in sf_test: return "test"
        if raw_id in sf_val: return "val"
        return "train"
        
    df["split"] = df["subject_id"].apply(get_split)
    
    train_df = df[df["split"] == "train"]
    val_df = df[df["split"] == "val"]
    
    if len(val_df) == 0:
        print("No val subjects found in proxy dataset. Falling back to random split.")
        from sklearn.model_selection import train_test_split
        train_df, val_df = train_test_split(df, test_size=0.15, random_state=42)
    
    # Baseline
    base_auprc = run_baseline(train_df, val_df)
    
    # Lightning
    ft_auprc = run_lightning(train_df, val_df)
    
    print("\n=== Epic 8.8 Checkpoint ===")
    print(f"Baseline AUPRC:  {base_auprc:.4f}")
    print(f"Finetuned AUPRC: {ft_auprc:.4f}")
    
    if ft_auprc > base_auprc:
        print("SUCCESS: Finetuned model beats the static baseline on the proxy task!")
    else:
        print("WARNING: Finetuned model did not beat baseline. Check pretraining or finetuning params.")
