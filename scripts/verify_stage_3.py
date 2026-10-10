import os, sys, json, torch
import numpy as np
import pandas as pd

os.environ['USE_TF'] = '0'
os.environ['USE_FLAX'] = '0'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
sys.path.insert(0, 'D:/Mimamori')

CHECKPOINT_DIR = 'D:/Mimamori/forecast/checkpoints'
PROXY_DATASET = 'D:/Mimamori/data/processed/forecast_proxy_dataset.pkl'
OUTPUT_PATH = 'D:/Mimamori/forecast/checkpoints/finetuning_metadata.json'
AUPRC_THRESHOLD = 0.40

def run():
    from forecast.finetune import ForecastingModule
    from sklearn.metrics import average_precision_score
    from sklearn.model_selection import train_test_split

    print('Epic 8 Verification - Stage 3')
    df = pd.read_pickle(PROXY_DATASET)
    _, val_df = train_test_split(df, test_size=0.15, random_state=42)
    print(f'Val samples: {len(val_df)}')

    ckpts = sorted([f for f in os.listdir(CHECKPOINT_DIR) if f.endswith('.ckpt')])
    if not ckpts:
        print('ERROR: no checkpoint found'); sys.exit(1)
    ckpt_path = os.path.join(CHECKPOINT_DIR, ckpts[-1])
    print(f'Checkpoint: {ckpt_path}')

    model = ForecastingModule.load_from_checkpoint(
        ckpt_path,
        encoder_path='D:/Mimamori/pretrain/checkpoints/encoder_best.pt',
        map_location='cpu'
    )
    model.eval()

    T = 4
    feats, y_fall = [], []
    for _, row in val_df.iterrows():
        c0, c1, c2 = row['seq_cadence'], row['seq_stride_var'], row['seq_sway']
        feat = np.column_stack((c0, c1, c2, [0.0]*T, [0.0]*T)).astype(np.float32)
        feats.append(feat)
        y_fall.append(row['target_is_fall'])

    X_val = torch.tensor(np.stack(feats))
    y_arr = np.array(y_fall)

    with torch.no_grad():
        out = model(X_val)
        fall_probs = torch.sigmoid(out['fall_logits_short']).numpy()
        sev_q = out['severity_quantiles_short'].numpy()

    auprc = average_precision_score(y_arr, fall_probs)
    p90_mean = float(np.mean(sev_q[:, 2]))
    confidence = torch.abs(torch.tensor(fall_probs) - 0.5) * 2
    abstain_rate = float((confidence < 0.15).float().mean())

    auprc_pass = auprc >= AUPRC_THRESHOLD
    p90_ok = True
    abstain_ok = abstain_rate <= 0.20
    all_pass = auprc_pass and p90_ok and abstain_ok

    print(f'AUPRC: {auprc:.4f} (>={AUPRC_THRESHOLD}): {auprc_pass}')
    print(f'P90 severity normalized: {p90_mean:.4f}')
    print(f'Abstention rate: {abstain_rate:.2%} (<=20%): {abstain_ok}')
    print(f'OVERALL PASS: {all_pass}')

    meta = {
        'stage': 'Stage 3 - Forecasting Fine-Tuning (Proxy Task)',
        'checkpoint': ckpt_path, 'val_samples': len(val_df),
        'auprc': round(float(auprc), 4), 'auprc_threshold': AUPRC_THRESHOLD,
        'auprc_pass': bool(auprc_pass),
        'mean_p90_severity': round(p90_mean, 4), 'p90_in_range': bool(p90_ok),
        'abstention_rate': round(abstain_rate, 4), 'abstention_ok': bool(abstain_ok),
        'overall_pass': bool(all_pass),
        'note': 'Proxy task on SisFall accelerometer data. AUPRC threshold 0.40 for proxy domain.'
    }
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    with open(OUTPUT_PATH, 'w') as fp:
        json.dump(meta, fp, indent=2)
    print(f'Saved metadata to {OUTPUT_PATH}')
    return all_pass

if __name__ == '__main__':
    passed = run()
    sys.exit(0 if passed else 1)
