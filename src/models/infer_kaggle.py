# src/models/infer_kaggle.py
from __future__ import annotations
import os
from pathlib import Path
import numpy as np
import pandas as pd
import torch

from src.models.timexer import build_timexer
from src.models.adapters.lora import apply_lora_to_attn

REPO_ROOT   = Path(os.environ.get("REPO_ROOT", "/home/ubuntu/repos/kaggle-mitsui"))
DATA_MODELS = REPO_ROOT / "data/models"
CKPT_PATH   = DATA_MODELS / "timexer_kaggle_ft" / "kaggle_ft_v3" / "best.pt"   # <- your FT ckpt
COMP_DIR    = REPO_ROOT / "data/kaggle"                                  # where you download Kaggle data
WINDOW      = 160
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"

def pick_feature_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if (c != "date_id" and not c.startswith("target_"))]

@torch.no_grad()
def main():
    # 1) Load Kaggle files locally
    train  = pd.read_csv(COMP_DIR / "train.csv")
    test   = pd.read_csv(COMP_DIR / "test.csv")
    sub    = pd.read_csv(COMP_DIR / "sample_submission.csv")  # defines exact cols/order

    # 2) Build feature frame (train + test) and compute scaler from TRAIN ONLY
    full = pd.concat([train, test], ignore_index=True).sort_values("date_id")
    feature_cols = pick_feature_cols(full)
    mu   = train[feature_cols].median().to_numpy(np.float32)
    iqr  = (train[feature_cols].quantile(0.75) - train[feature_cols].quantile(0.25)).to_numpy(np.float32)
    std  = train[feature_cols].std(ddof=0).to_numpy(np.float32)
    sig  = np.where(iqr > 1e-6, iqr, std)
    sig  = np.where(sig > 1e-6, sig, 1.0).astype(np.float32)

    # 3) Rebuild model skeleton with correct shapes
    num_features = len(feature_cols)
    num_targets  = 424
    model = build_timexer(
        num_features=num_features, num_targets=num_targets,
        d_model=512, n_layers=6, n_heads=8, ffn_hidden=2048,
        dropout=0.1, patch_size=20, stride=20, use_cls=False
    )
    apply_lora_to_attn(model, r=8, alpha=16, freeze_base=True)
    model.to(DEVICE)

    # 4) Load FT checkpoint (works because you saved feature_cols/target_cols there)
    ckpt = torch.load(str(CKPT_PATH), map_location=DEVICE)
    model.load_state_dict(ckpt["model"], strict=True)

    # Enforce EXACT training column order (and fill any missing with 0)
    expected_feats = ckpt.get("feature_cols", feature_cols)
    for c in expected_feats:
        if c not in full.columns:
            full[c] = 0.0
    full = full[["date_id"] + list(expected_feats)]
    mu   = pd.Series(mu, index=expected_feats).to_numpy(np.float32)
    sig  = pd.Series(sig, index=expected_feats).to_numpy(np.float32)

    # 5) Prepare normalized design matrix (one row per date)
    F = full.sort_values("date_id").reset_index(drop=True)
    X = F[expected_feats].astype(np.float32).replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy()
    X = (X - mu) / sig
    X = np.clip(X, -8.0, 8.0)
    dates = F["date_id"].to_numpy(np.int32)

    # 6) Map date_id -> model window (use previous WINDOW-1 rows + current)
    #    Build predictions only for rows requested by sample_submission
    pred_cols = [c for c in sub.columns if c.startswith("prediction_")]
    want_dates = sub["date_id"].to_numpy() if "date_id" in sub.columns else np.unique(sub["row_id"])
    # If the competition uses row_id instead, parse the date_id part from row_id here.

    date_to_idx = {d:i for i,d in enumerate(dates)}
    preds = []
    batch_idx = []
    B = 2048  # large batch: (B, T, C)
    xs, ids = [], []

    for d in want_dates:
        i = date_to_idx.get(int(d))
        if i is None or i < WINDOW-1:
            # not enough history → predict zeros (or last-known)
            preds.append(np.zeros(len(pred_cols), dtype=np.float32))
            continue
        xt = torch.from_numpy(X[i-WINDOW+1:i+1]).unsqueeze(0)  # (1, T, C)
        xs.append(xt)
        ids.append(len(preds))
        preds.append(None)  # placeholder

        if len(xs) == B:
            xb = torch.cat(xs, dim=0).to(DEVICE)
            pb = model(xb).detach().cpu().numpy()
            for k, p in zip(ids, pb):
                preds[k] = p
            xs, ids = [], []

    if xs:
        xb = torch.cat(xs, dim=0).to(DEVICE)
        pb = model(xb).detach().cpu().numpy()
        for k, p in zip(ids, pb):
            preds[k] = p

    P = np.vstack([p if p is not None else np.zeros(len(pred_cols), np.float32) for p in preds])

    # 7) Fill submission (keep exact column order)
    out = sub.copy()
    for i, c in enumerate(pred_cols):
        out[c] = P[:, i].astype(np.float32)

    out.to_csv(REPO_ROOT / "submission.csv", index=False)
    print("Wrote:", REPO_ROOT / "submission.csv")

if __name__ == "__main__":
    main()
