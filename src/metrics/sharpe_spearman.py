# src/metrics/sharpe_spearman.py
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

def sharpe_of_daily_spearman(
    df: pd.DataFrame,
    target_prefix: str = "target_",
    pred_prefix: str   = "prediction_",
    date_col: str      = "date_id",
) -> float:
    """
    Kaggle Mitsui metric: for each date, compute the mean Spearman rank correlation
    across all targets; then return mean/std (Sharpe) across dates.

    df: must contain columns: date_id, target_*, prediction_*
    """
    # gather columns
    target_cols = [c for c in df.columns if c.startswith(target_prefix)]
    pred_cols   = [c for c in df.columns if c.startswith(pred_prefix)]
    assert len(target_cols) == len(pred_cols) and len(target_cols) > 0, "Mismatched target/pred columns"

    # align target_i with prediction_i by suffix
    suffix = [c.replace(target_prefix, "") for c in target_cols]
    pred_cols = [f"{pred_prefix}{s}" for s in suffix]
    for c in pred_cols:
        if c not in df.columns:
            raise ValueError(f"Missing prediction column: {c}")

    daily_scores = []
    for dt, g in df.groupby(date_col):
        # per target correlation on this date (across rows/assets)
        scores = []
        for tcol, pcol in zip(target_cols, pred_cols):
            tgt = g[tcol].values
            prd = g[pcol].values
            if np.all(np.isnan(tgt)) or np.all(np.isnan(prd)):
                continue
            # handle constant arrays / nan-safe
            if np.nanstd(tgt) == 0 or np.nanstd(prd) == 0:
                continue
            rho, _ = spearmanr(tgt, prd, nan_policy="omit")
            if np.isfinite(rho):
                scores.append(rho)
        if scores:
            daily_scores.append(np.mean(scores))

    if len(daily_scores) < 2:
        # std would be zero/undefined; return 0 to be safe in early training
        return 0.0

    arr = np.array(daily_scores, dtype=float)
    mean = arr.mean()
    std  = arr.std(ddof=0)
    if std == 0:
        return 0.0
    return float(mean / std)
