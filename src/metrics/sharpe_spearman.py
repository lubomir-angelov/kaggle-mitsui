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
    For each date, compute Spearman correlation ACROSS TARGETS between
    target_* and prediction_* vectors. Then return mean/std (Sharpe) across dates.

    The frame may have multiple rows per date; we collapse them by mean first.
    """
    # collect and align columns by suffix
    target_cols = [c for c in df.columns if c.startswith(target_prefix)]
    assert len(target_cols) > 0, "No target_* columns found"

    suffixes   = [c[len(target_prefix):] for c in target_cols]
    pred_cols  = [f"{pred_prefix}{s}" for s in suffixes]
    missing    = [c for c in pred_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing prediction columns: {missing[:3]}...")

    # collapse to one row per date (in case there are duplicates)
    T = df.groupby(date_col, as_index=True)[target_cols].mean(numeric_only=True)
    P = df.groupby(date_col, as_index=True)[pred_cols].mean(numeric_only=True)

    daily_rhos = []
    for d in T.index:
        t = T.loc[d].to_numpy(dtype=float)
        p = P.loc[d].to_numpy(dtype=float)
        m = np.isfinite(t) & np.isfinite(p)
        if m.sum() >= 2:
            rho = spearmanr(t[m], p[m], nan_policy="omit").correlation
            if np.isfinite(rho):
                daily_rhos.append(rho)

    if not daily_rhos:
        return 0.0
    arr = np.asarray(daily_rhos, dtype=float)
    std = arr.std(ddof=0)
    if std == 0:
        return 0.0
    return float(arr.mean() / std)
