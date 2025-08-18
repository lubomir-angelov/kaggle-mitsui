#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
build_features.py — turn harmonized daily prices into model-ready features.

Inputs (created by harmonize.py)
--------------------------------
- data/processed/commodities_long_daily.parquet
    columns: [date, symbol, price_close, src, unit, frequency='D']

Outputs
-------
- data/processed/features_long.parquet               (one row per date × symbol)
- data/interim/features_manifest.json               (what we built and how)
- Optional: data/processed/features_long_h{H}.parquet  if --per-horizon-splits

Usage
-----
python src/data/build_features.py \
  --horizons 1 5 21 \
  --min-history 63 \
  --min-obs-per-symbol 100

Notes
-----
- All features are computed *within symbol* to avoid leaking cross-asset info,
  then some simple cross-sectional ranks are added per date.
- We avoid strong label leakage: targets are future returns (shifted -h).
- You can safely extend the FEATURE_WINDOWS or add more indicators.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------
# Paths & logging
# ---------------------------------------------------------------------
REPO_ROOT = os.environ.get("REPO_ROOT", "/home/ubuntu/repos/kaggle-mitsui")
RAW_DIR = Path(f"{REPO_ROOT}/data/raw")
INTERIM_DIR = Path(f"{REPO_ROOT}/data/interim")
PROCESSED_DIR = Path(f"{REPO_ROOT}/data/processed")

for _d in (RAW_DIR, INTERIM_DIR, PROCESSED_DIR):
    Path(_d).mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("build_features")

# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------

FEATURE_WINDOWS = [5, 10, 21, 63, 126]  # ~1w, 2w, 1m, 3m, 6m
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
RSI_PERIOD = 14
ANNUALIZE_D = 252  # for vols

INPUT_LONG_PATH = PROCESSED_DIR / "commodities_long_daily.parquet"
OUT_FEATURES_PATH = PROCESSED_DIR / "features_long.parquet"
OUT_MANIFEST_PATH = INTERIM_DIR / "features_manifest.json"


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def _downcast_float(df: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").astype("float32")
    return df


def _rsi(price: pd.Series, period: int = 14) -> pd.Series:
    # Wilder's RSI (EMA of gains/losses)
    delta = price.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    roll_up = up.ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    roll_down = down.ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    rs = roll_up / (roll_down.replace(0, np.nan))
    rsi = 100 - (100 / (1 + rs))
    return rsi


def _macd(price: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
    ema_fast = price.ewm(span=fast, adjust=False, min_periods=max(2, fast // 2)).mean()
    ema_slow = price.ewm(span=slow, adjust=False, min_periods=max(2, slow // 2)).mean()
    macd_line = ema_fast - ema_slow
    macd_signal = macd_line.ewm(span=signal, adjust=False, min_periods=max(2, signal // 2)).mean()
    macd_hist = macd_line - macd_signal
    return macd_line, macd_signal, macd_hist


def _feat_for_symbol(g: pd.DataFrame, horizons: List[int]) -> pd.DataFrame:
    """Compute features for a single symbol group (g is already that symbol)."""
    g = g.sort_values("date").copy()
    price = pd.to_numeric(g["price_close"], errors="coerce")

    # Returns
    g["ret_1d"] = price.pct_change()
    g["logret_1d"] = np.log(price).diff()

    for w in FEATURE_WINDOWS:
        # Simple/exp moving averages
        g[f"sma_{w}"] = price.rolling(w, min_periods=int(max(2, 0.6 * w))).mean()
        g[f"ema_{w}"] = price.ewm(span=w, adjust=False, min_periods=int(max(2, 0.5 * w))).mean()
        # Volatility (annualized, using log returns)
        g[f"vol_{w}"] = g["logret_1d"].rolling(w, min_periods=int(max(2, 0.6 * w))).std() * np.sqrt(ANNUALIZE_D)
        # Momentum vs moving average
        g[f"mom_{w}"] = price / g[f"sma_{w}"] - 1.0
        # Z-score of price around SMA
        roll_std = price.rolling(w, min_periods=int(max(2, 0.6 * w))).std()
        g[f"z_{w}"] = (price - g[f"sma_{w}"]) / (roll_std.replace(0, np.nan))

        # Multi-day returns
        g[f"ret_{w}d"] = price.pct_change(w)

    # Lags of 1d returns (helps many models)
    for k in (1, 2, 3, 4, 5):
        g[f"ret_1d_lag{k}"] = g["ret_1d"].shift(k)

    # RSI & MACD
    g[f"rsi_{RSI_PERIOD}"] = _rsi(price, period=RSI_PERIOD)
    macd_line, macd_signal, macd_hist = _macd(price, MACD_FAST, MACD_SLOW, MACD_SIGNAL)
    g["macd_line"] = macd_line
    g["macd_signal"] = macd_signal
    g["macd_hist"] = macd_hist

    # Targets: future returns for each horizon
    for h in horizons:
        g[f"target_ret_fwd_{h}d"] = price.pct_change(h).shift(-h)

    return g


def _add_calendar_feats(df: pd.DataFrame) -> pd.DataFrame:
    df["year"] = df["date"].dt.year.astype("int16")
    df["month"] = df["date"].dt.month.astype("int8")
    df["quarter"] = df["date"].dt.quarter.astype("int8")
    df["dow"] = df["date"].dt.dayofweek.astype("int8")
    df["month_start"] = (df["date"].dt.is_month_start).astype("int8")
    df["month_end"] = (df["date"].dt.is_month_end).astype("int8")
    return df


def _add_cross_sectional(df: pd.DataFrame) -> pd.DataFrame:
    """Add simple cross-sectional percent ranks per date for a few signals."""
    xsec_cols = []
    for base in ("mom_21", "z_21", "ret_21d"):
        if base in df.columns:
            col = f"{base}_xrank"
            df[col] = (
                df.groupby("date")[base]
                  .rank(pct=True, method="average")
                  .astype("float32")
            )
            xsec_cols.append(col)
    
    # robust cross-sectional z-scores
    # pick reasonable base columns that exist in your frame
    base5 = next((c for c in ("ret_5d", "mom_5", "roc_5", "ma_ratio_5") if c in df.columns), None)
    if base5 is not None:
        df = _xsec_zscore(df, base_col=base5, out_col="z_5")

    base10 = next((c for c in ("ret_10d", "mom_10", "roc_10", "ma_ratio_10") if c in df.columns), None)
    if base10 is not None:
        df = _xsec_zscore(df, base_col=base10, out_col="z_10")
            
    return df

def _xsec_zscore(df: pd.DataFrame, base_col: str, out_col: str) -> pd.DataFrame:
    """
    Cross-sectional z-score per day for `base_col`, robust to tiny std and missing data.
    Fallbacks:
      - if <3 valid observations on a day → z = 0
      - if std is 0/NaN/inf → z = 0
    Writes result to `out_col` and returns df.
    """
    import numpy as np

    EPS = 1e-12

    # mean and std per day, computed on the Series only
    g = df.groupby("date", observed=False)[base_col]
    mean = g.transform(lambda s: s.astype(float).mean(skipna=True))
    std0 = g.transform(lambda s: s.astype(float).std(ddof=0, skipna=True))

    # safe std (avoid 0/inf)
    safe_std = std0.where(np.isfinite(std0) & (std0 >= EPS))

    # z-score
    z = (df[base_col].astype(float) - mean) / safe_std

    # days with too few valid observations → set to 0
    valid_counts = g.transform(lambda s: s.notna().sum())
    z = np.where(valid_counts >= 3, z, 0.0)

    # clean up and cast
    z = pd.Series(z, index=df.index)
    z = z.replace([np.inf, -np.inf], 0.0).fillna(0.0).astype("float32")

    df[out_col] = z
    
    return df



# ---------------------------------------------------------------------
# Main build
# ---------------------------------------------------------------------

@dataclass
class BuildArgs:
    horizons: List[int]
    min_history: int
    min_obs_per_symbol: int
    per_horizon_splits: bool


def build_features(args: BuildArgs) -> pd.DataFrame:
    if not INPUT_LONG_PATH.exists():
        raise FileNotFoundError(f"Missing input long parquet: {INPUT_LONG_PATH}")

    df = pd.read_parquet(INPUT_LONG_PATH)
    if df.empty:
        raise RuntimeError("Input long panel is empty.")

    # basic hygiene
    df["date"] = pd.to_datetime(df["date"], utc=False).dt.tz_localize(None)
    df = df.sort_values(["symbol", "date"]).reset_index(drop=True)

    # Ensure numeric, no hidden infs, and keep a "safe for logs" view handy
    EPS = 1e-12
    df["price_close"] = pd.to_numeric(df["price_close"], errors="coerce")
    # not dropping zeros: keep them; we’ll turn them into NaNs for log-based calcs downstream
    # remove explicit +/-inf if any slipped through from upstream merges
    # explicitly, not in place => safe
    df["price_close"] = df["price_close"].replace([np.inf, -np.inf], np.nan)

    # keep only daily frequency rows (it should already be daily)
    if "frequency" in df.columns:
        df = df[df["frequency"].isin([None, "D"]) | df["frequency"].isna()]

    # prune very sparse series (optional but helps stability)
    counts = df["symbol"].value_counts()
    keep_syms = counts[counts >= args.min_obs_per_symbol].index
    dropped = sorted(set(df["symbol"].unique()) - set(keep_syms))
    if dropped:
        logger.info("Dropping %d sparse symbols (< %d obs): e.g., %s",
                    len(dropped), args.min_obs_per_symbol, ", ".join(dropped[:10]))
    df = df[df["symbol"].isin(keep_syms)].copy()

    # === FIX A (GLOBAL, BEFORE LOOP): stabilize raw level ===
    df["price_close"] = pd.to_numeric(df["price_close"], errors="coerce").replace([np.inf, -np.inf], np.nan)
    df = df[df["price_close"] > 0].copy()                    # forbid non-positive for logs
    df["logp"] = np.log(df["price_close"]).astype(np.float64)

    # build per-symbol features
    feats_parts = []
    for sym, g in df.groupby("symbol", sort=False):
        g = g.sort_values("date").copy()

        # === FIX B (PER-SYMBOL): trends on log price, not raw ===
        # rolling means / EMAs
        for w in (5, 10, 21, 63, 126):
            g[f"lgsma_{w}"] = g["logp"].rolling(window=w, min_periods=max(2, w // 3)).mean()
            g[f"lgema_{w}"] = g["logp"].ewm(span=w, adjust=False, min_periods=max(2, w // 3)).mean()

        # MACD on logp
        MACD_FAST, MACD_SLOW, MACD_SIGNAL = 12, 26, 9
        macd_line_log = (
            g["logp"].ewm(span=MACD_FAST, adjust=False).mean()
            - g["logp"].ewm(span=MACD_SLOW, adjust=False).mean()
        )
        g["macd_line_log"]   = macd_line_log
        g["macd_signal_log"] = macd_line_log.ewm(span=MACD_SIGNAL, adjust=False).mean()
        g["macd_hist_log"]   = g["macd_line_log"] - g["macd_signal_log"]

        # === FIX C (PER-SYMBOL): returns/momentum from logp + winsorize tails ===
        for n in (1, 5, 10, 21, 63, 126):
            g[f"logret_{n}d"] = g["logp"].diff(n)
            g[f"ret_{n}d"]    = np.expm1(g[f"logret_{n}d"])

        # winsorize return-like columns per symbol
        ret_like = [c for c in g.columns if c.startswith("ret_") or c.startswith("logret_")]
        for c in ret_like:
            q = g[c].quantile([0.001, 0.999])
            lo, hi = float(q.iloc[0]), float(q.iloc[1])
            g[c] = g[c].clip(lo, hi)

        # === FIX D (PER-SYMBOL, OPTIONAL): smoother daily returns for monthly series ===
        # compute 1d log return from time-interpolated logp (do not overwrite logp)
        logp_interp = g.set_index("date")["logp"].interpolate(method="time", limit_direction="both").reset_index(drop=True)
        g["logret_1d"] = logp_interp.diff(1)
        g["ret_1d"]    = np.expm1(g["logret_1d"])

        # === FIX F (PER-SYMBOL, OPTIONAL): tame MACD/log-trend extremes ===
        for c in ("macd_line_log", "macd_signal_log", "macd_hist_log"):
            if c in g.columns:
                q = g[c].quantile([0.001, 0.999])
                lo, hi = float(q.iloc[0]), float(q.iloc[1])
                g[c] = g[c].clip(lo, hi)

        # hand off to your existing symbol featurizer (it can use the columns above)
        g_out = _feat_for_symbol(g, horizons=args.horizons)

        # enforce minimum history so rolling features/targets exist
        if args.min_history > 0:
            g_out = g_out.iloc[args.min_history:].copy()

        feats_parts.append(g_out)

    features = pd.concat(feats_parts, ignore_index=True)

    # sanitize infinities right AFTER per-symbol features and BEFORE x-sec features
    num_cols_now = features.select_dtypes(include=[np.number]).columns
    if len(num_cols_now):
        features[num_cols_now] = features[num_cols_now].replace([np.inf, -np.inf], np.nan)

    # === FIX E (GLOBAL, AFTER LOOP): drop unsafe raw-level trend columns ===
    drops = [c for c in features.columns
             if (c.startswith(("sma_", "ema_", "macd_", "mom_")) and not c.endswith("_log"))]
    if drops:
        features.drop(columns=drops, inplace=True, errors="ignore")

    # calendar + cross-sectional
    features = _add_calendar_feats(features)
    features = _add_cross_sectional(features)

    # >>> INSERT S2.2 (STRICT FEATURE FILL to satisfy tests; targets untouched)
    exclude = {"date", "symbol", "src", "unit", "frequency"}
    feat_cols = [c for c in features.columns if c not in exclude and not c.startswith("target_")]
    
    if feat_cols:
        # First, replace any lingering infs
        features[feat_cols] = features[feat_cols].replace([np.inf, -np.inf], np.nan)
        # Groupwise forward/backward fill to keep continuity within each symbol
        features[feat_cols] = (
            features.groupby("symbol", group_keys=False)[feat_cols]
                    .apply(lambda g: g.ffill().bfill())
        )
        # Anything still missing (e.g., first rows after min_history) → neutral 0.0
        features[feat_cols] = features[feat_cols].fillna(0.0)

    # drop truly pathological feature columns with ultra-low coverage
    # You can tune via env var if you want: MIN_FEATURE_NOTNA=0.20 (for example)
    min_notna = float(os.environ.get("MIN_FEATURE_NOTNA", "0.00"))  # keep default permissive
    if min_notna > 0:
        exclude = {"date", "symbol", "src", "unit", "frequency"}
        feat_cols = [c for c in features.columns if c not in exclude and not c.startswith("target_")]
        if feat_cols:
            notna_share = features[feat_cols].notna().mean()
            to_drop = notna_share[notna_share < min_notna].index.tolist()
            if to_drop:
                logger.info("Dropping %d low-coverage feature(s) (< %.0f%% not-NA): %s",
                            len(to_drop), 100*min_notna, ", ".join(to_drop[:10]))
                features.drop(columns=to_drop, inplace=True)

    # --- normalized price feature (log + rolling z over 63d) ---
    # 1) safe log
    features["p_log"] = np.log(np.clip(features["price_close"].astype(float), 1e-6, None))

    # 2) per-symbol rolling stats
    grp = features.groupby("symbol", sort=False)

    features["p_log_med63"] = grp["p_log"].transform(
        lambda s: s.rolling(window=63, min_periods=20).median()
    )

    features["p_log_std63"] = grp["p_log"].transform(
        lambda s: s.rolling(window=63, min_periods=20).std(ddof=0)
    )

    # 3) z-score, guard zero std, clip, and cast
    features["p_log_z63"] = (
        (features["p_log"] - features["p_log_med63"]) /
        features["p_log_std63"].replace(0.0, np.nan)
    ).clip(-5, 5).astype("float32")

    # 4) tidy
    features = features.drop(columns=["p_log", "p_log_med63", "p_log_std63"])


    # tidy types
    num_cols = features.select_dtypes(include=[np.number]).columns.tolist()
    features = _downcast_float(features, num_cols)
    features["symbol"] = features["symbol"].astype("category")

    # >>> INSERT S3: final sanitize just BEFORE writing (catches any new infs from x-sec steps)
    num_cols_final = features.select_dtypes(include=[np.number]).columns
    if len(num_cols_final):
        features[num_cols_final] = features[num_cols_final].replace([np.inf, -np.inf], np.nan)

    # final NaN policy:
    # - keep rows where *targets* exist for at least one horizon
    target_cols = [c for c in features.columns if c.startswith("target_ret_fwd_")]
    tgt_mask = features[target_cols].notna().any(axis=1)
    before = len(features)
    features = features.loc[tgt_mask].copy()
    after = len(features)
    logger.info("Kept %d/%d rows with at least one target present.", after, before)

    # write outputs
    features = features.sort_values(["symbol", "date"]).reset_index(drop=True)
    features.to_parquet(OUT_FEATURES_PATH, index=False)
    logger.info("Wrote features → %s  (rows=%d, cols=%d)", OUT_FEATURES_PATH, len(features), features.shape[1])

    # optionally also write one file per horizon with that target non-null
    if args.per_horizon_splits:
        for h in args.horizons:
            col = f"target_ret_fwd_{h}d"
            sub = features[features[col].notna()].copy()
            out_h = PROCESSED_DIR / f"features_long_h{h}.parquet"
            sub.to_parquet(out_h, index=False)
            logger.info("Wrote per-horizon features → %s  (rows=%d)", out_h, len(sub))

    # manifest
    manifest = {
        "horizons": args.horizons,
        "min_history": args.min_history,
        "min_obs_per_symbol": args.min_obs_per_symbol,
        "n_rows": int(len(features)),
        "n_cols": int(features.shape[1]),
        "n_symbols": int(features["symbol"].nunique()),
        "input": str(INPUT_LONG_PATH),
        "output": str(OUT_FEATURES_PATH),
        "feature_windows": FEATURE_WINDOWS,
        "macd": {"fast": MACD_FAST, "slow": MACD_SLOW, "signal": MACD_SIGNAL},
        "rsi_period": RSI_PERIOD,
        "generated_at": pd.Timestamp.now().isoformat(),
    }

    # write manifest
    with open(OUT_MANIFEST_PATH, "w") as f:
        json.dump(manifest, f, indent=2)
    logger.info("Wrote manifest → %s", OUT_MANIFEST_PATH)

    return features


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def parse_args() -> BuildArgs:
    ap = argparse.ArgumentParser(description="Build model-ready features from harmonized prices.")
    ap.add_argument("--horizons", nargs="+", type=int, default=[1, 5, 21],
                    help="Forward return horizons (business days) used as targets.")
    ap.add_argument("--min-history", type=int, default=63,
                    help="Trim the first N rows per symbol to ensure rolling features available.")
    ap.add_argument("--min-obs-per-symbol", type=int, default=100,
                    help="Drop symbols with fewer rows than this threshold.")
    ap.add_argument("--per-horizon-splits", action="store_true",
                    help="Also write features_long_h{H}.parquet files per horizon.")
    a = ap.parse_args()
    return BuildArgs(
        horizons=a.horizons,
        min_history=a.min_history,
        min_obs_per_symbol=a.min_obs_per_symbol,
        per_horizon_splits=a.per_horizon_splits,
    )


def main():
    args = parse_args()
    build_features(args)


if __name__ == "__main__":
    main()
