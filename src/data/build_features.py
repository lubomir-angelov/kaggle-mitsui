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

    # build per-symbol features
    feats_parts = []
    for sym, g in df.groupby("symbol", sort=False):
        g_out = _feat_for_symbol(g, horizons=args.horizons)
        # enforce minimum history so rolling features/targets exist
        if args.min_history > 0:
            g_out = g_out.iloc[args.min_history:].copy()
        feats_parts.append(g_out)

    features = pd.concat(feats_parts, ignore_index=True)

    # calendar + cross-sectional
    features = _add_calendar_feats(features)
    features = _add_cross_sectional(features)

    # tidy types
    num_cols = features.select_dtypes(include=[np.number]).columns.tolist()
    features = _downcast_float(features, num_cols)
    features["symbol"] = features["symbol"].astype("category")

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
