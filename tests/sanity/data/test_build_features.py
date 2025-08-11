# tests/sanity/data/test_build_features.py
"""
Lightweight sanity checks for features_long.parquet produced by build_features.py

Env overrides:
- FEATURES_LONG_PATH: absolute path to features_long.parquet
- MIN_TARGET_COVERAGE: minimum allowed non-NA fraction per target column (default 0.20)
- MIN_FEATURE_NOTNA: minimum allowed non-NA fraction per feature column (default 0.50)
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

DEFAULT_PATH = "/home/ubuntu/repos/kaggle-mitsui/data/processed/features_long.parquet"
FEATURES_PATH = Path(os.environ.get("FEATURES_LONG_PATH", DEFAULT_PATH))


@pytest.fixture(scope="session")
def features_df() -> pd.DataFrame:
    if not FEATURES_PATH.exists():
        pytest.skip(f"features parquet not found at {FEATURES_PATH}. "
                    f"Set FEATURES_LONG_PATH to override.")
    df = pd.read_parquet(FEATURES_PATH)

    # Basic required columns present
    required = {"date", "symbol"}
    missing = required - set(df.columns)
    assert not missing, f"Missing required columns: {missing}"

    # Ensure datetime dtype and no timezone
    if not np.issubdtype(df["date"].dtype, np.datetime64):
        df["date"] = pd.to_datetime(df["date"], errors="raise")
    assert getattr(df["date"].dt, "tz", None) is None, "date column should be timezone-naive"

    return df


def test_no_duplicate_symbol_date(features_df: pd.DataFrame) -> None:
    dup_ct = int(features_df.duplicated(["symbol", "date"]).sum())
    assert dup_ct == 0, f"Found {dup_ct} duplicate (symbol,date) rows"


def test_target_columns_exist_and_numeric(features_df: pd.DataFrame) -> None:
    tcols = [c for c in features_df.columns if c.startswith("target_ret_fwd_")]
    assert tcols, "No forward target columns found (expected columns starting with 'target_ret_fwd_')."
    non_numeric = [c for c in tcols if not pd.api.types.is_numeric_dtype(features_df[c])]
    assert not non_numeric, f"Non-numeric target columns: {non_numeric}"


def test_target_coverage_reasonable(features_df: pd.DataFrame) -> None:
    min_cov = float(os.environ.get("MIN_TARGET_COVERAGE", "0.20"))
    tcols = [c for c in features_df.columns if c.startswith("target_ret_fwd_")]
    cov = features_df[tcols].notna().mean()
    # nothing should be entirely empty
    assert (cov > 0).all(), f"Some target columns are entirely NaN:\n{cov}"
    too_low = cov[cov < min_cov]
    assert too_low.empty, (
        f"Some target columns have coverage below {min_cov:.0%}:\n{too_low.sort_values()}"
    )


def test_feature_nan_share(features_df: pd.DataFrame) -> None:
    exclude = {"date", "symbol", "src", "unit", "frequency"}
    feat_cols = [c for c in features_df.columns if c not in exclude and not c.startswith("target_")]
    assert feat_cols, "No feature columns detected."

    min_notna = float(os.environ.get("MIN_FEATURE_NOTNA", "0.50"))
    notna = features_df[feat_cols].notna().mean()
    too_low = notna[notna < min_notna]
    assert too_low.empty, (
        f"Some feature columns have <{min_notna:.0%} non-NA values:\n{too_low.sort_values()}"
    )


def test_no_inf_values_in_numeric_columns(features_df: pd.DataFrame) -> None:
    num_cols = features_df.select_dtypes(include=[np.number]).columns
    if not len(num_cols):
        pytest.skip("No numeric columns found to test for inf values.")
    bad = ~np.isfinite(features_df[num_cols].to_numpy())
    has_bad = bool(bad.any())
    assert not has_bad, "Found inf/-inf/NaN beyond expected NaNs in numeric columns."


def test_business_day_density_sample(features_df: pd.DataFrame) -> None:
    """
    Sample a few symbols and ensure reasonable business-day density between their
    min/max dates. Monthly-resampled series should be dense (ffill). Sparse symbols
    with very short histories are skipped.
    """
    symbols = features_df["symbol"].drop_duplicates()
    if symbols.empty:
        pytest.skip("No symbols available.")

    sample_n = int(min(10, len(symbols)))
    sample_syms = symbols.sample(n=sample_n, random_state=0).tolist()

    for sym in sample_syms:
        g = features_df.loc[features_df["symbol"] == sym, "date"].sort_values()
        if len(g) < 20:
            # too short to assert density robustly
            continue
        bdays = pd.date_range(g.iloc[0], g.iloc[-1], freq="B")
        density = g.nunique() / max(1, len(bdays))
        assert density > 0.60, f"Low business-day density for {sym}: {density:.2%}"
