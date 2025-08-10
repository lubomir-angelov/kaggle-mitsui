#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
harmonize.py  —  build a canonical, daily commodity panel from heterogeneous raw sources.

What it does
------------
- Reads whatever raw files are already in data/raw/ (downloaded by download.py).
- Adapts each source into a common schema: [date, symbol, price_close, src, unit].
- Normalizes calendars (monthly/hourly → daily B-days) with sensible rules.
- Optionally applies symbol lookup mapping if data/raw/symbol_lookup.csv exists.
- Writes:
    data/interim/commodities_merged_daily.parquet    (wide, per symbol)
    data/processed/commodities_long_daily.parquet    (long/tidy)
    data/interim/source_manifest.csv                 (provenance of each row)

Notes
-----
- This is conservative: it won’t fail the whole run if one source is missing or fails to parse.
- The World Bank Pink Sheet (monthly) is treated as USD; most series are USD already.
- If you have FX series and non-USD prices, plug them into `apply_fx_normalization()`.

Requirements
------------
pip install pandas numpy python-dateutil openpyxl pyxlsb xlrd

Usage
-----
python harmonize.py
"""
from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


import re
import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta

# --------------------------------------------------------------------
# Paths & logging
# --------------------------------------------------------------------
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
logger = logging.getLogger("harmonize")

# --------------------------------------------------------------------
# Schema & helpers
# --------------------------------------------------------------------

CANON_COLUMNS = ["date", "symbol", "price_close", "src", "unit", "frequency"]
BUSINESS_CALENDAR_FREQ = "B"

# Optional user-maintained mapping file (raw_symbol → symbol)
SYMBOL_LK_PATH = RAW_DIR / "symbol_lookup.csv"

# A tiny built-in fallback mapping for common commodities
FALLBACK_SYMBOL_MAP = {
    # energy
    "crude oil, average spot price of brent, dubai and wti": "WTI",
    "crude oil, brent": "BRENT",
    "crude oil": "OIL",
    "natural gas, u.s.": "NG_US",
    "natural gas, europe": "NG_EU",
    "coal, australia": "COAL_AUS",
    # metals
    "aluminum": "AL",
    "copper": "CU",
    "iron ore": "FE",
    "nickel": "NI",
    "zinc": "ZN",
    "lead": "PB",
    "tin": "SN",
    # precious
    "gold": "XAU",
    "silver": "XAG",
    "platinum": "XPT",
    # agri (examples)
    "wheat, us hrw": "WHEAT",
    "maize": "CORN",
    "corn": "CORN",
    "soybeans": "SOY",
    "rice": "RICE",
    "sugar, world": "SUGAR",
    "coffee, other mild arabicas": "COFFEE",
}

DATE_COL_NAMES = {"date", "Date", "DATE", "timestamp", "Timestamp", "ds"}
PRICE_COL_CANDIDATES = [
    "price_close",
    "close",
    "Close",
    "Price",
    "Adj Close",
    "last",
    "value",
    "Value",
    "PRICE",
    "PX_LAST",
]

# --------------------------------------------------------------------
# Generic utilities
# --------------------------------------------------------------------

def _coerce_datetime(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce", utc=False).dt.tz_localize(None)


def _canonicalize_symbol(raw: str, user_map: Dict[str, str]) -> str:
    if not isinstance(raw, str):
        return ""
    key = raw.strip().lower()
    if key in user_map:
        return user_map[key]
    if key in FALLBACK_SYMBOL_MAP:
        return FALLBACK_SYMBOL_MAP[key]
    # Short alphanumeric slug as last resort
    sym = re.sub(r"[^A-Za-z0-9]+", "_", raw.upper()).strip("_")
    return sym[:20]


def _load_symbol_map() -> Dict[str, str]:
    if SYMBOL_LK_PATH.exists():
        try:
            m = pd.read_csv(SYMBOL_LK_PATH)
            m.columns = [c.strip().lower() for c in m.columns]
            # Expect columns like: raw_symbol, symbol
            if "raw_symbol" in m.columns and "symbol" in m.columns:
                return {str(r["raw_symbol"]).strip().lower(): str(r["symbol"]).strip() for _, r in m.iterrows()}
        except Exception as e:
            logger.warning("Failed to read symbol_lookup.csv: %s", e)
    return {}  # fallback used in _canonicalize_symbol


def _finalize_frame(df: pd.DataFrame, src: str, unit_default: str = "USD") -> pd.DataFrame:
    """Ensure schema, types, and drop invalid rows."""
    if df.empty:
        return df

    df = df.copy()
    # Ensure required columns exist
    for req in ("date", "symbol", "price_close"):
        if req not in df.columns:
            df[req] = np.nan

    df["date"] = _coerce_datetime(df["date"])
    df["symbol"] = df["symbol"].astype(str)
    df["price_close"] = pd.to_numeric(df["price_close"], errors="coerce")
    df["src"] = src
    if "unit" not in df.columns:
        df["unit"] = unit_default
    if "frequency" not in df.columns:
        df["frequency"] = None

    df = df.dropna(subset=["date", "symbol", "price_close"])
    df = df[df["symbol"].str.len() > 0]
    df = df.sort_values(["symbol", "date"]).reset_index(drop=True)
    return df[CANON_COLUMNS]

def _resample_to_business_daily(df: pd.DataFrame) -> pd.DataFrame:
    """
    Per symbol → business daily.
      - M: value on first business day of month, then ffill.
      - Q: value on first business day of quarter, then ffill.
      - A: value on first business day of year, then ffill.
      - H/sub-daily: daily mean.
      - D/None/unknown: last by B-day, ffill.
    """
    if df.empty:
        return df

    def _first_bday_range(start: pd.Timestamp, end: pd.Timestamp) -> pd.DatetimeIndex:
        return pd.date_range(start=start, end=end, freq=BUSINESS_CALENDAR_FREQ)

    def _place_first_bday(period_starts: pd.DatetimeIndex, values: pd.Series) -> pd.DataFrame:
        rows = []
        for ts, val in zip(period_starts, values):
            fb = pd.date_range(start=ts, end=ts + relativedelta(months=1, days=-1), freq=BUSINESS_CALENDAR_FREQ)
            if len(fb) > 0:
                rows.append((fb[0], val))
        if not rows:
            return pd.DataFrame(columns=["date", "price_close"])
        out = pd.DataFrame(rows, columns=["date", "price_close"]).set_index("date")
        return out

    def _resample_group(g: pd.DataFrame) -> pd.DataFrame:
        g = g.sort_values("date").set_index("date")
        freq = (g["frequency"].dropna().iloc[-1] if "frequency" in g.columns and not g["frequency"].dropna().empty else None)
        start, end = g.index.min(), g.index.max()
        if pd.isna(start) or pd.isna(end):
            return pd.DataFrame()

        if freq == "M":
            ms = pd.date_range(start=start, end=end, freq="MS")
            base = g.resample("MS").last()
            placed = _place_first_bday(ms, base["price_close"])
            bcal = _first_bday_range(start, end)
            out = placed.reindex(bcal).ffill()

        elif freq == "Q":
            qs = pd.date_range(start=start, end=end, freq="QS")
            base = g.resample("QS").last()
            # place on first business day of each quarter
            rows = []
            for ts, val in zip(qs, base["price_close"].reindex(qs)):
                fb = pd.date_range(start=ts, end=ts + relativedelta(months=3, days=-1), freq=BUSINESS_CALENDAR_FREQ)
                if len(fb) > 0:
                    rows.append((fb[0], val))
            out = pd.DataFrame(rows, columns=["date", "price_close"]).set_index("date")
            bcal = _first_bday_range(start, end)
            out = out.reindex(bcal).ffill()

        elif freq == "A":
            ys = pd.date_range(start=start, end=end, freq="AS")
            base = g.resample("AS").last()
            rows = []
            for ts, val in zip(ys, base["price_close"].reindex(ys)):
                fb = pd.date_range(start=ts, end=ts + relativedelta(years=1, days=-1), freq=BUSINESS_CALENDAR_FREQ)
                if len(fb) > 0:
                    rows.append((fb[0], val))
            out = pd.DataFrame(rows, columns=["date", "price_close"]).set_index("date")
            bcal = _first_bday_range(start, end)
            out = out.reindex(bcal).ffill()

        elif freq in ("H", "30T", "60T"):
            out = g.resample("B").mean(numeric_only=True)

        else:
            out = g.resample("B").last()
            out["price_close"] = out["price_close"].ffill()

        # carry metadata
        out["symbol"] = g["symbol"].iloc[0]
        out["src"] = g["src"].iloc[0]
        out["unit"] = g["unit"].iloc[0] if "unit" in g.columns else "USD"
        out["frequency"] = "D"
        return out.reset_index().rename(columns={"index": "date"})

    parts = [ _resample_group(g) for _, g in df.groupby("symbol", sort=False) ]
    out = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=CANON_COLUMNS)
    out = out.dropna(subset=["date", "symbol", "price_close"]).sort_values(["symbol", "date"]).reset_index(drop=True)
    return out



def _wide_from_long(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    w = df.pivot_table(index="date", columns="symbol", values="price_close", aggfunc="last")
    w = w.sort_index()
    return w


# --------------------------------------------------------------------
# Source adapters (each returns long/tidy frame with CANON_COLUMNS)
# --------------------------------------------------------------------
# --- replace your `adapt_worldbank_pink_sheet(...)` with this ---
def adapt_worldbank_pink_sheet_folder(folder: Path, symbol_map: Dict[str, str]) -> pd.DataFrame:
    """
    Parse multiple World Bank Pink Sheet workbooks from `folder`:
      - Monthly files (e.g., CMO-Historical-Data-Monthly.xlsx)
      - Annual files (e.g., CMO-Data-Annual.xlsx)
      - Older April/October 'historical' XLSX vintages

    Returns long frame with schema [date, symbol, price_close, src, unit, frequency]
    and resolves overlaps by: Monthly > Annual; within same freq keep newest vintage.
    """
    if not folder.exists():
        logger.info("World Bank folder not found: %s", folder)
        return pd.DataFrame(columns=CANON_COLUMNS)

    logger.info("Scanning World Bank Pink Sheet folder: %s", folder)
    files = sorted([p for p in folder.glob("*.xlsx") if p.is_file()])
    if not files:
        logger.info("No .xlsx files in %s", folder)
        return pd.DataFrame(columns=CANON_COLUMNS)

    def _read_all_sheets(xlsx: Path) -> Dict[str, pd.DataFrame]:
        # Some files have a one-row banner; try header=0 then header=1
        try:
            return pd.read_excel(xlsx, sheet_name=None, header=0)
        except Exception:
            try:
                return pd.read_excel(xlsx, sheet_name=None, header=1)
            except Exception as e:
                logger.warning("Failed reading %s: %s", xlsx.name, e)
                return {}

    # patterns
    re_month = re.compile(r"^\d{4}[Mm]\d{2}$")  # 1960M01, 2024m09
    re_year  = re.compile(r"^\d{4}$")           # 1960, 2024

    frames: List[pd.DataFrame] = []

    for xlsx in files:
        sheets = _read_all_sheets(xlsx)
        if not sheets:
            continue

        vintage = pd.Timestamp(xlsx.stat().st_mtime, unit="s")
        file_lower = xlsx.name.lower()

        for sname, df in sheets.items():
            if not isinstance(df, pd.DataFrame) or df.empty:
                continue
            # flatten columns to plain strings to avoid MultiIndex → DataFrame selections
            df.columns = [
                            "_".join(map(str, c)).strip() if isinstance(c, tuple) else str(c).strip()
                            for c in df.columns
                        ]
            cols_before = list(df.columns)

            # 1) drop duplicated columns (keeps the first)
            df = df.loc[:, ~pd.Index(df.columns).duplicated()].copy()

            # 2) re-materialize cols list after dedupe
            cols = [str(c) for c in df.columns]

            # (optional) quick guard: if we dropped stuff, you can log it
            dropped = set(cols_before) - set(cols)
            if dropped:
                logger.debug("Deduped duplicate columns on %s / %s (dropped: %s)",
                             xlsx.name, sname, list(dropped)[:5])

            # Identify a commodity/series name column
            name_candidates = [
                "Commodity", "Commodity name", "Series Name", "Series", "Indicator Name",
                "Description", "Item", "Name"
            ]
            name_col = next((c for c in name_candidates if c in cols), df.columns[0])

            # Heuristic: decide frequency by inspecting columns
            month_cols = [c for c in cols if re_month.match(str(c))]
            year_cols  = [c for c in cols if re_year.match(str(c))]

            freq = None
            if month_cols:
                freq = "M"
                subset = [name_col] + month_cols
            elif year_cols:
                freq = "A"
                subset = [name_col] + year_cols
            else:
                # Try classic Year column + many numeric columns (annual)
                # ==== annual branch (replace your current Year+numeric logic with this) ====
                low = [c.lower() for c in cols]
                if "year" in low:
                    ycol = df.columns[low.index("year")]
                    # pick numeric columns *only* using select_dtypes to avoid df[c] being a DataFrame
                    num_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c != ycol]
                    if len(num_cols) >= 1:
                        freq = "A"
                        subset = [name_col, ycol] + list(num_cols)
                    else:
                        continue
                else:
                    continue  # skip sheet

            try:
                slim = df[subset].copy()
            except Exception:
                # If header row was off, skip this sheet
                continue

            slim = slim.rename(columns={name_col: "raw_symbol"})

            if freq == "M":
                long = slim.melt(id_vars=["raw_symbol"], var_name="date_raw", value_name="price_close")
                # parse 1960M01 → 1960-01-01
                def _parse_m(x):
                    m = re.match(r"^(\d{4})[Mm](\d{2})$", str(x))
                    if m:
                        return pd.Timestamp(int(m.group(1)), int(m.group(2)), 1)
                    # sometimes monthly tabs use YYYY-MM
                    try:
                        dt = pd.to_datetime(x, errors="coerce")
                        if pd.notna(dt):
                            # coerce to month-start
                            return pd.Timestamp(dt.year, dt.month, 1)
                    except Exception:
                        pass
                    return pd.NaT

                long["date"] = long["date_raw"].map(_parse_m)
            else:  # Annual
                # Two forms:
                # 1) Columns are 4-digit years: melt like monthly
                if any(c in year_cols for c in slim.columns):
                    long = slim.melt(id_vars=["raw_symbol"], var_name="year", value_name="price_close")
                    long["date"] = pd.to_datetime(long["year"].astype(str) + "-01-01", errors="coerce")
                else:
                    # 2) Has a 'Year' column + many numeric series columns (rare)
                    ycol = next((c for c in slim.columns if str(c).lower() == "year"), None)
                    if ycol is None:
                        continue
                    tmp = slim.rename(columns={ycol: "year"})
                    long = tmp.melt(id_vars=["raw_symbol", "year"], var_name="series", value_name="price_close")
                    long["date"] = pd.to_datetime(tmp["year"].astype(str) + "-01-01", errors="coerce")

            long = long.dropna(subset=["date"])
            long["symbol"] = long["raw_symbol"].map(lambda s: _canonicalize_symbol(str(s), symbol_map))
            long["unit"] = "USD"
            long["src"] = "worldbank_pinksheet"
            long["frequency"] = freq
            long["__vintage"] = vintage  # used for tie-breaks
            frames.append(long[["date", "symbol", "price_close", "src", "unit", "frequency", "__vintage"]])

    if not frames:
        logger.warning("No usable Pink Sheet tables found across XLSX files.")
        return pd.DataFrame(columns=CANON_COLUMNS)

    wb_all = pd.concat(frames, ignore_index=True)
    wb_all = _finalize_frame(wb_all, src="worldbank_pinksheet", unit_default="USD")

    # Resolve overlaps: prefer Monthly over Annual, and newest vintage inside same freq
    freq_rank = {"M": 0, "Q": 1, "A": 2, None: 9}
    wb_all["__freq_rank"] = wb_all["frequency"].map(freq_rank).fillna(9).astype(int)
    # If __vintage missing (shouldn't), fill with epoch
    wb_all["__vintage"] = wb_all.get("__vintage", pd.Timestamp("1970-01-01"))

    wb_all = (
        wb_all.sort_values(["symbol", "date", "__freq_rank", "__vintage"], ascending=[True, True, True, False])
              .drop_duplicates(subset=["symbol", "date"], keep="first")
              .drop(columns=["__freq_rank", "__vintage"])
              .reset_index(drop=True)
    )
    return wb_all



def adapt_kaggle_commodity_prices(folder: Path, symbol_map: Dict[str, str]) -> pd.DataFrame:
    """
    Adapt daily commodity CSVs from the Kaggle commodity prices dataset into the canonical schema.
    We scan CSV files and try to infer date & price columns.
    """
    if not folder.exists():
        logger.info("Kaggle commodity prices folder not found: %s", folder)
        return pd.DataFrame(columns=CANON_COLUMNS)

    logger.info("Parsing Kaggle commodity prices from %s", folder)
    parts: List[pd.DataFrame] = []

    for p in sorted(folder.rglob("*.csv")):
        try:
            df = pd.read_csv(p)
        except Exception as e:
            logger.warning("Skip unreadable CSV %s: %s", p.name, e)
            continue
        if df.empty:
            continue

        # date column
        dcol = next((c for c in df.columns if c in DATE_COL_NAMES), None)
        if dcol is None:
            # try common alternatives
            for c in df.columns:
                if str(c).lower() in ("date", "datetime", "time"):
                    dcol = c
                    break
        if dcol is None:
            continue

        # price column
        pcol = next((c for c in PRICE_COL_CANDIDATES if c in df.columns), None)
        if pcol is None:
            # try last numeric column
            num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
            pcol = num_cols[-1] if num_cols else None
        if pcol is None:
            continue

        # symbol source: filename or a 'symbol'/'commodity' column
        if "symbol" in df.columns:
            raw_symbol = df["symbol"].astype(str)
        elif "Commodity" in df.columns:
            raw_symbol = df["Commodity"].astype(str)
        else:
            raw_symbol = pd.Series([p.stem] * len(df))

        tmp = pd.DataFrame(
            {
                "date": _coerce_datetime(df[dcol]),
                "symbol": [ _canonicalize_symbol(s, symbol_map) for s in raw_symbol ],
                "price_close": pd.to_numeric(df[pcol], errors="coerce"),
                "src": "kaggle_commodities",
                "unit": "USD",
                "frequency": "D",
            }
        )
        tmp = tmp.dropna(subset=["date", "symbol", "price_close"])
        parts.append(tmp)

    if not parts:
        return pd.DataFrame(columns=CANON_COLUMNS)

    out = pd.concat(parts, ignore_index=True)
    out = _finalize_frame(out, src="kaggle_commodities", unit_default="USD")
    return out


def adapt_agri_prices_kaggle(folder: Path, symbol_map: Dict[str, str]) -> pd.DataFrame:
    """Adapt agricultural prices CSVs (irregular or daily)."""
    if not folder.exists():
        logger.info("Agri prices folder not found: %s", folder)
        return pd.DataFrame(columns=CANON_COLUMNS)

    logger.info("Parsing Kaggle agricultural prices from %s", folder)
    parts: List[pd.DataFrame] = []
    for p in sorted(folder.rglob("*.csv")):
        try:
            df = pd.read_csv(p)
        except Exception as e:
            logger.warning("Skip unreadable CSV %s: %s", p.name, e)
            continue
        if df.empty:
            continue

        # dates
        dcol = next((c for c in df.columns if c in DATE_COL_NAMES), None)
        if dcol is None:
            continue

        # price
        pcol = next((c for c in PRICE_COL_CANDIDATES if c in df.columns), None)
        if pcol is None:
            num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
            pcol = num_cols[-1] if num_cols else None
        if pcol is None:
            continue

        # symbol
        if "commodity" in (c.lower() for c in df.columns):
            cname = next(c for c in df.columns if c.lower() == "commodity")
            raw_symbol = df[cname].astype(str)
        else:
            raw_symbol = pd.Series([p.stem] * len(df))

        tmp = pd.DataFrame(
            {
                "date": _coerce_datetime(df[dcol]),
                "symbol": [ _canonicalize_symbol(s, symbol_map) for s in raw_symbol ],
                "price_close": pd.to_numeric(df[pcol], errors="coerce"),
                "src": "kaggle_agri",
                "unit": "USD",
                "frequency": None,
            }
        )
        tmp = tmp.dropna(subset=["date", "symbol", "price_close"])
        parts.append(tmp)

    out = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=CANON_COLUMNS)
    out = _finalize_frame(out, src="kaggle_agri", unit_default="USD")
    return out


def adapt_eia_mer_folder(folder: Path, symbol_map: Dict[str, str]) -> pd.DataFrame:
    """
    Parse *all* EIA MER tables from a folder (CSV/XLSX/XLS).
    Heuristics handled:
      - Annual wide (Year, 1990..2025...)
      - Monthly wide (Year, Jan..Dec)
      - Tidy (Year + Month columns)
      - Multi-series tables (uses a 'name' column if present; else builds from file/sheet)

    Returns: [date, symbol, price_close, src, unit, frequency]
    (frequency: 'M' for monthly, 'A' for annual)
    """

    if not folder.exists():
        logger.info("EIA MER folder missing: %s", folder)
        return pd.DataFrame(columns=CANON_COLUMNS)

    files = sorted(list(folder.glob("*.csv")) +
                   list(folder.glob("*.xlsx")) +
                   list(folder.glob("*.xls")))
    if not files:
        logger.info("No MER files in %s", folder)
        return pd.DataFrame(columns=CANON_COLUMNS)

    month_map = {
        "jan": 1, "january": 1, "feb": 2, "february": 2, "mar": 3, "march": 3,
        "apr": 4, "april": 4, "may": 5, "jun": 6, "june": 6, "jul": 7, "july": 7,
        "aug": 8, "august": 8, "sep": 9, "sept": 9, "september": 9,
        "oct":10, "october":10, "nov":11, "november":11, "dec":12, "december":12,
    }
    re_year  = re.compile(r"^\d{4}$")
    re_monthcode = re.compile(r"^(\d{4})[-/]?([01]\d)$")  # e.g., 2024-07 or 202407

    def _flatten_cols(df: pd.DataFrame) -> list[str]:
        df.columns = [
            "_".join(map(str, c)).strip() if isinstance(c, tuple) else str(c).strip()
            for c in df.columns
        ]
        return [str(c) for c in df.columns]

    def _pick_name_cols(df: pd.DataFrame, ycol: str | None) -> list[str]:
        # Prefer semantic name columns; else any object cols except Year/Month.
        candidates = [
            "Series", "Series Name", "Description", "Commodity", "Product",
            "MSN", "Category", "Region", "Area", "Data Series", "Label", "Name"
        ]
        cols = list(df.columns)
        low  = [c.lower() for c in cols]
        for c in candidates:
            if c in cols:
                return [c]
            if c.lower() in low:
                return [cols[low.index(c.lower())]]
        obj_cols = df.select_dtypes(include=["object"]).columns.tolist()
        obj_cols = [c for c in obj_cols if c != ycol and c.lower() not in ("year", "month", "date")]
        # cap to 2 to avoid over-IDs
        return obj_cols[:2]

    def _rows_to_symbol(row: pd.Series, name_cols: list[str], fallback: str) -> str:
        if name_cols:
            parts = [str(row[c]) for c in name_cols if pd.notna(row.get(c)) and str(row[c]).strip()]
            if parts:
                return " / ".join(parts)
        return fallback

    frames: list[pd.DataFrame] = []

    def _read_sheets_any_header(path: Path) -> dict[str, pd.DataFrame]:
        """Try several header rows; return {sheet_name: DataFrame}."""
        out = {}
        if path.suffix.lower() == ".csv":
            try:
                df = pd.read_csv(path, low_memory=False)
                out[path.stem] = df
            except Exception as e:
                logger.debug("CSV read failed %s: %s", path.name, e)
            return out

        # Excel: sweep header rows 0..10
        engine = "openpyxl" if path.suffix.lower() in (".xlsx", ".xlsm") else "xlrd"
        for hdr in range(0, 11):
            try:
                sheets = pd.read_excel(path, sheet_name=None, header=hdr, engine=engine)
                if not isinstance(sheets, dict):
                    continue
                # accept sheets that have a Year col or >=3 month cols or 4-digit year columns
                accepted = {}
                for sname, df0 in sheets.items():
                    if not isinstance(df0, pd.DataFrame) or df0.empty:
                        continue
                    cols0 = [str(c).strip() for c in df0.columns]
                    low0  = [c.lower() for c in cols0]
                    has_year = "year" in low0
                    month_count = sum(1 for c in low0 if c in month_map)
                    has_yearcols = any(re_year.match(str(c)) for c in cols0)
                    if has_year or month_count >= 3 or has_yearcols:
                        accepted[sname] = df0
                if accepted:
                    out.update(accepted)
                    # keep going; sometimes different sheets need different header rows
            except Exception:
                pass

        # Fallback: header=None and try to promote header row
        if not out:
            try:
                raw = pd.read_excel(path, sheet_name=None, header=None, engine=engine)
                for sname, df0 in raw.items():
                    if not isinstance(df0, pd.DataFrame) or df0.empty:
                        continue
                    # find a row that looks like header: contains 'Year' or many month names
                    best = None
                    best_score = -1
                    for ridx in range(min(15, len(df0))):
                        row = df0.iloc[ridx].astype(str).str.strip()
                        low = row.str.lower().tolist()
                        score = 0
                        score += 5 if "year" in low else 0
                        score += sum(1 for v in low if v in month_map)
                        score += sum(1 for v in row if re_year.match(v))
                        if score > best_score:
                            best = ridx
                            best_score = score
                    if best is not None and best_score >= 3:
                        hdr = best
                        dfh = df0.copy()
                        dfh.columns = dfh.iloc[hdr].astype(str).str.strip()
                        dfh = dfh.iloc[hdr+1:].reset_index(drop=True)
                        out[sname] = dfh
            except Exception:
                pass

        return out

    for path in files:
        sheets = _read_sheets_any_header(path)
        if not sheets:
            logger.debug("No parsable sheets in %s", path.name)
            continue

        for sname, df in sheets.items():
            if not isinstance(df, pd.DataFrame) or df.empty:
                continue

            cols = _flatten_cols(df)
            low  = [c.lower() for c in cols]

            # Normalize common noise rows
            df = df.dropna(how="all").copy()

            # Detect Year column
            ycol = None
            for key in ("Year", "YEAR", "year"):
                if key in df.columns:
                    ycol = key
                    break
            if ycol is None and "yyyymm" in low:
                # tables with YYYYMM numeric value
                yyyymm_col = df.columns[low.index("yyyymm")]
                # convert to date
                ser = pd.to_numeric(df[yyyymm_col], errors="coerce").dropna().astype(int)
                dt = pd.to_datetime(ser.astype(str), format="%Y%m", errors="coerce")
                tidy = pd.DataFrame({"date": dt, "raw_symbol": path.stem, "price_close": df.select_dtypes(include=[np.number]).iloc[:, -1]})
                tidy["symbol"] = tidy["raw_symbol"].map(lambda s: _canonicalize_symbol(s, symbol_map))
                tidy["src"] = "eia_mer"
                tidy["unit"] = None
                tidy["frequency"] = "M"
                frames.append(tidy[["date", "symbol", "price_close", "src", "unit", "frequency"]])
                continue

            # Monthly wide (Year + Jan..Dec)
            month_cols = []
            for c in df.columns:
                cl = str(c).strip().lower()
                if cl in month_map:
                    month_cols.append(c)
            # Some tables use full month names with year separated; ensure enough months present
            is_monthly_wide = ycol is not None and len(month_cols) >= 3

            # Annual wide (Year + 1990..2025 across columns OR tidy year/value)
            numeric_year_cols = [c for c in df.columns if re_year.match(str(c))]

            # Tidy monthly (Year + Month columns)
            has_tidy_month = (ycol is not None) and any(c.lower() == "month" for c in df.columns)

            # Pull units if a dedicated column exists
            unit_col = None
            for u in ("Unit", "Units", "unit", "units"):
                if u in df.columns:
                    unit_col = u
                    break

            fallback_name = f"{path.stem}:{sname}"

            if is_monthly_wide:
                name_cols = _pick_name_cols(df, ycol)
                keep = [ycol] + name_cols + month_cols
                slim = df[keep].copy()

                long = slim.melt(id_vars=[ycol] + name_cols,
                                 value_vars=month_cols,
                                 var_name="month_name",
                                 value_name="price_close")
                long = long.dropna(subset=[ycol, "month_name"])

                long["month_num"] = long["month_name"].map(lambda x: month_map.get(str(x).strip().lower(), np.nan))
                long = long.dropna(subset=["month_num"])
                long["date"] = pd.to_datetime(dict(
                    year=pd.to_numeric(long[ycol], errors="coerce").astype("Int64"),
                    month=long["month_num"].astype(int),
                    day=1
                ), errors="coerce")
                long = long.dropna(subset=["date"])

                long["raw_symbol"] = long.apply(lambda r: _rows_to_symbol(r, name_cols, fallback_name), axis=1)
                long["symbol"] = long["raw_symbol"].map(lambda s: _canonicalize_symbol(str(s), symbol_map))
                long["src"] = "eia_mer"
                long["unit"] = long[unit_col] if unit_col else None
                long["frequency"] = "M"

                frames.append(long[["date", "symbol", "price_close", "src", "unit", "frequency"]])
                continue

            if has_tidy_month:
                name_cols = _pick_name_cols(df, ycol)
                mcol = next(c for c in df.columns if c.lower() == "month")
                keep = [ycol, mcol] + name_cols + [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
                slim = df[keep].dropna(subset=[ycol, mcol]).copy()

                # choose the last numeric column as the main series value
                num_cols = [c for c in slim.columns if pd.api.types.is_numeric_dtype(slim[c])]
                val_col = num_cols[-1] if num_cols else None
                if val_col is None:
                    continue

                # Month could be string ('Jan') or number (1..12)
                def _to_month(v):
                    if pd.isna(v):
                        return np.nan
                    s = str(v).strip().lower()
                    if s.isdigit():
                        n = int(s)
                        return n if 1 <= n <= 12 else np.nan
                    return month_map.get(s, np.nan)

                slim["month_num"] = slim[mcol].map(_to_month)
                slim = slim.dropna(subset=["month_num"])

                slim["date"] = pd.to_datetime(dict(
                    year=pd.to_numeric(slim[ycol], errors="coerce").astype("Int64"),
                    month=slim["month_num"].astype(int),
                    day=1
                ), errors="coerce")
                slim = slim.dropna(subset=["date"])

                slim["raw_symbol"] = slim.apply(lambda r: _rows_to_symbol(r, name_cols, fallback_name), axis=1)
                out = slim.rename(columns={val_col: "price_close"})

                out["symbol"] = out["raw_symbol"].map(lambda s: _canonicalize_symbol(str(s), symbol_map))
                out["src"] = "eia_mer"
                out["unit"] = out[unit_col] if unit_col else None
                out["frequency"] = "M"

                frames.append(out[["date", "symbol", "price_close", "src", "unit", "frequency"]])
                continue

            if ycol is not None and numeric_year_cols:
                # Columns are years: melt to 'A'
                name_cols = _pick_name_cols(df, ycol=None)  # there may not be series names; that's fine
                keep = name_cols + numeric_year_cols
                slim = df[keep].copy()

                long = slim.melt(id_vars=name_cols,
                                 var_name="year",
                                 value_name="price_close")
                long["date"] = pd.to_datetime(long["year"].astype(str) + "-01-01", errors="coerce")
                long = long.dropna(subset=["date"])

                long["raw_symbol"] = long.apply(lambda r: _rows_to_symbol(r, name_cols, fallback_name), axis=1)
                long["symbol"] = long["raw_symbol"].map(lambda s: _canonicalize_symbol(str(s), symbol_map))
                long["src"] = "eia_mer"
                long["unit"] = long[unit_col] if unit_col else None
                long["frequency"] = "A"

                frames.append(long[["date", "symbol", "price_close", "src", "unit", "frequency"]])
                continue

            # If we get here, try a last generic path: a single numeric time-like column
            # e.g., a 'Date' column or 'YYYY-MM' headers across columns
            # Try YYYYMM headers across columns:
            ym_cols = [c for c in df.columns if re_monthcode.match(str(c))]
            if ym_cols:
                name_cols = _pick_name_cols(df, ycol=None)
                slim = df[name_cols + ym_cols].copy()
                long = slim.melt(id_vars=name_cols, var_name="ym", value_name="price_close")
                long["date"] = pd.to_datetime(long["ym"].astype(str).str.replace(r"[-/]", "", regex=True),
                                              format="%Y%m", errors="coerce")
                long = long.dropna(subset=["date"])
                long["raw_symbol"] = long.apply(lambda r: _rows_to_symbol(r, name_cols, fallback_name), axis=1)
                long["symbol"] = long["raw_symbol"].map(lambda s: _canonicalize_symbol(str(s), symbol_map))
                long["src"] = "eia_mer"
                long["unit"] = long[unit_col] if unit_col else None
                long["frequency"] = "M"
                frames.append(long[["date", "symbol", "price_close", "src", "unit", "frequency"]])
                continue

            # Otherwise skip this sheet
            logger.debug("Unrecognized MER layout → skip: %s / %s", path.name, sname)

    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=CANON_COLUMNS)
    if out.empty:
        logger.info("No rows parsed from MER folder %s", folder)
        return out

    # finalize; don't force USD because MER mixes units — let unit be None/column if present
    out = _finalize_frame(out, src="eia_mer", unit_default=None)

    # Prefer monthly where present; if both A and M exist for same symbol & date, keep M
    out["__rank"] = out["frequency"].map({"M": 0, "A": 1}).fillna(2).astype(int)
    out = (out.sort_values(["symbol", "date", "__rank"])
              .drop_duplicates(["symbol", "date", "src"], keep="first")
              .drop(columns="__rank")
              .reset_index(drop=True))

    logger.info("Got %d rows from eia_mer (files=%d)", len(out), len(files))
    return out


def adapt_kaggle_etf_stock(folder: Path, symbol_map: Dict[str, str]) -> pd.DataFrame:
    """
    Adapt US stocks/ETFs dataset: produce daily close for symbols that look like commodities/ETFs.
    This is auxiliary (cross-asset covariates), not core commodity series.
    """
    if not folder.exists():
        logger.info("ETF/Stock folder not found: %s", folder)
        return pd.DataFrame(columns=CANON_COLUMNS)

    logger.info("Parsing ETF/Stock price files from %s", folder)
    parts: List[pd.DataFrame] = []
    for p in sorted(folder.rglob("*.csv")):
        try:
            df = pd.read_csv(p)
        except Exception:
            continue
        if df.empty:
            continue

        # Must have date and close
        dcol = next((c for c in df.columns if c in DATE_COL_NAMES), None)
        pcol = next((c for c in df.columns if c in ("Close", "Adj Close", "PX_LAST", "close")), None)
        sym_col = next((c for c in df.columns if c.lower() in ("symbol", "ticker")), None)

        if not dcol or not pcol:
            continue

        if sym_col:
            raw_symbol = df[sym_col].astype(str)
        else:
            raw_symbol = pd.Series([p.stem] * len(df))

        tmp = pd.DataFrame(
            {
                "date": _coerce_datetime(df[dcol]),
                "symbol": [ _canonicalize_symbol(s, symbol_map) for s in raw_symbol ],
                "price_close": pd.to_numeric(df[pcol], errors="coerce"),
                "src": "kaggle_etf_stock",
                "unit": "USD",
                "frequency": "D",
            }
        )
        tmp = tmp.dropna(subset=["date", "symbol", "price_close"])
        parts.append(tmp)

    out = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=CANON_COLUMNS)
    out = _finalize_frame(out, src="kaggle_etf_stock", unit_default="USD")
    return out


# --------------------------------------------------------------------
# FX normalization (optional hook)
# --------------------------------------------------------------------

def apply_fx_normalization(df: pd.DataFrame, fx_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """
    If you have non-USD prices and an FX panel, convert to USD.
    Expected fx_df: columns ['date','ccy','usd_per_ccy'] or similar.
    This stub currently returns df unchanged since most sources are USD.
    """
    return df


# --------------------------------------------------------------------
# Merge & precedence
# --------------------------------------------------------------------

@dataclass
class SourceSpec:
    name: str
    path: Path
    adapter: callable
    precedence: int  # lower number = preferred when overlapping same (date, symbol)


def build_harmonized_panel() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns:
        long_daily : tidy long DataFrame [date, symbol, price_close, src, unit, frequency='D']
        wide_daily : wide DataFrame, index=date, columns=symbol
    """
    symbol_map = _load_symbol_map()

    sources: List[SourceSpec] = [
        # name, path, adapter, precedence
        SourceSpec("kaggle_commodities", RAW_DIR / "kaggle_commodities_prices", adapt_kaggle_commodity_prices, 0),
        SourceSpec("worldbank", RAW_DIR / "worldbank", adapt_worldbank_pink_sheet_folder, 1),
        SourceSpec("eia_mer", RAW_DIR / "eia_energy" / "mer_zip", adapt_eia_mer_folder, 2),
        SourceSpec("kaggle_agri", RAW_DIR / "kaggle_agri_prices_2019", adapt_agri_prices_kaggle, 3),
        SourceSpec("kaggle_etf_stock", RAW_DIR / "kaggle_etf_stock", adapt_kaggle_etf_stock, 5),
    ]

    long_frames: List[pd.DataFrame] = []
    manifest_rows: List[Dict] = []

    for spec in sources:
        try:
            df_src = spec.adapter(spec.path, symbol_map)
            if df_src.empty:
                logger.info("No rows from %s", spec.name)
                continue
            df_src["__precedence"] = spec.precedence
            long_frames.append(df_src)
            manifest_rows.append({"source": spec.name, "rows": len(df_src), "path": str(spec.path)})
            logger.info("Got %d rows from %s", len(df_src), spec.name)
        except Exception as e:
            logger.error("Adapter failed for %s: %s", spec.name, e, exc_info=False)

    if not long_frames:
        logger.warning("No sources produced data. Exiting with empty frames.")
        return pd.DataFrame(columns=CANON_COLUMNS), pd.DataFrame()

    long_all = pd.concat(long_frames, ignore_index=True)

    # FX normalization hook (currently no-op)
    long_all = apply_fx_normalization(long_all, fx_df=None)

    # Resample each source to business-daily first (keeps provenance)
    long_daily_parts: List[pd.DataFrame] = []
    for src, g in long_all.groupby("src", sort=False):
        long_daily_parts.append(_resample_to_business_daily(g[CANON_COLUMNS]))

    long_daily = pd.concat(long_daily_parts, ignore_index=True)
    long_daily["__precedence"] = long_daily["src"].map({s.name: s.precedence for s in sources}).fillna(99).astype(int)

    # When multiple sources provide the same (date, symbol), keep the lowest precedence
    long_daily = (
        long_daily
        .sort_values(["date", "symbol", "__precedence"])
        .drop_duplicates(subset=["date", "symbol"], keep="first")
        .drop(columns="__precedence")
        .sort_values(["symbol", "date"])
        .reset_index(drop=True)
    )

    # Wide
    wide_daily = _wide_from_long(long_daily)

    # Write outputs
    manifest = pd.DataFrame(manifest_rows)
    INTERIM_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    manifest_path = INTERIM_DIR / "source_manifest.csv"
    long_path = PROCESSED_DIR / "commodities_long_daily.parquet"
    wide_path = INTERIM_DIR / "commodities_merged_daily.parquet"

    try:
        manifest.to_csv(manifest_path, index=False)
        long_daily.to_parquet(long_path, index=False)
        wide_daily.to_parquet(wide_path)
    except Exception as e:
        logger.warning("Failed writing some outputs: %s", e)

    logger.info("Wrote long daily → %s  (rows=%d)", long_path, len(long_daily))
    logger.info("Wrote wide daily → %s  (rows=%d, cols=%d)", wide_path, len(wide_daily), wide_daily.shape[1])
    logger.info("Wrote manifest   → %s", manifest_path)

    return long_daily, wide_daily


# --------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------

def main():
    build_harmonized_panel()


if __name__ == "__main__":
    main()
