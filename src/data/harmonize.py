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
    # guard: keep only strictly positive prices (avoids later log/return pathologies)
    df = df[df["price_close"] > 0]
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
        # ensure numeric to avoid object-dtype ffill warnings
        out["price_close"] = pd.to_numeric(out["price_close"], errors="coerce")
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
            out = placed.reindex(bcal)
            # avoid FutureWarning: make sure value col is numeric, then ffill
            out["price_close"] = pd.to_numeric(out["price_close"], errors="coerce")
            out = out.ffill()

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
            ys = pd.date_range(start=start, end=end, freq="YS")
            base = g.resample("YS").last()
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

        out = out.reset_index().rename(columns={"index": "date"})
        # enforce unique columns + canonical order to avoid concat crash
        out = out.loc[:, ~pd.Index(out.columns).duplicated()].copy()
        out = out.reindex(columns=["date", "symbol", "price_close", "src", "unit", "frequency"])

        return out

    # Group by symbol, resample each group, guard against duplicates
    parts = []
    for _, g in df.groupby("symbol", sort=False):
        pg = _resample_group(g)
        if pg is None or pg.empty:
            continue
        # double-guard against dup columns
        if not pd.Index(pg.columns).is_unique:
            pg = pg.loc[:, ~pd.Index(pg.columns).duplicated()].copy()
        parts.append(pg)

    if not parts:
        return pd.DataFrame(columns=CANON_COLUMNS)

    out = pd.concat(parts, ignore_index=True)
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
def adapt_worldbank_pink_sheet_folder(folder: Path, symbol_map: Dict[str, str]) -> pd.DataFrame:
    """
    Parse multiple World Bank workbooks (monthly, annual, and historical vintages) found in `folder`.
    Returns long frame with: [date, symbol, price_close, src, unit, frequency]
    - frequency: 'M' for monthly, 'A' for annual/marketing-year.
    - unit: assumes USD (consistent with Pink Sheet prices).
    """
    if not folder.exists():
        logger.info("World Bank folder not found: %s", folder)
        return pd.DataFrame(columns=CANON_COLUMNS)

    logger.info("Scanning World Bank Pink Sheet folder: %s", folder)
    files = sorted(folder.rglob("*.xlsx"))
    if not files:
        logger.info("No .xlsx files in %s", folder)
        return pd.DataFrame(columns=CANON_COLUMNS)

    import re
    month_map = {
        "jan": 1, "january": 1, "feb": 2, "february": 2, "mar": 3, "march": 3,
        "apr": 4, "april": 4, "may": 5, "jun": 6, "june": 6, "jul": 7, "july": 7,
        "aug": 8, "august": 8, "sep": 9, "sept": 9, "september": 9,
        "oct":10, "october":10, "nov":11, "november":11, "dec":12, "december":12,
    }
    re_year = re.compile(r"^\d{4}$")
    re_monthname = re.compile(r"^(jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)$", re.I)
    re_mcode = re.compile(r"^(\d{4})[Mm\-\/]?([01]\d)$")  # 1960M01, 2024-07, 202407

    def _flatten_dedup(df: pd.DataFrame) -> pd.DataFrame:
        # flatten MultiIndex headers and drop Unnamed
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [" | ".join([str(x) for x in tup if str(x) != "nan"]).strip() for tup in df.columns]
        else:
            df.columns = [str(c).strip() for c in df.columns]
        df = df.loc[:, [c for c in df.columns if not c.lower().startswith("unnamed")]]
        # deduplicate names → foo, foo.1, foo.2 ...
        new_cols, seen = [], {}
        for c in df.columns:
            if c in seen:
                seen[c] += 1
                new_cols.append(f"{c}.{seen[c]}")
            else:
                seen[c] = 0
                new_cols.append(c)
        df.columns = new_cols
        return df

    def _best_date_col(df: pd.DataFrame) -> Tuple[Optional[str], Optional[pd.Series]]:
        # prefer explicitly named date-ish columns
        for name in df.columns:
            if name.lower() in ("date", "month", "period"):
                ser = pd.to_datetime(df[name], errors="coerce", utc=False, format="mixed")
                if ser.notna().sum() >= max(3, int(len(df)*0.2)):
                    return name, ser
        # otherwise, try each column; pick first where majority parses as dates
        for name in df.columns:
            ser = pd.to_datetime(df[name], errors="coerce", utc=False, format="mixed")
            if ser.notna().sum() >= max(6, int(len(df)*0.5)):
                return name, ser
        return None, None

    def _parse_marketing_year(val) -> pd.Timestamp | pd.NaT:
        if pd.isna(val):
            return pd.NaT
        s = str(val).strip()
        m = re.match(r"^(\d{4})\s*/\s*(\d{4})$", s)
        if m:
            return pd.Timestamp(int(m.group(1)), 1, 1)  # take first year
        try:
            return pd.Timestamp(int(float(s)), 1, 1)
        except Exception:
            return pd.NaT

    frames: List[pd.DataFrame] = []

    for xlsx in files:
        try_headers = (0, [0, 1], 1)  # try simple first; many WB sheets work with header=0
        book = None
        for hdr in try_headers:
            try:
                book = pd.read_excel(xlsx, sheet_name=None, header=hdr, engine="openpyxl")
                break
            except Exception:
                continue
        if not isinstance(book, dict):
            logger.warning("Skip unreadable WB file: %s", xlsx.name)
            continue

        vintage = pd.Timestamp(xlsx.stat().st_mtime, unit="s")

        for sname, raw in book.items():
            if not isinstance(raw, pd.DataFrame) or raw.empty:
                continue
            df = raw.dropna(how="all")
            if df.empty:
                continue
            df = _flatten_dedup(df)
            # drop fully-empty columns after dedup
            df = df.loc[:, [c for c in df.columns if not df[c].isna().all()]]
            if df.shape[1] < 1:
                continue

            cols = list(df.columns)
            low = [c.lower() for c in cols]
            year_col = next((c for c in cols if c.lower() == "year"), None)

            # -------- Layout 1: tidy Date + series columns (e.g., Aluminum, Coal, etc.) --------
            dname, dser = _best_date_col(df)
            if dname is not None:
                # numeric columns = candidate series
                num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and c != dname]
                if not num_cols:
                    # try to coerce non-numeric to numeric if they look like values
                    for c in df.columns:
                        if c == dname:
                            continue
                        coerced = pd.to_numeric(df[c], errors="coerce")
                        if coerced.notna().sum() >= 6:
                            num_cols.append(c)
                if num_cols:
                    slim = df[[dname] + num_cols].dropna(subset=[dname]).copy()
                    # melt safely (id_vars must be *unique* column names)
                    slim = slim.loc[:, ~slim.columns.duplicated()].copy()
                    long = slim.melt(id_vars=[dname], var_name="series", value_name="price_close")
                    long["date"] = pd.to_datetime(long[dname], errors="coerce", utc=False, format="mixed")
                    long = long.dropna(subset=["date"])
                    base = f"{xlsx.stem} | {sname}"
                    long["raw_symbol"] = base + " | " + long["series"].astype(str)
                    long["symbol"] = long["raw_symbol"].map(lambda s: _canonicalize_symbol(s, symbol_map))
                    long["src"] = "worldbank_pinksheet"
                    long["unit"] = "USD"
                    long["frequency"] = "M"
                    long["__vintage"] = vintage
                    frames.append(long[["date", "symbol", "price_close", "src", "unit", "frequency", "__vintage"]])
                    continue

            # -------- Layout 2: Year + Jan..Dec (monthly wide) --------
            if year_col and any(re_monthname.match(c.split("|")[0].strip().lower()) for c in cols):
                mcols = [c for c in cols if re_monthname.match(c.split("|")[0].strip().lower())]
                keep = [year_col] + mcols
                slim = df[keep].dropna(subset=[year_col]).copy()
                slim = slim.loc[:, ~slim.columns.duplicated()].copy()
                long = slim.melt(id_vars=[year_col], var_name="month_name", value_name="price_close")
                long["month_num"] = long["month_name"].map(lambda x: month_map.get(x.split("|")[0].strip().lower(), np.nan))
                long = long.dropna(subset=["month_num"])
                long["date"] = pd.to_datetime(
                    dict(year=pd.to_numeric(long[year_col], 
                                            errors="coerce"), 
                            month=long["month_num"].astype(int), 
                            day=1,
                            format="mixed"),
                    errors="coerce",
                    format="mixed"
                )
                long = long.dropna(subset=["date"])
                base = f"{xlsx.stem} | {sname}"
                long["raw_symbol"] = base
                long["symbol"] = long["raw_symbol"].map(lambda s: _canonicalize_symbol(s, symbol_map))
                long["src"] = "worldbank_pinksheet"
                long["unit"] = "USD"
                long["frequency"] = "M"
                long["__vintage"] = vintage
                frames.append(long[["date", "symbol", "price_close", "src", "unit", "frequency", "__vintage"]])
                continue

            # -------- Layout 3: 4-digit years across columns (annual wide) --------
            year_header_cols = [c for c in cols if re_year.match(str(c))]
            if year_header_cols:
                # a series-name column if present
                name_candidates = [c for c in cols if c.lower() in ("commodity", "series name", "indicator name", "series", "description", "item", "name")]
                name_cols = name_candidates[:1]
                keep = name_cols + year_header_cols
                slim = df[keep].copy()
                if not name_cols:
                    slim.insert(0, "Series", f"{xlsx.stem} | {sname}")
                    name_cols = ["Series"]
                slim = slim.loc[:, ~slim.columns.duplicated()].copy()
                long = slim.melt(id_vars=name_cols, var_name="year", value_name="price_close")
                long["date"] = pd.to_datetime(long["year"].astype(str) + "-01-01", errors="coerce", format="mixed")
                long = long.dropna(subset=["date"])
                long["raw_symbol"] = long[name_cols[0]].astype(str)
                long["symbol"] = long["raw_symbol"].map(lambda s: _canonicalize_symbol(s, symbol_map))
                long["src"] = "worldbank_pinksheet"
                long["unit"] = "USD"
                long["frequency"] = "A"
                long["__vintage"] = vintage
                frames.append(long[["date", "symbol", "price_close", "src", "unit", "frequency", "__vintage"]])
                continue

            # -------- Layout 4: Year column + multiple numeric series (annual tidy / marketing-year) --------
            if year_col:
                num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and c != year_col]
                if num_cols:
                    slim = df[[year_col] + num_cols].dropna(subset=[year_col]).copy()
                    slim = slim.loc[:, ~slim.columns.duplicated()].copy()
                    long = slim.melt(id_vars=[year_col], var_name="series", value_name="price_close")
                    long["date"] = long[year_col].map(_parse_marketing_year)
                    long = long.dropna(subset=["date"])
                    base = f"{xlsx.stem} | {sname}"
                    long["raw_symbol"] = base + " | " + long["series"].astype(str)
                    long["symbol"] = long["raw_symbol"].map(lambda s: _canonicalize_symbol(s, symbol_map))
                    long["src"] = "worldbank_pinksheet"
                    long["unit"] = "USD"
                    long["frequency"] = "A"
                    long["__vintage"] = vintage
                    frames.append(long[["date", "symbol", "price_close", "src", "unit", "frequency", "__vintage"]])
                    continue

            # -------- Layout 5: YYYYMM-ish across columns --------
            ym_cols = [c for c in cols if re_mcode.match(str(c))]
            if ym_cols:
                name_candidates = [c for c in cols if c.lower() in ("commodity", "series name", "indicator name", "series", "description", "item", "name")]
                name_cols = name_candidates[:1]
                keep = name_cols + ym_cols
                slim = df[keep].copy()
                if not name_cols:
                    slim.insert(0, "Series", f"{xlsx.stem} | {sname}")
                    name_cols = ["Series"]
                slim = slim.loc[:, ~slim.columns.duplicated()].copy()
                long = slim.melt(id_vars=name_cols, var_name="ym", value_name="price_close")

                def _to_date(s):
                    s = str(s).replace("/", "").replace("-", "")
                    m = re.match(r"^(\d{4})(\d{2})$", s)
                    return pd.Timestamp(int(m.group(1)), int(m.group(2)), 1) if m else pd.NaT

                long["date"] = long["ym"].map(_to_date)
                long = long.dropna(subset=["date"])
                long["raw_symbol"] = long[name_cols[0]].astype(str)
                long["symbol"] = long["raw_symbol"].map(lambda s: _canonicalize_symbol(s, symbol_map))
                long["src"] = "worldbank_pinksheet"
                long["unit"] = "USD"
                long["frequency"] = "M"
                long["__vintage"] = vintage
                frames.append(long[["date", "symbol", "price_close", "src", "unit", "frequency", "__vintage"]])
                continue
            # otherwise: skip non-data/contents sheets

    if not frames:
        logger.warning("No usable World Bank tables found across XLSX files.")
        return pd.DataFrame(columns=CANON_COLUMNS)

    wb = pd.concat(frames, ignore_index=True)
    wb["price_close"] = pd.to_numeric(wb["price_close"], errors="coerce")
    wb = wb.dropna(subset=["date", "symbol", "price_close"])

    # Resolve overlaps: prefer monthly over annual; for same freq keep newest vintage
    freq_rank = {"M": 0, "Q": 1, "A": 2, None: 9}
    wb["__rank"] = wb["frequency"].map(freq_rank).fillna(9).astype(int)
    wb["__vintage"] = wb.get("__vintage", pd.Timestamp("1970-01-01"))

    wb = (
        wb.sort_values(["symbol", "date", "__rank", "__vintage"], ascending=[True, True, True, False])
          .drop_duplicates(subset=["symbol", "date"], keep="first")
          .drop(columns=["__rank", "__vintage"])
          .reset_index(drop=True)
    )
    return _finalize_frame(wb, src="worldbank_pinksheet", unit_default="USD")




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
    Parse EIA Monthly Energy Review (MER) tables exported by download.py into long tidy rows.
    Handles the common MER shape: banner rows, header row, *units* row, then data.
    Sheets parsed: 'Monthly Data' (frequency='M') and 'Annual Data' (frequency='A').
    """
    import re
    import numpy as np
    import pandas as pd

    if not folder.exists():
        logger.info("EIA MER folder missing: %s", folder)
        return pd.DataFrame(columns=CANON_COLUMNS)

    files = sorted(list(folder.glob("*.csv")) +
                   list(folder.glob("*.xlsx")) +
                   list(folder.glob("*.xls")))
    if not files:
        logger.info("No MER files in %s", folder)
        return pd.DataFrame(columns=CANON_COLUMNS)

    def _uniquify(cols: list[str]) -> list[str]:
        seen, out = {}, []
        for c in cols:
            k = str(c)
            if k in seen:
                seen[k] += 1
                out.append(f"{k}__{seen[k]}")
            else:
                seen[k] = 0
                out.append(k)
        return out

    def _looks_like_units(texts: list[str]) -> bool:
        if not texts:
            return False
        keys = [
            "btu","barrel","dollar","percent","cubic","short","tons",
            "bcm","gwh","quadrillion","million","billion","per","british"
        ]
        hits = sum(("(" in t or ")" in t or any(k in t.lower() for k in keys)) for t in texts)
        return hits / len(texts) >= 0.5

    def _units_row_to_map(df: pd.DataFrame, row_idx: int, ncols: int) -> dict[int, str]:
        if row_idx is None or row_idx >= len(df):
            return {}
        row = df.iloc[row_idx].astype(str).tolist()[:ncols]
        return {i: (s.strip() if str(s).strip().lower() != "nan" else None) for i, s in enumerate(row)}

    def _parse_monthly(df: pd.DataFrame, file_stem: str) -> Optional[pd.DataFrame]:
        # find header row (line with 'Month' or 'Year and Month')
        header_row = None
        for i in range(min(80, len(df))):
            row = df.iloc[i].astype(str).str.strip().tolist()
            rl = [c.lower() for c in row]
            if any(c == "month" for c in rl) or any("year and month" in c for c in rl):
                header_row = i
                break
        # fallback: first row whose col0 looks like "YYYY Month"
        if header_row is None:
            patt = re.compile(r"^\s*\d{4}\s+[A-Za-z]+")
            for i in range(min(80, len(df))):
                if patt.match(str(df.iloc[i, 0])):
                    header_row = max(i - 3, 0)
                    break
        if header_row is None:
            return None

        cols_raw = df.iloc[header_row].astype(str).tolist()
        cols = _uniquify(cols_raw)
        date_col = next((c for c in cols if "month" in c.lower()), None) or "Year and Month"
        cols[0] = date_col  # ensure the first column is the date label

        # data start: first row after header with "YYYY Month"
        patt = re.compile(r"^\s*\d{4}\s+[A-Za-z]+")
        data_start = None
        for i in range(header_row + 1, len(df)):
            if patt.match(str(df.iloc[i, 0])):
                data_start = i
                break
        if data_start is None:
            return None

        data = df.iloc[data_start:].copy()
        base_cols = cols[: len(data.columns)]
        if len(base_cols) < len(data.columns):
            base_cols += [f"col_{i}" for i in range(len(base_cols), len(data.columns))]
        data.columns = base_cols

        # units
        units_map_idx = _units_row_to_map(df, header_row + 1, len(base_cols))
        # parse date
        ser = data[date_col].astype(str).str.strip()
        d1 = pd.to_datetime(ser, format="%Y %B", errors="coerce")
        if d1.isna().mean() > 0.5:
            d1 = pd.to_datetime(ser, format="%Y %b", errors="coerce")
        if d1.isna().mean() > 0.5:
            d1 = pd.to_datetime(ser, errors="coerce")
        data["date"] = d1
        data = data.dropna(subset=["date"])

        value_cols = [c for c in data.columns if c not in (date_col, "date")]
        parts = []
        unit_for = {}
        for idx, c in enumerate(value_cols):
            s = pd.to_numeric(data[c], errors="coerce")
            if s.notna().sum() == 0:
                continue
            tmp = pd.DataFrame(
                {"date": data["date"], "raw_symbol": c, "price_close": s, "unit": units_map_idx.get(idx)}
            )
            parts.append(tmp)
            unit_for[c] = units_map_idx.get(idx)

        if not parts:
            return None
        out = pd.concat(parts, ignore_index=True)
        out["symbol"] = out["raw_symbol"].map(lambda s: _canonicalize_symbol(str(s), symbol_map))
        out["src"] = "eia_mer"
        out["frequency"] = "M"
        return out[["date", "symbol", "price_close", "src", "unit", "frequency"]]

    def _parse_annual(df: pd.DataFrame, file_stem: str) -> Optional[pd.DataFrame]:
        # find first row where col0 is a 4-digit year
        year_idx = None
        for i in range(min(200, len(df))):
            v = str(df.iloc[i, 0]).strip()
            if re.fullmatch(r"\d{4}", v):
                year_idx = i
                break
        if year_idx is None:
            for i in range(min(200, len(df))):
                val = df.iloc[i, 0]
                if isinstance(val, (int, float)) and not pd.isna(val) and 1900 <= int(val) <= 2100:
                    year_idx = i
                    break
        if year_idx is None:
            return None

        # header row = closest non-unit text row above year_idx
        header_row = None
        for j in range(year_idx - 1, max(-1, year_idx - 8), -1):
            row = df.iloc[j].tolist()
            texts = [str(x).strip() for x in row[:40] if isinstance(x, str) and x.strip()]
            if len(texts) >= 2 and not _looks_like_units(texts):
                header_row = j
                break
        if header_row is None:
            header_row = max(0, year_idx - 1)

        cols_raw = df.iloc[header_row].astype(str).tolist()
        cols = _uniquify(cols_raw)

        data = df.iloc[year_idx:].copy()
        base_cols = cols[: len(data.columns)]
        if len(base_cols) < len(data.columns):
            base_cols += [f"col_{i}" for i in range(len(base_cols), len(data.columns))]
        data.columns = base_cols

        # Year column name can be 'Year', 'End of Year', etc.
        year_candidates = [c for c in data.columns if "year" in str(c).lower()]
        ycol = year_candidates[0] if year_candidates else data.columns[0]

        # units
        units_map_idx = _units_row_to_map(df, header_row + 1, len(base_cols))

        y = pd.to_numeric(data[ycol], errors="coerce").astype("Int64")
        data["date"] = pd.to_datetime(y.astype(str) + "-01-01", errors="coerce")
        data = data.dropna(subset=["date"])

        value_cols = [c for c in data.columns if c not in (ycol, "date")]
        parts = []
        for i, c in enumerate(value_cols):
            s = pd.to_numeric(data[c], errors="coerce")
            if s.notna().sum() == 0:
                continue
            tmp = pd.DataFrame(
                {"date": data["date"], "raw_symbol": c, "price_close": s, "unit": units_map_idx.get(i)}
            )
            parts.append(tmp)

        if not parts:
            return None
        out = pd.concat(parts, ignore_index=True)
        out["symbol"] = out["raw_symbol"].map(lambda s: _canonicalize_symbol(str(s), symbol_map))
        out["src"] = "eia_mer"
        out["frequency"] = "A"
        return out[["date", "symbol", "price_close", "src", "unit", "frequency"]]

    frames: list[pd.DataFrame] = []

    for path in files:
        # MER bundle contains a state workbook that doesn't match the pattern—skip it
        if re.search(r"state[_\- ]?data", path.name, flags=re.I):
            continue

        try:
            if path.suffix.lower() == ".csv":
                df = pd.read_csv(path, low_memory=False)
                # very rare in MER zip; keep the old generic CSV branch if you like
                continue
            else:
                sheets = pd.read_excel(path, sheet_name=None, header=None, engine=("openpyxl" if path.suffix.lower() in (".xlsx", ".xlsm") else "xlrd"))
        except Exception as e:
            logger.warning("Skip MER file %s: %s", path.name, e)
            continue

        monthly = None
        annual = None
        for sname, df in sheets.items():
            s = sname.strip().lower()
            if "monthly" in s:
                monthly = _parse_monthly(df, path.stem)
                if monthly is not None:
                    frames.append(monthly)
            elif "annual" in s:
                annual = _parse_annual(df, path.stem)
                if annual is not None:
                    frames.append(annual)

    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=CANON_COLUMNS)
    if out.empty:
        logger.info("No rows parsed from MER folder %s", folder)
        return out

    # finalize; keep native units (do not force USD)
    out = _finalize_frame(out, src="eia_mer", unit_default=None)

    # Prefer monthly over annual where both exist
    out["__rank"] = out["frequency"].map({"M": 0, "A": 1}).fillna(2).astype(int)
    out = (out.sort_values(["symbol", "date", "__rank"])
              .drop_duplicates(["symbol", "date"], keep="first")
              .drop(columns="__rank")
              .reset_index(drop=True))

    logger.info("Got %d rows from eia_mer (files=%d)", len(out), len(files))
    return out


def adapt_kaggle_etf_stock(folder: Path, symbol_map: Dict[str, str]) -> pd.DataFrame:
    """
    Adapt Kaggle ETFs/Stocks text files into [date, symbol, price_close, src, unit, frequency].

    - Supports *.txt and *.csv
    - Supports both .../Data/ETFs and .../ETFs (and likewise for Stocks)
    - By default, only keeps commodity-oriented ETFs (set below). Change the allowlist
      or set it to None to ingest everything.
    """
    if not folder.exists():
        logger.info("ETF/Stock folder not found: %s", folder)
        return pd.DataFrame(columns=CANON_COLUMNS)

    # ---- Adjust this allowlist to your taste (or set to None to take all ETFs) ----
    commodity_etf_allowlist = {
        "GLD","SLV","PPLT","PALL",                 # precious metals
        "DBC","GSG","DJP",                         # broad commodity indices
        "DBB","JJC","CPER",                        # base metals (broad/copper)
        "USO","BNO","UNG","UGA",                   # energy (oil/NG/gasoline)
        "DBA","SOYB","WEAT","CORN","CANE","JO","WOOD"  # ags & timber
    }
    keep_only_etfs = True  # flip to False if you also want Stocks

    logger.info("Parsing ETF/Stock price files from %s", folder)

    paths = (
        list(folder.rglob("**/*.txt")) +
        list(folder.rglob("**/*.csv"))
    )
    if not paths:
        logger.info("No ETF/Stock files found under %s", folder)
        return pd.DataFrame(columns=CANON_COLUMNS)

    parts: List[pd.DataFrame] = []

    for p in sorted(paths):
        # optionally ignore stocks
        if keep_only_etfs and not any("etf" in part.lower() for part in p.parts):
            continue

        # read
        try:
            if p.suffix.lower() == ".csv":
                df = pd.read_csv(p, low_memory=False)
            else:
                # Kaggle .txt files are plain CSV
                df = pd.read_csv(p, low_memory=False)
        except Exception as e:
            logger.debug("Skip unreadable file %s: %s", p, e)
            continue
        if df.empty:
            continue

        # date column
        dcol = next((c for c in df.columns if str(c).lower() in {"date","datetime","time"}), None)
        if dcol is None:
            continue

        # price column: prefer Adj Close if present
        price_candidates = ["Adj Close", "adj close", "adj_close", "Close", "close", "PX_LAST", "price"]
        pcol = next((c for c in price_candidates if c in df.columns), None)
        if pcol is None:
            # last numeric as fallback
            num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
            pcol = num_cols[-1] if num_cols else None
        if pcol is None:
            continue

        # symbol from filename (e.g., 'gld.us.txt' -> 'GLD')
        stem = p.stem  # e.g., 'gld.us'
        base = stem.split(".")[0] if "." in stem else stem
        ticker = base.upper().strip()

        # optional filter to commodity ETFs
        if commodity_etf_allowlist is not None and ticker not in commodity_etf_allowlist:
            # if you're keeping all ETFs, set commodity_etf_allowlist = None above
            continue

        tmp = pd.DataFrame({
            "date": pd.to_datetime(df[dcol], errors="coerce", utc=False).dt.tz_localize(None),
            "symbol": _canonicalize_symbol(ticker, symbol_map),
            "price_close": pd.to_numeric(df[pcol], errors="coerce"),
            "src": "kaggle_etf_stock",
            "unit": "USD",
            "frequency": "D",
        }).dropna(subset=["date","price_close"])

        parts.append(tmp)

    if not parts:
        logger.info("No rows from kaggle_etf_stock")
        return pd.DataFrame(columns=CANON_COLUMNS)

    out = pd.concat(parts, ignore_index=True)

    # De-dupe in case the same ticker exists in multiple folders
    out = (out.sort_values(["symbol","date"])
              .drop_duplicates(subset=["symbol","date"], keep="last"))

    out = _finalize_frame(out, src="kaggle_etf_stock", unit_default="USD")
    logger.info("Got %d rows from kaggle_etf_stock (files scanned=%d)", len(out), len(paths))
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
