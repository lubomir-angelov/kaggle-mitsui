#!/usr/bin/env python
"""
Downloader script for the *commodity‑tsmixer* project.

This utility fetches **raw, untouched** files for all external data sources used
in the pipeline and stores them below `data/raw/` in a reproducible folder
layout.  Run it ad‑hoc (`python downloader.py`) or schedule it in CI.

Data sources covered
-------------------
1. Kaggle Commodity Prices Dataset               (daily, 20+ commodities)
2. World Bank "Pink Sheet" (monthly spot prices)
3. Prices of Agricultural Commodities (Kaggle)
4. Commodity Price Prediction repo   (GitHub)
5. Forecasting Energy Prices         (US‑EIA tables)
6. MvTS benchmark (multi‑source; electricity / FX)
7. Cross‑asset ETFs & Stocks         (Kaggle)
8. FinMultiTime (HuggingFace dataset)

The script is **idempotent**: if a target path already exists it skips the
transfer unless `--force` is passed.

Requirements
~~~~~~~~~~~~
- `requests`  (pip install requests)
- `kaggle`    (pip install kaggle) ‑ make sure you have `~/.kaggle/kaggle.json`
- `git` (CLI) for GitHub / HuggingFace clones
- `unzip` or Python's `zipfile` (used automatically)

Example
~~~~~~~
```bash
# download everything
python downloader.py

# download only World Bank + EIA and overwrite
python downloader.py --sources worldbank eia --force
```
"""
from __future__ import annotations

import argparse
import logging
import os
import pathlib
import shutil
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path
from typing import List

from bs4 import BeautifulSoup
import requests

try:
    from kaggle import api as kaggle_api  # type: ignore
except (ImportError, ModuleNotFoundError):
    kaggle_api = None

# add your repo root
REPO_ROOT = "/home/ubuntu/repos/kaggle-mitsui"
RAW_DIR = Path(f"{REPO_ROOT}/data/raw")
RAW_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("downloader")

############################################################
# Utility helpers                                          #
############################################################

def skip_if_exists(path: Path, force: bool):
    if path.exists() and not force:
        logger.info("%s already exists → skip (use --force to overwrite)", path)
        return True
    if path.exists() and force:
        if path.is_file():
            path.unlink()
        else:
            shutil.rmtree(path)
    return False


def stream_download(url: str, out_path: Path, chunk: int = 2 ** 16):
    """HTTP streaming download with progress to stdout."""
    r = requests.get(url, stream=True, timeout=60)
    r.raise_for_status()
    total = int(r.headers.get("content-length", 0))
    done = 0
    with open(out_path, "wb") as f:
        for buf in r.iter_content(chunk):
            f.write(buf)
            done += len(buf)
            if total:
                pct = done / total * 100
                sys.stdout.write(f"\r… {pct:5.1f}% ({done // 1024 ** 2} MB)")
                sys.stdout.flush()
    sys.stdout.write("\n")


def run(cmd: List[str], cwd: Path | None = None):
    logger.info("$ %s", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=cwd)


def unzip(src: Path, dest: Path):
    logger.info("Unzipping %s → %s", src.name, dest)
    with zipfile.ZipFile(src) as zf:
        zf.extractall(dest)


def untar(src: Path, dest: Path):
    logger.info("Extracting %s → %s", src.name, dest)
    mode = "r:gz" if src.suffixes[-1] == ".gz" else "r:*"
    with tarfile.open(src, mode) as tf:
        tf.extractall(dest)

############################################################
# Individual downloaders                                   #
############################################################

def kaggle_download(dataset: str, target: Path, force: bool):
    if kaggle_api is None:
        raise RuntimeError("'kaggle' package not available ‑ pip install kaggle")
    if skip_if_exists(target, force):
        return
    target.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading Kaggle dataset %s", dataset)
    kaggle_api.dataset_download_files(dataset, path=target, unzip=True, quiet=False)


def download_commodity_prices_kaggle(force: bool):
    kaggle_download("debashish311601/commodity-prices", RAW_DIR / "kaggle_commodities_prices", force)


def download_agri_prices_kaggle(force: bool):
    kaggle_download("syedjaferk/agriculture-commodity-data-2019", RAW_DIR / "kaggle_agri_prices_2019", force)


def download_etf_stock_dataset(force: bool):
    kaggle_download("borismarjanovic/price-volume-data-for-all-us-stocks-etfs", RAW_DIR / "kaggle_etf_stock", force)



def download_world_bank_pink_sheet(force: bool):
    """

    monthly https://thedocs.worldbank.org/en/doc/18675f1d1639c7a34d463f59263ba0a2-0050012025/related/CMO-Historical-Data-Monthly.xlsx
    annual https://thedocs.worldbank.org/en/doc/18675f1d1639c7a34d463f59263ba0a2-0050012025/related/CMO-Historical-Data-Annual.xlsx

    Historical data is available at e.g.:
    https://bit.ly/CMO-October-2024-Data
    """

    logger.info("The World Bank data needs to be downaloaded manually! See here https://www.worldbank.org/en/research/commodity-markets#1")

        


def download_commodity_prediction_repo(force: bool):
    repo_url = "https://github.com/amk1997/commodity-price-prediction.git"
    folder = RAW_DIR / "commodity_price_prediction_repo"
    if skip_if_exists(folder, force):
        return
    logger.info("Cloning %s", repo_url)
    run(["git", "clone", "--depth", "1", repo_url, str(folder)])


def download_energy_eia(force: bool):
    """
    # Note for the adapter:
    # - Prefer reading .xlsx with engine='openpyxl'
    # - If you see .xls (rare nowadays), use engine='xlrd' and ensure xlrd==1.2.0
    # - You can also skip Excel entirely and fetch per-table CSVs from the MER page.
    
    Download the current EIA Monthly Energy Review (MER) tables as a ZIP and extract them.

    Saves:
      data/raw/eia_energy/mer_all_tables.zip
      data/raw/eia_energy/mer_zip/  (unzipped tables: .xlsx / .csv)
    """
    import re
    from bs4 import BeautifulSoup

    base = "https://www.eia.gov/totalenergy/data/monthly/"
    dest_dir = RAW_DIR / "eia_energy"
    zip_path = dest_dir / "mer_all_tables.zip"
    extract_dir = dest_dir / "mer_zip"

    # idempotency
    if zip_path.exists() and extract_dir.exists() and not force:
        logger.info("EIA MER already present → skip (use --force to overwrite)")
        return

    dest_dir.mkdir(parents=True, exist_ok=True)
    if extract_dir.exists() and force:
        shutil.rmtree(extract_dir)

    logger.info("Fetching EIA Monthly Energy Review landing page…")
    resp = requests.get(base, timeout=30)
    resp.raise_for_status()

    soup = BeautifulSoup(resp.text, "html.parser")

    # Look for the “Download all tables ZIP” anchor; href usually ends with MER_Excel_Zip.zip
    zip_href = None
    for a in soup.find_all("a", href=True):
        text = (a.get_text() or "").strip().lower()
        href = a["href"]
        if "download all tables" in text and href.lower().endswith(".zip"):
            zip_href = href
            break
    if not zip_href:
        # fallback: any .zip with MER in path
        for a in soup.find_all("a", href=True):
            href = a["href"]
            if href.lower().endswith(".zip") and ("mer" in href.lower() or "zip_excel_month_end" in href.lower()):
                zip_href = href
                break

    if not zip_href:
        raise RuntimeError("Could not locate MER ZIP on the EIA page; the page layout may have changed.")

    if not zip_href.startswith("http"):
        zip_href = requests.compat.urljoin(base, zip_href)

    logger.info("Downloading MER ZIP: %s", zip_href)
    with requests.get(zip_href, stream=True, timeout=60) as r:
        r.raise_for_status()
        ctype = r.headers.get("Content-Type", "").lower()
        # Basic safety: fail if we’re not getting a real zip
        if "zip" not in ctype and not zip_href.lower().endswith(".zip"):
            text_sample = r.content[:500].decode("utf-8", errors="ignore")
            raise RuntimeError(
                f"Expected a ZIP but got Content-Type={ctype}. "
                f"First bytes:\n{text_sample[:300]}…"
            )
        with open(zip_path, "wb") as f:
            for chunk in r.iter_content(1 << 15):
                if chunk:
                    f.write(chunk)

    logger.info("Extracting MER tables → %s", extract_dir)
    extract_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_dir)

    logger.info("EIA MER downloaded and extracted. Examples inside: .xlsx and/or .csv table files.")

    


############################################################
# CLI                                                      #
############################################################

ALL_SOURCES = {
    "kaggle_commodities": download_commodity_prices_kaggle,
    "kaggle_agri": download_agri_prices_kaggle,
    "kaggle_etf": download_etf_stock_dataset,
    "worldbank": download_world_bank_pink_sheet,
    "repo_prediction": download_commodity_prediction_repo,
    "eia": download_energy_eia,
}


def parse_args():
    p = argparse.ArgumentParser(description="Fetch raw datasets for commodity‑tsmixer")
    p.add_argument(
        "--sources",
        nargs="*",
        choices=ALL_SOURCES.keys(),
        default=list(ALL_SOURCES.keys()),
        help="Which sources to download (default: all)",
    )
    p.add_argument("--force", action="store_true", help="Overwrite existing files")
    return p.parse_args()


def main():
    args = parse_args()
    for key in args.sources:
        try:
            ALL_SOURCES[key](args.force)
        except Exception as e:
            logger.error("%s: %s", key, e, exc_info=True)
            continue
    logger.info("All requests completed")


if __name__ == "__main__":
    download_world_bank_pink_sheet(force=True)  # Ensure the Pink Sheet is downloaded first
    main()
