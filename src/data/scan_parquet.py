# src/tools/scan_parquet.py
from __future__ import annotations
import argparse
import math
import os
import re
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional

import numpy as np

# hard dependency for streaming parquet (file or directory)
import pyarrow as pa
import pyarrow.dataset as ds


# ---------- numeric streaming aggregator (Welford + reservoir sample) ----------
class NumAgg:
    def __init__(self, sample_size: int):
        self.n = 0                   # count of finite values
        self.mean = 0.0
        self.M2 = 0.0                # sum of squares of diffs for variance
        self.min = math.inf
        self.max = -math.inf

        self.n_total = 0             # total elements seen (finite + nan/inf)
        self.n_nan = 0
        self.n_posinf = 0
        self.n_neginf = 0
        self.n_zero = 0

        self.sample_size = int(sample_size)
        self.sample: List[float] = []

    def ingest(self, arr: np.ndarray):
        # arr is float64; may contain nan/inf
        self.n_total += arr.size

        # count special cases
        nan_mask = np.isnan(arr)
        posinf_mask = np.isposinf(arr)
        neginf_mask = np.isneginf(arr)
        self.n_nan += int(nan_mask.sum())
        self.n_posinf += int(posinf_mask.sum())
        self.n_neginf += int(neginf_mask.sum())

        finite_mask = np.isfinite(arr)
        if not finite_mask.any():
            return

        x = arr[finite_mask]

        # zeros (finite zeros)
        self.n_zero += int((x == 0.0).sum())

        # update min/max
        cur_min = float(np.min(x))
        cur_max = float(np.max(x))
        if cur_min < self.min: self.min = cur_min
        if cur_max > self.max: self.max = cur_max

        # Welford updates
        for v in x:
            self.n += 1
            delta = v - self.mean
            self.mean += delta / self.n
            delta2 = v - self.mean
            self.M2 += delta * delta2

            # reservoir sample for quantiles
            if self.sample_size > 0:
                if len(self.sample) < self.sample_size:
                    self.sample.append(float(v))
                else:
                    j = np.random.randint(0, self.n)
                    if j < self.sample_size:
                        self.sample[j] = float(v)

    def finalize(self) -> Dict[str, float]:
        var = (self.M2 / self.n) if self.n > 0 else float("nan")
        std = math.sqrt(var) if var >= 0 and math.isfinite(var) else float("nan")
        # quantiles from sample (if any)
        p1 = p50 = p99 = float("nan")
        if self.sample:
            s = np.array(self.sample, dtype=np.float64)
            p1 = float(np.nanpercentile(s, 1))
            p50 = float(np.nanpercentile(s, 50))
            p99 = float(np.nanpercentile(s, 99))
        miss_ratio = float(self.n_nan / max(1, self.n_total))
        zero_ratio = float(self.n_zero / max(1, (self.n))) if self.n > 0 else float("nan")
        return {
            "count_total": float(self.n_total),
            "count_finite": float(self.n),
            "count_nan": float(self.n_nan),
            "count_posinf": float(self.n_posinf),
            "count_neginf": float(self.n_neginf),
            "missing_ratio": miss_ratio,
            "min": float(self.min) if self.n > 0 else float("nan"),
            "max": float(self.max) if self.n > 0 else float("nan"),
            "mean": float(self.mean) if self.n > 0 else float("nan"),
            "std": float(std),
            "p01": p1,
            "p50": p50,
            "p99": p99,
            "zero_ratio_finite": zero_ratio,
        }


@dataclass
class ColReport:
    column: str
    dtype: str
    count_total: float
    count_finite: float
    count_nan: float
    count_posinf: float
    count_neginf: float
    missing_ratio: float
    min: float
    max: float
    mean: float
    std: float
    p01: float
    p50: float
    p99: float
    zero_ratio_finite: float
    flags: str


# ---------- helpers ----------
RET_PATTERN = re.compile(r"(?:^|_)ret|logret", re.IGNORECASE)
ZS_PATTERN  = re.compile(r"^z[_\d]*", re.IGNORECASE)

def flag_column(name: str, stats: Dict[str, float]) -> List[str]:
    f: List[str] = []
    max_abs = max(abs(stats["min"]), abs(stats["max"])) if np.isfinite(stats["min"]) and np.isfinite(stats["max"]) else float("inf")

    # generic issues
    if stats["count_posinf"] > 0 or stats["count_neginf"] > 0:
        f.append("has_inf")
    if stats["missing_ratio"] > 0.5:
        f.append("missing>50%")
    if stats["count_finite"] > 0 and stats["std"] == 0.0:
        f.append("constant")
    if np.isfinite(stats["std"]) and stats["std"] > 1e3:
        f.append("std>1e3")
    if np.isfinite(max_abs) and max_abs > 1e6:
        f.append("|x|>1e6")
    if np.isfinite(stats["zero_ratio_finite"]) and stats["zero_ratio_finite"] > 0.95:
        f.append("mostly_zero")

    # semantics by name
    if RET_PATTERN.search(name):
        # returns/log-returns should rarely exceed |2| if expressed in decimal
        if np.isfinite(stats["p99"]) and abs(stats["p99"]) > 2.0:
            f.append("ret_p99>|2|")
        if np.isfinite(stats["p01"]) and abs(stats["p01"]) > 2.0:
            f.append("ret_p01>|2|")
    if ZS_PATTERN.search(name):
        # z-scores should be ~ N(0,1)
        if np.isfinite(stats["mean"]) and abs(stats["mean"]) > 0.5:
            f.append("z_mean_shift")
        if np.isfinite(stats["std"]) and not (0.4 <= stats["std"] <= 2.5):
            f.append("z_bad_std")

    # common raw culprits
    if name.lower() in {"price_close", "close", "px_last"}:
        if np.isfinite(max_abs) and max_abs > 1e4:
            f.append("raw_level_scale")

    return f


def arrow_numeric_to_numpy(col: pa.Array) -> Optional[np.ndarray]:
    """
    Convert a numeric Arrow array to float64 numpy with NaNs for nulls.
    Return None for non-numeric.
    """
    if pa.types.is_null(col.type):
        return None
    if pa.types.is_floating(col.type) or pa.types.is_integer(col.type) or pa.types.is_decimal(col.type):
        # pandas conversion handles nulls -> NaN for floats; for ints/decimals we cast to float64
        ser = col.to_pandas(types_mapper=None)  # pandas Series
        return ser.astype("float64").to_numpy()
    return None


# ---------- main scan ----------
def scan_parquet(
    path: str,
    include: Optional[List[str]],
    exclude: Optional[List[str]],
    exclude_prefix: Optional[List[str]],
    batch_size: int,
    sample_size: int,
    use_threads: bool,
    limit_rows: int,
    progress_every: int,
) -> List[ColReport]:

    dataset = ds.dataset(path, format="parquet")
    schema = dataset.schema

    all_cols = [f.name for f in schema]
    # column selection
    cols = list(all_cols)
    if include:
        keep = set(include)
        cols = [c for c in cols if c in keep]
    if exclude:
        bad = set(exclude)
        cols = [c for c in cols if c not in bad]
    if exclude_prefix:
        def ok(c: str) -> bool:
            return not any(c.startswith(p) for p in exclude_prefix)
        cols = [c for c in cols if ok(c)]

    # Build scanner
    scanner = ds.Scanner.from_dataset(
        dataset,
        columns=cols,
        batch_size=batch_size,
        use_threads=use_threads,
    ) 
    reader = scanner.to_reader()

    # per-column aggregators
    aggs: Dict[str, NumAgg] = {c: NumAgg(sample_size=sample_size) for c in cols}
    dtypes: Dict[str, str] = {c: str(schema.field(c).type) for c in cols}

    rows_seen = 0

    for batch in reader:
        nrows = batch.num_rows
        rows_seen += nrows

        # ingest columns
        for i, name in enumerate(batch.schema.names):
            col = batch.column(i)
            arr = arrow_numeric_to_numpy(col)
            if arr is not None:
                aggs[name].ingest(arr)

        # progress
        if progress_every > 0 and (rows_seen % progress_every) < nrows:
            print(f"[scan] rows_seen={rows_seen:,}")

        # early stop
        if limit_rows > 0 and rows_seen >= limit_rows:
            print(f"[scan] hit limit_rows={limit_rows:,}, stopping early.")
            break

    # finalize
    reports: List[ColReport] = []
    for name in cols:
        stats = aggs[name].finalize()
        flags = flag_column(name, stats)
        rep = ColReport(
            column=name,
            dtype=dtypes.get(name, ""),
            count_total=stats["count_total"],
            count_finite=stats["count_finite"],
            count_nan=stats["count_nan"],
            count_posinf=stats["count_posinf"],
            count_neginf=stats["count_neginf"],
            missing_ratio=stats["missing_ratio"],
            min=stats["min"],
            max=stats["max"],
            mean=stats["mean"],
            std=stats["std"],
            p01=stats["p01"],
            p50=stats["p50"],
            p99=stats["p99"],
            zero_ratio_finite=stats["zero_ratio_finite"],
            flags=";".join(flags),
        )
        reports.append(rep)

    return reports


def main():
    ap = argparse.ArgumentParser(description="Scan Parquet features and flag suspicious columns.")
    ap.add_argument("--path", required=True, help="Parquet file or directory (dataset).")
    ap.add_argument("--out", required=True, help="Where to write CSV report.")
    ap.add_argument("--batch-size", type=int, required=True, help="Arrow batch size for streaming scan.")
    ap.add_argument("--sample-size", type=int, required=True, help="Reservoir sample per column for quantiles.")
    ap.add_argument("--include", nargs="*", default=None, help="Optional explicit list of columns to include.")
    ap.add_argument("--exclude", nargs="*", default=None, help="Column names to exclude (e.g. date symbol src unit).")
    ap.add_argument("--exclude-prefix", nargs="*", default=None, help="Prefixes to exclude (e.g. target_).")
    ap.add_argument("--use-threads", type=lambda s: s.lower()=="true", required=True, help="Explicitly enable/disable pyarrow threaded decoding (True/False).")
    ap.add_argument("--limit-rows", type=int, required=True, help="Stop after reading this many rows (use for quick triage).")
    ap.add_argument("--progress-every", type=int, required=True, help="Print a progress message every N rows.")
    args = ap.parse_args()

    reports = scan_parquet(
        path=args.path,
        include=args.include,
        exclude=args.exclude,
        exclude_prefix=args.exclude_prefix,
        batch_size=args.batch_size,
        sample_size=args.sample_size,
        use_threads=args.use_threads,
        limit_rows=args.limit_rows,
        progress_every=args.progress_every,
    )

    # write CSV
    import csv
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(asdict(reports[0]).keys()) if reports else [])
        if reports:
            w.writeheader()
            for r in reports:
                w.writerow(asdict(r))

    # print a quick summary of flagged columns
    flagged = [r for r in reports if r.flags]
    flagged.sort(key=lambda r: (("raw_level_scale" in r.flags) or ("std>1e3" in r.flags), -abs(r.std if np.isfinite(r.std) else 0.0)), reverse=True)
    print(f"Scanned {len(reports)} columns. Flagged {len(flagged)}:")
    for r in flagged[:25]:
        print(f"  - {r.column:30s} | flags={r.flags:40s} | std={r.std:.3g} | max={r.max:.3g} | miss={r.missing_ratio:.2%}")


if __name__ == "__main__":
    main()
