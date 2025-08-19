# src/data/prepare_kaggle_alignment.py
from __future__ import annotations
import os
from pathlib import Path
import numpy as np
import pandas as pd

REPO_ROOT = os.environ.get("REPO_ROOT", "/home/ubuntu/repos/kaggle-mitsui")
RAW_DIR    = Path(f"{REPO_ROOT}/data/kaggle")  
RAW_DIR_LABELS = Path(f"{REPO_ROOT}/data/lagged_test_labels")
OUT_DIR    = Path(f"{REPO_ROOT}/data/processed")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---- explicit file paths (edit if yours differ) ----
TRAIN_CSV   = RAW_DIR / "train.csv"
LABELS_CSV  = RAW_DIR / "train_labels.csv"
PAIRS_CSV   = RAW_DIR / "target_pairs.csv"
TEST_CSV    = RAW_DIR / "test.csv"

# optional (to augment train with test labels)
TEST_LAG_CSV = {
    1: RAW_DIR_LABELS / "test_labels_lag_1.csv",
    2: RAW_DIR_LABELS / "test_labels_lag_2.csv",
    3: RAW_DIR_LABELS / "test_labels_lag_3.csv",
    4: RAW_DIR_LABELS / "test_labels_lag_4.csv",
}

def main() -> None:
    # -------- load core frames --------
    train = pd.read_csv(TRAIN_CSV)
    labels = pd.read_csv(LABELS_CSV)
    pairs = pd.read_csv(PAIRS_CSV)

    # sanity
    assert "date_id" in train.columns, "train.csv must have date_id"
    assert "date_id" in labels.columns, "train_labels.csv must have date_id"
    assert set(["target","lag"]).issubset(pairs.columns), "target_pairs.csv needs ['target','lag']"

    # feature columns = everything but date_id / is_scored / target_*
    feature_cols = [c for c in train.columns
                    if c != "date_id" and c != "is_scored" and not c.startswith("target_")]
    target_cols  = [c for c in labels.columns if c.startswith("target_")]
    print(f"[prep] features={len(feature_cols)} targets={len(target_cols)}")

    # ---- align labels so each row at label_date_id = t has y_j = label at (t + lag_j) ----
    # build mapping target -> lag
    lag_for = dict(zip(pairs["target"], pairs["lag"]))
    # make a copy we can shift
    lab = labels.set_index("date_id").sort_index()

    aligned_parts = []
    for lag_val in sorted(set(lag_for.values())):
        cols = [t for t in target_cols if lag_for.get(t, None) == lag_val]
        if not cols:
            continue
        # shift backward so row at t holds label from t+lag
        block = lab[cols].shift(-int(lag_val))
        block.index.name = "label_date_id"
        aligned_parts.append(block)

    if not aligned_parts:
        raise RuntimeError("No targets aligned. Check target_pairs.csv contents.")

    aligned = pd.concat(aligned_parts, axis=1).sort_index()
    aligned = aligned.loc[~aligned.index.isna()].copy()
    aligned.index = aligned.index.astype(int)
    aligned = aligned.reset_index()

    # ---- augment with test-lag labels (optional but nice) ----
    # we’ll add rows keyed by label_date_id from test, using the provided lag files.
    # to avoid polluting targets with lags that don't belong, we only keep targets whose lag==that file's lag.
    for lag_val, path in TEST_LAG_CSV.items():
        if not path.exists():
            continue
        tlag = pd.read_csv(path)
        assert {"date_id","label_date_id"}.issubset(tlag.columns)
        cols = [c for c in tlag.columns if c.startswith("target_") and lag_for.get(c, None) == lag_val]
        if not cols:
            continue
        # we want a frame keyed by label_date_id, with only those columns
        add = tlag[["label_date_id"] + cols].copy()
        # combine: prefer existing train labels, fill where missing
        aligned = aligned.set_index("label_date_id").combine_first(add.set_index("label_date_id")).reset_index()

    # ---- features for both train and test (keyed by label_date_id) ----
    # training features are keyed by date_id (these ARE the label_date_id after alignment).
    X_tr = train[["date_id"] + feature_cols].rename(columns={"date_id": "label_date_id"})
    # include test features too (they are keyed by date_id that *is* label_date_id in lag files)
    if TEST_CSV.exists():
        test = pd.read_csv(TEST_CSV)
        X_te = test[["date_id"] + [c for c in test.columns if c in feature_cols]].rename(columns={"date_id": "label_date_id"})
        X_all = pd.concat([X_tr, X_te], ignore_index=True).drop_duplicates(subset=["label_date_id"], keep="last")
    else:
        X_all = X_tr

    # ---- join features+labels on label_date_id ----
    df = pd.merge(X_all, aligned, on="label_date_id", how="inner").sort_values("label_date_id").reset_index(drop=True)

    # tidy types and NaNs (features only)
    for c in feature_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    # replace infs and keep NaNs in targets (loss is masked later)
    feat_mat = df[feature_cols].replace([np.inf,-np.inf], np.nan)
    # simple robust center/scale saved for the trainer to recompute on train split
    # we keep raw here; trainer will normalize from train-only stats

    # Save a single parquet for the FT trainer
    out_path = OUT_DIR / "kaggle_aligned.parquet"
    df.to_parquet(out_path, index=False)
    print(f"[prep] wrote → {out_path}  rows={len(df)}  feat={len(feature_cols)}  tgt={len(target_cols)}")

if __name__ == "__main__":
    main()
