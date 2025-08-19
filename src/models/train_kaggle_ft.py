# src/models/train_kaggle_ft.py
from __future__ import annotations
import os
import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Dict, Any
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from src.models.timexer import build_timexer
from src.models.adapters.lora import apply_lora_to_attn
from src.metrics.sharpe_spearman import sharpe_of_daily_spearman

REPO_ROOT = os.environ.get("REPO_ROOT", "/home/ubuntu/repos/kaggle-mitsui")
DATA_PROCESSED = Path(f"{REPO_ROOT}/data/processed")
DATA_MODELS = Path(f"{REPO_ROOT}/data/models")
DATA_MODELS.mkdir(parents=True, exist_ok=True)
ALIGNED_PATH = DATA_PROCESSED / "kaggle_aligned.parquet"


# ---------- helpers ----------
def report_feature_overlap(
    ckpt: Dict[str, Any], current_features: List[str], max_examples: int
) -> Dict[str, Any]:
    """
    Print a precise overlap report between checkpoint features and current features.
    Returns a dict with counts and lists so you can branch program logic.
    """

    if "feature_cols" not in ckpt or not isinstance(ckpt["feature_cols"], list):
        print("[overlap] checkpoint has no 'feature_cols'; Option B is not possible.")
        return {
            "has_old_list": False,
            "n_old": 0,
            "n_new": len(current_features),
            "n_overlap": 0,
            "overlap_ratio_new": 0.0,
            "overlap_ratio_old": 0.0,
            "overlap": [],
            "only_in_new": current_features,
            "only_in_old": [],
        }
    
    old_features: List[str] = list(ckpt["feature_cols"])
    new_features: List[str] = list(current_features)

    old_set = set(old_features)
    new_set = set(new_features)

    overlap = sorted(old_set & new_set)

    only_in_new = sorted(new_set - old_set)
    only_in_old = sorted(old_set - new_set)

    n_old = len(old_features)
    n_new = len(new_features)
    n_overlap = len(overlap)

    overlap_ratio_new = n_overlap / n_new if n_new > 0 else 0.0
    overlap_ratio_old = n_overlap / n_old if n_old > 0 else 0.0

    print("\n[overlap] feature list comparison")
    print(f"  old_features (ckpt): {n_old}")
    print(f"  new_features (data): {n_new}")
    print(f"  overlap:             {n_overlap}")
    print(f"  coverage of new:     {overlap_ratio_new:.1%}  (overlap / new)")
    print(f"  coverage of old:     {overlap_ratio_old:.1%}  (overlap / old)")

    if n_overlap:
        print("  examples in overlap:", ", ".join(overlap[:max_examples]))

    if only_in_new:
        print(
            "  examples only in NEW:",
            ", ".join(only_in_new[:max_examples]),
            ("..." if len(only_in_new) > max_examples else ""),
        )

    if only_in_old:
        print(
            "  examples only in OLD:",
            ", ".join(only_in_old[:max_examples]),
            ("..." if len(only_in_old) > max_examples else ""),
        )

    # quick recommendation
    if overlap_ratio_new >= 0.30:
        print(
            "  [overlap] Recommendation: Option B (transplant matching patcher weights)."
        )
    else:
        print("  [overlap] Recommendation: Option A (fresh patcher init).")

    return {
        "has_old_list": True,
        "n_old": n_old,
        "n_new": n_new,
        "n_overlap": n_overlap,
        "overlap_ratio_new": overlap_ratio_new,
        "overlap_ratio_old": overlap_ratio_old,
        "overlap": overlap,
        "only_in_new": only_in_new,
        "only_in_old": only_in_old,
    }


def load_backbone_except_patcher_and_head(model, ckpt_state: dict) -> None:
    # remove incompatible layers
    drop_keys = [
        k
        for k in ckpt_state.keys()
        if k.startswith("patcher.proj.") or k.startswith("head.")
    ]

    for k in drop_keys:
        ckpt_state.pop(k, None)
    # load remaining (encoder/FFN/attn)
    model.load_state_dict(ckpt_state, strict=False)


def pick_feature_cols(df: pd.DataFrame) -> List[str]:
    return [
        c for c in df.columns if (c != "label_date_id" and not c.startswith("target_"))
    ]


def pick_target_cols(df: pd.DataFrame) -> List[str]:
    return [c for c in df.columns if c.startswith("target_")]


def time_split(dates: np.ndarray, val_frac: float) -> Tuple[np.ndarray, np.ndarray]:
    u = np.unique(dates)
    n_val = max(10, int(round(len(u) * val_frac)))
    val_days = u[-n_val:]
    train_days = u[: len(u) - n_val]

    return train_days, val_days


def robust_mu_sigma(X_tr: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    mu = X_tr.median().to_numpy(np.float32)
    iqr = (X_tr.quantile(0.75) - X_tr.quantile(0.25)).to_numpy(np.float32)
    std = X_tr.std(ddof=0).to_numpy(np.float32)
    sig = np.where(iqr > 1e-6, iqr, std)
    sig = np.where(sig > 1e-6, sig, 1.0).astype(np.float32)

    return mu, sig


class GlobalSeqDataset(Dataset):
    """
    Sliding windows over a single global time series (no symbol grouping).
    Each sample: x: (T,C), y: (P,), date_id: int (window end).
    """

    def __init__(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        target_cols: List[str],
        window: int,
        stride: int,
        days_keep: np.ndarray,
        mu: np.ndarray,
        sigma: np.ndarray,
        clip: float,
    ):
        self.feature_cols = feature_cols
        self.target_cols = target_cols
        self.window = int(window)
        self.stride = int(stride)
        self.clip = float(clip)

        # keep rows by day split
        df = (
            df[df["label_date_id"].isin(days_keep)]
            .sort_values("label_date_id")
            .reset_index(drop=True)
        )

        # matrices
        X = (
            df[feature_cols]
            .astype(np.float32)
            .replace([np.inf, -np.inf], np.nan)
            .values
        )

        Y = df[target_cols].astype(np.float32).values
        D = df["label_date_id"].to_numpy(np.int32)
        self.mu = torch.tensor(mu, dtype=torch.float32)
        self.sigma = torch.tensor(sigma, dtype=torch.float32)

        # build end indices with stride and with at least one non-NaN target
        self.ends: List[int] = []
        start = window - 1

        for i in range(start, len(df)):
            if ((i - start) % self.stride) != 0:
                continue
            if not np.isfinite(Y[i]).any():
                continue
            self.ends.append(i)
        self.X = X
        self.Y = Y
        self.D = D

    def __len__(self) -> int:
        return len(self.ends)

    def __getitem__(self, idx: int):
        i = self.ends[idx]
        x = self.X[i - self.window + 1 : i + 1]  # (T,C)
        y = self.Y[i]  # (P,)
        d = self.D[i]
        # normalize per feature with train-only stats
        x = torch.tensor(x, dtype=torch.float32)
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        x = (x - self.mu) / self.sigma
        x = torch.clamp(x, -8.0, 8.0)
        y = torch.tensor(y, dtype=torch.float32)

        return {"x": x, "y": y, "date_id": torch.tensor(d, dtype=torch.int64)}


def masked_mse(pred: torch.Tensor, y: torch.Tensor) -> torch.Tensor:

    mask = torch.isfinite(y)

    if not mask.any():
        return pred.new_tensor(0.0)
    
    return ((pred - torch.nan_to_num(y, nan=0.0)) ** 2)[mask].mean()


# ---------- main ----------
@dataclass
class CFG:
    window: int
    stride: int
    batch_size: int
    epochs: int
    val_frac: float
    lr: float
    weight_decay: float
    d_model: int
    n_layers: int
    n_heads: int
    ffn: int
    dropout: float
    lora_enable: bool
    lora_rank: int
    resume_ckpt: str
    out_run_name: str


def main(cfg: CFG) -> None:
    df = pd.read_parquet(ALIGNED_PATH)
    feature_cols = pick_feature_cols(df)
    target_cols = pick_target_cols(df)

    print(f"[ft] rows={len(df)}  feat={len(feature_cols)}  tgt={len(target_cols)}")

    days = df["label_date_id"].to_numpy(np.int32)
    train_days, val_days = time_split(days, cfg.val_frac)

    # compute scaler on TRAIN DAYS only
    mu, sigma = robust_mu_sigma(df[df["label_date_id"].isin(train_days)][feature_cols])
    ds_tr = GlobalSeqDataset(
        df,
        feature_cols,
        target_cols,
        cfg.window,
        cfg.stride,
        train_days,
        mu,
        sigma,
        clip=8.0,
    )

    ds_va = GlobalSeqDataset(
        df,
        feature_cols,
        target_cols,
        cfg.window,
        cfg.stride,
        val_days,
        mu,
        sigma,
        clip=8.0,
    )

    dl_tr = DataLoader(
        ds_tr,
         batch_size=min(cfg.batch_size, max(1, len(ds_tr))), # You can also increase sample count by reducing stride (e.g., stride=10 or 5).
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
        drop_last=False,
    )

    dl_va = DataLoader(
        ds_va,
        batch_size=cfg.batch_size * 2,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # build model (num_features = Kaggle features)
    model = build_timexer(
        num_features=len(feature_cols),
        num_targets=len(target_cols),
        d_model=cfg.d_model,
        n_layers=cfg.n_layers,
        n_heads=cfg.n_heads,
        ffn_hidden=cfg.ffn,
        dropout=cfg.dropout,
        patch_size=20,
        stride=20,
        use_cls=False,
    ).to(device)

    if cfg.lora_enable:
        apply_lora_to_attn(
            model, r=cfg.lora_rank, alpha=2 * cfg.lora_rank, freeze_base=True
        )
        model.to(device)

    # resume base weights
    if cfg.resume_ckpt:
        print(f"[ft] loading checkpoint: {cfg.resume_ckpt}")
        # uncomment to check feature overlap
        # ckpt = torch.load(cfg.resume_ckpt, map_location="cpu")
        # # sanity check before building the model or touching weights
        # ov = report_feature_overlap(
        #     ckpt=ckpt,
        #     current_features=feature_cols,   # your freshly built list
        #     max_examples=10                  # show up to 10 examples per list
        # )
        # # Example branching (you can keep it manual and just read the printout if you prefer):
        # if ov["has_old_list"] and ov["overlap_ratio_new"] >= 0.30:
        #     print("[ft] Will use Option B after model is constructed.")
        # else:
        #     print("[ft] Will use Option A (fresh patcher init).")
        # Option A: fresh patcher init

        ckpt = torch.load(cfg.resume_ckpt, map_location=device)
        state = dict(ckpt["model"])  # make a mutable copy

        load_backbone_except_patcher_and_head(model, state)

        # model.load_state_dict(ckpt["model"], strict=False)

    optim = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )

    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=max(1, cfg.epochs))

    out_dir = DATA_MODELS / "timexer_kaggle_ft" / cfg.out_run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    best = -1e9

    for ep in range(1, cfg.epochs + 1):
        model.train()
        tot, steps = 0.0, 0

        for b in dl_tr:
            x = b["x"].to(device)
            y = b["y"].to(device)
            p = model(x)
            loss = masked_mse(p, y)
            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            tot += float(loss.item())
            steps += 1

        # at epoch end
        if steps > 0:
            sched.step()
        else:
            print("[warn] no training steps this epoch (len(ds_tr) < batch_size?). "
                  "Reduce batch_size or set drop_last=False or reduce stride.")

        # validation: Sharpe of daily Spearman (expects date_id + target_* + prediction_*)
        model.eval()

        with torch.no_grad():
            rows = []
            for b in dl_va:
                x = b["x"].to(device)
                y = b["y"].cpu().numpy()
                p = model(x).detach().cpu().numpy()
                d = b["date_id"].cpu().numpy()
                rows.append((d, y, p))

        date_ids = np.concatenate([r[0] for r in rows], axis=0)
        Y = np.concatenate([r[1] for r in rows], axis=0)
        P = np.concatenate([r[2] for r in rows], axis=0)

        num_targets = Y.shape[1]
        assert P.shape[1] == num_targets == len(target_cols), \
            f"Shape mismatch: P:{P.shape}, Y:{Y.shape}, len(target_cols):{len(target_cols)}"

        # 1) derive suffixes from the actual column names, preserving order used during training
        suffixes = [c.split("target_", 1)[1] for c in target_cols]

        # 2) build DataFrame in one shot (also avoids fragmentation warnings)
        frame = pd.concat(
            [
                pd.DataFrame({"date_id": date_ids}),
                pd.DataFrame(Y, columns=[f"target_{s}"     for s in suffixes]),
                pd.DataFrame(P, columns=[f"prediction_{s}" for s in suffixes]),
            ],
            axis=1,
        )

        # 3) optional: safety assertion to catch any future mismatch early
        assert all(f"target_{s}" in frame.columns for s in suffixes)
        assert all(f"prediction_{s}" in frame.columns for s in suffixes)

        # one time checks, uncomment below to print out additional sanity checks
        ###
        # how many validation days will be scored?
        # print("val unique days:", frame["date_id"].nunique())

        # # count finite targets per day (Spearman needs >= 2)
        # tgt_cols = [c for c in frame.columns if c.startswith("target_")]
        # valid_per_day = frame[tgt_cols].apply(lambda r: np.isfinite(r).sum(), axis=1)
        # print("finite targets per day: min", int(valid_per_day.min()), "median", int(valid_per_day.median()))

        # # Spearman for a single day (first row) to verify it's not identically zero
        # from scipy.stats import spearmanr
        # r0 = spearmanr(
        #     frame.loc[frame.index[0], tgt_cols].values,
        #     frame.loc[frame.index[0], [f"prediction_{c.split('target_',1)[1]}" for c in tgt_cols]].values,
        #     nan_policy="omit",
        # ).correlation
        # print("sample day Spearman:", r0)
        ###

        score = sharpe_of_daily_spearman(
            frame,
            target_prefix="target_",
            pred_prefix="prediction_",
            date_col="date_id",
        )

        tr_loss = tot / max(1, steps)

        print(
            f"[{ep:03d}/{cfg.epochs}] train_loss={tr_loss:.6f}  val_SharpeSpearman={score:.6f}  lr={sched.get_last_lr()[0]:.2e}"
        )

        if score > best:
            best = score

            torch.save(
                {
                    "model": model.state_dict(),
                    "best_score": float(best),
                    "epoch": ep,
                    "feature_cols": feature_cols,
                    "target_cols": target_cols,
                },
                out_dir / "best.pt",
            )

            frame.to_parquet(out_dir / "val_preds.parquet", index=False)

            with open(out_dir / "val_report.json", "w") as f:
                json.dump(
                    {
                        "epoch": ep,
                        "val_sharpe_spearman": float(best),
                        "train_loss": float(tr_loss),
                    },
                    f,
                    indent=2,
                )

    print(f"[ft] best val SharpeSpearman = {best:.4f}")
    print(f"[ft] saved → {out_dir / 'best.pt'}")


if __name__ == "__main__":
    # EDIT these explicitly for your run
    cfg = CFG(
        window=160,
        # With window=160 and stride=20, you’ll only get ~((N - 160)/20) windows.
        # Lower stride (e.g., 10 or 5) to multiply samples.
        stride=10,
        batch_size=32,
        epochs=10,  # short FT
        val_frac=0.2,
        lr=5e-4,  # smaller LR for FT
        weight_decay=0.05,
        d_model=512,
        n_layers=6,
        n_heads=8,
        ffn=2048,
        dropout=0.1,
        lora_enable=True,  # mirrors your setup
        lora_rank=8,
        # "last_run.txt").read_text().strip() + # for later runs
        resume_ckpt=str(DATA_MODELS / "timexer" / "best.pt"),
        out_run_name="kaggle_ft_v3",
    )

    main(cfg)  # run the training script with the specified config
