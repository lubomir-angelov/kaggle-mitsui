# src/train.py
from __future__ import annotations
import argparse, json, os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional, Dict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import Subset
import torch.nn.functional as F

from src.models.timexer import build_timexer
from src.models.adapters.lora import apply_lora_to_attn
from src.models.data_loader import DynamicDataLoader
from src.metrics.sharpe_spearman import sharpe_of_daily_spearman
from src.pretrain.masked_recon import pretrain_masked_reconstruction, ReconHead


# ---------------------------------------------------------------------
# Paths & basic setup
# ---------------------------------------------------------------------
REPO_ROOT = os.environ.get("REPO_ROOT", "/home/ubuntu/repos/kaggle-mitsui")
DATA_PROCESSED = Path(f"{REPO_ROOT}/data/processed")
DATA_MODELS    = Path(f"{REPO_ROOT}/data/models")
DATA_MODELS.mkdir(parents=True, exist_ok=True)

FEATURES_PATH  = DATA_PROCESSED / "features_long.parquet"


# ---------------------------------------------------------------------
# Utilities: column selection, date splits
# ---------------------------------------------------------------------
EXCLUDE = {"date", "symbol", "src", "unit", "frequency"}
def pick_feature_cols(df: pd.DataFrame) -> List[str]:
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    feat_cols = [c for c in num_cols if not c.startswith("target_")]
    return feat_cols

def pick_target_cols(df: pd.DataFrame, like: str = "target_ret_fwd_") -> List[str]:
    return [c for c in df.columns if c.startswith(like)]

def make_time_split_by_fraction(
    dates: pd.Series,
    val_frac: float = 0.2,
    embargo_days: int = 0,
) -> Tuple[pd.DatetimeIndex, pd.DatetimeIndex, pd.DatetimeIndex]:
    """
    Split unique dates into train / (embargo) / val, contiguous by time.
    Returns: train_days, emb_days, val_days
    """
    u = pd.Index(pd.to_datetime(dates)).sort_values().unique()
    n_val = max(10, int(round(len(u) * val_frac)))
    val_days = u[-n_val:]
    emb = pd.Timedelta(days=embargo_days)
    train_days = u[u <= (val_days[0] - emb)]
    if embargo_days > 0:
        emb_days = u[(u > (val_days[0] - emb)) & (u < (val_days[0]))]
    else:
        emb_days = u[[]]
    return train_days, emb_days, val_days

# ---------------------------------------------------------------------
# Normalize Data Set, remove NaNs
# ---------------------------------------------------------------------

def compute_robust_scaler(df: pd.DataFrame, feature_cols: list[str], train_days: pd.DatetimeIndex) -> tuple[np.ndarray, np.ndarray]:
    train_mask = df["date"].isin(train_days)
    S = df.loc[train_mask, feature_cols].replace([np.inf, -np.inf], np.nan)
    mu = S.median().to_numpy(dtype=np.float32)
    iqr = (S.quantile(0.75) - S.quantile(0.25)).to_numpy(dtype=np.float32)
    std = S.std(ddof=0).to_numpy(dtype=np.float32)

    # choose a stable scale per feature
    sigma = np.where(iqr > 1e-6, iqr, std)
    sigma = np.where(sigma > 1e-6, sigma, 1.0).astype(np.float32)
    return mu, sigma


class NormalizeDataset(Dataset):
    """Wrap another dataset; standardize x and make it finite."""
    def __init__(self, base: Dataset, mu: np.ndarray, sigma: np.ndarray, clip: float = 8.0):
        self.base = base
        self.mu = torch.tensor(mu, dtype=torch.float32)
        self.sigma = torch.tensor(sigma, dtype=torch.float32)
        self.clip = float(clip)

    def __len__(self): 
        return len(self.base)

    def __getitem__(self, i: int):
        item = self.base[i]
        x = item["x"].to(torch.float32)              # (T,C)
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        # (T,C) − (C,) / (C,)
        x = (x - self.mu) / self.sigma
        x = torch.clamp(x, -self.clip, self.clip)
        item["x"] = x
        return item


# ---------------------------------------------------------------------
# Dataset: builds sliding windows per symbol
# ---------------------------------------------------------------------
@dataclass
class WindowedArrays:
    X: np.ndarray        # (N, T, C)
    y: np.ndarray        # (N, P) with NaN allowed; loss masks will be applied
    date_id: np.ndarray  # (N,) end-date per window
    symbol_id: np.ndarray# (N,)

class SeqDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray, date_id: np.ndarray):
        self.X = X
        self.y = y
        self.date_id = date_id

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, idx: int):
        x = self.X[idx]
        y = self.y[idx]
        d = self.date_id[idx]
        # to tensors
        x = torch.tensor(x, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32)
        return {"x": x, "y": y, "date_id": int(d)}

def build_windows(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_cols: List[str],
    window: int = 160,
    min_obs: int = 300,
) -> WindowedArrays:
    """
    Build (X,y,date_id) by sliding windows **per symbol**.
    - Uses rows where targets exist (at least one non-NaN) at the window end.
    - Fills NaNs in X with 0; model can learn a 'missing' value if you kept mask channels.
    """
    parts_X, parts_y, parts_date, parts_sym = [], [], [], []
    sym_to_id: Dict[str, int] = {}

    for sid, (sym, g) in enumerate(df.groupby("symbol", sort=False)):
        if len(g) < min_obs:
            continue
        sym_to_id[sym] = sid

        g = g.sort_values("date").reset_index(drop=True)
        Xmat = g[feature_cols].to_numpy(dtype=np.float32)
        Ymat = g[target_cols].to_numpy(dtype=np.float32)
        dates = g["date"].to_numpy()

        # fill NaNs in features with 0 (simple and stable)
        Xmat = np.nan_to_num(Xmat, nan=0.0, posinf=0.0, neginf=0.0)

        # slide
        for t in range(window - 1, len(g)):
            y_row = Ymat[t]
            if np.all(np.isnan(y_row)):
                continue
            x_win = Xmat[t - window + 1 : t + 1]           # (T, C)
            parts_X.append(x_win)
            parts_y.append(y_row)
            parts_date.append(dates[t])
            parts_sym.append(sid)

    if not parts_X:
        raise RuntimeError("No windows built. Check `window` and data coverage.")
    X = np.stack(parts_X, axis=0)
    y = np.stack(parts_y, axis=0)
    date_id = np.array([pd.Timestamp(d).toordinal() for d in parts_date], dtype=np.int32)
    symbol_id = np.array(parts_sym, dtype=np.int32)
    return WindowedArrays(X=X, y=y, date_id=date_id, symbol_id=symbol_id)

def select_by_dates(arr: WindowedArrays, keep_days: pd.DatetimeIndex) -> WindowedArrays:
    mask = np.isin(arr.date_id, np.array([d.toordinal() for d in keep_days], dtype=np.int32))
    return WindowedArrays(
        X=arr.X[mask],
        y=arr.y[mask],
        date_id=arr.date_id[mask],
        symbol_id=arr.symbol_id[mask],
    )

# ---------------------------------------------------------------------
# Training / eval
# ---------------------------------------------------------------------
def masked_mse_loss(pred: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    pred,y: (B, P). Compute MSE only on non-NaN targets.
    """
    mask = torch.isfinite(y)
    if not mask.any():
        return torch.tensor(0.0, device=pred.device)
    diff = (pred - torch.nan_to_num(y, nan=0.0)) ** 2
    return diff[mask].mean()

@torch.no_grad()
def eval_sharpe_spearman(model: nn.Module, loader: DataLoader, device: str,
                         target_cols: List[str]) -> Tuple[float, pd.DataFrame]:
    model.eval()
    all_date, all_y, all_p = [], [], []
    for batch in loader:
        x = batch["x"].to(device)        # (B,T,C)
        y = batch["y"].cpu().numpy()     # (B,P) with NaN possible
        p = model(x).detach().cpu().numpy()

        all_date.append(batch["date_id"].cpu().numpy())
        all_y.append(y)
        all_p.append(p)

    date_ids = np.concatenate(all_date, axis=0)
    Y = np.concatenate(all_y, axis=0)
    P = np.concatenate(all_p, axis=0)

    df = pd.DataFrame({"date_id": date_ids})
    for i, tcol in enumerate(target_cols):
        # tcol already looks like "target_ret_fwd_5d"
        df[tcol] = Y[:, i]
        pred_name = "prediction_" + tcol.replace("target_", "")
        df[pred_name] = P[:, i]
    
    # optional sanity check
    expected_pred = {"prediction_" + c.replace("target_", "") for c in target_cols}
    missing = expected_pred - set(df.columns)
    if missing:
        raise ValueError(f"Missing prediction columns: {sorted(missing)}")


    score = sharpe_of_daily_spearman(
        df.rename(columns={"date_id": "date_id"}),
        target_prefix="target_",
        pred_prefix="prediction_",
        date_col="date_id",
    )
    return float(score), df


# ---------------------------------------------------------------------
# Main training
# ---------------------------------------------------------------------
@dataclass
class Args:
    window: int
    batch_size: int
    epochs: int
    lr: float
    weight_decay: float
    val_frac: float
    embargo_days: int
    d_model: int
    n_layers: int
    n_heads: int
    ffn: int
    dropout: float
    patch_size: int
    stride: Optional[int]
    use_cls: bool
    device: str
    lora: bool
    lora_rank: int
    pretrain_epochs: int
    pretrain_mask_ratio: float
    pretrain_span: int

def parse_args() -> Args:
    import yaml

    # Locate config file:
    # 1) ENV var TIMEXER_TRAIN_YAML
    # 2) <REPO_ROOT>/timexer_train.yaml
    # 3) <REPO_ROOT>/config/timexer_train.yaml
    # 4) <REPO_ROOT>/configs/timexer_train.yaml
    cand = []
    env_path = os.environ.get("TIMEXER_TRAIN_YAML")
    if env_path:
        cand.append(Path(env_path))
    cand += [
        Path(REPO_ROOT) / "timexer_train.yaml",
        Path(REPO_ROOT) / "config" / "timexer_train.yaml",
        Path(REPO_ROOT) / "configs" / "timexer_train.yaml",
    ]
    cfg_path = next((p for p in cand if p.exists()), None)
    if cfg_path is None:
        raise FileNotFoundError("Could not find timexer_train.yaml. "
                                "Set $TIMEXER_TRAIN_YAML or place it in REPO_ROOT/[config|configs]/.")

    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f) or {}

    d = cfg.get("data", {})
    m = cfg.get("model", {})
    t = cfg.get("train", {})
    l = cfg.get("lora", {})
    # metric block is read elsewhere when needed

    # Helpers with sensible defaults
    def _get_int(dct, key, default):
        v = dct.get(key, default)
        return None if v in (None, "None") else int(v)

    def _get_float(dct, key, default):
        v = dct.get(key, default)
        return float(v)

    def _get_bool(dct, key, default=False):
        v = dct.get(key, default)
        return bool(v)

    args = Args(
        # data
        window=_get_int(d, "window", 160),
        patch_size=_get_int(d, "patch_size", 20),
        stride=_get_int(d, "stride", 20),

        # train (finetune)
        batch_size=_get_int(t, "batch_size", 128),
        epochs=_get_int(t, "epochs_finetune", 15),
        lr=_get_float(t, "lr", 1e-3),
        weight_decay=_get_float(t, "weight_decay", 0.05),

        # extra split knobs (not in yaml → defaults)
        val_frac=_get_float(t, "val_frac", 0.20),
        embargo_days=_get_int(t, "embargo_days", 5),

        # model
        d_model=_get_int(m, "d_model", 512),
        n_layers=_get_int(m, "n_layers", 6),
        n_heads=_get_int(m, "n_heads", 8),
        ffn=_get_int(m, "ffn", 2048),
        dropout=_get_float(m, "dropout", 0.1),
        use_cls=_get_bool(m, "use_cls", False),

        # device
        device="cuda" if torch.cuda.is_available() else "cpu",

        # lora
        lora=_get_bool(l, "enable", False),
        lora_rank=_get_int(l, "rank", 8),

        # pretrain
        pretrain_epochs=_get_int(t, "epochs_pretrain", 0),
        pretrain_mask_ratio=_get_float(t, "mask_ratio", 0.2),
        pretrain_span=_get_int(t, "span", 4),
    )
    return args


def _subset_apply_embargo(subset: Subset, base_ds: DynamicDataLoader, embargo_days: int, val_start: pd.Timestamp) -> Subset:
    """
    Remove samples in the embargo window: [val_start - embargo_days, val_start).
    Assumes `subset.dataset is base_ds` and base_ds has `index_pairs` and `series_dates`.
    """
    if embargo_days <= 0:
        return subset

    idx = np.asarray(subset.indices, dtype=np.int64)
    # dates for these sample indices
    dates = np.array([ base_ds.series_dates[s][i] for (s, i) in base_ds.index_pairs[idx] ], dtype='datetime64[ns]')
    cutoff = (val_start.to_datetime64() - np.timedelta64(embargo_days, "D"))
    keep = dates < cutoff
    return Subset(base_ds, idx[keep].tolist())


class _PretrainXOnlyWrapper(Dataset):
    """Wrap a dataset/subset and expose only 'x' for masked reconstruction pretrain."""
    def __init__(self, base: Dataset):
        self.base = base
    def __len__(self) -> int:
        return len(self.base)
    def __getitem__(self, i: int):
        return {"x": self.base[i]["x"]}


def main():
    args = parse_args()

    print("[boot] parsed args:", json.dumps(vars(args), indent=2), flush=True)
    print("[boot] FEATURES_PATH:", FEATURES_PATH, "exists:", FEATURES_PATH.exists(), flush=True)

    # ------------ Load panel ------------
    df = pd.read_parquet(FEATURES_PATH)
    df["date"] = pd.to_datetime(df["date"], utc=False).dt.tz_localize(None)
    df = df.sort_values(["date", "symbol"]).reset_index(drop=True)

    feature_cols = pick_feature_cols(df)
    target_cols = pick_target_cols(df)  # e.g., ["target_ret_fwd_5d", "target_ret_fwd_21d", ...]

    print(f"Using {len(feature_cols)} features, {len(target_cols)} targets")
    if args.window < args.patch_size:
        raise ValueError("window must be >= patch_size for patching to work.")

    # ------------ Dynamic dataset (on-the-fly windows) ------------
    ds_all = DynamicDataLoader(
        df=df,
        feature_cols=feature_cols,
        target_cols=target_cols,
        window=args.window,
        min_obs=300,
        stride=int(args.stride),          # explicit decimation of end indices
        min_target_notna=1,               # require at least one target at window end
    )

    # Time split by fraction → cutoff = start of validation span
    train_days, emb_days, val_days = make_time_split_by_fraction(
        dates=df["date"],
        val_frac=float(args.val_frac),
        embargo_days=int(args.embargo_days),
    )
    val_start = val_days[0]

    # Split the dataset by date (train: < cutoff, valid: >= cutoff)
    train_ds, valid_ds = ds_all.split_by_date(val_start.to_datetime64())

    # Apply embargo on the training subset explicitly (drop near-future leakage)
    train_ds = _subset_apply_embargo(train_ds, base_ds=ds_all,
                                     embargo_days=int(args.embargo_days),
                                     val_start=val_start)
    
    # —— robust feature scaling (train-only stats) ——
    mu, sigma = compute_robust_scaler(df, feature_cols, train_days)

    # wrap both subsets so pretrain AND finetune see normalized inputs
    train_ds = NormalizeDataset(train_ds, mu, sigma, clip=8.0)
    valid_ds = NormalizeDataset(valid_ds, mu, sigma, clip=8.0)


    # DataLoaders
    dl_tr = DataLoader(
        train_ds,
        batch_size=int(args.batch_size),
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
        drop_last=True,
    )

    dl_va = DataLoader(
        valid_ds,
        batch_size=int(args.batch_size) * 2,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
    )


    device = args.device

    # ------------ Model ------------
    model = build_timexer(
        num_features=len(feature_cols),
        num_targets=len(target_cols),
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        ffn_hidden=args.ffn,
        dropout=args.dropout,
        patch_size=args.patch_size,
        stride=args.stride,
        use_cls=args.use_cls,
    ).to(device)

    # LoRA (optional)
    if args.lora:
        apply_lora_to_attn(model, r=args.lora_rank, alpha=2*args.lora_rank, freeze_base=True)

    # ------------ Optional masked-span pretrain ------------
    if args.pretrain_epochs > 0:
        print(f"[pretrain] epochs={args.pretrain_epochs}, mask_ratio={args.pretrain_mask_ratio}, span={args.pretrain_span}")
        # tiny wrapper: encoder = model up to tokens (reuse model.forward via a lambda)
        # Simpler: reuse model but grab tokens by return_tokens=True and put a small recon head.
        recon_head = ReconHead(d_model=args.d_model, out_channels=len(feature_cols)).to(device)

        # lightweight dataloader that yields only 'x' from the *training subset*
        pre_ds = _PretrainXOnlyWrapper(train_ds)
        pre_dl = DataLoader(
            pre_ds,
            batch_size=int(args.batch_size),
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            persistent_workers=True,
        )

        # encoder function: patch + blocks → tokens (match masked_recon expectation)
        class EncoderWrapper(nn.Module):
            def __init__(self, timexer: nn.Module):
                super().__init__()
                self.timexer = timexer
            def forward(self, x):
                # run through patcher + blocks, return tokens
                tokens = self.timexer.patcher(x)
                h = tokens
                for blk in self.timexer.blocks:
                    h = blk(h)
                return h  # (B,L,d)

        encoder = EncoderWrapper(model).to(device)
        pretrain_masked_reconstruction(
            encoder=encoder,
            recon_head=recon_head,
            dataloader=pre_dl,
            device=device,
            epochs=args.pretrain_epochs,
            lr=1e-3,
            mask_ratio=args.pretrain_mask_ratio,
            span=args.pretrain_span,
        )
        # Drop recon head afterwards.

    # ------------ Supervised fine-tune ------------
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=max(1, args.epochs))

    best_score = -1e9
    out_dir = DATA_MODELS / "timexer"
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / "best.pt"

    for ep in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        steps = 0
        for batch in dl_tr:
            x = batch["x"].to(device)          # (B,T,C)
            y = batch["y"].to(device)          # (B,P)
            pred = model(x)                    # (B,P)

            loss = masked_mse_loss(pred, y)

            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optim.step()

            total += float(loss.item()); steps += 1

        sched.step()
        train_loss = total / max(1, steps)

        # validation: Sharpe of daily Spearman
        val_score, val_frame = eval_sharpe_spearman(model, dl_va, device, target_cols)

        print(f"[{ep:03d}/{args.epochs}] train_loss={train_loss:.6f}  val_SharpeSpearman={val_score:.4f}  lr={sched.get_last_lr()[0]:.2e}")

        # keep best
        if val_score > best_score:
            best_score = val_score
            torch.save({
                "model": model.state_dict(),
                "args": vars(args),
                "feature_cols": feature_cols,
                "target_cols": target_cols,
                "best_score": best_score,
            }, ckpt_path)
            # also persist a small val report
            rep = {
                "epoch": ep,
                "val_sharpe_spearman": float(val_score),
                "train_loss": float(train_loss),
                "n_train_samples": int(len(train_ds)),
                "n_val_samples": int(len(valid_ds)),
            }
            with open(out_dir / "val_report.json", "w") as f:
                json.dump(rep, f, indent=2)
            # store a feather/parquet of the latest val preds if you like
            try:
                val_frame.to_parquet(out_dir / "val_preds.parquet", index=False)
            except Exception:
                pass

    print(f"Best val SharpeSpearman = {best_score:.4f}")
    print(f"Saved best checkpoint → {ckpt_path}")

if __name__ == "__main__":
    main()
