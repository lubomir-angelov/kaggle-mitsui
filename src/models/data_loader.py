import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, Subset

class DynamicDataLoader(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        feature_cols: list[str],
        target_cols: list[str],
        window: int,
        min_obs: int,
        stride: int,
        min_target_notna: int,
    ):
        assert window > 0
        assert min_obs >= 0
        assert stride >= 1
        assert min_target_notna >= 1

        self.feature_cols = feature_cols
        self.target_cols = target_cols
        self.window = window

        self.series_X: list[np.ndarray] = []
        self.series_y: list[np.ndarray] = []
        self.series_dates: list[np.ndarray] = []
        self.series_symid: list[int] = []

        index_pairs: list[tuple[int, int]] = []  # (series_idx, end_idx)

        for sid, (sym, g) in enumerate(df.groupby("symbol", sort=False)):
            g = g.sort_values("date")
            if len(g) < max(window, min_obs):
                continue

            Xg = g[feature_cols].to_numpy(dtype=np.float32, copy=False)
            yg = g[target_cols].to_numpy(dtype=np.float32, copy=False)
            dg = g["date"].to_numpy(copy=False)

            targ_mask = np.isfinite(yg).sum(axis=1) >= min_target_notna

            start = window - 1
            for i in range(start, len(g)):
                if ((i - start) % stride) != 0:
                    continue
                if i + 1 < min_obs:
                    continue
                if not targ_mask[i]:
                    continue
                # accept; materialize later
                index_pairs.append((len(self.series_X), i))

            # store series containers once
            self.series_X.append(Xg)
            self.series_y.append(yg)
            self.series_dates.append(dg)
            self.series_symid.append(sid)

        self.index_pairs = np.array(index_pairs, dtype=np.int64)
        if self.index_pairs.size == 0:
            raise RuntimeError("No windows indexed. Relax filters.")

    def __len__(self) -> int:
        return int(self.index_pairs.shape[0])

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        sidx, i = self.index_pairs[idx]
        Xg = self.series_X[sidx]
        yg = self.series_y[sidx]
        dg = self.series_dates[sidx]
        # slice window ending at i
        x = Xg[i - self.window + 1 : i + 1]   # (W, C)
        y = yg[i]                              # (P,)
        d = dg[i]

        return {
            "x": torch.from_numpy(x),                      # float32
            "y": torch.from_numpy(y),                      # float32
            "date_id": torch.tensor(d, dtype=torch.int64),
            "sym_id": torch.tensor(self.series_symid[sidx], dtype=torch.int32),
        }

    def split_by_date(self, cutoff: np.datetime64) -> tuple[Subset, Subset]:
        # masks per sample from stored dates
        dates = np.array([self.series_dates[s][i] for s, i in self.index_pairs], dtype="datetime64[ns]")
        train_idx = np.nonzero(dates < cutoff)[0]
        valid_idx = np.nonzero(dates >= cutoff)[0]
        return Subset(self, train_idx.tolist()), Subset(self, valid_idx.tolist())
