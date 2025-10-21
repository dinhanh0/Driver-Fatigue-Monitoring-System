from pathlib import Path
import json
from typing import List, Dict

import numpy as np
import torch
from torch.utils.data import Dataset


def _list_windows(root: Path, split: str) -> List[Path]:
    p = root / split
    if not p.exists():
        return []
    windows: List[Path] = []
    for sub in p.iterdir():
        if not sub.is_dir():
            continue
        for fn in sub.iterdir():
            if fn.suffix == ".npz":
                windows.append(fn)
    return sorted(windows)


class ClipDataset(Dataset):
    """Dataset that reads precomputed sliding-window .npz files and returns fixed-length clips.

    Each .npz is expected to contain arrays: X_video (T,C,H,W) or (T,3,H,W) float32, X_feats (T, F), y (int)

    Args:
        splits_csv: path to the dataset index CSV (not used directly here but kept for API compatibility)
        subset: one of 'train','val','test' or a custom split name
        windows_root: base folder where windows were written (defaults to data/processed/windows)
        img_size: expected image size (kept for API; dataset reads image tensors stored in .npz)
        seq_len: desired sequence length (T) returned by the dataset. If a window has more frames
                 than seq_len it will be center-cropped; if fewer frames it will be padded/repeated.
    """

    def __init__(self, splits_csv: str, subset: str = "train", windows_root: str = "data/processed/windows",
                 img_size: int = 224, seq_len: int = 32, cache_size: int = 0):
        super().__init__()
        self.subset = subset
        self.seq_len = int(seq_len)
        self.img_size = int(img_size)
        self.windows_root = Path(windows_root)
        # store paths as plain strings to avoid multiprocessing pickling issues on Windows
        self.windows = [str(p) for p in _list_windows(self.windows_root, self.subset)]

        # Try to load label mapping if present
        lm = self.windows_root / "label_mapping.json"
        if lm.exists():
            try:
                with open(lm, 'r') as f:
                    self.label_map = json.load(f)
            except Exception:
                self.label_map = {}
        else:
            self.label_map = {}

        # Simple in-memory LRU cache for loaded .npz files. cache_size=0 disables caching.
        self.cache_size = int(cache_size)
        self._cache = {}  # path_str -> dict(X_video, X_feats, y)
        self._cache_order = []  # list of path_str for LRU ordering

    def __len__(self):
        return len(self.windows)

    def _read_npz(self, p) -> Dict:
        # p may be a Path or a string; normalize to string
        pstr = str(p)
        # Serve from cache if present
        if self.cache_size > 0 and pstr in self._cache:
            # move to end as most-recently-used
            try:
                self._cache_order.remove(pstr)
            except ValueError:
                pass
            self._cache_order.append(pstr)
            return self._cache[pstr]

        try:
            data = np.load(pstr)
            Xv = data["X_video"]  # expected shape (T,C,H,W)
            Xf = data["X_feats"]
            y = int(data["y"].tolist())
            rec = {"X_video": Xv, "X_feats": Xf, "y": y}
            # insert into cache
            if self.cache_size > 0:
                self._cache[pstr] = rec
                self._cache_order.append(pstr)
                # evict if over capacity
                if len(self._cache_order) > self.cache_size:
                    old = self._cache_order.pop(0)
                    try:
                        del self._cache[old]
                    except KeyError:
                        pass
            return rec
        except Exception:
            # Return a dummy empty example that will be skipped later; but to keep indexing stable
            return {"X_video": np.zeros((self.seq_len, 3, self.img_size, self.img_size), dtype="float32"),
                    "X_feats": np.zeros((self.seq_len, 4), dtype="float32"),
                    "y": 0}

    def _sample_clip(self, X: np.ndarray) -> np.ndarray:
        # X shape: (T, C, H, W) or (T, F)
        T = X.shape[0]
        if T == self.seq_len:
            return X
        if T > self.seq_len:
            # center crop
            start = max(0, (T - self.seq_len) // 2)
            return X[start:start + self.seq_len]
        # T < seq_len -> pad by repeating last frame
        pad = self.seq_len - T
        last = X[-1:]
        pads = np.repeat(last, pad, axis=0)
        return np.concatenate([X, pads], axis=0)

    def __getitem__(self, idx: int):
        p = self.windows[idx]
        rec = self._read_npz(p)
        Xv = rec["X_video"]
        Xf = rec["X_feats"]
        y = int(rec["y"]) if rec.get("y") is not None else 0

        # Ensure shapes: (T,C,H,W) and (T,F)
        Xv_clip = self._sample_clip(Xv)
        Xf_clip = self._sample_clip(Xf)

        # Convert to torch tensors: clip -> (T,C,H,W) -> we want (C,T,H,W) or (T,C,H,W)? Training expects (B,T,C,H,W)
        # We'll return clip as float32 tensor shaped (T,C,H,W) and collate_fn in DataLoader will stack into (B,T,C,H,W)
        clip_tensor = torch.from_numpy(Xv_clip).float()
        feats_tensor = torch.from_numpy(Xf_clip).float()
        label_tensor = torch.tensor(y, dtype=torch.long)

        return {"clip": clip_tensor, "feats": feats_tensor, "label": label_tensor, "meta": str(p)}

    def prefetch(self, n: int = 100):
        """Warm the in-memory cache by loading the first `n` windows."""
        if self.cache_size <= 0:
            return 0
        n = min(n, len(self.windows))
        for p in self.windows[:n]:
            self._read_npz(p)
        return min(n, len(self._cache))

