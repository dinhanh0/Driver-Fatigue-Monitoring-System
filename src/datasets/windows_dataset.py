import os, glob
import numpy as np
import torch
from torch.utils.data import Dataset

class WindowsNPZDataset(Dataset):
    def __init__(self, root_dir: str, split: str, limit: int=None):
        split_dir = os.path.join(root_dir, split)
        self.paths = sorted(glob.glob(os.path.join(split_dir, "**", "*.npz"), recursive=True))
        if limit:
            self.paths = self.paths[:limit]
        if not self.paths:
            raise FileNotFoundError(f"No .npz found under {split_dir}")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        with np.load(path, allow_pickle=False) as npz:
            if "X_video" in npz:
                Xv = npz["X_video"]
            elif "video" in npz:
                Xv = npz["video"]
            else:
                raise KeyError(f"{path} missing X_video/video")

            if "X_feats" in npz:
                Xf = npz["X_feats"]
            elif "feats" in npz:
                Xf = npz["feats"]
            else:
                raise KeyError(f"{path} missing X_feats/feats")

            y = int(npz["y"] if "y" in npz else npz["label"])

        # Ensure types & shapes
        Xv = torch.from_numpy(Xv).float()           # [T,3,224,224]
        Xf = torch.from_numpy(Xf).float()           # [T,D]
        y  = torch.tensor(y, dtype=torch.long)

        return {"video": Xv, "feats": Xf, "label": y, "path": path}