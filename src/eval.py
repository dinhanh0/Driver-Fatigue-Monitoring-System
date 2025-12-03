# src/eval.py
# Evaluate VideoFeatResNet18LSTM binary model on precomputed video windows.
# - Uses both X_video (frames) and X_feats (EAR, MAR, blur, illum)
# - Applies ImageNet normalization (must match train.py)
# - Reports per-class precision/recall/F1 and macro F1

from pathlib import Path
from collections import Counter
import sys
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

# ------------------------------------------------------------------
# Paths / imports
# ------------------------------------------------------------------
THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[1]

# So `models` is importable
sys.path.append(str(PROJECT_ROOT))

from models.resnet18_lstm import ResNet18LSTM  # same backbone as train.py

WINDOWS_ROOT = PROJECT_ROOT / "data" / "processed" / "windows_from_index"
MODEL_DIR = PROJECT_ROOT / "models"
# Evaluate the best-accuracy model by default (you can swap to best_loss if you want)
MODEL_PATH = MODEL_DIR / "resnet18_lstm_binary_best_acc.pt"

# ------------------------------------------------------------------
# ImageNet normalization (MUST match train.py)
# ------------------------------------------------------------------
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1)
IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1)

# ------------------------------------------------------------------
# Binary label mapping (same idea as train.py)
# ------------------------------------------------------------------
def raw_to_binary(y_raw: int) -> int:
    """
    Map raw labels to binary:
      0 -> normal
      1 -> impaired

    Any other value is treated as:
      0 if y_raw <= 0
      1 if y_raw > 0
    """
    if y_raw in (0, 1):
        return int(y_raw)
    return 0 if y_raw <= 0 else 1


# ------------------------------------------------------------------
# Dataset: X_video + X_feats (mirrors WindowVideoFeatDataset in train.py)
# ------------------------------------------------------------------
class WindowVideoFeatDataset(Dataset):
    def __init__(self, split: str):
        self.split = split
        split_dir = WINDOWS_ROOT / split
        if not split_dir.exists():
            raise RuntimeError(f"Split folder does not exist: {split_dir}")

        self.files = sorted(split_dir.rglob("*.npz"))
        if not self.files:
            raise RuntimeError(f"No .npz files found in {split_dir}")

        # Probe one file to get sequence / spatial dims and feat dim
        sample = np.load(self.files[0])
        X_video = sample["X_video"]    # [T, C, H, W]
        X_feats = sample["X_feats"]    # [T, F]
        self.seq_len = X_video.shape[0]
        self.C = X_video.shape[1]
        self.H = X_video.shape[2]
        self.W = X_video.shape[3]
        self.feat_dim = X_feats.shape[1]

        print(f"[{split.upper()}] {len(self.files)} windows")
        print(f"  Sequence length: {self.seq_len}, CxHxW = {self.C}x{self.H}x{self.W}")
        print(f"  Feature dim: {self.feat_dim}")

        # Raw + binary label histogram (sanity check)
        raw_counts = Counter()
        for p in self.files:
            d = np.load(p)
            y_raw = int(d["y"])
            raw_counts[y_raw] += 1
        print(f"  Raw label counts (y values): {dict(raw_counts)}")

        bin_counts = Counter()
        for y_raw, c in raw_counts.items():
            bin_counts[raw_to_binary(y_raw)] += c
        print(f"  Binary label counts (0=normal, 1=impaired): {dict(bin_counts)}\n")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        d = np.load(self.files[idx])
        clip = d["X_video"]   # [T, C, H, W]
        feats = d["X_feats"]  # [T, F]
        y_raw = int(d["y"])
        y_bin = raw_to_binary(y_raw)

        # Ensure consistent T (pad or trim) for both clip and feats
        x = clip
        f = feats
        if x.shape[0] > self.seq_len:
            x = x[: self.seq_len]
            f = f[: self.seq_len]
        elif x.shape[0] < self.seq_len:
            pad_len = self.seq_len - x.shape[0]
            x_pad = np.zeros((pad_len, self.C, self.H, self.W), dtype=np.float32)
            f_pad = np.zeros((pad_len, self.feat_dim), dtype=np.float32)
            x = np.concatenate([x, x_pad], axis=0)
            f = np.concatenate([f, f_pad], axis=0)

        # To float32 tensors
        x = x.astype(np.float32)  # [T, C, H, W], in [0,1] from preprocess
        f = f.astype(np.float32)  # [T, F]
        x = torch.from_numpy(x)   # [T, C, H, W]
        f = torch.from_numpy(f)   # [T, F]

        # Apply ImageNet normalization per-frame (broadcast mean/std over T)
        x = (x - IMAGENET_MEAN) / IMAGENET_STD

        y = torch.tensor([y_bin], dtype=torch.float32)  # [1]

        return x, f, y


# ------------------------------------------------------------------
# Model: video encoder (ResNet18LSTM) + aggregated handcrafted feats
# (same as VideoFeatResNet18LSTM in train.py)
# ------------------------------------------------------------------
class VideoFeatResNet18LSTM(nn.Module):
    """
    Use ResNet18LSTM as a video encoder, then concatenate
    an aggregated version of X_feats and classify with a final linear layer.
    """
    def __init__(
        self,
        feat_dim_extra: int,        # dimension of handcrafted features per frame (F)
        lstm_hidden: int = 256,
        lstm_layers: int = 1,
        bidirectional: bool = False,
        dropout: float = 0.5,
        use_nat: bool = False,
        nat_kernel: int = 7,
        nat_dilation: int = 1,
        nat_blocks: int = 1,
        backbone_name: str = "resnet18",
    ):
        super().__init__()

        # This is the feature dimension of the LSTM output at the last time step
        backbone_feat_dim = lstm_hidden * (2 if bidirectional else 1)
        self.backbone_feat_dim = backbone_feat_dim
        self.feat_dim_extra = feat_dim_extra

        # ResNet18LSTM outputs a vector of size backbone_feat_dim
        self.backbone = ResNet18LSTM(
            num_classes=backbone_feat_dim,
            feat_dim=512,
            lstm_hidden=lstm_hidden,
            lstm_layers=lstm_layers,
            bidirectional=bidirectional,
            dropout=dropout,
            use_nat=use_nat,
            nat_kernel=nat_kernel,
            nat_dilation=nat_dilation,
            nat_blocks=nat_blocks,
            backbone_name=backbone_name,
        )

        # Final classifier takes [backbone_feat | agg_feats] -> 1 logit
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(backbone_feat_dim + feat_dim_extra, 1),
        )

    def forward(self, clip, feats_seq):
        """
        clip:      [B, T, C, H, W]
        feats_seq: [B, T, F]  (EAR, MAR, blur, illum per frame)
        """
        # Backbone gives [B, backbone_feat_dim]
        backbone_vec = self.backbone(clip)               # [B, D]

        # Aggregate handcrafted features over time (simple mean)
        feats_agg = feats_seq.mean(dim=1)                # [B, F]

        # Concatenate and classify
        x = torch.cat([backbone_vec, feats_agg], dim=1)  # [B, D+F]
        logits = self.classifier(x).squeeze(-1)          # [B]
        return logits


# ------------------------------------------------------------------
# Metrics helper (same structure as train.py)
# ------------------------------------------------------------------
def _compute_class_metrics(tp: int, tn: int, fp: int, fn: int):
    """
    Given confusion matrix entries for positive class (1):
        tp = true positives  (pred=1, y=1)
        tn = true negatives  (pred=0, y=0)
        fp = false positives (pred=1, y=0)
        fn = false negatives (pred=0, y=1)

    Returns a dict with precision, recall, F1 for class 0 and 1,
    plus macro F1.
    """
    # Class 1 (impaired)
    prec1 = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec1  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1_1  = 2 * prec1 * rec1 / (prec1 + rec1) if (prec1 + rec1) > 0 else 0.0

    # Class 0 (normal): treat TN as "TP0"
    prec0 = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    rec0  = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    f1_0  = 2 * prec0 * rec0 / (prec0 + rec0) if (prec0 + rec0) > 0 else 0.0

    macro_f1 = (f1_0 + f1_1) / 2.0

    return {
        "prec0": prec0,
        "rec0": rec0,
        "f1_0": f1_0,
        "prec1": prec1,
        "rec1": rec1,
        "f1_1": f1_1,
        "macro_f1": macro_f1,
    }


# ------------------------------------------------------------------
# Load model (same arch as train.py, including feats)
# ------------------------------------------------------------------
def load_model(device, use_nat: bool = False, freeze_backbone: bool = False):
    """
    Build the same architecture as train.py and load weights from MODEL_PATH.
    We read feat_dim from the checkpoint metadata to match training.
    """
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Checkpoint not found at {MODEL_PATH}")

    ckpt = torch.load(MODEL_PATH, map_location=device)
    state_dict = ckpt["model_state"]
    feat_dim_extra = ckpt.get("feat_dim", 4)  # fallback if not present

    model = VideoFeatResNet18LSTM(
        feat_dim_extra=feat_dim_extra,
        lstm_hidden=256,
        lstm_layers=1,
        bidirectional=False,
        dropout=0.5,
        use_nat=use_nat,
        nat_kernel=7,
        nat_dilation=1,
        nat_blocks=1,
        backbone_name="resnet18",
    ).to(device)

    try:
        model.load_state_dict(state_dict)
    except Exception as e:
        print("ERROR loading state_dict from checkpoint.")
        print("Make sure use_nat / backbone settings / feat dims match training.")
        raise e

    print(f"Loaded weights from {MODEL_PATH}")
    print(f"  (seq_len={ckpt.get('seq_len')}, C={ckpt.get('C')}, "
          f"H={ckpt.get('H')}, W={ckpt.get('W')}, feat_dim={feat_dim_extra})")

    if freeze_backbone:
        for _, p in model.backbone.named_parameters():
            p.requires_grad = False

    model.eval()
    return model


# ------------------------------------------------------------------
# Evaluation
# ------------------------------------------------------------------
def evaluate(
    split: str = "test",
    batch_size: int = 4,
    threshold: float = 0.5,
    max_files: int | None = None,
    use_nat: bool = False,
    freeze_backbone: bool = False,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Dataset / subset
    ds = WindowVideoFeatDataset(split)
    if max_files is not None:
        ds.files = ds.files[:max_files]
        print(f"Restricting to first {len(ds.files)} files for split='{split}'")

    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    # Model
    model = load_model(device, use_nat=use_nat, freeze_backbone=freeze_backbone)
    criterion = nn.BCEWithLogitsLoss()

    all_labels = []
    all_probs = []
    all_preds = []
    total_loss = 0.0
    total_samples = 0

    start = time.time()

    with torch.no_grad():
        for batch_idx, (xb, fb, yb) in enumerate(loader, start=1):
            xb = xb.to(device)              # [B, T, C, H, W]
            fb = fb.to(device)              # [B, T, F]
            yb = yb.to(device).view(-1)     # [B]

            logits = model(xb, fb)          # [B]
            loss = criterion(logits, yb)

            probs = torch.sigmoid(logits)   # [B]
            preds = (probs >= threshold).long()

            total_loss += float(loss.item()) * xb.size(0)
            total_samples += xb.size(0)

            all_labels.extend(yb.cpu().numpy().tolist())
            all_probs.extend(probs.cpu().numpy().tolist())
            all_preds.extend(preds.cpu().numpy().tolist())

            if batch_idx % 50 == 0 or batch_idx == len(loader):
                print(f"[eval] batch {batch_idx}/{len(loader)}  loss={loss.item():.4f}")

    total_time = time.time() - start

    # Convert to numpy arrays for metrics
    labels_np = np.array(all_labels, dtype=np.int64)
    preds_np = np.array(all_preds, dtype=np.int64)

    acc = float((labels_np == preds_np).mean())

    # Confusion matrix (1 = impaired / positive)
    tp = int(((preds_np == 1) & (labels_np == 1)).sum())
    tn = int(((preds_np == 0) & (labels_np == 0)).sum())
    fp = int(((preds_np == 1) & (labels_np == 0)).sum())
    fn = int(((preds_np == 0) & (labels_np == 1)).sum())

    metrics = _compute_class_metrics(tp, tn, fp, fn)
    avg_loss = total_loss / max(1, total_samples)

    print("\n================ EVAL SUMMARY ================")
    print(f"Split:          {split}")
    print(f"Num samples:    {total_samples}")
    print(f"Batch size:     {batch_size}")
    print(f"Threshold:      {threshold:.2f}")
    print(f"Avg loss:       {avg_loss:.4f}")
    print(f"Accuracy:       {acc * 100:.2f}%")
    print()
    print("Confusion matrix (positive = impaired = 1):")
    print(f"  TP (1->1): {tp}")
    print(f"  TN (0->0): {tn}")
    print(f"  FP (0->1): {fp}")
    print(f"  FN (1->0): {fn}")
    print()
    print("Per-class metrics:")
    print(f"  Class 0 (normal):   "
          f"P={metrics['prec0']*100:.2f}%, "
          f"R={metrics['rec0']*100:.2f}%, "
          f"F1={metrics['f1_0']*100:.2f}%")
    print(f"  Class 1 (impaired): "
          f"P={metrics['prec1']*100:.2f}%, "
          f"R={metrics['rec1']*100:.2f}%, "
          f"F1={metrics['f1_1']*100:.2f}%")
    print(f"  Macro F1:            {metrics['macro_f1']*100:.2f}%")
    print()
    print(f"Total eval time: {total_time:.1f}s ({total_time/60:.2f} min)")
    print("==============================================")


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Evaluate VideoFeatResNet18LSTM binary model on windows_from_index."
    )
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Probability threshold for impaired=1")
    parser.add_argument("--max-files", type=int, default=None,
                        help="Optional cap on number of NPZ files")
    parser.add_argument("--use-nat", action="store_true",
                        help="Must match training; default False")
    parser.add_argument("--freeze-backbone", action="store_true",
                        help="Freeze ResNet backbone (for debugging)")

    args = parser.parse_args()

    evaluate(
        split=args.split,
        batch_size=args.batch_size,
        threshold=args.threshold,
        max_files=args.max_files,
        use_nat=args.use_nat,
        freeze_backbone=args.freeze_backbone,
    )
