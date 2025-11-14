# src/eval.py

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

# so `models` is importable
sys.path.append(str(PROJECT_ROOT))

from models.resnet18_lstm import ResNet18LSTM  # same backbone as train.py

WINDOWS_ROOT = PROJECT_ROOT / "data" / "processed" / "windows_from_index"
MODEL_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODEL_DIR / "resnet18_lstm_binary.pt"


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
# Dataset (same structure as in train.py)
# ------------------------------------------------------------------
class WindowVideoDataset(Dataset):
    def __init__(self, split: str):
        self.split = split
        split_dir = WINDOWS_ROOT / split
        if not split_dir.exists():
            raise RuntimeError(f"Split folder does not exist: {split_dir}")

        self.files = sorted(split_dir.rglob("*.npz"))
        if not self.files:
            raise RuntimeError(f"No .npz files found in {split_dir}")

        # Probe one file to get sequence / spatial dims
        sample = np.load(self.files[0])
        X_video = sample["X_video"]  # shape [T, C, H, W]
        self.seq_len = X_video.shape[0]
        self.C = X_video.shape[1]
        self.H = X_video.shape[2]
        self.W = X_video.shape[3]

        print(f"[{split.upper()}] {len(self.files)} windows")
        print(f"  Sequence length: {self.seq_len}, CxHxW = {self.C}x{self.H}x{self.W}")

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
        clip = d["X_video"]  # [T, C, H, W]
        y_raw = int(d["y"])
        y_bin = raw_to_binary(y_raw)

        # Ensure consistent T (pad or trim)
        x = clip
        if x.shape[0] > self.seq_len:
            x = x[: self.seq_len]
        elif x.shape[0] < self.seq_len:
            pad_len = self.seq_len - x.shape[0]
            pad = np.zeros((pad_len, self.C, self.H, self.W), dtype=np.float32)
            x = np.concatenate([x, pad], axis=0)

        x = x.astype(np.float32)
        x = torch.from_numpy(x)  # [T, C, H, W]

        y = torch.tensor([y_bin], dtype=torch.float32)  # [1]

        return x, y


# ------------------------------------------------------------------
# Wrapper: ResNet18LSTM -> single logit
# ------------------------------------------------------------------
class BinaryHeadResNet18LSTM(nn.Module):
    """
    Same wrapper as train.py: outputs a single logit for BCEWithLogitsLoss.
    """
    def __init__(self, **backbone_kwargs):
        super().__init__()
        self.model = ResNet18LSTM(num_classes=1, **backbone_kwargs)

    def forward(self, clip):
        # clip: [B, T, C, H, W]
        return self.model(clip).squeeze(-1)  # [B]


# ------------------------------------------------------------------
# Evaluation
# ------------------------------------------------------------------
def load_model(device, use_nat: bool = False, freeze_backbone: bool = False):
    """
    Build the same architecture as train.py and load weights from MODEL_PATH.
    """
    model = BinaryHeadResNet18LSTM(
        feat_dim=512,
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

    if MODEL_PATH.exists():
        ckpt = torch.load(MODEL_PATH, map_location=device)
        try:
            model.load_state_dict(ckpt["model_state"])
        except Exception as e:
            print("ERROR loading state_dict from checkpoint.")
            print("Make sure use_nat / backbone settings match training.")
            raise e
        print(f"Loaded weights from {MODEL_PATH}")
    else:
        raise FileNotFoundError(f"Checkpoint not found at {MODEL_PATH}")

    if freeze_backbone:
        for _, p in model.model.backbone.named_parameters():
            p.requires_grad = False

    model.eval()
    return model


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
    ds = WindowVideoDataset(split)
    if max_files is not None:
        ds.files = ds.files[:max_files]
        print(f"Restricting to first {len(ds.files)} files for split='{split}'")

    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)

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
        for batch_idx, (xb, yb) in enumerate(loader, start=1):
            xb = xb.to(device)        # [B, T, C, H, W]
            yb = yb.to(device).view(-1)  # [B]

            logits = model(xb)        # [B]
            loss = criterion(logits, yb)

            probs = torch.sigmoid(logits)  # [B]
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

    # Safety for division
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

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
    print(f"  TP (1→1): {tp}")
    print(f"  TN (0→0): {tn}")
    print(f"  FP (0→1): {fp}")
    print(f"  FN (1→0): {fn}")
    print()
    print(f"Precision (imp.): {precision * 100:.2f}%")
    print(f"Recall    (imp.): {recall * 100:.2f}%")
    print(f"F1        (imp.): {f1 * 100:.2f}%")
    print()
    print(f"Total eval time: {total_time:.1f}s ({total_time/60:.2f} min)")
    print("==============================================")


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate ResNet18+LSTM binary model on windows_from_index.")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--threshold", type=float, default=0.5, help="Probability threshold for impaired=1")
    parser.add_argument("--max-files", type=int, default=None, help="Optional cap on number of NPZ files")
    parser.add_argument("--use-nat", action="store_true", help="Must match training; default False")
    parser.add_argument("--freeze-backbone", action="store_true", help="Freeze ResNet backbone (for debugging)")

    args = parser.parse_args()

    evaluate(
        split=args.split,
        batch_size=args.batch_size,
        threshold=args.threshold,
        max_files=args.max_files,
        use_nat=args.use_nat,
        freeze_backbone=args.freeze_backbone,
    )
