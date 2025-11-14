# src/train_resnet18_lstm.py

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

# Put project root on sys.path so "models" is importable
sys.path.append(str(PROJECT_ROOT))

from models.resnet18_lstm import ResNet18LSTM  # <-- your backbone model

# Match preprocess.py output dir
WINDOWS_ROOT = PROJECT_ROOT / "data" / "processed" / "windows_from_index"
MODEL_DIR = PROJECT_ROOT / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
MODEL_PATH = MODEL_DIR / "resnet18_lstm_binary.pt"


# ------------------------------------------------------------------
# Binary label mapping
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
# Dataset using X_video (frames)
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

        # Raw label histogram
        raw_counts = Counter()
        for p in self.files:
            d = np.load(p)
            y_raw = int(d["y"])
            raw_counts[y_raw] += 1
        print(f"  Raw label counts (y values): {dict(raw_counts)}")

        # Binary histogram
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

        x = x.astype(np.float32)        # [T, C, H, W]
        x = torch.from_numpy(x)         # Tensor [T, C, H, W]

        y = torch.tensor([y_bin], dtype=torch.float32)  # [1]

        return x, y


# ------------------------------------------------------------------
# Wrapper: ResNet18LSTM -> single logit for BCEWithLogitsLoss
# ------------------------------------------------------------------
class BinaryHeadResNet18LSTM(nn.Module):
    """
    Wrap ResNet18LSTM to output a single logit for BCEWithLogitsLoss.
    """
    def __init__(self, **backbone_kwargs):
        super().__init__()
        # num_classes=1 so the head outputs a single logit
        self.model = ResNet18LSTM(num_classes=1, **backbone_kwargs)

    def forward(self, clip):
        # clip: [B, T, C, H, W]
        return self.model(clip).squeeze(-1)  # [B]


def train_binary_resnet18_lstm(
    batch_size: int = 4,       # tune based on GPU/CPU memory
    num_epochs: int = 3,
    lr: float = 1e-4,
    use_nat: bool = False,     # NAT off by default
    freeze_backbone: bool = False,
    max_train_files: int | None = None,
    max_val_files: int | None = None,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Datasets
    train_ds = WindowVideoDataset("train")
    if max_train_files is not None:
        train_ds.files = train_ds.files[:max_train_files]

    val_ds = WindowVideoDataset("val")
    if max_val_files is not None:
        val_ds.files = val_ds.files[:max_val_files]

    # Loaders
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    # Model: ResNet18 backbone + LSTM + (optional) NAT
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

    # Optional freezing
    if freeze_backbone:
        for _, param in model.model.backbone.named_parameters():
            param.requires_grad = False

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
    )

    print(f"\nUsing device: {device}")
    print(f"Saving best model to: {MODEL_PATH}")
    print(f"Backbone frozen: {freeze_backbone}, NAT enabled: {use_nat}")
    print(f"Train files: {len(train_ds.files)}, Val files: {len(val_ds.files)}\n")

    best_val_loss = float("inf")

    for epoch in range(1, num_epochs + 1):
        epoch_start = time.time()

        # ----------------- TRAIN -----------------
        model.train()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        print(f"Epoch {epoch}/{num_epochs} - TRAIN")
        for batch_idx, (xb, yb) in enumerate(train_loader, start=1):
            xb = xb.to(device)                 # [B, T, C, H, W]
            yb = yb.to(device).view(-1)        # [B]

            logits = model(xb)                 # [B]
            loss = criterion(logits, yb)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += float(loss.item()) * xb.size(0)
            preds = (torch.sigmoid(logits) >= 0.5).long()
            total_correct += int((preds == yb.long()).sum().item())
            total_samples += xb.size(0)

            if batch_idx % 100 == 0 or batch_idx == len(train_loader):
                print(
                    f"  [train] batch {batch_idx}/{len(train_loader)} "
                    f"loss={loss.item():.4f}"
                )

        train_loss = total_loss / total_samples
        train_acc = total_correct / total_samples * 100.0

        # ----------------- VAL -----------------
        model.eval()
        val_loss_sum = 0.0
        val_correct = 0
        val_samples = 0

        print(f"Epoch {epoch}/{num_epochs} - VAL")
        with torch.no_grad():
            for batch_idx, (xb, yb) in enumerate(val_loader, start=1):
                xb = xb.to(device)
                yb = yb.to(device).view(-1)

                logits = model(xb)
                loss = criterion(logits, yb)

                val_loss_sum += float(loss.item()) * xb.size(0)
                preds = (torch.sigmoid(logits) >= 0.5).long()
                val_correct += int((preds == yb.long()).sum().item())
                val_samples += xb.size(0)

                if batch_idx % 100 == 0 or batch_idx == len(val_loader):
                    print(
                        f"  [val]   batch {batch_idx}/{len(val_loader)} "
                        f"loss={loss.item():.4f}"
                    )

        val_loss = val_loss_sum / val_samples
        val_acc = val_correct / val_samples * 100.0
        epoch_time = time.time() - epoch_start

        print(
            f"Epoch {epoch}/{num_epochs} SUMMARY\n"
            f"  Train loss: {train_loss:.4f}  |  Train acc: {train_acc:.2f}%\n"
            f"  Val   loss: {val_loss:.4f}  |  Val   acc: {val_acc:.2f}%\n"
            f"  Epoch time: {epoch_time:.1f} s ({epoch_time/60:.2f} min)"
        )

        # Save best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "seq_len": train_ds.seq_len,
                    "C": train_ds.C,
                    "H": train_ds.H,
                    "W": train_ds.W,
                },
                MODEL_PATH,
            )
            print(f"  New best model saved to: {MODEL_PATH}\n")
        else:
            print()

    print("Training complete.")


# ------------------------------------------------------------------
# Entry point with CLI args + total runtime
# ------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--use-nat", action="store_true")
    parser.add_argument("--freeze-backbone", action="store_true")
    parser.add_argument("--max-train", type=int, default=None)
    parser.add_argument("--max-val", type=int, default=None)

    args = parser.parse_args()

    total_start = time.time()
    train_binary_resnet18_lstm(
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        lr=args.lr,
        use_nat=args.use_nat,
        freeze_backbone=args.freeze_backbone,
        max_train_files=args.max_train,
        max_val_files=args.max_val,
    )
    total_end = time.time()
    total = total_end - total_start

    mins = int(total // 60)
    secs = total - mins * 60
    print(f"\nTotal runtime: {mins}m {secs:.1f}s ({total:.1f} s)")
