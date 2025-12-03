# src/train.py
# ResNet18 + LSTM, binary classification on video windows
# Uses both X_video (frames) and X_feats (EAR, MAR, blur, illum)
# - Class-weighted BCEWithLogitsLoss
# - LR scheduler on val loss
# - Saves best model by val loss AND best model by val accuracy

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

from models.resnet18_lstm import ResNet18LSTM  # your backbone model

# Match preprocess.py output dir
WINDOWS_ROOT = PROJECT_ROOT / "data" / "processed" / "windows_from_index"
MODEL_DIR = PROJECT_ROOT / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

BEST_LOSS_PATH = MODEL_DIR / "resnet18_lstm_binary_best_loss.pt"
BEST_ACC_PATH  = MODEL_DIR / "resnet18_lstm_binary_best_acc.pt"

# ------------------------------------------------------------------
# NOTE: no fixed random seeds here on purpose (you wanted varied runs)
# ------------------------------------------------------------------

# ------------------------------------------------------------------
# ImageNet normalization (for pretrained ResNet18)
# ------------------------------------------------------------------
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1)
IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1)

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
# Dataset using both X_video (frames) and X_feats (per-frame features)
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

        # Save for later (for logging / class weighting)
        self.bin_counts = bin_counts

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
        x = x.astype(np.float32)  # [T, C, H, W], values in [0,1] from preprocess
        f = f.astype(np.float32)  # [T, F]
        x = torch.from_numpy(x)   # Tensor [T, C, H, W]
        f = torch.from_numpy(f)   # Tensor [T, F]

        # ImageNet channel-wise normalization (broadcasted over T)
        x = (x - IMAGENET_MEAN) / IMAGENET_STD

        y = torch.tensor([y_bin], dtype=torch.float32)  # [1]

        return x, f, y


# ------------------------------------------------------------------
# Model: video encoder (ResNet18LSTM) + aggregated handcrafted feats
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

        # We instantiate ResNet18LSTM to output a vector of size backbone_feat_dim
        # (we treat that as an embedding, not final logits).
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
# Metrics helper
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
# Training function
# ------------------------------------------------------------------
def train_binary_resnet18_lstm(
    batch_size: int = 4,       # tune based on GPU/CPU memory
    num_epochs: int = 10,
    lr: float = 1e-4,
    use_nat: bool = False,     # NAT off by default
    freeze_backbone: bool = False,
    max_train_files: int | None = None,
    max_val_files: int | None = None,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ----------------- DATASETS -----------------
    train_ds = WindowVideoFeatDataset("train")
    if max_train_files is not None:
        train_ds.files = train_ds.files[:max_train_files]

    val_ds = WindowVideoFeatDataset("val")
    if max_val_files is not None:
        val_ds.files = val_ds.files[:max_val_files]

    # ----------------- CLASS WEIGHTS -----------------
    train_bin_counts = train_ds.bin_counts
    n0 = train_bin_counts.get(0, 0)
    n1 = train_bin_counts.get(1, 0)
    print(f"[TRAIN] Binary label counts (0=normal,1=impaired): {dict(train_bin_counts)}")
    if n0 > 0 and n1 > 0:
        pos_weight_value = n0 / n1  # < 1, down-weight impaired class a bit
    else:
        pos_weight_value = 1.0
    print(f"[CLASS WEIGHTS] n0={n0}, n1={n1}, pos_weight={pos_weight_value:.4f}\n")

    # ----------------- DATALOADERS -----------------
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    # ----------------- MODEL -----------------
    model = VideoFeatResNet18LSTM(
        feat_dim_extra=train_ds.feat_dim,
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
        for _, param in model.backbone.named_parameters():
            param.requires_grad = False

    # Class-weighted BCE
    pos_weight = torch.tensor([pos_weight_value], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
    )

    # Reduce LR if val loss plateaus to avoid late-epoch swings
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=1,
        verbose=True,
    )

    print(f"\nUsing device: {device}")
    print(f"Best-loss model will be saved to: {BEST_LOSS_PATH}")
    print(f"Best-acc  model will be saved to: {BEST_ACC_PATH}")
    print(f"Backbone frozen: {freeze_backbone}, NAT enabled: {use_nat}")
    print(f"Train files: {len(train_ds.files)}, Val files: {len(val_ds.files)}\n")

    best_val_loss = float("inf")
    best_val_acc = 0.0
    DECISION_THRESHOLD = 0.5  # can be tuned later

    for epoch in range(1, num_epochs + 1):
        epoch_start = time.time()

        # ----------------- TRAIN -----------------
        model.train()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        train_tp = train_tn = train_fp = train_fn = 0

        print(f"Epoch {epoch}/{num_epochs} - TRAIN")
        for batch_idx, (xb, fb, yb) in enumerate(train_loader, start=1):
            xb = xb.to(device)                 # [B, T, C, H, W]
            fb = fb.to(device)                 # [B, T, F]
            yb = yb.to(device).view(-1)        # [B], float 0/1

            logits = model(xb, fb)             # [B]
            loss = criterion(logits, yb)

            optimizer.zero_grad()
            loss.backward()
            # Gradient clipping to stabilize training
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            total_loss += float(loss.item()) * xb.size(0)

            probs = torch.sigmoid(logits)
            preds = (probs >= DECISION_THRESHOLD).long()  # [B]
            y_int = yb.long()

            total_correct += int((preds == y_int).sum().item())
            total_samples += xb.size(0)

            train_tp += int(((preds == 1) & (y_int == 1)).sum().item())
            train_tn += int(((preds == 0) & (y_int == 0)).sum().item())
            train_fp += int(((preds == 1) & (y_int == 0)).sum().item())
            train_fn += int(((preds == 0) & (y_int == 1)).sum().item())

            if batch_idx % 100 == 0 or batch_idx == len(train_loader):
                print(
                    f"  [train] batch {batch_idx}/{len(train_loader)} "
                    f"loss={loss.item():.4f}"
                )

        train_loss = total_loss / total_samples
        train_acc = total_correct / total_samples * 100.0
        train_metrics = _compute_class_metrics(train_tp, train_tn, train_fp, train_fn)

        # ----------------- VAL -----------------
        model.eval()
        val_loss_sum = 0.0
        val_correct = 0
        val_samples = 0

        val_tp = val_tn = val_fp = val_fn = 0

        print(f"Epoch {epoch}/{num_epochs} - VAL")
        with torch.no_grad():
            for batch_idx, (xb, fb, yb) in enumerate(val_loader, start=1):
                xb = xb.to(device)
                fb = fb.to(device)
                yb = yb.to(device).view(-1)

                logits = model(xb, fb)
                loss = criterion(logits, yb)

                val_loss_sum += float(loss.item()) * xb.size(0)

                probs = torch.sigmoid(logits)
                preds = (probs >= DECISION_THRESHOLD).long()
                y_int = yb.long()

                val_correct += int((preds == y_int).sum().item())
                val_samples += xb.size(0)

                val_tp += int(((preds == 1) & (y_int == 1)).sum().item())
                val_tn += int(((preds == 0) & (y_int == 0)).sum().item())
                val_fp += int(((preds == 1) & (y_int == 0)).sum().item())
                val_fn += int(((preds == 0) & (y_int == 1)).sum().item())

                if batch_idx % 100 == 0 or batch_idx == len(val_loader):
                    print(
                        f"  [val]   batch {batch_idx}/{len(val_loader)} "
                        f"loss={loss.item():.4f}"
                    )

        val_loss = val_loss_sum / val_samples
        val_acc = val_correct / val_samples * 100.0
        val_metrics = _compute_class_metrics(val_tp, val_tn, val_fp, val_fn)

        # Step LR scheduler based on val loss
        scheduler.step(val_loss)

        epoch_time = time.time() - epoch_start

        print(
            f"Epoch {epoch}/{num_epochs} SUMMARY\n"
            f"  Train loss: {train_loss:.4f}  |  Train acc: {train_acc:.2f}%\n"
            f"  Val   loss: {val_loss:.4f}  |  Val   acc: {val_acc:.2f}%\n"
            f"  Epoch time: {epoch_time:.1f} s ({epoch_time/60:.2f} min)\n"
            f"  Train metrics:\n"
            f"    Class 0 (normal):   P={train_metrics['prec0']:.3f}, "
            f"R={train_metrics['rec0']:.3f}, F1={train_metrics['f1_0']:.3f}\n"
            f"    Class 1 (impaired): P={train_metrics['prec1']:.3f}, "
            f"R={train_metrics['rec1']:.3f}, F1={train_metrics['f1_1']:.3f}\n"
            f"    Macro F1: {train_metrics['macro_f1']:.3f}\n"
            f"  Val metrics:\n"
            f"    Class 0 (normal):   P={val_metrics['prec0']:.3f}, "
            f"R={val_metrics['rec0']:.3f}, F1={val_metrics['f1_0']:.3f}\n"
            f"    Class 1 (impaired): P={val_metrics['prec1']:.3f}, "
            f"R={val_metrics['rec1']:.3f}, F1={val_metrics['f1_1']:.3f}\n"
            f"    Macro F1: {val_metrics['macro_f1']:.3f}"
        )

        # ----- Save best by VAL LOSS -----
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "seq_len": train_ds.seq_len,
                    "C": train_ds.C,
                    "H": train_ds.H,
                    "W": train_ds.W,
                    "feat_dim": train_ds.feat_dim,
                    "best_val_loss": best_val_loss,
                    "best_val_acc": best_val_acc,
                },
                BEST_LOSS_PATH,
            )
            print(f"\n  New best-LOSS model saved to: {BEST_LOSS_PATH}\n")
        else:
            print("")

        # ----- Save best by VAL ACCURACY -----
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "seq_len": train_ds.seq_len,
                    "C": train_ds.C,
                    "H": train_ds.H,
                    "W": train_ds.W,
                    "feat_dim": train_ds.feat_dim,
                    "best_val_loss": best_val_loss,
                    "best_val_acc": best_val_acc,
                },
                BEST_ACC_PATH,
            )
            print(f"  New best-ACC  model saved to: {BEST_ACC_PATH}\n")

    print("Training complete.")


# ------------------------------------------------------------------
# Entry point with CLI args + total runtime
# ------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=10)
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
