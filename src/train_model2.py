import os, math, time, argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.datasets.windows_dataset import WindowsNPZDataset
from src.models.model2_hmgru_tna import HMGRU_TNA

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def accuracy(logits, y):
    preds = logits.argmax(dim=1)
    return (preds == y).float().mean().item()

def build_loaders(data_root, train_split, val_split, bs, workers, limit=None):
    train_ds = WindowsNPZDataset(data_root, train_split, limit=limit)
    val_ds   = WindowsNPZDataset(data_root, val_split,   limit=limit)
    train_ld = DataLoader(train_ds, batch_size=bs, shuffle=True,  num_workers=workers, pin_memory=False)
    val_ld   = DataLoader(val_ds,   batch_size=bs, shuffle=False, num_workers=workers, pin_memory=False)
    return train_ld, val_ld

def freeze(module, requires_grad: bool):
    for p in module.parameters():
        p.requires_grad = requires_grad

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=str, default="data/processed/windows")
    ap.add_argument("--train-split", type=str, default="train")
    ap.add_argument("--val-split", type=str, default="val")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--limit", type=int, default=None, help="debug: limit samples per split")
    ap.add_argument("--num-classes", type=int, default=2)
    ap.add_argument("--freeze-backbone-epochs", type=int, default=1, help="initial epochs to freeze CNN")
    args = ap.parse_args()

    device = get_device()
    print(f"Using device: {device}")

    train_ld, val_ld = build_loaders(args.data_root, args.train_split, args.val_split,
                                     args.batch_size, args.workers, args.limit)

    model = HMGRU_TNA(num_classes=args.num_classes, feat_dim=4)  # 4 = EAR/MAR/BLUR/ILLUM
    model.to(device)

    # Optionally freeze CNN backbone at start
    freeze(model.backbone, True)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                                  lr=args.lr, weight_decay=args.weight-decay if hasattr(args,'weight-decay') else args.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)

    best_val = -1.0
    os.makedirs("checkpoints", exist_ok=True)

    for epoch in range(1, args.epochs+1):
        t0 = time.time()
        # Unfreeze after N epochs
        if epoch == args.freeze_backbone_epochs + 1:
            freeze(model.backbone, False)
            # re-add unfrozen params to optimizer
            optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)
            print("Unfroze backbone.")

        # -------- train --------
        model.train()
        tr_loss, tr_acc, n = 0.0, 0.0, 0
        for batch in train_ld:
            Xv = batch["video"].to(device)   # [B,T,3,224,224]
            Xf = batch["feats"].to(device)   # [B,T,D]
            y  = batch["label"].to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(Xv, Xf)
            loss = criterion(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            bs = y.size(0)
            tr_loss += loss.item() * bs
            tr_acc  += accuracy(logits.detach(), y) * bs
            n       += bs

        tr_loss /= n
        tr_acc  /= n

        # -------- validate --------
        model.eval()
        va_loss, va_acc, m = 0.0, 0.0, 0
        with torch.no_grad():
            for batch in val_ld:
                Xv = batch["video"].to(device)
                Xf = batch["feats"].to(device)
                y  = batch["label"].to(device)

                logits = model(Xv, Xf)
                loss = criterion(logits, y)

                bs = y.size(0)
                va_loss += loss.item() * bs
                va_acc  += accuracy(logits, y) * bs
                m       += bs
        va_loss /= m
        va_acc  /= m

        scheduler.step()
        dt = time.time() - t0
        print(f"[Epoch {epoch:02d}] {dt:.1f}s  train loss {tr_loss:.4f} acc {tr_acc:.3f}  "
              f"val loss {va_loss:.4f} acc {va_acc:.3f}")

        if va_acc > best_val:
            best_val = va_acc
            ckpt = f"checkpoints/model2_hmgru_tna_best.pt"
            torch.save({"model": model.state_dict(),
                        "epoch": epoch,
                        "val_acc": va_acc}, ckpt)
            print(f"  ✔ Saved best checkpoint → {ckpt} (acc={va_acc:.3f})")

if __name__ == "__main__":
    main()