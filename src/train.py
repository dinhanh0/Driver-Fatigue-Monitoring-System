import argparse, os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from data.dataset import ClipDataset
from .models.resnet18_lstm import ResNet18LSTM
from .utils.metrics import MetricTracker

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--splits_csv', type=str, required=True)
    ap.add_argument('--subset_train', type=str, default='train')
    ap.add_argument('--subset_val', type=str, default='val')
    ap.add_argument('--img_size', type=int, default=160)
    ap.add_argument('--seq_len', type=int, default=32)
    ap.add_argument('--batch_size', type=int, default=4)
    ap.add_argument('--epochs', type=int, default=5)
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--num_classes', type=int, default=3)
    ap.add_argument('--out_dir', type=str, default='outputs')
    ap.add_argument('--num_workers', type=int, default=2)
    ap.add_argument('--use_amp', action='store_true', help='Enable automatic mixed precision if CUDA is available')
    ap.add_argument('--backbone', type=str, default='resnet18', help='Backbone: resnet18 or mobilenet_v2')
    ap.add_argument('--windows_root', type=str, default=None, help='Root directory for preprocessed windows (auto-detected if not specified)')
    # NAT flags
    ap.add_argument('--use_nat', action='store_true')
    ap.add_argument('--nat_kernel', type=int, default=7)
    ap.add_argument('--nat_dilation', type=int, default=1)
    ap.add_argument('--nat_blocks', type=int, default=1)
    # Training improvements (defaults enabled)
    ap.add_argument('--patience', type=int, default=15, help='Early stopping patience')
    return ap.parse_args()

def main():
    print("[TRAIN] Starting training initialization...")
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[TRAIN] Using device: {device}")
    
    # Enable cuDNN autotuner for potential speedups on fixed-size inputs
    try:
        if device.type == 'cuda':
            torch.backends.cudnn.benchmark = True
            print("[TRAIN] Enabled cuDNN benchmark mode")
    except Exception:
        pass

    print("[TRAIN] Loading datasets...")
    import pandas as pd
    from pathlib import Path
    splits_df = pd.read_csv(args.splits_csv)
    print("\nDataset distribution:")
    try:
        dataset_counts = splits_df['relpath'].apply(lambda x: x.split('\\')[2] if '\\' in x else x.split('/')[2]).value_counts()
        print(dataset_counts)
    except Exception:
        print("Could not parse dataset distribution")
    print("\nLabel distribution:")
    if 'label' in splits_df.columns:
        print(splits_df['label'].value_counts())
    print("\n")
    
    # Auto-detect windows directory if not specified
    if args.windows_root is None:
        base_dir = Path(args.splits_csv).parent
        # Try to find windows directory (prefer windows_bench_mp)
        for candidate in ['windows_bench_mp', 'windows_bench_serial', 'windows']:
            test_path = base_dir / candidate
            if test_path.exists() and (test_path / args.subset_train).exists():
                args.windows_root = str(test_path)
                print(f"[TRAIN] Auto-detected windows directory: {args.windows_root}")
                break
        if args.windows_root is None:
            args.windows_root = str(base_dir / 'windows')
            print(f"[TRAIN] Using default windows directory: {args.windows_root}")
    
    train_ds = ClipDataset(args.splits_csv, args.subset_train, windows_root=args.windows_root, img_size=args.img_size, seq_len=args.seq_len, use_augmentation=True)
    print(f"[TRAIN] Loaded training dataset with {len(train_ds)} samples (augmentation enabled)")
    val_ds   = ClipDataset(args.splits_csv, args.subset_val, windows_root=args.windows_root, img_size=args.img_size, seq_len=args.seq_len, use_augmentation=False)
    print(f"[TRAIN] Loaded validation dataset with {len(val_ds)} samples")

    print("[TRAIN] Creating data loaders...")
    pin_mem = torch.cuda.is_available()
    train_ld = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=pin_mem)
    val_ld   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=pin_mem)
    print(f"[TRAIN] Created data loaders with {args.num_workers} workers, pin_memory={pin_mem}")

    print("[TRAIN] Initializing model...")
    model = ResNet18LSTM(num_classes=args.num_classes, use_nat=args.use_nat,
                         nat_kernel=args.nat_kernel, nat_dilation=args.nat_dilation, nat_blocks=args.nat_blocks,
                         backbone_name=args.backbone, feat_dim=512).to(device)
    print(f"[TRAIN] Using {args.backbone} backbone")
    
    print("[TRAIN] Setting up optimizer and loss...")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    
    # Calculate class weights for imbalanced data
    labels = [train_ds[i]['label'].item() for i in range(len(train_ds))]
    counts = torch.bincount(torch.tensor(labels), minlength=args.num_classes).float()
    weights = 1.0 / (counts + 1e-6)
    weights = weights / weights.sum() * args.num_classes
    crit = nn.CrossEntropyLoss(weight=weights.to(device))
    print(f"[TRAIN] Using class weights: {weights.cpu().numpy()}")
    print(f"[TRAIN] Using AdamW optimizer with lr={args.lr}")
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=3)

    best_f1 = -1.0
    patience_counter = 0
    tracker = MetricTracker()

    use_amp = args.use_amp and device.type == 'cuda'
    # Use the top-level amp API (torch.amp.GradScaler) to avoid deprecation warnings
    scaler = torch.amp.GradScaler(enabled=use_amp)

    print("[TRAIN] Starting training loop...")
    for epoch in range(1, args.epochs+1):
        model.train()
        print(f"\n[TRAIN] Epoch {epoch}/{args.epochs}")
        print("[TRAIN] Training phase...")
        for i, batch in enumerate(train_ld):
            if i % 10 == 0:  # Print every 10 batches
                print(f"[TRAIN] Processing batch {i+1}/{len(train_ld)}")
            clip = batch['clip'].to(device)  # (B,T,C,H,W)
            label = batch['label'].to(device)
            opt.zero_grad()
            if use_amp:
                with torch.cuda.amp.autocast():
                    logits = model(clip)
                    loss = crit(logits, label)
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
            else:
                logits = model(clip)
                loss = crit(logits, label)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            if i % 10 == 0:  # Print loss every 10 batches
                print(f"[TRAIN] Current loss: {loss.item():.4f}")
            tracker.update_train(loss.item(), logits.detach().cpu(), label.detach().cpu())

        model.eval()
        with torch.no_grad():
            for batch in val_ld:
                clip = batch['clip'].to(device)
                label = batch['label'].to(device)
                logits = model(clip)
                loss = crit(logits, label)
                tracker.update_val(loss.item(), logits.detach().cpu(), label.detach().cpu())

        tr = tracker.summarize(reset=True)
        print(f"Epoch {epoch:02d} | Train loss {tr['train_loss']:.4f} acc {tr['train_acc']:.3f} f1 {tr['train_f1']:.3f} | "
              f"Val loss {tr['val_loss']:.4f} acc {tr['val_acc']:.3f} f1 {tr['val_f1']:.3f}")

        # Learning rate scheduling
        scheduler.step(tr['val_loss'])
        print(f"Current LR: {opt.param_groups[0]['lr']:.6f}")

        if tr['val_f1'] > best_f1:
            best_f1 = tr['val_f1']
            patience_counter = 0
            ckpt_path = os.path.join(args.out_dir, 'best.pt')
            torch.save({'model': model.state_dict(), 'args': vars(args), 'epoch': epoch}, ckpt_path)
            print(f"Saved best checkpoint → {ckpt_path} (val_f1={best_f1:.3f})")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"Early stopping triggered at epoch {epoch}")
                break
    
    print(f"\nTraining complete. Best val F1: {best_f1:.3f}")

if __name__ == '__main__':
    main()
