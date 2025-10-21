import time
import argparse
import torch
from torch.utils.data import DataLoader
from data.dataset import ClipDataset
from .models.resnet18_lstm import ResNet18LSTM

def measure_once(splits_csv, subset, seq_len, batch_size, num_workers, cache_size, backbone):
    ds = ClipDataset(splits_csv, subset, seq_len=seq_len, cache_size=cache_size)
    if cache_size > 0:
        warmed = ds.prefetch(min(200, len(ds)))
        print(f"Prefetched {warmed} entries into cache (size={cache_size})")

    ld = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ResNet18LSTM(num_classes=3, backbone_name=backbone).to(device)
    model.eval()

    batch = next(iter(ld))
    clip = batch['clip']  # (B,T,C,H,W)
    clip = clip.to(device)

    # measure backbone+pool
    t0 = time.time()
    with torch.no_grad():
        out = model._encode_batch_frames(clip)
    t1 = time.time()

    # measure LSTM
    t2 = time.time()
    with torch.no_grad():
        out2, _ = model.lstm(out)
    t3 = time.time()

    print(f"Data batch shape: {clip.shape}")
    print(f"Backbone+pool time: {t1-t0:.4f}s")
    print(f"LSTM time: {t3-t2:.4f}s")
    return (t1-t0, t3-t2)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--splits_csv', required=True)
    ap.add_argument('--subset', default='train')
    ap.add_argument('--seq_len', type=int, default=8)
    ap.add_argument('--batch_size', type=int, default=2)
    ap.add_argument('--num_workers', type=int, default=0)
    ap.add_argument('--cache_size', type=int, default=0)
    ap.add_argument('--backbone', type=str, default='resnet18')
    args = ap.parse_args()

    measure_once(args.splits_csv, args.subset, args.seq_len, args.batch_size, args.num_workers, args.cache_size, args.backbone)

if __name__ == '__main__':
    main()
