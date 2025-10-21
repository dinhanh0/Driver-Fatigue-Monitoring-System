import argparse, os
import torch
from torch.utils.data import DataLoader
from data.dataset import ClipDataset
from .models.resnet18_lstm import ResNet18LSTM
from .utils.metrics import compute_metrics, softmax

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--splits_csv', type=str, required=True)
    ap.add_argument('--subset', type=str, default='test')
    ap.add_argument('--ckpt', type=str, required=True)
    ap.add_argument('--img_size', type=int, default=160)
    ap.add_argument('--seq_len', type=int, default=32)
    ap.add_argument('--batch_size', type=int, default=4)
    ap.add_argument('--num_workers', type=int, default=2)
    ap.add_argument('--num_classes', type=int, default=3)
    # NAT flags (match training if NAT was used)
    ap.add_argument('--use_nat', action='store_true')
    ap.add_argument('--nat_kernel', type=int, default=7)
    ap.add_argument('--nat_dilation', type=int, default=1)
    ap.add_argument('--nat_blocks', type=int, default=1)
    return ap.parse_args()

def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ds = ClipDataset(args.splits_csv, args.subset, img_size=args.img_size, seq_len=args.seq_len)
    pin_mem = torch.cuda.is_available()
    ld = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=pin_mem)

    model = ResNet18LSTM(num_classes=args.num_classes, use_nat=args.use_nat,
                         nat_kernel=args.nat_kernel, nat_dilation=args.nat_dilation, nat_blocks=args.nat_blocks).to(device)
    ckpt = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ckpt['model'])
    model.eval()

    logits_all, labels_all = [], []
    with torch.no_grad():
        for batch in ld:
            clip = batch['clip'].to(device)
            label = batch['label'].to(device)
            logits = model(clip)
            logits_all.append(logits.cpu())
            labels_all.append(label.cpu())

    import torch as th
    logits = th.cat(logits_all, dim=0)
    labels = th.cat(labels_all, dim=0)
    probs = softmax(logits, dim=1)
    metrics = compute_metrics(labels.numpy(), probs.numpy())
    print("Eval results:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")

if __name__ == '__main__':
    main()
