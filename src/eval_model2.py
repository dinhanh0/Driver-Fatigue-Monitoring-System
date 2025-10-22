import argparse, torch
from torch.utils.data import DataLoader
from src.datasets.windows_dataset import WindowsNPZDataset
from src.models.model2_hmgru_tna import HMGRU_TNA

def get_device():
    if torch.cuda.is_available(): return torch.device("cuda")
    if torch.backends.mps.is_available(): return torch.device("mps")
    return torch.device("cpu")

def accuracy(logits, y):
    return (logits.argmax(1) == y).float().mean().item()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=str, default="data/processed/windows")
    ap.add_argument("--split", type=str, default="test")
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--workers", type=int, default=2)
    args = ap.parse_args()

    import os, glob
    if not os.path.isfile(args.ckpt):
        print(f"\n[eval] Checkpoint not found: {args.ckpt}")
        candidates = sorted(glob.glob("checkpoints/*.pt"))
        more = sorted(glob.glob("**/*.pt", recursive=True))
        seen = set(candidates)
        for p in more:
            if p not in seen:
                candidates.append(p)
                seen.add(p)
        if candidates:
            print("Available .pt files I can see:")
            for p in candidates[:15]:
                print("  -", p)
            if len(candidates) > 15:
                print(f"  ... and {len(candidates)-15} more")
        raise FileNotFoundError(args.ckpt)

    ds = WindowsNPZDataset(args.data_root, args.split)
    ld = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

    device = get_device()
    model = HMGRU_TNA(num_classes=2, feat_dim=4).to(device)
    state = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(state["model"])
    model.eval()

    tot, correct = 0, 0
    with torch.no_grad():
        for batch in ld:
            Xv = batch["video"].to(device)
            Xf = batch["feats"].to(device)
            y  = batch["label"].to(device)
            logits = model(Xv, Xf)
            correct += (logits.argmax(1) == y).sum().item()
            tot += y.numel()
    print(f"Accuracy on {args.split}: {correct/tot:.3f} ({correct}/{tot})")

if __name__ == "__main__":
    main()