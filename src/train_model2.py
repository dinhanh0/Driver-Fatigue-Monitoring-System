import os, math, time, argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import WeightedRandomSampler

import sys
try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

import numpy as np
import random
import json

from src.datasets.windows_dataset import WindowsNPZDataset
from src.models.model2_hmgru_tna import HMGRU_TNA

# --- Collate utility to handle variable-length windows and non-contiguous tensors
def _to_tensor_contig(x):
    import torch, numpy as _np
    if isinstance(x, torch.Tensor):
        return x.contiguous()
    if isinstance(x, _np.ndarray):
        return torch.as_tensor(x).contiguous()
    return torch.tensor(x).contiguous()

# --- Picklable collate: class-based to support multiprocessing workers on macOS (spawn)
class CollatePadTrim:
    def __init__(self, target_T: int | None = None):
        self.target_T = target_T

    @staticmethod
    def _pad_trim_time(t: torch.Tensor, T: int) -> torch.Tensor:
        # t: [T, ...]
        cur = t.shape[0]
        if cur == T:
            return t.contiguous()
        if cur > T:
            return t[:T].contiguous()
        pad_shape = (T - cur, *t.shape[1:])
        pad = t.new_zeros(pad_shape)
        return torch.cat([t, pad], dim=0).contiguous()

    def __call__(self, batch):
        # batch: list of dicts with keys: video [T,3,H,W], feats [T,D], label int
        vids  = [_to_tensor_contig(b["video"]) for b in batch]
        feats = [_to_tensor_contig(b["feats"]) for b in batch]
        labels = torch.as_tensor([int(b["label"]) for b in batch], dtype=torch.long)

        # Decide temporal length for this batch
        T = self.target_T
        if T is None:
            Ts = [v.shape[0] for v in vids]
            T = int(np.median(Ts)) if len(Ts) else 1

        vids  = [self._pad_trim_time(v, T) for v in vids]
        feats = [self._pad_trim_time(f, T) for f in feats]

        xV = torch.stack(vids,  dim=0)  # [B,T,3,H,W]
        xF = torch.stack(feats, dim=0)  # [B,T,D]
        return {"video": xV, "feats": xF, "label": labels}

def make_collate_fn(target_T: int | None = None):
    # Return a picklable callable for DataLoader workers
    return CollatePadTrim(target_T)

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def accuracy(logits, y):
    preds = logits.argmax(dim=1)
    return (preds == y).float().mean().item()

def build_loaders(data_root, train_split, val_split, bs, workers, limit=None, train_sampler=None, collate_fn=None, train_indices=None, val_indices=None):
    train_ds = WindowsNPZDataset(data_root, train_split, limit=limit)
    val_ds   = WindowsNPZDataset(data_root, val_split,   limit=limit)

    if train_indices is not None:
        from torch.utils.data import Subset
        train_ds = Subset(train_ds, train_indices)
    if val_indices is not None:
        from torch.utils.data import Subset
        val_ds = Subset(val_ds, val_indices)

    train_ld = DataLoader(train_ds, batch_size=bs, shuffle=(train_sampler is None),
                          sampler=train_sampler, num_workers=workers, pin_memory=False,
                          collate_fn=collate_fn)
    val_ld   = DataLoader(val_ds,   batch_size=bs, shuffle=False,
                          num_workers=workers, pin_memory=False,
                          collate_fn=collate_fn)
    return train_ld, val_ld

def set_seed(seed: int):
    if seed is None:
        return
    try:
        import numpy as _np
    except Exception:
        _np = None
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if _np is not None:
        _np.random.seed(seed)

def scan_labels_npz(data_root: str, split: str):
    labels = []
    root = os.path.join(data_root, split)
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            if not fn.endswith(".npz"):
                continue
            p = os.path.join(dirpath, fn)
            try:
                with np.load(p) as z:
                    labels.append(int(z["y"]))
            except Exception:
                pass
    return labels

def make_weighted_sampler_for_split(data_root: str, split: str, id_remap: dict | None, num_classes: int):
    """
    Build a class-balanced WeightedRandomSampler for a given split.

    Returns:
        (sampler_or_None, class_counts_list, n_labeled)
    """
    # Try to get the file list from the dataset first (preserves dataset order)
    file_list = []
    try:
        _tmp_ds = WindowsNPZDataset(data_root, split, limit=None)
        file_list = getattr(_tmp_ds, "files", None) or []
    except Exception:
        file_list = []

    # Fallback: scan the filesystem
    if not file_list:
        root = os.path.join(data_root, split)
        for dirpath, _, filenames in os.walk(root):
            for fn in filenames:
                if fn.endswith(".npz"):
                    file_list.append(os.path.join(dirpath, fn))
        file_list.sort()

    # Load labels and apply optional remap
    labels = []
    for p in file_list:
        try:
            with np.load(p) as z:
                y = int(z["y"])
        except Exception:
            continue
        if id_remap is not None:
            if y not in id_remap:
                continue  # drop labels not present in the remap
            y = id_remap[y]
        labels.append(y)

    # Compute class counts and per-sample weights
    counts = [0] * num_classes
    for y in labels:
        if 0 <= y < num_classes:
            counts[y] += 1
    total = max(1, sum(counts))
    class_w = [total / (num_classes * max(1, c)) for c in counts]  # inverse-freq-ish

    sample_w = [class_w[y] if (0 <= y < num_classes) else 1.0 for y in labels]
    if len(sample_w) == 0:
        return None, counts, 0

    sampler = WeightedRandomSampler(
        weights=torch.as_tensor(sample_w, dtype=torch.double),
        num_samples=len(sample_w),
        replacement=True,
    )
    return sampler, counts, len(sample_w)

# --- Helper functions for effective class counts in stratified subsets ---
def _get_file_list_for_split(data_root: str, split: str):
    """
    Return the dataset's file list in the same order as WindowsNPZDataset would,
    falling back to a filesystem scan if needed.
    """
    file_list = []
    try:
        _tmp_ds = WindowsNPZDataset(data_root, split, limit=None)
        file_list = getattr(_tmp_ds, "files", None) or []
    except Exception:
        file_list = []
    if not file_list:
        root = os.path.join(data_root, split)
        for dirpath, _, filenames in os.walk(root):
            for fn in filenames:
                if fn.endswith(".npz"):
                    file_list.append(os.path.join(dirpath, fn))
        file_list.sort()
    return file_list

def count_labels_for_indices(data_root: str, split: str, indices, id_remap: dict | None, num_classes: int):
    """
    Count labels for a specific index subset of a split, using the dataset's file order.
    Applies optional id_remap and returns a length-`num_classes` list of counts.
    """
    files = _get_file_list_for_split(data_root, split)
    counts = [0] * num_classes
    for idx in indices or []:
        if 0 <= idx < len(files):
            try:
                with np.load(files[idx]) as z:
                    y = int(z["y"])
            except Exception:
                continue
            if id_remap is not None:
                y = id_remap.get(y, None)
                if y is None:
                    continue
            if 0 <= y < num_classes:
                counts[y] += 1
    return counts
def make_stratified_indices(data_root: str, split: str, per_class: int, id_remap: dict | None, num_classes: int):
    """
    Build a stratified subset by taking up to `per_class` samples for each class
    (after optional id_remap). Returns a sorted list of dataset indices that
    correspond to the dataset's internal file order.
    """
    # Prefer the dataset's file ordering if available
    file_list = []
    try:
        _tmp_ds = WindowsNPZDataset(data_root, split, limit=None)
        file_list = getattr(_tmp_ds, "files", None) or []
    except Exception:
        file_list = []

    # Fallback: scan the filesystem
    if not file_list:
        root = os.path.join(data_root, split)
        for dirpath, _, filenames in os.walk(root):
            for fn in filenames:
                if fn.endswith(".npz"):
                    file_list.append(os.path.join(dirpath, fn))
        file_list.sort()

    # Bucket indices per class
    buckets = {c: [] for c in range(num_classes)}
    for idx, p in enumerate(file_list):
        try:
            with np.load(p) as z:
                y = int(z["y"])
        except Exception:
            continue
        if id_remap is not None:
            y = id_remap.get(y, None)
            if y is None:
                continue
        if 0 <= y < num_classes and len(buckets[y]) < per_class:
            buckets[y].append(idx)
        # Early exit if all buckets are filled
        if all(len(buckets[c]) >= per_class for c in range(num_classes)):
            break

    indices = []
    for c in range(num_classes):
        indices.extend(buckets[c])
    indices.sort()
    return indices

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
    ap.add_argument("--progress", action="store_true", help="show per-batch tqdm progress bars")
    ap.add_argument("--log-file", type=str, default=None, help="optional path to append logs (also prints to console)")
    ap.add_argument("--progress-to-log", type=int, default=0,
                    help="If >0, write a plain-text progress line to the log every N batches (train/val).")
    ap.add_argument("--run-tag", type=str, default=None,
                    help="Optional tag to prefix checkpoint filenames, e.g., 'ddd', 'sust_nity', 'all3'. If not given, uses basename of --data-root.")
    ap.add_argument("--log-to-file-only", action="store_true",
                    help="If set, suppress console logging (except tqdm bars) and write logs only to --log-file.")
    ap.add_argument("--binary", action="store_true", help="Enable binary label remap (e.g., for DDD).")
    ap.add_argument("--pos-label-ids", type=str, default="3", help="Comma-separated original class ids to map to POSITIVE=1 when --binary.")
    ap.add_argument("--neg-label-ids", type=str, default="5", help="Comma-separated original class ids to map to NEGATIVE=0 when --binary.")
    ap.add_argument("--weighted-loss", action="store_true",
                    help="Use class-weighted CrossEntropy (weights from train split distribution; applies after binary remap if set).")
    ap.add_argument("--balanced-sampler", action="store_true",
                    help="Use a class-balanced WeightedRandomSampler for the train dataloader (recommended with imbalanced data).")
    ap.add_argument("--early-stop-patience", type=int, default=0,
                    help="Enable early stopping if >0 (number of epochs without val acc improvement).")
    ap.add_argument("--early-stop-min-delta", type=float, default=0.0,
                    help="Minimum improvement in val acc to reset early stopping patience.")
    ap.add_argument(
        "--early-metric",
        type=str,
        default="acc",
        choices=["acc", "f1", "spec", "balanced_acc"],
        help="Metric to monitor for best checkpoint and early stopping (binary only: f1/spec/balanced_acc)."
    )
    ap.add_argument("--seed", type=int, default=None, help="Set random seed for reproducibility.")
    ap.add_argument("--time-len", type=int, default=None,
                    help="Force temporal length T in the collate. If not set, uses per-batch median T.")
    ap.add_argument("--limit-train-per-class", type=int, default=None,
                    help="Debug: cap TRAIN to at most N samples per class (after binary remap).")
    ap.add_argument("--limit-val-per-class", type=int, default=None,
                    help="Debug: cap VAL to at most N samples per class (after binary remap).")
    ap.add_argument("--resume", type=str, default=None, help="Path to a FULL checkpoint to resume training from.")
    args = ap.parse_args()

    # Build a run tag to avoid checkpoint overwrites across datasets
    if args.run_tag and len(args.run_tag.strip()) > 0:
        _tag = "".join(c if (c.isalnum() or c in "-._") else "_" for c in args.run_tag.strip())
    else:
        _tag = os.path.basename(args.data_root.rstrip("/"))
        _tag = "".join(c if (c.isalnum() or c in "-._") else "_" for c in _tag)
    def _ckpt_path(name: str) -> str:
        # name like 'model2_hmgru_tna_best.pt'
        return os.path.join("checkpoints", f"{_tag}_{name}") if _tag else os.path.join("checkpoints", name)

    set_seed(args.seed)

    id_remap = None
    remap = None
    if args.binary:
        pos_ids = set(map(int, args.pos_label_ids.split(","))) if args.pos_label_ids else set()
        neg_ids = set(map(int, args.neg_label_ids.split(","))) if args.neg_label_ids else set()
        id_remap = {i: 1 for i in pos_ids}
        id_remap.update({i: 0 for i in neg_ids})

        def _remap_labels(t: torch.Tensor) -> torch.Tensor:
            out = torch.empty_like(t)
            uniq = torch.unique(t).tolist()
            for v in uniq:
                if v not in id_remap:
                    raise ValueError(f"Label {v} not in binary remap; add it via --pos-label-ids/--neg-label-ids")
                out[t == v] = id_remap[v]
            return out

        remap = _remap_labels
        args.num_classes = 2
    else:
        remap = None

    log_f = open(args.log_file, "a", buffering=1) if args.log_file else None
    def log_print(*a, **k):
        # If --log-to-file-only is set, keep console clean (tqdm bars still show),
        # but continue writing all messages to the log file.
        if not getattr(args, "log_to_file_only", False):
            print(*a, **k)
        if log_f is not None:
            print(*a, **k, file=log_f)
            log_f.flush()

    progress_ok = bool(args.progress and (tqdm is not None))
    if args.progress and tqdm is None:
        log_print("WARNING: --progress requested but tqdm is not installed. Run `pip install tqdm` to enable.")

    device = get_device()
    log_print(f"Using device: {device}")
    if not getattr(args, "resume", None):
        log_print("Starting a fresh run (no --resume).")

    train_labels_raw = scan_labels_npz(args.data_root, args.train_split)
    val_labels_raw   = scan_labels_npz(args.data_root, args.val_split)

    def _apply_remap_to_list(lst, id_map):
        if id_map is None:
            return lst
        out = []
        for v in lst:
            if v in id_map:
                out.append(id_map[v])
            else:
                continue
        return out

    train_labels = _apply_remap_to_list(train_labels_raw, id_remap)
    val_labels   = _apply_remap_to_list(val_labels_raw,   id_remap)

    def _counts(labels, k):
        c = [0]*k
        for v in labels:
            if 0 <= v < k:
                c[v] += 1
        return c

    train_counts = _counts(train_labels, args.num_classes)
    val_counts   = _counts(val_labels,   args.num_classes)
    log_print(f"Class counts (train): {train_counts}")
    log_print(f"Class counts (val)  : {val_counts}")

    train_sampler = None
    if args.balanced_sampler:
        train_sampler, counted_train, n_labeled = make_weighted_sampler_for_split(
            args.data_root, args.train_split, id_remap, args.num_classes
        )
        if train_sampler is not None:
            log_print(f"Using WeightedRandomSampler (balanced). Labeled samples: {n_labeled}  counts: {counted_train}")
        else:
            log_print("WARNING: Failed to build WeightedRandomSampler; falling back to plain shuffle.")

    # --- Optional stratified per-class limits for smoke tests ---
    train_indices = None
    val_indices   = None
    limit_for_builders = args.limit
    if (args.limit_train_per_class is not None) or (args.limit_val_per_class is not None):
        if args.limit is not None:
            log_print("NOTE: Ignoring --limit because per-class limits are set.")
            limit_for_builders = None
        if args.limit_train_per_class is not None:
            train_indices = make_stratified_indices(args.data_root, args.train_split,
                                                    args.limit_train_per_class, id_remap, args.num_classes)
            log_print(f"Using stratified train subset per-class={args.limit_train_per_class} → {len(train_indices)} samples")
            if train_sampler is not None:
                log_print("NOTE: Disabling balanced sampler due to stratified train subset.")
                train_sampler = None
        if args.limit_val_per_class is not None:
            val_indices = make_stratified_indices(args.data_root, args.val_split,
                                                  args.limit_val_per_class, id_remap, args.num_classes)
            log_print(f"Using stratified val subset per-class={args.limit_val_per_class} → {len(val_indices)} samples")

    # --- If we're using a stratified subset for TRAIN, recompute class counts from that subset ---
    effective_train_counts = train_counts
    if train_indices is not None:
        effective_train_counts = count_labels_for_indices(
            args.data_root, args.train_split, train_indices, id_remap, args.num_classes
        )
        log_print(f"Effective class counts (train subset): {effective_train_counts}")

    collate = make_collate_fn(args.time_len)
    train_ld, val_ld = build_loaders(args.data_root, args.train_split, args.val_split,
                                     args.batch_size, args.workers, limit_for_builders,
                                     train_sampler=train_sampler, collate_fn=collate,
                                     train_indices=train_indices, val_indices=val_indices)
    log_print(f"Loaded dataset: train={len(train_ld.dataset)}  val={len(val_ld.dataset)}")
    log_print(f"Early stopping monitors: {args.early_metric} (min_delta={args.early_stop_min_delta}, patience={args.early_stop_patience})")

    model = HMGRU_TNA(num_classes=args.num_classes, feat_dim=4)  # 4 = EAR/MAR/BLUR/ILLUM
    model.to(device)

    # --- Resume & freeze bootstrap ---
    start_epoch = 1
    best_metric = -1.0
    no_improve = 0
    backbone_frozen = True
    resume_loaded = False
    ckpt = None
    best_epoch = 0
    best_val_acc = 0.0
    early_stopped = False
    stop_epoch = None
    if getattr(args, "resume", None):
        try:
            ckpt = torch.load(args.resume, map_location=device)
            model.load_state_dict(ckpt["model"])
            start_epoch = int(ckpt.get("epoch", 0)) + 1
            best_metric = float(ckpt.get("best_metric", -1.0))
            no_improve = int(ckpt.get("no_improve", 0))
            backbone_frozen = bool(ckpt.get("backbone_frozen", start_epoch <= args.freeze_backbone_epochs))
            log_print(f"Resuming from {args.resume} at epoch {start_epoch} (best_metric={best_metric:.3f})")
            resume_loaded = True
        except Exception as e:
            log_print(f"WARNING: Failed to resume from {args.resume}: {e}")
    # Set backbone freeze state based on resume or schedule
    if start_epoch <= args.freeze_backbone_epochs:
        backbone_frozen = True if ckpt is None else backbone_frozen
    else:
        backbone_frozen = False if ckpt is None else backbone_frozen
    freeze(model.backbone, backbone_frozen)

    if args.weighted_loss:
        K = args.num_classes
        counts_for_weights = effective_train_counts  # uses subset counts if a stratified subset is active
        n_total = max(1, sum(counts_for_weights))
        w = []
        for n_c in counts_for_weights:
            w.append(n_total / (K * max(1, n_c)))
        w = torch.tensor(w, dtype=torch.float32, device=device)
        log_print(f"Using weighted CE loss with weights (from {'subset' if train_indices is not None else 'full train'} counts): {w.detach().cpu().numpy().round(4).tolist()}")
        criterion = nn.CrossEntropyLoss(weight=w)
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)

    # If resuming, restore optimizer/scheduler states when available
    if resume_loaded and ckpt is not None:
        if "optimizer" in ckpt:
            try:
                optimizer.load_state_dict(ckpt["optimizer"])
            except Exception as e:
                log_print(f"WARNING: could not load optimizer state: {e}")
        if "scheduler" in ckpt:
            try:
                scheduler.load_state_dict(ckpt["scheduler"])
            except Exception as e:
                log_print(f"WARNING: could not load scheduler state: {e}")

    os.makedirs("checkpoints", exist_ok=True)

    for epoch in range(start_epoch, args.epochs+1):
        t0 = time.time()
        if backbone_frozen and epoch == args.freeze_backbone_epochs + 1:
            freeze(model.backbone, False)
            backbone_frozen = False
            # Rebuild optimizer to include newly-trainable params (matches original behavior)
            optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)
            log_print("Unfroze backbone.")

        model.train()
        tr_loss, tr_acc, n = 0.0, 0.0, 0
        train_iter = train_ld if not progress_ok else tqdm(train_ld, total=len(train_ld), desc=f"Train {epoch:02d}/{args.epochs}", leave=True)
        for i, batch in enumerate(train_iter, 1):
            Xv = batch["video"].to(device)   # [B,T,3,224,224]
            Xf = batch["feats"].to(device)   # [B,T,D]
            y  = batch["label"].to(device)
            y_use = remap(y) if remap is not None else y

            optimizer.zero_grad(set_to_none=True)
            logits = model(Xv, Xf)
            loss = criterion(logits, y_use)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            bs = y.size(0)
            tr_loss += loss.item() * bs
            tr_acc  += accuracy(logits.detach(), y_use) * bs
            n       += bs

            if progress_ok:
                train_iter.set_postfix(loss=f"{loss.item():.4f}")

            if log_f is not None and args.progress_to_log and (i % args.progress_to_log == 0 or i == len(train_ld)):
                log_print(f"[Train {epoch:02d}] step {i}/{len(train_ld)}  loss={loss.item():.4f}")

        tr_loss /= n
        tr_acc  /= n


        model.eval()
        va_loss, va_acc, m = 0.0, 0.0, 0
        y_true_all, y_pred_all = [], []
        val_iter = val_ld if not progress_ok else tqdm(val_ld, total=len(val_ld), desc=f"Val   {epoch:02d}/{args.epochs}", leave=True)
        with torch.no_grad():
            for j, batch in enumerate(val_iter, 1):
                Xv = batch["video"].to(device)
                Xf = batch["feats"].to(device)
                y  = batch["label"].to(device)
                y_use = remap(y) if remap is not None else y

                logits = model(Xv, Xf)
                loss = criterion(logits, y_use)

                bs = y_use.size(0)
                m += bs
                va_loss += loss.item() * bs
                va_acc  += accuracy(logits, y_use) * bs

                preds = logits.argmax(dim=1)
                y_true_all.extend(y_use.detach().cpu().tolist())
                y_pred_all.extend(preds.detach().cpu().tolist())

                if log_f is not None and args.progress_to_log and (j % args.progress_to_log == 0 or j == len(val_ld)):
                    log_print(f"[Val   {epoch:02d}] step {j}/{len(val_ld)}  loss={loss.item():.4f}")
        if m > 0:
            va_loss /= m
            va_acc  /= m
        else:
            va_loss = float("nan")
            va_acc  = 0.0

        import numpy as _np
        cm = _np.zeros((args.num_classes, args.num_classes), dtype=int)
        for t, p in zip(y_true_all, y_pred_all):
            if 0 <= t < args.num_classes and 0 <= p < args.num_classes:
                cm[t, p] += 1
        log_print("Confusion matrix (rows=true, cols=pred):")
        for row in cm:
            log_print(" ".join(f"{int(x):6d}" for x in row))

        if args.num_classes == 2:
            tn, fp = int(cm[0,0]), int(cm[0,1])
            fn, tp = int(cm[1,0]), int(cm[1,1])
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1   = (2*prec*rec / (prec+rec)) if (prec+rec) > 0 else 0.0
            spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            log_print(f"Binary metrics: precision={prec:.3f}  recall={rec:.3f}  F1={f1:.3f}  specificity={spec:.3f}")
        metric_val = va_acc
        if args.num_classes == 2:
            if args.early_metric == "f1":
                metric_val = f1
            elif args.early_metric == "spec":
                metric_val = spec
            elif args.early_metric == "balanced_acc":
                metric_val = 0.5 * (rec + spec)

        metric_val = locals().get("metric_val", va_acc)
        scheduler.step()
        dt = time.time() - t0
        log_print(f"[Epoch {epoch:02d}] {dt:.1f}s  train loss {tr_loss:.4f} acc {tr_acc:.3f}  "
              f"val loss {va_loss:.4f} acc {va_acc:.3f}")

        # Save rolling full-state checkpoint each epoch (for exact resume)
        last_full = _ckpt_path("model2_hmgru_tna_last_full.pt")
        torch.save({
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch,
            "val_acc": va_acc,
            "early_metric": args.early_metric,
            "best_metric": best_metric,
            "no_improve": no_improve,
            "backbone_frozen": backbone_frozen,
        }, last_full)

        improved = (metric_val > (best_metric + args.early_stop_min_delta))
        if improved:
            best_metric = metric_val
            best_epoch = epoch
            best_val_acc = va_acc
            no_improve = 0
            ckpt = _ckpt_path("model2_hmgru_tna_best.pt")
            torch.save({"model": model.state_dict(),
                        "epoch": epoch,
                        "val_acc": va_acc,
                        "early_metric": args.early_metric,
                        "best_metric": best_metric}, ckpt)
            log_print(f"  ✔ Saved best checkpoint → {ckpt} ({args.early_metric}={best_metric:.3f}, acc={va_acc:.3f})")
            # Also save a full-state BEST checkpoint (resume-able)
            best_full = _ckpt_path("model2_hmgru_tna_best_full.pt")
            torch.save({
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "epoch": epoch,
                "val_acc": va_acc,
                "early_metric": args.early_metric,
                "best_metric": best_metric,
                "no_improve": no_improve,
                "backbone_frozen": backbone_frozen,
            }, best_full)
        else:
            no_improve += 1

        if args.early_stop_patience and args.early_stop_patience > 0:
            if no_improve >= args.early_stop_patience:
                early_stopped = True
                stop_epoch = epoch
                log_print(f"EARLY STOPPED at epoch {epoch}: no improvement in {args.early_metric} for ≥ {args.early_stop_patience} epochs. "
                          f"Best epoch {best_epoch} with {args.early_metric}={best_metric:.3f} (val_acc={best_val_acc:.3f}).")
                break

    # --- Write a compact run summary JSON for quick inspection ---
    try:
        os.makedirs("runs", exist_ok=True)
        summary = {
            "run_tag": _tag,
            "data_root": args.data_root,
            "train_split": args.train_split,
            "val_split": args.val_split,
            "early_metric": args.early_metric,
            "stopped_early": early_stopped,
            "stop_epoch": stop_epoch,
            "best_epoch": best_epoch,
            "best_metric": float(best_metric),
            "best_val_acc": float(best_val_acc),
            "epochs_planned": args.epochs,
            "epochs_completed": epoch
        }
        with open(os.path.join("runs", f"{_tag}_summary.json"), "w") as sf:
            json.dump(summary, sf, indent=2)
        log_print(f"Saved run summary → runs/{_tag}_summary.json")
    except Exception as e:
        log_print(f"WARNING: could not write run summary: {e}")

    if log_f is not None:
        log_f.close()

if __name__ == "__main__":
    main()