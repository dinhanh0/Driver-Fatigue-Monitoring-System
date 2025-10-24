from pathlib import Path
import os
import json
import numpy as np, pandas as pd, cv2
from tqdm import tqdm
from typing import Optional, Tuple, List

# ---------------------------------------------
# Feature extractor import (falls back to stub)
# ---------------------------------------------
try:
    from .features import process_frame, FEAT_DIM  # FEAT_DIM should be an int, e.g., 4
except Exception:  # features not implemented yet
    FEAT_DIM = 4
    def process_frame(frame_bgr, out_size=224):
        # Temporary stub: return None so preprocessing still runs and fills zeros.
        return None

# ---------------------------------------------
# Sliding window helper
# ---------------------------------------------

def _windows(n, win, stride):
    i = 0
    while i + win <= n:
        yield i, i + win
        i += stride

# ---------------------------------------------
# Label mapping (string → int) and utility
# ---------------------------------------------

LABEL_TO_INT = {
    "microsleep": 0,
    "yawning": 1,
    "fatigue": 2,
    "drowsy": 3,
    "sleep": 4,
    "alert": 5,
    "normal": 5,  # map normal to same as alert
    "distraction": 6,
    "unknown": 7,
}

def _normalize_label_name(v) -> str:
    """Normalize free-form labels to a canonical name used by LABEL_TO_INT."""
    if v is None:
        return "unknown"
    s = str(v).strip().lower()
    # unify separators
    s = s.replace("_", " ").replace("-", " ")
    # collapse repeated whitespace
    s = " ".join(s.split())
    # synonyms
    if s in {"non drowsy", "awake", "alert", "normal"}:
        return "normal"
    if s in {"drowsy", "sleepy"}:
        return "drowsy"
    return s


def _coerce_label(v) -> int:
    try:
        # Already an int-like label
        return int(v)
    except Exception:
        s = _normalize_label_name(v)
        return LABEL_TO_INT.get(s, LABEL_TO_INT["unknown"])

# ---------------------------------------------
# Unified frame source (video file OR image sequence dir)
# ---------------------------------------------
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


class FrameSource:
    def __init__(self, media_type: str, source_path: str, fps: float):
        self.media_type = (media_type or "").lower() or "video"
        self.source_path = source_path
        self.fps = float(fps) if fps else 25.0
        self.cap = None
        self.files: List[str] = []
        if self.media_type == "images":
            p = Path(source_path)
            if p.is_dir():
                self.files = sorted(
                    [str(x) for x in p.iterdir() if x.suffix.lower() in IMAGE_EXTS]
                )
            self.n_frames = len(self.files)
        else:
            self.cap = cv2.VideoCapture(str(source_path))
            self.n_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0

    def get(self, idx: int) -> Tuple[bool, Optional[np.ndarray]]:
        if self.media_type == "images":
            if 0 <= idx < self.n_frames:
                frame = cv2.imread(self.files[idx])
                return (frame is not None), frame
            return False, None
        else:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = self.cap.read()
            return ok, frame

    def release(self):
        if self.cap is not None:
            self.cap.release()

# ---------------------------------------------
# Worker that processes windows for ONE media item
# ---------------------------------------------

def _process_task(task):
    import time as _time
    import numpy as _np
    import cv2 as _cv2

    # Local import for features to isolate workers
    try:
        from .features import process_frame as _process_frame, FEAT_DIM as _FEAT_DIM
    except Exception:
        def _process_frame(x, out_size=224):
            return None
        _FEAT_DIM = 4

    split = task["split"]
    key = task["key"]  # unique identifier (filename/video_path/images_dir)
    media_type = task["media_type"]
    path = task["path"]
    fps = float(task["fps"]) if task.get("fps") is not None else 25.0
    win_len = int(task["win_len"])  # in frames
    stride = int(task["stride"])    # in frames
    segs = task["segs"]              # list of {f_start,f_end,label}
    default_label = int(task.get("default_label", LABEL_TO_INT["unknown"]))
    img_size = int(task.get("img_size", 224))
    sample_rate = int(task.get("sample_rate", 1))
    out_dir = Path(task["out_dir"])
    skip_existing = bool(task.get("skip_existing", False))
    save_dir = out_dir / str(split) / Path(key).stem

    start_time = _time.time()
    src = FrameSource(media_type, path, fps)
    n_frames = int(src.n_frames) or 0
    if n_frames <= 0:
        src.release()
        return {"path": path, "time": _time.time() - start_time, "processed": 0}

    # If requested, skip this item if all expected window files already exist
    if skip_existing:
        expected_pairs = list(_windows(n_frames, win_len, stride))
        if save_dir.exists() and all((save_dir / f"win_{s}_{e}.npz").exists() for (s, e) in expected_pairs):
            src.release()
            return {"path": path, "time": _time.time() - start_time, "processed": 0, "skipped": True}

    processed_windows = 0
    for s, e in _windows(n_frames, win_len, stride):
        Xv, Xf = [], []
        feat_dim = None
        ok = True
        frame_idx = s
        while frame_idx < e:
            ok, frame = src.get(frame_idx)
            if not ok or frame is None:
                # keep timing with zeros
                Xv.append(_np.zeros((3, img_size, img_size), dtype="float32"))
                Xf.append(_np.zeros((feat_dim or _FEAT_DIM,), dtype="float32"))
                frame_idx += sample_rate
                continue

            out = _process_frame(frame, out_size=img_size)
            if out is None:
                Xv.append(_np.zeros((3, img_size, img_size), dtype="float32"))
                Xf.append(_np.zeros((feat_dim or _FEAT_DIM,), dtype="float32"))
            else:
                img, feats = out
                if feat_dim is None:
                    feat_dim = int(_np.asarray(feats).shape[0])
                else:
                    if int(_np.asarray(feats).shape[0]) != feat_dim:
                        if int(_np.asarray(feats).shape[0]) < feat_dim:
                            pad = feat_dim - int(_np.asarray(feats).shape[0])
                            feats = _np.pad(_np.asarray(feats, dtype="float32"), (0, pad), mode="constant")
                        else:
                            feats = _np.asarray(feats, dtype="float32")[:feat_dim]
                Xv.append(img)
                Xf.append(_np.asarray(feats, dtype="float32"))

            frame_idx += sample_rate

        # choose label: default per-file, overridden by majority overlap if segs provided
        y = int(default_label)
        if segs:
            overlaps = []
            for seg in segs:
                fs, fe = int(seg.get("f_start", 0)), int(seg.get("f_end", 0))
                ov = max(0, min(fe, e) - max(fs, s))
                lbl = _coerce_label(seg.get("label"))
                overlaps.append((ov, lbl))
            if overlaps and max(ov for ov, _ in overlaps) > 0:
                y = max(overlaps)[1]

        final_dim = int(feat_dim or _FEAT_DIM)
        if any(len(f) != final_dim for f in Xf):
            Xf = [
                _np.pad(_np.asarray(f, dtype="float32"), (0, max(0, final_dim - len(f))), mode="constant")[:final_dim]
                for f in Xf
            ]

        save_dir.mkdir(parents=True, exist_ok=True)
        _np.savez_compressed(
            save_dir / f"win_{s}_{e}.npz",
            X_video=_np.stack(Xv),
            X_feats=_np.stack(Xf),
            y=int(y),
            start=s,
            end=e,
            fps=fps,
            feat_dim=final_dim,
            media_type=media_type,
            source=str(path),
        )
        processed_windows += 1

    src.release()
    return {"path": path, "time": _time.time() - start_time, "processed": processed_windows}

# ---------------------------------------------
# Public API: build_windows
# ---------------------------------------------

def build_windows(index_csv: Path, out_dir: Path, win_sec: float, stride_sec: float,
                  img_size: int = 224, sample_rate: int = 1, max_files: Optional[int] = None,
                  skip_existing: bool = False):
    """Build sliding-window .npz files from a mixed media (video or images) index.

    Args:
        ...
        skip_existing: if True, skip a (split,key) group when all expected window files already exist.

    The index CSV can be one of two styles:
      • Legacy: columns include filename, relpath/std_relpath, standardize, fps, split, and optional f_start/f_end/label rows.
      • New:    columns include media_type (video/images), video_path/images_dir, fps, split, label.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(index_csv)
    # Ensure a usable split column
    # If the CSV has no explicit 'split', fall back to 'split_hint' when present; else default everything to 'train'.
    if "split" not in df.columns:
        if "split_hint" in df.columns:
            df["split"] = df["split_hint"].fillna("train").astype(str)
        else:
            df["split"] = "train"

    # Persist label mapping for training code
    with open(out_dir / "label_mapping.json", "w") as f:
        json.dump(LABEL_TO_INT, f, indent=2)

    # Normalize paths into common columns: key (unique id), media_type, open_path, fps, label
    if ("media_type" in df.columns) or ("video_path" in df.columns) or ("images_dir" in df.columns):  # New/mixed index style
        # Prepare candidate columns safely
        vp = df["video_path"] if "video_path" in df.columns else pd.Series(pd.NA, index=df.index)
        idirs = df["images_dir"] if "images_dir" in df.columns else pd.Series(pd.NA, index=df.index)

        # media_type: prefer explicit column; otherwise infer from which path is present
        if "media_type" in df.columns:
            mt = df["media_type"].fillna("")
            inferred = pd.Series("video", index=df.index)
            inferred = inferred.mask(idirs.notna(), "images")
            df["media_type"] = mt.where(mt != "", inferred).str.lower()
        else:
            df["media_type"] = pd.Series("video", index=df.index)
            df.loc[idirs.notna(), "media_type"] = "images"

        # open_path: choose the concrete path field
        df["open_path"] = vp.fillna(idirs).fillna("")

        # Unique key: use the concrete path
        df["key"] = df["open_path"]

        # Ensure fps numeric
        df["fps"] = pd.to_numeric(df.get("fps", 25.0), errors="coerce").fillna(25.0)

        # Default label per file
        df["default_label"] = df.get("label", None).apply(_coerce_label)

        # Segments (optional): expect f_start/f_end/label rows if present
        key_col = "key"
    else:  # Legacy/minimal index style (relpath/std_relpath/filename)
        # Build soft defaults for possibly-missing columns
        std = df["standardize"] if "standardize" in df.columns else pd.Series(False, index=df.index)
        std_rel = df["std_relpath"] if "std_relpath" in df.columns else pd.Series(pd.NA, index=df.index)
        # Prefer `relpath`; fall back to a `path` column if present; else empty
        if "relpath" in df.columns:
            rel = df["relpath"]
        elif "path" in df.columns:
            rel = df["path"]
        else:
            rel = pd.Series("", index=df.index)

        # Choose standardized path when available, otherwise rel/path, and finally filename
        open_path = np.where((std == True) & (std_rel.notna()), std_rel, rel)
        if "filename" in df.columns:
            # Fill remaining empties with filename
            open_path = pd.Series(open_path).fillna(df["filename"])  # ensure Series for fillna
        df["open_path"] = open_path

        df["media_type"] = "video"
        df["key"] = df["filename"] if "filename" in df.columns else df["open_path"]
        df["fps"] = pd.to_numeric(df.get("fps", 25.0), errors="coerce").fillna(25.0)
        df["default_label"] = df.get("label", None).apply(_coerce_label)
        key_col = "filename" if "filename" in df.columns else "key"

    # Gather label intervals per key (frame space), if provided
    lab_cols = [c for c in [key_col, "f_start", "f_end", "label"] if c in df.columns]
    if len(lab_cols) == 4:
        intervals = (
            df[lab_cols].dropna()
            .groupby(key_col)[["f_start", "f_end", "label"]]
            .apply(lambda g: g.sort_values("f_start").to_dict("records"))
            .to_dict()
        )
    else:
        intervals = {}

    # Build tasks per (split, key)
    groups = list(df.groupby(["split", "key"], sort=False))
    print(f"Preprocessing {len(groups)} (split, key) groups; sample_rate={sample_rate}; max_files={max_files}")

    tasks = []
    for (split, key), rows in groups:
        if max_files is not None and len(tasks) >= max_files:
            break
        row0 = rows.iloc[0]
        media_type = str(row0.get("media_type", "video")).lower()
        path = str(row0.get("open_path"))
        fps = float(row0.get("fps", 25.0))
        win_len = int(round(float(win_sec) * fps))
        stride = int(round(float(stride_sec) * fps))
        raw_segs = intervals.get(key, [])
        segs = []
        for s in raw_segs:
            try:
                segs.append({
                    "f_start": int(s.get("f_start", 0)),
                    "f_end": int(s.get("f_end", 0)),
                    "label": s.get("label", "unknown"),
                })
            except Exception:
                segs.append({"f_start": 0, "f_end": 0, "label": "unknown"})
        # Default label from CSV (after normalization)
        def_label = int(row0.get("default_label", LABEL_TO_INT["unknown"]))
        # If still unknown, fall back to hints in the path (helps datasets like DDD)
        if def_label == LABEL_TO_INT["unknown"]:
            pl = str(path).lower().replace("_", " ").replace("-", " ")
            if any(k in pl for k in ("non drowsy", "awake", "alert", "normal")):
                def_label = LABEL_TO_INT["normal"]
            elif any(k in pl for k in ("drowsy", "sleep")):
                def_label = LABEL_TO_INT["drowsy"]

        tasks.append({
            "split": split,
            "key": key,
            "media_type": media_type,
            "path": path,
            "fps": fps,
            "win_len": win_len,
            "stride": stride,
            "segs": segs,
            "default_label": def_label,
            "img_size": int(img_size),
            "sample_rate": int(sample_rate),
            "out_dir": str(out_dir),
            "skip_existing": bool(skip_existing),
        })

    # Run (optionally parallel)
    workers = int(os.environ.get("PREPROCESS_WORKERS", "0"))
    results = []
    if workers and workers > 1:
        import multiprocessing as _mp
        print(f"Running preprocessing with {workers} worker processes...")
        with _mp.Pool(processes=workers) as pool:
            for res in tqdm(
                pool.imap_unordered(_process_task, tasks),
                total=len(tasks),
                unit="task",
                desc="Preprocess",
                dynamic_ncols=True,
            ):
                results.append(res)
    else:
        print("Running preprocessing in serial (single process)")
        for t in tqdm(
            tasks,
            total=len(tasks),
            unit="task",
            desc="Preprocess",
            dynamic_ncols=True,
        ):
            res = _process_task(t)
            results.append(res)

    # Report slow tasks
    slow = sorted(
        [r for r in results if r and r.get("time", 0) > 5.0],
        key=lambda x: -x["time"]
    )[:10]
    if slow:
        print("Top slow tasks:")
        for t in slow:
            print(f" - {t['path']}: {t['time']:.2f}s (windows={t['processed']})")


# ---------------------------------------------
# CLI Entrypoint
# ---------------------------------------------
if __name__ == "__main__":
    import argparse
    from pathlib import Path as _Path

    parser = argparse.ArgumentParser(description="Build sliding-window NPZs from an index CSV (videos and/or image folders)")
    parser.add_argument("--index-csv", type=_Path, required=True, help="Path to combined index CSV")
    parser.add_argument("--out-dir", type=_Path, required=True, help="Output directory for window NPZ files")
    parser.add_argument("--win-sec", type=float, default=4.0, help="Window length in seconds (default: 4.0)")
    parser.add_argument("--stride-sec", type=float, default=1.0, help="Stride in seconds (default: 1.0)")
    parser.add_argument("--img-size", type=int, default=224, help="Square image size (default: 224)")
    parser.add_argument("--sample-rate", type=int, default=1, help="Frame sampling step (default: 1)")
    parser.add_argument("--max-files", type=int, default=None, help="Optional cap on number of (split,key) groups to process")
    parser.add_argument("--skip-existing", action="store_true", help="Skip a group if all its expected window files already exist")

    args = parser.parse_args()

    build_windows(
        index_csv=args.index_csv,
        out_dir=args.out_dir,
        win_sec=args.win_sec,
        stride_sec=args.stride_sec,
        img_size=args.img_size,
        sample_rate=args.sample_rate,
        max_files=args.max_files,
        skip_existing=args.skip_existing,
    )