from pathlib import Path
import os
import numpy as np, pandas as pd, cv2
from tqdm import tqdm


try:
    from .features import process_frame, FEAT_DIM  # FEAT_DIM should be an int, e.g., 4
except Exception:  # features not implemented yet
    FEAT_DIM = 4
    def process_frame(frame_bgr, out_size=224):
        # Temporary stub: return None so preprocessing still runs and fills zeros.
        return None

def _windows(n, win, stride):
    i=0
    while i+win<=n:
        yield i, i+win
        i+=stride

# Define mapping of string labels to integers
LABEL_TO_INT = {
    "microsleep": 0,
    "yawning": 1,
    "fatigue": 2,
    "drowsy": 3,
    "sleep": 4,
    "alert": 5,
    "normal": 5,  # map normal to same as alert
    "distraction": 6,
    "Unknown": 7
}


def _process_task(task):
    """Module-level worker for multiprocessing. Receives a task dict and processes windows for one file."""
    import time as _time
    import numpy as _np
    import cv2 as _cv2
    try:
        from .features import process_frame as _process_frame, FEAT_DIM as _FEAT_DIM
    except Exception:
        def _process_frame(x, out_size=224):
            return None
        _FEAT_DIM = 4

    split = task["split"]
    fname = task["filename"]
    path = task["path"]
    fps = task["fps"]
    win_len = task["win_len"]
    stride = task["stride"]
    segs = task["segs"]
    img_size = int(task.get("img_size", 224))
    sample_rate = int(task.get("sample_rate", 1))
    out_dir = Path(task["out_dir"])

    start_time = _time.time()
    cap = _cv2.VideoCapture(str(path))
    n_frames = int(cap.get(_cv2.CAP_PROP_FRAME_COUNT)) or 0
    if n_frames <= 0:
        cap.release()
        return {"path": path, "time": _time.time() - start_time, "processed": 0}

    processed_windows = 0
    for s, e in _windows(n_frames, win_len, stride):
        Xv, Xf = [], []
        feat_dim = None
        ok = True
        frame_idx = s
        while frame_idx < e:
            cap.set(_cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ok, frame = cap.read()
            if not ok:
                break
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

        if not ok:
            break

        final_dim = feat_dim or _FEAT_DIM
        if any(len(f) != final_dim for f in Xf):
            Xf = [_np.pad(_np.asarray(f, dtype="float32"), (0, max(0, final_dim - len(f))), mode="constant")[:final_dim] for f in Xf]

        y = LABEL_TO_INT["Unknown"]
        if segs:
            overlaps = []
            for seg in segs:
                fs, fe = int(seg["f_start"]), int(seg["f_end"])
                ov = max(0, min(fe, e) - max(fs, s))
                label_str = str(seg.get("label", "")).lower()
                label_int = LABEL_TO_INT.get(label_str, LABEL_TO_INT["Unknown"])
                overlaps.append((ov, label_int))
            if overlaps and max(ov for ov, _ in overlaps) > 0:
                y = max(overlaps)[1]

        save_dir = out_dir / str(split) / Path(fname).stem
        save_dir.mkdir(parents=True, exist_ok=True)
        _np.savez_compressed(
            save_dir / f"win_{s}_{e}.npz",
            X_video=_np.stack(Xv),
            X_feats=_np.stack(Xf),
            y=int(y),
            start=s,
            end=e,
            fps=fps,
            feat_dim=int(final_dim),
        )
        processed_windows += 1

    cap.release()
    return {"path": path, "time": _time.time() - start_time, "processed": processed_windows}

def build_windows(index_csv: Path, out_dir: Path, win_sec: float, stride_sec: float, img_size: int = 224, sample_rate: int = 1, max_files: int | None = None):
    """Build sliding-window .npz files from the dataset index.

    Args:
        index_csv: path to dataset_index.csv
        out_dir: directory where windows will be saved
        win_sec: window length in seconds
        stride_sec: stride length in seconds
        img_size: image size for feature extractor
        sample_rate: process every `sample_rate`-th frame (1 = every frame)
        max_files: optional cap on number of (split,filename) groups to process for quick tests
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(index_csv)

    # Save label mapping for training
    label_map_file = out_dir / "label_mapping.json"
    import json
    with open(label_map_file, 'w') as f:
        json.dump(LABEL_TO_INT, f, indent=2)

    df["open_path"] = np.where(
        (df.get("standardize", False) == True) & df.get("std_relpath").notna(),
        df["std_relpath"], df["relpath"]
    )

    # Gather label intervals per video (frame space)
    lab_cols = [c for c in ["filename", "f_start", "f_end", "label"] if c in df.columns]
    intervals = (df[lab_cols].dropna()
                 .groupby("filename")[ ["f_start", "f_end", "label"] ]
                 .apply(lambda g: g.sort_values("f_start").to_dict("records"))
                 .to_dict())

    # Process each file once per split
    fps_map = dict(zip(df["filename"], df["fps"]))

    import time
    groups = list(df.groupby(["split", "filename"], sort=False))
    total_groups = len(groups)
    print(f"Preprocessing {total_groups} (split,filename) groups; sample_rate={sample_rate}; max_files={max_files}")

    # Build lightweight task list so worker processes don't need the entire df
    tasks = []
    for (split, fname), rows in groups:
        if max_files is not None and len(tasks) >= max_files:
            break
        path = rows.iloc[0]["open_path"]
        fps = float(fps_map.get(fname, 25.0))
        win_len = int(round(win_sec * fps))
        stride = int(round(stride_sec * fps))
        raw_segs = intervals.get(fname, [])
        # sanitize segments to plain Python types for pickling
        segs = []
        for s in raw_segs:
            try:
                segs.append({
                    "f_start": int(s.get("f_start", 0)),
                    "f_end": int(s.get("f_end", 0)),
                    "label": str(s.get("label", "")),
                })
            except Exception:
                segs.append({"f_start": 0, "f_end": 0, "label": str(s)})
        tasks.append({
            "split": split,
            "filename": fname,
            "path": path,
            "fps": fps,
            "win_len": win_len,
            "stride": stride,
            "segs": segs,
            "img_size": img_size,
            "sample_rate": sample_rate,
            "out_dir": str(out_dir),
        })

    # Use module-level _process_task for multiprocessing (defined above)

    # Decide on parallel vs serial execution
    workers = int(os.environ.get("PREPROCESS_WORKERS", "0"))
    results = []
    if workers and workers > 1:
        import multiprocessing as _mp
        print(f"Running preprocessing with {workers} worker processes...")
        with _mp.Pool(processes=workers) as pool:
            for i, res in enumerate(pool.imap_unordered(_process_task, tasks), 1):
                results.append(res)
                if i % 50 == 0:
                    print(f"Completed {i}/{len(tasks)} tasks")
    else:
        print("Running preprocessing in serial (single process)")
        for i, t in enumerate(tasks, 1):
            res = _process_task(t)
            results.append(res)
            if i % 50 == 0:
                print(f"Completed {i}/{len(tasks)} tasks")

    # Summarize slow tasks
    slow_tasks = sorted([r for r in results if r and r.get("time", 0) > 5.0], key=lambda x: -x["time"])[:10]
    if slow_tasks:
        print("Top slow tasks:")
        for t in slow_tasks:
            print(f" - {t['path']}: {t['time']:.2f}s (windows={t['processed']})")