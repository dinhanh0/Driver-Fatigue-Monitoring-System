import os, sys, csv, re, argparse, subprocess
from pathlib import Path
from typing import Dict, List, Tuple, Optional

def probe_video(path: str) -> Tuple[Optional[float], Optional[int]]:
    fps = frames = None
    try:
        from pymediainfo import MediaInfo
        mi = MediaInfo.parse(path)
        for track in mi.tracks:
            if track.track_type == "Video":
                if track.frame_rate:
                    try:
                        fps = float(str(track.frame_rate).split()[0])
                    except Exception:
                        pass
                if track.frame_count:
                    try:
                        frames = int(track.frame_count)
                    except Exception:
                        pass
                break
    except Exception:
        pass
    if fps is None or frames is None:
        try:
            import cv2
            cap = cv2.VideoCapture(path)
            if fps is None:
                f = cap.get(cv2.CAP_PROP_FPS)
                fps = float(f) if f and f > 1e-3 else None
            if frames is None:
                c = cap.get(cv2.CAP_PROP_FRAME_COUNT)
                frames = int(c) if c and c > 0 else None
            cap.release()
        except Exception:
            pass
    return fps, frames

POS = {"drowsy","sleep","sleepy","yawn","yawning","closed","microsleep","fatigue","drowsiness"}
NEG = {"alert","awake","normal","open","nondrowsy","non_drowsy","safe","neutral"}

def infer_label_from_path(p: Path) -> Optional[int]:
    parts = [s.lower() for s in p.parts]
    for s in reversed(parts):
        tokens = re.split(r"[_\-\s]", s)
        for t in tokens:
            if t in POS:  return 1
            if t in NEG:  return 0
    return None

def find_videos(root: str) -> List[Path]:
    exts = {".mp4",".avi",".mov",".mkv",".m4v"}
    out = []
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            if Path(fn).suffix.lower() in exts:
                out.append(Path(dirpath) / fn)
    return out

def write_csv(rows: List[Dict], out_csv: str):
    cols = ["video_path","dataset","subject_id","label","fps","frames","split_hint"]
    os.makedirs(Path(out_csv).parent, exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in cols})

def build_index_for_dataset(name: str, root: str, label_map: Dict[str,int]=None) -> List[Dict]:
    vids = find_videos(root)
    rows = []
    for v in vids:
        label = None
        if label_map:
            for part in reversed([s.lower() for s in v.parts]):
                for k, val in label_map.items():
                    if re.search(rf"(^|[_\-\s]){re.escape(k)}([_\-\s]|$)", part):
                        label = int(val); break
                if label is not None: break
        if label is None:
            label = infer_label_from_path(v)
        if label is None:
            continue

        fps, frames = probe_video(str(v))
        subj = None
        for part in v.parts:
            if re.match(r"^[PpSs]\d+", part) or re.match(r"^subject[_\-]?\d+", part, re.I):
                subj = part; break

        rows.append({
            "video_path": str(v),
            "dataset": name,
            "subject_id": subj if subj else "",
            "label": label,          # 0 alert, 1 drowsy/fatigue
            "fps": fps if fps is not None else "",
            "frames": frames if frames is not None else "",
            "split_hint": ""         # optional; fill later if dataset has provided split
        })
    return rows

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nitymed-root", type=str, default="data/raw/nitymed")
    ap.add_argument("--ddd-root", type=str, default="data/raw/ddd")
    ap.add_argument("--sust-root", type=str, default="data/raw/sust_ddd")
    ap.add_argument("--out-dir", type=str, default="data/index")
    args = ap.parse_args()

    # If you know exact folder labels for a dataset, declare here to be precise:
    # (Adjust after you inspect data/raw/* structure)
    nitymed_map = {}  # unknown -> rely on infer_label_from_path
    ddd_map = { "drowsy":1, "yawning":1, "sleep":1, "closed":1, "open":0, "normal":0, "alert":0 }
    sust_map = { "drowsy":1, "yawn":1, "sleep":1, "closed":1, "open":0, "normal":0, "alert":0 }

    rows_n = build_index_for_dataset("nitymed", args.nitymed_root, nitymed_map)
    rows_d = build_index_for_dataset("ddd",     args.ddd_root,     ddd_map)
    rows_s = build_index_for_dataset("sust_ddd",args.sust_root,    sust_map)

    print(f"NITYMED: {len(rows_n)} videos")
    print(f"DDD:     {len(rows_d)} videos")
    print(f"SUST-DDD:{len(rows_s)} videos")

    write_csv(rows_n, os.path.join(args.out_dir, "nitymed_index.csv"))
    write_csv(rows_d, os.path.join(args.out_dir, "ddd_index.csv"))
    write_csv(rows_s, os.path.join(args.out_dir, "sust_index.csv"))

    # Merge for a combined index (handy later)
    all_rows = rows_n + rows_d + rows_s
    write_csv(all_rows, os.path.join(args.out_dir, "combined_all.csv"))
    print("Wrote indices to", args.out_dir)

if __name__ == "__main__":
    main()