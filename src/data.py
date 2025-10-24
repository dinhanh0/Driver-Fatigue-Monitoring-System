from pathlib import Path #cleaner file path
import argparse #used for flagging
import os 
import random

# To run ffmpeg (used to standadize videos to 25fps, not used here)
import subprocess

import pandas as pd
import cv2


# General settings (tweakable)
VIDEO_PATTERNS = ["*.mp4", "*.MP4"]  # video file extensions to scan
IMAGE_PATTERNS = ["*.jpg", "*.jpeg", "*.png"]  # image file extensions to scan
FALLBACK_FPS = 25.0  # used if a video file reports 0 or missing FPS
TRAIN_RATIO, VAL_RATIO = 0.70, 0.15   # test = 1 - TRAIN_RATIO - VAL_RATIO
RANDOM_SEED = 42  # ensure output result is reproducible

# Standardization policy (flags + optional re-encode)
TARGET_FPS = 25.0
TARGET_SHORT_EDGE = 720 #
STANDARDIZED_DIR = Path("data/processed/videos_std")  # output for re-encoded files
# ----------------------------------------------------------------


def scan_files(root_dir: Path, patterns: list[str]) -> list[Path]:
    """Find all files under root_dir that match the given patterns."""
    found: list[Path] = []
    for pattern in patterns:
        # rglob walks all subfolders; extend the list of matches
        found += list(root_dir.rglob(pattern))
    # keep only files (exclude dirs), and sort for determinism
    return sorted([p for p in found if p.is_file()])

def scan_videos(videos_root_dir: Path) -> list[Path]:
    """Find all videos under videos_root_dir that match VIDEO_PATTERNS."""
    return scan_files(videos_root_dir, VIDEO_PATTERNS)

def scan_images(images_root_dir: Path) -> list[Path]:
    """Find all images under images_root_dir that match IMAGE_PATTERNS."""
    return scan_files(images_root_dir, IMAGE_PATTERNS)


# Cache for video metadata to avoid re-reading the same files
_video_meta_cache = {}

def probe_video(video_path: Path) -> dict:
    """Lightweight QC: open with OpenCV and read basic metadata."""
    video_path_str = str(video_path)
    
    # Check cache first
    if video_path_str in _video_meta_cache:
        return _video_meta_cache[video_path_str]
        
    cap = cv2.VideoCapture(video_path_str)      # open a handle to the file
    can_open = cap.isOpened()                    # did OpenCV successfully open it?

    # Query metadata if open; else use zeros
    fps = float(cap.get(cv2.CAP_PROP_FPS)) if can_open else 0.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if can_open else 0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) if can_open else 0
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) if can_open else 0

    cap.release()                                # free the file handle

    # If FPS reported as 0, fall back to a sensible default
    fps = fps if fps > 0 else FALLBACK_FPS

    # Duration: frames / fps (guard when frames==0)
    duration_sec = (frame_count / fps) if frame_count > 0 else 0.0

    # If frames is 0, approximate using duration * fps (keeps downstream math nonzero)
    if frame_count <= 0:
        frame_count = int(round(duration_sec * fps))

    # Create metadata dictionary
    meta = {
        "filename": video_path.name,             # basename (no folders)
        "relpath": str(video_path),              # full path string for CSV
        "can_decode": int(can_open),             # 1 if openable, else 0
        "fps": fps,
        "frames": frame_count,
        "duration_sec": float(duration_sec),
        "width": width,
        "height": height,
        "type": "video"
    }
    
    # Cache the result
    _video_meta_cache[video_path_str] = meta.copy()
    return meta

def probe_image(image_path: Path) -> dict:
    """Lightweight QC: open with OpenCV and read basic metadata."""
    img = cv2.imread(str(image_path))  # open image
    can_open = img is not None

    # Query metadata if open; else use zeros
    height, width = img.shape[:2] if can_open else (0, 0)

    return {
        "filename": image_path.name,             # basename (no folders)
        "relpath": str(image_path),              # full path string for CSV
        "can_decode": int(can_open),             # 1 if openable, else 0
        "fps": 0.0,                              # not applicable for images
        "frames": 1,                             # single frame
        "duration_sec": 0.0,                     # not applicable for images
        "width": width,
        "height": height,
        "type": "image"
    }


def infer_category_and_gender(path_str: str) -> tuple[str, str]:
    """
    Heuristic labels from folder names and filenames:
    - category: one of known keywords in the path/filename
    - gender: 'Female'/'Male' if present in path/filename, else 'Unknown'
    
    Handles multiple dataset formats:
    - DSM-Dataset: microsleep/yawning in folders
    - SUST-DDD: sleep/fatigue/distraction in folders
    - DriverDrowsiness: drowsy/alert in filenames
    """
    path = Path(path_str)
    parts = path.parts
    stem = path.stem.lower()  # filename without extension
    
    # Combined keywords from all datasets
    known_categories = {
        # DSM Dataset
        "microsleep": "microsleep",
        "yawning": "yawning",
        # SUST-DDD
        "sleep": "sleep",
        "fatigue": "fatigue",
        "distraction": "distraction",
        # Driver Drowsiness Dataset
        "drowsy": "drowsy",
        "alert": "alert",
        # Generic terms
        "tired": "fatigue",
        "normal": "alert"
    }
    
    # First check path parts for category
    for part in parts:
        part_lower = part.lower()
        if part_lower in known_categories:
            category = known_categories[part_lower]
            break
    else:
        # Then check filename for category keywords
        for keyword, mapping in known_categories.items():
            if keyword in stem:
                category = mapping
                break
        else:
            category = "Unknown"

    # Gender detection
    lower_parts = [p.lower() for p in parts] + [stem]
    if any("female" in p for p in lower_parts) or any("f_" == p[:2] for p in lower_parts):
        gender = "Female"
    elif any("male" in p for p in lower_parts) or any("m_" == p[:2] for p in lower_parts):
        gender = "Male"
    else:
        gender = "Unknown"

    return category, gender


def _merge_overlapping_segments(aligned_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge touching/overlapping [f_start, f_end) intervals per (filename, label).
    This keeps a clean set of non-overlapping segments.
    """
    if aligned_df.empty:
        return aligned_df

    aligned_df = aligned_df.sort_values(["filename", "label", "f_start", "f_end"])

    merged_rows = []
    for (fname, lab), group in aligned_df.groupby(["filename", "label"], sort=False):
        current = None
        for _, row in group.iterrows():
            row_dict = row.to_dict()
            if current is None:
                current = row_dict
                continue

            # If intervals overlap or touch, extend the current interval
            if row_dict["f_start"] <= current["f_end"]:
                current["f_end"] = max(current["f_end"], row_dict["f_end"])
                current["t_end_sec"] = max(current["t_end_sec"], row_dict["t_end_sec"])
            else:
                merged_rows.append(current)
                current = row_dict

        if current is not None:
            merged_rows.append(current)

    return pd.DataFrame(merged_rows)


def build_table(raw_data_dir: Path) -> pd.DataFrame:
    """
    Build a consolidated table with:
    - file metadata (QC) for both videos and images
    - labels aligned to timebase (for videos) or single frame (for images)
    - standardization flags
    """
    print(f"Scanning for media files in {raw_data_dir}...")
    
    # Scan both videos and images
    video_paths = scan_videos(raw_data_dir)
    image_paths = scan_images(raw_data_dir)
    
    total_files = len(video_paths) + len(image_paths)
    print(f"Found {len(video_paths)} videos and {len(image_paths)} images")
    
    # Probe every file into a metadata row with progress reporting
    meta_rows = []
    
    print("Processing videos...")
    for i, path in enumerate(video_paths, 1):
        if i % 10 == 0:  # Progress update every 10 files
            print(f"Processing video {i}/{len(video_paths)}")
        meta_rows.append(probe_video(path))
    
    print("Processing images...")
    for i, path in enumerate(image_paths, 1):
        if i % 100 == 0:  # Progress update every 100 files (images are faster)
            print(f"Processing image {i}/{len(image_paths)}")
        meta_rows.append(probe_image(path))
    
    print(f"Creating DataFrame with {len(meta_rows)} total entries...")
    meta_df = pd.DataFrame(meta_rows)

    # If no files found, return an empty but well-shaped table so callers don't crash.
    if meta_df.empty:
        print(f"[WARN] No media files found under {raw_data_dir}. Writing an empty dataset index.")
        cols = [
            "filename",
            "relpath",
            "can_decode",
            "fps",
            "frames",
            "duration_sec",
            "width",
            "height",
            "subject_id",
            "label",
            "t_start_sec",
            "t_end_sec",
            "f_start",
            "f_end",
            "needs_fps_fix",
            "needs_resize",
            "standardize",
            "std_relpath",
            "split",
        ]
        return pd.DataFrame(columns=cols)

    # If hand labels exist, load them. Else, infer one whole-video segment per file.
    labels_csv = raw_data_dir / "labels.csv"
    if labels_csv.exists():
        labels_df = pd.read_csv(labels_csv)
        # Normalize key to basename (so we match meta_df["filename"])
        labels_df["video"] = labels_df["video"].apply(lambda s: Path(str(s)).name)
    else:
        inferred = []
        for _, mrow in meta_df.iterrows():
            category, gender = infer_category_and_gender(mrow["relpath"])
            subject_id = f"{gender}_{Path(mrow['filename']).stem}"
            inferred.append({
                "video": mrow["filename"],
                "subject_id": subject_id,
                "t_start_sec": 0.0,
                "t_end_sec": float(mrow["duration_sec"]),
                "label": category,
            })
        labels_df = pd.DataFrame(inferred)

    # Align label times to frame indices using file metadata
    meta_by_filename = meta_df.set_index("filename")
    aligned_rows = []

    for _, lrow in labels_df.iterrows():
        filename = Path(str(lrow["video"])).name
        if filename not in meta_by_filename.index:
            # label references a file we didn't scan
            continue

        # Handle duplicate basenames by picking the first row deterministically
        file_meta = meta_by_filename.loc[filename]
        if isinstance(file_meta, pd.DataFrame):
            file_meta = file_meta.iloc[0]

        fps = float(file_meta.get("fps", 0.0)) or FALLBACK_FPS
        duration = float(file_meta.get("duration_sec", 0.0))
        total_frames = int(file_meta.get("frames", 0))

        # Clip label times to [0, duration]
        t0 = max(0.0, float(lrow["t_start_sec"]))
        t1 = min(duration, max(t0, float(lrow["t_end_sec"])))

        # Map times to frame indices; ensure at least one frame
        f0 = int(t0 * fps)
        if total_frames <= 0:
            total_frames = int(round(duration * fps))
        f1 = max(f0 + 1, min(int(round(t1 * fps)), total_frames))

        aligned_rows.append({
            "filename": filename,
            "subject_id": lrow["subject_id"],
            "label": lrow["label"],
            "t_start_sec": t0,
            "t_end_sec": t1,
            "f_start": f0,
            "f_end": f1,
        })

    aligned_df = pd.DataFrame(aligned_rows)
    aligned_df = _merge_overlapping_segments(aligned_df)

    # Join per-file metadata with aligned labels
    table = meta_by_filename.reset_index().merge(aligned_df, on="filename", how="left")

    # ---------- Standardization flags (does not re-encode here) ----------
    def _needs_resize(w: int, h: int) -> bool:
        if int(w) <= 0 or int(h) <= 0:
            return False
        short_edge = min(int(w), int(h))
        return short_edge not in (720, 1080)     # tweak if your policy differs

    table["needs_fps_fix"] = (table["fps"].round(3) != TARGET_FPS)
    table["needs_resize"] = table.apply(lambda r: _needs_resize(r["width"], r["height"]), axis=1)
    table["standardize"] = table[["needs_fps_fix", "needs_resize"]].any(axis=1)

    # Suggest a standardized output path (used if you enable --standardize)
    table["std_relpath"] = table["relpath"].apply(lambda p: str(STANDARDIZED_DIR / Path(p).name))

    return table


def split_by_subject(table: pd.DataFrame, seed: int = RANDOM_SEED) -> pd.DataFrame:
    """Assign each subject_id to train/val/test without leakage."""
    rng = random.Random(seed)

    subjects = sorted(table["subject_id"].dropna().unique().tolist())
    rng.shuffle(subjects)

    n = len(subjects)
    n_train = int(round(n * TRAIN_RATIO))
    n_val = int(round(n * VAL_RATIO))
    train_subj = set(subjects[:n_train])
    val_subj = set(subjects[n_train:n_train + n_val])
    test_subj = set(subjects[n_train + n_val:])

    def in_group(s: set):
        return table["subject_id"].isin(s)

    table.loc[in_group(train_subj), "split"] = "train"
    table.loc[in_group(val_subj), "split"] = "val"
    table.loc[in_group(test_subj), "split"] = "test"

    # Safety: the same subject cannot appear in multiple splits
    check = table.dropna(subset=["subject_id"])[["subject_id", "split"]].drop_duplicates()
    group_sizes = check.groupby("subject_id")["split"].nunique()
    assert (group_sizes <= 1).all(), "Subject leak detected across splits!"

    return table


def reencode_to_standard(
    src_path: str,
    dst_path: str,
    fps: float = TARGET_FPS,
    short_edge: int = TARGET_SHORT_EDGE,
) -> None:
    """
    Optional: actually re-encode using ffmpeg (requires ffmpeg on PATH).
    - sets FPS
    - scales so the shorter image edge == short_edge (keeps aspect ratio)
    - outputs H.264/AAC MP4 ready for streaming
    """
    os.makedirs(Path(dst_path).parent, exist_ok=True)

    # ffmpeg command (arguments explained below in the walkthrough)
    cmd = [
        "ffmpeg", "-y", "-i", src_path,
        "-r", str(fps),
        "-vf", f"scale='if(gte(iw,ih),-2,{short_edge})':'if(gte(iw,ih),{short_edge},-2)'",
        "-c:v", "libx264", "-preset", "medium", "-crf", "23",
        "-c:a", "aac", "-movflags", "+faststart",
        dst_path
    ]
    subprocess.run(cmd, check=True)


def maybe_standardize(table: pd.DataFrame, do_standardize: bool) -> pd.DataFrame:
    """
    If do_standardize=True, re-encode rows marked 'standardize' and update metadata.
    Otherwise, return table unchanged.
    """
    if not do_standardize:
        return table

    STANDARDIZED_DIR.mkdir(parents=True, exist_ok=True)

    to_fix = table[table["standardize"] == True].copy()
    for idx, row in to_fix.iterrows():
        try:
            src = row["relpath"]
            dst = row["std_relpath"]
            reencode_to_standard(src, dst, fps=TARGET_FPS, short_edge=TARGET_SHORT_EDGE)

            # Re-probe the new file and update the row
            q = probe_video(Path(dst))
            for col in ["relpath", "fps", "frames", "duration_sec", "width", "height"]:
                table.loc[idx, col] = q[col]

            short_edge = min(int(q["width"]), int(q["height"]))
            table.loc[idx, "needs_fps_fix"] = (q["fps"] != TARGET_FPS)
            table.loc[idx, "needs_resize"] = (short_edge not in (720, 1080))
            table.loc[idx, "standardize"] = False
        except Exception as e:
            print(f"[WARN] standardize failed for {row['relpath']}: {e}")

    return table


def run(project_data_dir: Path, do_standardize: bool = False) -> None:
    """End-to-end pipeline that writes one consolidated CSV."""
    raw_dir = project_data_dir / "raw_datasets"
    processed_dir = project_data_dir / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    table = build_table(raw_dir)                     # QC + labels + alignment + std flags
    table = maybe_standardize(table, do_standardize) # optional re-encode + refresh QC
    table = split_by_subject(table)                  # subject-wise split
    table = table.drop_duplicates(                   # one row per (file, subject, label)
        subset=["filename", "subject_id", "label"]
    )

    out_csv = processed_dir / "dataset_index.csv"
    table.to_csv(out_csv, index=False)
    print(f"Wrote {out_csv} with {len(table)} rows. Standardize run: {do_standardize}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data", help="project data folder (contains raw/ and processed/)")
    parser.add_argument("--standardize", action="store_true", help="actually re-encode off-spec videos with ffmpeg")
    args = parser.parse_args()
    run(Path(args.data), do_standardize=args.standardize)
 

