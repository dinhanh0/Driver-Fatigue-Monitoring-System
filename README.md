## 1) Setup


# Python 3.10+ (we used 3.11)
python -m venv .venv311
source .venv311/bin/activate  # on Windows: .venv311\Scripts\activate
pip install -r requirements.txt


Optional (Apple Silicon):

export PYTORCH_ENABLE_MPS_FALLBACK=1


Folder layout (created automatically by the scripts below):

data/
  raw/
    ddd/               # Driver Drowsiness Dataset (DDD)
    nitymed/           # NITYMED DSM_Dataset-HDTV720
    sust_ddd/          # SUST Driver Drowsiness Dataset
  index/
  processed/
runs/
checkpoints/


---

## 2) Datasets & expected labels

Download sources (examples):

- DDD: https://www.kaggle.com/datasets/ismailnasri20/driver-drowsiness-dataset-ddd  
- NITYMED: https://www.kaggle.com/datasets/nikospetrellis/nitymed  
- SUST DDD: https://www.kaggle.com/datasets/esrakavalci/sust-ddd

Place videos under:


data/raw/ddd/Driver Drowsiness Dataset (DDD)/{Drowsy,Non Drowsy}/...
data/raw/nitymed/DSM_Dataset-HDTV720/{Microsleep,Yawning}/{Male,Female}/HDTV720/*.mp4
data/raw/sust_ddd/SUST Driver Drowsiness Dataset/{drowsiness,not drowsiness}/*.mp4


**Binary label mapping we use:**

- **DDD** → pos/drowsy = `3`, neg/normal = `5` (the repo’s internal mapping has many classes; we pass ids at train time).  
- **NITYMED** → `1 = drowsy`, `0 = non‑drowsy` (both *Microsleep* and *Yawning* are treated as **drowsy**).  
- **SUST** → parent folder name: `drowsiness → 1`, `not drowsiness → 0`.

---

## 3) Build indexes

You can use the high‑level entrypoint:


# creates CSVs in data/index/
python src/main.py --build-index


Or, if you prefer explicit per‑dataset steps, run the small helpers below.

### 3.1 DDD index (binary)


python - <<'PY'
import pandas as pd, pathlib, re

rows = []
root = pathlib.Path("data/raw/ddd/Driver Drowsiness Dataset (DDD)")
for cls_dir in ("Drowsy","Non Drowsy"):
    for p in (root/cls_dir).rglob("*.mp4"):
        rows.append({"video_path": str(p), "dataset":"DDD",
                     "label": 3 if "Drowsy" in str(p) else 5})

df = pd.DataFrame(rows)
df.to_csv("data/index/ddd_only.csv", index=False)
print("WROTE data/index/ddd_only.csv rows=", len(df))
PY


### 3.2 NITYMED index (treat Microsleep/Yawning as drowsy=1)


python - <<'PY'
import pandas as pd, pathlib

root = pathlib.Path("data/raw/nitymed/DSM_Dataset-HDTV720")
rows = []
for cls in ("Microsleep","Yawning"):
    for sex in ("Male","Female"):
        for p in (root/cls/sex/"HDTV720").rglob("*.mp4"):
            rows.append({"video_path": str(p), "dataset":"NITYMED",
                         "label": 1})
df = pd.DataFrame(rows)
df.to_csv("data/index/nitymed_only.csv", index=False)
print("WROTE data/index/nitymed_only.csv rows=", len(df))
PY


### 3.3 SUST index (drowsiness=1, not drowsiness=0)


python - <<'PY'
import pandas as pd, pathlib

root = pathlib.Path("data/raw/sust_ddd/SUST Driver Drowsiness Dataset")
rows = []
for p in (root).rglob("*.mp4"):
    lab = 1 if "drowsiness" in str(p).lower() else 0
    rows.append({"video_path": str(p), "dataset":"SUST", "label": lab})
df = pd.DataFrame(rows)
df.to_csv("data/index/sust_only.csv", index=False)
print("WROTE data/index/sust_only.csv rows=", len(df))
PY


---

## 4) Preprocess (windowing + features → NPZ)

High‑level:


export PREPROCESS_WORKERS=4
python src/main.py --preprocess


Low‑level (explicit):


# DDD (2s windows / 2s stride / 128px frames)
python -m src.preprocess \
  --index-csv data/index/ddd_only.csv \
  --out-dir   data/processed/windows_ddd_2s2_128 \
  --win-sec 2 --stride-sec 2 --img-size 128 --sample-rate 1 \
  --skip-existing


Repeat for NITYMED and SUST with their index CSVs and different output folders, e.g.:


python -m src.preprocess \
  --index-csv data/index/nitymed_only.csv \
  --out-dir   data/processed/windows_nitymed_2s2_128 \
  --win-sec 2 --stride-sec 2 --img-size 128 --sample-rate 1 \
  --skip-existing

python -m src.preprocess \
  --index-csv data/index/sust_only.csv \
  --out-dir   data/processed/windows_sust_2s2_128 \
  --win-sec 2 --stride-sec 2 --img-size 128 --sample-rate 1 \
  --skip-existing


**What preprocessing does**

- Extracts **video windows** (default 2s) and rescales frames to **128×128**.  
- Computes per‑frame embeddings with a CNN backbone later during training.  
- Computes **heuristics** per frame (EAR, MAR, blur, illumination).  
- Packs each window into a compressed `.npz` with keys: `X_video (T,3,128,128)`, `X_feats (T,4)`, `y`, `fps`, `start`, `end`, …  
- You can smoke‑test with `--max-files 6`.

---

## 5) Train/Val/Test split (80/10/10)

Many preprocess folders start with everything under `train/`. Use this helper to split:


python - <<'PY'
import numpy as np, pathlib, random, shutil, collections, os
random.seed(1337)

ROOT = pathlib.Path("data/processed/windows_sust_2s2_128")  # change per dataset
VAL_FRAC = 0.10
TEST_FRAC = 0.10

def move_some(src_dir, dst_dir, per_class_take):
    dst_dir.mkdir(parents=True, exist_ok=True)
    taken = 0
    for y, n in per_class_take.items():
        ys = [p for p in src_dir.rglob("*.npz") if int(np.load(p)["y"])==y]
        random.shuffle(ys)
        for p in ys[:n]:
            q = dst_dir/p.name
            try:
                os.symlink(os.path.relpath(p, q.parent), q)
            except OSError:
                shutil.copy2(p, q)
            taken += 1
    return taken

train_dir = ROOT/"train"
files = [p for p in train_dir.rglob("*.npz")]
counts = collections.Counter(int(np.load(p)["y"]) for p in files)
print("Found labels in train:", sorted(counts))
print(counts)

take_val = {y: int(n*VAL_FRAC) for y,n in counts.items()}
take_tst = {y: int(n*TEST_FRAC) for y,n in counts.items()}

move_some(train_dir, ROOT/"val",  take_val)
move_some(train_dir, ROOT/"test", take_tst)

def cnt(d): 
    import collections, numpy as np
    c = collections.Counter(int(np.load(p)["y"]) for p in d.rglob("*.npz"))
    return dict(c), len(list(d.rglob("*.npz")))

print("train:", *cnt(ROOT/"train"))
print("  val:", *cnt(ROOT/"val"))
print(" test:", *cnt(ROOT/"test"))
PY


---

## 6) (Optional) Combine datasets

Example: **SUST + NITYMED** into a single root (keeps 0/1 labels):


python - <<'PY'
import pathlib, os, shutil, numpy as np, json, collections
OUT   = pathlib.Path("data/processed/windows_sust_nitymed_2s2_128")
SUST  = pathlib.Path("data/processed/windows_sust_2s2_128")
NITY  = pathlib.Path("data/processed/windows_nitymed_2s2_128")
OUT.mkdir(parents=True, exist_ok=True)

def link_all(src, split):
    dest = OUT/split
    dest.mkdir(parents=True, exist_ok=True)
    for p in (src/split).rglob("*.npz"):
        q = dest/p.name
        try: os.symlink(os.path.relpath(p, q.parent), q)
        except OSError: shutil.copy2(p, q)

for s in ("train","val","test"):
    link_all(SUST, s); link_all(NITY, s)

with open(OUT/"label_mapping.json","w") as f:
    json.dump({"non_drowsy":0, "drowsy":1}, f, indent=2)
PY


---


## 7) Training

### 7A) Quick smoke test (recommended during development)

Use the lightweight **smoke test** to sanity‑check data loading, padding, class balance, loss wiring, and logging—without running a full training job.

**Run (module form, if the file lives at `src/smoke_test.py`):**

python -m src.smoke_test \
  --data-root data/processed/windows_sust_nitymed_2s2_128 \
  --train-split train --val-split val \
  --pos-label-ids 1 --neg-label-ids 0 \
  --time-len 60 \
  --limit-train-per-class 32 \
  --limit-val-per-class 32 \
  --weighted-loss --balanced-sampler \
  --epochs 1 --batch-size 4 --workers 0 \
  --seed 1337 --progress \
  --log-file runs/smoke.log


**Alternative (script form, if the file is at repo root as `smoke_test.py`):**

python smoke_test.py \
  --data-root data/processed/windows_sust_nitymed_2s2_128 \
  --train-split train --val-split val \
  --pos-label-ids 1 --neg-label-ids 0 \
  --time-len 60 \
  --limit-train-per-class 32 \
  --limit-val-per-class 32 \
  --weighted-loss --balanced-sampler \
  --epochs 1 --batch-size 4 --workers 0 \
  --seed 1337 --progress \
  --log-file runs/smoke.log


**What this does**
- Picks a **balanced subset** (32 per class for train and val) so each minibatch sees both classes.
- Applies `--time-len 60` so variable‑length windows are **padded/trimmed** consistently at load time (prevents tensor stacking errors).
- Uses `--workers 0` for clear error traces during debugging.
- Enables `--weighted-loss` + `--balanced-sampler` to stabilize training on small, imbalanced samples.
- Runs **1 epoch** end‑to‑end with logs you can tail in real time:
  
  tail -f runs/smoke.log
  

**Single‑dataset variants**
- SUST only: change `--data-root` to `data/processed/windows_sust_2s2_128`.
- NITYMED only: change `--data-root` to `data/processed/windows_nitymed_2s2_128`.
- DDD (if preprocessed to 0/1): set `--pos-label-ids 1 --neg-label-ids 0` accordingly (or keep original 3/5 when using multi‑id mapping).

**If `smoke_test.py` isn’t available yet**, you can emulate the same behavior with the main trainer:

python -m src.train_model2 \
  --data-root data/processed/windows_sust_nitymed_2s2_128 \
  --train-split train --val-split val \
  --epochs 1 --batch-size 4 --workers 0 \
  --binary --pos-label-ids 1 --neg-label-ids 0 \
  --weighted-loss --balanced-sampler \
  --limit-train-per-class 32 \
  --limit-val-per-class 32 \
  --time-len 60 \
  --seed 1337 --progress \
  --log-file runs/smoke_emulated.log


### 7B) Full training
High‑level:


python src/main.py --train


Direct:


mkdir -p runs checkpoints

# Example: train on combined SUST+NITYMED (binary: pos=1, neg=0)
python -m src.train_model2 \
  --data-root data/processed/windows_sust_nitymed_2s2_128 \
  --train-split train --val-split val \
  --epochs 20 --batch-size 4 --workers 2 \
  --binary --pos-label-ids 1 --neg-label-ids 0 \
  --weighted-loss --balanced-sampler \
  --early-stop-patience 6 --early-stop-min-delta 0.001 \
  --seed 1337 --progress \
  --log-file runs/combined_train.log


**Useful flags**

- `--freeze-backbone-epochs N` (e.g., 2) to warm up the head.  
- `--time-len 60` to **pad/trim** windows at load time (avoid variable‑length errors).  
- `--limit-train-per-class K` / `--limit-val-per-class K` for **balanced smoke tests**.  
- `--limit N`: hard cap on number of samples for quick runs.  
- `--weighted-loss` + `--balanced-sampler`: handle class imbalance.

Track progress:


tail -f runs/combined_train.log


---

## 8) Evaluation (quick sanity check)


# reuse the training entry to evaluate on the test split
python -m src.train_model2 \
  --data-root data/processed/windows_sust_nitymed_2s2_128 \
  --train-split test --val-split test \
  --epochs 1 --batch-size 4 --workers 0 \
  --binary --pos-label-ids 1 --neg-label-ids 0 \
  --weighted-loss \
  --early-stop-patience 1 --early-stop-min-delta 1e9 \
  --seed 1337 --progress --log-file runs/eval_on_test.log


---

## 9) Model‑2 (HMGRU‑TNA) in words

- Per‑frame images flow through a shared **Backbone CNN** → embeddings.  
- A **visual BiGRU** models short/long motion.  
- Hand‑crafted heuristics (EAR/MAR/blur/illum) are normalized → **Feature MLP**.  
- **Gated Fusion** learns how much to trust each stream at each time.  
- Post‑fusion **BiGRU** + **Temporal Neighborhood Attention** (local self‑attention along time) refine context.  
- **Classifier** (**FC → BN → Dropout → FC(2)**) outputs drowsy logits.

The code lives in `src/models/model2_hmgru_tna.py`.

---

## 10) Troubleshooting

- `RuntimeError: stack expects each tensor to be equal size`  
  → Use `--time-len 60` (or consistent windowing in preprocessing).

- DataLoader worker crash during debugging  
  → Set `--workers 0` for a clearer stack trace.

- All predictions are one class on tiny val sets  
  → Use `--balanced-sampler` and/or `--limit-*-per-class` for smoke tests.

---

## 11) Acknowledgments

Datasets: DDD, NITYMED, and SUST (links above).  
This repo: https://github.com/zachkhan99/Facial-Recognition-Driver.git