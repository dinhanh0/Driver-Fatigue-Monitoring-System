import csv, os, argparse, random, collections
from pathlib import Path

BASE_COLS = [
    "media_type","video_path","images_dir","dataset","subject_id",
    "label","fps","frames","split_hint","split"
]

def read_rows(csv_path):
    with open(csv_path) as f:
        rdr = csv.DictReader(f)
        return list(rdr)

def write_rows(rows, out_csv):
    # Preserve known columns first, then append any extras we saw
    cols, seen = [], set()
    for c in BASE_COLS:
        if c not in seen:
            cols.append(c); seen.add(c)
    for r in rows:
        for k in r.keys():
            if k not in seen:
                cols.append(k); seen.add(k)
    os.makedirs(Path(out_csv).parent, exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in cols})

def main():
    ap = argparse.ArgumentParser(
        description="Split an index CSV into train/val/test splits."
    )
    ap.add_argument("--in-csv", type=str, required=True,
                    help="Path to input index CSV (e.g., data/index/nitymed_all.csv)")
    ap.add_argument("--out-csv", type=str, required=True,
                    help="Path to output split CSV (e.g., data/index/nitymed_split.csv)")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--train", type=float, default=0.7,
                    help="Proportion for train split")
    ap.add_argument("--val",   type=float, default=0.15,
                    help="Proportion for val split; test is 1-train-val")
    args = ap.parse_args()

    rows = read_rows(args.in_csv)
    rng = random.Random(args.seed)

    # group by (dataset, label) to approximately stratify
    buckets = collections.defaultdict(list)
    for r in rows:
        key = (r.get("dataset",""), r.get("label",""))
        buckets[key].append(r)

    out = []
    for key, items in buckets.items():
        rng.shuffle(items)
        n = len(items)
        n_tr = int(n * args.train)
        n_va = int(n * args.val)
        for i, r in enumerate(items):
            if i < n_tr:
                r["split"] = "train"
            elif i < n_tr + n_va:
                r["split"] = "val"
            else:
                r["split"] = "test"
            out.append(r)

    write_rows(out, args.out_csv)
    print(f"Wrote {len(out)} rows -> {args.out_csv}")
    from collections import Counter
    ctr = Counter((r["split"], r.get("dataset",""), r.get("label","")) for r in out)
    for k, v in sorted(ctr.items()):
        print(k, v)

if __name__ == "__main__":
    main()