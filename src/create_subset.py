"""Create a subset of dataset_index.csv for fast iteration.

Usage:
    python -m src.create_subset --index data/processed/dataset_index.csv --out data/processed/dataset_index_subset.csv --per_label 10
"""
from pathlib import Path
import argparse
import pandas as pd


def create_subset(index_path: Path, out_path: Path, per_label: int = 10):
    df = pd.read_csv(index_path)
    # Prefer 'label' column, fallback to 'subject_id' grouping
    if 'label' in df.columns:
        groups = df.groupby('label')
    else:
        groups = df.groupby('subject_id')

    keep = []
    for k, g in groups:
        keep.append(g.head(per_label))

    subset = pd.concat(keep, axis=0)
    subset.to_csv(out_path, index=False)
    print(f"Wrote subset with {len(subset)} rows to {out_path}")


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--index', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--per_label', type=int, default=10)
    args = ap.parse_args()
    create_subset(Path(args.index), Path(args.out), args.per_label)
