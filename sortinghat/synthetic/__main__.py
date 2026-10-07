"""python -m sortinghat.synthetic --out data/synthetic [--n 3000] [--seed N]

Writes the REAL HEEDB layout (EEG/..., OMOP/Merged/<table>/*.parquet) under ``--out``. Needs pyarrow."""

import argparse

from . import generate, write_tables


def main() -> int:
    ap = argparse.ArgumentParser(description="Generate synthetic HEEDB-shaped tables (fake data).")
    ap.add_argument("--out", default="data/synthetic")
    ap.add_argument("--n", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=20260101)
    a = ap.parse_args()
    tables, truth = generate(a.n, a.seed)
    paths = write_tables(tables, a.out, truth)
    print(f"wrote {len(paths)} synthetic table files to {a.out} (seed={a.seed})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
