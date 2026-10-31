"""Save SID-Set real photos (OpenImages, label 0) as files, for scripts that take image lists.

Used for the Muse training edits (scripts/make_muse_edits.py): everyday amateur photos add variety
to the Commons training photos. The file bytes are written unchanged (no re-saving). The photos go to
D:/Datasets (never git); the list gets the columns path, label, source like every image list.

Usage (from the repo root):
  python scripts/export_sid_reals.py D:/Datasets/SID_Set/data/train-0002*-of-00249.parquet
      --out-dir D:/Datasets/SID_Set/real_export --list features/lists/sid_reals_20_29.csv --per-file 150
"""

import argparse
import csv
import glob
from pathlib import Path

import pyarrow.parquet as pq


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("inputs", nargs="+", help="SID-Set Parquet files (globs allowed)")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--list", type=Path, required=True)
    parser.add_argument("--per-file", type=int, default=150, help="real photos per Parquet file")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for path in sorted({f for pattern in args.inputs for f in glob.glob(pattern)}):
        saved = 0
        for batch in pq.ParquetFile(path).iter_batches(batch_size=64, columns=["img_id", "label", "image"]):
            for row in batch.to_pylist():
                if row["label"] != 0 or saved >= args.per_file:
                    continue
                out = args.out_dir / f"{row['img_id']}.jpg"
                out.write_bytes(row["image"]["bytes"])
                rows.append({"path": str(out), "label": 0, "source": "sid_set/openimages"})
                saved += 1
            if saved >= args.per_file:
                break
        print(f"{Path(path).name}: {saved} real photos")
    with open(args.list, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["path", "label", "source"], lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"{len(rows)} photos -> {args.out_dir}, list {args.list}")


if __name__ == "__main__":
    main()
