"""Save SID-Set pictures as files, for scripts that take image lists (columns path, label, source).

Real photos (OpenImages, label 0) are used as sources for the Muse training edits (scripts/make_muse_edits.py);
real photos and fully synthetic FLUX pictures (label 1) together feed the LayerNorm-tuning
(src/models/tune_layernorm.py), which reads image files, not Parquet. The file bytes are written unchanged
(no re-saving; real = JPEG, FLUX = PNG). The pictures go to D:/Datasets (never git).

Usage (from the repo root):
  python scripts/export_sid_images.py D:/Datasets/SID_Set/data/train-0002*-of-00249.parquet
      --out-dir D:/Datasets/SID_Set/real_export --list features/lists/sid_reals_20_29.csv --per-file 150
  python scripts/export_sid_images.py D:/Datasets/SID_Set/data/train-0001[0-4]-of-00249.parquet --labels 0 1
      --out-dir D:/Datasets/SID_Set/export_10_14 --list features/lists/sid_10_14.csv --per-file 9999
"""

import argparse
import csv
import glob
from pathlib import Path

import pyarrow.parquet as pq

SOURCES = {0: "sid_set/openimages", 1: "sid_set/flux"}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("inputs", nargs="+", help="SID-Set Parquet files (globs allowed)")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--list", type=Path, required=True)
    parser.add_argument("--labels", type=int, nargs="+", default=[0], help="0 = real, 1 = fully synthetic (FLUX)")
    parser.add_argument("--per-file", type=int, default=150, help="pictures per label and Parquet file")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for path in sorted({f for pattern in args.inputs for f in glob.glob(pattern)}):
        saved = {label: 0 for label in args.labels}
        for batch in pq.ParquetFile(path).iter_batches(batch_size=64, columns=["img_id", "label", "image"]):
            for row in batch.to_pylist():
                label = row["label"]
                if label not in saved or saved[label] >= args.per_file:
                    continue
                data = row["image"]["bytes"]
                out = args.out_dir / (row["img_id"] + (".png" if data[:4] == b"\x89PNG" else ".jpg"))
                out.write_bytes(data)
                rows.append({"path": str(out), "label": label, "source": SOURCES[label]})
                saved[label] += 1
            if all(n >= args.per_file for n in saved.values()):
                break
        print(f"{Path(path).name}: {saved}")
    with open(args.list, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["path", "label", "source"], lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"{len(rows)} pictures -> {args.out_dir}, list {args.list}")


if __name__ == "__main__":
    main()
