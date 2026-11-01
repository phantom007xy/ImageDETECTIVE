"""Nano Banana (Google Gemini 2.5 Flash Image) pictures from BitMind's public dataset (Hugging Face, MIT licence).

Why: Gemini pictures are our weakest popular generator (74% caught, results/commons_ai_v1.txt) and none was in training.
bitmind/nano-banana (https://huggingface.co/datasets/bitmind/nano-banana) has 9,457 of them in 31 parquet files
(1024 x 1024 PNG): landscapes and fantasy scenes in the first files, then people, animals, interiors and objects.
Whole files are split: --train-shards for training, --dev-shards only for testing (never trained on).

Writes the pictures to --out-dir/<shard>/<id>.png (never git) and features/lists/bitmind_nb_train.csv and
bitmind_nb_dev.csv (path, label 1, source bitmind/nano_banana).

Usage:  python scripts/export_bitmind.py --out-dir D:/Datasets/BitMindNB --train-shards 0 5 10 15 20 --dev-shards 25 30
"""

import argparse
import csv
import os
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import pyarrow.parquet as pq  # noqa: E402
from huggingface_hub import hf_hub_download  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--train-shards", type=int, nargs="+", required=True)
    parser.add_argument("--dev-shards", type=int, nargs="+", required=True)
    args = parser.parse_args()

    for name, shards in (("train", args.train_shards), ("dev", args.dev_shards)):
        rows = []
        for shard in shards:
            table = pq.read_table(hf_hub_download("bitmind/nano-banana", f"data/train-{shard:05d}-of-00031.parquet",
                                                  repo_type="dataset"), columns=["id", "image"])
            folder = args.out_dir / f"{shard:02d}"
            folder.mkdir(parents=True, exist_ok=True)
            for pid, image in zip(table.column("id").to_pylist(), table.column("image").to_pylist()):
                path = folder / f"{pid}.png"
                if not path.exists():
                    path.write_bytes(image["bytes"])
                rows.append({"path": str(path), "label": 1, "source": "bitmind/nano_banana"})
            print(f"  shard {shard}: {table.num_rows} pictures", flush=True)
        with open(f"features/lists/bitmind_nb_{name}.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["path", "label", "source"], lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
        print(f"{name}: {len(rows)} pictures -> features/lists/bitmind_nb_{name}.csv")


if __name__ == "__main__":
    main()
