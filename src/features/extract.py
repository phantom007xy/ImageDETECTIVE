"""Extract DINOv3 average patch vectors for Path A (the whole-image score).

For every image we store the average of its patch features (1,024 numbers, ~2 KB). A linear
head on this average equals the same head applied to every patch and then averaged, so this is
all the frozen baseline needs.

Path A never trains on local edits, so by default only real (0) and fully synthetic (1) images
are used; tampered images (2) are for the edited-region head.

Inputs are SID-Set Parquet files or CSV image lists (columns: path, label, source; see
src/data/folder.py), picked by file extension.

Usage (from the repo root):
  python -m src.features.extract INPUT [INPUT ...] --out-dir features --mode train --copies 4
"""

import argparse
import random
import time
import zlib
from pathlib import Path

import numpy as np
import torch

from src.data import folder, preprocess, sid_set
from src.models.backbone import Backbone


def view_rng(img_id, copy, seed):
    """A random generator that gives the same copy on every run (Python's hash() changes per run)."""
    return random.Random(zlib.crc32(f"{seed}:{img_id}:{copy}".encode()))


def read_input(path, labels):
    """SID-Set Parquet file or CSV image list, with only the wanted labels."""
    if path.suffix == ".csv":
        return (row for row in folder.iter_rows(path) if row["label"] in labels)
    return sid_set.iter_rows(path, labels=labels)


def extract_file(backbone, input_path, out_path, mode, copies, seed, batch_size, labels):
    features, labels_out, img_ids, copy_ids, sources = [], [], [], [], []
    batch = []

    def run_batch():
        if batch:
            features.append(backbone.mean_patch_vector(torch.stack(batch)).cpu().numpy().astype(np.float16))
            batch.clear()

    start, images = time.time(), 0
    for row in read_input(input_path, labels):
        images += 1
        for copy in range(copies if mode == "train" else 1):
            if mode == "train":
                view = preprocess.train_view(row["image"], view_rng(row["img_id"], copy, seed))
            else:
                view = preprocess.eval_view(row["image"])
            batch.append(preprocess.to_tensor(view))
            labels_out.append(row["label"])
            img_ids.append(row["img_id"])
            copy_ids.append(copy)
            sources.append(row["source"])
            if len(batch) == batch_size:
                run_batch()
        if images % 200 == 0:
            print(f"  {images} images, {time.time() - start:.0f} s")
    run_batch()

    np.savez_compressed(
        out_path,
        features=np.concatenate(features),
        labels=np.array(labels_out, dtype=np.int8),
        img_ids=np.array(img_ids),
        copies=np.array(copy_ids, dtype=np.int8),
        sources=np.array(sources),
        mode=mode,
    )
    print(f"{input_path.name}: {images} images -> {len(labels_out)} vectors in {time.time() - start:.0f} s -> {out_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("inputs", nargs="+", type=Path, help="SID-Set .parquet files or .csv image lists")
    parser.add_argument("--out-dir", type=Path, default=Path("features"))
    parser.add_argument("--mode", choices=["train", "eval"], default="train")
    parser.add_argument("--copies", type=int, default=4, help="random copies per image (train mode only)")
    parser.add_argument("--labels", type=int, nargs="+", default=[0, 1], help="0 = real, 1 = synthetic, 2 = tampered")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    backbone = Backbone()
    print(f"DINOv3 on {backbone.device} ({backbone.dtype})")
    for path in args.inputs:
        out_path = args.out_dir / f"{path.stem}_{args.mode}.npz"
        extract_file(backbone, path, out_path, args.mode, args.copies, args.seed, args.batch_size, tuple(args.labels))


if __name__ == "__main__":
    main()
