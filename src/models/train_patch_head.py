"""Train Path B's head: a linear head on single patch features (src/features/extract_patches.py).

The same logistic regression as the whole-image head (src/models/linear_head.py), but every patch is
one example: 1 = inside an AI-edited area, 0 = not edited. Classes get equal weight.

Usage (from the repo root):
  python -m src.models.train_patch_head --train "features/patches/*_train.npz" --out weights/path_b/head_B.pt
"""

import argparse
import glob
from pathlib import Path

import numpy as np
import torch

from src.models.linear_head import LinearHead


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--train", nargs="+", required=True, help="patch feature files (globs allowed)")
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    files = sorted({f for pattern in args.train for f in glob.glob(pattern)})
    parts = [np.load(f) for f in files]
    x = torch.from_numpy(np.concatenate([p["features"] for p in parts]).astype(np.float32))
    y = torch.from_numpy(np.concatenate([p["labels"] for p in parts]).astype(np.float32))
    print(f"{len(files)} files: {len(y)} patches, {int(y.sum())} edited")
    head = LinearHead().fit(x, y, weight_decay=args.weight_decay)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    head.save(args.out)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
