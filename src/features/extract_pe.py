"""PE-Core-L features (6 layers x 1,024 numbers) for the Muse-edit check (src/models/pe_core.py).

Views:
  eval   the centre square at 336 x 336, JPEG 90; one per picture (how the edit check is trained)
  three  the app's view: three squares along the long side (start, centre, end), features averaged
  train  random training views at 336 (src/data/preprocess.train_view: random shrink, crop, brightness, JPEG;
         identical for real and AI), --copies per picture; the same copy on every run
Inputs: CSV image lists (path, label, source); --labels keeps only these labels.
Output per list: <out-dir>/<list name>_<mode>.npz with features [N, 6, 1024] (float16), labels, sources, img_ids, copies.

Usage (from the repo root):
  python -m src.features.extract_pe features/lists/commons_train.csv ... --mode train --copies 2 --out-dir features/pe
"""

import argparse
import csv
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps

from src.data import preprocess
from src.features.extract import view_rng
from src.models.pe_core import PECore


class Views(torch.utils.data.Dataset):
    """(picture, copy) -> normalised view; runs in the DataLoader workers."""

    def __init__(self, rows, mode, copies, size, mean, std, seed):
        self.items = [(r, c) for r in rows for c in range(copies if mode == "train" else 1)]
        self.mode, self.size, self.mean, self.std, self.seed = mode, size, mean, std, seed

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        row, copy = self.items[i]
        with Image.open(row["path"]) as im:
            image = ImageOps.exif_transpose(im).convert("RGB")
        if self.mode == "train":
            views = [preprocess.train_view(image, view_rng(row["path"], copy, self.seed), size=self.size)]
        elif self.mode == "three":  # the app's three squares along the long side (src/models/pe_core.PECore.views)
            w, h = image.size
            side = min(w, h)
            views = [preprocess.jpeg(preprocess.square_crop(image, side, round((w - side) * f), round((h - side) * f),
                                                             self.size), 90) for f in (0.0, 0.5, 1.0)]
        else:
            views = [preprocess.eval_view(image, self.size)]
        pixels = torch.stack([torch.from_numpy(np.asarray(v, dtype=np.float32) / 255.0).permute(2, 0, 1) for v in views])
        return (pixels - self.mean) / self.std, i


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("inputs", nargs="+", type=Path, help="CSV image lists")
    parser.add_argument("--mode", choices=["train", "eval", "three"], default="eval",
                        help="three = the app's three squares along the long side, features averaged")
    parser.add_argument("--copies", type=int, default=2, help="random views per picture (train mode)")
    parser.add_argument("--labels", type=int, nargs="+", help="keep only these labels (default: all)")
    parser.add_argument("--out-dir", type=Path, default=Path("features/pe"))
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    pe = PECore()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for path in args.inputs:
        with open(path, newline="") as f:
            rows = [r for r in csv.DictReader(f) if args.labels is None or int(r["label"]) in args.labels]
        data = Views(rows, args.mode, args.copies, pe.size, pe.mean, pe.std, args.seed)
        loader = torch.utils.data.DataLoader(data, batch_size=args.batch_size, num_workers=args.workers)
        feats, order, start = [], [], time.time()
        for pixels, index in loader:  # pixels: [B, views, 3, 336, 336]
            b, v = pixels.shape[:2]
            f = pe.layer_features(pixels.flatten(0, 1)).reshape(b, v, -1, 1024).mean(1)  # average over the views
            feats.append(f.numpy().astype(np.float16))
            order.append(index.numpy())
        items = [data.items[i] for i in np.concatenate(order)]
        out = args.out_dir / f"{path.stem}_{args.mode}.npz"
        np.savez_compressed(out, features=np.concatenate(feats),
                            labels=np.array([int(r["label"]) for r, _ in items], dtype=np.int8),
                            sources=np.array([r["source"] for r, _ in items]),
                            img_ids=np.array([r["path"] for r, _ in items]),
                            copies=np.array([c for _, c in items], dtype=np.int8))
        print(f"{path.name}: {len(rows)} pictures -> {len(items)} views in {time.time() - start:.0f} s -> {out}", flush=True)


if __name__ == "__main__":
    main()
