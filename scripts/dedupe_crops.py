"""Near-duplicate check: no training face may be (almost) the same picture as a dev or test face.

Different datasets can share photos (e.g. FairFace and Open Images both come from Flickr), one photo can be uploaded
twice under different names, and OpenForensics reuses the same base photos in different splits with different fake
faces put in (its real faces then repeat exactly; found 2 Nov). Every face crop (src/faces/make_crops.py) gets a
256-bit difference hash (16 x 17 grey pixels) of its face at the app's framing (a wide 5-eye-distance crop is cut to
its central 4 eye distances; app crops already are). Measured 2 Nov: the nearest dev/test face of a DFDC or Commons
training face is 30+ bits away, while OpenForensics has exact repeats (0-3 bits). Training faces within --distance
bits (default 16) of any dev/test face are removed from their manifest (the full one is kept as
<name>_before_dedup.csv). Also reports how many dev/test faces had such a twin in training before the cleaning.

Usage (from the repo root):
  python scripts/dedupe_crops.py --crops /content/FaceCrops --train of_val_train dfdc_train real_sid ...
      --held pool_commons_dev pool_sid_val of_val_dev of_test_dev dfdc_p3_dev ...
"""

import argparse
import csv
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

WIDE_K, APP_K, N = 5.0, 4.0, 16


def dhash(path, kind):
    with Image.open(path) as im:
        img = im.convert("L")
    if kind == "wide":  # the central 4 of 5 eye distances: the app's framing
        side = img.size[0]
        cut = side * APP_K / WIDE_K
        m = (side - cut) / 2
        img = img.crop((round(m), round(m), round(m + cut), round(m + cut)))
    a = np.asarray(img.resize((N + 1, N), Image.BILINEAR), dtype=np.int16)
    return (a[:, 1:] > a[:, :-1]).flatten()


def load(crops, stem, workers):
    rows = list(csv.DictReader(open(crops / f"{stem}.csv", newline="")))
    with ThreadPoolExecutor(workers) as pool:
        bits = list(pool.map(lambda r: dhash(r["path"], r.get("kind", "app")), rows))
    return rows, np.stack(bits).astype(np.uint8) if bits else np.zeros((0, N * N), np.uint8)


def nearest(a, b, chunk=256):
    """For every row of a: the smallest Hamming distance to any row of b."""
    out = np.empty(len(a), dtype=np.int32)
    for i in range(0, len(a), chunk):
        out[i:i + chunk] = (a[i:i + chunk, None, :] != b[None, :, :]).sum(-1).min(1)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--crops", type=Path, required=True)
    parser.add_argument("--train", nargs="+", required=True)
    parser.add_argument("--held", nargs="+", required=True)
    parser.add_argument("--distance", type=int, default=16, help="of 256 bits")
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()

    held = {s: load(args.crops, s, args.workers)[1] for s in args.held if (args.crops / f"{s}.csv").exists()}
    all_held = np.concatenate(list(held.values()))
    print(f"{len(all_held)} dev/test faces hashed", flush=True)
    trains = {s: load(args.crops, s, args.workers) for s in args.train if (args.crops / f"{s}.csv").exists()}
    all_train = np.concatenate([bits for _, bits in trains.values()])
    for stem, bits in held.items():  # leakage before cleaning, per dev/test set
        twins = int((nearest(bits, all_train) <= args.distance).sum()) if len(bits) else 0
        print(f"  {stem}: {twins} of {len(bits)} faces had a near-twin in training", flush=True)
    for stem, (rows, bits) in trains.items():
        far = nearest(bits, all_held) > args.distance
        if not far.all():
            backup = args.crops / f"{stem}_before_dedup.csv"
            if not backup.exists():
                shutil.copy(args.crops / f"{stem}.csv", backup)
            with open(args.crops / f"{stem}.csv", "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
                w.writeheader()
                w.writerows(r for r, keep in zip(rows, far) if keep)
        print(f"{stem}: {len(rows)} faces, {int((~far).sum())} near-duplicates of dev/test faces removed", flush=True)


if __name__ == "__main__":
    main()
