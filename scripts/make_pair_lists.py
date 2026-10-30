"""Same-content pairs, step 3: match each Muse Image fake to its real photo and write the image lists.

Muse Image returns ~2 MP images; the real photos are smaller. So that size can't tell the classes
apart, every fake is centre-cropped to its real photo's aspect ratio (cropping, never squashing) and
resized to exactly the real photo's size, saved losslessly as PNG in fake_matched/. Preprocessing
then treats both the same way (src/data/preprocess.py).

Writes two CSV lists (path, label, source) in --out-dir: muse_pairs_train.csv and muse_pairs_val.csv.
A real photo and its fake are always in the same list.

Usage:
  python scripts/make_pair_lists.py --pairs-dir D:/Datasets/MuseImage/pairs_v1 --out-dir features/lists
"""

import argparse
import csv
from pathlib import Path

from PIL import Image


def matched_fake(fake_path, width, height, out_path):
    with Image.open(fake_path) as im:
        im = im.convert("RGB")
        target = width / height
        if im.width / im.height > target:  # too wide: crop the sides
            keep = round(im.height * target)
            left = (im.width - keep) // 2
            im = im.crop((left, 0, left + keep, im.height))
        else:  # too tall: crop top and bottom
            keep = round(im.width / target)
            top = (im.height - keep) // 2
            im = im.crop((0, top, im.width, top + keep))
        im.resize((width, height), Image.BICUBIC).save(out_path)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pairs-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("features/lists"))
    args = parser.parse_args()

    with open(args.pairs_dir / "pairs.csv", newline="", encoding="utf-8") as f:
        pairs = {r["pair_id"]: r for r in csv.DictReader(f)}
    with open(args.pairs_dir / "fake" / "manifest.csv", newline="", encoding="utf-8") as f:
        fakes = {r["id"]: r for r in csv.DictReader(f) if r["status"] == "ok"}

    (args.pairs_dir / "fake_matched").mkdir(exist_ok=True)
    lists = {"train": [], "val": []}
    for pair_id, fake in sorted(fakes.items()):
        p = pairs[pair_id]
        out_path = args.pairs_dir / "fake_matched" / f"{pair_id}.png"
        if not out_path.exists():
            matched_fake(args.pairs_dir / "fake" / fake["file"], int(p["width"]), int(p["height"]), out_path)
        lists[p["split"]] += [
            {"path": str(args.pairs_dir / p["real_file"]), "label": 0, "source": "muse_pairs/real"},
            {"path": str(out_path), "label": 1, "source": "muse_pairs/muse"},
        ]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for split, rows in lists.items():
        path = args.out_dir / f"muse_pairs_{split}.csv"
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["path", "label", "source"], lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
        print(f"wrote {len(rows) // 2} pairs ({len(rows)} images) to {path}")


if __name__ == "__main__":
    main()
