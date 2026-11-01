"""Real face close-ups (head and shoulders, like the GAN faces of SFHQ), cut from real photos with SAM 3.

Why: GAN face datasets are tight, centred face crops (FFHQ style). If the only real pictures are wider photos, a head
can learn "close-up face = AI" and then flag a real close-up selfie (the ImageCLEF "portrait" lesson). So the real side
gets close-ups of the same framing: for every big enough face SAM 3 finds (box at least --min-face px wide), a square
of k = 5..6.5 eye distances around it (src/faces/crop.py), skipped if it would leave the photo, saved as JPEG 95.

Writes the crops to --out-dir/<name>/ and features/lists/real_face_crops_<name>.csv (path, label 0, source).
Training photos give the training crops; dev photos give a dev set that checks for false alarms on real close-ups.

Usage:
  python scripts/make_real_face_crops.py --name train --lists features/lists/commons_train.csv ... --out-dir D:/Datasets/RealFaceCrops
  python scripts/make_real_face_crops.py --name dev --lists features/lists/commons_dev.csv ... --out-dir D:/Datasets/RealFaceCrops
"""

import argparse
import csv
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # run as a script from the repo root

from PIL import Image, ImageOps  # noqa: E402

from src.faces import crop  # noqa: E402
from src.faces.sam_faces import FaceFinder  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--name", required=True, help="train or dev")
    parser.add_argument("--lists", nargs="+", required=True, help="CSV lists of REAL photos (label 0)")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--min-face", type=int, default=200, help="smallest face box width in pixels")
    parser.add_argument("--per-photo", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    finder = FaceFinder()
    out = args.out_dir / args.name
    out.mkdir(parents=True, exist_ok=True)
    rows, photos = [], [r for path in args.lists for r in csv.DictReader(open(path, newline="")) if r["label"] == "0"]
    for n, r in enumerate(photos):
        with Image.open(r["path"]) as im:
            image = ImageOps.exif_transpose(im).convert("RGB")
        faces = [f for f in finder.find(image) if f["box"][2] - f["box"][0] >= args.min_face][: args.per_photo]
        for i, face in enumerate(faces):
            x0, y0, x1, y1 = crop.crop_box(face, k=rng.uniform(5.0, 6.5), shift=(rng.uniform(-0.15, 0.15), rng.uniform(-0.3, 0.1)))
            if x0 < 0 or y0 < 0 or x1 > image.width or y1 > image.height:
                continue  # it would leave the photo (black borders would be a giveaway)
            path = out / f"{Path(r['path']).stem}_{i}.jpg"
            image.crop(tuple(round(v) for v in (x0, y0, x1, y1))).save(path, quality=95)
            rows.append({"path": str(path), "label": 0, "source": f"real_face_crops/{r['source']}"})
        if (n + 1) % 200 == 0:
            print(f"  {n + 1} of {len(photos)} photos, {len(rows)} close-ups", flush=True)
    with open(f"features/lists/real_face_crops_{args.name}.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["path", "label", "source"], lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"{len(rows)} real close-ups from {len(photos)} photos -> features/lists/real_face_crops_{args.name}.csv")


if __name__ == "__main__":
    main()
