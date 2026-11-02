"""Face crops as image files, for fine-tuning a face-swap model (src/models/tune_face_pe.py) and testing it.

Every face is found by SAM 3 with the app's settings (src/faces/sam_faces.py) and cut with the app's crop rule
(src/faces/crop.py), so training faces and app faces look alike. Two kinds of crop:
  - "wide" (training): 5 eye distances around the face at 420 x 420 (one eye distance = 84 px), JPEG 95, plus SAM's
    face mask cut the same way (PNG). Training takes a random sub-crop of 2.6-4.0 eye distances from it (the same range
    as crop.train_face_view) and may blend the face onto itself (self-blended fakes, needs the mask).
  - "app" (testing): exactly the app's crop, 4 eye distances, at 336 x 336 (the Perception Encoder's input), JPEG 90.

Sources (one manifest CSV per run: path, mask, label, source, group, kind):
  of     OpenForensics photos (scripts/download_openforensics.py) with each SAM face labelled by the annotated box it
         overlaps (src/features/extract_openforensics_faces.py). Val: 15% of the photos (by name) are the dev part
         (app crops), the rest train (wide crops). Test-Dev: app crops only.
  list   CSV image lists (path, label, source): every face gets the photo's label (real photo lists: all faces real)
  dfdc   DFDC parts: real video frames and the same frames of up to 2 fakes made from them (the same face boxes;
         fake crops kept only if the face really changed: src/features/extract_dfdc_faces.py)

Usage (from the repo root):
  python -m src.faces.make_crops of --split Val --out D:/Datasets/FaceCrops
  python -m src.faces.make_crops list features/lists/commons_dev.csv ... --kind app --out D:/Datasets/FaceCrops
  python -m src.faces.make_crops dfdc D:/Datasets/DFDC/dfdc_train_part_0 ... --kind wide --name dfdc_train --out ...
"""

import argparse
import csv
import json
import random
import time
import zlib
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from src.data import sid_set
from src.faces import crop
from src.faces.sam_faces import FaceFinder

WIDE_K, WIDE_SIZE = 5.0, 420  # one eye distance = 84 px in a wide crop
APP_SIZE = 336  # the Perception Encoder's input
FIELDS = ["path", "mask", "label", "source", "group", "kind"]


class Writer:
    """Saves crops under out/name/ and collects the manifest rows."""

    def __init__(self, out, name):
        self.dir = out / name
        self.dir.mkdir(parents=True, exist_ok=True)
        self.manifest = out / f"{name}.csv"
        self.rows, self.count = [], 0

    def add(self, image, face, label, source, group, kind, mask=None):
        self.count += 1
        stem = self.dir / f"{self.count:06d}"
        if kind == "wide":
            box = tuple(round(v) for v in crop.crop_box(face, k=WIDE_K))
            image.crop(box).resize((WIDE_SIZE, WIDE_SIZE), Image.BICUBIC).save(f"{stem}.jpg", quality=95)
            mask_path = ""
            if mask is not None and mask.any():
                Image.fromarray(mask.astype(np.uint8) * 255).crop(box).resize((WIDE_SIZE, WIDE_SIZE), Image.NEAREST) \
                    .save(f"{stem}_mask.png")
                mask_path = f"{stem}_mask.png"
        else:  # the app's crop at the PE's size, JPEG 90 (the file is the JPEG-90 view)
            crop.face_crop(image, face, size=APP_SIZE).save(f"{stem}.jpg", quality=90)
            mask_path = ""
        self.rows.append({"path": f"{stem}.jpg", "mask": mask_path, "label": int(label), "source": source,
                          "group": group, "kind": kind})

    def save(self):
        with open(self.manifest, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDS, lineterminator="\n")
            writer.writeheader()
            writer.writerows(self.rows)
        print(f"{len(self.rows)} crops -> {self.manifest}", flush=True)


def run_of(args, finder):
    from src.features.extract_openforensics_faces import annotated_faces, match
    truth_of = annotated_faces(args.data_dir / f"{args.split}_poly.json")
    with open(f"features/lists/openforensics_{args.split}.csv", newline="") as f:
        paths = [Path(r["path"]) for r in csv.DictReader(f)][: args.max_images]
    tag = args.split.lower().replace("-", "_")
    train, dev = Writer(args.out, f"of_{tag}_train"), Writer(args.out, f"of_{tag}_dev" if args.split == "Val" else f"of_{tag}")
    start = time.time()
    for n, path in enumerate(paths, 1):
        width, height, truth = truth_of[path.name]
        with Image.open(path) as im:
            image = im.convert("RGB")  # not turned upright: the annotation boxes refer to the stored pixels
        if image.size != (width, height):
            continue
        # Val: 15% of the photos are the dev part; Train and Test-Challenge (perturbed copies, used for TRAINING here):
        # all train; Test-Dev (our test): all app crops
        is_dev = args.split == "Test-Dev" or (args.split == "Val" and zlib.crc32(path.name.encode()) % 100 < 15)
        found = finder.find(image, with_masks=not is_dev)
        for i, is_fake, _ in match(found, truth, 0.3):
            face = found[i]
            if face["too_small"]:
                continue
            source = "openforensics/" + ("fake" if is_fake else "real")
            if is_dev:
                dev.add(image, face, is_fake, source, path.name, "app")
            else:
                train.add(image, face, is_fake, source, path.name, "wide", face.get("mask"))
        if n % 500 == 0:
            print(f"  {n}/{len(paths)} photos, {train.count} train + {dev.count} dev/test crops, {time.time() - start:.0f} s",
                  flush=True)
    for w in (train, dev):
        if w.rows:
            w.save()


def images_of(path, max_images):
    """(image, group) for a CSV list (turned upright, as the app sees it) or a SID-Set parquet file (its real photos)."""
    if path.suffix == ".parquet":
        for n, row in enumerate(sid_set.iter_rows(path, labels=(0,))):
            if max_images and n >= max_images:
                return
            yield row["image"], str(row["img_id"]), 0, "sid_set/real"
        return
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))[:max_images]
    for row in rows:
        try:
            with Image.open(row["path"]) as im:
                image = ImageOps.exif_transpose(im).convert("RGB")
        except Exception:
            continue
        yield image, row["path"], int(row["label"] != "0"), row["source"]


def run_list(args, finder):
    held_paths, held_ids = set(), set()
    for spec in args.exclude:  # photos that are in a test or calibration pool must not become training faces
        with open(spec, newline="") as f:
            for r in csv.DictReader(f):
                held_paths.add(r["path"].replace("\\", "/").lower())
                held_ids.add(Path(r["path"].replace("\\", "/")).stem)  # SID-Set photos: the OpenImages id
    writer = Writer(args.out, args.name)
    start, n = time.time(), 0
    for path in args.inputs:
        for image, group, label, source in images_of(path, args.max_images):
            if (path.suffix == ".parquet" and group in held_ids) or str(group).replace("\\", "/").lower() in held_paths:
                continue
            n += 1
            for face in finder.find(image, with_masks=args.kind == "wide"):
                if not face["too_small"]:
                    writer.add(image, face, label, source, group, args.kind, face.get("mask"))
            if n % 500 == 0:
                print(f"  {n} photos, {writer.count} crops, {time.time() - start:.0f} s", flush=True)
    writer.save()


def run_dfdc(args, finder):
    from src.features.extract_dfdc_faces import changed, read_frames
    writer = Writer(args.out, args.name)
    rng = random.Random(args.seed)
    start = time.time()
    for part in args.inputs:
        meta = json.loads((part / "metadata.json").read_text())
        fakes_of = defaultdict(list)
        for name, info in meta.items():
            if info["label"] == "FAKE":
                fakes_of[info["original"]].append(name)
        reals = sorted(name for name, info in meta.items() if info["label"] == "REAL")[: args.max_images]
        for n, real in enumerate(reals, 1):
            fakes = rng.sample(sorted(fakes_of[real]), min(2, len(fakes_of[real])))
            real_frames = read_frames(part / real, 6)
            fake_frames = {f: read_frames(part / f, 6) for f in fakes}
            for index, real_img in real_frames.items():
                for face in (f for f in finder.find(real_img, with_masks=args.kind == "wide") if not f["too_small"]):
                    writer.add(real_img, face, 0, "dfdc/real", real, args.kind, face.get("mask"))
                    for fake in fakes:
                        fake_img = fake_frames[fake].get(index)
                        if fake_img is not None and fake_img.size == real_img.size and changed(real_img, fake_img, face)[0]:
                            writer.add(fake_img, face, 1, "dfdc/fake", real, args.kind)
            if n % 50 == 0:
                print(f"  {part.name}: {n}/{len(reals)} real videos, {writer.count} crops, {time.time() - start:.0f} s",
                      flush=True)
    writer.save()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mode", choices=["of", "list", "dfdc"])
    parser.add_argument("inputs", nargs="*", type=Path, help="list mode: CSV lists / SID-Set parquet files; dfdc: parts")
    parser.add_argument("--out", type=Path, default=Path("D:/Datasets/FaceCrops"))
    parser.add_argument("--name", help="list/dfdc mode: name of this crop set")
    parser.add_argument("--kind", choices=["wide", "app"], default="wide")
    parser.add_argument("--split", default="Val", help="of mode: Val or Test-Dev")
    parser.add_argument("--data-dir", type=Path, default=Path("D:/Datasets/OpenForensics"))
    parser.add_argument("--exclude", nargs="*", default=[], help="list mode: CSV lists whose photos are skipped")
    parser.add_argument("--max-images", type=int, help="only the first N photos (videos) of each input (quick test)")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    finder = FaceFinder()
    {"of": run_of, "list": run_list, "dfdc": run_dfdc}[args.mode](args, finder)


if __name__ == "__main__":
    main()
