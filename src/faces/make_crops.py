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
  rtfs   RTFS-10k (scripts/download_rtfs.py): video-call clips and their face swaps (inswapper, UniFace). The real
         clips are H.264 (x264 CRF 23), the swaps HEVC (x265 CRF 28, read from their files), so every real frame is
         re-encoded like the swaps first: then only the face differs. Split by YouTube video (70/15/15% train/dev/test)
         and by source photo (dev/test swaps only from photos never used in training swaps); children's photos skipped.
         Run in shards (--shard 0/3 ...), then --merge 3 joins them and runs the coding check (see rtfs_merge).

Usage (from the repo root):
  python -m src.faces.make_crops of --split Val --out D:/Datasets/FaceCrops
  python -m src.faces.make_crops list features/lists/commons_dev.csv ... --kind app --out D:/Datasets/FaceCrops
  python -m src.faces.make_crops dfdc D:/Datasets/DFDC/dfdc_train_part_0 ... --kind wide --name dfdc_train --out ...
  python -m src.faces.make_crops rtfs --data-dir /content/data/rtfs --shard 0/3 --out /content/FaceCrops  (and 1/3, 2/3)
  python -m src.faces.make_crops rtfs --merge 3 --out /content/FaceCrops
"""

import argparse
import csv
import json
import os
import random
import subprocess
import tempfile
import time
import zlib
from collections import defaultdict
from pathlib import Path

import cv2
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


RTFS_GOP, RTFS_OFFSET = 250, 6  # keyframes at frames 0, 250, 500 in the real and the swapped clips; we take 6 after one
RTFS_X265 = "-c:v libx265 -preset medium -crf 28 -pix_fmt yuv420p"  # the swaps' own encoder settings


def ffmpeg_frames(ffmpeg, inputs, count, size):
    """The first `count` frames ffmpeg decodes from `inputs` (input options + file), as RGB arrays ([] on a size mismatch)."""
    w, h = size
    out = subprocess.run([ffmpeg, "-v", "error", *inputs, "-frames:v", str(count), "-f", "rawvideo", "-pix_fmt", "rgb24",
                          "-"], capture_output=True).stdout
    if not out or len(out) % (w * h * 3):
        return []
    return [np.frombuffer(out, np.uint8, w * h * 3, i * w * h * 3).reshape(h, w, 3) for i in range(len(out) // (w * h * 3))]


def ring_sharpness(image, face):
    """Variance of the Laplacian in the outer ring of the wide crop (hair, neck, background). A swap leaves the ring
    alone, so there a real and a swapped frame differ only by their video coding."""
    box = tuple(round(v) for v in crop.crop_box(face, k=WIDE_K))
    gray = np.asarray(image.crop(box).resize((WIDE_SIZE, WIDE_SIZE), Image.BICUBIC).convert("L"), dtype=np.float32)
    lap = cv2.Laplacian(gray, cv2.CV_32F)
    m = round(WIDE_SIZE * 0.15)
    ring = np.ones(lap.shape, bool)
    ring[m:-m, m:-m] = False
    return float(lap[ring].var())


def rtfs_split(clip, photo=None):
    """'train', 'dev', 'test' (or None: a swap not used). Clips by YouTube video (its person and room): 70/15/15%.
    Source photos: 80% for training swaps, 20% held out; dev and test swaps use only held-out photos, so neither the
    person in the clip nor the identity swapped in was seen in training."""
    v = zlib.crc32(clip.rsplit("_", 2)[0].encode()) % 100
    split = "train" if v < 70 else "dev" if v < 85 else "test"
    if photo is None:
        return split
    held = zlib.crc32(photo.encode()) % 100 >= 80
    return split if held == (split != "train") else None


def rtfs_clip(args, finder, writer, real, fakes, split, rng, checks):
    """Crops of one real clip and its swaps. At up to 3 keyframe starts (0, 250, 500): frame start+6 of the real clip
    re-encoded the way the swaps were (H.264 -> x265 CRF 28; else "HEVC look = swap" would be a shortcut) and the same
    frame of its swaps (each swap at one random start), all cut at the real frame's largest face (the swapped person;
    the same box for real and swap, as for DFDC). Swaps whose face did not change are skipped. The coding check
    (rtfs_merge) also needs the original real frame: decoded at the clip's first start only (time)."""
    from src.features.extract_dfdc_faces import changed
    cap = cv2.VideoCapture(str(real))
    n, fps = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), cap.get(cv2.CAP_PROP_FPS) or 30.0
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    cap.release()
    starts = [s for s in range(0, n, RTFS_GOP) if s + RTFS_OFFSET + 4 <= n]
    if split != "train":
        starts = sorted(rng.sample(starts, min(2, len(starts))))
    if not starts or not all(size):
        return
    at = {s: [] for s in starts}
    for fake in fakes:
        at[rng.choice(starts)].append(fake)
    kind = "wide" if split == "train" else "app"
    group = real.stem.rsplit("_", 2)[0]
    seg = Path(tempfile.gettempdir()) / f"rtfs_seg_{os.getpid()}.mp4"
    for s in starts:
        seek = ["-ss", f"{s / fps:.4f}", "-i"]
        encoded = subprocess.run([args.ffmpeg, "-v", "error", "-y", *seek, str(real), "-frames:v", str(RTFS_OFFSET + 4),
                                  *args.encoder.split(), "-an", str(seg)], capture_output=True)
        frames = ffmpeg_frames(args.ffmpeg, ["-i", str(seg)], RTFS_OFFSET + 1, size) if encoded.returncode == 0 else []
        if len(frames) <= RTFS_OFFSET:
            continue
        real_img = Image.fromarray(frames[RTFS_OFFSET])
        faces = [f for f in finder.find(real_img, with_masks=kind == "wide") if not f["too_small"]]
        if not faces:
            continue
        face = max(faces, key=lambda f: (f["box"][2] - f["box"][0]) * (f["box"][3] - f["box"][1]))
        writer.add(real_img, face, 0, "rtfs/real", group, kind, face.get("mask"))
        original = ffmpeg_frames(args.ffmpeg, [*seek, str(real)], RTFS_OFFSET + 1, size) if s == starts[0] else []
        for path, variant in at[s]:
            swapped = ffmpeg_frames(args.ffmpeg, [*seek, str(path)], RTFS_OFFSET + 1, size)
            if len(swapped) <= RTFS_OFFSET:
                continue
            fake_img = Image.fromarray(swapped[RTFS_OFFSET])
            if not changed(real_img, fake_img, face)[0]:
                continue
            writer.add(fake_img, face, 1, "rtfs/fake", f"{group}|{variant}", kind)
            if len(original) > RTFS_OFFSET:
                checks.append({"split": split, "matched": ring_sharpness(real_img, face), "fake": ring_sharpness(fake_img, face),
                               "original": ring_sharpness(Image.fromarray(original[RTFS_OFFSET]), face)})


def run_rtfs(args, finder):
    codec = args.encoder.split()[args.encoder.split().index("-c:v") + 1]
    encoders = subprocess.run([args.ffmpeg, "-hide_banner", "-encoders"], capture_output=True, text=True).stdout
    if f" {codec} " not in encoders:
        raise SystemExit(f"{args.ffmpeg} has no {codec} encoder (Colab: apt-get install ffmpeg)")
    root = args.data_dir
    with open(root / "photos.csv", newline="", encoding="utf-8") as f:
        children = {r["filename"] for r in csv.DictReader(f) if "child" in r["tags"]}
    fakes_of = defaultdict(list)
    for tool in ("inswapper", "uniface"):
        if not (root / f"{tool}.csv").exists():
            continue
        with open(root / f"{tool}.csv", newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                path = root / tool / r["filename"]
                if (r["success"] == "True" and r["photo"] not in children and rtfs_split(r["video"], r["photo"])
                        and path.exists()):
                    fakes_of[r["video"]].append((path, tool + ("+enhancer" if r["additional_features"].strip() else "")))
    shard, shards = map(int, args.shard.split("/"))
    reals = [p for p in sorted((root / "original_videos").glob("*.mp4")) if zlib.crc32(p.name.encode()) % shards == shard]
    reals = reals[: args.max_images]
    writers = {s: Writer(args.out, f"rtfs_{s}_s{shard}") for s in ("train", "dev", "test")}
    rng, checks, start = random.Random(args.seed + shard), [], time.time()
    print(f"RTFS shard {shard}/{shards}: {len(reals)} real clips, {sum(len(v) for v in fakes_of.values())} usable swaps "
          f"in all shards ({len(children)} photos of children skipped)", flush=True)
    for n, real in enumerate(reals, 1):
        split = rtfs_split(real.name)
        rtfs_clip(args, finder, writers[split], real, fakes_of.get(real.name, []), split, rng, checks)
        if n % 50 == 0:
            print(f"  {n}/{len(reals)} clips, " + ", ".join(f"{s} {w.count}" for s, w in writers.items())
                  + f" crops, {time.time() - start:.0f} s", flush=True)
    for w in writers.values():
        w.save()
    with open(args.out / f"rtfs_check_s{shard}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["split", "matched", "fake", "original"], lineterminator="\n")
        w.writeheader()
        w.writerows(checks)


def rtfs_merge(args):
    """Joins the shards into rtfs_train / rtfs_dev / rtfs_test, then checks the coding match: the sharpness of the ring
    around the face (which a swap leaves alone) must not tell swapped frames from the re-encoded real frames (AUC within
    0.5 +- 0.1, median swapped/real ratio within 1 +- 0.03). If it does, rtfs_train is renamed *_rejected (training
    then skips it) and the reason is printed."""
    from src.eval import metrics
    for split in ("train", "dev", "test"):
        rows = []
        for shard in range(args.merge):
            path = args.out / f"rtfs_{split}_s{shard}.csv"
            if path.exists():
                with open(path, newline="") as f:
                    rows += list(csv.DictReader(f))
        with open(args.out / f"rtfs_{split}.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS, lineterminator="\n")
            w.writeheader()
            w.writerows(rows)
        fakes = sum(r["label"] == "1" for r in rows)
        print(f"rtfs_{split}: {len(rows) - fakes} real + {fakes} swapped faces from "
              f"{len({r['group'].split('|')[0] for r in rows})} YouTube videos", flush=True)
    checks = []
    for shard in range(args.merge):
        path = args.out / f"rtfs_check_s{shard}.csv"
        if path.exists():
            with open(path, newline="") as f:
                checks += list(csv.DictReader(f))
    if not checks:
        print("!!! no RTFS crops at all", flush=True)
        return
    m, f, o = (np.array([float(c[k]) for c in checks]) for k in ("matched", "fake", "original"))
    swapped = np.r_[np.ones(len(f)), np.zeros(len(f))]
    auc_m, auc_o = metrics.auc(np.r_[f, m], swapped), metrics.auc(np.r_[f, o], swapped)
    ratio_m, ratio_o = float(np.median(f / np.maximum(m, 1e-6))), float(np.median(f / np.maximum(o, 1e-6)))
    print(f"coding check on {len(f)} real/swapped frame pairs (sharpness around the face, which a swap leaves alone): "
          f"swapped vs re-encoded real AUC {auc_m:.3f}, median ratio {ratio_m:.3f}, swap sharper in {(f > m).mean():.0%} "
          f"of pairs; swapped vs the original H.264 real AUC {auc_o:.3f}, ratio {ratio_o:.3f}, swap sharper in "
          f"{(f > o).mean():.0%}", flush=True)
    if abs(auc_m - 0.5) > 0.1 or abs(ratio_m - 1) > 0.03:
        (args.out / "rtfs_train.csv").replace(args.out / "rtfs_train_rejected.csv")
        print("!!! the re-encoded real frames still differ from the swaps outside the face: rtfs_train NOT used for "
              "training (renamed rtfs_train_rejected.csv)", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mode", choices=["of", "list", "dfdc", "rtfs"])
    parser.add_argument("inputs", nargs="*", type=Path, help="list mode: CSV lists / SID-Set parquet files; dfdc: parts")
    parser.add_argument("--out", type=Path, default=Path("D:/Datasets/FaceCrops"))
    parser.add_argument("--name", help="list/dfdc mode: name of this crop set")
    parser.add_argument("--kind", choices=["wide", "app"], default="wide")
    parser.add_argument("--split", default="Val", help="of mode: Val or Test-Dev")
    parser.add_argument("--data-dir", type=Path, default=Path("D:/Datasets/OpenForensics"))
    parser.add_argument("--exclude", nargs="*", default=[], help="list mode: CSV lists whose photos are skipped")
    parser.add_argument("--max-images", type=int, help="only the first N photos (videos) of each input (quick test)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--shard", default="0/1", help="rtfs mode: this process's share of the clips, e.g. 0/3")
    parser.add_argument("--merge", type=int, help="rtfs mode: join this many shards and run the coding check")
    parser.add_argument("--ffmpeg", default="ffmpeg", help="rtfs mode: the ffmpeg program (needs libx265)")
    parser.add_argument("--encoder", default=RTFS_X265, help="rtfs mode: how the real clips are re-encoded")
    args = parser.parse_args()

    if args.mode == "rtfs" and args.merge:
        rtfs_merge(args)
        return
    finder = FaceFinder()
    {"of": run_of, "list": run_list, "dfdc": run_dfdc, "rtfs": run_rtfs}[args.mode](args, finder)


if __name__ == "__main__":
    main()
