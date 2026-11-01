"""DINOv3 features of OpenForensics face crops: real faces and AI-replaced faces from the same photos (face swaps in
still photos, type b). Photos from scripts/download_openforensics.py.

For every photo, SAM 3 finds the faces with the app's settings (src/faces/sam_faces.py). Each SAM face is matched to
the annotated face boxes (<split>_poly.json) by box overlap (IoU >= --min-iou, best pairs first, one to one); the
matched annotation gives the label (Real / Fake). SAM faces without a match are skipped (label unknown), as are faces
too small to check. Crops as everywhere: train = random crops (crop.train_face_view), eval = the app's crop.
A photo and all its faces share one group (the photo name), so a photo never ends up on both sides of a split.

The photos are NOT turned upright from EXIF: the annotation boxes refer to the stored pixels.
With --layers, the mean patch tokens after those DINOv3 blocks are saved too (features_L<k>), from the same pass:
middle layers keep more low-level traces (they found Muse edits far better: results/backbone_probe_v1.txt).

Usage (from the repo root):
  python -m src.features.extract_openforensics_faces --split Val --mode train --copies 4
  python -m src.features.extract_openforensics_faces --split Test-Dev --mode eval
"""

import argparse
import csv
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from src.data import preprocess
from src.faces import crop
from src.faces.sam_faces import FaceFinder
from src.features.extract import view_rng
from src.models.backbone import Backbone


def iou(a, b):
    x0, y0, x1, y1 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def annotated_faces(path):
    """Photo name -> (width, height, [(box x0 y0 x1 y1, is_fake), ...])."""
    data = json.loads(path.read_text())
    faces = defaultdict(list)
    for a in data["annotations"]:
        x, y, w, h = a["bbox"]
        faces[a["image_id"]].append(([x, y, x + w, y + h], a["category_id"] == 1))
    return {Path(img["file_name"]).name: (img["width"], img["height"], faces[img["id"]]) for img in data["images"]}


def match(sam_faces, truth, min_iou):
    """Pairs (sam index, is_fake, iou), best overlaps first, each SAM face and each annotation used once."""
    pairs = sorted(((iou(f["box"], box), i, j) for i, f in enumerate(sam_faces) for j, (box, _) in enumerate(truth)),
                   reverse=True)
    used_sam, used_truth, out = set(), set(), []
    for overlap, i, j in pairs:
        if overlap < min_iou:
            break
        if i not in used_sam and j not in used_truth:
            used_sam.add(i), used_truth.add(j)
            out.append((i, truth[j][1], overlap))
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", required=True, help="Val, Test-Dev, Test-Challenge or Train")
    parser.add_argument("--mode", choices=["train", "eval"], default="eval")
    parser.add_argument("--copies", type=int, default=4, help="random copies per face (train mode only)")
    parser.add_argument("--min-iou", type=float, default=0.3)
    parser.add_argument("--max-photos", type=int, help="only the first N photos of the list (quick test)")
    parser.add_argument("--data-dir", type=Path, default=Path("D:/Datasets/OpenForensics"))
    parser.add_argument("--out", type=Path, help="default: features/faces/openforensics_<split>_<mode>.npz")
    parser.add_argument("--layers", type=int, nargs="*", default=[8, 12, 16], help="also save these blocks' means")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    truth_of = annotated_faces(args.data_dir / f"{args.split}_poly.json")
    with open(f"features/lists/openforensics_{args.split}.csv", newline="") as f:
        paths = [Path(r["path"]) for r in csv.DictReader(f)][: args.max_photos]
    out = args.out or Path(f"features/faces/openforensics_{args.split.lower().replace('-', '_')}_{args.mode}.npz")

    backbone, finder = Backbone(), FaceFinder()
    feats, labels, ids, groups, ious, batch = defaultdict(list), [], [], [], [], []
    stats = defaultdict(int)

    def run_batch():
        if batch:
            for key, value in backbone.layer_means(torch.stack(batch), args.layers).items():
                feats[key].append(value.cpu().numpy().astype(np.float16))
            batch.clear()

    start = time.time()
    for n, path in enumerate(paths, 1):
        width, height, truth = truth_of[path.name]
        with Image.open(path) as im:
            image = im.convert("RGB")
        if image.size != (width, height):
            stats["photos with a size mismatch (skipped)"] += 1
            continue
        stats["annotated real faces"] += sum(not fake for _, fake in truth)
        stats["annotated fake faces"] += sum(fake for _, fake in truth)
        found = finder.find(image)
        stats["SAM faces"] += len(found)
        for i, is_fake, overlap in match(found, truth, args.min_iou):
            face = found[i]
            if face["too_small"]:
                stats["matched but too small"] += 1
                continue
            stats["fake faces kept" if is_fake else "real faces kept"] += 1
            key = f"{args.split}/{path.name}#{i}"
            for copy in range(args.copies if args.mode == "train" else 1):
                view = (crop.train_face_view(image, face, view_rng(key, copy, args.seed)) if args.mode == "train"
                        else crop.eval_face_view(image, face))
                batch.append(preprocess.to_tensor(view))
                labels.append(int(is_fake))
                ids.append(key)
                groups.append(path.name)
                ious.append(overlap)
                if len(batch) == 32:
                    run_batch()
        if n % 200 == 0:
            print(f"  {n}/{len(paths)} photos, {dict(stats)}, {time.time() - start:.0f} s", flush=True)
    run_batch()

    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **{k: np.concatenate(v) for k, v in feats.items()}, labels=np.array(labels, dtype=np.int8),
                        img_ids=np.array(ids), groups=np.array(groups), iou=np.array(ious, dtype=np.float16),
                        sources=np.array(["openforensics/" + ("fake" if lab else "real") for lab in labels]),
                        mode=args.mode)
    print(f"done: {len(paths)} photos, {dict(stats)} -> {len(labels)} vectors in {time.time() - start:.0f} s -> {out}",
          flush=True)


if __name__ == "__main__":
    main()
