"""Forwarded copies and the face-swap check: do they push REAL faces over the line, and are swaps still caught?

Whole photos are damaged as a forward would (src/eval/robustness_check.py: JPEG 70, WhatsApp-like, forwarded twice,
small copy, screenshot), then the app's face path runs on the copy: SAM 3 faces, the app's crop at 336, the face-swap
Perception Encoder and its head. A photo counts as flagged if any of its faces is above the line.
  - REAL: photos with faces from the polished Commons dev set and the unseen OpenImages photos
  - "phone": real faces given beauty-mode or portrait-mode processing (src/models/tune_face_pe.phone_process)
  - SWAPS: OpenForensics dev photos (15% of Val; for choices) and Test-Dev photos (the test); each has at least one
    replaced face (a real face above the line would also count, which is rare: 0.2% of its real faces)
Several models can be compared on the same copies (SAM runs once per copy): --model "NAME:PE weights:head:line".
Totals only. Usage (from the repo root):
  python -m src.eval.face_swap_robustness --model "v1:weights/face_pe_v1/pe_step4000.pt:weights/face_pe_v1/head_facepe.pt:9.89"
"""

import argparse
import csv
import random
import zlib
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps

from src.data.preprocess import jpeg
from src.eval.robustness_check import VERSIONS
from src.faces import crop
from src.faces.sam_faces import FaceFinder
from src.models.linear_head import LinearHead
from src.models.pe_core import PECore
from src.models.tune_face_pe import phone_process


def phone_copy(image, finder, seed):
    """The photo as a phone with a beauty mode or portrait mode would have made it: every face found (SAM 3 masks) gets
    skin smoothing, a brighter skin tone or a blurred background around it. The faces stay REAL faces."""
    rng = random.Random(seed)
    out = image
    for face in finder.find(image, with_masks=True):
        if not face["too_small"] and face["mask"].any():
            out = Image.fromarray(phone_process(out, face["mask"], rng))
    return out

SETS = {"REAL polished dev": ("features/lists/commons_dev.csv", "features/faces/commons_dev_eval.npz"),
        "REAL OpenImages unseen": ("features/lists/sid_reals_0_9_unseen.csv", "features/faces/sid_reals_0_9_unseen_eval.npz"),
        "SWAPS OpenForensics dev": ("features/lists/openforensics_Val.csv", "dev"),
        "SWAPS OpenForensics test": ("features/lists/openforensics_Test-Dev.csv", None)}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", action="append", required=True, help='"NAME:PE weights:head:line (logit)"')
    parser.add_argument("--sets", nargs="+", help="only these sets (titles)")
    parser.add_argument("--n", type=int, default=120, help="photos per set")
    parser.add_argument("--versions", nargs="+", default=["original", "jpeg70", "whatsapp", "whatsapp2", "small", "screenshot"])
    args = parser.parse_args()

    finder, models = FaceFinder(), {}
    for spec in args.model:
        name, pe_path, head_path, line = spec.rsplit(":", 3) if spec.count(":") > 3 else spec.split(":")
        models[name] = (PECore(tuned_path=pe_path), LinearHead.load(head_path), float(line))
    for title, (list_path, faces_path) in SETS.items():
        if args.sets and title not in args.sets:
            continue
        rows = list(csv.DictReader(open(list_path, newline="")))
        if faces_path == "dev":  # the OpenForensics Val photos of the dev part (src/faces/make_crops.py)
            rows = [r for r in rows if zlib.crc32(Path(r["path"]).name.encode()) % 100 < 15]
        elif faces_path:  # only photos with at least one checked face
            with_faces = set(np.load(faces_path)["img_ids"].astype(str))
            rows = [r for r in rows if r["path"] in with_faces]
        rows = random.Random(0).sample(rows, min(args.n, len(rows)))
        flagged = {(m, v): [] for m in models for v in args.versions}
        for r in rows:
            with Image.open(r["path"]) as im:
                image = (ImageOps.exif_transpose(im) if title.startswith("REAL") else im).convert("RGB")
            for v in args.versions:
                copy = phone_copy(image, finder, zlib.crc32(r["path"].encode())) if v == "phone" else VERSIONS[v](image)
                faces = [f for f in finder.find(copy) if not f["too_small"]]
                for m, (pe, head, line) in models.items():
                    if faces:
                        views = torch.stack([pe.to_tensor(jpeg(crop.face_crop(copy, f, size=pe.size), 90)) for f in faces])
                        flagged[m, v].append(bool((head.logits(pe.layer_features(views).flatten(1)).numpy() > line).any()))
                    else:
                        flagged[m, v].append(False)  # no face big enough in this copy
        print(f"{title} ({len(rows)} photos), share with a face above the line:", flush=True)
        for m in models:
            print(f"  {m:<12} " + " | ".join(f"{v} {np.mean(flagged[m, v]):5.1%}" for v in args.versions), flush=True)


if __name__ == "__main__":
    main()
