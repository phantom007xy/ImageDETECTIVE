"""Brightness / resolution check: does a head's AI score move when the same REAL images change?

A detector should not care about brightness or resolution. If darker images score higher, real
night photos and low-light selfies get more false alarms (priority #1); if smaller images score
higher, small real images (thumbnails, old forwards) do. The check uses only the real images of a
list, changes them (--change brightness: brighter/darker by the factor; --change downscale: width
and height times the factor) and reports the mean score and the share called AI.

Usage (from the repo root):
  python -m src.eval.brightness_check --heads weights/head_path_a_sidset30.pt weights/head_face_sidset10.pt
      --list features/lists/wilddeepfake.csv --view ready-crop
  python -m src.eval.brightness_check --heads ... --list features/lists/commons_dev.csv --view app
      --change downscale --factors 1 0.5 0.35 0.25
"""

import argparse
import csv
import random
from pathlib import Path

import torch
from PIL import Image, ImageEnhance

from src.data import preprocess
from src.faces import crop
from src.models.backbone import Backbone
from src.models.linear_head import LinearHead

VIEWS = {"whole": preprocess.eval_view, "ready-crop": crop.ready_crop_view, "app": preprocess.inference_view}


def change(image, kind, factor):
    if kind == "brightness":
        return ImageEnhance.Brightness(image).enhance(factor)
    w, h = image.size
    return image.resize((max(16, round(w * factor)), max(16, round(h * factor))), Image.BICUBIC)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--heads", type=Path, nargs="+", required=True)
    parser.add_argument("--list", type=Path, required=True, help="CSV image list (path, label, source)")
    parser.add_argument("--view", choices=list(VIEWS), default="whole",
                        help="whole = centre crop (448 px); ready-crop = the image is a face crop already; app = the app's view")
    parser.add_argument("--change", choices=["brightness", "downscale"], default="brightness")
    parser.add_argument("--n", type=int, default=300, help="real images to use")
    parser.add_argument("--factors", type=float, nargs="+", default=[0.5, 0.7, 1.0, 1.3, 1.6])
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    with open(args.list, newline="") as f:
        paths = [r["path"] for r in csv.DictReader(f) if r["label"] == "0"]
    paths = random.Random(args.seed).sample(paths, min(args.n, len(paths)))
    images = []
    for p in paths:
        with Image.open(p) as im:
            images.append(im.convert("RGB"))

    backbone = Backbone()
    view = VIEWS[args.view]
    features = {}
    for factor in args.factors:
        out = []
        tensors = [preprocess.to_tensor(view(change(im, args.change, factor))) for im in images]
        for i in range(len(tensors)):  # app views differ in shape, so one image per pass there
            if args.view == "app":
                out.append(backbone.mean_patch_vector(tensors[i].unsqueeze(0)).cpu())
            elif i % 32 == 0:
                out.append(backbone.mean_patch_vector(torch.stack(tensors[i:i + 32])).cpu())
        features[factor] = torch.cat(out)

    print(f"{len(images)} real images from {args.list.name} ({args.view}); "
          f"per {args.change} factor: mean logit (share called AI at 0.5)")
    for head_path in args.heads:
        head = LinearHead.load(head_path)
        cells = []
        for factor in args.factors:
            scores = head.logits(features[factor])
            cells.append(f"x{factor:g}: {scores.mean():+6.2f} ({(scores > 0).float().mean():4.0%})")
        print(f"  {head_path.stem:<24} " + " | ".join(cells))


if __name__ == "__main__":
    main()
