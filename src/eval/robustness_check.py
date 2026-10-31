"""Robustness check: does the app's verdict hold when a picture is forwarded, shrunk or screenshotted?

Every image of the given lists goes through the same "forwarding damage" in several versions, then
through exactly the app's path (src/inference.py: app view, the head, its calibrated thresholds).
Reported per list and version, real and AI separately: the share called AI (Likely AI-generated or
AI-generated), Uncertain and Likely authentic. For real photos the AI share is a false alarm; for AI
images the "authentic" share is a miss. Whole image only (no faces, no metadata: a forwarded copy has
lost its metadata anyway).

Versions:
  original    the file as it is
  jpeg70      saved once as JPEG, quality 70
  whatsapp    long side at most 1600 px, JPEG quality 75 (a typical WhatsApp forward)
  whatsapp2   the same step twice (forwarded again)
  small       long side at most 800 px, JPEG quality 60 (old forwards, thumbnails)
  screenshot  scaled to a phone screen 1080 px wide, saved as PNG
  crop        the middle 70% of each side, JPEG quality 85 (the cropped-copy test)

Usage (from the repo root):
  python -m src.eval.robustness_check --head weights/open_gen_v1/head_F.pt
      --list "polished dev:features/lists/commons_dev.csv" --list "SDXL:features/lists/sd_faces.csv@sd_faces/sdxl"
"""

import argparse
import csv
import io
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from src.data import preprocess
from src.inference import load_calibration
from src.models.backbone import Backbone
from src.models.linear_head import LinearHead


def jpeg_bytes(image, quality):
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=quality)
    return Image.open(io.BytesIO(buffer.getvalue())).convert("RGB")


def shrink(image, long_side):
    image = image.copy()
    image.thumbnail((long_side, long_side), Image.BICUBIC)  # only ever makes it smaller
    return image


def screenshot(image, width=1080):
    w, h = image.size
    shot = image.resize((width, round(h * width / w)), Image.BICUBIC)
    buffer = io.BytesIO()
    shot.save(buffer, format="PNG")
    return Image.open(io.BytesIO(buffer.getvalue())).convert("RGB")


def crop(image, keep=0.7):
    """The middle 70% of each side, re-saved as JPEG 85: the "cropped copy" that hid Muse images from
    Meta's watermark-based detector (Reuters, July 2026). Our detector reads pixels, not the watermark."""
    w, h = image.size
    cw, ch = round(w * keep), round(h * keep)
    left, top = (w - cw) // 2, (h - ch) // 2
    return jpeg_bytes(image.crop((left, top, left + cw, top + ch)), 85)


VERSIONS = {
    "original": lambda im: im,
    "jpeg70": lambda im: jpeg_bytes(im, 70),
    "whatsapp": lambda im: jpeg_bytes(shrink(im, 1600), 75),
    "whatsapp2": lambda im: jpeg_bytes(shrink(jpeg_bytes(shrink(im, 1600), 75), 1600), 75),
    "small": lambda im: jpeg_bytes(shrink(im, 800), 60),
    "screenshot": screenshot,
    "crop": crop,
}


def read_list(spec, n, seed):
    path, _, source = spec.partition("@")
    with open(path, newline="") as f:
        rows = [r for r in csv.DictReader(f) if not source or r["source"] == source]
    return random.Random(seed).sample(rows, min(n, len(rows)))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--head", type=Path, required=True)
    parser.add_argument("--list", action="append", required=True, help='"NAME:image list csv[@source]"')
    parser.add_argument("--n", type=int, default=150, help="images per list (random sample)")
    parser.add_argument("--versions", nargs="+", default=list(VERSIONS))
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    head, c = LinearHead.load(args.head), load_calibration(args.head)
    backbone = Backbone()
    print(f"head {args.head.name} ({'calibrated' if c['calibrated'] else 'provisional thresholds'}); "
          f"shares: AI (likely or clear) | Uncertain | Likely authentic\n")
    for item in args.list:
        name, spec = item.split(":", 1)
        rows = read_list(spec, args.n, args.seed)
        labels = {int(r["label"]) for r in rows}
        kind = "REAL" if labels == {0} else "AI" if labels == {1} else "MIXED"
        logits = {v: [] for v in args.versions}
        for row in rows:
            with Image.open(row["path"]) as im:
                image = ImageOps.exif_transpose(im).convert("RGB")
            for version in args.versions:
                view = preprocess.inference_view(VERSIONS[version](image))
                vector = backbone.mean_patch_vector(preprocess.to_tensor(view).unsqueeze(0)).cpu()
                logits[version].append(float(head.logits(vector)[0]))
        print(f"{kind} {name} ({len(rows)} images)")
        for version in args.versions:
            z = np.array(logits[version])
            ai, auth = (z > c["ai_logit"]).mean(), (z <= c["real_logit"]).mean()
            print(f"  {version:<11} AI {ai:6.1%} | Uncertain {1 - ai - auth:6.1%} | Likely authentic {auth:6.1%} "
                  f"| mean logit {z.mean():+6.2f}")
        print()


if __name__ == "__main__":
    main()
