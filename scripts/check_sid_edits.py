"""Check whether SID-Set's tampered images were regenerated as a whole, or only inside the mask.

Why: if the whole image went through the diffusion decoder, AI traces are everywhere, and an
edited-region head trained on these masks would learn "this image was decoded", not "this region
was edited".

Idea: every camera photo has fine sensor/JPEG noise; a decoded image has a different, smoother
texture. We measure that fine texture (the residual after a 3x3 median filter) in:
  - real images (whole image)
  - fully synthetic images (whole image)
  - tampered images, inside and outside the mask
If "tampered, outside the mask" looks like the synthetic images, the whole image was regenerated.

Also reports file format, size and mask area per class (format and size shortcuts).

Usage:
  python -I scripts/check_sid_edits.py PARQUET_FILE [--panel OUT.png]
"""

import argparse
import io
import statistics
from collections import Counter

import cv2
import numpy as np
import pyarrow.parquet as pq
from PIL import Image

LABELS = {0: "real", 1: "synthetic", 2: "tampered"}


def image_format(data):
    if data[:2] == b"\xff\xd8":
        return "JPEG"
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return "PNG"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "WEBP"
    return "other"


def decode_grey(data):
    with Image.open(io.BytesIO(data)) as im:
        return np.array(im.convert("L"))


def residual(grey):
    """Fine texture: absolute difference from a 3x3 median-filtered copy."""
    return np.abs(grey.astype(np.float32) - cv2.medianBlur(grey, 3).astype(np.float32))


# Edges also produce a large residual, so objects (inside masks) look "noisier" than backgrounds.
# Measuring only in flat areas (small gradient) keeps the noise and drops most of the content.
FLAT_GRADIENT = 6.0


def flat_area(grey):
    smooth = cv2.GaussianBlur(grey, (5, 5), 0).astype(np.float32)
    gradient = np.hypot(cv2.Sobel(smooth, cv2.CV_32F, 1, 0), cv2.Sobel(smooth, cv2.CV_32F, 0, 1)) / 8
    return gradient < FLAT_GRADIENT


def flat_noise(res, region):
    """Mean residual over the flat pixels of a region, or None if the region has too few of them."""
    return float(res[region].mean()) if region.sum() >= 500 else None


def auc(a, b):
    """How well a value separates group a from group b (0.5 = chance, 1.0 = perfect)."""
    values = np.concatenate([a, b])
    is_b = np.concatenate([np.zeros(len(a), bool), np.ones(len(b), bool)])
    uniq, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
    ranks = (np.cumsum(counts) - counts + 1 + (counts - 1) / 2)[inverse]  # average ranks for ties
    n1, n0 = is_b.sum(), (~is_b).sum()
    result = (ranks[is_b].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
    return max(result, 1 - result)


def describe(values):
    if not values:
        return "n/a"
    q = np.percentile(values, [10, 50, 90])
    return f"median {q[1]:.2f} [10%: {q[0]:.2f}, 90%: {q[2]:.2f}]"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("parquet")
    parser.add_argument("--panel", help="save image | mask | residual for a few tampered images")
    args = parser.parse_args()

    formats = {name: Counter() for name in LABELS.values()}
    sizes = {name: [] for name in LABELS.values()}
    texture = {"real": [], "synthetic": [], "tampered inside": [], "tampered outside": []}
    flat = {"real": [], "synthetic": [], "tampered inside": [], "tampered outside": []}
    mask_area = []
    mask_size_matches = Counter()
    panels = []

    table = pq.ParquetFile(args.parquet)
    for batch in table.iter_batches(batch_size=32, columns=["label", "image", "mask"]):
        for row in batch.to_pylist():
            name = LABELS[row["label"]]
            data = row["image"]["bytes"]
            formats[name][image_format(data)] += 1
            grey = decode_grey(data)
            h, w = grey.shape
            sizes[name].append((w, h))
            res = residual(grey)
            is_flat = flat_area(grey)

            if name != "tampered":
                texture[name].append(float(res.mean()))
                value = flat_noise(res, is_flat)
                if value is not None:
                    flat[name].append(value)
                continue

            mask = decode_grey(row["mask"]["bytes"])
            mask_size_matches[mask.shape == grey.shape] += 1
            if mask.shape != grey.shape:
                mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
            inside = mask > 127
            if inside.sum() < 100 or (~inside).sum() < 100:
                continue
            mask_area.append(float(inside.mean()))
            texture["tampered inside"].append(float(res[inside].mean()))
            texture["tampered outside"].append(float(res[~inside].mean()))
            # Only images where both sides have enough flat area, so inside and outside stay paired.
            flat_in, flat_out = flat_noise(res, inside & is_flat), flat_noise(res, ~inside & is_flat)
            if flat_in is not None and flat_out is not None:
                flat["tampered inside"].append(flat_in)
                flat["tampered outside"].append(flat_out)

            if args.panel and len(panels) < 3:
                small = (512, 512 * h // w)
                image_rgb = cv2.cvtColor(np.array(Image.open(io.BytesIO(data)).convert("RGB")), cv2.COLOR_RGB2BGR)
                amplified = np.clip(res * 12, 0, 255).astype(np.uint8)
                panels.append(np.hstack([
                    cv2.resize(image_rgb, small),
                    cv2.resize(cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR), small),
                    cv2.resize(cv2.cvtColor(amplified, cv2.COLOR_GRAY2BGR), small),
                ]))

    print("=== format and size per class ===")
    for name in LABELS.values():
        n = len(sizes[name])
        square = sum(w == h for w, h in sizes[name])
        common = Counter(sizes[name]).most_common(1)[0][0] if n else None
        print(f"{name:10s} {n:4d} images | formats {dict(formats[name])} | square {square}/{n} | most common size {common}")

    print("\n=== masks (tampered) ===")
    print(f"mask has the same size as the image: {dict(mask_size_matches)}")
    print(f"edited share of the image: {describe(mask_area)}")

    print("\n=== fine texture (mean residual, higher = more fine noise) ===")
    for key, values in texture.items():
        print(f"{key:17s} {describe(values)}")

    real, synth = np.array(texture["real"]), np.array(texture["synthetic"])
    outside, inside = np.array(texture["tampered outside"]), np.array(texture["tampered inside"])
    print("\n=== how different are they? (AUC: 0.5 = same, 1.0 = fully separable) ===")
    print(f"real vs synthetic                : {auc(real, synth):.2f}")
    print(f"real vs tampered OUTSIDE the mask: {auc(real, outside):.2f}   <- near 0.5 = outside looks like a real photo")
    print(f"synthetic vs tampered OUTSIDE    : {auc(synth, outside):.2f}   <- near 0.5 = outside looks synthetic")
    print(f"tampered inside vs outside       : {auc(inside, outside):.2f}   <- near 0.5 = no local difference")

    print("\n=== noise in FLAT areas only (edges removed, so content matters less) ===")
    for key, values in flat.items():
        print(f"{key:17s} {len(values):4d} images | {describe(values)}")
    f_real, f_synth = np.array(flat["real"]), np.array(flat["synthetic"])
    f_in, f_out = np.array(flat["tampered inside"]), np.array(flat["tampered outside"])
    print(f"real vs synthetic                : {auc(f_real, f_synth):.2f}")
    print(f"real vs tampered OUTSIDE the mask: {auc(f_real, f_out):.2f}")
    print(f"synthetic vs tampered OUTSIDE    : {auc(f_synth, f_out):.2f}")
    print(f"tampered inside vs outside       : {auc(f_in, f_out):.2f}")
    smoother_inside = float(np.mean(f_in < f_out)) if len(f_in) else float("nan")
    print(f"share of tampered images where the edit is smoother than its surroundings: {smoother_inside:.0%}")

    if panels:
        cv2.imwrite(args.panel, np.vstack([p[:, :, :] for p in panels if p.shape[1] == panels[0].shape[1]]))
        print(f"\nsaved {args.panel}")


if __name__ == "__main__":
    main()
