"""Keep only photos with NO person in them (SAM 3, prompt "person"), before any AI edit of a real photo.

Our rule (plan Section 0): never make AI edits of photos of real people. A face filter is not enough:
people seen from behind, small or turned away have no checkable face. So SAM 3 looks for any "person",
with a low score threshold (0.3, stricter than the app's 0.5): better to drop a photo too many than to
edit one with a person in it.

Usage:
  python scripts/find_people_free.py --list features/lists/commons_dev.csv --out features/lists/commons_dev_no_people.csv
"""

import argparse
import csv
import sys
from pathlib import Path

import torch
from PIL import Image, ImageOps

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root, so `src` can be imported
from transformers import Sam3Model, Sam3Processor  # noqa: E402

from src.models.backbone import load_local_first  # noqa: E402

THRESHOLD = 0.3


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--list", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()

    torch.set_num_threads(args.threads)
    processor = load_local_first(Sam3Processor.from_pretrained, "facebook/sam3")
    model = load_local_first(Sam3Model.from_pretrained, "facebook/sam3").to(args.device).eval()
    with open(args.list, newline="") as f:
        rows = [r for r in csv.DictReader(f) if r["label"] == "0"]
    keep = []
    for n, row in enumerate(rows, 1):
        with Image.open(row["path"]) as im:
            image = ImageOps.exif_transpose(im).convert("RGB")
        inputs = processor(images=image, text="person", return_tensors="pt").to(args.device)
        with torch.no_grad():
            outputs = model(**inputs)
        found = processor.post_process_instance_segmentation(
            outputs, threshold=THRESHOLD, mask_threshold=0.5, target_sizes=inputs["original_sizes"].tolist())[0]
        if len(found["scores"]) == 0:
            keep.append(row)
        if n % 20 == 0:
            print(f"  {n}/{len(rows)} checked, {len(keep)} without people", flush=True)
    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["path", "label", "source"], lineterminator="\n")
        writer.writeheader()
        writer.writerows(keep)
    print(f"{len(keep)} of {len(rows)} photos have no person -> {args.out}")


if __name__ == "__main__":
    main()
