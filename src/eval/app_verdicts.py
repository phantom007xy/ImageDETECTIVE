"""Run the app's detector (src/inference.py) on an image list and count its verdicts.

Unlike evaluate_head (fixed centre crop), this uses exactly what the app does: the whole image
with its aspect ratio kept, faces, and the app's verdict thresholds. Results are given per source,
real and AI separately. Pixels only: the metadata check is left out on purpose, so the numbers show
what the detector itself can do (a forwarded or edited copy has lost its metadata anyway).

Usage (from the repo root):
  python -m src.eval.app_verdicts --head weights/head_path_a_sidset30.pt --list features/lists/muse_image_test_v1.csv
"""

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image

from src.inference import Detector

VERDICTS = ["AI-generated", "Likely AI-generated", "This face looks AI-made or AI-changed", "Uncertain",
            "Likely authentic"]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--head", type=Path, required=True)
    parser.add_argument("--list", type=Path, required=True, help="CSV image list (path, label, source)")
    args = parser.parse_args()

    detector = Detector(args.head)
    groups = defaultdict(list)  # (label, source) -> verdicts
    with open(args.list, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            with Image.open(row["path"]) as im:
                result = detector.analyze(im.convert("RGB"))
            groups[(int(row["label"]), row["source"])].append(result["verdict"])

    print(f"{args.list.name}, head {args.head.stem} (the app's view and thresholds)")
    for label, name in ((0, "REAL"), (1, "AI")):
        keys = sorted(k for k in groups if k[0] == label)
        if not keys:
            continue
        all_verdicts = [v for k in keys for v in groups[k]]
        for title, verdicts in [(f"{name}, all", all_verdicts)] + [(f"  {k[1]}", groups[k]) for k in keys]:
            counts = Counter(verdicts)
            cells = " | ".join(f"{v} {counts[v]:3d} ({counts[v] / len(verdicts):4.0%})" for v in VERDICTS)
            print(f"{title:<34} n={len(verdicts):4d} | {cells}")


if __name__ == "__main__":
    main()
