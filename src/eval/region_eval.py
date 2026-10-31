"""Evaluate Path B (edited regions) on whole pictures: does it find the edited spot, and only there?

Input: patch feature files made with src/features/extract_patches.py --mode eval (ALL patches of every
picture, with the patch grid). Per picture, Path B's patch scores are smoothed over each patch's 3 x 3
neighbourhood (one noisy patch alone shouldn't count); the picture's score is its strongest spot.

The threshold is set on the real pools (at most --max-false-alarms of their pictures above it, the
strictest pool decides). Then, per edited set: the share of pictures found, and how well the found region
(smoothed score above the threshold) overlaps the true edited region (IoU, mean over found pictures).

Usage (from the repo root):
  python -m src.eval.region_eval --head weights/path_b/head_B.pt --real "polished dev:features/patches/commons_dev_eval.npz"
      --edited "SID val:features/patches/sid_val_eval.npz"
"""

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from src.eval import metrics
from src.eval.fair_compare import threshold_for
from src.models.linear_head import LinearHead


def smooth(grid):
    """Mean over each cell's 3 x 3 neighbourhood (edges use the cells that exist)."""
    padded = np.pad(grid, 1, constant_values=np.nan)
    stack = np.stack([padded[r:r + grid.shape[0], c:c + grid.shape[1]] for r in range(3) for c in range(3)])
    return np.nanmean(stack, axis=0)


def pictures(head, path):
    """{picture: (smoothed score grid, true edited grid)} for one eval feature file."""
    data = np.load(path)
    shapes = dict(zip(data["grid_ids"], map(tuple, data["grid_shapes"])))
    scores = head.logits(torch.from_numpy(data["features"].astype(np.float32))).numpy()
    by_pic = defaultdict(list)
    for i, img_id in enumerate(data["img_ids"]):
        by_pic[str(img_id).rsplit("#", 1)[0]].append(i)
    out = {}
    for pic, idx in by_pic.items():
        rows, cols = shapes[pic]
        grid = np.full(rows * cols, np.nan, dtype=np.float32)
        truth = np.zeros(rows * cols, dtype=bool)
        grid[data["positions"][idx]] = scores[idx]
        truth[data["positions"][idx]] = data["labels"][idx] == 1
        out[pic] = (smooth(grid.reshape(rows, cols)), truth.reshape(rows, cols))
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--head", type=Path, required=True)
    parser.add_argument("--real", action="append", required=True, help='"NAME:patch eval file" (real photos)')
    parser.add_argument("--edited", action="append", required=True, help='"NAME:patch eval file" (edited pictures)')
    parser.add_argument("--max-false-alarms", type=float, nargs="+", default=[0.01, 0.05])
    args = parser.parse_args()

    head = LinearHead.load(args.head)
    reals = {name: pictures(head, path) for name, path in (s.split(":", 1) for s in args.real)}
    edited = {name: pictures(head, path) for name, path in (s.split(":", 1) for s in args.edited)}
    real_scores = {n: np.array([g.max() for g, _ in pics.values()]) for n, pics in reals.items()}

    # Patch level: can single patches tell edited from not edited at all? (AUC over all patches of edited pictures)
    s = np.concatenate([g.reshape(-1) for pics in edited.values() for g, _ in pics.values()])
    t = np.concatenate([tr.reshape(-1) for pics in edited.values() for _, tr in pics.values()])
    print(f"patch level (edited pictures): AUC {metrics.auc(s, t):.3f} over {len(t)} patches ({int(t.sum())} edited)\n")

    for rate in args.max_false_alarms:
        threshold = max(threshold_for(sc, rate) for sc in real_scores.values())
        print(f"threshold for at most {rate:.0%} false alarms on every real pool: {threshold:+.2f}")
        for name, sc in real_scores.items():
            print(f"  REAL   {name:<22}{len(sc):>5} pictures | flagged {(sc > threshold).mean():6.1%}")
        for name, pics in edited.items():
            found, ious = 0, []
            for grid, truth in pics.values():
                if grid.max() > threshold:
                    found += 1
                    pred = grid > threshold
                    union = (pred | truth).sum()
                    ious.append((pred & truth).sum() / union if union else 0.0)
            iou = f"{np.mean(ious):.2f}" if ious else "-"
            print(f"  EDITED {name:<22}{len(pics):>5} pictures | found {found / len(pics):6.1%} | region IoU {iou}")
        print()


if __name__ == "__main__":
    main()
