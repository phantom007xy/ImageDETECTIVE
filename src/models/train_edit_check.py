"""Train and test the Muse-edit check: a linear head over PE-Core-L's 6-layer features (src/models/pe_core.py).

Label 1 = rendered by Muse Image: the Muse training edits of people-free photos and the Muse pair fakes. Label 0 = real
photos: the edits' own originals, Commons training photos, SID-Set OpenImages photos and the pair reals. Other
generators are not used: they are head F's job. Each class counts half (src/models/linear_head.py).

Test, on dev sets only, in the app's view of the check (centre square, 336): the line where at most --rates of EVERY
real pool is called "edited" (the strictest pool decides); then the share of every AI set above it. Other generators
(SDXL, PixArt, SD 3.5) are shown for information only.

Usage (from the repo root):
  python -m src.models.train_edit_check --variant "app:features/pe/*_eval.npz" --save-dir weights/edit_check_v1
"""

import argparse
import glob
import math
from pathlib import Path

import numpy as np
import torch

from src.eval import metrics
from src.eval.fair_compare import threshold_for
from src.models.linear_head import LinearHead

DEV = "features/pe/{}_eval.npz"
REAL_POOLS = {"polished dev": ("commons_dev", 0, None), "OpenImages val": ("muse_pairs_val", 0, None),
              "originals v1": ("muse_edits_v1", 0, None), "originals v2": ("muse_edits_test_v2", 0, None)}
AI_SETS = {"edits v1 local": ("muse_edits_v1", 2, None), "edits v1 whole": ("muse_edits_v1", 1, None),
           "edits v2 local": ("muse_edits_test_v2", 2, None), "edits v2 whole": ("muse_edits_test_v2", 1, None),
           "Muse test v1": ("muse_image_test_v1", 1, None), "Muse pairs val": ("muse_pairs_val", 1, None),
           "SDXL": ("sd_faces", 1, "sd_faces/sdxl"), "PixArt": ("open_gen_dev", 1, "open_gen/pixart"),
           "SD3.5 dev": ("open_gen_dev", 1, "open_gen/sd35m_devcaption")}
DEV_FILES = {"muse_edits_v1", "muse_edits_test_v2", "commons_dev", "muse_pairs_val", "muse_image_test_v1", "sd_faces",
             "open_gen_dev"}


def flat(features):
    return torch.from_numpy(features.reshape(len(features), -1).astype(np.float32))


def load_dev(name, label, source):
    d = np.load(DEV.format(name))
    keep = d["labels"] == label
    if source:
        keep &= d["sources"] == source
    return flat(d["features"][keep])


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--variant", action="append", required=True, help='"NAME:glob1,glob2" (training files)')
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--rates", type=float, nargs="+", default=[0.01, 0.02, 0.05])
    parser.add_argument("--save-dir", type=Path)
    args = parser.parse_args()

    reals = {n: load_dev(*spec) for n, spec in REAL_POOLS.items()}
    fakes = {n: load_dev(*spec) for n, spec in AI_SETS.items()}
    for item in args.variant:
        name, patterns = item.split(":", 1)
        # Never train on a dev set (compared by file name: glob mixes / and \ on Windows).
        files = sorted(f for f in {f for p in patterns.split(",") for f in glob.glob(p)}
                       if Path(f).stem not in {f"{d}_eval" for d in DEV_FILES})
        parts = [np.load(f) for f in files]
        x = torch.cat([flat(p["features"]) for p in parts])
        # Label 1 = Muse (edits and pair fakes); 2 never occurs in training lists; FLUX etc. are not in these files.
        y = torch.from_numpy(np.concatenate([p["labels"] for p in parts]).astype(np.float32).clip(0, 1))
        head = LinearHead().fit(x, y, weight_decay=args.weight_decay)
        print(f"\n{name}: {len(files)} files, {len(y)} views ({int(y.sum())} Muse, {int((y == 0).sum())} real), "
              f"train AUC {metrics.auc(head.logits(x).numpy(), y.numpy() == 1):.3f}")
        score = {n: head.logits(v).numpy() for n, v in {**reals, **fakes}.items()}
        edits = np.r_[score["edits v2 local"], score["edits v2 whole"]]
        print(f"  AUC originals v2 vs edits v2: {metrics.auc(np.r_[score['originals v2'], edits], np.r_[np.zeros(len(score['originals v2'])), np.ones(len(edits))] == 1):.3f}")
        for rate in args.rates:
            line = max(threshold_for(score[n], rate) for n in reals)
            print(f"  line: at most {rate:.0%} of every real pool (p={1 / (1 + math.exp(-line)):.3f})")
            print("    real called edited: " + " | ".join(f"{n} {(score[n] > line).mean():.1%}" for n in reals))
            print("    AI caught:          " + " | ".join(f"{n} {(score[n] > line).mean():.1%}" for n in fakes))
        if args.save_dir:
            args.save_dir.mkdir(parents=True, exist_ok=True)
            head.save(args.save_dir / f"head_{name}.pt")


if __name__ == "__main__":
    main()
