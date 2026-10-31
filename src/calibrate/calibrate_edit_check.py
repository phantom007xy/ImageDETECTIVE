"""Calibrate the Muse-edit check (src/models/train_edit_check.py) jointly with the whole-image head.

One false-alarm budget for the final verdict (plan Section 5.6): a real photo counts as a false alarm if the whole-image
head says AI (its calibrated line, src/calibrate/calibrate.py) OR the edit check says edited. The edit check's line is
the lowest one that keeps
  1. that union at most 5% in EVERY real pool (the head often uses most of the budget already: then the edit check
     may add no extra false alarm there), and
  2. the edit check alone at most --rate (default 0.5%) in every real pool.
Each pool needs both feature files of the SAME photos in the same order: the whole-image app features
(src/features/extract.py --mode app; my phone photos: data/private/<folder>_app.npz) and the PE-Core features
(src/features/extract_pe.py --mode eval; my photos: data/private/<folder>_pe.npz). Only totals are printed.

Usage (from the repo root):
  python -m src.calibrate.calibrate_edit_check --head weights/edit_check_v1/head_EC_both.pt
      --whole-head weights/open_gen_v1/head_F.pt
      --pool "polished dev:features/commons_dev_app.npz:features/pe/commons_dev_eval.npz" ...
      --fake "edits v2:features/muse_edits_test_v2_app.npz:features/pe/muse_edits_test_v2_eval.npz"
      --out weights/edit_check_v1/head_EC_both_calibration.json
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from src.eval.fair_compare import threshold_for
from src.inference import load_calibration
from src.models.linear_head import LinearHead


def paired(whole, edit, spec, real):
    """(whole-image logits, edit logits) of the real (label 0) or AI (label > 0) photos of a pool."""
    whole_file, pe_file = spec.rsplit(":", 1)
    a, b = np.load(whole_file), np.load(pe_file)
    if not (len(a["labels"]) == len(b["labels"]) and (a["img_ids"] == b["img_ids"]).all()):
        raise ValueError(f"{whole_file} and {pe_file} don't hold the same photos in the same order")
    keep = a["labels"] == 0 if real else a["labels"] > 0
    w = whole.logits(torch.from_numpy(a["features"][keep].astype(np.float32))).numpy()
    x = b["features"][keep].astype(np.float32)
    return w, edit.logits(torch.from_numpy(x.reshape(len(x), -1))).numpy()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--head", type=Path, required=True, help="the edit check's head")
    parser.add_argument("--whole-head", type=Path, required=True, help="the whole-image head (with its calibration)")
    parser.add_argument("--pool", action="append", required=True, help='"NAME:whole-image app file:PE file" (real photos)')
    parser.add_argument("--fake", action="append", default=[], help='"NAME:whole-image app file:PE file" (for the report)')
    parser.add_argument("--rate", type=float, default=0.005)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    edit, whole, c = LinearHead.load(args.head), LinearHead.load(args.whole_head), load_calibration(args.whole_head)
    split = lambda item: item.split(":", 1)  # noqa: E731
    pools = {name: paired(whole, edit, spec, True) for name, spec in map(split, args.pool)}
    fakes = {name: paired(whole, edit, spec, False) for name, spec in map(split, args.fake)}

    line = max(threshold_for(e, args.rate) for _, e in pools.values())
    for w, e in pools.values():
        whole_flag = w > c["ai_logit"]
        room = int(np.floor(0.05 * len(w))) - int(whole_flag.sum())  # false alarms this pool can still take
        rest = np.sort(e[~whole_flag])[::-1]
        if room < len(rest):
            line = max(line, float(rest[max(room, 0)]))

    print(f"edit check {args.head.name}: edited if logit > {line:+.2f}; with {args.whole_head.name} "
          f"(AI if logit > {c['ai_logit']:+.2f})")
    for name, (w, e) in pools.items():
        print(f"  REAL {name:<18}{len(w):>5} | whole-image AI {(w > c['ai_logit']).mean():5.1%} | edited {(e > line).mean():5.1%}"
              f" | together {((w > c['ai_logit']) | (e > line)).mean():5.1%}")
    for name, (w, e) in fakes.items():
        print(f"  AI   {name:<18}{len(w):>5} | whole-image AI {(w > c['ai_logit']).mean():5.1%} | edited {(e > line).mean():5.1%}"
              f" | together {((w > c['ai_logit']) | (e > line)).mean():5.1%}")
    if args.out:
        args.out.write_text(json.dumps({"head": args.head.as_posix(), "whole_head": args.whole_head.as_posix(),
                                        "edit_logit": float(line), "rate": args.rate,
                                        "real_pools": {n: len(w) for n, (w, _) in pools.items()}}, indent=2))
        print(f"saved {args.out}")


if __name__ == "__main__":
    main()
