"""Calibrate the Muse-edit check (src/models/train_edit_check.py) jointly with the whole-image head.

One false-alarm budget for the final verdict (plan Section 5.6): a real photo counts as a false alarm if the whole-image
head says AI (its calibrated line, src/calibrate/calibrate.py) OR the edit check says edited. The edit check's line is
the lowest one that keeps
  1. that union at most 5% in EVERY real pool (the head often uses most of the budget already: then the edit check
     may add no extra false alarm there), and
  2. the edit check alone at most --rate (default 0.5%) in every real pool.
The joint trade (--give-up N): the whole-image head gives up its N lowest flags in the pool where it has the least
room left, i.e. its AI line is raised a little, so that a single real outlier in that pool can't block the edit check.
The false alarms per pool stay within the same budget; the raised line is saved too (whole_ai_logit) and the app uses
it together with the edit check (src/inference.load_edit_check). Measured: results/edit_check_v1.txt, v3.
Each pool needs both feature files of the SAME photos in the same order: the whole-image app features
(src/features/extract.py --mode app; my phone photos: data/private/<folder>_app.npz) and the PE-Core features
(src/features/extract_pe.py --mode eval; a fine-tuned PE: --pe-weights; my photos: data/private/<folder>_pe*.npz).
--check pools are real photos that are only reported (an independent check), never used for the lines.
"...@prefix" after a pool keeps only the photos whose source starts with prefix. Only totals are printed.

Usage (from the repo root):
  python -m src.calibrate.calibrate_edit_check --head weights/edit_check_v3/head_EC3.pt
      --pe-weights weights/edit_check_v3/pe_EC3.pt --whole-head weights/open_gen_v1/head_F.pt --give-up 1
      --pool "polished dev:features/commons_dev_app.npz:features/pe_ec3/commons_dev_eval.npz" ...
      --fake "edits v2 local:features/muse_edits_test_v2_app.npz:features/pe_ec3/muse_edits_test_v2_eval.npz@muse_edits/local"
      --out weights/edit_check_v3/head_EC3_calibration.json
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from src.eval.fair_compare import threshold_for
from src.inference import load_calibration
from src.models.linear_head import LinearHead

MAX_FALSE_ALARMS = 0.05  # the shared budget of the final verdict in every real pool


def paired(whole, edit, spec, real):
    """(whole-image logits, edit logits) of the real (label 0) or AI (label > 0) photos of a pool."""
    spec, _, prefix = spec.partition("@")
    whole_file, pe_file = spec.rsplit(":", 1)
    a, b = np.load(whole_file), np.load(pe_file)
    if not (len(a["labels"]) == len(b["labels"]) and (a["img_ids"] == b["img_ids"]).all()):
        raise ValueError(f"{whole_file} and {pe_file} don't hold the same photos in the same order")
    keep = a["labels"] == 0 if real else a["labels"] > 0
    if prefix:
        keep &= np.char.startswith(a["sources"].astype(str), prefix)
    w = whole.logits(torch.from_numpy(a["features"][keep].astype(np.float32))).numpy()
    x = b["features"][keep].astype(np.float32)
    return w, edit.logits(torch.from_numpy(x.reshape(len(x), -1))).numpy()


def traded_line(pools, line, give_up):
    """The whole-image head's AI line after giving up its `give_up` lowest flags in its tightest pool."""
    room = {n: int(np.floor(MAX_FALSE_ALARMS * len(w))) - int((w > line).sum()) for n, (w, _) in pools.items()}
    tightest = min(room, key=room.get)
    flagged = np.sort(pools[tightest][0][pools[tightest][0] > line])[::-1]
    if give_up == 0 or len(flagged) < give_up:
        return line, None
    return float(flagged[len(flagged) - give_up]), tightest  # flags are strictly above the line


def edit_line(pools, whole_line, rate):
    """The lowest edit line that keeps (whole OR edit) <= 5% and the edit check alone <= rate in every pool."""
    line = max(threshold_for(e, rate) for _, e in pools.values())
    for w, e in pools.values():
        whole_flag = w > whole_line
        room = int(np.floor(MAX_FALSE_ALARMS * len(w))) - int(whole_flag.sum())  # false alarms this pool can still take
        rest = np.sort(e[~whole_flag])[::-1]
        if room < len(rest):
            line = max(line, float(rest[max(room, 0)]))
    return line


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--head", type=Path, required=True, help="the edit check's head")
    parser.add_argument("--pe-weights", type=Path, help="the fine-tuned PE its features come from (default: the original)")
    parser.add_argument("--whole-head", type=Path, required=True, help="the whole-image head (with its calibration)")
    parser.add_argument("--pool", action="append", required=True, help='"NAME:whole-image app file:PE file" (real photos)')
    parser.add_argument("--check", action="append", default=[], help='like --pool, but only reported (not used for the lines)')
    parser.add_argument("--fake", action="append", default=[], help='"NAME:whole-image app file:PE file" (for the report)')
    parser.add_argument("--rate", type=float, default=0.005)
    parser.add_argument("--give-up", type=int, default=0, help="the joint trade: flags the whole-image head gives up")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    edit, whole, c = LinearHead.load(args.head), LinearHead.load(args.whole_head), load_calibration(args.whole_head)
    split = lambda item: item.split(":", 1)  # noqa: E731
    pools = {name: paired(whole, edit, spec, True) for name, spec in map(split, args.pool)}
    checks = {name: paired(whole, edit, spec, True) for name, spec in map(split, args.check)}
    fakes = {name: paired(whole, edit, spec, False) for name, spec in map(split, args.fake)}

    whole_line, tightest = traded_line(pools, c["ai_logit"], args.give_up)
    line = edit_line(pools, whole_line, args.rate)

    print(f"edit check {args.head.name}{' (PE ' + args.pe_weights.name + ')' if args.pe_weights else ''}: edited if "
          f"logit > {line:+.2f}; with {args.whole_head.name}: AI if logit > {whole_line:+.2f}"
          + (f" (its calibrated line {c['ai_logit']:+.2f}, raised: it gives up {args.give_up} flag(s) in {tightest})"
             if tightest else ""))
    for title, group in (("REAL", pools), ("CHECK", checks), ("AI", fakes)):
        for name, (w, e) in group.items():
            print(f"  {title:<5} {name:<20}{len(w):>5} | whole-image AI {(w > whole_line).mean():5.1%} | edited "
                  f"{(e > line).mean():5.1%} | together {((w > whole_line) | (e > line)).mean():5.1%}")
    if args.out:
        args.out.write_text(json.dumps({
            "head": args.head.as_posix(), "pe_weights": args.pe_weights.as_posix() if args.pe_weights else None,
            "whole_head": args.whole_head.as_posix(), "edit_logit": float(line), "rate": args.rate,
            "give_up": args.give_up, "whole_ai_logit": whole_line, "whole_ai_logit_calibrated": c["ai_logit"],
            "real_pools": {n: len(w) for n, (w, _) in pools.items()},
            "check_pools": {n: len(w) for n, (w, _) in checks.items()}}, indent=2))
        print(f"saved {args.out}")


if __name__ == "__main__":
    main()
