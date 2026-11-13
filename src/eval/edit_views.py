"""Crops for the Muse-edit check compared on the SAME fine-tuned Perception Encoder and head: the app scores one
centre square at 336 px, so a small local edit near the picture's edge is never seen, and fine detail is shrunk away
(B-Free, SSP: average several crops at a higher resolution).

Crop schemes of one picture (every crop 336 x 336, JPEG 90):
  centre      the app's view: the largest centre square (the edit check was trained on 60-100% squares)
  three       the three largest squares along the long side (start, centre, end): mean and max of their logits
  grid70      squares of 70% of the short side on a 2 x k grid (inside the training range, overlapping): mean, max
  half        squares of 50% of the short side on a 2 x k grid (just below the training range): mean, max
  any         max over the centre square and the half tiles
Line rule (src/calibrate/calibrate_edit_check): at most --rate of the real photos of EVERY real set above the line.
Reported per scheme: the real sets' shares above the line, each fake set's share caught, and its AUC against its own
originals when a set of originals is named with "=ORIGINALS" (e.g. "edits v2 local:...@muse_edits/local=originals v2").

Usage (from the repo root):
  python -m src.eval.edit_views --head weights/edit_check_v3/head_EC3pf.pt --pe-weights weights/edit_check_v3/pe_EC3.pt
      --real "polished dev:features/lists/commons_dev.csv" --real "originals v2:features/lists/muse_edits_test_v2.csv@muse_edits/original"
      --fake "edits v2 local:features/lists/muse_edits_test_v2.csv@muse_edits/local=originals v2" ... --out results/edit_views_v1.txt
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from src.data import preprocess
from src.eval import metrics
from src.eval.image_sets import open_upright, parse_set
from src.models.linear_head import LinearHead
from src.models.pe_core import PECore

SCHEMES = ["centre", "three mean", "three max", "grid70 mean", "grid70 max", "half mean", "half max", "any"]


def squares(image, fraction, size):
    """Squares of `fraction` of the short side on a 2 x k grid (k along the long side), resized to size."""
    w, h = image.size
    side = max(16, round(min(w, h) * fraction))
    nx, ny = (max(2, round(w / side)), 2) if w >= h else (2, max(2, round(h / side)))
    xs = [round(i * (w - side) / (nx - 1)) for i in range(nx)]
    ys = [round(j * (h - side) / (ny - 1)) for j in range(ny)]
    return [preprocess.jpeg(preprocess.square_crop(image, side, x, y, size), 90) for y in ys for x in xs]


def three(image, size):
    w, h = image.size
    side = min(w, h)
    return [preprocess.jpeg(preprocess.square_crop(image, side, round((w - side) * f), round((h - side) * f), size), 90)
            for f in (0.0, 0.5, 1.0)]


@torch.no_grad()
def logits_of(pe, head, views):
    out = []
    for start in range(0, len(views), 16):
        pixels = torch.stack([pe.to_tensor(v) for v in views[start:start + 16]])
        feats = pe.layer_features(pixels).reshape(len(pixels), -1).float()
        out += head.logits(feats).reshape(-1).tolist()
    return out


def score_picture(pe, head, image):
    s = {}
    centre = logits_of(pe, head, [preprocess.eval_view(image, pe.size)])[0]
    s["centre"] = centre
    for name, views in (("three", three(image, pe.size)), ("grid70", squares(image, 0.7, pe.size)),
                        ("half", squares(image, 0.5, pe.size))):
        logits = logits_of(pe, head, views)
        s[f"{name} mean"], s[f"{name} max"], s[f"{name} n"] = float(np.mean(logits)), float(np.max(logits)), len(logits)
    s["any"] = max(centre, s["half max"])
    return s


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--head", type=Path, required=True)
    parser.add_argument("--pe-weights", type=Path, required=True)
    parser.add_argument("--real", action="append", required=True, help='"NAME:list.csv[@prefix]" or "NAME:folder"')
    parser.add_argument("--fake", action="append", required=True, help='"NAME:list.csv[@prefix][=ORIGINALS set name]"')
    parser.add_argument("--rate", type=float, default=0.02, help="real photos above the line, per real set")
    parser.add_argument("--limit", type=int, help="first N pictures of every set (quick test)")
    parser.add_argument("--save-dir", type=Path, default=Path("features/edit_views_v1"))
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    pe, head = PECore(tuned_path=args.pe_weights), LinearHead.load(args.head)
    args.save_dir.mkdir(parents=True, exist_ok=True)
    sets, originals_of = {}, {}
    start = time.time()
    for is_fake, specs in ((False, args.real), (True, args.fake)):
        for spec in specs:
            spec, _, originals = spec.partition("=")
            name, paths = parse_set(spec)
            if originals:
                originals_of[name] = originals
            paths = paths[: args.limit] if args.limit else paths
            cache = args.save_dir / f"{name.replace(' ', '_').replace('/', '_')}.json"
            if cache.exists() and not args.limit:
                rows = json.loads(cache.read_text())
            else:
                rows = []
                for p in paths:
                    try:
                        rows.append(score_picture(pe, head, open_upright(p)))
                    except Exception as e:
                        print(f"  skipped {Path(p).name}: {type(e).__name__}", flush=True)
                if not args.limit:
                    cache.write_text(json.dumps(rows))
            sets[name] = (is_fake, {v: np.array([r[v] for r in rows]) for v in SCHEMES},
                          {k: float(np.mean([r[k] for r in rows])) for k in ("grid70 n", "half n")})
            print(f"{name}: {len(rows)} pictures ({time.time() - start:.0f} s)", flush=True)

    reals = {n: s for n, (f, s, _) in sets.items() if not f}
    fakes = {n: s for n, (f, s, _) in sets.items() if f}
    line = {v: max(float(np.quantile(s[v], 1 - args.rate)) for s in reals.values()) for v in SCHEMES}
    lines = [f"{args.head.name} on {args.pe_weights.name}: crop schemes for the edit check, line = at most {args.rate:.0%} "
             f"of the real photos of every real set above it",
             "set".ljust(26) + "".join(f"{v:>13}" for v in SCHEMES),
             "line (logit)".ljust(26) + "".join(f"{line[v]:>13.2f}" for v in SCHEMES)]
    for n, s in reals.items():
        lines.append(f"REAL {n}".ljust(26) + "".join(f"{(s[v] > line[v]).mean():>12.1%} " for v in SCHEMES))
    for n, s in fakes.items():
        lines.append(f"AI   {n}".ljust(26) + "".join(f"{(s[v] > line[v]).mean():>12.1%} " for v in SCHEMES))
        if n in originals_of and originals_of[n] in reals:
            o = reals[originals_of[n]]
            lines.append(f"     AUC vs {originals_of[n]}".ljust(26)
                         + "".join(f"{metrics.auc(np.r_[s[v], o[v]], np.r_[np.ones(len(s[v])), np.zeros(len(o[v]))]):>13.3f}"
                                   for v in SCHEMES))
    lines.append("crops per picture (mean): " + ", ".join(f"{n} grid70 {t['grid70 n']:.0f} / half {t['half n']:.0f}"
                                                            for n, (_, _, t) in sets.items()))
    report = "\n".join(lines)
    print(report)
    if args.out:
        with open(args.out, "a", encoding="utf-8") as f:
            f.write(report + "\n\n")


if __name__ == "__main__":
    main()
