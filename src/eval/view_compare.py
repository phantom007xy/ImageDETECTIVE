"""Whole-image views compared on the SAME head: does the app's view (the whole photo shrunk to long side 448) lose the
fine detail that the detectors in the literature rely on (CNNDetection: uncropped beats resized; B-Free, SSP: crops at
native resolution)?

Views of one photo, all scored by the linear head on DINOv3's mean patch vector (the app's head is linear, so the mean
of several tiles' logits is exactly the head on the mean of their patch vectors):
  long448    the app's view: long side 448 (results/calibration: every line is set on this view)
  short448   short side 448, long side at most 896 (preprocess.view_size "short448")
  half       the photo resized to short side 896, covered by 448 x 448 tiles (2 x k): each tile shows half of the short
             side, just below the 60-100% squares the head was trained on; "mean" = mean of the tile logits,
             "max" = the strongest tile
  native     448 x 448 tiles at the photo's own resolution (no resizing; a 12 MP phone photo gives ~60 tiles); mean, max
Each tile is saved as JPEG 90 like every app view. A photo smaller than a tile falls back to long448.

Comparison (src/eval/fair_compare rule): per view and combination, the line is set so that at most --rate of the real
photos of EVERY real set are above it; then the share of each fake set above that line. Also the AUC of each fake set
against all real sets together. Private sets (folders): totals only.

Usage (from the repo root):
  python -m src.eval.view_compare --head weights/google_v1/head_G3tt.pt
      --real "polished dev:features/lists/commons_dev.csv" --real "phone:data/private/phone_v1"
      --fake "Nano Banana dev:features/lists/bitmind_nb_dev.csv" ... --out results/views_v1.txt
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from src.data import preprocess
from src.eval import metrics
from src.eval.image_sets import open_upright, parse_set
from src.models.backbone import Backbone
from src.models.linear_head import LinearHead

TILE = preprocess.MODEL_SIZE  # 448
VIEWS = ["long448", "short448", "half mean", "half max", "native mean", "native max"]


def tile_boxes(w, h, side):
    """Evenly spaced side x side tiles covering a w x h picture (at least one per axis)."""
    nx, ny = max(1, round(w / side)), max(1, round(h / side))
    xs = [round(i * (w - side) / (nx - 1)) if nx > 1 else (w - side) // 2 for i in range(nx)]
    ys = [round(j * (h - side) / (ny - 1)) if ny > 1 else (h - side) // 2 for j in range(ny)]
    return [(x, y, x + side, y + side) for y in ys for x in xs]


def tiles(image, short_side=None):
    """448-px tiles of the picture resized to `short_side` (None: native size); [] when it is smaller than a tile."""
    w, h = image.size
    if short_side and min(w, h) > short_side:
        s = short_side / min(w, h)
        image = image.resize((round(w * s), round(h * s)), Image.BICUBIC)
        w, h = image.size
    if min(w, h) < TILE:
        return []
    return [preprocess.jpeg(image.crop(box), 90) for box in tile_boxes(w, h, TILE)]


@torch.no_grad()
def logits_of(backbone, head, views):
    """Head logits of a list of PIL views (any sizes: one forward pass per distinct size)."""
    out = []
    by_size = {}
    for i, v in enumerate(views):
        by_size.setdefault(v.size, []).append(i)
    result = [None] * len(views)
    for size, idx in by_size.items():
        pixels = torch.stack([preprocess.to_tensor(views[i]) for i in idx])
        for start in range(0, len(idx), 32):
            feats = backbone.layer_means(pixels[start:start + 32])["features"].float().cpu()
            for k, logit in zip(idx[start:start + 32], head.logits(feats).reshape(-1).tolist()):
                result[k] = logit
    return result


def score_picture(backbone, head, image):
    """The six view scores of one picture."""
    scores = {}
    base = preprocess.inference_view(image, "long448")
    scores["long448"] = logits_of(backbone, head, [base])[0]
    scores["short448"] = logits_of(backbone, head, [preprocess.inference_view(image, "short448")])[0]
    for name, short in (("half", 2 * TILE), ("native", None)):
        views = tiles(image, short) or [base]
        logits = logits_of(backbone, head, views)
        scores[f"{name} mean"], scores[f"{name} max"] = float(np.mean(logits)), float(np.max(logits))
        scores[f"{name} tiles"] = len(views)
    return scores


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--head", type=Path, required=True)
    parser.add_argument("--real", action="append", required=True, help='"NAME:list.csv[@prefix]" or "NAME:folder"')
    parser.add_argument("--fake", action="append", required=True)
    parser.add_argument("--rate", type=float, default=0.05, help="real photos above the line, per real set")
    parser.add_argument("--limit", type=int, help="first N pictures of every set (quick test)")
    parser.add_argument("--save-dir", type=Path, default=Path("features/views_v1"), help="per-set scores (JSON)")
    parser.add_argument("--out", type=Path, help="append the report here")
    args = parser.parse_args()

    backbone, head = Backbone(), LinearHead.load(args.head)
    args.save_dir.mkdir(parents=True, exist_ok=True)
    sets = {}  # name -> (is_fake, {view: scores})
    start = time.time()
    for is_fake, specs in ((False, args.real), (True, args.fake)):
        for spec in specs:
            name, paths = parse_set(spec)
            paths = paths[: args.limit] if args.limit else paths
            cache = args.save_dir / f"{name.replace(' ', '_').replace('/', '_')}.json"
            if cache.exists() and not args.limit:
                rows = json.loads(cache.read_text())
            else:
                rows = []
                for p in paths:
                    try:
                        rows.append(score_picture(backbone, head, open_upright(p)))
                    except Exception as e:  # an unreadable file is skipped, never a crash
                        print(f"  skipped {Path(p).name}: {type(e).__name__}", flush=True)
                if not args.limit:
                    cache.write_text(json.dumps(rows))
            sets[name] = (is_fake, {v: np.array([r[v] for r in rows]) for v in VIEWS},
                          {k: float(np.mean([r[k] for r in rows])) for k in ("half tiles", "native tiles")})
            print(f"{name}: {len(rows)} pictures ({time.time() - start:.0f} s)", flush=True)

    lines = [f"{args.head.name}: whole-image views, line = at most {args.rate:.0%} of the real photos of every real set "
             f"above it (the fair_compare rule); AUC = the fake set against all real sets together",
             "set" + "".ljust(26) + "".join(f"{v:>14}" for v in VIEWS)]
    reals = {n: s for n, (f, s, _) in sets.items() if not f}
    fakes = {n: s for n, (f, s, _) in sets.items() if f}
    line = {v: max(float(np.quantile(s[v], 1 - args.rate)) for s in reals.values()) for v in VIEWS}
    lines.append("line (logit)".ljust(29) + "".join(f"{line[v]:>14.2f}" for v in VIEWS))
    for n, s in reals.items():
        lines.append(f"REAL {n}".ljust(29) + "".join(f"{(s[v] > line[v]).mean():>13.1%} " for v in VIEWS))
    all_real = {v: np.concatenate([s[v] for s in reals.values()]) for v in VIEWS}
    for n, s in fakes.items():
        caught = "".join(f"{(s[v] > line[v]).mean():>13.1%} " for v in VIEWS)
        auc = "".join(f"{metrics.auc(np.r_[s[v], all_real[v]], np.r_[np.ones(len(s[v])), np.zeros(len(all_real[v]))]):>14.3f}"
                      for v in VIEWS)
        lines.append(f"AI   {n}".ljust(29) + caught)
        lines.append("     (AUC)".ljust(29) + auc)
    lines.append("tiles per picture (mean): " + ", ".join(f"{n} half {t['half tiles']:.0f} / native {t['native tiles']:.0f}"
                                                            for n, (_, _, t) in sets.items()))
    report = "\n".join(lines)
    print(report)
    if args.out:
        with open(args.out, "a", encoding="utf-8") as f:
            f.write(report + "\n\n")


if __name__ == "__main__":
    main()
