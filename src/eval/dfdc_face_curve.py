"""Learning curve and first numbers for the DFDC face head (face swaps, type b).

Trains the linear face head on DFDC parts 0, 0-1, 0-2 (features from src/features/extract_dfdc_faces.py)
and tests every version on:
  - DFDC part 3: never trained on (our dev set for face swaps), per face and per video
  - WildDeepfake test: face swaps from internet videos, on its own tighter crops, per face and per sequence
  - real faces from the Commons polished dev set: do clean photo faces get flagged?
  - AI faces (Muse test v1, SD faces): does a swap head also notice fully generated faces?
Head F (the app's current face scorer) is shown for comparison. If part 0-2 is still clearly better than
0-1, more DFDC parts are worth it (plan 31 Oct: learning curve first).

Usage (from the repo root):
  python -m src.eval.dfdc_face_curve --save weights/dfdc_v1/head_dfdc.pt
"""

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from src.eval import metrics
from src.models.linear_head import LinearHead

FACES = Path("features/faces")


def load(name):
    d = np.load(FACES / f"{name}.npz")
    return d["features"].astype(np.float32), d["labels"].astype(int), d["img_ids"]


def per_group(scores, labels, groups):
    """Average score per video / sequence (neighbouring frames are near-copies, so they are not independent)."""
    sums = defaultdict(list)
    for s, lab, g in zip(scores, labels, groups):
        sums[(g, lab)].append(s)
    keys = list(sums)
    return np.array([np.mean(sums[k]) for k in keys]), np.array([k[1] for k in keys])


def report(name, scores, labels, threshold, groups=None):
    is_ai = labels == 1
    line = f"  {name:<26} AUC {metrics.auc(scores, is_ai):.3f}"
    line += f" | at the DFDC 5% line: real called AI {(scores[~is_ai] > threshold).mean():5.1%}"
    line += f", caught {(scores[is_ai] > threshold).mean():5.1%}"
    if groups is not None:
        g_scores, g_labels = per_group(scores, labels, groups)
        line += f" | per video: AUC {metrics.auc(g_scores, g_labels == 1):.3f} ({len(g_labels)} videos)"
    print(line)


def flagged(name, scores, threshold, label):
    print(f"  {name:<26} {'flagged as AI' if label == 0 else 'caught'} at the DFDC 5% line: "
          f"{(scores > threshold).mean():5.1%} of {len(scores)} faces")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--weight-decay", type=float, nargs="+", default=[1e-3])
    parser.add_argument("--save", type=Path, help="save the head trained on all three parts here")
    args = parser.parse_args()

    x_train, y_train, ids_train = load("dfdc_train")
    part_of = np.array([i.split("/")[0] for i in ids_train])
    x_dev, y_dev, ids_dev = load("dfdc_part3_eval")
    video_dev = np.array([i.split("#")[0] for i in ids_dev])
    x_wild, y_wild, ids_wild = load("wilddeepfake_eval")
    seq_wild = np.array([str(Path(i).parent) for i in ids_wild])
    x_commons, _, _ = load("commons_dev_eval")
    x_muse, _, _ = load("muse_image_test_v1_eval")
    x_sd, _, _ = load("sd_faces_eval")

    heads = {"head F (app now)": LinearHead.load("weights/open_gen_v1/head_F.pt")}
    for wd in args.weight_decay:
        for n in (1, 2, 3):
            keep = np.isin(part_of, [f"dfdc_train_part_{p}" for p in range(n)])
            x, y = torch.from_numpy(x_train[keep]), torch.from_numpy(y_train[keep].astype(np.float32))
            name = f"DFDC parts 0-{n - 1} (wd {wd:g})"
            print(f"{name}: {int((y == 0).sum())} real / {int((y == 1).sum())} fake training vectors")
            heads[name] = LinearHead().fit(x, y, weight_decay=wd)

    for name, head in heads.items():
        score = lambda x: head.logits(torch.from_numpy(x)).numpy()  # noqa: E731
        dev = score(x_dev)
        threshold = np.quantile(dev[y_dev == 0], 0.95)  # 5% of DFDC part 3 real faces above it
        print(f"\n{name}")
        report("DFDC part 3 (held out)", dev, y_dev, threshold, video_dev)
        report("WildDeepfake test", score(x_wild), y_wild, threshold, seq_wild)
        flagged("Commons dev real faces", score(x_commons), threshold, 0)
        flagged("Muse test v1 faces", score(x_muse), threshold, 1)
        flagged("SD faces", score(x_sd), threshold, 1)

    if args.save:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        heads[f"DFDC parts 0-2 (wd {args.weight_decay[-1]:g})"].save(args.save)
        print(f"\nsaved {args.save}")


if __name__ == "__main__":
    main()
