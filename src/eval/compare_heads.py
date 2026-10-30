"""Train several head variants and compare them on the same test sets, in one table.

Each variant is a name and one or more feature-file globs (training data); each test is a name and
one eval feature file. Sets with real and AI images get AUC, false alarms and AI caught at 0.5, and
AI caught at 5% false alarms; AI-only sets get AI caught at 0.5 and at a reference threshold: the
score above which 5% of the real images of the --reference test fall (indicative only).

Usage (from the repo root):
  python -m src.eval.compare_heads
      --variant "B:features/aug_v2/train-*_train.npz"
      --variant "C:features/aug_v2/train-*_train.npz,features/aug_v2/muse_pairs_train_train.npz"
      --test "Chameleon random:features/chameleon_random_eval.npz" --test "Muse test:features/muse_image_test_v1_eval.npz"
      --reference "Chameleon random" --save-dir weights/compare
"""

import argparse
import glob
from pathlib import Path

import numpy as np
import torch

from src.eval import metrics
from src.models.linear_head import LinearHead


def load(paths):
    parts = [np.load(p) for p in paths]
    x = torch.from_numpy(np.concatenate([p["features"] for p in parts]).astype(np.float32))
    y = torch.from_numpy(np.concatenate([p["labels"] for p in parts]).astype(np.float32))
    return x, y


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--variant", action="append", required=True, help='"NAME:glob1,glob2"')
    parser.add_argument("--test", action="append", required=True, help='"NAME:path"')
    parser.add_argument("--reference", help="test whose real images set the reference threshold")
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--save-dir", type=Path)
    args = parser.parse_args()

    tests = {}
    for item in args.test:
        name, path = item.split(":", 1)
        tests[name] = load([path])
    reference = None
    if args.reference:
        x, y = tests[args.reference]
        reference = x[y == 0]

    for item in args.variant:
        name, patterns = item.split(":", 1)
        files = sorted({f for pattern in patterns.split(",") for f in glob.glob(pattern)})
        x, y = load(files)
        head = LinearHead().fit(x, y, weight_decay=args.weight_decay)
        print(f"{name}: {len(files)} files, {len(y)} training vectors ({int(y.sum())} AI)")
        t5 = None if reference is None else float(np.quantile(head.logits(reference).numpy(), 0.95))
        for tname, (xt, yt) in tests.items():
            s = head.logits(xt).numpy()
            is_ai = yt.numpy() == 1
            at = metrics.rates_at(s, is_ai, threshold=0.0)
            if (~is_ai).any():
                line = (f"AUC {metrics.auc(s, is_ai):.3f} | real called AI {at['false_alarms']:5.1%} | AI caught "
                        f"{at['caught']:5.1%} | caught at 5% FA {metrics.caught_at_false_alarm_rate(s, is_ai):5.1%}")
            else:
                line = f"AI caught {at['caught']:5.1%}"
                if t5 is not None:
                    line += f" | at the {args.reference} 5%-FA threshold {(s > t5).mean():5.1%}"
            print(f"    {tname:<16} {line}")
        if args.save_dir:
            args.save_dir.mkdir(parents=True, exist_ok=True)
            head.save(args.save_dir / f"head_{name}.pt")


if __name__ == "__main__":
    main()
