"""Learning curve for the Path A head: does more SID-Set data still help?

Trains the linear head on the features of the first 5, 10, 20, ... training files and tests every
version on the same unseen test features (e.g. a SID-Set validation file in eval mode). If the
test numbers stop improving, more of the same data is not worth it (plan Section 5.8).

Usage (from the repo root):
  python -m src.eval.learning_curve --train features/train-*_train.npz --test features/validation-00000-of-00034_eval.npz
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
    x = np.concatenate([p["features"] for p in parts]).astype(np.float32)
    y = np.concatenate([p["labels"] for p in parts]).astype(np.float32)  # 1 = AI (synthetic), 0 = real
    images = sum(len(set(p["img_ids"])) for p in parts)
    return torch.from_numpy(x), torch.from_numpy(y), images


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--train", nargs="+", required=True, help="training feature files (globs allowed)")
    parser.add_argument("--test", nargs="+", required=True, help="test feature files (eval mode)")
    parser.add_argument("--steps", type=int, nargs="+", default=[5, 10, 20, 30], help="numbers of training files")
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--save", type=Path, help="save the head trained on the most files here")
    args = parser.parse_args()

    train_files = sorted({f for pattern in args.train for f in glob.glob(pattern)})
    x_test, y_test, test_images = load(args.test)
    is_ai = y_test.numpy() == 1
    print(f"test: {test_images} images ({int((~is_ai).sum())} real, {int(is_ai.sum())} AI vectors)\n")
    print(f"{'files':>5} {'images':>7} | {'train AUC':>9} | {'test AUC':>8} | {'real called AI':>14} "
          f"{'AI caught':>9} {'balanced':>8} (threshold 0.5) | {'AI caught at 5% false alarms':>28}")

    head = None
    for n in [s for s in args.steps if s <= len(train_files)]:
        x, y, images = load(train_files[:n])
        head = LinearHead().fit(x, y, weight_decay=args.weight_decay)
        train_auc = metrics.auc(head.logits(x).numpy(), y.numpy() == 1)
        scores = head.logits(x_test).numpy()
        at_half = metrics.rates_at(scores, is_ai, threshold=0.0)  # logit 0 = probability 0.5
        print(f"{n:>5} {images:>7} | {train_auc:>9.3f} | {metrics.auc(scores, is_ai):>8.3f} | "
              f"{at_half['false_alarms']:>14.1%} {at_half['caught']:>9.1%} {at_half['balanced_accuracy']:>8.1%}"
              f"{'':>17} | {metrics.caught_at_false_alarm_rate(scores, is_ai):>28.1%}")

    if args.save and head is not None:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        head.save(args.save)
        print(f"\nsaved the head trained on {n} files to {args.save}")


if __name__ == "__main__":
    main()
