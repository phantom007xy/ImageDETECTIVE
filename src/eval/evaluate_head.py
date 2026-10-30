"""Evaluate a saved Path A head on feature files from other datasets.

Reports real and AI separately: with both classes, AUC, false alarms and AI caught at threshold
0.5, and AI caught at 5% false alarms; with AI images only, the share caught. Each test set is
also broken down by source (e.g. the SD version), since one generator can hide another.

Usage (from the repo root):
  python -m src.eval.evaluate_head --head weights/head_path_a_sidset30.pt --test features/chameleon_random_eval.npz ...
"""

import argparse
from pathlib import Path

import numpy as np
import torch

from src.eval import metrics
from src.models.linear_head import LinearHead


def report(name, scores, is_ai):
    has_real, has_ai = (~is_ai).any(), is_ai.any()
    at_half = metrics.rates_at(scores, is_ai, threshold=0.0)  # logit 0 = probability 0.5
    parts = [f"{name:<34} {int((~is_ai).sum()):>5} real {int(is_ai.sum()):>5} AI"]
    if has_real and has_ai:
        parts.append(f"AUC {metrics.auc(scores, is_ai):.3f}")
    if has_real:
        parts.append(f"real called AI {at_half['false_alarms']:.1%}")
    if has_ai:
        parts.append(f"AI caught {at_half['caught']:.1%}")
    if has_real and has_ai:
        parts.append(f"AI caught at 5% false alarms {metrics.caught_at_false_alarm_rate(scores, is_ai):.1%}")
    print(" | ".join(parts))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--head", type=Path, required=True)
    parser.add_argument("--test", type=Path, nargs="+", required=True)
    args = parser.parse_args()

    head = LinearHead.load(args.head)
    for path in args.test:
        data = np.load(path)
        x = torch.from_numpy(data["features"].astype(np.float32))
        scores = head.logits(x).numpy()
        is_ai = data["labels"] == 1
        report(path.stem, scores, is_ai)
        sources = data["sources"]
        if len(set(sources)) > 2 or not (~is_ai).any():
            for source in sorted(set(sources)):
                keep = sources == source
                report(f"  {source}", scores[keep], is_ai[keep])


if __name__ == "__main__":
    main()
