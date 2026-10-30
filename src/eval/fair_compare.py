"""Fair head comparison (plan Section 0, decided 31 Oct): every head gets its own threshold.

For each head the threshold is set so that at most 5% of the real photos in EACH real dev pool are
called AI (e.g. my phone photos and the polished Commons dev set); the strictest pool decides.
Then we compare how many fakes each head catches at its own threshold. The feature files must
come from the app's preprocessing (python -m src.features.extract --mode app), so the numbers are
what the app would do. Only dev sets belong here, never Chameleon or a final test set.

Real pools use the real images (label 0) of their files, fake sets the AI images (label 1).
A real pool may be private (data/private/...): only its saved feature vectors are read, never
the photos, and only totals are printed.

Usage (from the repo root):
  python -m src.eval.fair_compare --head "app:weights/head_path_a_sidset30.pt" --head "C:weights/muse_pairs_v1/head_C.pt"
      --real "phone:data/private/phone_v1_app.npz" --real "polished dev:features/commons_v1_dev_app.npz"
      --fake "Muse test v1:features/muse_image_test_v1_app.npz" --fake "Muse pairs val:features/muse_pairs_val_app.npz"
"""

import argparse
import math
from pathlib import Path

import numpy as np
import torch

from src.models.linear_head import LinearHead

MAX_FALSE_ALARMS = 0.05


def load(path, label):
    data = np.load(path)
    keep = data["labels"] == label
    return torch.from_numpy(data["features"][keep].astype(np.float32))


def threshold_for(scores, max_rate=MAX_FALSE_ALARMS):
    """The lowest threshold (on the logit) with at most max_rate of these real scores above it."""
    allowed = math.floor(max_rate * len(scores))  # how many real photos may be above
    return float(np.sort(scores)[::-1][allowed])


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--head", action="append", required=True, help='"NAME:path to a saved head"')
    parser.add_argument("--real", action="append", required=True, help='"NAME:feature file" (its real images)')
    parser.add_argument("--fake", action="append", required=True, help='"NAME:feature file" (its AI images)')
    args = parser.parse_args()

    split = lambda item: item.split(":", 1)  # noqa: E731
    reals = {name: load(path, 0) for name, path in map(split, args.real)}
    fakes = {name: load(path, 1) for name, path in map(split, args.fake)}
    print(f"Own threshold per head: at most {MAX_FALSE_ALARMS:.0%} of the real photos in every real pool called AI.")
    print("Real pools: " + ", ".join(f"{n} ({len(x)})" for n, x in reals.items()))
    print("Fake sets:  " + ", ".join(f"{n} ({len(x)})" for n, x in fakes.items()) + "\n")

    width = max(len(n) for n in list(reals) + list(fakes)) + 2
    header = f"{'head':<8}{'threshold':>10} |" + "".join(f"{n:>{width}}" for n in reals) + " |"
    header += "".join(f"{n:>{width}}" for n in fakes)
    print(" " * 20 + "real called AI".center(width * len(reals)) + "   " + "AI caught".center(width * len(fakes)))
    print(header)
    for name, path in map(split, args.head):
        head = LinearHead.load(Path(path))
        t = max(threshold_for(head.logits(x).numpy()) for x in reals.values())
        line = f"{name:<8}{1 / (1 + math.exp(-t)):>10.3f} |"
        line += "".join(f"{(head.logits(x).numpy() > t).mean():>{width}.1%}" for x in reals.values()) + " |"
        line += "".join(f"{(head.logits(x).numpy() > t).mean():>{width}.1%}" for x in fakes.values())
        print(line)
    print("\nthreshold = the AI probability above which the app would say AI, chosen per head on these dev pools")


if __name__ == "__main__":
    main()
