"""Image lists for the open-model fakes (scripts/generate_open_models.py, made on Kaggle).

Every fake was made from the caption of a pair photo (ids like v1_pair_0004 = pair_0004 of
D:/Datasets/MuseImage/pairs_v1), so it has a real photo with the same content.

  - training (open_gen_train.csv): fakes of the --train-models whose real photo is a TRAINING pair
    photo. Like the Muse pairs, each fake is centre-cropped to its real photo's aspect ratio and resized
    to its size (scripts/make_pair_lists.py), so size can't tell the classes apart.
  - dev (open_gen_dev.csv): every fake of the --dev-models as generated (PixArt-Sigma: it shares SDXL's
    decoder, so it never trains; a second generator the head has never seen), plus the --train-models
    fakes of DEV pair photos, as generated (a known generator, new captions).

Usage:
  python scripts/make_open_gen_lists.py --gen-dir D:/Datasets/OpenGen/v1
      --pairs-dir v1=D:/Datasets/MuseImage/pairs_v1 --pairs-dir v2=D:/Datasets/MuseImage/pairs_v2
"""

import argparse
import csv
from pathlib import Path

from make_pair_lists import matched_fake


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--gen-dir", type=Path, required=True)
    parser.add_argument("--pairs-dir", action="append", required=True, help='"v1=path": pair round and its folder')
    parser.add_argument("--train-models", nargs="+", default=["sd35m", "k22"])
    parser.add_argument("--dev-models", nargs="+", default=["pixart"])
    parser.add_argument("--out-dir", type=Path, default=Path("features/lists/open_gen_v1"))
    args = parser.parse_args()

    pairs = {}
    for item in args.pairs_dir:
        round_name, folder = item.split("=", 1)
        with open(Path(folder) / "pairs.csv", newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                pairs[f"{round_name}_{row['pair_id']}"] = {**row, "folder": Path(folder)}

    train, dev = [], []
    for model in args.train_models + args.dev_models:
        with open(args.gen_dir / model / "manifest.csv", newline="") as f:
            made = [r for r in csv.DictReader(f) if r["status"] == "ok"]
        matched_dir = args.gen_dir / f"{model}_matched"
        for r in made:
            pair, path = pairs[r["id"]], args.gen_dir / model / r["file"]
            if model in args.dev_models:
                dev.append({"path": str(path), "label": 1, "source": f"open_gen/{model}"})
            elif pair["split"] == "train":
                matched_dir.mkdir(exist_ok=True)
                out = matched_dir / r["file"]
                if not out.exists():
                    matched_fake(path, int(pair["width"]), int(pair["height"]), out)
                train.append({"path": str(out), "label": 1, "source": f"open_gen/{model}"})
            else:
                dev.append({"path": str(path), "label": 1, "source": f"open_gen/{model}_devcaption"})

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in (("open_gen_train", train), ("open_gen_dev", dev)):
        with open(args.out_dir / f"{name}.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["path", "label", "source"], lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
        counts = {}
        for r in rows:
            counts[r["source"]] = counts.get(r["source"], 0) + 1
        print(f"{name}: {len(rows)} images {counts}")


if __name__ == "__main__":
    main()
