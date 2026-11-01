"""The app's full verdict on a labelled image list, totals per source only (e.g. the judge test, or AI images from
Wikimedia Commons per generator).

Runs every picture through src/inference.Detector exactly as the app does (whole image, the Muse-edit check, every
face; the app's calibrated lines), and prints per source how many pictures get each verdict from their PIXELS, how
many carry an AI label in the file, and the final verdict with that label (what the app shows for the file as it is).

PRIVACY: no file name, picture or single score is printed, so it also works for private pictures (my phone photos,
selfies). Fully local: no API calls (the investigator is not run).

Usage (from the repo root):
  python -m src.eval.full_verdicts --list features/lists/commons_ai_v1_photo.csv
"""

import os

os.environ["HF_HUB_OFFLINE"] = "1"  # the models come from the local cache; nothing is sent anywhere
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import argparse  # noqa: E402
import csv  # noqa: E402
from collections import Counter, defaultdict  # noqa: E402
from pathlib import Path  # noqa: E402

from PIL import Image  # noqa: E402

from src.inference import Detector  # noqa: E402

HEAD = "weights/open_gen_v1/head_F.pt"
EDIT_CHECK = "weights/edit_check_v3/head_EC3.pt"
VERDICTS = ["AI-generated", "Likely AI-generated", "Likely AI-edited", "This face looks AI-made or AI-changed",
            "Uncertain", "Likely authentic"]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--head", default=HEAD, help="whole-image head (default: the app's)")
    parser.add_argument("--edit-check", default=EDIT_CHECK, help="edit check head with its calibration (default: the app's)")
    parser.add_argument("--list", type=Path, required=True, help="CSV image list: path, label (0 real, >0 AI), source")
    args = parser.parse_args()

    det = Detector(args.head, edit_check=args.edit_check)
    if det.edit_check is None:
        raise SystemExit("the edit check is not calibrated: run python -m src.calibrate.calibrate_app first")
    counts, unreadable = defaultdict(Counter), 0
    with open(args.list, newline="") as f:
        rows = list(csv.DictReader(f))
    for n, row in enumerate(rows):
        try:
            data = Path(row["path"]).read_bytes()
            with Image.open(row["path"]) as im:
                out = det.analyze(im, file_bytes=data)
        except Exception:
            unreadable += 1  # counted, never named
            continue
        c = counts[row["source"]]
        c["n"] += 1
        c["real" if row["label"] == "0" else "ai"] += 1
        c[out["pixel_verdict"]] += 1
        c["label"] += bool(out["metadata"] and out["metadata"]["ai_label"])
        c["final_ai"] += out["verdict"] not in ("Uncertain", "Likely authentic")
        if (n + 1) % 50 == 0:
            print(f"  {n + 1} of {len(rows)}", flush=True)

    print(f"\n{sum(c['n'] for c in counts.values())} pictures from {args.list.name} ({unreadable} unreadable). Pixels only, "
          f"as the app sees them; 'AI verdict' = any verdict that says AI. Then the file's AI label.")
    short = ["AI-gen", "Likely AI", "AI-edited", "AI face", "Uncertain", "Authentic"]
    print(f"  {'source':<40}{'n':>5} | " + " ".join(f"{s:>9}" for s in short) + " | AI verdict | AI label | with label")
    for source, c in sorted(counts.items()):
        n = c["n"]
        ai = sum(c[v] for v in VERDICTS[:4])
        print(f"  {source:<40}{n:>5} | " + " ".join(f"{c[v] / n:>9.1%}" for v in VERDICTS) +
              f" | {ai / n:>10.1%} | {c['label']:>8} | {c['final_ai'] / n:>9.1%}")


if __name__ == "__main__":
    main()
