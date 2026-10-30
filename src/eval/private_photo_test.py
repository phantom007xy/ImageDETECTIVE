"""Private real-photo test: how many of my own real phone photos does each head call AI?

PRIVACY (plan Section 0): the photos never leave this laptop. Everything runs locally: DINOv3 and
SAM 3 come from the local model cache in offline mode, and there are no API calls. The script
prints only totals per group: never a file name, a picture or the score of a single photo.
Per-photo results and the app-view features (1,024 numbers per photo, no pixels; for
src/eval/fair_compare.py) are written to data/private/ (ignored by git) and stay on this laptop.

Groups: every subfolder of the photo folder is a group (its name is printed, so use neutral names;
photos directly in the folder form the group "(no subfolder)"). Two extra splits: photos with or
without a face that SAM 3 can check (>= 64 px), and dark photos (mean brightness below 80 of 255).
Every photo is first turned upright as the phone shows it (EXIF orientation).

Heads compared (all on the same DINOv3 features):
  app = the head the app uses now (SID-Set files 0-29, old brightness range)
  A, B, C = results/muse_pairs_v1_heads.txt (C = with the Muse Image pairs)

Usage (from the repo root, in your own terminal, with the venv active):
  python -m src.eval.private_photo_test
  python -m src.eval.private_photo_test --folder data/private/phone_v1
"""

import os

os.environ["HF_HUB_OFFLINE"] = "1"  # never contact Hugging Face: the models come from the local cache
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import argparse  # noqa: E402
import csv  # noqa: E402
import math  # noqa: E402
import warnings  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
from PIL import Image, ImageOps  # noqa: E402

from src.data import preprocess  # noqa: E402
from src.faces.sam_faces import FaceFinder  # noqa: E402
from src.inference import AI_ABOVE, REAL_BELOW  # noqa: E402
from src.models.backbone import Backbone  # noqa: E402
from src.models.linear_head import LinearHead  # noqa: E402

HEADS = {
    "app": "weights/head_path_a_sidset30.pt",
    "A": "weights/muse_pairs_v1/head_A.pt",
    "B": "weights/muse_pairs_v1/head_B.pt",
    "C": "weights/muse_pairs_v1/head_C.pt",
}
DARK_BELOW = 80


def probability(head, vector):
    return 1 / (1 + math.exp(-float(head.logits(vector)[0])))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--folder", type=Path, default=Path("data/private/phone_v1"))
    args = parser.parse_args()
    warnings.filterwarnings("ignore")  # library warnings could mention a file

    files = sorted(p for p in args.folder.rglob("*") if p.is_file())
    heads = {name: LinearHead.load(path) for name, path in HEADS.items() if Path(path).exists()}
    backbone, finder = Backbone(), FaceFinder()

    rows, vectors, unreadable = [], [], 0
    for path in files:
        try:
            with Image.open(path) as im:
                image = ImageOps.exif_transpose(im).convert("RGB")
        except Exception:
            unreadable += 1  # e.g. HEIC files or videos; counted, never named
            continue
        relative = path.relative_to(args.folder)
        app_view = backbone.mean_patch_vector(preprocess.to_tensor(preprocess.inference_view(image)).unsqueeze(0)).cpu()
        crop_view = backbone.mean_patch_vector(preprocess.to_tensor(preprocess.eval_view(image)).unsqueeze(0)).cpu()
        vectors.append(app_view[0].numpy().astype(np.float16))
        row = {
            "file": str(relative),
            "group": relative.parts[0] if len(relative.parts) > 1 else "(no subfolder)",
            "faces": sum(not f["too_small"] for f in finder.find(image)),
            "brightness": round(float(np.asarray(image.convert("L")).mean()), 1),
        }
        for name, head in heads.items():
            row[f"p_{name}"] = round(probability(head, app_view), 4)
            row[f"p_{name}_centre_crop"] = round(probability(head, crop_view), 4)
        rows.append(row)

    if not rows:
        print(f"No readable photos in {args.folder} ({unreadable} files could not be read).")
        return
    out = Path("data/private") / f"{args.folder.name}_results.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    # The app-view features (1,024 numbers per photo, no pixels), so new heads can be compared
    # with src/eval/fair_compare.py without reading the photos again. Stays in data/private/.
    features_out = Path("data/private") / f"{args.folder.name}_app.npz"
    np.savez_compressed(features_out, features=np.stack(vectors), labels=np.zeros(len(rows), dtype=np.int8),
                        img_ids=np.array([r["file"] for r in rows]), sources=np.array(["private/phone"] * len(rows)))

    # ---- totals only from here on ----
    print(f"\n{len(rows)} photos checked, {unreadable} files could not be read. Fully offline, no API calls.")
    print(f"Per-photo results and features stay on this laptop: {out}, {features_out}\n")
    splits = [("all", lambda r: True)]
    splits += [(f"folder: {g}", lambda r, g=g: r["group"] == g) for g in sorted({r["group"] for r in rows})]
    splits += [("with a face", lambda r: r["faces"] > 0), ("no face", lambda r: r["faces"] == 0),
               (f"dark (brightness < {DARK_BELOW})", lambda r: r["brightness"] < DARK_BELOW)]

    print("Real photos called AI (probability > 0.5), as the app sees them:")
    print(f"  {'group':<30}{'n':>5}" + "".join(f"{name:>9}" for name in heads))
    for title, keep in splits:
        part = [r for r in rows if keep(r)]
        if part:
            cells = "".join(f"{np.mean([r[f'p_{n}'] > 0.5 for r in part]):>9.0%}" for n in heads)
            print(f"  {title:<30}{len(part):>5}{cells}")

    print(f"\nApp verdicts on all {len(rows)} photos (AI >= {AI_ABOVE}, authentic <= {REAL_BELOW}):")
    for name in heads:
        p = np.array([r[f"p_{name}"] for r in rows])
        print(f"  {name:<5} Likely AI-generated {int((p >= AI_ABOVE).sum()):>4} | "
              f"Uncertain {int(((p > REAL_BELOW) & (p < AI_ABOVE)).sum()):>4} | "
              f"Likely authentic {int((p <= REAL_BELOW).sum()):>4}")

    cells = "  ".join(f"{name} {np.mean([r[f'p_{name}_centre_crop'] > 0.5 for r in rows]):.0%}" for name in heads)
    print(f"\nFor comparison with the Chameleon numbers (centre crop), all photos called AI: {cells}")


if __name__ == "__main__":
    main()
