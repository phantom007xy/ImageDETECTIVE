"""Test the fine-tuned face-swap Perception Encoder (src/models/tune_face_pe.py) on face crops (src/faces/make_crops.py).

The strict line comes from the REAL face pools, as in the app's calibration: at most 0.5% of the faces of every pool
above it (polished Commons, OpenImages val and unseen, Commons v3 dev, close-ups, selfies+portraits, SID val, DFDC part
3 real faces). Then, at that line:
  - OpenForensics Test-Dev: swaps in still photos, never used for any choice
  - OpenForensics dev and DFDC part 3: the dev sets the checkpoint was chosen on (for reference)
  - WildDeepfake: video face swaps from the internet, never trained on; its own tighter 224 px crops, resized to 336
  - AI-made faces (Muse test, SD faces incl. SDXL, GAN faces): not this check's job, shown for information
With --save, the features (mean patch tokens after blocks 4..20 + last layer) of every set go to --save-dir, so the
app's calibration can use them.

Usage (from the repo root):
  python -m src.eval.face_pe_eval --pe weights/face_pe_v1/pe_step4000.pt --head weights/face_pe_v1/head_step4000.pt
"""

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from src.data.preprocess import jpeg
from src.eval import metrics
from src.models.linear_head import LinearHead
from src.models.pe_core import PECore

CROPS = Path("D:/Datasets/FaceCrops")
REAL_POOLS = {"polished dev": "pool_commons_dev", "OpenImages val": "pool_muse_pairs_val",
              "OpenImages unseen": "pool_sid_reals_0_9_unseen", "Commons v3 dev": "pool_commons_v3_dev",
              "close-ups": "pool_real_face_crops_dev_all", "selfies+portraits": "pool_commons_v4_dev",
              "SID val": "pool_sid_val", "DFDC p3 real": "dfdc_p3_dev"}
SWAPS = {"OpenForensics test": "of_test_dev", "OpenForensics dev": "of_val_dev", "DFDC p3": "dfdc_p3_dev"}
AI_FACES = {"Muse test faces": "pool_muse_image_test_v1", "SD faces": "pool_sd_faces", "GAN faces (TPDNE)": "pool_tpdne_test"}


def rows_of(stem):
    with open(CROPS / f"{stem}.csv", newline="") as f:
        return list(csv.DictReader(f))


def wilddeepfake_rows():
    with open("features/lists/wilddeepfake.csv", newline="") as f:
        return [{"path": r["path"], "label": r["label"], "group": str(Path(r["path"]).parent)} for r in csv.DictReader(f)]


@torch.no_grad()
def features(pe, rows, ready_crops=False):
    """[N, 6, 1024] for the crops; ready_crops: WildDeepfake's own crops, resized to 336 and saved as JPEG 90."""
    out, batch = [], []
    for r in rows:
        with Image.open(r["path"]) as im:
            image = im.convert("RGB")
        if ready_crops:
            image = jpeg(image.resize((pe.size, pe.size), Image.BICUBIC), 90)
        batch.append(pe.to_tensor(image))
        if len(batch) == 32:
            out.append(pe.layer_features(torch.stack(batch)))
            batch = []
    if batch:
        out.append(pe.layer_features(torch.stack(batch)))
    return torch.cat(out).numpy().astype(np.float16)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pe", type=Path, help="fine-tuned PE weights (default: the original PE)")
    parser.add_argument("--head", type=Path, required=True)
    parser.add_argument("--save-dir", type=Path, help="save every set's features here")
    args = parser.parse_args()

    pe, head = PECore(tuned_path=args.pe), LinearHead.load(args.head)
    sets = {}
    for name, stem in {**REAL_POOLS, **SWAPS, **AI_FACES}.items():
        if stem not in sets:
            rows = rows_of(stem)
            sets[stem] = (rows, features(pe, rows))
    rows = wilddeepfake_rows()
    sets["wilddeepfake"] = (rows, features(pe, rows, ready_crops=True))
    if args.save_dir:
        args.save_dir.mkdir(parents=True, exist_ok=True)
        for stem, (rows, x) in sets.items():
            np.savez_compressed(args.save_dir / f"{stem}.npz", features=x, labels=np.array([int(r["label"]) for r in rows]),
                                img_ids=np.array([r["path"] for r in rows]), groups=np.array([r["group"] for r in rows]))

    score = lambda stem: head.logits(torch.from_numpy(sets[stem][1].reshape(len(sets[stem][1]), -1).astype(np.float32))).numpy()  # noqa: E731
    label = lambda stem: np.array([int(r["label"]) for r in sets[stem][0]])  # noqa: E731
    reals = {n: score(s)[label(s) == 0] for n, s in REAL_POOLS.items()}
    line = max(float(np.quantile(s, 0.995)) for s in reals.values())
    tight = max(reals, key=lambda n: np.quantile(reals[n], 0.995))
    print(f"{args.pe.name if args.pe else 'original PE'} + {args.head.name}: strict line {line:+.2f} (set by {tight})")
    print("  REAL faces above the line: " + " | ".join(f"{n} {(s > line).mean():.1%} of {len(s)}" for n, s in reals.items()))
    for name, stem in {**SWAPS, "WildDeepfake (never trained on, tighter crops)": "wilddeepfake"}.items():
        s, y = score(stem), label(stem)
        print(f"  {name:<48} swaps caught {(s[y == 1] > line).mean():6.1%} of {int((y == 1).sum())} | AUC {metrics.auc(s, y == 1):.3f}"
              f" | its real faces above {(s[y == 0] > line).mean():.1%}")
    print("  AI-made faces above the line: " + " | ".join(f"{n} {(score(s) > line).mean():.1%}" for n, s in AI_FACES.items()))


if __name__ == "__main__":
    main()
