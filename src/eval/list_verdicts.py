"""App verdicts on a labelled image list, totals per source only (e.g. my selfies and their Muse edits).

Runs exactly the app's whole-image path (app view, head, its calibrated thresholds; pixels only, no metadata,
no faces) and prints, per source of the list (e.g. muse_edits/original, muse_edits/local, muse_edits/whole), how
many pictures get "AI-generated" (clear), "Likely AI-generated", "Uncertain" and "Likely authentic". It also
counts how many files carry an AI label in their metadata (what the app would add for unchanged files).

PRIVACY: made for pictures that must not leave the laptop or be seen (my selfies): no file name, picture or
single score is printed; the app-view features (1,024 numbers per picture, no pixels) go to --out, which should
be under data/private/ (ignored by git). Fully local: no API calls.

Usage (from the repo root, in your own terminal):
  python -m src.eval.list_verdicts --list features/lists/muse_self_edits_v1.csv --head weights/open_gen_v1/head_F.pt
      --out data/private/self_edits_v1_app.npz
  (a tuned backbone: add --tuned weights/ln_v1/ln_stepN.pt and give its head and calibration)
"""

import os

os.environ["HF_HUB_OFFLINE"] = "1"  # the models come from the local cache; nothing is sent anywhere
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import argparse  # noqa: E402
import csv  # noqa: E402
import json  # noqa: E402
import warnings  # noqa: E402
from collections import Counter, defaultdict  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image, ImageOps  # noqa: E402

from src.data import preprocess  # noqa: E402
from src.inference import load_calibration  # noqa: E402
from src.models.backbone import Backbone  # noqa: E402
from src.models.linear_head import LinearHead  # noqa: E402
from src.models.pe_core import PECore  # noqa: E402
from src.services import metadata  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--list", type=Path, required=True)
    parser.add_argument("--head", type=Path, required=True)
    parser.add_argument("--tuned", type=Path, help="tuned DINOv3 weights (src/models/tune_backbone.py)")
    parser.add_argument("--view", choices=["long448", "short448"], help="app view other than preprocess.APP_VIEW")
    parser.add_argument("--edit-check", type=Path, help="also the Muse-edit check (its head; line from its calibration)")
    parser.add_argument("--out", type=Path, required=True, help="feature file (keep it under data/private/)")
    args = parser.parse_args()
    warnings.filterwarnings("ignore")  # library warnings could mention a file

    with open(args.list, newline="") as f:
        rows = list(csv.DictReader(f))
    backbone = Backbone(tuned_path=args.tuned)
    head, c = LinearHead.load(args.head), load_calibration(args.head)
    pe, edit_head, edit_line, pe_vectors = None, None, None, []
    if args.edit_check:
        pe, edit_head = PECore(), LinearHead.load(args.edit_check)
        cal = args.edit_check.with_name(args.edit_check.stem + "_calibration.json")
        edit_line = json.loads(cal.read_text())["edit_logit"] if cal.exists() else 0.0
    vectors, verdicts, tagged = [], defaultdict(Counter), Counter()
    for row in rows:
        data = Path(row["path"]).read_bytes()
        if metadata.check_file(data)["ai_label"]:
            tagged[row["source"]] += 1
        with Image.open(row["path"]) as im:
            image = ImageOps.exif_transpose(im).convert("RGB")
        view = preprocess.inference_view(image, args.view)
        vector = backbone.mean_patch_vector(preprocess.to_tensor(view).unsqueeze(0)).cpu()
        vectors.append(vector[0].numpy().astype(np.float16))
        z = float(head.logits(vector)[0])
        edited = False
        if pe is not None:
            pe_vector = pe.layer_features(pe.to_tensor(pe.view(image)).unsqueeze(0))[0]
            pe_vectors.append(pe_vector.numpy().astype(np.float16))
            edited = float(edit_head.logits(pe_vector.reshape(1, -1).float())[0]) > edit_line
        if c["strong_ai_logit"] is not None and z > c["strong_ai_logit"]:
            verdict = "AI-generated"
        elif z > c["ai_logit"]:
            verdict = "Likely AI-generated"
        elif edited:
            verdict = "Likely AI-edited"
        elif z <= c["real_logit"]:
            verdict = "Likely authentic"
        else:
            verdict = "Uncertain"
        verdicts[row["source"]][verdict] += 1

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, features=np.stack(vectors), labels=np.array([int(r["label"]) for r in rows], dtype=np.int8),
                        sources=np.array([r["source"] for r in rows]), img_ids=np.arange(len(rows)))
    if pe_vectors:  # the PE-Core features next to it, e.g. data/private/self_edits_v1_app_pe.npz
        np.savez_compressed(args.out.with_name(args.out.stem + "_pe.npz"), features=np.stack(pe_vectors),
                            labels=np.array([int(r["label"]) for r in rows], dtype=np.int8),
                            sources=np.array([r["source"] for r in rows]), img_ids=np.arange(len(rows)))
    print(f"\n{len(rows)} pictures, head {args.head.name}{' + tuned DINOv3 ' + args.tuned.name if args.tuned else ''}, "
          f"view {args.view or preprocess.APP_VIEW} ({'calibrated' if c['calibrated'] else 'provisional thresholds'});"
          f" pixels only. Features (no pixels) -> {args.out}\n")
    names = ["AI-generated", "Likely AI-generated"] + (["Likely AI-edited"] if pe else []) + ["Uncertain", "Likely authentic"]
    print(f"  {'source':<24}{'n':>4} | " + " | ".join(f"{n:>19}" for n in names) + " | AI label in file")
    for source in sorted(verdicts):
        n = sum(verdicts[source].values())
        print(f"  {source:<24}{n:>4} | " + " | ".join(f"{verdicts[source][v]:>4} ({verdicts[source][v] / n:5.1%})    "
                                                     for v in names) + f" | {tagged[source]}")


if __name__ == "__main__":
    main()
