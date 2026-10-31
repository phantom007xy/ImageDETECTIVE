"""Private real-photo test: how many of my own real phone photos does each head call AI?

PRIVACY (plan Section 0): the photos never leave this laptop. Everything runs locally: DINOv3 and
SAM 3 come from the local model cache in offline mode, and there are no API calls. The script
prints only totals per group: never a file name, a picture or the score of a single photo.
Per-photo results and the app-view features of every photo and every face crop (1,024 numbers
each, no pixels; for src/eval/fair_compare.py and src/calibrate/calibrate.py) are written to
data/private/ (ignored by git) and stay on this laptop.

Groups: every subfolder of the photo folder is a group (its name is printed, so use neutral names;
photos directly in the folder form the group "(no subfolder)"). Two extra splits: photos with or
without a face that SAM 3 can check (>= 64 px), and dark photos (mean brightness below 80 of 255).
Every photo is first turned upright as the phone shows it (EXIF orientation).

Heads compared (all on the same DINOv3 features):
  app = the head the app uses now (SID-Set files 0-29, old brightness range)
  A, B, C = results/muse_pairs_v1_heads.txt (C = with the Muse Image pairs)
  C2, D = results/commons_v1_heads.txt (C2 = C + Commons real photos, D = C2 + Commons pairs)
  E = D + SD 1.5 / 2.1 faces (results/generator_variety_v1.txt)
  F = E + SD 3.5 Medium and Kandinsky 2.2 fakes (results/open_gen_v1_heads.txt)
  G = F + Muse Image edits of people-free training photos (results/muse_edits_v1.txt)

Usage (from the repo root, in your own terminal, with the venv active):
  python -m src.eval.private_photo_test
  python -m src.eval.private_photo_test --folder data/private/phone_v1
  python -m src.eval.private_photo_test --folder data/private/phone_v1 --tuned weights/ln_v1/ln_step2000.pt
      --tuned-head weights/ln_v1/head_step2000.pt   (also a tuned DINOv3: its features go to data/private too)
"""

import os

os.environ["HF_HUB_OFFLINE"] = "1"  # never contact Hugging Face: the models come from the local cache
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import argparse  # noqa: E402
import csv  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import warnings  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image, ImageOps  # noqa: E402

from src.data import preprocess  # noqa: E402
from src.faces import crop  # noqa: E402
from src.faces.sam_faces import FaceFinder  # noqa: E402
from src.inference import load_calibration  # noqa: E402
from src.models.backbone import Backbone  # noqa: E402
from src.models.linear_head import LinearHead  # noqa: E402
from src.models.pe_core import PECore  # noqa: E402

HEADS = {
    "app": "weights/head_path_a_sidset30.pt",
    "A": "weights/muse_pairs_v1/head_A.pt",
    "B": "weights/muse_pairs_v1/head_B.pt",
    "C": "weights/muse_pairs_v1/head_C.pt",
    "C2": "weights/commons_v1/head_C2.pt",
    "D": "weights/commons_v1/head_D.pt",
    "E": "weights/commons_v1/head_E.pt",
    "F": "weights/open_gen_v1/head_F.pt",
    "G": "weights/muse_edits_v1/head_G.pt",
}
DARK_BELOW = 80
VIEWS = ("long448", "short448")  # both app views are saved, so they can be compared (preprocess.APP_VIEW)
SWAP_HEAD = "weights/dfdc_v1/head_dfdc.pt"


def probability(head, vector):
    return 1 / (1 + math.exp(-float(head.logits(vector)[0])))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--folder", type=Path, default=Path("data/private/phone_v1"))
    parser.add_argument("--tuned", type=Path, help="also a tuned DINOv3 (src/models/tune_backbone.py) ...")
    parser.add_argument("--tuned-head", type=Path, help="... with its head")
    parser.add_argument("--edit-check", type=Path, help="also the Muse-edit check: its head (src/models/train_edit_check.py)")
    args = parser.parse_args()
    warnings.filterwarnings("ignore")  # library warnings could mention a file

    files = sorted(p for p in args.folder.rglob("*") if p.is_file())
    heads = {name: LinearHead.load(path) for name, path in HEADS.items() if Path(path).exists()}
    swap = LinearHead.load(SWAP_HEAD)  # face swaps (DFDC), on the same face crops
    backbone, finder = Backbone(), FaceFinder()
    # A tuned DINOv3 is a different backbone: its own pass over every photo (whole image, both views).
    tuned = Backbone(tuned_path=args.tuned) if args.tuned else None
    tuned_head = LinearHead.load(args.tuned_head) if args.tuned else None
    tuned_vectors = {view: [] for view in VIEWS}
    # The Muse-edit check: Meta's PE-Core on the centre square of every photo (its own pass).
    pe = PECore() if args.edit_check else None
    pe_vectors = []

    rows, vectors, face_vectors, face_files, unreadable = [], {view: [] for view in VIEWS}, [], [], 0
    for path in files:
        try:
            with Image.open(path) as im:
                image = ImageOps.exif_transpose(im).convert("RGB")
        except Exception:
            unreadable += 1  # e.g. HEIC files or videos; counted, never named
            continue
        relative = path.relative_to(args.folder)
        views = {view: backbone.mean_patch_vector(preprocess.to_tensor(preprocess.inference_view(image, view))
                                                  .unsqueeze(0)).cpu() for view in VIEWS}
        app_view = views[preprocess.APP_VIEW]
        crop_view = backbone.mean_patch_vector(preprocess.to_tensor(preprocess.eval_view(image)).unsqueeze(0)).cpu()
        for view, vector in views.items():
            vectors[view].append(vector[0].numpy().astype(np.float16))
        if tuned is not None:
            for view in VIEWS:
                v = tuned.mean_patch_vector(preprocess.to_tensor(preprocess.inference_view(image, view)).unsqueeze(0)).cpu()
                tuned_vectors[view].append(v[0].numpy().astype(np.float16))
        if pe is not None:
            pe_vectors.append(pe.layer_features(pe.to_tensor(pe.view(image)).unsqueeze(0))[0].numpy().astype(np.float16))
        faces = [f for f in finder.find(image) if not f["too_small"]]
        crops = [backbone.mean_patch_vector(preprocess.to_tensor(crop.eval_face_view(image, f)).unsqueeze(0)).cpu()
                 for f in faces]  # the app's face crop
        face_vectors += [c[0].numpy().astype(np.float16) for c in crops]
        face_files += [str(relative)] * len(crops)
        row = {
            "file": str(relative),
            "group": relative.parts[0] if len(relative.parts) > 1 else "(no subfolder)",
            "faces": len(faces),
            "brightness": round(float(np.asarray(image.convert("L")).mean()), 1),
        }
        row["face_max_swap_logit"] = round(max(float(swap.logits(c)[0]) for c in crops), 4) if crops else ""
        for name, head in heads.items():
            row[f"p_{name}"] = round(probability(head, app_view), 4)
            row[f"logit_{name}"] = round(float(head.logits(app_view)[0]), 4)
            row[f"p_{name}_centre_crop"] = round(probability(head, crop_view), 4)
            row[f"face_max_logit_{name}"] = round(max(float(head.logits(c)[0]) for c in crops), 4) if crops else ""
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
    # _app.npz = the app's current view; _app_<view>.npz = both views, to compare them (size fix, 31 Oct).
    features_out = Path("data/private") / f"{args.folder.name}_app.npz"
    for suffix, view in [("", preprocess.APP_VIEW)] + [(f"_{v}", v) for v in VIEWS]:
        np.savez_compressed(Path("data/private") / f"{args.folder.name}_app{suffix}.npz",
                            features=np.stack(vectors[view]), labels=np.zeros(len(rows), dtype=np.int8),
                            img_ids=np.array([r["file"] for r in rows]), sources=np.array(["private/phone"] * len(rows)))
    if tuned is not None:  # e.g. data/private/phone_v1_app_short448_ln_v1-ln_step2000.npz
        tag = f"{args.tuned.parent.name}-{args.tuned.stem}"
        for view in VIEWS:
            np.savez_compressed(Path("data/private") / f"{args.folder.name}_app_{view}_{tag}.npz",
                                features=np.stack(tuned_vectors[view]), labels=np.zeros(len(rows), dtype=np.int8),
                                img_ids=np.array([r["file"] for r in rows]), sources=np.array(["private/phone"] * len(rows)))
    if pe is not None:  # PE-Core features of every photo (6 x 1,024 numbers, no pixels) for the edit check's line
        np.savez_compressed(Path("data/private") / f"{args.folder.name}_pe.npz", features=np.stack(pe_vectors),
                            labels=np.zeros(len(rows), dtype=np.int8), img_ids=np.array([r["file"] for r in rows]),
                            sources=np.array(["private/phone"] * len(rows)))
    if face_vectors:  # the same for every face crop, so the face threshold can be checked on my faces too
        np.savez_compressed(Path("data/private") / f"{args.folder.name}_faces_app.npz", features=np.stack(face_vectors),
                            labels=np.zeros(len(face_vectors), dtype=np.int8), img_ids=np.array(face_files),
                            sources=np.array(["private/phone_face"] * len(face_vectors)))

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

    print(f"\nApp verdicts on all {len(rows)} photos (each head's own calibration if it has one; the same rules"
          f" as src/inference.py):")
    for name in heads:
        c = load_calibration(HEADS[name])
        counts = {"AI": 0, "face": 0, "Uncertain": 0, "authentic": 0}
        for r in rows:
            face, swap_face = r[f"face_max_logit_{name}"], r["face_max_swap_logit"]
            if r[f"logit_{name}"] > c["ai_logit"]:
                counts["AI"] += 1
            elif c["face_ai_logit"] is not None and face != "" and face > c["face_ai_logit"]:
                counts["face"] += 1
            elif c["swap_ai_logit"] is not None and swap_face != "" and swap_face > c["swap_ai_logit"]:
                counts["face"] += 1
            elif r[f"logit_{name}"] <= c["real_logit"]:
                counts["authentic"] += 1
            else:
                counts["Uncertain"] += 1
        print(f"  {name:<5} Likely AI-generated {counts['AI']:>4} | A face looks AI-made {counts['face']:>4} | "
              f"Uncertain {counts['Uncertain']:>4} | Likely authentic {counts['authentic']:>4}   "
              f"({'calibrated' if c['calibrated'] else 'provisional 0.65 / 0.35'})")

    cells = "  ".join(f"{name} {np.mean([r[f'p_{name}_centre_crop'] > 0.5 for r in rows]):.0%}" for name in heads)
    print(f"\nFor comparison with the Chameleon numbers (centre crop), all photos called AI: {cells}")
    if pe is not None:
        x = np.stack(pe_vectors).astype(np.float32)
        z = LinearHead.load(args.edit_check).logits(torch.from_numpy(x.reshape(len(x), -1))).numpy()
        cal = args.edit_check.with_name(args.edit_check.stem + "_calibration.json")
        line = json.loads(cal.read_text())["edit_logit"] if cal.exists() else 0.0
        print(f"Muse-edit check {args.edit_check.name}: called edited {(z > line).mean():.0%} of {len(z)} real photos "
              f"({'its calibrated line' if cal.exists() else 'p > 0.5, not calibrated yet'})")
    if tuned is not None:
        for view in VIEWS:
            z = tuned_head.logits(torch.from_numpy(np.stack(tuned_vectors[view]).astype(np.float32))).numpy()
            print(f"Tuned DINOv3 {args.tuned.name} + {args.tuned_head.name}, view {view}: called AI (p > 0.5) "
                  f"{(z > 0).mean():.0%} of {len(z)} (its fair line comes from src/eval/fair_compare.py)")


if __name__ == "__main__":
    main()
