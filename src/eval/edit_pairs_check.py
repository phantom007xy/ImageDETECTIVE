"""Why does a set of edits get missed? Totals only, made for private pairs (my selfies and their Muse edits).

For every edit and its own original (an image list written by scripts/make_muse_edits.py):
  1. What changed: the share of pixels whose colour clearly changed (difference > 25 after a 5x5 median, the edit
     resized to the original, as in make_muse_edits.py), and whether the rest of the picture was re-rendered: the
     fine-detail (high-pass) correlation with the original, at least 15 px away from any change. A plain resize of the
     original keeps ~0.98; people-free Muse edits gave 0.59 (results/muse_edits_v1.txt), i.e. redrawn everywhere.
  2. The Muse-edit check (fine-tuned PE) on the whole picture, exactly as the app runs it (src/inference.Detector).
  3. Every face SAM 3 finds (the app's crop): the edit check on the face crop (zoomed in, the crop at 336 px), and the
     app's own face checks (DINOv3 + head F; the face-swap head), each as the picture's highest face score.
Prints medians, AUCs (originals vs. edits: 0.5 = no signal, 1.0 = perfect) and shares above the app's lines. No file
name, picture or single score is printed. Fully local: no API calls.

Usage (from the repo root, in your own terminal for private pictures):
  python -m src.eval.edit_pairs_check --list features/lists/muse_self_edits_v1.csv
"""

import os

os.environ["HF_HUB_OFFLINE"] = "1"  # the models come from the local cache; nothing is sent anywhere
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import argparse  # noqa: E402
import csv  # noqa: E402
from collections import defaultdict  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
from PIL import Image, ImageFilter, ImageOps  # noqa: E402

from src.faces import crop  # noqa: E402
from src.inference import Detector  # noqa: E402

HEAD = "weights/open_gen_v1/head_F.pt"
EDIT_CHECK = "weights/edit_check_v3/head_EC3.pt"


def rerender(original, edited):
    """(changed share, fine-detail correlation outside the change, the same for a plain resize of the original)."""
    a = np.asarray(original, dtype=np.float32)
    small = edited.resize(original.size, Image.BICUBIC)
    diff = Image.fromarray(np.abs(a - np.asarray(small, dtype=np.float32)).mean(axis=2).clip(0, 255).astype(np.uint8))
    changed = np.asarray(diff.filter(ImageFilter.MedianFilter(5))) > 25
    near = np.asarray(Image.fromarray(changed.astype(np.uint8) * 255).filter(ImageFilter.MaxFilter(31))) > 0
    far = ~near  # at least 15 px away from any change

    def fine(img):  # high-pass: the picture minus a blurred copy of itself
        g = img.convert("L")
        return np.asarray(g, dtype=np.float32) - np.asarray(g.filter(ImageFilter.GaussianBlur(2)), dtype=np.float32)

    def corr(x, y):
        return float(np.corrcoef(x[far], y[far])[0, 1]) if far.sum() > 1000 else np.nan

    resized = original.resize(edited.size, Image.BICUBIC).resize(original.size, Image.BICUBIC)
    return changed.mean(), corr(fine(original), fine(small)), corr(fine(original), fine(resized))


def auc(neg, pos):
    """Chance that an edit scores above an original (ties count half); nan without both."""
    neg, pos = np.asarray(neg, float), np.asarray(pos, float)
    neg, pos = neg[np.isfinite(neg)], pos[np.isfinite(pos)]
    if not len(neg) or not len(pos):
        return np.nan
    return float(((pos[:, None] > neg[None]).sum() + 0.5 * (pos[:, None] == neg[None]).sum()) / (len(pos) * len(neg)))


def spread(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return f"{np.median(x):+.2f} ({np.percentile(x, 10):+.2f} .. {np.percentile(x, 90):+.2f})" if len(x) else "-"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--list", type=Path, required=True, help="image list from scripts/make_muse_edits.py")
    args = parser.parse_args()

    with open(args.list, newline="") as f:
        rows = list(csv.DictReader(f))
    originals = {Path(r["path"]).stem.removesuffix("_original"): r["path"] for r in rows if r["label"] == "0"}
    edits = [(Path(r["path"]).stem.removesuffix("_edited"), r["source"].split("/")[-1], r["path"])
             for r in rows if r["label"] != "0"]
    det = Detector(HEAD, edit_check=EDIT_CHECK)
    if det.edit_check is None:
        raise SystemExit("the edit check is not calibrated: run python -m src.calibrate.calibrate_app first")
    c, edit_line = det.calibration, det.edit_check["line"]
    head, pe = det.edit_check["head"], det.pe

    def scores(path):
        with Image.open(path) as im:
            image = ImageOps.exif_transpose(im).convert("RGB")
        r = det.analyze(image)
        faces = [f for f in r["faces"] if not f["too_small"]]
        face_edit = [float(head.logits(pe.layer_features(pe.to_tensor(pe.view(image.crop(tuple(
            round(v) for v in crop.crop_box(f))))).unsqueeze(0)).reshape(1, -1).float())[0]) for f in faces]
        best = lambda xs: max(xs) if xs else -np.inf  # noqa: E731
        return image, {"whole": r["score"], "edit": r["edit"]["logit"], "faces": len(faces), "face_edit": best(face_edit),
                       "face_f": best([f["logit"] for f in faces]), "swap": best([f.get("swap_logit", -np.inf) for f in faces])}

    seen, out = {}, defaultdict(list)
    for n, (key, kind, path) in enumerate(edits):
        if key not in originals:
            continue
        if key not in seen:
            seen[key] = scores(originals[key])
        original, s_orig = seen[key]
        edited, s_edit = scores(path)
        out[kind].append((s_orig, s_edit, rerender(original, edited)))
        if (n + 1) % 20 == 0:
            print(f"  {n + 1} of {len(edits)} edits", flush=True)

    print(f"\n{sum(map(len, out.values()))} edits with their originals ({args.list.name}); app lines: edit check "
          f"{edit_line:+.2f}, red face {c['face_ai_logit']:+.2f}, orange face {c['face_leans_logit']:+.2f}, "
          f"red swap {c['swap_ai_logit']:+.2f}")
    for kind, items in sorted(out.items()):
        o = {k: [s[0][k] for s in items] for k in items[0][0]}
        e = {k: [s[1][k] for s in items] for k in items[0][1]}
        changed, fine, plain = (np.array([s[2][i] for s in items]) for i in range(3))
        kept = changed < 0.5  # edits that kept the framing (a reframed picture has no "outside the change")
        print(f"\n== {kind} edits: {len(items)} ==")
        print(f"  changed area: median {np.median(changed):.1%} (10-90%: {np.percentile(changed, 10):.1%} .. "
              f"{np.percentile(changed, 90):.1%})")
        print(f"  fine detail outside the change vs. the original ({kept.sum()} edits that kept the framing): correlation "
              f"{spread(fine[kept])}; a plain resize of the original: {spread(plain[kept])}")
        print(f"  edit check, whole picture (as the app): originals {spread(o['edit'])} | edits {spread(e['edit'])} | AUC "
              f"{auc(o['edit'], e['edit']):.2f} | edits above the line {np.mean(np.array(e['edit']) > edit_line):.0%}, "
              f"originals above it {np.mean(np.array(o['edit']) > edit_line):.0%}")
        print(f"  whole-image head F: originals {spread(o['whole'])} | edits {spread(e['whole'])} | AUC "
              f"{auc(o['whole'], e['whole']):.2f}")
        has = np.array(e["faces"]) > 0
        print(f"  pictures with a checkable face: originals {np.mean(np.array(o['faces']) > 0):.0%}, edits {has.mean():.0%}")
        for name, key, red, orange in (("edit check on the face crop", "face_edit", edit_line, None),
                                       ("head F on the face (app)", "face_f", c["face_ai_logit"], c["face_leans_logit"]),
                                       ("face-swap head (app)", "swap", c["swap_ai_logit"], c["swap_leans_logit"])):
            line = f"  {name:<28}: AUC {auc(o[key], e[key]):.2f} | edits above {red:+.2f}: {np.mean(np.array(e[key]) > red):.0%}"
            if orange is not None:
                line += f", above {orange:+.2f}: {np.mean(np.array(e[key]) > orange):.0%}"
            print(line + f" | originals above {red:+.2f}: {np.mean(np.array(o[key]) > red):.0%}")


if __name__ == "__main__":
    main()
