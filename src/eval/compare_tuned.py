"""Fair comparison of tuned DINOv3 backbones (src/models/tune_backbone.py) with the frozen one, on all dev sets.

Each model is a backbone (frozen, or a tuned checkpoint) plus its head; every model sees exactly what the app sees
(src/data/preprocess.inference_view, both app views). The line per model and view: at most 5% of EVERY real pool
called AI (the strictest pool decides, as in src/eval/fair_compare.py); then the share of every AI set above it.
Only dev sets (plan: decisions only on dev sets). Features are cached in features/compare/<model>/.

Real pools: polished dev, OpenImages val, the originals of Muse edit test sets v1 (people-free Commons dev photos)
and v2 (people-free OpenImages photos from SID-Set files 0-9, used nowhere else). My phone photos and selfies are
private: their features come from my own runs of src/eval/private_photo_test.py (--tuned) and are added with
--private "NAME:<file with {view}>" (only totals are printed).

Usage (from the repo root):
  python -m src.eval.compare_tuned --model "F:none:weights/open_gen_v1/head_F.pt"
      --model "LN2000:weights/ln_v1/ln_step2000.pt:weights/ln_v1/head_step2000.pt"
"""

import argparse
import csv
import math
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps

from src.data import preprocess
from src.eval import metrics
from src.eval.fair_compare import threshold_for
from src.models.backbone import Backbone
from src.models.linear_head import LinearHead

# name: (list, labels, source or None)
REAL_POOLS = {
    "polished dev": ("features/lists/commons_dev.csv", {0}, None),
    "OpenImages val": ("features/lists/muse_pairs_val.csv", {0}, None),
    "originals v1": ("features/lists/muse_edits_v1.csv", {0}, None),
    "originals v2": ("features/lists/muse_edits_test_v2.csv", {0}, None),
}
AI_SETS = {
    "Muse test v1": ("features/lists/muse_image_test_v1.csv", {1}, None),
    "Muse pairs val": ("features/lists/muse_pairs_val.csv", {1}, None),
    "edits v1 local": ("features/lists/muse_edits_v1.csv", {2}, None),
    "edits v1 whole": ("features/lists/muse_edits_v1.csv", {1}, None),
    "edits v2 local": ("features/lists/muse_edits_test_v2.csv", {2}, None),
    "edits v2 whole": ("features/lists/muse_edits_test_v2.csv", {1}, None),
    "SDXL": ("features/lists/sd_faces.csv", {1}, "sd_faces/sdxl"),
    "PixArt": ("features/lists/open_gen_v1/open_gen_dev.csv", {1}, "open_gen/pixart"),
    "SD3.5 dev": ("features/lists/open_gen_v1/open_gen_dev.csv", {1}, "open_gen/sd35m_devcaption"),
}
VIEWS = ("long448", "short448")


def rows_of(path, labels, source):
    with open(path, newline="") as f:
        return [r for r in csv.DictReader(f) if int(r["label"]) in labels and (source is None or r["source"] == source)]


def features(model_name, tuned, sets):
    """{set: {view: [N, 1024]}}, computed once per model and cached."""
    cache = Path("features/compare") / model_name
    out, backbone = {}, None
    for name, spec in sets.items():
        path = cache / f"{name.replace(' ', '_')}.npz"
        if path.exists():
            data = np.load(path)
            out[name] = {v: data[v] for v in VIEWS}
            continue
        backbone = backbone or Backbone(tuned_path=tuned)
        vectors = {v: [] for v in VIEWS}
        for row in rows_of(*spec):
            with Image.open(row["path"]) as im:
                image = ImageOps.exif_transpose(im).convert("RGB")
            for v in VIEWS:
                pixels = preprocess.to_tensor(preprocess.inference_view(image, v)).unsqueeze(0)
                vectors[v].append(backbone.mean_patch_vector(pixels)[0].cpu().numpy())
        out[name] = {v: np.stack(vectors[v]).astype(np.float32) for v in VIEWS}
        cache.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, **out[name])
        print(f"  {model_name}: {name} ({len(out[name][VIEWS[0]])} pictures)", flush=True)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", action="append", required=True, help='"NAME:tuned weights or none:head"')
    parser.add_argument("--private", action="append", default=[],
                        help='"NAME:file pattern with {view} and {tag}" (a private real pool), e.g. '
                             '"phone:data/private/phone_v1_app_{view}{tag}.npz"; {tag} is empty for the frozen '
                             'backbone and _<folder>-<file> for a tuned one, as private_photo_test.py names them')
    args = parser.parse_args()

    for item in args.model:
        name, tuned, head_path = item.split(":", 2)
        feats = features(name, None if tuned == "none" else tuned, {**REAL_POOLS, **AI_SETS})
        head = LinearHead.load(head_path)
        tag = "" if tuned == "none" else f"_{Path(tuned).parent.name}-{Path(tuned).stem}"
        for v in VIEWS:
            score = {s: head.logits(torch.from_numpy(feats[s][v])).numpy() for s in feats}
            for item_p in args.private:
                p_name, pattern = item_p.split(":", 1)
                p_file = Path(pattern.format(tag=tag, view=v))
                if p_file.exists():
                    score[p_name] = head.logits(torch.from_numpy(np.load(p_file)["features"].astype(np.float32))).numpy()
            reals = [s for s in score if s in REAL_POOLS or s not in AI_SETS]
            line = max(threshold_for(score[s]) for s in reals)
            edits = np.concatenate([score["edits v2 local"], score["edits v2 whole"]])
            auc = metrics.auc(np.r_[score["originals v2"], edits], np.r_[np.zeros(len(score["originals v2"])),
                                                                          np.ones(len(edits))] == 1)
            print(f"\n{name} [{v}] line p={1 / (1 + math.exp(-line)):.3f} | AUC originals v2 vs edits v2 {auc:.3f}")
            print("  real called AI: " + " | ".join(f"{s} {(score[s] > line).mean():.1%} (n={len(score[s])})" for s in reals))
            print("  AI caught:      " + " | ".join(f"{s} {(score[s] > line).mean():.1%}" for s in AI_SETS))


if __name__ == "__main__":
    main()
