"""Re-score every checkpoint of the PE fine-tune (src/models/tune_pe.py) next to the frozen edit check, fairly.

Each candidate = a PE (frozen or a fine-tuned checkpoint) + its edit head. For every candidate: the edit scores of all
dev pictures (the app's view, centre square 336), then head F + the edit check together, with the edit line chosen
jointly (src/calibrate/calibrate_edit_check.py): together at most 5% of every real pool, the edit check alone at most
0.5%. Shown without and with the "trade" (head F gives up its top 1 flag in its tightest pool, so a single real
outlier can't set the edit line). Dev sets only.

Usage (from the repo root, after the fine-tune):  python -m src.eval.rescore_pe_ft --run weights/pe_ft_v1
"""

import argparse
import csv
import io
import json
import re
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps

from src.eval import metrics
from src.eval.fair_compare import threshold_for
from src.models.linear_head import LinearHead
from src.models.pe_core import PECore

REAL = {"polished dev": ("commons_dev", 0), "OpenImages val": ("muse_pairs_val", 0),
        "originals v1": ("muse_edits_v1", 0), "originals v2": ("muse_edits_test_v2", 0)}
AI = {"edits v1 local": ("muse_edits_v1", 2), "edits v1 whole": ("muse_edits_v1", 1),
      "edits v2 local": ("muse_edits_test_v2", 2), "edits v2 whole": ("muse_edits_test_v2", 1),
      "Muse test v1": ("muse_image_test_v1", 1), "Muse pairs val": ("muse_pairs_val", 1)}


def dev_views(pe):
    """{set: [normalised view tensors]} and head F's logits of the same pictures (aligned by path)."""
    head_f = LinearHead.load("weights/open_gen_v1/head_F.pt")
    views, f_logits = {}, {}
    for name, (stem, label) in {**REAL, **AI}.items():
        rows = list(csv.DictReader(open(f"features/lists/{stem}.csv", newline="")))
        app = np.load(f"features/{stem}_app.npz")
        assert list(app["img_ids"]) == [r["path"] for r in rows], stem
        keep = [i for i, r in enumerate(rows) if int(r["label"]) == label]
        f_logits[name] = head_f.logits(torch.from_numpy(app["features"][keep].astype(np.float32))).numpy()
        out = []
        for i in keep:
            with Image.open(rows[i]["path"]) as im:
                out.append(pe.to_tensor(pe.view(ImageOps.exif_transpose(im).convert("RGB"))))
        views[name] = out
    return views, f_logits


def edit_scores(pe, head, views):
    out = {}
    for name, tensors in views.items():
        s = []
        for start in range(0, len(tensors), 32):
            feats = pe.layer_features(torch.stack(tensors[start:start + 32]))
            s.append(head.logits(feats.reshape(len(feats), -1).float()).numpy())
        out[name] = np.concatenate(s)
    return out


def joint(edit, f_logits, f_line, give_up=0):
    """The edit line and the share flagged per set when head F gives up its top `give_up` flags in its tightest pool."""
    if give_up:
        pol = np.sort(f_logits["polished dev"])[::-1]
        f_line = float(pol[int(np.floor(0.05 * len(pol))) - give_up])
    line = max(threshold_for(edit[n], 0.005) for n in REAL)
    for n in REAL:
        f_flag = f_logits[n] > f_line
        room = int(np.floor(0.05 * len(f_flag))) - int(f_flag.sum())
        rest = np.sort(edit[n][~f_flag])[::-1]
        if room < len(rest):
            line = max(line, float(rest[max(room, 0)]))
    return line, {n: float(((f_logits[n] > f_line) | (edit[n] > line)).mean()) for n in edit}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", type=Path, required=True, help="folder with pe_stepN.pt + head_stepN.pt")
    parser.add_argument("--frozen-head", type=Path, default=Path("weights/edit_check_v2/head_EC2.pt"))
    args = parser.parse_args()

    f_line = json.load(open("weights/open_gen_v1/head_F_calibration.json"))["ai_logit"]
    steps = sorted(int(re.search(r"(\d+)", p.stem).group(1)) for p in args.run.glob("pe_step*.pt"))
    candidates = [("frozen PE + " + args.frozen_head.parent.name, None, args.frozen_head)]
    candidates += [(f"fine-tuned step {s}", args.run / f"pe_step{s}.pt", args.run / f"head_step{s}.pt") for s in steps]
    views = f_logits = None
    print("joint rule with head F; caught = (head F says AI) or (edit check above its line)")
    for name, pe_path, head_path in candidates:
        pe = PECore(tuned_path=pe_path)
        if views is None:
            views, f_logits = dev_views(pe)
        edit = edit_scores(pe, LinearHead.load(head_path), views)
        edits = np.r_[edit["edits v2 local"], edit["edits v2 whole"]]
        auc = metrics.auc(np.r_[edit["originals v2"], edits], np.r_[np.zeros(len(edit["originals v2"])), np.ones(len(edits))] == 1)
        for give_up in (0, 1):
            line, caught = joint(edit, f_logits, f_line, give_up)
            reals = " ".join(f"{caught[n]:.1%}" for n in REAL)
            fakes = " | ".join(f"{n} {caught[n]:.0%}" for n in AI)
            print(f"{name:<34} trade {give_up} | AUC {auc:.3f} | line {line:+6.2f} | real {reals} | {fakes}", flush=True)
        del pe
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
