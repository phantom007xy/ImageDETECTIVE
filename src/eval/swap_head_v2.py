"""Face-swap head v2: does OpenForensics (face swaps in still photos) make the face-swap check better?

The app's face-swap check is a linear head on the DINOv3 features of each face crop, trained on DFDC video frames
(results/dfdc_face_v1.txt). This script trains variants with OpenForensics faces too (src/features/
extract_openforensics_faces.py: real and AI-replaced faces from the same photos) and compares them on:
  - OpenForensics dev: 15% of the Val photos (by photo), never trained on; used to choose between the variants
  - OpenForensics Test-Dev: a different split of the dataset, the untouched test (per face and per photo)
  - DFDC part 3 (held-out videos) and WildDeepfake test (internet videos, its own tighter crops)
  - REAL faces from photos (polished Commons, OpenImages, close-ups, selfies and portraits, ...): false alarms
  - AI-made faces (Muse, SDXL, GAN): does the swap head notice them too? (that is the AI-made face check's job)
Every head gets its own strict line, as in the app: at most 0.5% of the real faces of EVERY public real pool above it
(src/calibrate/calibrate.py; my phone faces can raise it further when calibrate_app runs). Numbers at the strict line,
plus AUC; "5% line" = at most 5% of every real pool ("some swap signs" in the app).

--features picks which DINOv3 features the heads use: "features" (the last layer: what the app's heads use now) or
middle layers saved with --layers ("features_L12"), several joined with "+". Feature files without them are skipped.

Usage (from the repo root):
  python -m src.eval.swap_head_v2 [--features features_L12] [--faces-dir features/faces_ml]
      [--save weights/swap_v2/head_swap_v2.pt --save-variant "DFDC + OF"]
"""

import argparse
import zlib
from pathlib import Path

import numpy as np
import torch

from src.eval import metrics
from src.models.linear_head import LinearHead

REAL_POOLS = {"polished dev": "commons_dev", "OpenImages val": "muse_pairs_val", "OpenImages unseen": "sid_reals_0_9_unseen",
              "Commons v3 dev": "commons_v3_dev", "close-ups": "real_face_crops_dev_all", "selfies+portraits": "commons_v4_dev",
              "SID val": "validation-00000-of-00034", "DFDC p3 real": "dfdc_part3"}
AI_FACES = {"Muse test v1": "muse_image_test_v1", "SD faces": "sd_faces", "GAN faces (TPDNE)": "tpdne_test"}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--features", default="features", help='feature keys joined with "+", e.g. features+features_L12')
    parser.add_argument("--faces-dir", type=Path, nargs="+", default=[Path("features/faces")],
                        help="folders with face feature files (the first one that has a file with the keys wins)")
    parser.add_argument("--weight-decay", type=float, nargs="+", default=[1e-2])
    parser.add_argument("--save", type=Path, help="save one variant here (--save-variant)")
    parser.add_argument("--save-variant", default="DFDC + OF")
    args = parser.parse_args()
    keys = args.features.split("+")

    def find(stem):
        for folder in args.faces_dir:
            path = folder / f"{stem}.npz"
            if path.exists() and all(k in np.load(path).files for k in keys):
                return path
        return None

    def load(stem, label=None):
        """(x, labels, groups) with the chosen features, or None if no file has them."""
        path = find(stem)
        if path is None:
            print(f"  ({stem}: no file with {args.features}: skipped)")
            return None
        d = np.load(path)
        keep = np.ones(len(d["labels"]), bool) if label is None else d["labels"] == label
        x = np.concatenate([d[k].reshape(len(d[k]), -1) for k in keys], axis=1)[keep].astype(np.float32)
        groups = d["groups"] if "groups" in d.files else d["img_ids"]
        return x, d["labels"][keep].astype(int), np.asarray(groups)[keep].astype(str)

    x_of, y_of, g_of = load("openforensics_val_train")
    dev = np.array([zlib.crc32(g.encode()) % 100 < 15 for g in g_of])  # 15% of the photos: the dev part
    data = {"OF": (x_of[~dev], y_of[~dev])}
    dfdc = load("dfdc_train")
    if dfdc:
        data["DFDC"] = dfdc[:2]
    # Real faces of SID-Set train files 0-9 (OpenImages), without the photos of any real pool below: some unseen
    # OpenImages photos and the Muse pairs' real photos come from the same files.
    held = {Path(str(i).replace("\\", "/")).stem for s in REAL_POOLS.values() if find(f"{s}_eval")
            for i in np.load(find(f"{s}_eval"))["img_ids"]}
    sid = []
    for stem in sorted({p.stem for folder in args.faces_dir for p in folder.glob("train-000*-of-00249_train.npz")}):
        if (path := find(stem)):
            d = np.load(path)
            keep = (d["labels"] == 0) & ~np.isin(d["img_ids"].astype(str), list(held))
            sid.append(np.concatenate([d[k].reshape(len(d[k]), -1) for k in keys], axis=1)[keep].astype(np.float32))
    if sid:
        data["SID real"] = (np.concatenate(sid), np.zeros(sum(len(x) for x in sid), int))
    print("training faces (4 random copies each): " + " | ".join(
        f"{n} {int((y == 0).sum())} real / {int((y == 1).sum())} fake" for n, (_, y) in data.items())
        + f" | OF dev part held out: {int((y_of[dev] == 0).sum())} real / {int((y_of[dev] == 1).sum())} fake")

    mixes = {"OF": ["OF"], "DFDC + OF": ["DFDC", "OF"], "DFDC + OF + SID real faces": ["DFDC", "OF", "SID real"]}
    heads = {}
    if args.features == "features":
        heads["DFDC v1 (app now)"] = LinearHead.load("weights/dfdc_v1/head_dfdc.pt")
    for wd in args.weight_decay:
        for name, parts in mixes.items():
            if all(p in data for p in parts):
                x = torch.from_numpy(np.concatenate([data[p][0] for p in parts]))
                y = torch.from_numpy(np.concatenate([data[p][1] for p in parts]).astype(np.float32))
                heads[f"{name} (wd {wd:g})"] = LinearHead().fit(x, y, weight_decay=wd)

    reals = {n: r[0] for n, s in REAL_POOLS.items() if (r := load(f"{s}_eval", 0))}
    tests = {"OF dev (train views)": (x_of[dev], y_of[dev], g_of[dev])}
    for name, stem in (("OF Test-Dev", "openforensics_test_dev_eval"), ("DFDC p3", "dfdc_part3_eval")):
        if (t := load(stem)):
            tests[name] = t
    if (w := load("wilddeepfake_eval")):
        tests["WildDeepfake"] = (w[0], w[1], np.array([str(Path(i).parent) for i in w[2]]))
    ai = {n: r[0] for n, s in AI_FACES.items() if (r := load(f"{s}_eval"))}

    print(f"\nfeatures: {args.features} | real pools for the lines: {', '.join(reals)}")
    for name, head in heads.items():
        score = lambda x: head.logits(torch.from_numpy(x)).numpy()  # noqa: E731
        real_scores = {n: score(x) for n, x in reals.items()}
        strict = max(float(np.quantile(s, 0.995)) for s in real_scores.values())
        leans = max(float(np.quantile(s, 0.95)) for s in real_scores.values())
        tight = max(real_scores, key=lambda n: np.quantile(real_scores[n], 0.995))
        print(f"\n== {name}: strict line {strict:+.2f} (set by {tight}), 5% line {leans:+.2f}")
        for tname, (x, y, g) in tests.items():
            s = score(x)
            line = (f"  {tname:<20} AUC {metrics.auc(s, y == 1):.3f} | swaps caught: strict {(s[y == 1] > strict).mean():5.1%}, "
                    f"5% line {(s[y == 1] > leans).mean():5.1%} | its own real faces flagged (strict) {(s[y == 0] > strict).mean():5.1%}")
            if tname.startswith("OF"):
                fake_photos = {gg for gg, lab in zip(g, y) if lab == 1}
                hit = {gg for ss, lab, gg in zip(s, y, g) if lab == 1 and ss > strict}
                line += f" | photos with a caught swap {len(hit) / max(1, len(fake_photos)):5.1%}"
            print(line)
        print("  REAL faces flagged (strict): " + " | ".join(f"{n} {(s > strict).mean():.1%}" for n, s in real_scores.items()))
        print("  AI-made faces caught (strict): " + " | ".join(f"{n} {(score(x) > strict).mean():.1%}" for n, x in ai.items()))

    if args.save:
        key = next(k for k in heads if k.startswith(args.save_variant + " (wd"))
        args.save.parent.mkdir(parents=True, exist_ok=True)
        heads[key].save(args.save)
        print(f"\nsaved {key} -> {args.save}")


if __name__ == "__main__":
    main()
