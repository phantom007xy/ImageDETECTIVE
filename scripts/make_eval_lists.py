"""Build CSV image lists (path, label, source) for evaluating the Path A head on other datasets.

  - Chameleon, random: N real + N AI images.
  - Chameleon, size-matched: real and AI images taken in equal numbers per short-side bin, so the
    image size (which alone separates the classes with AUC 0.83) can't help.
  - SD faces: N AI images, spread evenly over SD 1.5 / 2.1 / XL and man / woman.
  - WildDeepfake (test split): a few evenly spaced frames per sequence (face swaps, ready-made crops).
  - Muse Image: every successfully generated image of a folder made by scripts/make_muse_images.py.

Usage:
  python scripts/make_eval_lists.py --chameleon D:/Datasets/Chameleon/test
      --sd-faces D:/Datasets/stable-diffusion-face-dataset/stable-diffusion-face-dataset --out-dir features/lists
"""

import argparse
import csv
import random
from collections import defaultdict
from pathlib import Path

from PIL import Image

BIN = 32  # short-side bin width in pixels


def write_list(path, rows):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["path", "label", "source"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {path}")


def short_side(path):
    with Image.open(path) as im:  # reads only the header, not the pixels
        return min(im.size)


def chameleon_lists(root, n, rng, out_dir):
    classes = {0: sorted((root / "0_real").glob("*.jpg")), 1: sorted((root / "1_fake").glob("*.jpg"))}
    source = {0: "chameleon/real", 1: "chameleon/ai"}

    random_rows = []
    for label, files in classes.items():
        random_rows += [{"path": str(p), "label": label, "source": source[label]} for p in rng.sample(files, n)]
    write_list(out_dir / "chameleon_random.csv", random_rows)

    bins = {0: defaultdict(list), 1: defaultdict(list)}
    for label, files in classes.items():
        for p in files:
            bins[label][short_side(p) // BIN].append(p)
    matched = {0: [], 1: []}
    for b in sorted(set(bins[0]) & set(bins[1])):
        k = min(len(bins[0][b]), len(bins[1][b]))
        for label in (0, 1):
            matched[label] += rng.sample(bins[label][b], k)
    k = min(n, len(matched[0]))
    matched_rows = []
    for label in (0, 1):
        matched_rows += [{"path": str(p), "label": label, "source": source[label]} for p in rng.sample(matched[label], k)]
    print(f"size-matched pool: {len(matched[0])} real + {len(matched[1])} AI share a short-side bin "
          f"(of {len(classes[0])} real, {len(classes[1])} AI)")
    write_list(out_dir / "chameleon_size_matched.csv", matched_rows)


def sd_faces_list(root, n, rng, out_dir):
    groups = [sorted((root / res / who).glob("*.jpg")) for res in ("512", "768", "1024") for who in ("man", "woman")]
    per_group = n // len(groups)
    rows = []
    for files in groups:
        version = {"512": "sd1.5", "768": "sd2.1", "1024": "sdxl"}[files[0].parent.parent.name]
        rows += [{"path": str(p), "label": 1, "source": f"sd_faces/{version}"} for p in rng.sample(files, per_group)]
    write_list(out_dir / "sd_faces.csv", rows)


def sd_faces_train_list(root, n_per_version, rng, out_dir):
    """Training list: SD 1.5 and SD 2.1 faces that are NOT in sd_faces.csv (the dev list).

    SDXL stays out completely, as the generator the head has never seen. n_per_version is kept
    small on purpose: many more AI portraits than real portraits would teach "portrait = AI".
    """
    with open(out_dir / "sd_faces.csv", newline="") as f:
        dev = {r["path"] for r in csv.DictReader(f)}
    rows = []
    for res, version in (("512", "sd1.5"), ("768", "sd2.1")):
        files = sorted(str(p) for who in ("man", "woman") for p in (root / res / who).glob("*.jpg") if str(p) not in dev)
        rows += [{"path": p, "label": 1, "source": f"sd_faces/{version}"} for p in rng.sample(files, n_per_version)]
    write_list(out_dir / "sd_faces_train.csv", rows)


def wilddeepfake_list(root, frames_per_seq, out_dir):
    """A few evenly spaced frames per sequence (neighbouring frames are near-identical)."""
    rows = []
    for group in ("real_test", "fake_test"):
        label = 0 if group == "real_test" else 1
        for seq in sorted(p for p in (root / group).rglob("*") if p.is_dir() and any(p.glob("*.png"))):
            frames = sorted(seq.glob("*.png"), key=lambda f: int(f.stem) if f.stem.isdigit() else f.stem)
            picks = sorted({round(i * (len(frames) - 1) / max(1, frames_per_seq - 1)) for i in range(frames_per_seq)})
            rows += [{"path": str(frames[i]), "label": label, "source": f"wilddeepfake/{'real' if label == 0 else 'fake'}"}
                     for i in picks]
    write_list(out_dir / "wilddeepfake.csv", rows)


def muse_image_list(folder, out_dir):
    """All images in a Muse Image folder that were generated successfully (from its manifest.csv)."""
    with open(folder / "manifest.csv", newline="", encoding="utf-8") as f:
        rows = [{"path": str(folder / r["file"]), "label": 1, "source": f"muse_image/{r['category']}"}
                for r in csv.DictReader(f) if r["status"] == "ok"]
    write_list(out_dir / f"muse_image_{folder.name}.csv", rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--chameleon", type=Path)
    parser.add_argument("--sd-faces", type=Path)
    parser.add_argument("--wilddeepfake", type=Path, help="the extracted WildDeepfake test folder")
    parser.add_argument("--frames-per-seq", type=int, default=8)
    parser.add_argument("--muse-image", type=Path, help="a folder made by scripts/make_muse_images.py")
    parser.add_argument("--sd-faces-train", type=int, help="with --sd-faces: also a training list, N per SD 1.5 / 2.1")
    parser.add_argument("--n", type=int, default=1000, help="images per class (Chameleon) or in total (SD faces)")
    parser.add_argument("--out-dir", type=Path, default=Path("features/lists"))
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    if args.chameleon:
        chameleon_lists(args.chameleon, args.n, rng, args.out_dir)
    if args.sd_faces and args.sd_faces_train:
        sd_faces_train_list(args.sd_faces, args.sd_faces_train, rng, args.out_dir)  # keeps sd_faces.csv as it is
    elif args.sd_faces:
        sd_faces_list(args.sd_faces, args.n, rng, args.out_dir)
    if args.wilddeepfake:
        wilddeepfake_list(args.wilddeepfake, args.frames_per_seq, args.out_dir)
    if args.muse_image:
        muse_image_list(args.muse_image, args.out_dir)


if __name__ == "__main__":
    main()
