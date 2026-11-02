"""Download REAL face photos for the face-swap model (src/models/tune_face_pe.py): more people, skin tones, ages,
cameras and selfies, so real faces are not mistaken for swaps. Made to run on Colab (fast network); also works locally.

Datasets (all CC BY; credited in the README):
  fairface    FairFace (Karkkainen & Joo, WACV 2021; CC BY 4.0): faces balanced over race, gender and age, from
              Flickr photos; the version with more margin (padding 1.25). --n photos sampled evenly over race x gender.
  openimages  Open Images V7 validation and test photos labelled "Human face" (images CC BY 2.0, each with its author and
              licence: written to data/attribution/openimages_faces_v1.csv). Photos in our test pools are left out.
  faceid550   FaceID-550 (Mendeley Data 10.17632/zs2vnf9692.2; CC BY 4.0): 550 people, 21 photos each (an ID photo
              and 20 daily/selfie photos), 256 x 256. 20% of the PEOPLE are held out as a dev pool (real selfies of
              people never seen in training): lists real_faceid550_train.csv and real_faceid550_dev.csv.
  indian      Dataset of Indian face images with various expressions (Mendeley Data 10.17632/zfcd4bny82.1;
              CC BY 4.0): 6,000+ photos from three directions (JPG only; the few HEIC files are skipped).
Writes the photos to --out/<name>/ and features/lists/real_<name>*.csv (path, label 0, source <name>/real).

Usage (from the repo root):  python scripts/download_real_faces.py fairface openimages faceid550 indian --out /content/real
"""

import argparse
import csv
import io
import random
import shutil
import subprocess
import zipfile
import zlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx

FAIRFACE_IMAGES = "1g7qNOZz9wC7OfOhcPqH1EZ5bk1UFGmlL"  # fairface-img-margin125-trainval.zip (official README)
FAIRFACE_LABELS = "1i1L3Yqwaio7YSOCj7ftgk8ZZchPG7dmH"  # fairface_label_train.csv
OI_LABELS = {"validation": "https://storage.googleapis.com/openimages/v7/oidv7-val-annotations-human-imagelabels.csv",
             "test": "https://storage.googleapis.com/openimages/v7/oidv7-test-annotations-human-imagelabels.csv"}
OI_IMAGES = "https://storage.googleapis.com/openimages/2018_04/{0}/{0}-images-with-rotation.csv"
OI_PHOTO = "https://open-images-dataset.s3.amazonaws.com/{}/{}.jpg"
HUMAN_FACE = "/m/0dzct"
MENDELEY = "https://data.mendeley.com/public-api/datasets/{}"
# photos in our real test/dev pools (by OpenImages id in the file name): never training photos
HELD_LISTS = ["features/lists/sid_reals_0_9_unseen.csv", "features/lists/muse_pairs_val.csv",
              "features/lists/muse_edits_test_v2.csv", "features/lists/muse_edits_v1.csv"]


def write_list(name, paths, source):
    out = Path(f"features/lists/real_{name}.csv")
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["path", "label", "source"], lineterminator="\n")
        w.writeheader()
        w.writerows({"path": str(p), "label": 0, "source": source} for p in paths)
    print(f"{len(paths)} photos -> {out}", flush=True)


def gdown(file_id, dest):
    if not dest.exists():
        subprocess.run(["gdown", "--fuzzy", f"https://drive.google.com/file/d/{file_id}/view", "-O", str(dest)], check=True)


def fairface(out, n, seed):
    folder = out / "fairface"
    folder.mkdir(parents=True, exist_ok=True)
    labels, images = out / "fairface_label_train.csv", out / "fairface-img-margin125-trainval.zip"
    gdown(FAIRFACE_LABELS, labels)
    gdown(FAIRFACE_IMAGES, images)
    rows = list(csv.DictReader(open(labels, newline="")))
    by_group = {}
    for r in rows:  # columns: file, age, gender, race, service_test
        by_group.setdefault((r["race"], r["gender"]), []).append(r["file"])
    rng, picked = random.Random(seed), []
    for files in by_group.values():
        picked += rng.sample(files, min(len(files), n // len(by_group)))
    with zipfile.ZipFile(images) as z:
        for name in picked:
            target = folder / name.replace("/", "_")
            if not target.exists():
                target.write_bytes(z.read(name))
    write_list("fairface", sorted(folder / p.replace("/", "_") for p in picked), "fairface/real")


def openimages(out, n, seed, workers, splits):
    folder = out / "openimages"
    folder.mkdir(parents=True, exist_ok=True)
    held = {Path(r["path"].replace("\\", "/")).stem for f in HELD_LISTS if Path(f).exists()
            for r in csv.DictReader(open(f, newline=""))}
    got, meta = [], {}
    with httpx.Client(timeout=300, follow_redirects=True) as client:
        for split in splits:
            labels = client.get(OI_LABELS[split]).text.splitlines()
            ids = sorted({line.split(",")[0] for line in labels[1:] if f",{HUMAN_FACE}," in line and line.endswith(",1")} - held)
            meta.update({r["ImageID"]: r for r in csv.DictReader(io.StringIO(client.get(OI_IMAGES.format(split)).text))})
            ids = [i for i in ids if i in meta and meta[i].get("License", "").startswith("https://creativecommons.org/licenses/by/2.0")
                   and meta[i].get("Rotation", "") in ("", "0", "0.0")]  # stored upright (no rotation needed)
            ids = sorted(random.Random(seed).sample(ids, min(n, len(ids))))
            print(f"Open Images {split} photos labelled 'Human face', CC BY 2.0, not in our pools: picked {len(ids)}", flush=True)

            def fetch(i, split=split):
                target = folder / f"{i}.jpg"
                for _ in range(3):
                    if target.exists():
                        break
                    try:
                        r = client.get(OI_PHOTO.format(split, i))
                        if r.status_code == 200:
                            target.write_bytes(r.content)
                    except httpx.HTTPError:
                        pass
                return target if target.exists() else None

            with ThreadPoolExecutor(workers) as pool:
                got += [p for p in pool.map(fetch, ids) if p]
    write_list("openimages", got, "openimages/real")
    att = Path("data/attribution/openimages_faces_v1.csv")
    with open(att, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["file", "author", "licence", "page_url"], lineterminator="\n")
        w.writeheader()
        w.writerows({"file": p.name, "author": meta[p.stem].get("Author", ""), "licence": meta[p.stem].get("License", ""),
                     "page_url": meta[p.stem].get("OriginalLandingURL", "")} for p in got)
    print(f"credits -> {att}", flush=True)


def mendeley_files(dataset):
    d = httpx.get(MENDELEY.format(dataset), headers={"Accept": "application/json"}, follow_redirects=True, timeout=120).json()
    return [(f["filename"], f["content_details"]["download_url"]) for f in d["files"]]


def unpack(archive, folder):
    """A zip with Python; a RAR (FaceID-550 is one) with unrar or 7-Zip (Colab: apt-get install unrar p7zip-full)."""
    if zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as z:
            z.extractall(folder)
        return
    for cmd in (["unrar", "x", "-o+", "-inul", str(archive), f"{folder}/"], ["7z", "x", "-y", f"-o{folder}", str(archive)],
                ["C:/Program Files/7-Zip/7z.exe", "x", "-y", f"-o{folder}", str(archive)]):
        if shutil.which(cmd[0]) or Path(cmd[0]).exists():
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)
            return
    raise RuntimeError(f"cannot unpack {archive.name}: install unrar or 7-Zip")


def faceid550(out):
    folder = out / "faceid550"
    folder.mkdir(parents=True, exist_ok=True)
    (name, url), = mendeley_files("zs2vnf9692")
    archive = out / name
    if not archive.exists():
        with httpx.stream("GET", url, follow_redirects=True, timeout=600) as r, open(archive, "wb") as f:
            for chunk in r.iter_bytes(1 << 20):
                f.write(chunk)
    unpack(archive, folder)
    photos = sorted(p for p in folder.rglob("*") if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp"))
    if len({p.parent for p in photos}) > 50:  # one folder per person
        person = lambda p: p.parent.name  # noqa: E731
    else:  # all in a few folders: the person is the file name's first part (e.g. 0123_05.jpg)
        person = lambda p: p.stem.replace("-", "_").split("_")[0]  # noqa: E731
    dev = [p for p in photos if zlib.crc32(person(p).encode()) % 100 < 20]
    train = [p for p in photos if zlib.crc32(person(p).encode()) % 100 >= 20]
    print(f"FaceID-550: {len(photos)} photos of {len({person(p) for p in photos})} people "
          f"({len({person(p) for p in dev})} held out)", flush=True)
    write_list("faceid550_train", train, "faceid550/real")
    write_list("faceid550_dev", dev, "faceid550/real")


def indian(out, workers):
    folder = out / "indian"
    folder.mkdir(parents=True, exist_ok=True)
    files = [(n, u) for n, u in mendeley_files("zfcd4bny82") if n.lower().endswith((".jpg", ".jpeg"))]

    def fetch(item):
        name, url = item
        target = folder / name
        for _ in range(3):
            if target.exists():
                break
            try:
                r = httpx.get(url, follow_redirects=True, timeout=300)
                if r.status_code == 200:
                    target.write_bytes(r.content)
            except httpx.HTTPError:
                pass
        return target if target.exists() else None

    with ThreadPoolExecutor(workers) as pool:
        got = [p for p in pool.map(fetch, files) if p]
    write_list("indian", sorted(got), "indian/real")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("datasets", nargs="+", choices=["fairface", "openimages", "faceid550", "indian"])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--fairface-n", type=int, default=30000)
    parser.add_argument("--openimages-n", type=int, default=10000, help="per split")
    parser.add_argument("--openimages-splits", nargs="+", default=["validation", "test"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    for name in args.datasets:
        print(f"=== {name}", flush=True)
        try:  # one dataset failing (e.g. a Drive download quota) must not stop the others
            if name == "fairface":
                fairface(args.out, args.fairface_n, args.seed)
            elif name == "openimages":
                openimages(args.out, args.openimages_n, args.seed, args.workers, args.openimages_splits)
            elif name == "faceid550":
                faceid550(args.out)
            else:
                indian(args.out, args.workers)
        except Exception as e:
            print(f"!!! {name} failed: {e!r} (skipped)", flush=True)


if __name__ == "__main__":
    main()
