"""Colab: bring our existing face crops (src/faces/make_crops.py; DFDC, OpenForensics Val, SID-Set, Commons, dev and
test pools) to a local folder, from either FaceCrops_colab.zip or an uploaded FaceCrops folder in the Drive folder.

The manifests' laptop paths (D:\\Datasets\\FaceCrops\\...) become paths under --out. A folder uploaded file by file may
be unfinished, so every manifest row whose image (or mask) is missing is dropped, with the count per set printed.
Copying from Drive is slow per file, hence many threads. Runs again quickly: files already copied are kept.

Usage (Colab):  python scripts/colab_facecrops.py --drive /content/drive/MyDrive/ImageDETECTIVE --out /content/FaceCrops
"""

import argparse
import csv
import shutil
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def relative(path):
    """'D:\\Datasets\\FaceCrops\\dfdc_train\\000001.jpg' -> 'dfdc_train/000001.jpg' ('' for an empty path)."""
    path = path.replace("\\", "/")
    return path.split("FaceCrops/", 1)[1] if "FaceCrops/" in path else ""


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--drive", type=Path, required=True, help="the ImageDETECTIVE folder in Drive")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=48)
    args = parser.parse_args()
    zipped, folder = args.drive / "FaceCrops_colab.zip", args.drive / "FaceCrops"
    args.out.mkdir(parents=True, exist_ok=True)
    start = time.time()
    if zipped.exists():
        local = args.out.parent / zipped.name
        shutil.copy(zipped, local)
        subprocess.run(["unzip", "-q", "-o", str(local), "-d", str(args.out.parent)], check=True)
        local.unlink()
        source = args.out
    elif folder.exists():
        source = folder
    else:
        raise SystemExit(f"neither {zipped} nor {folder} exists: upload one of them first")
    manifests = sorted(p for p in source.glob("*.csv") if not p.stem.endswith("_before_dedup"))
    print(f"{len(manifests)} crop lists in {source}", flush=True)
    tables = {}
    for m in manifests:
        with open(m, newline="") as f:
            tables[m.stem] = list(csv.DictReader(f))
    if source == folder:  # copy every listed image and mask from Drive, many at once
        wanted = sorted({rel for rows in tables.values() for r in rows for rel in (relative(r["path"]), relative(r["mask"])) if rel})

        def fetch(rel):
            target = args.out / rel
            if target.exists():
                return True
            try:
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(folder / rel, target)
                return True
            except OSError:
                return False

        copied = 0
        with ThreadPoolExecutor(args.workers) as pool:
            for n, ok in enumerate(pool.map(fetch, wanted), 1):
                copied += ok
                if n % 5000 == 0:
                    print(f"  {n}/{len(wanted)} files, {time.time() - start:.0f} s", flush=True)
        print(f"copied {copied} of {len(wanted)} files from Drive in {time.time() - start:.0f} s", flush=True)
    for stem, rows in tables.items():  # Colab paths; rows with a missing file dropped
        kept = []
        for r in rows:
            path, mask = relative(r["path"]), relative(r["mask"])
            if path and (args.out / path).exists() and (not mask or (args.out / mask).exists()):
                kept.append({**r, "path": str(args.out / path), "mask": str(args.out / mask) if mask else ""})
        with open(args.out / f"{stem}.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]) if rows else ["path"], lineterminator="\n")
            w.writeheader()
            w.writerows(kept)
        missing = len(rows) - len(kept)
        print(f"  {stem}: {len(kept)} crops" + (f" ({missing} missing in the upload: dropped)" if missing else ""), flush=True)


if __name__ == "__main__":
    main()
