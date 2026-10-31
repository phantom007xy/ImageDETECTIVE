"""Leak check: no dev or test photo may appear in any training list, also not as the source of an edit.

A photo is identified by its source: SID-Set exports by their image id (the same photo can be exported to several
folders), Commons photos by their full path (file numbers repeat across downloads), Muse edits and their originals by
the source photo in their manifest. Prints the overlaps; 0 everywhere = no leak.

Usage (from the repo root):  python -m src.eval.leak_check
"""

import csv
import glob
from pathlib import Path

TRAIN = ["features/lists/muse_edits_train_*_path_a.csv", "features/lists/muse_pairs_train.csv",
         "features/lists/pairs_v2/muse_pairs_train.csv", "features/lists/commons_train.csv",
         "features/lists/sid_10_14_reals.csv", "features/lists/sid_reals_15_29.csv", "features/lists/commons_v3_train.csv"]
DEV_TEST = ["features/lists/commons_dev.csv", "features/lists/muse_pairs_val.csv", "features/lists/muse_edits_v1.csv",
            "features/lists/muse_edits_test_v2.csv", "features/lists/muse_image_test_v1.csv",
            "features/lists/sid_reals_0_9_unseen.csv", "features/lists/commons_v3_dev.csv"]
MANIFESTS = "D:/Datasets/MuseImage/edits*/manifest.csv"


def edit_sources():
    """Muse edit file stem (edit dir / id) -> the source photo it was made from."""
    out = {}
    for manifest in glob.glob(MANIFESTS):
        folder = Path(manifest).parent.name
        for row in csv.DictReader(open(manifest, newline="")):
            out[f"{folder}/{row['id']}"] = row["source"]
    return out


def identity(path, sources):
    p = Path(path)
    if "MuseImage" in p.parts and p.parent.name.startswith("edits"):  # an edit or its original -> its source photo
        source = sources.get(f"{p.parent.name}/{'_'.join(p.stem.split('_')[:2])}")
        return identity(source, sources) if source else str(p)
    if "SID_Set" in p.parts:
        return "sid:" + p.stem  # the same SID-Set image id in any export folder
    return str(p).replace("\\", "/").lower()


def ids(patterns, sources):
    out = {}
    for pattern in patterns:
        for path in glob.glob(pattern):
            for row in csv.DictReader(open(path, newline="")):
                out.setdefault(identity(row["path"], sources), set()).add(Path(path).name)
    return out


def main():
    sources = edit_sources()
    train = ids(TRAIN, sources)
    print(f"training photos (by source): {len(train)}")
    total = 0
    for pattern in DEV_TEST:
        dev = ids([pattern], sources)
        overlap = sorted(set(dev) & set(train))
        total += len(overlap)
        print(f"  {Path(pattern).name:<26} {len(dev):>5} photos | also in training: {len(overlap)}"
              + (f"  e.g. {overlap[:2]} in {sorted(train[overlap[0]])}" if overlap else ""))
    print("no leak" if total == 0 else f"LEAK: {total} dev/test photos also in training")


if __name__ == "__main__":
    main()
