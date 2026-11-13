"""Named image sets for the view experiments: "NAME:list.csv[@source prefix]" or "NAME:folder/" (private photos).

A CSV list has the columns path, label, source (features/lists/*.csv); "@prefix" keeps only the rows whose source starts
with the prefix (e.g. muse_pairs_val.csv@muse_pairs/real). A folder gives every JPG/PNG/WEBP in it (my private phone
photos: only totals are ever printed). Pictures are opened upright (EXIF orientation applied), as the app does.
"""

import csv
from pathlib import Path

from PIL import Image, ImageOps

IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".webp")


def parse_set(spec):
    """'NAME:path[@prefix]' -> (name, [paths])."""
    name, rest = spec.split(":", 1)
    path, _, prefix = rest.partition("@")
    path = Path(path)
    if path.is_dir():
        paths = sorted(str(p) for p in path.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
    else:
        with open(path, newline="", encoding="utf-8") as f:
            paths = [r["path"] for r in csv.DictReader(f) if not prefix or r["source"].startswith(prefix)]
    return name, paths


def open_upright(path):
    with Image.open(path) as im:
        return ImageOps.exif_transpose(im).convert("RGB")
