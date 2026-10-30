"""Read images listed in a CSV file (for folder datasets such as Chameleon or the SD faces).

The CSV has one row per image with the columns: path, label (0 = real, 1 = AI), source.
"""

import csv

from PIL import Image, ImageOps


def iter_rows(list_path):
    """Yield one dict per image, in the same format as src.data.sid_set.iter_rows."""
    with open(list_path, newline="") as f:
        for row in csv.DictReader(f):
            with Image.open(row["path"]) as im:
                image = ImageOps.exif_transpose(im).convert("RGB")  # upright, as the app sees it
            yield {
                "img_id": row["path"],
                "label": int(row["label"]),
                "source": row["source"],
                "image": image,
                "mask": None,
            }
