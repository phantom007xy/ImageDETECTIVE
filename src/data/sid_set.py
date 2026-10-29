"""Read SID-Set Parquet files row by row (Hugging Face `saberzl/SID_Set`, CC BY 4.0).

Each row has: img_id, image (bytes), mask (bytes, tampered images only), width, height and
label (0 = real, 1 = fully synthetic, 2 = tampered).
"""

import io

import pyarrow.parquet as pq
from PIL import Image

LABEL_NAMES = {0: "real", 1: "synthetic", 2: "tampered"}
# Every feature carries a source tag, so data sources can be switched in or out later.
SOURCES = {0: "sid_set/openimages", 1: "sid_set/flux", 2: "sid_set/tampered"}


def iter_rows(parquet_path, labels=(0, 1, 2), batch_size=16):
    """Yield one dict per image: img_id, label, source, image (PIL RGB) and mask bytes (or None)."""
    table = pq.ParquetFile(parquet_path)
    for batch in table.iter_batches(batch_size=batch_size, columns=["img_id", "label", "image", "mask"]):
        for row in batch.to_pylist():
            if row["label"] not in labels:
                continue
            mask = row.get("mask")
            yield {
                "img_id": row["img_id"],
                "label": row["label"],
                "source": SOURCES[row["label"]],
                "image": Image.open(io.BytesIO(row["image"]["bytes"])).convert("RGB"),
                "mask": mask["bytes"] if mask and mask.get("bytes") else None,
            }
