"""SAM 3.1 outline (Meta Model API): the exact outline of the object at the strongest AI evidence.

The evidence map is coarse (one cell = 16 x 16 pixels of the 448-pixel view). Its hottest cell is
sent to SAM 3.1 as a point prompt; SAM 3.1 returns the outline (mask) of the object at that point,
so the app can show WHICH object carries the evidence and how strong the evidence inside it is.
SAM 3.1 never decides the verdict. It costs about $0.0025 per photo; it goes through
src/services/meta_api.py (daily cap, cache, fallback). The photo is sent to Meta's API in standard
mode (Meta does not train on it); a smaller copy is sent (long side 1024 px).

Response format: special-token text, parsed with Meta's own `meta-sam-parser` (SAM licence).
Each object comes as a box and a mask; the mask covers exactly the box region.
"""

import base64
import hashlib
import io

import numpy as np
from meta_sam_parser import (
    CompletedOutcome,
    FramePrompt,
    PromptObject,
    PromptPoint,
    SegmentationMaskRecord,
    build_object_prompt,
    decode_mask_to_raster,
    image_segmentation_format,
)
from PIL import Image

from src.services import meta_api

MODEL = "sam-3.1"
COST = 0.0025  # $2.50 per 1,000 images (Meta's pricing page)
SEND_LONG_SIDE = 1024


def hottest_point(patch_logits, size):
    """Centre of the patch with the strongest AI evidence, in pixels of an image of `size` (w, h)."""
    rows, cols = patch_logits.shape
    row, col = np.unravel_index(int(np.argmax(patch_logits)), patch_logits.shape)
    return int((col + 0.5) / cols * size[0]), int((row + 0.5) / rows * size[1])


def output_text(response):
    return "".join(part.get("text", "") for item in response.get("output", [])
                   for part in item.get("content", []) or [] if part.get("type") == "output_text")


def parse_mask(text, size):
    """The first object's mask as a boolean array of `size` (w, h), its box and confidence; None if empty."""
    parser = image_segmentation_format().create_parser()
    parser.push(text if text.endswith("\n") else text + "\n")
    records = parser.finish(CompletedOutcome()).result.records
    masks = [r for r in records if isinstance(r, SegmentationMaskRecord)]
    if not masks:
        return None
    record = masks[0]
    b = record.bounds
    raster = np.frombuffer(decode_mask_to_raster(record.mask), dtype=np.uint8)
    raster = raster.reshape(record.mask.height, record.mask.width)
    box_w, box_h = b.right - b.left, b.bottom - b.top
    if raster.shape != (box_h, box_w):  # the mask covers the box; fit it if the sizes differ
        raster = np.asarray(Image.fromarray(raster * 255).resize((box_w, box_h), Image.NEAREST)) > 0
    full = np.zeros((size[1], size[0]), dtype=bool)
    full[b.top:b.bottom, b.left:b.right] = raster[: size[1] - b.top, : size[0] - b.left] > 0
    return {"mask": full, "box": (b.left, b.top, b.right, b.bottom), "confidence": record.confidence}


def outline(image, patch_logits):
    """image: the upright PIL RGB photo. Returns the outline at the hottest point (in `image` pixels), or None."""
    small = image.copy()
    small.thumbnail((SEND_LONG_SIDE, SEND_LONG_SIDE))
    point = hottest_point(patch_logits, small.size)
    buffer = io.BytesIO()
    small.save(buffer, format="JPEG", quality=90)
    data = buffer.getvalue()

    def call(key):
        prompt = build_object_prompt(size=small.size, objects=[
            PromptObject(id=1, prompts=[FramePrompt(points=[PromptPoint(x=point[0], y=point[1])])])])
        body = {"model": MODEL, "metadata": {"include_confidence": "true"}, "input": [{
            "type": "message", "role": "user", "content": [
                {"type": "input_text", "text": prompt},
                {"type": "input_image", "image_url": "data:image/jpeg;base64," + base64.b64encode(data).decode()},
            ]}]}
        found = parse_mask(output_text(meta_api.post(key, "/responses", body)), small.size)
        if found is None:
            return {"mask": None, "point": point}, COST  # SAM found no object there: still paid
        # Back to the photo's own size, so the app can draw it on the full image.
        mask = np.asarray(Image.fromarray(found["mask"].astype(np.uint8) * 255).resize(image.size, Image.NEAREST)) > 0
        scale = image.size[0] / small.size[0]
        box = tuple(round(v * scale) for v in found["box"])
        return {"mask": mask, "box": box, "confidence": found["confidence"], "point": point}, COST

    return meta_api.paid_call((MODEL, hashlib.sha256(data).hexdigest(), point), COST, call)


def evidence_inside(patch_logits, mask):
    """Mean patch score inside the outline and outside it (patch grid sampled from the mask)."""
    rows, cols = patch_logits.shape
    grid = np.asarray(Image.fromarray(mask.astype(np.uint8) * 255).resize((cols, rows), Image.BILINEAR)) > 127
    if not grid.any() or grid.all():
        return None
    return float(patch_logits[grid].mean()), float(patch_logits[~grid].mean())
