"""Per-patch DINOv3 features for Path B, the edited-region head (type c: local AI edits).

Path B asks for every 16 x 16 patch: was this spot changed by AI? It learns from SID-Set's edited
("tampered") images, which come with a mask of the changed area:
  - patches at least half inside the mask   -> 1 (edited)
  - patches completely outside the mask     -> 0 (not edited); border patches are left out
  - every patch of a real photo             -> 0
Fully AI-generated images are left out on purpose: they are Path A's job, and Path B should learn
"this spot doesn't fit the rest of the photo", not "AI texture". Every picture is seen through the
app's view (src/data/preprocess.inference_view), the mask goes through the same resize and trim.

Modes:
  train: up to --per-image patches per picture (all edited ones + as many unedited), 2 copies:
         as it is, and a WhatsApp-like copy (shrunk, JPEG 65-85), so Path B survives forwarding.
  eval:  ALL patches of every picture, one copy, with the patch grid (for region maps and IoU).

Inputs: SID-Set Parquet files (tampered images with masks; --labels 2) or CSV image lists of real photos.

Usage (from the repo root):
  python -m src.features.extract_patches D:/Datasets/SID_Set/data/train-00010-of-00249.parquet ... --mode train --labels 2
      --out features/patches/sid_tampered_train.npz
"""

import argparse
import io
import random
import time
import zlib
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from src.data import preprocess
from src.features.extract import read_input
from src.models.backbone import Backbone

PATCH = 16


def view_with_mask(image, mask, rng=None, app_view=None):
    """The app view of the picture (aspect kept, preprocess.APP_VIEW or app_view, trimmed to whole patches), and
    the share of every patch inside the mask (grid of rows x cols; all zeros for a real photo without a mask)."""
    if rng is not None:  # the WhatsApp-like copy: shrink and re-save, the mask is not affected
        w, h = image.size
        scale = rng.uniform(0.5, 1.0)
        image = image.resize((max(64, round(w * scale)), max(64, round(h * scale))), Image.BICUBIC)
        image = preprocess.jpeg(image, rng.randint(65, 85))
    view = preprocess.inference_view(image, app_view)
    rows, cols = view.size[1] // PATCH, view.size[0] // PATCH
    if mask is None:
        return view, np.zeros((rows, cols), dtype=np.float32)
    # The same resize and centre trim as inference_view, done on the mask.
    w, h = preprocess.view_size(*image.size, app_view)
    m = np.asarray(mask.convert("L").resize(image.size, Image.NEAREST).resize((w, h), Image.BILINEAR), dtype=np.float32) / 255
    left, top = (w - cols * PATCH) // 2, (h - rows * PATCH) // 2
    m = m[top:top + rows * PATCH, left:left + cols * PATCH]
    return view, m.reshape(rows, PATCH, cols, PATCH).mean(axis=(1, 3))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--mode", choices=["train", "eval"], default="train")
    parser.add_argument("--labels", type=int, nargs="+", default=[0, 2], help="0 = real, 2 = tampered (SID-Set)")
    parser.add_argument("--per-image", type=int, default=96, help="train mode: patches kept per picture")
    parser.add_argument("--max-images", type=int, help="at most this many pictures per input")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--view", choices=["long448", "short448"], help="app view other than preprocess.APP_VIEW")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    backbone = Backbone()
    feats, labels, image_ids, positions, shares, grids = [], [], [], [], [], {}
    start, images, skipped = time.time(), 0, 0
    for path in args.inputs:
        for n, row in enumerate(read_input(path, tuple(args.labels))):
            if args.max_images and n >= args.max_images:
                break
            mask = Image.open(io.BytesIO(row["mask"])) if row.get("mask") else None
            if mask is not None and mask.size != row["image"].size:
                skipped += 1  # the few SID-Set masks of another size than their picture: left out (plan 2.9)
                continue
            if row["label"] == 2 and mask is None:
                skipped += 1
                continue
            images += 1
            rng = random.Random(zlib.crc32(f"{args.seed}:{row['img_id']}".encode()))
            copies = [None, rng] if args.mode == "train" else [None]
            for copy, copy_rng in enumerate(copies):
                view, share = view_with_mask(row["image"], mask, copy_rng, args.view)
                tokens = backbone.patch_tokens(preprocess.to_tensor(view).unsqueeze(0))[0].float().cpu().numpy()
                flat = share.reshape(-1)
                if args.mode == "train":
                    inside = np.flatnonzero(flat >= 0.5)
                    outside = np.flatnonzero(flat == 0)
                    k = min(len(inside), args.per_image // 2)
                    pick_in = rng.sample(list(inside), k) if k else []
                    pick_out = rng.sample(list(outside), min(len(outside), args.per_image - k))
                    keep = np.array(pick_in + pick_out, dtype=int)
                else:
                    keep = np.arange(len(flat))
                    grids[f"{row['img_id']}"] = share.shape
                feats.append(tokens[keep].astype(np.float16))
                labels.append((flat[keep] >= 0.5).astype(np.int8))
                shares.append(flat[keep].astype(np.float16))
                image_ids += [f"{row['img_id']}#{copy}"] * len(keep)
                positions.append(keep.astype(np.int16))
            if images % 200 == 0:
                print(f"  {images} pictures, {sum(map(len, labels))} patches, {time.time() - start:.0f} s", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, features=np.concatenate(feats), labels=np.concatenate(labels),
                        mask_share=np.concatenate(shares), positions=np.concatenate(positions),
                        img_ids=np.array(image_ids), grid_ids=np.array(list(grids)),
                        grid_shapes=np.array(list(grids.values()), dtype=np.int16).reshape(-1, 2), mode=args.mode)
    edited = int(sum(lab.sum() for lab in labels))
    print(f"done: {images} pictures ({skipped} skipped), {sum(map(len, labels))} patches ({edited} edited) "
          f"in {time.time() - start:.0f} s -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
