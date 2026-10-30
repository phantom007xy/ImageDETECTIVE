"""Extract DINOv3 average patch vectors of face crops, for the face head (Path F).

Every photo goes through SAM 3 (src/faces/sam_faces.py); every face that is not too small is
cropped with the calibrated rule (src/faces/crop.py) and described by DINOv3 at 224 x 224.
With --whole-image-faces the images already are face crops (e.g. WildDeepfake): no SAM, the
whole image is resized to 224 x 224.

Inputs are SID-Set Parquet files or CSV image lists (path, label, source), as in extract.py.

Usage (from the repo root):
  python -m src.features.extract_faces INPUT [INPUT ...] --out-dir features --mode train --copies 4
"""

import argparse
import time
from pathlib import Path

import numpy as np
import torch

from src.data import preprocess
from src.faces import crop
from src.faces.sam_faces import FaceFinder
from src.features.extract import read_input, view_rng
from src.models.backbone import Backbone


def extract_file(backbone, finder, input_path, out_path, args):
    features, labels, img_ids, face_ids, copy_ids, sources, face_scores = [], [], [], [], [], [], []
    batch = []

    def run_batch():
        if batch:
            features.append(backbone.mean_patch_vector(torch.stack(batch)).cpu().numpy().astype(np.float16))
            batch.clear()

    start, images, faces_total = time.time(), 0, 0
    for row in read_input(input_path, tuple(args.labels)):
        images += 1
        if args.whole_image_faces:
            faces = [None]
        else:
            faces = [f for f in finder.find(row["image"]) if not f["too_small"]]
        for face_id, face in enumerate(faces):
            faces_total += 1
            for copy in range(args.copies if args.mode == "train" else 1):
                if face is None:
                    view = crop.ready_crop_view(row["image"])
                elif args.mode == "train":
                    view = crop.train_face_view(row["image"], face, view_rng(f"{row['img_id']}#{face_id}", copy, args.seed))
                else:
                    view = crop.eval_face_view(row["image"], face)
                batch.append(preprocess.to_tensor(view))
                labels.append(row["label"])
                img_ids.append(row["img_id"])
                face_ids.append(face_id)
                copy_ids.append(copy)
                sources.append(row["source"])
                face_scores.append(face["score"] if face else 1.0)
                if len(batch) == args.batch_size:
                    run_batch()
        if images % 200 == 0:
            print(f"  {images} images, {faces_total} faces, {time.time() - start:.0f} s")
    run_batch()

    if not labels:
        print(f"{input_path.name}: {images} images, no faces found")
        return
    np.savez_compressed(
        out_path,
        features=np.concatenate(features),
        labels=np.array(labels, dtype=np.int8),
        img_ids=np.array(img_ids),
        face_ids=np.array(face_ids, dtype=np.int16),
        copies=np.array(copy_ids, dtype=np.int8),
        sources=np.array(sources),
        face_scores=np.array(face_scores, dtype=np.float16),
        mode=args.mode,
    )
    print(f"{input_path.name}: {images} images, {faces_total} faces -> {len(labels)} vectors "
          f"in {time.time() - start:.0f} s -> {out_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("inputs", nargs="+", type=Path, help="SID-Set .parquet files or .csv image lists")
    parser.add_argument("--out-dir", type=Path, default=Path("features/faces"))
    parser.add_argument("--mode", choices=["train", "eval"], default="train")
    parser.add_argument("--copies", type=int, default=4, help="random copies per face (train mode only)")
    parser.add_argument("--labels", type=int, nargs="+", default=[0, 1], help="0 = real, 1 = AI")
    parser.add_argument("--whole-image-faces", action="store_true", help="images are face crops already (no SAM)")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    backbone = Backbone()
    finder = None if args.whole_image_faces else FaceFinder()
    print(f"DINOv3 on {backbone.device} ({backbone.dtype})" + ("" if finder is None else f", SAM 3 on {finder.device}"))
    for path in args.inputs:
        extract_file(backbone, finder, path, args.out_dir / f"{path.stem}_{args.mode}.npz", args)


if __name__ == "__main__":
    main()
