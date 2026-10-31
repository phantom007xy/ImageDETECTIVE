"""DINOv3 features of DFDC face crops, real and face-swapped, as same-content pairs (face swaps, type b).

DFDC (Deepfake Detection Challenge, Meta 2020; paid actors who agreed to have their likeness
changed). Used as the organizers recommend: frames and crops are never stored in the repo, only
these feature files on the laptop; the trained weights are never published.

For every real video: a few evenly spaced frames, SAM 3 finds the faces (same settings as the app).
For each of up to N fake videos made from it: the SAME frames, cropped at the SAME face boxes (a
face swap keeps the head's position), with the same random crop and augmentation as the real crop.
So a real/fake pair differs only in the face. In many DFDC fakes only one of two people was swapped,
so a fake crop counts as "fake" only if it really changed: its difference to the real crop must be
clearly above the frame's overall difference (re-encoding noise). Unchanged crops are dropped.

Usage (from the repo root):
  python -m src.features.extract_dfdc_faces D:/Datasets/DFDC/dfdc_train_part_0 ... --mode train --out features/faces/dfdc_train.npz
"""

import argparse
import json
import random
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

from src.data import preprocess
from src.faces import crop
from src.faces.sam_faces import FaceFinder
from src.features.extract import view_rng
from src.models.backbone import Backbone


def read_frames(path, count):
    """`count` evenly spaced frames (skipping the first and last 5%), as PIL RGB images."""
    cap = cv2.VideoCapture(str(path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    wanted = sorted({int(total * (0.05 + 0.9 * i / max(1, count - 1))) for i in range(count)})
    frames, index = {}, 0
    while index <= wanted[-1]:
        ok, frame = cap.read()
        if not ok:
            break
        if index in wanted:
            frames[index] = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        index += 1
    cap.release()
    return frames


def changed(real_img, fake_img, face, ratio=2.0, minimum=3.0):
    """Did the face really change? Mean pixel difference inside SAM's (tight) face box vs. the whole frame.

    A swap changes the inner face; the wide app crop (hair, neck, background) would dilute it.
    """
    box = tuple(round(v) for v in face["box"])
    size = (112, 112)
    a = np.asarray(real_img.crop(box).resize(size), dtype=np.float32)
    b = np.asarray(fake_img.crop(box).resize(size), dtype=np.float32)
    face_diff = float(np.abs(a - b).mean())
    small = (320, 180)
    frame_diff = float(np.abs(np.asarray(real_img.resize(small), dtype=np.float32)
                              - np.asarray(fake_img.resize(small), dtype=np.float32)).mean())
    return face_diff > max(minimum, ratio * frame_diff), face_diff, frame_diff


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("parts", nargs="+", type=Path, help="DFDC part folders (with metadata.json)")
    parser.add_argument("--mode", choices=["train", "eval"], default="train")
    parser.add_argument("--frames", type=int, default=6, help="frames per video")
    parser.add_argument("--fakes-per-real", type=int, default=2)
    parser.add_argument("--copies", type=int, default=4, help="random copies per crop (train mode only)")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-reals", type=int, help="only the first N real videos per part (for a quick test)")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    backbone, finder = Backbone(), FaceFinder()
    rng_pick = random.Random(args.seed)
    feats, labels, ids, groups, diffs, batch = [], [], [], [], [], []
    stats = defaultdict(int)

    def add(view, label, item_id, group, diff):
        batch.append(preprocess.to_tensor(view))
        labels.append(label)
        ids.append(item_id)
        groups.append(group)
        diffs.append(diff)
        if len(batch) == 32:
            feats.append(backbone.mean_patch_vector(torch.stack(batch)).cpu().numpy().astype(np.float16))
            batch.clear()

    start = time.time()
    for part in args.parts:
        meta = json.loads((part / "metadata.json").read_text())
        fakes_of = defaultdict(list)
        for name, info in meta.items():
            if info["label"] == "FAKE":
                fakes_of[info["original"]].append(name)
        reals = sorted(name for name, info in meta.items() if info["label"] == "REAL")[: args.max_reals]
        for n, real in enumerate(reals, 1):
            fakes = rng_pick.sample(sorted(fakes_of[real]), min(args.fakes_per_real, len(fakes_of[real])))
            real_frames = read_frames(part / real, args.frames)
            fake_frames = {f: read_frames(part / f, args.frames) for f in fakes}
            for index, real_img in real_frames.items():
                for face_id, face in enumerate(f for f in finder.find(real_img) if not f["too_small"]):
                    key = f"{part.name}/{real}#{index}#{face_id}"
                    copies = args.copies if args.mode == "train" else 1
                    for copy in range(copies):
                        view = (crop.train_face_view(real_img, face, view_rng(key, copy, args.seed))
                                if args.mode == "train" else crop.eval_face_view(real_img, face))
                        add(view, 0, key, real, 0.0)
                    stats["real faces"] += 1
                    for fake in fakes:
                        fake_img = fake_frames[fake].get(index)
                        if fake_img is None or fake_img.size != real_img.size:
                            continue
                        is_changed, face_diff, frame_diff = changed(real_img, fake_img, face)
                        if not is_changed:
                            stats["fake crops unchanged (dropped)"] += 1
                            continue
                        stats["fake faces"] += 1
                        for copy in range(copies):  # the same random crop as the real face: a true pair
                            view = (crop.train_face_view(fake_img, face, view_rng(key, copy, args.seed))
                                    if args.mode == "train" else crop.eval_face_view(fake_img, face))
                            add(view, 1, f"{part.name}/{fake}#{index}#{face_id}", real, face_diff)
            if n % 25 == 0:
                print(f"  {part.name}: {n}/{len(reals)} real videos, {dict(stats)}, {time.time() - start:.0f} s", flush=True)
    if batch:
        feats.append(backbone.mean_patch_vector(torch.stack(batch)).cpu().numpy().astype(np.float16))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, features=np.concatenate(feats), labels=np.array(labels, dtype=np.int8),
                        img_ids=np.array(ids), groups=np.array(groups), face_diff=np.array(diffs, dtype=np.float16),
                        sources=np.array(["dfdc/" + ("fake" if lab else "real") for lab in labels]), mode=args.mode)
    print(f"done: {dict(stats)} -> {len(labels)} vectors in {time.time() - start:.0f} s -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
