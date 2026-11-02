"""Face-swap models on MY real photos (private): how many real faces would each model flag? Totals only.

For every photo of a private folder: SAM 3 faces (the app's settings), the app's face crop at 336 (JPEG 90), then each
face-swap model (a fine-tuned Perception Encoder + its head). Printed per model: faces checked, the share above the
model's strict line from the PUBLIC real pools (what it would flag before my photos raise its line), the share above
its 5%-style "some signs" level, and the highest score as a multiple of the line. No file name, picture or single
score is printed or saved; nothing is sent anywhere (fully local).

Usage (from the repo root; run it yourself, the photos are private):
  python -m src.eval.private_face_compare --folder data/private/phone_v1 \
      --model "v2:weights/face_pe_v2/pe_step1000.pt:weights/face_pe_v2/head_step1000.pt:12.60" \
      --model "v3:weights/face_pe_v3/pe_step1000.pt:weights/face_pe_v3/head_step1000.pt:8.93"
"""

import os

os.environ["HF_HUB_OFFLINE"] = "1"  # the models come from the local cache; nothing is sent anywhere
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import argparse  # noqa: E402
import warnings  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image, ImageOps  # noqa: E402

from src.data.preprocess import jpeg  # noqa: E402
from src.faces import crop  # noqa: E402
from src.faces.sam_faces import FaceFinder  # noqa: E402
from src.models.linear_head import LinearHead  # noqa: E402
from src.models.pe_core import PECore  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--folder", type=Path, required=True)
    parser.add_argument("--model", action="append", required=True, help='"NAME:PE weights:head:public strict line"')
    args = parser.parse_args()
    warnings.filterwarnings("ignore")  # library warnings could mention a file

    finder = FaceFinder()
    models = {}
    for spec in args.model:
        name, pe_path, head_path, line = spec.split(":")
        models[name] = (PECore(tuned_path=pe_path), LinearHead.load(head_path), float(line))
    scores = {name: [] for name in models}
    photos = unreadable = 0
    for path in sorted(p for p in args.folder.rglob("*") if p.is_file()):
        try:
            with Image.open(path) as im:
                image = ImageOps.exif_transpose(im).convert("RGB")
        except Exception:
            unreadable += 1
            continue
        photos += 1
        faces = [f for f in finder.find(image) if not f["too_small"]]
        if not faces:
            continue
        for name, (pe, head, _) in models.items():
            views = torch.stack([pe.to_tensor(jpeg(crop.face_crop(image, f, size=pe.size), 90)) for f in faces])
            scores[name] += head.logits(pe.layer_features(views).flatten(1)).numpy().tolist()
    print(f"\n{photos} photos ({unreadable} unreadable), fully offline. Real faces checked per model:")
    for name, (_, _, line) in models.items():
        s = np.array(scores[name])
        if not len(s):
            print(f"  {name}: no face big enough")
            continue
        print(f"  {name:<10} {len(s)} faces | above its public strict line ({line:+.2f}): {(s > line).mean():5.1%} "
              f"| above half that line: {(s > line / 2).mean():5.1%} | highest score = {s.max() / line:.2f} x the line")


if __name__ == "__main__":
    main()
