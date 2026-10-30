"""Does a face rule help the verdict under ONE shared false-alarm budget? (plan Section 5.6)

Per image: AI if the whole-image logit > t_ai OR its best face logit > t_face. For each t_face, t_ai
is set as low as possible so that every real dev pool keeps at most 5% of its images called AI;
then the fakes caught are counted. The first row (t_face = inf) is the whole image alone. Rows
where even t_ai = inf breaks the budget are marked "over budget" (the face rule alone is too loose).

Whole-image features: the app view (src.features.extract --mode app); face features: the app's face
crop (src.features.extract_faces --mode eval). Both are matched by image path. Dev sets only.

Usage (from the repo root):
  python -m src.eval.shared_budget --head weights/commons_v1/head_E.pt
"""

import argparse
import math

import numpy as np
import torch

from src.models.linear_head import LinearHead

REALS = {"polished dev": ("features/commons_dev_app.npz", "features/faces/commons_dev_eval.npz", None),
         "OpenImages val": ("features/muse_pairs_val_app.npz", "features/faces/muse_pairs_val_eval.npz", None)}
FAKES = {"Muse test v1": ("features/muse_image_test_v1_app.npz", "features/faces/muse_image_test_v1_eval.npz", None),
         "Muse pairs val": ("features/muse_pairs_val_app.npz", "features/faces/muse_pairs_val_eval.npz", None),
         "SDXL": ("features/sd_faces_app.npz", "features/faces/sd_faces_eval.npz", "sd_faces/sdxl")}
MAX_FALSE_ALARMS = 0.05


def per_image(head, whole_npz, faces_npz, label, source):
    """Whole-image logit and best face logit (-inf without a checkable face) of every image of one label."""
    w, f = np.load(whole_npz), np.load(faces_npz)
    keep = w["labels"] == label
    if source:
        keep &= w["sources"] == source
    ids = [str(i) for i in w["img_ids"][keep]]
    whole = head.logits(torch.from_numpy(w["features"][keep].astype(np.float32))).numpy()
    best = dict.fromkeys(ids, -np.inf)
    face_logits = head.logits(torch.from_numpy(f["features"].astype(np.float32))).numpy()
    for img_id, s, lab in zip(f["img_ids"], face_logits, f["labels"]):
        if str(img_id) in best and lab == label:
            best[str(img_id)] = max(best[str(img_id)], s)
    return whole, np.array([best[i] for i in ids])


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--head", required=True)
    args = parser.parse_args()

    head = LinearHead.load(args.head)
    reals = {name: per_image(head, *spec[:2], 0, spec[2]) for name, spec in REALS.items()}
    fakes = {name: per_image(head, *spec[:2], 1, spec[2]) for name, spec in FAKES.items()}

    def lowest_t_ai(t_face):
        for t in np.sort(np.concatenate([w for w, _ in reals.values()])):
            if all(((w > t) | (fc > t_face)).mean() <= MAX_FALSE_ALARMS for w, fc in reals.values()):
                return t
        return math.inf

    real_faces = np.concatenate([fc[np.isfinite(fc)] for _, fc in reals.values()])
    grid = [math.inf] + list(np.quantile(real_faces, [0.999, 0.995, 0.99, 0.98, 0.97, 0.95, 0.9]))
    print(f"{'t_face':>8} {'t_ai':>7} | AI caught: " + " ".join(f"{n:>15}" for n in fakes) +
          " | real called AI: " + " ".join(f"{n:>15}" for n in reals))
    for t_face in grid:
        t_ai = lowest_t_ai(t_face)
        rule = lambda w, fc: ((w > t_ai) | (fc > t_face)).mean()  # noqa: E731
        fa = [rule(w, fc) for w, fc in reals.values()]
        note = "  over budget" if max(fa) > MAX_FALSE_ALARMS else ""
        print(f"{t_face:>8.2f} {t_ai:>7.2f} |            " + " ".join(f"{rule(w, fc):>15.1%}" for w, fc in fakes.values())
              + " |                 " + " ".join(f"{x:>15.1%}" for x in fa) + note)
    print("(t_face = inf: the whole image alone)")


if __name__ == "__main__":
    main()
