"""Add the face-swap Perception Encoder's features of every face to the app's face files, as "features_facepe".

src/eval/face_pe_eval.py --save-dir saves them per crop set (src/faces/make_crops.py, same SAM faces in the same order
as src/features/extract_faces.py). Here each set is matched to its face file (features/faces/<stem>_eval.npz) row by
row and checked (photo of every row, labels), so the calibration (src/calibrate/calibrate.py, calibrate_edit_check.py)
can use a swap head that reads "features_facepe" without any new file layout. Rows a crop set does not have (e.g. the
FLUX faces of the SID-Set validation file: only its real faces were cropped) get zeros and are never used as real faces.

Usage (from the repo root):  python -m src.features.add_face_pe --from features/face_pe_v1
"""

import argparse
from pathlib import Path

import numpy as np

FACES = Path("features/faces")
# crop set -> face file (stem of features/faces/<stem>.npz)
PAIRS = {"pool_commons_dev": "commons_dev_eval", "pool_muse_pairs_val": "muse_pairs_val_eval",
         "pool_sid_reals_0_9_unseen": "sid_reals_0_9_unseen_eval", "pool_commons_v3_dev": "commons_v3_dev_eval",
         "pool_real_face_crops_dev_all": "real_face_crops_dev_all_eval", "pool_commons_v4_dev": "commons_v4_dev_eval",
         "pool_sid_val": "validation-00000-of-00034_eval", "dfdc_p3_dev": "dfdc_part3_eval",
         "of_test_dev": "openforensics_test_dev_eval", "wilddeepfake": "wilddeepfake_eval",
         "pool_muse_image_test_v1": "muse_image_test_v1_eval", "pool_sd_faces": "sd_faces_eval",
         "pool_tpdne_test": "tpdne_test_eval"}


def norm(values):
    return [str(v).replace("\\", "/").lower() for v in values]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--from", dest="source", type=Path, required=True)
    parser.add_argument("--key", default="features_facepe")
    args = parser.parse_args()

    for crop_set, stem in PAIRS.items():
        c, path = np.load(args.source / f"{crop_set}.npz"), FACES / f"{stem}.npz"
        f = dict(np.load(path))
        x = c["features"].reshape(len(c["features"]), -1)
        # the photo (or video) of every face, in order: face files keep it in "groups" (DFDC, OpenForensics) or img_ids
        face_photo = norm(f["groups"] if "groups" in f else f["img_ids"])
        crop_photo = norm(c["img_ids"] if crop_set == "wilddeepfake" else c["groups"])
        rows = np.arange(len(face_photo)) if len(crop_photo) == len(face_photo) else np.flatnonzero(f["labels"] == 0)
        if len(rows) != len(crop_photo) or [face_photo[i] for i in rows] != crop_photo \
                or not np.array_equal(f["labels"][rows].astype(int), c["labels"].astype(int)):
            raise SystemExit(f"{crop_set} does not line up with {path}: not changed")
        out = np.zeros((len(face_photo), x.shape[1]), dtype=np.float16)
        out[rows] = x
        f[args.key] = out
        np.savez_compressed(path, **f)
        print(f"{path.name}: {len(rows)} of {len(face_photo)} faces <- {crop_set}")


if __name__ == "__main__":
    main()
