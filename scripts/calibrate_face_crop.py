"""Translate the face-crop rule from eye distances into SAM-box units (offline, plan Section 0).

Our crop rule says: side = 4.0 eye distances, centre 0.6 eye distances below the eyes. In the
app only SAM's face box is available, so we measure on many frontal faces how SAM's box relates
to the eye distance. YuNet (non-Meta) is used here offline only, to find the eyes.

For every SAM face we compare three possible size bases with the eye distance: box width, box
height and the square root of the mask area. The base with the smallest spread wins. Everything
is measured separately for real and AI faces: a different ratio would be a shortcut.

Usage:
  python scripts/calibrate_face_crop.py --model weights/face_detection_yunet_2026may.onnx
      --out results/face_crop_calibration.csv D:/Datasets/SID_Set/data/train-00000-of-00249.parquet ...
"""

import argparse
import csv
import math
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root, so `src` can be imported
from src.data.sid_set import LABEL_NAMES, iter_rows  # noqa: E402
from src.faces.sam_faces import FaceFinder  # noqa: E402

CROP_EYE_DISTANCES = 4.0  # our app crop side (plan Section 5.8)
CENTRE_BELOW_EYES = 0.6  # in eye distances


def eyes_in_box(detector, image_bgr, box):
    """Run YuNet around one SAM box; return (right_eye, left_eye, nose) in photo pixels, or None."""
    x0, y0, x1, y1 = box
    w, h = x1 - x0, y1 - y0
    # Look at the box plus 30% on each side, scaled so the face is ~200 px (YuNet works best ~10-300 px).
    rx0, ry0 = max(0, int(x0 - 0.3 * w)), max(0, int(y0 - 0.3 * h))
    rx1, ry1 = min(image_bgr.shape[1], int(x1 + 0.3 * w)), min(image_bgr.shape[0], int(y1 + 0.3 * h))
    region = image_bgr[ry0:ry1, rx0:rx1]
    scale = min(1.0, 200 / max(w, h))
    region = cv2.resize(region, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    pad = int(0.25 * max(region.shape[:2]))
    region = cv2.copyMakeBorder(region, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=(127, 127, 127))
    detector.setInputSize((region.shape[1], region.shape[0]))
    _, found = detector.detect(region)
    if found is None or len(found) == 0:
        return None

    def to_photo(px, py):
        return np.array([(px - pad) / scale + rx0, (py - pad) / scale + ry0])

    # The YuNet face whose nose lies inside the SAM box, most confident first.
    for f in sorted(found, key=lambda f: -f[-1]):
        right_eye, left_eye, nose = to_photo(*f[4:6]), to_photo(*f[6:8]), to_photo(*f[8:10])
        if x0 <= nose[0] <= x1 and y0 <= nose[1] <= y1:
            return right_eye, left_eye, nose
    return None


def spread(values):
    """Relative spread: interquartile range divided by the median (smaller = more stable)."""
    q1, median, q3 = np.percentile(values, [25, 50, 75])
    return (q3 - q1) / median


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("parquet", nargs="+", type=Path)
    parser.add_argument("--model", required=True, help="YuNet .onnx file")
    parser.add_argument("--out", type=Path, required=True, help="CSV with one row per frontal face")
    args = parser.parse_args()

    finder = FaceFinder()
    detector = cv2.FaceDetectorYN.create(str(args.model), "", (320, 320))
    detector.setScoreThreshold(0.6)

    rows, seen, no_eyes, not_frontal = [], 0, 0, 0
    for path in args.parquet:
        for item in iter_rows(path, labels=(0, 1)):
            image_bgr = cv2.cvtColor(np.array(item["image"]), cv2.COLOR_RGB2BGR)
            for face in finder.find(item["image"]):
                if face["too_small"]:
                    continue
                seen += 1
                found = eyes_in_box(detector, image_bgr, face["box"])
                if found is None:
                    no_eyes += 1
                    continue
                right_eye, left_eye, nose = found
                eye_dist = float(np.linalg.norm(right_eye - left_eye))
                eyes_mid = (right_eye + left_eye) / 2
                tilt = math.degrees(math.atan2(abs(left_eye[1] - right_eye[1]), abs(left_eye[0] - right_eye[0])))
                # Frontal: nose roughly centred between the eyes, eyes roughly level.
                if eye_dist < 8 or abs(nose[0] - eyes_mid[0]) > 0.15 * eye_dist or tilt > 15:
                    not_frontal += 1
                    continue
                x0, y0, x1, y1 = face["box"]
                rows.append({
                    "label": LABEL_NAMES[item["label"]],
                    "img_id": item["img_id"],
                    "eye_dist": eye_dist,
                    "width_ratio": (x1 - x0) / eye_dist,
                    "height_ratio": (y1 - y0) / eye_dist,
                    "sqrt_mask_ratio": math.sqrt(face["mask_area"]) / eye_dist,
                    # Where the eyes sit relative to the box centre, in eye distances (+y = below the centre).
                    "eyes_dx": float((eyes_mid[0] - (x0 + x1) / 2) / eye_dist),
                    "eyes_dy": float((eyes_mid[1] - (y0 + y1) / 2) / eye_dist),
                })

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    print(f"SAM faces (>= 64 px): {seen} | no eyes found: {no_eyes} | not frontal: {not_frontal} | frontal: {len(rows)}")
    print(f"wrote {args.out}\n")
    best = None
    for base in ("width_ratio", "height_ratio", "sqrt_mask_ratio"):
        line = f"{base:16s}"
        for label in ("real", "synthetic", None):
            values = [r[base] for r in rows if label is None or r["label"] == label]
            name = label or "all"
            line += f" | {name}: median {np.median(values):.2f}, spread {spread(values):.2f} (n={len(values)})"
        print(line)
        all_spread = spread([r[base] for r in rows])
        if best is None or all_spread < best[1]:
            best = (base, all_spread)

    base = best[0]
    ratio = float(np.median([r[base] for r in rows]))
    eyes_dy = float(np.median([r["eyes_dy"] for r in rows]))
    print(f"\neyes relative to the box centre (eye distances): dx {np.median([r['eyes_dx'] for r in rows]):+.2f}, "
          f"dy {eyes_dy:+.2f} (real {np.median([r['eyes_dy'] for r in rows if r['label'] == 'real']):+.2f}, "
          f"synthetic {np.median([r['eyes_dy'] for r in rows if r['label'] == 'synthetic']):+.2f})")
    print(f"\nmost stable base: {base} (median {ratio:.2f} x eye distance)")
    print(f"crop rule in SAM units: eye distance ~= {base.replace('_ratio', '')} / {ratio:.2f}; "
          f"side = {CROP_EYE_DISTANCES / ratio:.2f} x {base.replace('_ratio', '')}; "
          f"centre = box centre + ({eyes_dy:+.2f} + {CENTRE_BELOW_EYES}) eye distances downwards")


if __name__ == "__main__":
    main()
