"""Measure ready-made face crops (e.g. WildDeepfake): framing, sharpness, brightness.

For every frame we detect the face with YuNet and record:
  - k: crop size divided by the eye distance (how many "eye distances" the crop spans).
    Eye distance is a detector-independent scale, so k describes the margin fairly.
  - where the crop centre sits relative to the point between the eyes
  - sharpness (variance of the Laplacian) and brightness (mean grey value)

Results are summarised per sequence (frames in a sequence are near-identical)
and then per class (real / fake).

Usage:
  python -I scripts/measure_face_crops.py --model weights/face_detection_yunet_2026may.onnx DIR [DIR ...]
"""

import argparse
import statistics
from pathlib import Path

import cv2
import numpy as np

# Grey border added around each crop before detection. Detectors often miss a face
# that fills the whole image; a plain colour (not a mirrored edge) can't create a fake second face.
PAD_FRACTION = 0.25
PAD_COLOUR = (127, 127, 127)


def measure_frame(detector, image):
    """Return a dict of measurements for one crop, or None if no face was found."""
    h, w = image.shape[:2]
    pad = int(round(PAD_FRACTION * max(h, w)))
    padded = cv2.copyMakeBorder(image, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=PAD_COLOUR)

    detector.setInputSize((padded.shape[1], padded.shape[0]))
    _, faces = detector.detect(padded)

    grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    result = {
        "sharpness": float(cv2.Laplacian(grey, cv2.CV_64F).var()),
        "brightness": float(grey.mean()),
    }
    if faces is None or len(faces) == 0:
        return result | {"found": False}

    # Keep the most confident face. Each row: x, y, w, h, right eye (x, y), left eye (x, y),
    # nose, right mouth corner, left mouth corner, score.
    face = max(faces, key=lambda f: f[-1])
    right_eye = np.array(face[4:6]) - pad
    left_eye = np.array(face[6:8]) - pad
    eye_dist = float(np.linalg.norm(right_eye - left_eye))
    eyes_mid = (right_eye + left_eye) / 2
    crop_centre = np.array([w / 2, h / 2])

    return result | {
        "found": True,
        "k": max(w, h) / eye_dist,
        "box_w": face[2] / w,
        "box_h": face[3] / h,
        # Positive offset_y = the crop centre lies below the eyes (in eye distances).
        "offset_x": float((crop_centre[0] - eyes_mid[0]) / eye_dist),
        "offset_y": float((crop_centre[1] - eyes_mid[1]) / eye_dist),
    }


def median_of(rows, key):
    values = [r[key] for r in rows if key in r]
    return statistics.median(values) if values else float("nan")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, help="path to the YuNet .onnx file")
    parser.add_argument("--frames-per-seq", type=int, default=0,
                        help="measure only this many evenly spaced frames per sequence (0 = all)")
    parser.add_argument("dirs", nargs="+", help="folders with extracted crops")
    args = parser.parse_args()

    detector = cv2.FaceDetectorYN.create(args.model, "", (320, 320))
    detector.setScoreThreshold(0.6)

    per_class = {}
    for root in map(Path, args.dirs):
        for seq in sorted(p for p in root.rglob("*") if p.is_dir() and any(p.glob("*.png"))):
            label = "fake" if "fake" in seq.parts else "real"
            frames = sorted(seq.glob("*.png"), key=lambda f: int(f.stem) if f.stem.isdigit() else f.stem)
            if args.frames_per_seq and len(frames) > args.frames_per_seq:
                # Neighbouring frames are near-identical, so a few evenly spaced ones are enough.
                picks = np.linspace(0, len(frames) - 1, args.frames_per_seq).round().astype(int)
                frames = [frames[i] for i in picks]
            rows = [measure_frame(detector, cv2.imread(str(f))) for f in frames]
            found = [r for r in rows if r["found"]]
            summary = {
                "sequence": str(seq.relative_to(root)),
                "frames": len(rows),
                "found_rate": len(found) / len(rows),
                "k": median_of(found, "k"),
                "offset_y": median_of(found, "offset_y"),
                "sharpness": median_of(rows, "sharpness"),
                "brightness": median_of(rows, "brightness"),
            }
            per_class.setdefault(label, []).append(summary)
            print(f"[{label}] {summary['sequence']}: {summary['frames']} frames, face found in "
                  f"{summary['found_rate']:.0%} | k = {summary['k']:.2f} eye distances | "
                  f"centre {summary['offset_y']:+.2f} below eyes | sharpness {summary['sharpness']:.0f} | "
                  f"brightness {summary['brightness']:.0f}")

    print("\n=== per class (median over sequences) ===")
    for label, seqs in sorted(per_class.items()):
        print(f"{label}: {len(seqs)} sequences | k = {median_of(seqs, 'k'):.2f} | "
              f"centre {median_of(seqs, 'offset_y'):+.2f} below eyes | "
              f"sharpness {median_of(seqs, 'sharpness'):.0f} | brightness {median_of(seqs, 'brightness'):.0f}")


if __name__ == "__main__":
    main()
