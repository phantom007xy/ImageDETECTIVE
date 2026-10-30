"""Square face crops around SAM faces, with the rule calibrated in scripts/calibrate_face_crop.py.

Our rule is defined in eye distances: side = k eye distances, centre 0.6 eye distances below the
eyes. SAM gives no eye points, so we use the calibration (30 Oct, 403 frontal faces,
results/face_crop_calibration.txt):
  - eye distance ~= sqrt(mask area) / 2.30 (the most stable of box width, box height, sqrt(mask))
  - the eyes sit 0.37 eye distances above the centre of SAM's box
The app always uses k = 4.0. Training uses a random k from 2.6 (the tight end of WildDeepfake's
crops) to 4.0 plus a small shift, identically for real and AI faces.
"""

import math

from PIL import Image, ImageEnhance

from src.data.preprocess import jpeg

SQRT_MASK_PER_EYE_DISTANCE = 2.30
BOX_WIDTH_PER_EYE_DISTANCE = 2.58  # fallback if a mask is empty
EYES_ABOVE_BOX_CENTRE = 0.37
CENTRE_BELOW_EYES = 0.6
APP_K = 4.0
TRAIN_K_RANGE = (2.6, 4.0)
FACE_SIZE = 224  # DINOv3 input for a face: 14 x 14 = 196 patches


def eye_distance(face):
    if face["mask_area"] > 0:
        return math.sqrt(face["mask_area"]) / SQRT_MASK_PER_EYE_DISTANCE
    x0, _, x1, _ = face["box"]
    return (x1 - x0) / BOX_WIDTH_PER_EYE_DISTANCE


def crop_box(face, k=APP_K, shift=(0.0, 0.0)):
    """Square crop (x0, y0, x1, y1) in photo pixels. shift moves the centre, in eye distances."""
    x0, y0, x1, y1 = face["box"]
    eye = eye_distance(face)
    cx = (x0 + x1) / 2 + shift[0] * eye
    cy = (y0 + y1) / 2 + (CENTRE_BELOW_EYES - EYES_ABOVE_BOX_CENTRE + shift[1]) * eye
    half = k * eye / 2
    return (cx - half, cy - half, cx + half, cy + half)


def face_crop(image, face, k=APP_K, shift=(0.0, 0.0), size=FACE_SIZE):
    """The crop, resized to size x size. Parts outside the photo become black (same for every class)."""
    box = tuple(round(v) for v in crop_box(face, k, shift))
    return image.crop(box).resize((size, size), Image.BICUBIC)


def eval_face_view(image, face):
    """The fixed version for the app and for test numbers: k = 4.0, JPEG quality 90."""
    return jpeg(face_crop(image, face), 90)


def ready_crop_view(image):
    """For images that already are face crops (e.g. WildDeepfake): the whole image, 224 x 224, JPEG 90."""
    return jpeg(image.resize((FACE_SIZE, FACE_SIZE), Image.BICUBIC), 90)


def train_face_view(image, face, rng):
    """One random training copy of a face. rng is a random.Random."""
    k = rng.uniform(*TRAIN_K_RANGE)
    crop = face_crop(image, face, k=k, shift=(rng.uniform(-0.1, 0.1), rng.uniform(-0.1, 0.1)))
    crop = ImageEnhance.Brightness(crop).enhance(rng.uniform(0.6, 1.5))  # wide: see preprocess.train_view
    crop = ImageEnhance.Contrast(crop).enhance(rng.uniform(0.85, 1.15))
    if rng.random() < 0.3:  # like a low-resolution face: shrink and enlarge again
        small = rng.randint(64, 160)
        crop = crop.resize((small, small), Image.BICUBIC).resize((FACE_SIZE, FACE_SIZE), Image.BICUBIC)
    return jpeg(crop, rng.randint(65, 95))
