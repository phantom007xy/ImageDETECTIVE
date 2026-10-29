"""Our own preprocessing. Never DINOv3's default, which squashes every image to a square.

Every image, real or fake, goes through exactly the same steps, so file format, size or shape
can't tell the classes apart (in SID-Set, fakes are square PNGs and real photos are JPEGs):
  1. sometimes an earlier JPEG save (evens out "fakes are clean PNGs, reals are JPEGs")
  2. random downscale
  3. random square crop (cropping, never squashing)
  4. resize to the model size; this resampling also wipes out the 8x8 grid of earlier JPEG saves
  5. small random brightness and contrast change
  6. a final JPEG save at a random quality (like WhatsApp or a re-upload)

"eval" mode is one fixed version of these steps, so test numbers are repeatable.
"""

import io

import numpy as np
import torch
from PIL import Image, ImageEnhance

MODEL_SIZE = 448  # a multiple of 16 -> 28 x 28 = 784 patches
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)  # ImageNet values that DINOv3 expects
STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def jpeg(image, quality):
    """Save to JPEG in memory and load it back."""
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    return Image.open(buffer).convert("RGB")


def square_crop(image, side, left, top, size):
    return image.crop((left, top, left + side, top + side)).resize((size, size), Image.BICUBIC)


def train_view(image, rng, size=MODEL_SIZE):
    """One random training copy of a PIL RGB image. rng is a random.Random."""
    if rng.random() < 0.5:
        image = jpeg(image, rng.randint(70, 95))

    w, h = image.size
    smallest_scale = min(1.0, max(0.5, size / min(w, h)))  # don't shrink below the model size
    scale = rng.uniform(smallest_scale, 1.0)
    if scale < 1.0:
        image = image.resize((round(w * scale), round(h * scale)), Image.BICUBIC)

    w, h = image.size
    side = round(min(w, h) * rng.uniform(0.6, 1.0))
    image = square_crop(image, side, rng.randint(0, w - side), rng.randint(0, h - side), size)

    image = ImageEnhance.Brightness(image).enhance(rng.uniform(0.85, 1.15))
    image = ImageEnhance.Contrast(image).enhance(rng.uniform(0.85, 1.15))
    return jpeg(image, rng.randint(65, 95))


def eval_view(image, size=MODEL_SIZE):
    """The fixed version for test numbers: centre square crop, resize, JPEG quality 90."""
    w, h = image.size
    side = min(w, h)
    return jpeg(square_crop(image, side, (w - side) // 2, (h - side) // 2, size), 90)


def to_tensor(image):
    """PIL RGB -> normalised [3, H, W] float tensor."""
    pixels = torch.from_numpy(np.asarray(image, dtype=np.float32) / 255.0).permute(2, 0, 1)
    return (pixels - MEAN) / STD
