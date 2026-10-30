"""Check one photo: DINOv3 patch features -> linear head per patch -> score, verdict, evidence map.

Thin version: whole-image path (Path A) only. Faces, edited regions, SAM and Muse come later.
The thresholds are NOT calibrated yet, so the verdict is a preview.
"""

import math
import time

import numpy as np
import torch
from PIL import Image

from src.data import preprocess
from src.models.backbone import Backbone
from src.models.linear_head import LinearHead

# Provisional thresholds on the AI probability, until calibration with real photos (plan Section 5.6).
AI_ABOVE = 0.65
REAL_BELOW = 0.35


def verdict(probability_ai):
    if probability_ai >= AI_ABOVE:
        return "Likely AI-generated"
    if probability_ai <= REAL_BELOW:
        return "Likely authentic"
    return "Uncertain"


class Detector:
    def __init__(self, head_path, device=None):
        self.backbone = Backbone(device=device)
        self.head = LinearHead.load(head_path)

    def analyze(self, image):
        start = time.perf_counter()
        view = preprocess.inference_view(image.convert("RGB"))
        pixels = preprocess.to_tensor(view).unsqueeze(0)
        tokens = self.backbone.patch_tokens(pixels)[0].float().cpu()
        if not torch.isfinite(tokens).all():
            raise RuntimeError("DINOv3 returned NaN or inf values")

        grid_w, grid_h = view.size[0] // 16, view.size[1] // 16
        # The head on each patch. Their average is exactly the head on the average patch vector.
        patch_logits = (tokens @ self.head.weight + self.head.bias).reshape(grid_h, grid_w)
        score = patch_logits.mean().item()
        probability_ai = 1 / (1 + math.exp(-score))
        return {
            "probability_ai": probability_ai,
            "verdict": verdict(probability_ai),
            "patch_logits": patch_logits.numpy(),
            "seconds": time.perf_counter() - start,
        }


def evidence_overlay(image, patch_logits, scale=4.0):
    """Colour the patches over the photo: red = points to AI, blue = points to a real photo.

    tanh squashes the patch scores into -1..1; `scale` sets how strong a score must be to show
    clearly. The same fixed scale is used for every photo, so a calm map really means calm.
    """
    strength = np.tanh(patch_logits / scale)
    heat = np.zeros(strength.shape + (4,), dtype=np.uint8)
    heat[..., 0] = np.where(strength > 0, 255, 0)  # red
    heat[..., 2] = np.where(strength < 0, 255, 0)  # blue
    heat[..., 3] = (np.abs(strength) * np.where(strength > 0, 160, 90)).astype(np.uint8)
    heat_image = Image.fromarray(heat, "RGBA").resize(image.size, Image.BILINEAR)
    return Image.alpha_composite(image.convert("RGBA"), heat_image).convert("RGB")
