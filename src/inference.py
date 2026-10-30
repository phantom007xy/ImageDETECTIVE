"""Check one photo: DINOv3 patch features -> linear head per patch -> score, verdict, evidence map.

Thin version: whole-image path (Path A) only. Faces, edited regions, SAM and Muse come later.
If the head has a calibration file next to it (<head>_calibration.json, from
src/calibrate/calibrate.py), the verdict uses its thresholds and temperature; otherwise the
provisional thresholds below are used and the verdict is only a preview.
"""

import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps

from src.data import preprocess
from src.models.backbone import Backbone
from src.models.linear_head import LinearHead

# Provisional thresholds on the AI probability, only for heads without a calibration file.
AI_ABOVE = 0.65
REAL_BELOW = 0.35


def load_calibration(head_path):
    """(temperature, AI threshold, authentic threshold, calibrated?) with both thresholds on the logit."""
    path = Path(head_path).with_name(Path(head_path).stem + "_calibration.json")
    if path.exists():
        c = json.loads(path.read_text())
        return c["temperature"], c["ai_logit"], c["real_logit"], True
    return 1.0, math.log(AI_ABOVE / (1 - AI_ABOVE)), math.log(REAL_BELOW / (1 - REAL_BELOW)), False


class Detector:
    def __init__(self, head_path, device=None):
        self.backbone = Backbone(device=device)
        self.head = LinearHead.load(head_path)
        self.temperature, self.ai_logit, self.real_logit, self.calibrated = load_calibration(head_path)

    def verdict(self, score):
        """score = the image logit. AI only strictly above its threshold (as in the calibration)."""
        if score > self.ai_logit:
            return "Likely AI-generated"
        if score <= self.real_logit:
            return "Likely authentic"
        return "Uncertain"

    def analyze(self, image):
        start = time.perf_counter()
        # Phones often store photos sideways plus an EXIF "rotate" tag: turn the photo upright first.
        view = preprocess.inference_view(ImageOps.exif_transpose(image).convert("RGB"))
        pixels = preprocess.to_tensor(view).unsqueeze(0)
        tokens = self.backbone.patch_tokens(pixels)[0].float().cpu()
        if not torch.isfinite(tokens).all():
            raise RuntimeError("DINOv3 returned NaN or inf values")

        grid_w, grid_h = view.size[0] // 16, view.size[1] // 16
        # The head on each patch. Their average is exactly the head on the average patch vector.
        patch_logits = (tokens @ self.head.weight + self.head.bias).reshape(grid_h, grid_w)
        score = patch_logits.mean().item()
        probability_ai = 1 / (1 + math.exp(-score / self.temperature))  # calibrated, if a calibration exists
        return {
            "score": score,
            "probability_ai": probability_ai,
            "verdict": self.verdict(score),
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
