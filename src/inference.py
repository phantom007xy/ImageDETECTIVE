"""Check one photo: whole image (Path A) and every face (Path F), one verdict, an evidence map.

Whole image: DINOv3 patch features -> linear head per patch -> score and evidence map.
Faces: SAM 3 finds them, each face crop (src/faces/crop.py) goes through the same DINOv3 and the
same head (AI-made faces) and, if calibrated, the DFDC face-swap head on the same features (swapped
faces). If the head has a calibration file next to it (<head>_calibration.json, from
src/calibrate/calibrate.py), the verdict uses its thresholds and temperature; otherwise the
provisional thresholds below are used and the verdict is only a preview. Edited regions, SAM 3.1
and Muse come later.

Verdict from the pixels (plan Section 5.6; Path B not built yet):
  0. whole-image score above every real dev photo    -> "AI-generated" (the clear level)
  1. whole-image score above the AI threshold        -> "Likely AI-generated"
  2. otherwise a face above the (very strict) face threshold -> "This face looks AI-made or AI-changed"
  3. otherwise whole-image score at or below the authentic threshold -> "Likely authentic"
  4. everything else                                 -> "Uncertain"
If the file's own metadata says AI (src/services/metadata.py, only when the file bytes are given), that
decides: "made by AI" -> "AI-generated"; "partly changed by AI" -> "Part of this photo looks AI-made
or AI-changed" (unless the pixels already say "Likely AI-generated"). The pixel verdict is always kept as
well, and test numbers are always made from the pixels alone.
"""

import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFilter, ImageFont, ImageOps

from src.data import preprocess
from src.faces import crop
from src.faces.sam_faces import FaceFinder
from src.models.backbone import Backbone
from src.models.linear_head import LinearHead
from src.services import metadata

# Provisional thresholds on the AI probability, only for heads without a calibration file.
AI_ABOVE = 0.65
REAL_BELOW = 0.35


def load_calibration(head_path):
    """Temperature and thresholds (on the logit) of a head; provisional values if it has no calibration."""
    path = Path(head_path).with_name(Path(head_path).stem + "_calibration.json")
    if path.exists():
        c = json.loads(path.read_text())
        return {"temperature": c["temperature"], "ai_logit": c["ai_logit"], "real_logit": c["real_logit"],
                "strong_ai_logit": c.get("strong_ai_logit"), "face_ai_logit": c.get("face_ai_logit"),
                "face_leans_logit": c.get("face_leans_logit"), "face_platt": c.get("face_platt"),
                "swap_head": c.get("swap_head"), "swap_ai_logit": c.get("swap_ai_logit"),
                "swap_leans_logit": c.get("swap_leans_logit"), "calibrated": True}
    return {"temperature": 1.0, "ai_logit": math.log(AI_ABOVE / (1 - AI_ABOVE)),
            "real_logit": math.log(REAL_BELOW / (1 - REAL_BELOW)), "strong_ai_logit": None, "face_ai_logit": None,
            "face_leans_logit": None, "face_platt": None, "swap_head": None, "swap_ai_logit": None,
            "swap_leans_logit": None, "calibrated": False}


class Detector:
    def __init__(self, head_path, device=None, faces=True):
        self.backbone = Backbone(device=device)
        self.head = LinearHead.load(head_path)
        self.calibration = load_calibration(head_path)
        self.calibrated = self.calibration["calibrated"]
        self.finder = FaceFinder(device=device) if faces else None
        # The face-swap head (DFDC) scores the same face features; only used if it is calibrated and present.
        swap_path = self.calibration["swap_head"] and Path(self.calibration["swap_head"].replace("\\", "/"))
        self.swap_head = LinearHead.load(swap_path) if swap_path and swap_path.exists() else None
        if swap_path and self.swap_head is None:
            print(f"WARNING: face-swap head {swap_path} not found: the face-swap check is off")

    def probability(self, logit):
        return 1 / (1 + math.exp(-logit / self.calibration["temperature"]))

    def face_probability(self, logit):
        """Face crops score lower than whole images, so faces get their own fitted scale (Platt scaling)."""
        if self.calibration["face_platt"] is None:
            return None
        w, b = self.calibration["face_platt"]
        return 1 / (1 + math.exp(-(w * logit + b)))

    def verdict(self, score, faces):
        c = self.calibration
        if c["strong_ai_logit"] is not None and score > c["strong_ai_logit"]:  # no real dev photo scores this high
            return "AI-generated"
        if score > c["ai_logit"]:  # strictly above, as in the calibration
            return "Likely AI-generated"
        if any(f.get("flagged") for f in faces):
            return "This face looks AI-made or AI-changed"
        if score <= c["real_logit"]:
            return "Likely authentic"
        return "Uncertain"

    def check_faces(self, image):
        """Every face SAM 3 finds: its box and, if it is big enough, its score from the same head."""
        if self.finder is None:
            return []
        faces = self.finder.find(image)
        for face in faces:
            if face["too_small"]:
                continue
            pixels = preprocess.to_tensor(crop.eval_face_view(image, face)).unsqueeze(0)
            vector = self.backbone.mean_patch_vector(pixels).cpu()
            logit = float(self.head.logits(vector)[0])
            face["logit"] = logit
            face["probability_ai"] = self.face_probability(logit)  # None without a face calibration
            c = self.calibration
            made = c["face_ai_logit"] is not None and logit > c["face_ai_logit"]
            leans = c["face_leans_logit"] is not None and logit > c["face_leans_logit"]
            swapped = swap_leans = False
            if self.swap_head is not None:  # face swaps: the DFDC head on the same face features
                face["swap_logit"] = swap = float(self.swap_head.logits(vector)[0])
                swapped, swap_leans = swap > c["swap_ai_logit"], swap > c["swap_leans_logit"]
            face["flagged"] = made or swapped  # only this changes the verdict
            face["kind"] = "AI-made" if made else ("swapped" if swapped else None)
            # "leans_ai": more AI-like (or swap-like) than 95% of real dev faces; "clear": no clear signs in
            # this face (which is not proof that the face is real: the face checks miss some AI faces).
            face["level"] = "ai" if face["flagged"] else ("leans_ai" if leans or swap_leans else "clear")
        return faces

    def analyze(self, image, file_bytes=None):
        """image: PIL image. file_bytes: the original file (optional), for the metadata check."""
        start = time.perf_counter()
        # Phones often store photos sideways plus an EXIF "rotate" tag: turn the photo upright first.
        image = ImageOps.exif_transpose(image).convert("RGB")
        view = preprocess.inference_view(image)
        pixels = preprocess.to_tensor(view).unsqueeze(0)
        tokens = self.backbone.patch_tokens(pixels)[0].float().cpu()
        if not torch.isfinite(tokens).all():
            raise RuntimeError("DINOv3 returned NaN or inf values")

        grid_w, grid_h = view.size[0] // 16, view.size[1] // 16
        # The head on each patch. Their average is exactly the head on the average patch vector.
        patch_logits = (tokens @ self.head.weight + self.head.bias).reshape(grid_h, grid_w)
        score = patch_logits.mean().item()
        faces = self.check_faces(image)
        pixel_verdict = self.verdict(score, faces)
        meta = metadata.check_file(file_bytes) if file_bytes is not None else None
        verdict, reason = pixel_verdict, "pixels"
        if meta and meta["ai_label"] == "generated":
            verdict, reason = "AI-generated", "metadata"
        elif meta and meta["ai_label"] == "edited" and "AI-generated" not in pixel_verdict:
            verdict, reason = "Part of this photo looks AI-made or AI-changed", "metadata"
        return {
            "score": score,
            "probability_ai": self.probability(score),  # calibrated, if a calibration exists
            "verdict": verdict,
            "verdict_reason": reason,  # "pixels" or "metadata"
            "pixel_verdict": pixel_verdict,  # what the detector alone says
            "metadata": meta,
            "patch_logits": patch_logits.numpy(),
            "faces": faces,
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


def draw_outline(image, mask, colour=(255, 215, 0)):
    """The SAM 3.1 outline: a light fill and a thick edge around the object (mask = bool array, image size)."""
    shape = Image.fromarray(mask.astype(np.uint8) * 255)
    width = max(3, round(min(image.size) / 250))
    edge = np.asarray(shape.filter(ImageFilter.MaxFilter(2 * width + 1))) > 0
    edge &= ~(np.asarray(shape.filter(ImageFilter.MinFilter(2 * width + 1))) > 0)
    layer = np.zeros((image.size[1], image.size[0], 4), dtype=np.uint8)
    layer[mask] = colour + (60,)
    layer[edge] = colour + (255,)
    return Image.alpha_composite(image.convert("RGBA"), Image.fromarray(layer, "RGBA")).convert("RGB")


FACE_COLOURS = {"ai": (220, 30, 30), "leans_ai": (245, 150, 20), "clear": (30, 170, 70), "too_small": (150, 150, 150)}


def draw_faces(image, faces):
    """A numbered box per face: red = looks AI-made (changes the verdict), orange = some AI signs,
    green = no clear AI signs, grey = too small to check.

    A white edge around every box keeps it visible on the red/blue evidence map.
    """
    image = image.copy()
    draw = ImageDraw.Draw(image)
    width = max(2, round(min(image.size) / 300))
    font = ImageFont.load_default(size=max(14, round(min(image.size) / 40)))
    for n, face in enumerate(faces, 1):
        colour = FACE_COLOURS["too_small" if face["too_small"] else face["level"]]
        x0, y0, x1, y1 = face["box"]
        draw.rectangle((x0 - width, y0 - width, x1 + width, y1 + width), outline="white", width=width)
        draw.rectangle((x0, y0, x1, y1), outline=colour, width=width)
        label = draw.textbbox((x0, y0), str(n), font=font)  # number tag just above the box
        pad = width
        tag = (x0, y0 - (label[3] - label[1]) - 3 * pad, x0 + (label[2] - label[0]) + 2 * pad, y0)
        draw.rectangle(tag, fill=colour)
        draw.text((tag[0] + pad, tag[1] + pad), str(n), fill="white", font=font)
    return image
