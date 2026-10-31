"""Find faces with SAM 3 (local, Meta). The same settings in data preparation and in the app,
so training crops and app crops match.

SAM 3 takes a short text prompt ("face") and returns a box, a mask and a score for every match.
It works on a 1008 x 1008 version of the photo; boxes come back in original-photo pixels.
"""

import torch
from transformers import Sam3Model, Sam3Processor

from src.models.backbone import load_local_first, pick_dtype

MODEL_NAME = "facebook/sam3"
PROMPT = "face"
SCORE_THRESHOLD = 0.5
MIN_FACE = 64  # pixels (shorter box side, in the original photo); smaller faces are "too small to check"


class FaceFinder:
    def __init__(self, device=None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.dtype = pick_dtype(self.device)  # bf16 gave the same faces as fp32 (checked 29 Oct)
        self.processor = load_local_first(Sam3Processor.from_pretrained, MODEL_NAME)
        model = load_local_first(Sam3Model.from_pretrained, MODEL_NAME)
        self.model = model.to(device=self.device, dtype=self.dtype).eval()

    @torch.no_grad()
    def find(self, image):
        """image: PIL RGB -> list of faces, each {"box": [x0, y0, x1, y1], "score", "mask_area", "too_small"}."""
        inputs = self.processor(images=image, text=PROMPT, return_tensors="pt")
        batch = {k: (v.to(self.device, self.dtype) if v.is_floating_point() else v.to(self.device))
                 for k, v in inputs.items()}
        outputs = self.model(**batch)
        result = self.processor.post_process_instance_segmentation(
            outputs, threshold=SCORE_THRESHOLD, mask_threshold=0.5,
            target_sizes=inputs["original_sizes"].tolist())[0]

        faces = []
        for box, score, mask in zip(result["boxes"].float().cpu().tolist(),
                                    result["scores"].float().cpu().tolist(), result["masks"]):
            x0, y0, x1, y1 = box
            faces.append({
                "box": box,
                "score": score,
                "mask_area": float(mask.sum()),
                "too_small": min(x1 - x0, y1 - y0) < MIN_FACE,
            })
        return faces
