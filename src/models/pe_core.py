"""Meta's Perception Encoder, PE-Core-L/14 (336 px; Apache 2.0; loaded with timm): the Muse-edit check.

Why PE, and why all its layers: a Muse Image edit re-renders every pixel of a real photo (results/muse_edits_v1.txt),
and the frozen features of PE's middle layers separate edits from their own real originals far better than any
last layer (results/backbone_probe_v1.txt, 198 held-out edits: AUC 0.97 for all PE layers, 0.75 for DINOv3's last
layer). The last layer mostly says WHAT is in the picture; the middle layers still carry HOW it was rendered.

Features of a picture: the mean patch token after blocks 4, 8, 12, 16 and 20 and of the last layer (6 x 1,024).
A linear head over them is, per patch, one logit summed over the layers, so an exact "edit map" is possible.

The view: the centre square of the photo at 336 x 336, JPEG 90 (src/data/preprocess.eval_view). Muse re-renders the
whole photo, so the centre carries the trace wherever the asked-for change is.
"""

import numpy as np
import timm
import torch
from huggingface_hub import hf_hub_download
from timm.data import resolve_data_config

from src.data import preprocess

PE_NAME = "vit_pe_core_large_patch14_336.fb"
PE_REPO = "timm/vit_pe_core_large_patch14_336.fb"
LAYERS = (4, 8, 12, 16, 20)  # blocks after which the middle-layer features are taken (PE-Core-L has 24)
NUM_FEATURES = len(LAYERS) + 1  # + the last layer


def load_pe():
    """PE-Core-L from the local Hugging Face cache if it is there (the app then starts offline), else online."""
    try:
        weights = hf_hub_download(PE_REPO, "model.safetensors", local_files_only=True)
        return timm.create_model(PE_NAME, pretrained=True, num_classes=0, pretrained_cfg_overlay={"file": weights})
    except Exception:  # not in the cache yet
        return timm.create_model(PE_NAME, pretrained=True, num_classes=0)


class PECore:
    def __init__(self, device=None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        # bf16 on GPUs that support it, else fp32 (as for DINOv3; fp16 is not used)
        cuda_bf16 = self.device.type == "cuda" and torch.cuda.get_device_capability(self.device)[0] >= 8
        self.dtype = torch.bfloat16 if cuda_bf16 else torch.float32
        self.model = load_pe().to(self.device, self.dtype).eval()
        cfg = resolve_data_config({}, model=self.model)
        self.size = cfg["input_size"][1]  # 336
        self.mean = torch.tensor(cfg["mean"]).view(3, 1, 1)
        self.std = torch.tensor(cfg["std"]).view(3, 1, 1)

    def view(self, image):
        """The centre square at 336 x 336, JPEG 90 (how the edit check is trained)."""
        return preprocess.eval_view(image, self.size)

    def views(self, image):
        """The app's view for the edit check: three squares along the long side (start, centre, end), each at
        336 x 336, JPEG 90, so the whole photo is seen. The linear head's score of the photo is the average of the
        three crops' scores (= the head on the averaged features)."""
        w, h = image.size
        side = min(w, h)
        out = []
        for f in (0.0, 0.5, 1.0):
            left, top = round((w - side) * f), round((h - side) * f)
            out.append(preprocess.jpeg(preprocess.square_crop(image, side, left, top, self.size), 90))
        return out

    def to_tensor(self, view):
        pixels = torch.from_numpy(np.asarray(view, dtype=np.float32) / 255.0).permute(2, 0, 1)
        return (pixels - self.mean) / self.std

    @torch.no_grad()
    def layer_features(self, pixels):
        """pixels: [B, 3, 336, 336] normalised -> [B, 6, 1024]: mean patch token after blocks 4..20 and of the last layer."""
        pixels = pixels.to(self.device, self.dtype)
        # One pass: the patch tokens after blocks 4..20 and after the last block (24); the last one gets PE's final
        # norm, exactly as forward_features does.
        tokens = self.model.forward_intermediates(pixels, indices=[k - 1 for k in LAYERS] + [23], intermediates_only=True,
                                                  output_fmt="NLC")
        tokens[-1] = self.model.norm(tokens[-1])
        out = torch.stack([t.float().mean(1) for t in tokens], dim=1)
        if not torch.isfinite(out).all():
            raise RuntimeError("PE-Core returned NaN or inf values")
        return out.cpu()
