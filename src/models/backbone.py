"""DINOv3 backbone: turns a batch of images into patch features.

DINOv3 returns one token per 16x16 patch, plus 1 class token and 4 "register" tokens at the
start. We only use the patch tokens.
"""

import torch
from transformers import AutoModel

MODEL_NAME = "facebook/dinov3-vitl16-pretrain-lvd1689m"


def load_local_first(loader, name, **kwargs):
    """Load from the local Hugging Face cache; go online only if something is missing there.

    Online, gated models (DINOv3, SAM 3) also check for optional extra files, which needs a token;
    from the cache the app starts without internet and without a token.
    """
    try:
        return loader(name, local_files_only=True, **kwargs)
    except OSError:
        return loader(name, **kwargs)


def pick_dtype(device):
    """bf16 on GPUs that support it natively (RTX 30xx and newer, A100, L4), otherwise fp32.

    Not fp16: DINOv3 ViT-L overflows inside the network in fp16 and every output becomes NaN
    (checked 29 Oct). bf16 has the same range as fp32 and matched it (cosine 0.9999).
    """
    if device.type == "cuda" and torch.cuda.get_device_capability(device)[0] >= 8:
        return torch.bfloat16
    return torch.float32


class Backbone:
    def __init__(self, name=MODEL_NAME, device=None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.dtype = pick_dtype(self.device)
        self.model = load_local_first(AutoModel.from_pretrained, name).to(device=self.device, dtype=self.dtype).eval()
        self.skip = 1 + self.model.config.num_register_tokens  # class token + register tokens

    @torch.no_grad()
    def patch_tokens(self, pixels):
        """pixels: [B, 3, H, W] normalised images -> [B, num_patches, 1024] patch features."""
        tokens = self.model(pixel_values=pixels.to(self.device, self.dtype)).last_hidden_state
        return tokens[:, self.skip:]

    def mean_patch_vector(self, pixels):
        """The average of the patch features: one vector of 1,024 numbers per image.

        A linear head on this average gives exactly the same score as the same head applied to
        every patch and averaged afterwards. That keeps the evidence map honest, and training
        only needs ~2 KB per image instead of all patch features.
        """
        vectors = self.patch_tokens(pixels).float().mean(dim=1)
        if not torch.isfinite(vectors).all():
            raise RuntimeError("DINOv3 returned NaN or inf values; check the precision (dtype)")
        return vectors
