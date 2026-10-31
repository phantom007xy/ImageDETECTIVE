"""Which frozen backbone sees Muse Image edits best? DINOv3 vs. Meta's Perception Encoder (PE-Core) vs. both.

Same pictures, same view for every model (the centre square crop at the model's own input size, JPEG 90:
src/data/preprocess.eval_view), frozen weights. A linear probe (logistic regression, src/models/linear_head.py)
is trained on the Muse TRAINING edits (edit = 1, its real original = 0) and tested on the held-out Muse TEST edits:
AUC on the test edits vs. their originals, for local and whole-image edits. Only Meta models are candidates for the
app (rule since 28 Oct): DINOv3 and PE-Core-L/14 (Apache 2.0, loaded with timm).

Features per backbone (all frozen):
  dinov3       mean of the last layer's patch tokens (what our heads use), 448 x 448
  dinov3_L<k>  mean of the patch tokens after block k (4, 8, ..., 20): middle layers keep more low-level traces
               (RINE, ECCV 2024: intermediate CLIP blocks detect generated images far better than the output)
  pe_patch     PE-Core-L/14: mean of the last layer's patch tokens, 336 x 336
  pe_L<k>      PE-Core-L/14: mean of the patch tokens after block k ("the best visual embeddings are not at the
               output of the network", the PE paper's title)
  pe_pooled    PE-Core-L/14: its pooled image embedding (the CLIP-style output that text is matched to)
A linear head over a middle layer's patch tokens still gives an exact evidence map (one logit per patch).

Usage (from the repo root):
  python -m src.eval.backbone_probe --train features/lists/muse_edits_train_commons_path_a.csv
      features/lists/muse_edits_train_openimages_path_a.csv --test features/lists/muse_edits_v1.csv
"""

import argparse
import csv
from pathlib import Path

import numpy as np
import timm
import torch
from PIL import Image, ImageOps
from timm.data import resolve_data_config

from src.data import preprocess
from src.eval import metrics
from src.models.backbone import Backbone
from src.models.linear_head import LinearHead

PE_NAME = "vit_pe_core_large_patch14_336.fb"
OUT_DIR = Path("features/probe")
LAYERS = (4, 8, 12, 16, 20)  # blocks after which the middle-layer features are taken (both models have 24)


class PECore:
    """PE-Core-L/14 from timm, frozen, bf16 on a GPU (fp32 on CPU)."""

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.bfloat16 if self.device.type == "cuda" else torch.float32
        self.model = timm.create_model(PE_NAME, pretrained=True, num_classes=0).to(self.device, self.dtype).eval()
        cfg = resolve_data_config({}, model=self.model)
        self.size = cfg["input_size"][1]
        self.mean = torch.tensor(cfg["mean"]).view(3, 1, 1)
        self.std = torch.tensor(cfg["std"]).view(3, 1, 1)

    @torch.no_grad()
    def features(self, image):
        view = preprocess.eval_view(image, self.size)
        pixels = torch.from_numpy(np.asarray(view, dtype=np.float32) / 255.0).permute(2, 0, 1)
        pixels = ((pixels - self.mean) / self.std).unsqueeze(0).to(self.device, self.dtype)
        tokens = self.model.forward_features(pixels)
        out = {"pe_patch": tokens[:, self.model.num_prefix_tokens:].float().mean(1),
               "pe_pooled": self.model.forward_head(tokens, pre_logits=True).float()}
        middle = self.model.forward_intermediates(pixels, indices=[k - 1 for k in LAYERS], intermediates_only=True,
                                                  output_fmt="NLC")  # patch tokens only (no class token)
        out.update({f"pe_L{k}": m.float().mean(1) for k, m in zip(LAYERS, middle)})
        if not all(torch.isfinite(v).all() for v in out.values()):
            raise RuntimeError("PE-Core returned NaN or inf values")
        return {k: v[0].cpu().numpy() for k, v in out.items()}


@torch.no_grad()
def dinov3_features(backbone, image):
    """DINOv3: mean patch tokens of the last layer (after the final norm) and after the middle blocks."""
    pixels = preprocess.to_tensor(preprocess.eval_view(image)).unsqueeze(0).to(backbone.device, backbone.dtype)
    result = backbone.model(pixel_values=pixels, output_hidden_states=True)
    out = {"dinov3": result.last_hidden_state[:, backbone.skip:].float().mean(1)}
    out.update({f"dinov3_L{k}": result.hidden_states[k][:, backbone.skip:].float().mean(1) for k in LAYERS})
    if not all(torch.isfinite(v).all() for v in out.values()):
        raise RuntimeError("DINOv3 returned NaN or inf values")
    return {k: v[0].cpu().numpy() for k, v in out.items()}


def read(paths):
    rows = []
    for path in paths:
        with open(path, newline="") as f:
            rows += [r for r in csv.DictReader(f)]
    return rows


def extract(rows, name):
    """Features of every picture for every backbone, cached in features/probe/<name>.npz."""
    out = OUT_DIR / f"{name}.npz"
    if out.exists():
        return dict(np.load(out))
    dino, pe = Backbone(), PECore()
    feats = {}
    for i, row in enumerate(rows):
        with Image.open(row["path"]) as im:
            image = ImageOps.exif_transpose(im).convert("RGB")
        for k, v in {**dinov3_features(dino, image), **pe.features(image)}.items():
            feats.setdefault(k, []).append(v)
        if i % 500 == 0:
            print(f"  {name}: {i}/{len(rows)}", flush=True)
    data = {k: np.stack(v).astype(np.float32) for k, v in feats.items()}
    data["labels"] = np.array([1 if int(r["label"]) else 0 for r in rows], dtype=np.int8)
    data["sources"] = np.array([r["source"] for r in rows])
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **data)
    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--train", nargs="+", type=Path, required=True, help="Path A lists of the training edits")
    parser.add_argument("--test", type=Path, required=True, help="list of the test edits (labels 0 / 1 / 2)")
    args = parser.parse_args()

    train, test = extract(read(args.train), "train_edits"), extract(read([args.test]), args.test.stem)
    is_local = test["sources"] == "muse_edits/local"
    is_whole = test["sources"] == "muse_edits/whole"
    is_orig = test["labels"] == 0
    combos = {"dinov3": ["dinov3"], **{f"dinov3_L{k}": [f"dinov3_L{k}"] for k in LAYERS},
              "dinov3 all layers": ["dinov3"] + [f"dinov3_L{k}" for k in LAYERS],
              "pe_patch": ["pe_patch"], "pe_pooled": ["pe_pooled"], **{f"pe_L{k}": [f"pe_L{k}"] for k in LAYERS},
              "pe all layers": ["pe_patch"] + [f"pe_L{k}" for k in LAYERS],
              "dinov3 + pe_patch": ["dinov3", "pe_patch"],
              "both, all layers": ["dinov3", "pe_patch"] + [f"dinov3_L{k}" for k in LAYERS] + [f"pe_L{k}" for k in LAYERS]}
    print(f"\ntrain: {len(train['labels'])} pictures, test: {len(test['labels'])} "
          f"({is_orig.sum()} originals, {is_local.sum()} local, {is_whole.sum()} whole-image edits)")
    print(f"{'features':<20}{'wd':>6} | {'train AUC':>9} | test AUC: {'all':>5} {'local':>6} {'whole':>6} | caught at 5% FA")
    for name, keys in combos.items():
        x_tr = torch.from_numpy(np.concatenate([train[k] for k in keys], axis=1))
        x_te = torch.from_numpy(np.concatenate([test[k] for k in keys], axis=1))
        y_tr = torch.from_numpy(train["labels"].astype(np.float32))
        for wd in (1e-3, 1e-2, 1e-1):
            head = LinearHead().fit(x_tr, y_tr, weight_decay=wd)
            s_tr, s = head.logits(x_tr).numpy(), head.logits(x_te).numpy()
            auc = lambda keep: metrics.auc(s[keep | is_orig], ~is_orig[keep | is_orig])  # noqa: E731
            print(f"{name:<20}{wd:>6g} | {metrics.auc(s_tr, train['labels'] == 1):>9.3f} | "
                  f"{auc(~is_orig):>14.3f} {auc(is_local):>6.3f} {auc(is_whole):>6.3f} | "
                  f"{metrics.caught_at_false_alarm_rate(s, ~is_orig):.1%}")

    # Anomaly-style check (the idea behind PaDiM / FastFlow in Anomalib): model ONLY the real training photos with
    # one Gaussian and score every test picture by how unusual it is (Mahalanobis distance). No fakes are used.
    print("\nanomaly-style (fit on the real training photos only; score = Mahalanobis distance):")
    for name in ["dinov3"] + [f"dinov3_L{k}" for k in LAYERS] + ["pe_patch", "pe_pooled"] + [f"pe_L{k}" for k in LAYERS]:
        real = train[name][train["labels"] == 0].astype(np.float64)
        mean = real.mean(0)
        cov = np.cov(real, rowvar=False)
        cov += 0.01 * np.trace(cov) / len(cov) * np.eye(len(cov))  # shrinkage: far fewer photos than numbers
        inv = np.linalg.inv(cov)
        d = test[name].astype(np.float64) - mean
        s = np.einsum("ij,jk,ik->i", d, inv, d)
        auc = lambda keep: metrics.auc(s[keep | is_orig], ~is_orig[keep | is_orig])  # noqa: E731
        print(f"  {name:<18} test AUC: all {auc(~is_orig):.3f}, local {auc(is_local):.3f}, whole {auc(is_whole):.3f}")


if __name__ == "__main__":
    main()
