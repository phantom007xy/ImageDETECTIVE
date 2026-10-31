"""Full fine-tuning of Meta's Perception Encoder (PE-Core-L/14-336) for the Muse-edit check.

Why: the frozen PE separates Muse edits from their own originals well (probe AUC 0.97, results/backbone_probe_v1.txt),
and the edit signal sits in its MIDDLE layers, so tuning only the last blocks would barely touch it: all layers are
tuned. What limits the frozen check is a handful of real photos (e.g. heavily retouched portraits) at the strict line
(results/edit_check_v1.txt); fine-tuning may learn to separate them from edits, or overfit, so it is measured.

How (standard practice for fine-tuning a ViT, chosen to protect what PE knows about real photos):
  - start: pretrained PE + the frozen edit check's head (step 0 = today's edit check, weights/edit_check_v1)
  - features as in the app: mean patch token after blocks 4, 8, 12, 16, 20 and of the last layer -> linear head
  - layer-wise learning-rate decay: --lr for the last block, x --decay per block downwards (the patch embedding
    almost frozen); head --lr-head
  - a decoupled pull towards the pretrained weights after every step (L2-SP), stochastic depth 0.1, gradient
    clipping 1.0, warmup then cosine schedule
  - bf16 compute, fp32 weights, gradient checkpointing (fits an 8 GB GPU at batch 16)
  - data: label 1 = Muse renderings (training edits + Muse pair fakes), label 0 = real photos; each class counts half;
    the random training view at 336 (src/data/preprocess.train_view)
Validation every --eval-every steps on dev sets only, in the app's view (centre square 336): the edit line chosen
JOINTLY with head F (exactly as src/calibrate/calibrate_edit_check.py: together at most 5% of every real pool, the edit
check alone at most 0.5%), then the share caught of every AI set. Every validation saves a checkpoint.

Usage (from the repo root):
  python -m src.models.tune_pe --train features/lists/muse_edits_train_*_path_a.csv ... --out-dir weights/pe_ft_v1
"""

import argparse
import csv
import glob
import io
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps
from timm.data import resolve_data_config

from src.data import preprocess
from src.eval import metrics
from src.eval.fair_compare import threshold_for
from src.models.linear_head import LinearHead
from src.models.pe_core import LAYERS, load_pe

# Dev sets: name -> (list stem, label); every dev list has an aligned head-F app feature file features/<stem>_app.npz.
REAL_POOLS = {"polished dev": ("commons_dev", 0), "OpenImages val": ("muse_pairs_val", 0),
              "originals v1": ("muse_edits_v1", 0), "originals v2": ("muse_edits_test_v2", 0)}
AI_SETS = {"edits v1 local": ("muse_edits_v1", 2), "edits v1 whole": ("muse_edits_v1", 1),
           "edits v2 local": ("muse_edits_test_v2", 2), "edits v2 whole": ("muse_edits_test_v2", 1),
           "Muse test v1": ("muse_image_test_v1", 1), "Muse pairs val": ("muse_pairs_val", 1)}
LIST_DIR = {"muse_pairs_val": "features/lists", "commons_dev": "features/lists", "muse_edits_v1": "features/lists",
            "muse_edits_test_v2": "features/lists", "muse_image_test_v1": "features/lists"}


def open_upright(path):
    with Image.open(path) as im:
        return ImageOps.exif_transpose(im).convert("RGB")


class TrainImages(torch.utils.data.Dataset):
    def __init__(self, rows, size, mean, std):
        self.rows, self.size, self.mean, self.std = rows, size, mean, std

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        row = self.rows[i]
        view = preprocess.train_view(open_upright(row["path"]), random, size=self.size)
        pixels = torch.from_numpy(np.asarray(view, dtype=np.float32) / 255.0).permute(2, 0, 1)
        return ((pixels - self.mean) / self.std, torch.tensor(1.0 if int(row["label"]) else 0.0),
                torch.tensor(row["source"].startswith("muse_edits")))


def layer_features(model, pixels):
    """[B, 6, 1024]: mean patch token after blocks 4..20 and of the normed last layer (as src/models/pe_core.py)."""
    tokens = model.forward_intermediates(pixels, indices=[k - 1 for k in LAYERS] + [23], intermediates_only=True,
                                         output_fmt="NLC")
    tokens[-1] = model.norm(tokens[-1])
    return torch.stack([t.float().mean(1) for t in tokens], dim=1)


def dev_data(size):
    """Every dev picture as its app view (PNG bytes), plus head F's logit of the same picture (aligned by path)."""
    head_f = LinearHead.load("weights/open_gen_v1/head_F.pt")
    out = {}
    for name, (stem, label) in {**REAL_POOLS, **AI_SETS}.items():
        with open(f"{LIST_DIR[stem]}/{stem}.csv", newline="") as f:
            rows = [r for r in csv.DictReader(f)]
        app = np.load(f"features/{stem}_app.npz")
        if list(app["img_ids"]) != [r["path"] for r in rows]:
            raise ValueError(f"features/{stem}_app.npz is not aligned with its list")
        keep = [i for i, r in enumerate(rows) if int(r["label"]) == label]
        f_logits = head_f.logits(torch.from_numpy(app["features"][keep].astype(np.float32))).numpy()
        views = []
        for i in keep:
            buffer = io.BytesIO()
            preprocess.eval_view(open_upright(rows[i]["path"]), size).save(buffer, format="PNG")
            views.append(buffer.getvalue())
        out[name] = (views, f_logits)
    return out


@torch.no_grad()
def validate(model, w, b, dev, mean, std, f_line):
    model.eval()
    scores = {}
    for name, (views, _) in dev.items():
        s = []
        for start in range(0, len(views), 32):
            batch = torch.stack([(torch.from_numpy(np.asarray(Image.open(io.BytesIO(v)).convert("RGB"), dtype=np.float32)
                                                   / 255.0).permute(2, 0, 1) - mean) / std for v in views[start:start + 32]])
            with torch.autocast("cuda", dtype=torch.bfloat16):
                feats = layer_features(model, batch.cuda())
            s.append((feats.flatten(1) @ w + b).float().cpu().numpy())
        scores[name] = np.concatenate(s)
    model.train()
    # the joint line with head F (as src/calibrate/calibrate_edit_check.py)
    line = max(threshold_for(scores[n], 0.005) for n in REAL_POOLS)
    for n in REAL_POOLS:
        f_flag = dev[n][1] > f_line
        room = int(np.floor(0.05 * len(f_flag))) - int(f_flag.sum())
        rest = np.sort(scores[n][~f_flag])[::-1]
        if room < len(rest):
            line = max(line, float(rest[max(room, 0)]))
    together = {n: float(((dev[n][1] > f_line) | (scores[n] > line)).mean()) for n in scores}
    edits = np.r_[scores["edits v2 local"], scores["edits v2 whole"]]
    auc = metrics.auc(np.r_[scores["originals v2"], edits], np.r_[np.zeros(len(scores["originals v2"])), np.ones(len(edits))] == 1)
    return line, together, auc


def report(step, line, together, auc):
    reals = " | ".join(f"{n} {together[n]:.1%}" for n in REAL_POOLS)
    fakes = " | ".join(f"{n} {together[n]:.0%}" for n in AI_SETS)
    print(f"  step {step}: edit line {line:+.2f} | F + edit check, real called AI: {reals} || caught: {fakes} "
          f"|| AUC originals v2 vs edits v2 {auc:.3f}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--train", nargs="+", required=True, help="CSV image lists (globs allowed)")
    parser.add_argument("--init-head", type=Path, default=Path("weights/edit_check_v1/head_EC_both.pt"))
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=4000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5, help="for the last block (layer-wise decay below it)")
    parser.add_argument("--decay", type=float, default=0.85, help="layer-wise learning-rate decay per block")
    parser.add_argument("--lr-head", type=float, default=1e-4)
    parser.add_argument("--pull", type=float, default=1.0, help="decoupled pull towards the pretrained weights")
    parser.add_argument("--eval-every", type=int, default=500)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    files = sorted({f for pattern in args.train for f in glob.glob(pattern)})
    rows = [r for path in files for r in csv.DictReader(open(path, newline=""))]
    labels = np.array([1 if int(r["label"]) else 0 for r in rows])
    print(f"training pictures: {len(rows)} ({int((labels == 0).sum())} real, {int(labels.sum())} Muse) from {len(files)} lists",
          flush=True)
    class_weight = torch.tensor([len(rows) / (2 * (labels == 0).sum()), len(rows) / (2 * labels.sum())],
                                dtype=torch.float32).cuda()

    model = load_pe(drop_path_rate=0.1).cuda()  # fp32 weights; bf16 compute under autocast
    model.set_grad_checkpointing(True)
    model.train()
    cfg = resolve_data_config({}, model=model)
    size, mean, std = cfg["input_size"][1], torch.tensor(cfg["mean"]).view(3, 1, 1), torch.tensor(cfg["std"]).view(3, 1, 1)
    head = LinearHead.load(args.init_head)
    w = head.weight.float().cuda().requires_grad_()
    b = head.bias.float().reshape(()).cuda().requires_grad_()

    # Layer-wise learning-rate decay: depth 0 = patch embedding etc., 1..24 = blocks, 25 = the final norm.
    groups = {}  # depth -> [(weight, its pretrained value in bf16, for the pull)]
    for name, p in model.named_parameters():
        if name.startswith("attn_pool"):  # PE's pooling head is not used by the edit check
            p.requires_grad = False
            continue
        depth = int(name.split(".")[1]) + 1 if name.startswith("blocks.") else (25 if name.startswith("norm.") else 0)
        groups.setdefault(depth, []).append((p, p.detach().to(torch.bfloat16).clone()))
    depths = sorted(groups)
    param_groups = [{"params": [p for p, _ in groups[d]], "lr": args.lr * args.decay ** (25 - d),
                     "base_lr": args.lr * args.decay ** (25 - d)} for d in depths]
    param_groups.append({"params": [w, b], "lr": args.lr_head, "base_lr": args.lr_head})
    optimizer = torch.optim.AdamW(param_groups, weight_decay=0.0)
    warmup = 300

    def factor(s):
        return min(1.0, (s + 1) / warmup) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / args.steps)))

    print(f"tuning all {sum(p.numel() for pairs in groups.values() for p, _ in pairs):,} PE weights; lr {args.lr:g} (last block) "
          f"... {args.lr * args.decay ** 25:.1e} (patch embedding); head lr {args.lr_head:g}", flush=True)
    print("preparing dev views ...", flush=True)
    dev = dev_data(size)
    f_line = json.load(open("weights/open_gen_v1/head_F_calibration.json"))["ai_logit"]
    report(0, *validate(model, w, b, dev, mean, std, f_line))

    loader = torch.utils.data.DataLoader(TrainImages(rows, size, mean, std), batch_size=args.batch_size, shuffle=True,
                                         drop_last=True, num_workers=args.workers, persistent_workers=True, pin_memory=True)
    step, t0, running, edit_losses = 0, time.time(), [], []
    while step < args.steps:
        for pixels, y, is_edit in loader:
            for g in optimizer.param_groups:
                g["lr"] = g["base_lr"] * factor(step)
            pixels, y = pixels.cuda(non_blocking=True), y.cuda(non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                feats = layer_features(model, pixels)
            logit = feats.flatten(1) @ w + b
            per_picture = F.binary_cross_entropy_with_logits(logit, y, reduction="none")
            loss = (per_picture * class_weight[y.long()]).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for g in optimizer.param_groups for p in g["params"]], 1.0)
            optimizer.step()
            with torch.no_grad():  # decoupled pull towards the pretrained weights (L2-SP), scaled like each group's lr
                for g, d in zip(optimizer.param_groups[:-1], depths):
                    for p, p0 in groups[d]:
                        p.sub_(g["lr"] * args.pull * (p - p0.float()))
            step += 1
            running.append(loss.item())
            edit_losses += per_picture[is_edit.cuda()].tolist()
            if step % 100 == 0:
                print(f"step {step}: loss {np.mean(running):.4f}, edit pairs {np.mean(edit_losses):.4f}, "
                      f"{(time.time() - t0) / step:.2f} s/step", flush=True)
                running, edit_losses = [], []
            if step % args.eval_every == 0 or step == args.steps:
                torch.save(model.state_dict(), args.out_dir / f"pe_step{step}.pt")
                LinearHead(w.detach().cpu().clone(), b.detach().cpu().clone()).save(args.out_dir / f"head_step{step}.pt")
                report(step, *validate(model, w, b, dev, mean, std, f_line))
            if step >= args.steps:
                break
    print(f"done in {(time.time() - t0) / 60:.0f} min; checkpoints in {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
