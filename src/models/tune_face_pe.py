"""Fine-tune Meta's Perception Encoder (PE-Core-L/14-336) as the face-swap check, on face crops.

Why: frozen features + a linear head catch few face swaps at the strict line (results/swap_head_v2.txt). Tuning the
backbone is what makes face-swap detectors generalise (GenD, WACV 2026: frozen linear probe ~75 vs LayerNorm-tuned ~91
mean cross-dataset AUROC), and the same recipe took our Muse-edit check from AUC 0.75 to 0.97 (edit_check_v3.txt).

Data (src/faces/make_crops.py; label 0 = real face, 1 = swapped or replaced face):
  - real swaps: OpenForensics (faces replaced by GAN faces in still photos), DFDC (video face swaps), RTFS-10k (video
    calls swapped with inswapper and UniFace, today's common swap tools; real frames re-encoded like the swaps)
  - self-blended faces (Shiohara & Yamasaki, CVPR 2022): a real face blended onto itself through its SAM mask after
    small changes of colour, sharpness, resolution and position. It teaches the seam a face swap leaves without any
    swap tool, which is why it carries over to swap methods never seen in training. Made on the fly from real faces.
  - real faces from photos (OpenForensics, SID-Set, Commons) and video frames (DFDC)
  Each source gets a fixed share of every batch (--shares), whatever its size.
Training view: a random 2.6-4.0 eye-distance square of the wide crop (as src/faces/crop.train_face_view) at 336,
flips, brightness, contrast, low resolution and JPEG 60-95, the same for every class.
Model: as src/models/tune_pe.py (layer-wise lr decay, a pull towards the pretrained weights, stochastic depth, bf16 +
gradient checkpointing); features: mean patch tokens after blocks 4..20 and of the last layer -> linear head. The head
starts as a linear probe of the frozen PE (step 0 = the frozen baseline).
Validation every --eval-every steps on DEV crops in the app's view: the strict line (at most 0.5% of the faces of every
real dev pool above it), then the share of OpenForensics dev and DFDC part 3 swaps above it, and AUCs.

Usage (from the repo root):
  python -m src.models.tune_face_pe --crops D:/Datasets/FaceCrops --out-dir weights/face_pe_v1
"""

import argparse
import csv
import math
import random
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageEnhance
from timm.data import resolve_data_config

from src.data.preprocess import jpeg
from src.eval import metrics
from src.faces.make_crops import WIDE_K, WIDE_SIZE
from src.models.linear_head import LinearHead
from src.models.pe_core import load_pe
from src.models.tune_pe import layer_features

# Training crop sets (make_crops manifests) -> the source names inside them; shares of a batch per source group.
TRAIN_SETS = ["of_val_train", "dfdc_train", "real_sid", "real_commons"]  # default; --train-sets adds more
# Batch shares: the swaps keep 40%; the real faces share 60% equally per source (--share overrides one group).
FAKE_SHARES = {"openforensics/fake": 0.20, "dfdc/fake": 0.20, "rtfs/fake": 0.15}
REAL_TOTAL = 0.60
# Dev crop sets (app view): real pools set the strict line; swap sets are measured at it.
DEV_REAL = {"polished dev": "pool_commons_dev", "OpenImages val": "pool_muse_pairs_val",
            "OpenImages unseen": "pool_sid_reals_0_9_unseen", "close-ups": "pool_real_face_crops_dev_all",
            "selfies+portraits": "pool_commons_v4_dev", "SID val": "pool_sid_val",
            "FaceID-550 new people": "pool_faceid550_dev",  # selfies of people never in training (if downloaded)
            "RTFS video calls": "rtfs_dev"}  # its real faces (the swaps are below)
DEV_SWAPS = {"OF dev": "of_val_dev", "DFDC p3": "dfdc_p3_dev", "RTFS dev": "rtfs_dev"}


def self_blend(img, mask, rng):
    """A real face blended onto itself through its (changed) mask after small changes: the seam of a face swap.
    img: H x W x 3 uint8, mask: H x W bool -> H x W x 3 uint8."""
    h, w = mask.shape
    base = img.astype(np.float32)
    src = (base - 127.5) * rng.uniform(0.9, 1.1) + 127.5 + rng.uniform(-12, 12) + np.array([rng.uniform(-10, 10) for _ in range(3)])
    r = rng.random()
    if r < 0.33:
        src = cv2.GaussianBlur(src, (0, 0), rng.uniform(0.6, 1.6))
    elif r < 0.66:
        src = src + rng.uniform(0.3, 1.0) * (src - cv2.GaussianBlur(src, (0, 0), 1.0))
    if rng.random() < 0.5:  # resolution: down and up again
        f = rng.uniform(0.4, 0.9)
        src = cv2.resize(cv2.resize(src, (max(8, int(w * f)), max(8, int(h * f))), interpolation=cv2.INTER_AREA), (w, h),
                         interpolation=cv2.INTER_LINEAR)
    ys, xs = np.nonzero(mask)  # position: a small scale and shift around the face centre
    cx, cy = xs.mean(), ys.mean()
    s = rng.uniform(0.96, 1.04)
    tx, ty = rng.uniform(-0.02, 0.02) * w, rng.uniform(-0.02, 0.02) * h
    src = cv2.warpAffine(src, np.float32([[s, 0, (1 - s) * cx + tx], [0, s, (1 - s) * cy + ty]]), (w, h),
                         borderMode=cv2.BORDER_REFLECT)
    m = mask.astype(np.uint8)  # the mask: grown or shrunk, softened, a random strength
    size = rng.randrange(3, 22) | 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    changed = cv2.dilate(m, kernel) if rng.random() < 0.5 else cv2.erode(m, kernel)
    m = changed if changed.sum() > 0.3 * m.sum() else m
    m = cv2.GaussianBlur(m.astype(np.float32), (0, 0), rng.uniform(2, 10))
    m = m / max(float(m.max()), 1e-6) * rng.choice([0.5, 0.75, 1.0, 1.0, 1.0])
    out = m[..., None] * src + (1 - m[..., None]) * base
    return np.clip(out, 0, 255).astype(np.uint8)


def photo_damage(img, rng):
    """Like a compressed or small copy of the whole photo (a WhatsApp forward): the face shrunk to 25-100%, JPEG 45-90 at
    that size, enlarged again. Compression at the photo's scale wipes far more of a swap's traces than JPEG on the
    enlarged 336 px crop (the first run's view): results/face_pe_v1.txt, forwarded copies."""
    side = img.size[0]
    small = img.resize((max(32, round(side * rng.uniform(0.25, 1.0))),) * 2, Image.BICUBIC)
    return jpeg(small, rng.randint(45, 90)).resize((side, side), Image.BICUBIC)


def phone_process(img, mask, rng):
    """Real phone processing of a REAL face (it stays real): skin smoothing, a brighter skin tone or a portrait-mode
    background blur, all aligned with the face (no shift, no resolution seam). Teaches the model that a face processed
    differently from its surroundings is not by itself a swap: real phone faces with such processing scored far above
    real public photos (2 Nov), while self-blended fakes keep their misalignment and resolution seams."""
    base = np.asarray(img).astype(np.float32)
    soft = cv2.dilate(mask.astype(np.uint8), np.ones((7, 7), np.uint8)).astype(np.float32)
    soft = cv2.GaussianBlur(soft, (0, 0), rng.uniform(3, 8))
    soft = (soft / max(float(soft.max()), 1e-6))[..., None]
    r = rng.random()
    if r < 0.4:  # skin smoothing (beauty filter)
        smooth = cv2.bilateralFilter(np.asarray(img), rng.choice([5, 7, 9]), rng.uniform(30, 90), rng.uniform(5, 15))
        out = soft * (base + rng.uniform(0.5, 1.0) * (smooth.astype(np.float32) - base)) + (1 - soft) * base
    elif r < 0.7:  # a brighter, evener skin tone
        tone = (base - 127.5) * rng.uniform(0.9, 1.0) + 127.5 + rng.uniform(5, 25) + np.array([rng.uniform(-6, 6) for _ in range(3)])
        out = soft * tone + (1 - soft) * base
    else:  # portrait mode: the background blurred around the (much enlarged) face
        size = rng.randrange(41, 121) | 1
        head = cv2.dilate(mask.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size)))
        keep = cv2.GaussianBlur(head.astype(np.float32), (0, 0), rng.uniform(3, 8))[..., None]
        out = keep * base + (1 - keep) * cv2.GaussianBlur(base, (0, 0), rng.uniform(2, 8))
    return np.clip(out, 0, 255).astype(np.uint8)


def phone_process_dev(img):
    """A fixed version for the dev check (app crops have no mask: an ellipse where the face sits in the app crop)."""
    a = np.asarray(img)
    mask = np.zeros(a.shape[:2], np.uint8)
    cv2.ellipse(mask, (168, 150), (80, 105), 0, 0, 360, 1, -1)
    soft = cv2.GaussianBlur(mask.astype(np.float32), (0, 0), 5)[..., None]
    base = a.astype(np.float32)
    smooth = cv2.bilateralFilter(a, 9, 60, 9).astype(np.float32)
    face = base + 0.8 * (smooth - base) + 12
    big = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (81, 81))).astype(np.float32)
    big = cv2.GaussianBlur(big, (0, 0), 5)[..., None]
    out = big * (soft * face + (1 - soft) * base) + (1 - big) * cv2.GaussianBlur(base, (0, 0), 4)
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))


def face_train_view(img, rng, size):
    """A random 2.6-4.0 eye-distance square of a wide crop (one eye distance = WIDE_SIZE / WIDE_K px), at size x size."""
    eye = WIDE_SIZE / WIDE_K
    k = rng.uniform(2.6, 4.0)
    cx, cy = WIDE_SIZE / 2 + rng.uniform(-0.1, 0.1) * eye, WIDE_SIZE / 2 + rng.uniform(-0.1, 0.1) * eye
    half = k * eye / 2
    view = img.crop((round(cx - half), round(cy - half), round(cx + half), round(cy + half))).resize((size, size), Image.BICUBIC)
    if rng.random() < 0.5:
        view = view.transpose(Image.FLIP_LEFT_RIGHT)
    view = ImageEnhance.Brightness(view).enhance(rng.uniform(0.6, 1.5))
    view = ImageEnhance.Contrast(view).enhance(rng.uniform(0.85, 1.15))
    if rng.random() < 0.3:  # like a small face: shrink and enlarge again
        small = rng.randint(64, 200)
        view = view.resize((small, small), Image.BICUBIC).resize((size, size), Image.BICUBIC)
    return jpeg(view, rng.randint(60, 95))


class FaceCrops(torch.utils.data.Dataset):
    def __init__(self, rows, size, mean, std, p_blend, p_damage=0.0, p_phone=0.0):
        self.rows, self.size, self.mean, self.std, self.p_blend, self.p_damage = rows, size, mean, std, p_blend, p_damage
        self.p_phone = p_phone

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        rng = random  # each DataLoader worker process has its own generator
        row = self.rows[i]
        with Image.open(row["path"]) as im:
            img = im.convert("RGB")
        label, kind = int(row["label"]), 1 if int(row["label"]) else 0  # 0 real, 1 swap, 2 self-blended, 3 phone-processed real
        if label == 0 and row["mask"]:
            with Image.open(row["mask"]) as m:
                mask = np.asarray(m.convert("L")) > 127
            if mask.sum() > 200:
                if rng.random() < self.p_blend:
                    img, label, kind = Image.fromarray(self_blend(np.asarray(img), mask, rng)), 1, 2
                elif rng.random() < self.p_phone:
                    img, kind = Image.fromarray(phone_process(img, mask, rng)), 3
        if rng.random() < self.p_damage:  # every class alike (after self-blending: the seam gets compressed too)
            img = photo_damage(img, rng)
        view = face_train_view(img, rng, self.size)
        pixels = torch.from_numpy(np.asarray(view, dtype=np.float32) / 255.0).permute(2, 0, 1)
        return (pixels - self.mean) / self.std, torch.tensor(float(label)), torch.tensor(kind)


def read_manifest(crops, name):
    path = crops / f"{name}.csv"
    if not path.exists():
        print(f"  (no {path}: skipped)", flush=True)
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def source_group(row):
    """Swaps: one group per dataset (openforensics/fake, dfdc/fake, rtfs/fake); real faces: one group per dataset."""
    s = row["source"]
    return s if s.endswith("/fake") else f"{s.split('/')[0]}/real"


def dev_crops(crops, mean, std, max_per_set, seed):
    """name -> (pixels tensor [N,3,336,336] as uint8 to save memory, labels); swap sets subsampled by photo/video group."""
    out = {}
    for name, stem in {**DEV_REAL, **DEV_SWAPS}.items():
        rows = read_manifest(crops, stem)
        if name in DEV_REAL:
            rows = [r for r in rows if r["label"] == "0"]
        if name in DEV_SWAPS and len(rows) > max_per_set:
            groups = sorted({r["group"] for r in rows})
            random.Random(seed).shuffle(groups)
            keep, chosen = set(), 0
            for g in groups:
                if chosen >= max_per_set:
                    break
                keep.add(g)
                chosen += sum(r["group"] == g for r in rows)
            rows = [r for r in rows if r["group"] in keep]
        if rows:
            images = [Image.open(r["path"]).convert("RGB") for r in rows]
            labels = np.array([int(r["label"]) for r in rows])
            out[name] = (torch.from_numpy(np.stack([np.asarray(i) for i in images])).permute(0, 3, 1, 2).contiguous(), labels)
            # a fixed compressed copy (the face at 60%, JPEG 70, enlarged again): roughly a forwarded photo's face
            damaged = [jpeg(i.resize((202, 202), Image.BICUBIC), 70).resize((336, 336), Image.BICUBIC) for i in images]
            out[name + " (compressed)"] = (torch.from_numpy(np.stack([np.asarray(i) for i in damaged])).permute(0, 3, 1, 2)
                                           .contiguous(), labels)
            if name in DEV_REAL:  # real faces with phone-like processing (beauty smoothing, portrait blur): must stay real
                processed = [phone_process_dev(i) for i in images]
                out[name + " (phone-processed)"] = (torch.from_numpy(np.stack([np.asarray(i) for i in processed]))
                                                    .permute(0, 3, 1, 2).contiguous(), labels)
    return out


@torch.no_grad()
def validate(model, w, b, dev, mean, std):
    model.eval()
    scores = {}
    for name, (imgs, _) in dev.items():
        s = []
        for start in range(0, len(imgs), 32):
            pixels = ((imgs[start:start + 32].float() / 255.0 - mean) / std).cuda()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                feats = layer_features(model, pixels)
            s.append((feats.flatten(1) @ w + b).float().cpu().numpy())
        scores[name] = np.concatenate(s)
    model.train()
    reals = {n: scores[n] for n in DEV_REAL if n in scores}
    line = max(float(np.quantile(s, 0.995)) for s in reals.values())
    result = {"line": line}
    for n in list(DEV_SWAPS) + [d + " (compressed)" for d in DEV_SWAPS]:
        if n in scores:
            labels = dev[n][1]
            result[n] = (float((scores[n][labels == 1] > line).mean()), metrics.auc(scores[n], labels == 1))
    result["reals"] = {n: float((s > line).mean()) for n, s in reals.items()}
    result["reals compressed"] = max(float((scores[n + " (compressed)"] > line).mean()) for n in reals)
    result["reals phone"] = max(float((scores[n + " (phone-processed)"] > line).mean()) for n in reals)
    return result


def report(step, r):
    swaps = " | ".join(f"{n}: caught {r[n][0]:.1%} (AUC {r[n][1]:.3f})" for n in r
                       if n not in ("line", "reals", "reals compressed", "reals phone"))
    swaps += (f" | real faces above (worst pool): compressed {r['reals compressed']:.1%}, phone-processed "
              f"{r['reals phone']:.1%}")
    print(f"  step {step}: strict line {r['line']:+.2f} | {swaps} | real faces above: "
          + " ".join(f"{n} {v:.1%}" for n, v in r["reals"].items()), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--crops", type=Path, default=Path("D:/Datasets/FaceCrops"))
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5, help="for the last block (layer-wise decay below it)")
    parser.add_argument("--decay", type=float, default=0.85)
    parser.add_argument("--lr-head", type=float, default=1e-4)
    parser.add_argument("--pull", type=float, default=1.0)
    parser.add_argument("--p-blend", type=float, default=0.5, help="share of real training faces turned self-blended")
    parser.add_argument("--probe-samples", type=int, default=4000, help="training views for the frozen probe (head start)")
    parser.add_argument("--dev-max", type=int, default=1200, help="dev swap-set size (whole photos/videos)")
    parser.add_argument("--eval-every", type=int, default=500)
    parser.add_argument("--init-pe", type=Path, help="continue from this fine-tuned PE (the pull then anchors to it)")
    parser.add_argument("--init-head", type=Path, help="... and this head (instead of a frozen probe)")
    parser.add_argument("--p-damage", type=float, default=0.0, help="share of training faces given photo-scale damage")
    parser.add_argument("--p-phone", type=float, default=0.0, help="share of the other real faces given phone processing")
    parser.add_argument("--warmup", type=int, default=300)
    parser.add_argument("--train-sets", nargs="+", default=TRAIN_SETS, help="crop manifests (stems) to train on")
    parser.add_argument("--share", action="append", default=[], help='"GROUP=SHARE", e.g. "fairface/real=0.1"')
    parser.add_argument("--no-checkpointing", action="store_true", help="faster on a big GPU (more memory)")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows = [r for name in args.train_sets for r in read_manifest(args.crops, name)]
    groups = np.array([source_group(r) for r in rows])
    present = sorted(set(groups))
    reals = [g for g in present if g.endswith("/real")]
    shares = {g: s for g, s in FAKE_SHARES.items() if g in present}
    shares.update({g: REAL_TOTAL / len(reals) for g in reals})
    for spec in args.share:
        g, v = spec.split("=")
        shares[g] = float(v)
    shares = {g: s for g, s in shares.items() if g in present}  # a dataset that failed to download drops out
    counts = {g: int((groups == g).sum()) for g in present}
    print("training faces: " + " | ".join(f"{g} {counts[g]} (share {shares.get(g, 0):.0%})" for g in present), flush=True)
    total = sum(shares.values())
    weights = np.array([shares.get(g, 0) / total / max(counts.get(g, 1), 1) for g in groups])
    real_share = sum(s for g, s in shares.items() if g.endswith("/real")) / total
    fake_share = (1 - real_share) + real_share * args.p_blend  # real swaps + self-blended
    class_weight = torch.tensor([0.5 / (1 - fake_share), 0.5 / fake_share], dtype=torch.float32).cuda()
    print(f"batch mix: real {1 - fake_share:.0%}, swapped {1 - real_share:.0%}, self-blended {real_share * args.p_blend:.0%}",
          flush=True)

    model = load_pe(pretrained=True, drop_path_rate=0.1).cuda()
    model.set_grad_checkpointing(not args.no_checkpointing)
    cfg = resolve_data_config({}, model=model)
    size, mean, std = cfg["input_size"][1], torch.tensor(cfg["mean"]).view(3, 1, 1), torch.tensor(cfg["std"]).view(3, 1, 1)
    if args.init_pe:
        model.load_state_dict(torch.load(args.init_pe, map_location="cpu"), strict=True)
    dataset = FaceCrops(rows, size, mean, std, args.p_blend, args.p_damage, args.p_phone)
    sampler = torch.utils.data.WeightedRandomSampler(weights, num_samples=10 ** 7, replacement=True)
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, sampler=sampler, drop_last=True,
                                         num_workers=args.workers, persistent_workers=True, pin_memory=True)

    print("preparing dev crops ...", flush=True)
    dev = dev_crops(args.crops, mean, std, args.dev_max, args.seed)
    print("dev: " + " | ".join(f"{n} {len(v[1])}" for n, v in dev.items()), flush=True)

    # The head starts as a linear probe of the frozen PE on training views (step 0 = the frozen baseline),
    # or from --init-head when continuing a run.
    if args.init_head:
        probe = LinearHead.load(args.init_head)
    else:
        model.eval()
        feats, labels = [], []
        with torch.no_grad():
            for pixels, y, _ in loader:
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    feats.append(layer_features(model, pixels.cuda()).flatten(1).float().cpu())
                labels.append(y)
                if sum(len(f) for f in feats) >= args.probe_samples:
                    break
        probe = LinearHead().fit(torch.cat(feats), torch.cat(labels), weight_decay=1e-2)
    model.train()
    w = probe.weight.float().cuda().requires_grad_()
    b = probe.bias.float().reshape(()).cuda().requires_grad_()
    report(0, validate(model, w, b, dev, mean, std))
    LinearHead(w.detach().cpu().clone(), b.detach().cpu().clone()).save(args.out_dir / "head_step0.pt")

    groups_lr = {}
    for name, p in model.named_parameters():
        if name.startswith("attn_pool"):
            p.requires_grad = False
            continue
        depth = int(name.split(".")[1]) + 1 if name.startswith("blocks.") else (25 if name.startswith("norm.") else 0)
        groups_lr.setdefault(depth, []).append((p, p.detach().to(torch.bfloat16).clone()))
    depths = sorted(groups_lr)
    param_groups = [{"params": [p for p, _ in groups_lr[d]], "lr": args.lr * args.decay ** (25 - d),
                     "base_lr": args.lr * args.decay ** (25 - d)} for d in depths]
    param_groups.append({"params": [w, b], "lr": args.lr_head, "base_lr": args.lr_head})
    optimizer = torch.optim.AdamW(param_groups, weight_decay=0.0)
    warmup = args.warmup

    def factor(s):
        return min(1.0, (s + 1) / warmup) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / args.steps)))

    step, t0, running, by_kind = 0, time.time(), [], {0: [], 1: [], 2: [], 3: []}
    for pixels, y, kind in loader:
        for g in optimizer.param_groups:
            g["lr"] = g["base_lr"] * factor(step)
        pixels, y = pixels.cuda(non_blocking=True), y.cuda(non_blocking=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            feats = layer_features(model, pixels)
        per_face = F.binary_cross_entropy_with_logits(feats.flatten(1) @ w + b, y, reduction="none")
        loss = (per_face * class_weight[y.long()]).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for g in optimizer.param_groups for p in g["params"]], 1.0)
        optimizer.step()
        with torch.no_grad():  # decoupled pull towards the pretrained weights (L2-SP)
            for g, d in zip(optimizer.param_groups[:-1], depths):
                for p, p0 in groups_lr[d]:
                    p.sub_(g["lr"] * args.pull * (p - p0.float()))
        step += 1
        running.append(loss.item())
        for k in (0, 1, 2, 3):
            by_kind[k] += per_face[kind.cuda() == k].tolist()
        if step % 100 == 0:
            parts = " ".join(f"{n} {np.mean(by_kind[k]):.3f}" for k, n in ((0, "real"), (1, "swap"), (2, "blended"),
                                                                           (3, "phone-real")) if by_kind[k])
            print(f"step {step}: loss {np.mean(running):.4f} ({parts}), {(time.time() - t0) / step:.2f} s/step", flush=True)
            running, by_kind = [], {0: [], 1: [], 2: [], 3: []}
        if step % args.eval_every == 0 or step == args.steps:
            torch.save(model.state_dict(), args.out_dir / f"pe_step{step}.pt")
            LinearHead(w.detach().cpu().clone(), b.detach().cpu().clone()).save(args.out_dir / f"head_step{step}.pt")
            report(step, validate(model, w, b, dev, mean, std))
        if step >= args.steps:
            break
    print(f"done in {(time.time() - t0) / 60:.0f} min; checkpoints in {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
