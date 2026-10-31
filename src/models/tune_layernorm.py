"""LayerNorm-tuning of DINOv3 for Path A, the whole-image check (plan Section 5.4, step 2).

Why: the frozen DINOv3 + linear head can't tell Muse Image edits from their real originals (AUC 0.65-0.75,
results/muse_edits_v1.txt), and the papers agree that tuning helps a lot (GenD: frozen 74.9 -> LayerNorm-
tuned 91.3 mean AUROC on face deepfakes). Here only the LayerNorm weights of DINOv3 (100,352 of 303 million
numbers: the two norms of every block and the final norm) and the linear head are trained, by
backpropagation through the rest of the network, which stays frozen.

What stays as before:
  - The head is linear over the patch features (average of the patch tokens -> one logit), so the evidence
    map in the app stays exact.
  - Training pictures go through the same random training view as for the frozen heads
    (src/data/preprocess.train_view: random crop at the training scale, brightness, JPEG; identical for real
    and AI). Each class counts half, whatever the numbers.
  - Start: the original DINOv3 + head F (--init-head), so step 0 IS head F. An L2 pull towards the original
    LayerNorm weights (--pull; "L2-SP") keeps the backbone close to what it knows about real photos.

Validation every --eval-every steps, on dev sets only (plan: decisions only on dev sets), exactly like the fair
comparison: the line where at most 5% of EVERY real pool is called AI, then the share of every AI set above it,
in both app views. Every validation saves a checkpoint (LayerNorm weights + head, ~0.4 MB) to --out-dir.

Usage (from the repo root; all lists are CSV image lists: path, label, source):
  python -m src.models.tune_layernorm --train features/lists/sid_10_14.csv features/lists/commons_train.csv ...
      --init-head weights/open_gen_v1/head_F.pt --out-dir weights/ln_v1 --steps 6000
"""

import argparse
import csv
import io
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps
from transformers import AutoModel

from src.data import preprocess
from src.eval import metrics
from src.models.backbone import MODEL_NAME, is_layernorm, load_local_first
from src.models.linear_head import LinearHead

# Dev sets (never trained on): (list, labels to keep, source to keep or None). REAL pools set the line.
REAL_POOLS = {
    "polished dev": ("features/lists/commons_dev.csv", {0}, None),
    "OpenImages val": ("features/lists/muse_pairs_val.csv", {0}, None),
    "edit originals": ("features/lists/muse_edits_v1.csv", {0}, None),
}
AI_SETS = {
    "Muse test v1": ("features/lists/muse_image_test_v1.csv", {1}, None),
    "Muse pairs val": ("features/lists/muse_pairs_val.csv", {1}, None),
    "Muse edits local": ("features/lists/muse_edits_v1.csv", {2}, None),
    "Muse edits whole": ("features/lists/muse_edits_v1.csv", {1}, None),
    "SDXL (unseen)": ("features/lists/sd_faces.csv", {1}, "sd_faces/sdxl"),
    "PixArt (unseen)": ("features/lists/open_gen_v1/open_gen_dev.csv", {1}, "open_gen/pixart"),
    "SD3.5 dev": ("features/lists/open_gen_v1/open_gen_dev.csv", {1}, "open_gen/sd35m_devcaption"),
}
DEV_CAP = 200  # at most this many pictures per dev set (a fixed random sample), to keep validation short
VIEWS = ("long448", "short448")


def read_rows(path, labels=None, source=None):
    with open(path, newline="") as f:
        return [r for r in csv.DictReader(f)
                if (labels is None or int(r["label"]) in labels) and (source is None or r["source"] == source)]


def open_upright(path):
    with Image.open(path) as im:
        return ImageOps.exif_transpose(im).convert("RGB")


class TrainImages(torch.utils.data.Dataset):
    """One random training view per call (Python's random module is seeded per DataLoader worker)."""

    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        row = self.rows[i]
        view = preprocess.train_view(open_upright(row["path"]), random)
        return preprocess.to_tensor(view), torch.tensor(1.0 if int(row["label"]) else 0.0)


def dev_views(rng):
    """Every dev picture in both app views, kept in memory as PNG bytes (exactly what the app would see)."""
    out = {}
    for name, (path, labels, source) in {**REAL_POOLS, **AI_SETS}.items():
        rows = read_rows(path, labels, source)
        rows = rng.sample(rows, min(DEV_CAP, len(rows)))
        views = {v: [] for v in VIEWS}
        for row in rows:
            image = open_upright(row["path"])
            for v in VIEWS:
                buffer = io.BytesIO()
                preprocess.inference_view(image, v).save(buffer, format="PNG")
                views[v].append(buffer.getvalue())
        out[name] = views
        print(f"  dev {name}: {len(rows)} pictures", flush=True)
    return out


@torch.no_grad()
def dev_logits(model, w, b, views, skip):
    logits = []
    for data in views:
        pixels = preprocess.to_tensor(Image.open(io.BytesIO(data)).convert("RGB")).unsqueeze(0).cuda()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            tokens = model(pixel_values=pixels).last_hidden_state[:, skip:]
        logits.append(float(tokens.float().mean(1)[0] @ w + b))
    return np.array(logits)


def validate(model, w, b, dev, skip):
    """Fair line per view (at most 5% of every real pool above it) and the share of every AI set above it."""
    model.eval()
    report = {}
    for v in VIEWS:
        scores = {name: dev_logits(model, w, b, views[v], skip) for name, views in dev.items()}
        line = max(np.sort(scores[n])[::-1][math.floor(0.05 * len(scores[n]))] for n in REAL_POOLS)
        report[v] = {n: float((scores[n] > line).mean()) for n in scores}
        reals = np.concatenate([scores[n] for n in REAL_POOLS])
        report[v]["AUC edits"] = metrics.auc(np.concatenate([reals, scores["Muse edits local"], scores["Muse edits whole"]]),
                                             np.r_[np.zeros(len(reals)), np.ones(len(scores["Muse edits local"]) +
                                                                                 len(scores["Muse edits whole"]))] == 1)
    model.train()
    return report


def print_report(step, report):
    names = list(REAL_POOLS) + list(AI_SETS)
    for v, r in report.items():
        cells = " | ".join(f"{n} {r[n]:.0%}" for n in names)
        print(f"  step {step} [{v}] real called AI / AI caught: {cells} | AUC real vs edits {r['AUC edits']:.3f}", flush=True)


def save(out_dir, step, model, w, b):
    ln = {n: p.detach().cpu().clone() for n, p in model.named_parameters() if is_layernorm(n)}
    torch.save(ln, out_dir / f"ln_step{step}.pt")
    LinearHead(w.detach().cpu().clone(), b.detach().cpu().clone()).save(out_dir / f"head_step{step}.pt")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--train", nargs="+", type=Path, required=True, help="CSV image lists (path, label, source)")
    parser.add_argument("--init-head", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=6000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4, help="for the LayerNorm weights and the head")
    parser.add_argument("--pull", type=float, default=1e-3, help="L2 pull towards the original LayerNorm weights")
    parser.add_argument("--eval-every", type=int, default=1000)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows = [r for path in args.train for r in read_rows(path)]
    labels = np.array([1 if int(r["label"]) else 0 for r in rows])
    print(f"training pictures: {len(rows)} ({int((labels == 0).sum())} real, {int(labels.sum())} AI) from {len(args.train)} lists")
    # Each class counts half (as in LinearHead.fit).
    class_weight = torch.tensor([len(rows) / (2 * (labels == 0).sum()), len(rows) / (2 * labels.sum())], dtype=torch.float32)

    model = load_local_first(AutoModel.from_pretrained, MODEL_NAME).cuda()  # fp32 weights, bf16 compute
    for name, p in model.named_parameters():
        p.requires_grad = is_layernorm(name)
    start = {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}
    # Recompute activations in the backward pass (fits in 8 GB). Non-reentrant: with the default reentrant kind,
    # blocks whose inputs don't need gradients (the frozen patch embedding) would pass none to their LayerNorms.
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.train()  # DINOv3 has no dropout or drop-path, so train and eval mode compute the same
    skip = 1 + model.config.num_register_tokens
    head = LinearHead.load(args.init_head)
    w = head.weight.float().cuda().requires_grad_()
    b = head.bias.float().reshape(()).cuda().requires_grad_()
    tuned = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW([{"params": tuned}, {"params": [w, b]}], lr=args.lr, weight_decay=0.0)
    warmup = 200
    schedule = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: min(1.0, (s + 1) / warmup) * (
        0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1.0, s / args.steps)))))

    print("preparing dev views ...", flush=True)
    dev = dev_views(random.Random(args.seed))
    report = validate(model, w, b, dev, skip)
    print_report(0, report)

    loader = torch.utils.data.DataLoader(TrainImages(rows), batch_size=args.batch_size, shuffle=True, drop_last=True,
                                         num_workers=args.workers, persistent_workers=True, pin_memory=True)
    step, t0, running = 0, time.time(), []
    while step < args.steps:
        for pixels, y in loader:
            pixels, y = pixels.cuda(non_blocking=True), y.cuda(non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                tokens = model(pixel_values=pixels).last_hidden_state[:, skip:]
            logit = tokens.float().mean(1) @ w + b
            loss = F.binary_cross_entropy_with_logits(logit, y, weight=class_weight.cuda()[y.long()])
            pull = sum(((p - start[n]) ** 2).sum() for n, p in model.named_parameters() if p.requires_grad)
            total = loss + args.pull * pull
            optimizer.zero_grad(set_to_none=True)
            total.backward()
            optimizer.step()
            schedule.step()
            step += 1
            running.append(float(loss))
            if step % 100 == 0:
                print(f"step {step}: loss {np.mean(running):.4f}, LayerNorm drift {float(pull) ** 0.5:.3f}, "
                      f"{(time.time() - t0) / step:.2f} s/step", flush=True)
                running = []
            if step % args.eval_every == 0 or step == args.steps:
                save(args.out_dir, step, model, w, b)
                print_report(step, validate(model, w, b, dev, skip))
            if step >= args.steps:
                break
    print(f"done in {(time.time() - t0) / 60:.0f} min; checkpoints in {args.out_dir}")


if __name__ == "__main__":
    main()
