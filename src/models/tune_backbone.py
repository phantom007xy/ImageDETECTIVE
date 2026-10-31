"""Tune DINOv3 for Path A, the whole-image check: LayerNorms only, the last blocks, or everything (plan Section 5.4).

Why: the frozen DINOv3 + linear head can't tell Muse Image edits from their real originals (AUC 0.65-0.75,
results/muse_edits_v1.txt). Experiments, smallest change first, all from the same start, data and validation:
  --tune layernorm  only the LayerNorm weights (100,352 of 303 million numbers) + head (the GenD recipe)
  --tune last2/last4  the last 2 / 4 transformer blocks + the final norm + head (~25 / ~50 million numbers);
                    backpropagation only runs through these blocks, so a step is faster than with layernorm
  --tune full       everything (303 million numbers; needs a big GPU, e.g. Colab A100; overfits most easily)

What stays as before:
  - The head is linear over the patch features (average of the patch tokens -> one logit), so the evidence
    map in the app stays exact.
  - Training pictures go through the same random training view as for the frozen heads
    (src/data/preprocess.train_view: random crop at the training scale, brightness, JPEG; identical for real
    and AI). Each class counts half, whatever the numbers.
  - Start: the original DINOv3 + head F (--init-head), so step 0 IS head F. An L2 pull towards the original
    weights of everything that is tuned (--pull; "L2-SP") keeps the backbone close to what it knows.

Validation every --eval-every steps, on dev sets only (plan: decisions only on dev sets), like the fair
comparison: the line where at most 5% of EVERY real pool is called AI, then the share of every AI set above it,
in both app views. Every validation saves a checkpoint (the tuned weights + head) to --out-dir; the backbone
loads it with Backbone(tuned_path=...).

Usage (from the repo root; all lists are CSV image lists: path, label, source):
  python -m src.models.tune_backbone --tune last4 --train features/lists/sid_10_14.csv ...
      --init-head weights/open_gen_v1/head_F.pt --out-dir weights/last4_v1 --steps 4000
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
LAYERS = 24  # DINOv3 ViT-L/16


def is_tuned(name, mode):
    """Which DINOv3 weights the mode trains (parameter names: model.layer.<i>.*, norm.*, embeddings.*)."""
    if mode == "layernorm":
        return is_layernorm(name)
    if mode == "full":
        return True
    first = LAYERS - int(mode.removeprefix("last"))
    return name.startswith("norm.") or any(name.startswith(f"model.layer.{i}.") for i in range(first, LAYERS))


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
        is_edit_pair = row["source"].startswith("muse_edits")  # a Muse edit or its real original
        return preprocess.to_tensor(view), torch.tensor(1.0 if int(row["label"]) else 0.0), torch.tensor(is_edit_pair)


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
        edits = np.concatenate([scores["Muse edits local"], scores["Muse edits whole"]])
        report[v]["AUC edits"] = metrics.auc(np.r_[reals, edits], np.r_[np.zeros(len(reals)), np.ones(len(edits))] == 1)
    model.train()
    return report


def print_report(step, report):
    names = list(REAL_POOLS) + list(AI_SETS)
    for v, r in report.items():
        cells = " | ".join(f"{n} {r[n]:.0%}" for n in names)
        print(f"  step {step} [{v}] real called AI / AI caught: {cells} | AUC real vs edits {r['AUC edits']:.3f}", flush=True)


def save(out_dir, step, model, w, b, mode):
    tuned = {n: p.detach().cpu().clone() for n, p in model.named_parameters() if is_tuned(n, mode)}
    torch.save(tuned, out_dir / f"tuned_step{step}.pt")
    LinearHead(w.detach().cpu().clone(), b.detach().cpu().clone()).save(out_dir / f"head_step{step}.pt")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tune", choices=["layernorm", "last2", "last4", "full"], required=True)
    parser.add_argument("--train", nargs="+", type=Path, required=True, help="CSV image lists (path, label, source)")
    parser.add_argument("--init-head", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=4000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, help="for the tuned DINOv3 weights (default: 1e-4 layernorm, 2e-5 otherwise)")
    parser.add_argument("--lr-head", type=float, default=1e-4)
    parser.add_argument("--pull", type=float, default=1e-3, help="L2 pull towards the original weights of the tuned part")
    parser.add_argument("--eval-every", type=int, default=500)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    lr = args.lr or (1e-4 if args.tune == "layernorm" else 2e-5)

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
        p.requires_grad = is_tuned(name, args.tune)
    tuned = [p for p in model.parameters() if p.requires_grad]
    print(f"tuning {args.tune}: {sum(p.numel() for p in tuned):,} of {sum(p.numel() for p in model.parameters()):,} "
          f"DINOv3 weights, lr {lr:g}", flush=True)
    start = {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}
    # Recompute activations in the backward pass (fits in 8 GB). Non-reentrant: with the default reentrant kind,
    # blocks whose inputs don't need gradients (the frozen patch embedding) would pass none to their weights.
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.train()  # DINOv3 has no dropout or drop-path, so train and eval mode compute the same
    skip = 1 + model.config.num_register_tokens
    head = LinearHead.load(args.init_head)
    w = head.weight.float().cuda().requires_grad_()
    b = head.bias.float().reshape(()).cuda().requires_grad_()
    optimizer = torch.optim.AdamW([{"params": tuned, "lr": lr}, {"params": [w, b], "lr": args.lr_head}], weight_decay=0.0)
    warmup = 200
    schedule = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: min(1.0, (s + 1) / warmup) * (
        0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1.0, s / args.steps)))))

    print("preparing dev views ...", flush=True)
    dev = dev_views(random.Random(args.seed))
    print_report(0, validate(model, w, b, dev, skip))

    loader = torch.utils.data.DataLoader(TrainImages(rows), batch_size=args.batch_size, shuffle=True, drop_last=True,
                                         num_workers=args.workers, persistent_workers=True, pin_memory=True)
    step, t0, running, edit_losses = 0, time.time(), [], []
    while step < args.steps:
        for pixels, y, is_edit_pair in loader:
            pixels, y = pixels.cuda(non_blocking=True), y.cuda(non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                tokens = model(pixel_values=pixels).last_hidden_state[:, skip:]
            logit = tokens.float().mean(1) @ w + b
            per_picture = F.binary_cross_entropy_with_logits(logit, y, reduction="none")
            loss = (per_picture * class_weight.cuda()[y.long()]).mean()
            pull = sum(((p - start[n]) ** 2).sum() for n, p in model.named_parameters() if p.requires_grad)
            optimizer.zero_grad(set_to_none=True)
            (loss + args.pull * pull).backward()
            optimizer.step()
            schedule.step()
            step += 1
            running.append(loss.item())
            # Plain loss on the Muse edit pairs alone: ~0.69 = can't tell edits from their originals; lower = learning it.
            edit_losses += per_picture[is_edit_pair.cuda()].tolist()
            if step % 100 == 0:
                print(f"step {step}: loss {np.mean(running):.4f}, edit pairs {np.mean(edit_losses):.4f}, "
                      f"drift {pull.item() ** 0.5:.3f}, {(time.time() - t0) / step:.2f} s/step", flush=True)
                running, edit_losses = [], []
            if step % args.eval_every == 0 or step == args.steps:
                save(args.out_dir, step, model, w, b, args.tune)
                print_report(step, validate(model, w, b, dev, skip))
            if step >= args.steps:
                break
    print(f"done in {(time.time() - t0) / 60:.0f} min; checkpoints in {args.out_dir}")


if __name__ == "__main__":
    main()
