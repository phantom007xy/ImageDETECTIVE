"""Same-content pairs, step 1: pick real photos and let Muse Spark describe each one as a prompt.

The real photos come from SID-Set training files (public OpenImages photos) or from a public image
list such as the Commons training photos (--list), never from private photos. Muse Spark runs in standard mode (muse-spark-1.3, not the "-contributor" variant, so Meta
does not train on them). Step 2 is scripts/make_muse_images.py with the prompts.csv written here.
Real and fake then show the same content, so a head trained on them has to learn the generator's
traces, not "what is in the photo".

The pairs are split into train and val before anything is generated; a photo and its fake always
stay in the same split. Photos that show any child or anything sensitive are skipped.

Without --spend: only picks the photos, saves them unchanged and writes pairs.csv (no API call).
With --spend: captions every photo not yet captioned. COSTS CREDITS (about $0.002 per photo).

Usage:
  python scripts/caption_photos.py --parquet "D:/Datasets/SID_Set/data/train-0000[0-9]-of-00249.parquet"
      --faces "features/faces/train-0000[0-9]-of-00249_train.npz" --out-dir D:/Datasets/MuseImage/pairs_v1
  python scripts/caption_photos.py ... --limit 5 --spend
  python scripts/caption_photos.py --list features/lists/commons_v1_train.csv --quota commons/people=150
      --quota commons/portrait=120 --n-val 0 --out-dir D:/Datasets/MuseImage/pairs_v2
"""

import argparse
import base64
import csv
import glob
import io
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import httpx
import numpy as np
import pyarrow.parquet as pq
from PIL import Image

from make_muse_images import api_key  # same .env reading as for Muse Image

API_URL = "https://api.meta.ai/v1/chat/completions"
MODEL = "muse-spark-1.3"
PRICE_IN, PRICE_OUT = 1.25 / 1e6, 4.25 / 1e6  # $ per token (standard mode)
INSTRUCTION = (
    "Write a prompt for an image generator that would recreate a photo like this one. "
    "One or two sentences, at most 60 words. Describe the main subjects and what they are doing, "
    "the setting, the lighting, the camera angle and distance, and what kind of photo it is "
    "(for example a phone snapshot or a DSLR photo). Describe people only in general terms "
    "(for example 'a woman in her 30s in a red jacket'); never name or identify anyone, and do not "
    "copy any text, logos or brand names. If the photo shows any child or teenager (anyone who may "
    "be under 18), or nudity, violence or anything else sensitive, reply with the single word SKIP. "
    "Otherwise reply with the prompt only."
)
CAPTION_FIELDS = ["pair_id", "status", "caption", "input_tokens", "output_tokens"]


def pick_pairs(args):
    """Choose the real photos (deterministic) and save them; returns the rows of pairs.csv."""
    with_face = set()
    for path in glob.glob(args.faces):
        data = np.load(path)
        with_face |= {str(i) for i, label in zip(data["img_ids"], data["labels"]) if label == 0}
    parquets = sorted(glob.glob(args.parquet))
    real_ids = []
    for path in parquets:
        table = pq.read_table(path, columns=["img_id", "label"]).to_pylist()
        real_ids += [r["img_id"] for r in table if r["label"] == 0]
    rng = random.Random(args.seed)
    faces = rng.sample(sorted(i for i in real_ids if i in with_face), args.n_faces)
    others = rng.sample(sorted(i for i in real_ids if i not in with_face), args.n_other)
    chosen = faces + others
    rng.shuffle(chosen)

    (args.out_dir / "real").mkdir(parents=True, exist_ok=True)
    wanted, rows = set(chosen), {}
    for path in parquets:
        for batch in pq.ParquetFile(path).iter_batches(batch_size=16, columns=["img_id", "image"]):
            for r in batch.to_pylist():
                if r["img_id"] not in wanted:
                    continue
                data = r["image"]["bytes"]
                with Image.open(io.BytesIO(data)) as im:
                    width, height, fmt = im.width, im.height, im.format.lower()
                name = f"{r['img_id']}.{'jpg' if fmt == 'jpeg' else fmt}"
                (args.out_dir / "real" / name).write_bytes(data)  # unchanged bytes
                rows[r["img_id"]] = {"img_id": r["img_id"], "real_file": f"real/{name}",
                                     "has_face": int(r["img_id"] in with_face), "width": width, "height": height}
    pairs = []
    for n, img_id in enumerate(chosen):
        split = "val" if n < args.n_val else "train"
        pairs.append({"pair_id": f"pair_{n:04d}", "split": split, **rows[img_id]})
    return pairs


def pick_from_list(args):
    """Choose real photos from a CSV image list with a quota per source, copy them unchanged."""
    with open(args.list, newline="") as f:
        rows = [r for r in csv.DictReader(f) if r["label"] == "0"]
    rng = random.Random(args.seed)
    chosen = []
    for item in args.quota:
        source, count = item.split("=")
        pool = sorted((r for r in rows if r["source"] == source), key=lambda r: r["path"])
        chosen += rng.sample(pool, min(int(count), len(pool)))
    rng.shuffle(chosen)
    (args.out_dir / "real").mkdir(parents=True, exist_ok=True)
    pairs = []
    for n, r in enumerate(chosen):
        src = Path(r["path"])
        name = f"{src.parent.parent.name}_{src.name}"  # e.g. v2_commons_00001.jpg: names repeat across downloads
        (args.out_dir / "real" / name).write_bytes(src.read_bytes())  # unchanged bytes
        with Image.open(src) as im:
            width, height = im.size
        pairs.append({"pair_id": f"pair_{n:04d}", "split": "val" if n < args.n_val else "train", "img_id": Path(name).stem,
                      "real_file": f"real/{name}", "has_face": int(r["source"] in ("commons/people", "commons/portrait")),
                      "width": width, "height": height})
    return pairs


def caption(client, key, image_path):
    with Image.open(image_path) as im:
        im = im.convert("RGB")
        im.thumbnail((768, 768))  # smaller image = fewer tokens; enough for a description
        buffer = io.BytesIO()
        im.save(buffer, format="JPEG", quality=90)
    b64 = base64.b64encode(buffer.getvalue()).decode()
    # A description needs little thinking; reasoning tokens are billed like output tokens.
    body = {"model": MODEL, "reasoning_effort": "minimal", "messages": [{"role": "user", "content": [
        {"type": "text", "text": INSTRUCTION},
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
    ]}]}
    for attempt in range(5):
        response = client.post(API_URL, headers={"Authorization": f"Bearer {key}"}, json=body)
        if response.status_code != 429 and response.status_code < 500:
            return response
        time.sleep(15 * (attempt + 1))
    return response


def write_prompts(out_dir, pairs):
    """prompts.csv for scripts/make_muse_images.py: one prompt per captioned pair, at the real photo's aspect ratio."""
    with open(out_dir / "captions.csv", newline="", encoding="utf-8") as f:
        captions = {r["pair_id"]: r for r in csv.DictReader(f) if r["status"] == "ok"}
    with open(out_dir / "prompts.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "category", "style", "size", "prompt"], lineterminator="\n")
        writer.writeheader()
        for p in pairs:
            if p["pair_id"] in captions:
                writer.writerow({"id": p["pair_id"], "category": f"pair_{p['split']}/{'face' if p['has_face'] else 'noface'}",
                                 "style": "caption", "size": f"{p['width']}x{p['height']}",
                                 "prompt": captions[p["pair_id"]]["caption"]})
    print(f"wrote {len(captions)} prompts to {out_dir / 'prompts.csv'}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--parquet", help="glob of SID-Set training files")
    parser.add_argument("--faces", help="glob of face feature files of the same SID-Set files")
    parser.add_argument("--list", type=Path, help="instead of SID-Set: a CSV image list (e.g. Commons train photos)")
    parser.add_argument("--quota", action="append", default=[], help='with --list: "source=count", e.g. commons/people=150')
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--n-faces", type=int, default=300, help="photos with at least one face")
    parser.add_argument("--n-other", type=int, default=200, help="photos without a checkable face")
    parser.add_argument("--n-val", type=int, default=100, help="pairs kept for testing")
    parser.add_argument("--limit", type=int, help="caption at most this many photos")
    parser.add_argument("--spend", action="store_true", help="really call Muse Spark (costs credits)")
    parser.add_argument("--workers", type=int, default=1, help="requests at the same time")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    pairs_csv = args.out_dir / "pairs.csv"
    if pairs_csv.exists():
        with open(pairs_csv, newline="", encoding="utf-8") as f:
            pairs = list(csv.DictReader(f))
    else:
        pairs = pick_from_list(args) if args.list else pick_pairs(args)
        with open(pairs_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(pairs[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(pairs)
    print(f"{len(pairs)} pairs ({sum(p['split'] == 'val' for p in pairs)} val), "
          f"{sum(int(p['has_face']) for p in pairs)} with a face -> {pairs_csv}")

    captions_csv = args.out_dir / "captions.csv"
    done = set()
    if captions_csv.exists():
        with open(captions_csv, newline="", encoding="utf-8") as f:
            done = {r["pair_id"] for r in csv.DictReader(f) if r["status"] in ("ok", "skipped")}
    todo = [p for p in pairs if p["pair_id"] not in done][: args.limit]
    print(f"{len(done)} already captioned or skipped, {len(todo)} to caption")
    if not args.spend:
        print("dry run: nothing was sent. Add --spend to caption.")
        return

    key = api_key()
    new_file = not captions_csv.exists() or captions_csv.stat().st_size == 0  # an empty file needs the header too
    tokens_in = tokens_out = 0
    with (open(captions_csv, "a", newline="", encoding="utf-8") as out, httpx.Client(timeout=300) as client,
          ThreadPoolExecutor(args.workers) as pool):
        writer = csv.DictWriter(out, fieldnames=CAPTION_FIELDS, lineterminator="\n")
        if new_file:
            writer.writeheader()
        futures = {pool.submit(caption, client, key, args.out_dir / p["real_file"]): p for p in todo}
        for i, future in enumerate(as_completed(futures), 1):
            if future.cancelled():
                continue
            p, response = futures[future], future.result()  # each result is saved as soon as it arrives
            if response.status_code != 200:
                print(f"  [{i}/{len(todo)}] {p['pair_id']} error {response.status_code}: {response.text[:300]}")
                writer.writerow({"pair_id": p["pair_id"], "status": "error", "caption": response.text[:200]})
                if response.status_code in (400, 401, 402, 403):
                    print("stopping: this error will repeat for every photo")
                    for f in futures:
                        f.cancel()  # requests not started yet are dropped; running ones are still saved
                continue
            result = response.json()
            text = " ".join(result["choices"][0]["message"]["content"].split()).strip().strip('"')
            usage = result.get("usage", {})
            tokens_in += usage.get("prompt_tokens", 0)
            tokens_out += usage.get("completion_tokens", 0)
            status = "skipped" if text.upper().startswith("SKIP") else "ok"
            writer.writerow({"pair_id": p["pair_id"], "status": status, "caption": text,
                             "input_tokens": usage.get("prompt_tokens", ""),
                             "output_tokens": usage.get("completion_tokens", "")})
            out.flush()
            print(f"  [{i}/{len(todo)}] {p['pair_id']} {status} ({usage.get('prompt_tokens')} in, "
                  f"{usage.get('completion_tokens')} out): {text[:90]}")
        cost = tokens_in * PRICE_IN + tokens_out * PRICE_OUT
        print(f"this run: {tokens_in} input + {tokens_out} output tokens = ${cost:.3f} "
              f"(${cost / max(1, len(todo)):.4f} per photo)")
    write_prompts(args.out_dir, pairs)


if __name__ == "__main__":
    main()
