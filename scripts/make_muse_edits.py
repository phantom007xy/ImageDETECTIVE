"""Muse Image EDITS of real photos (types c and c+), for testing or training. COSTS META API CREDITS: $0.01 per edit.

Source photos: the Commons polished dev set (CC0 / CC BY / public domain; never trained on), only
photos with NO person in them (scripts/find_people_free.py: SAM 3, prompt "person"), so no real person
is ever edited (plan Section 0). A face filter alone is not enough: people seen from behind or far
away have no checkable face (found on 31 Oct when a first test edited a meeting photo).
Edits: about 2/3 local ("add a dog", "remove the bicycle": type c) and 1/3 whole-image ("make it a
snowy winter day": type c+). For every pair the script measures how much of the picture the edit
clearly changed (pixel difference). Measured 31 Oct: Muse re-renders every pixel of an edit, also
outside the asked-for change (fine detail there correlates 0.59 with the original, vs. 0.98 for a plain
resize), and returns ~2.5 MP whatever size was sent.

Without --spend it only picks the photos and shows the cost.
  --prompts test:  a held-out test set from DEV photos, never used for training.
  --prompts train: training edits from TRAINING photos, with a different, broader prompt list.

Usage:
  python scripts/find_people_free.py --list ... --out features/lists/commons_dev_no_people.csv
  python scripts/make_muse_edits.py --list features/lists/commons_dev_no_people.csv
      --out-dir D:/Datasets/MuseImage/edits_v1 --n 40 --spend --workers 4
  python scripts/make_muse_edits.py --list features/lists/commons_train_no_people.csv --prompts train
      --edits-per-photo 3 --n 9999 --out-dir D:/Datasets/MuseImage/edits_train_commons --spend --workers 4
  python scripts/make_muse_edits.py --list D:/Datasets/SelfEdits/originals --prompts person --edits-per-photo 2
      --n 20 --out-dir D:/Datasets/SelfEdits/self_edits_v1 --spend   (my own selfies, test only)
  The image list is written to features/lists/muse_<out-dir name>.csv, so every out-dir needs its own name.
"""

import argparse
import base64
import csv
import io
import random
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter, ImageOps

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root, so `src` can be imported
from src.services import meta_api  # noqa: E402

COST = 0.01
SEND_LONG_SIDE = 1536
# Test prompts (held-out edit set) and a different, broader training list, so the test also shows
# whether Path B finds kinds of edits it never trained on.
PROMPTS = {
    "test": {
        "local": ["add a small brown dog sitting in the foreground", "add a red bicycle somewhere in the scene",
                  "add a few birds flying in the sky", "remove the most prominent object in the foreground and "
                  "fill in the background naturally", "add a parked blue car", "add a wooden bench",
                  "add a potted plant", "remove one object from the scene and fill in the background naturally"],
        "whole": ["make it look like a snowy winter day", "turn it into a night scene with warm street lights",
                  "make it look like a rainy evening", "make it look like it was taken at golden hour sunset"],
    },
    "train": {
        "local": ["add a black cat sitting somewhere in the scene", "add a white delivery van", "add a yellow umbrella",
                  "add a stack of cardboard boxes", "add a street lamp", "add a small tree", "add a basket of fruit",
                  "add a motorcycle", "add a flock of pigeons on the ground", "add a fire hydrant",
                  "add a hot air balloon in the sky", "remove the largest object and fill in the background naturally",
                  "remove a small object and fill in the background naturally", "replace one object with a large vase",
                  "change the colour of the most prominent object to bright red", "add a shop sign with no text",
                  "add a puddle of water on the ground", "add a laptop on a table or surface", "add a horse",
                  "add some fallen autumn leaves on the ground"],
        "whole": ["make it look like a foggy morning", "make it look like a hot summer midday",
                  "turn it into a dusk scene with a purple sky", "make it look like it is snowing heavily",
                  "make it look like after heavy rain with wet surfaces", "make it look like a cloudy winter afternoon",
                  "replace the sky with a dramatic sunset sky", "change the season to autumn",
                  "make it look like spring with blooming flowers", "make the lighting warmer and softer",
                  "change the background to a sandy beach", "change the background to a busy city street",
                  "make it look like a sunny day with a clear blue sky", "make it look like it was taken at night"],
    },
    # Test only, for my own selfies (D:/Datasets/SelfEdits/originals; nobody else in the frame; never published):
    # what a judge might do to their own photo.
    "person": {
        "local": ["add sunglasses", "change the colour of my shirt to bright red", "add a cap on my head",
                  "add a small dog next to me", "make me smile", "remove the object behind me and fill in the background"],
        "whole": ["change the background to a sandy beach at sunset", "put me in a busy city street at night",
                  "make it look like it is snowing", "change the background to a mountain landscape",
                  "make it look like a professional studio portrait"],
    },
}
FIELDS = ["id", "source", "kind", "prompt", "width", "height", "edited_width", "edited_height", "changed_share", "status"]


def changed_mask(original, edited):
    """Where the colour clearly changed (edited resized to the original's size), as a bool array."""
    a = np.asarray(original, dtype=np.float32)
    b = np.asarray(edited.resize(original.size, Image.BICUBIC), dtype=np.float32)
    diff = Image.fromarray(np.abs(a - b).mean(axis=2).clip(0, 255).astype(np.uint8)).filter(ImageFilter.MedianFilter(5))
    return np.asarray(diff) > 25


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--list", type=Path, required=True,
                        help="photos without people (scripts/find_people_free.py), or a folder (--prompts person)")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--n", type=int, default=40, help="photos to edit")
    parser.add_argument("--edits-per-photo", type=int, default=1, help="different prompts per photo")
    parser.add_argument("--prompts", choices=list(PROMPTS), default="test")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--spend", action="store_true")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if args.list.is_dir():  # a folder of photos, e.g. my selfies for the person-edit test
        photos = sorted(str(p) for p in args.list.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp"))
    else:
        with open(args.list, newline="") as f:
            photos = sorted(r["path"] for r in csv.DictReader(f) if r["label"] == "0")
    rng = random.Random(args.seed)
    picked = rng.sample(photos, min(args.n, len(photos)))
    jobs = []
    for i, path in enumerate(picked):
        used = set()
        for e in range(args.edits_per_photo):
            n = i * args.edits_per_photo + e
            kind = "whole" if n % 3 == 2 else "local"
            choices = [p for p in PROMPTS[args.prompts][kind] if p not in used]
            prompt = rng.choice(choices)
            used.add(prompt)
            jobs.append({"id": f"edit_{n:04d}", "source": path, "kind": kind, "prompt": prompt})

    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest = args.out_dir / "manifest.csv"
    done = set()
    if manifest.exists() and manifest.stat().st_size > 0:
        with open(manifest, newline="") as f:
            done = {r["id"] for r in csv.DictReader(f) if r["status"] in ("ok", "refused")}
    todo = [j for j in jobs if j["id"] not in done][: args.limit]
    print(f"{len(photos)} photos without people; {len(jobs)} edits planned ({sum(j['kind'] == 'local' for j in jobs)} "
          f"local, {sum(j['kind'] == 'whole' for j in jobs)} whole-image); {len(todo)} to do -> at most ${len(todo) * COST:.2f}")
    if not args.spend:
        print("dry run: nothing was sent. Add --spend to edit.")
        return
    key = meta_api.api_key()

    def edit(job):
        with Image.open(job["source"]) as im:
            original = ImageOps.exif_transpose(im).convert("RGB")
        original.thumbnail((SEND_LONG_SIDE, SEND_LONG_SIDE))
        buffer = io.BytesIO()
        original.save(buffer, format="JPEG", quality=95)
        original.save(args.out_dir / f"{job['id']}_original.jpg", quality=95)  # exactly what was sent (one per edit)
        body = {"model": "muse-image-1.0", "prompt": job["prompt"], "n": 1, "response_format": "b64_json",
                "output_format": "png", "images": [{"image_url": "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode()}],
                "tool_enablement": {"enable_image_search": False, "enable_web_search": False, "enable_shell": False}}
        response = meta_api.client().post("/images/edits", headers={"Authorization": f"Bearer {key}"}, json=body)
        row = {**job, "width": original.width, "height": original.height}
        if response.status_code != 200:
            code = response.json().get("error", {}).get("code", "") if "json" in response.headers.get("content-type", "") else ""
            return {**row, "status": "refused" if code == "content_policy_violation" else f"error {response.status_code}"}
        data = base64.b64decode(response.json()["data"][0]["b64_json"])
        (args.out_dir / f"{job['id']}_edited.png").write_bytes(data)
        with Image.open(io.BytesIO(data)) as im:
            edited = im.convert("RGB")
        mask = changed_mask(original, edited)
        # The approximate edited area, at the edited picture's size (the "true region" for Path B's test).
        Image.fromarray(mask.astype(np.uint8) * 255).resize(edited.size, Image.NEAREST).save(
            args.out_dir / f"{job['id']}_mask.png")
        # For training: the edit at the original's size (Muse returns ~2.5 MP), so "bigger = AI" can't be learned.
        edited.resize(original.size, Image.BICUBIC).save(args.out_dir / f"{job['id']}_edited_matched.png")
        return {**row, "edited_width": edited.width, "edited_height": edited.height,
                "changed_share": round(float(mask.mean()), 4), "status": "ok"}

    new_file = not manifest.exists() or manifest.stat().st_size == 0
    with open(manifest, "a", newline="") as out, ThreadPoolExecutor(args.workers) as pool:
        writer = csv.DictWriter(out, fieldnames=FIELDS, lineterminator="\n")
        if new_file:
            writer.writeheader()
        for future in as_completed([pool.submit(edit, j) for j in todo]):
            row = future.result()
            writer.writerow({k: row.get(k, "") for k in FIELDS})
            out.flush()
            print(f"  {row['id']} {row['kind']:<5} {row['status']:<8} changed {row.get('changed_share', '')} | {row['prompt'][:50]}",
                  flush=True)

    # Image list for the tests: originals (real, 0), local edits (2, with mask), whole-image edits (1).
    with open(manifest, newline="") as f:
        made = [r for r in csv.DictReader(f) if r["status"] == "ok"]
    rows = []
    for r in made:
        rows.append({"path": str(args.out_dir / f"{r['id']}_original.jpg"), "label": 0, "source": "muse_edits/original", "mask": ""})
        local = r["kind"] == "local"
        rows.append({"path": str(args.out_dir / f"{r['id']}_edited.png"), "label": 2 if local else 1,
                     "source": f"muse_edits/{r['kind']}", "mask": str(args.out_dir / f"{r['id']}_mask.png") if local else ""})
    out_list = Path("features/lists") / f"muse_{args.out_dir.name}.csv"
    with open(out_list, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["path", "label", "source", "mask"], lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    # Path A training list: Muse re-renders EVERY pixel of an edit, also outside the asked-for change
    # (31 Oct, results/muse_edits_v1.txt), so the whole edited picture is AI (1) next to its original (0).
    # Edits at the original's size (the first test set, edits_v1, was made before this file existed).
    path_a = [{"path": str(args.out_dir / f"{r['id']}_original.jpg"), "label": 0, "source": "muse_edits/original"}
              for r in made]
    path_a += [{"path": str(args.out_dir / f"{r['id']}_edited_matched.png"), "label": 1, "source": f"muse_edits/{r['kind']}"}
               for r in made if (args.out_dir / f"{r['id']}_edited_matched.png").exists()]
    path_a_list = Path("features/lists") / f"muse_{args.out_dir.name}_path_a.csv"
    with open(path_a_list, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["path", "label", "source"], lineterminator="\n")
        writer.writeheader()
        writer.writerows(path_a)
    print(f"wrote {out_list}: {len(made)} originals, {sum(r['kind'] == 'local' for r in made)} local and "
          f"{sum(r['kind'] == 'whole' for r in made)} whole-image edits; Path A list {path_a_list}")


if __name__ == "__main__":
    main()
