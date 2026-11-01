"""Google's image models (Imagen, Gemini "Nano Banana") on Vertex AI, for TRAINING and DEV data only (offline).
COSTS GOOGLE CLOUD CREDITS (the free-trial credits of my project); without --spend it only plans and shows the cost.

Why: Gemini pictures are our weakest generator among the popular tools (74% caught), and AI restorations or edits of
real photos made with tools other than Muse are mostly missed (19%; results/commons_ai_v1.txt). The app never calls
these models (runtime = Meta models only); they only make training and dev pictures.

Modes:
  t2i      pictures from our pair captions (data/prompts/pairs_v*_prompts.csv): the same content as a real photo, so
           real and AI differ only in how they were made. Train captions -> training; val captions -> dev only.
  edit     Nano Banana edits of people-free TRAINING photos (--list, e.g. the Muse training edits' originals), with
           the training edit prompts of scripts/make_muse_edits.py: original (real) next to its edit (AI).
  restore  an "old photo" made from a people-free training photo (grey or sepia, soft, grainy, small, low-quality JPEG)
           goes to Nano Banana with "restore and colourise"; its output (AI) is paired with the real colour original.
Outputs (never git): --out-dir/<model>/ with manifest.csv (as scripts/generate_open_models.py, so
scripts/make_open_gen_lists.py builds the t2i lists), and for edit/restore features/lists/google_<mode>_<model>_path_a.csv
(original 0, edit resized to the original's size 1), like the Muse training edits.

Setup (once): Vertex AI API enabled in the project; my own Google login on this laptop instead of a key file (keys are
blocked by my organisation's "secure by default" policy): Google Cloud CLI, then gcloud auth application-default login
and gcloud auth application-default set-quota-project <project id>; pip install google-genai. --project or
GOOGLE_CLOUD_PROJECT in .env names the project. Nothing secret is in the repo.

Usage:
  python scripts/make_google_images.py --mode t2i --model imagen-4.0-fast-generate-001 --n 400 --out-dir D:/Datasets/GoogleGen
  python scripts/make_google_images.py --mode restore --model gemini-2.5-flash-image --list ... --n 150 --out-dir ...
  (add --spend to really make them)
"""

import argparse
import csv
import io
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image, ImageFilter, ImageOps

PRICE = {"imagen-4.0-fast": 0.02, "imagen-4.0-generate": 0.04, "imagen-4.0-ultra": 0.06, "gemini": 0.039}
ASPECTS = {"1:1": 1.0, "4:3": 4 / 3, "3:4": 3 / 4, "16:9": 16 / 9, "9:16": 9 / 16}
FIELDS = ["id", "file", "width", "height", "seconds", "status"]
EDIT_PROMPTS = ["add a black cat sitting somewhere in the scene", "add a white delivery van", "add a yellow umbrella",
                "add a stack of cardboard boxes", "add a small tree", "add a basket of fruit", "add a motorcycle",
                "remove the largest object and fill in the background naturally", "add a red balloon",
                "change the colour of the most prominent object to bright red", "add a cup of coffee",
                "make it look like a foggy morning", "change the season to autumn", "make it look like it is snowing",
                "replace the sky with a dramatic sunset sky", "make the lighting warmer and softer"]
RESTORE_PROMPT = ("Restore this old, damaged photograph: remove the grain and blur, sharpen the details and colourise it "
                  "with natural, realistic colours. Keep the same scene and framing.")


def price(model):
    return next((p for key, p in PRICE.items() if model.startswith(key)), PRICE["gemini"])


def old_photo(image, rng):
    """A real photo made to look old: grey or sepia, soft, grainy, small, low-quality JPEG."""
    img = ImageOps.grayscale(image)
    if rng.random() < 0.5:
        img = ImageOps.colorize(img, black=(40, 26, 13), white=(250, 236, 205))
    img = img.convert("RGB").filter(ImageFilter.GaussianBlur(rng.uniform(0.8, 1.8)))
    img.thumbnail((rng.randint(420, 640),) * 2)
    noise = Image.effect_noise(img.size, rng.uniform(18, 35)).convert("RGB")
    img = Image.blend(img, noise, 0.12)
    buffer = io.BytesIO()
    img.save(buffer, "JPEG", quality=rng.randint(30, 50))
    return Image.open(io.BytesIO(buffer.getvalue())).convert("RGB")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=["t2i", "edit", "restore"], required=True)
    parser.add_argument("--model", required=True, help="e.g. imagen-4.0-fast-generate-001, gemini-2.5-flash-image")
    parser.add_argument("--prompts", nargs="+", default=["data/prompts/pairs_v1_prompts.csv", "data/prompts/pairs_v2_prompts.csv"])
    parser.add_argument("--list", type=Path, help="edit/restore: CSV of people-free TRAINING photos (path, label 0)")
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--backend", choices=["vertex", "studio"], default="vertex",
                        help="vertex: Google Cloud (billing); studio: a Gemini API key in .env (GEMINI_API_KEY; free tier: "
                             "Google may use what is sent, so only captions and public photos, never my own photos)")
    parser.add_argument("--project", help="Google Cloud project id (default: GOOGLE_CLOUD_PROJECT in .env)")
    parser.add_argument("--location", default="us-central1")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--spend", action="store_true")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    if args.mode == "t2i":
        rows = [r for p in args.prompts for r in csv.DictReader(open(p, newline="", encoding="utf-8"))]
        jobs = [{"id": r["id"], "prompt": r["prompt"], "size": r["size"]} for r in rng.sample(rows, min(args.n, len(rows)))]
    else:
        with open(args.list, newline="") as f:
            photos = sorted({r["path"] for r in csv.DictReader(f) if r["label"] == "0"})
        jobs = [{"id": f"{args.mode}_{i:04d}", "source": p, "prompt": RESTORE_PROMPT if args.mode == "restore" else rng.choice(EDIT_PROMPTS)}
                for i, p in enumerate(rng.sample(photos, min(args.n, len(photos))))]
    out = args.out_dir / (args.model if args.mode == "t2i" else f"{args.mode}_{args.model}")
    out.mkdir(parents=True, exist_ok=True)
    manifest = out / "manifest.csv"
    done = set()
    if manifest.exists() and manifest.stat().st_size > 0:
        with open(manifest, newline="") as f:
            done = {r["id"] for r in csv.DictReader(f) if r["status"] in ("ok", "refused")}
    todo = [j for j in jobs if j["id"] not in done]
    print(f"{args.mode} with {args.model}: {len(jobs)} planned, {len(todo)} to do -> at most ${len(todo) * price(args.model):.2f}")
    if not args.spend:
        print("dry run: nothing was sent. Add --spend to make them.")
        return

    from google import genai  # noqa: E402
    from google.genai import types  # noqa: E402
    if Path(".env").exists():  # the key path and the project id live in .env, never in git
        for line in Path(".env").read_text().splitlines():
            name, _, value = line.partition("=")
            if name.strip() in ("GOOGLE_APPLICATION_CREDENTIALS", "GOOGLE_CLOUD_PROJECT", "GEMINI_API_KEY"):
                os.environ.setdefault(name.strip(), value.strip().strip('"').strip("'"))
    if args.backend == "studio":
        if args.list and "private" in str(args.list).lower():
            raise SystemExit("the free tier may use what is sent: never send my own photos with --backend studio")
        client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    else:
        client = genai.Client(vertexai=True, project=args.project or os.environ["GOOGLE_CLOUD_PROJECT"], location=args.location)

    def make(job):
        start = time.time()
        try:
            if args.model.startswith("imagen"):
                w, h = (int(v) for v in job["size"].split("x"))
                aspect = min(ASPECTS, key=lambda a: abs(ASPECTS[a] - w / h))
                result = client.models.generate_images(model=args.model, prompt=job["prompt"], config=types.GenerateImagesConfig(
                    number_of_images=1, aspect_ratio=aspect, person_generation="allow_adult"))
                if not result.generated_images:
                    return job, None, "refused"
                data = result.generated_images[0].image.image_bytes
            else:
                contents = [job["prompt"]]
                if args.mode != "t2i":
                    with Image.open(job["source"]) as im:
                        source = ImageOps.exif_transpose(im).convert("RGB")
                    job["original"] = source
                    sent = old_photo(source, random.Random(job["id"])) if args.mode == "restore" else source
                    sent.thumbnail((1536, 1536))
                    contents = [sent, job["prompt"]]
                result = client.models.generate_content(model=args.model, contents=contents,
                                                        config=types.GenerateContentConfig(response_modalities=["IMAGE"]))
                parts = [p for c in (result.candidates or []) for p in (c.content.parts if c.content else []) if p.inline_data]
                if not parts:
                    return job, None, "refused"
                data = parts[0].inline_data.data
            return job, Image.open(io.BytesIO(data)).convert("RGB"), f"{time.time() - start:.1f}"
        except Exception as error:  # recorded and retried on the next run (only ok/refused count as done)
            return job, None, f"error {type(error).__name__}: {str(error)[:80]}"

    new_file = not manifest.exists() or manifest.stat().st_size == 0
    path_a = []
    with open(manifest, "a", newline="") as f, ThreadPoolExecutor(args.workers) as pool:
        writer = csv.DictWriter(f, fieldnames=FIELDS, lineterminator="\n")
        if new_file:
            writer.writeheader()
        for job, image, info in pool.map(make, todo):
            status = "ok" if image is not None else ("refused" if info == "refused" else "error")
            if image is not None:
                image.save(out / f"{job['id']}.png")
                if args.mode != "t2i":  # pairs for training: the real original next to the edit at its size
                    job["original"].save(out / f"{job['id']}_original.jpg", quality=95)
                    image.resize(job["original"].size, Image.BICUBIC).save(out / f"{job['id']}_edited_matched.png")
            writer.writerow({"id": job["id"], "file": f"{job['id']}.png" if image is not None else "",
                             "width": image.width if image else "", "height": image.height if image else "",
                             "seconds": info if image is not None else "", "status": status})
            f.flush()
            print(f"  {job['id']} {status} {'' if image is not None else info}", flush=True)
    if args.mode != "t2i":
        with open(manifest, newline="") as f:
            made = [r["id"] for r in csv.DictReader(f) if r["status"] == "ok"]
        rows = [{"path": str(out / f"{i}_original.jpg"), "label": 0, "source": f"google_{args.mode}/original"} for i in made]
        rows += [{"path": str(out / f"{i}_edited_matched.png"), "label": 1, "source": f"google_{args.mode}/{args.model}"} for i in made]
        name = Path("features/lists") / f"google_{args.mode}_{args.model.replace('.', '_')}_path_a.csv"
        with open(name, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["path", "label", "source"], lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
        print(f"wrote {name}: {len(made)} pairs")


if __name__ == "__main__":
    main()
