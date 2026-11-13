"""Generate fakes with open text-to-image models from our pair captions (offline data prep, on Kaggle).

Why: a fake class with only FLUX and Muse Image is too narrow (results/generator_variety_v1.txt).
These models add two more architectures, from the same captions as the Muse pairs, so every fake
has a real photo with the same content.

Models (licences checked 31 Oct 2026):
  sd35m  = stabilityai/stable-diffusion-3.5-medium (Stability AI Community License: free below $1M
           revenue, we own the outputs, outputs must not train a *generative* foundation model;
           gated: accept the terms once on Hugging Face)
  pixart = PixArt-alpha/PixArt-Sigma-XL-2-1024-MS (OpenRAIL++, open). It uses SDXL's decoder (VAE),
           so its images are a dev test only, never training: SDXL must stay unseen.
  k22    = kandinsky-community/kandinsky-2-2-decoder + -prior (Apache 2.0, open); its own MoVQ decoder
No SDXL-based model for training: SDXL stays the generator the head has never seen.
Added 13 Nov 2026 (see MODELS): sana (Apache 2.0, DC-AE decoder), wuerstchen (MIT, VQGAN decoder), amused (OpenRAIL++,
VQ-VAE token model) for training variety; kolors (Apache 2.0, SDXL's VAE) as a dev set only.
Runtime rule: these are offline data-preparation tools only; the app never calls them.

Each model gets its own share of the prompts (--part / --parts), so every caption is used once.
Images keep the real photo's aspect ratio at about --megapixels, are saved as PNG to
--out-dir/<model>/, and every result is added to --out-dir/<model>/manifest.csv. Finished ids are
skipped, so a stopped run can simply be started again.

Usage (on Kaggle, see notebooks/kaggle_open_gen.ipynb):
  python scripts/generate_open_models.py --model pixart --prompts data/prompts/pairs_v1_prompts.csv
      data/prompts/pairs_v2_prompts.csv --part 0 --parts 2 --out-dir /kaggle/working/open_gen --device cuda:0
"""

import argparse
import csv
import math
import time
import zlib
from pathlib import Path

import numpy as np
import torch

MODELS = {
    "sd35m": {"repo": "stabilityai/stable-diffusion-3.5-medium", "steps": 28, "guidance": 4.5, "megapixels": 0.8},
    "pixart": {"repo": "PixArt-alpha/PixArt-Sigma-XL-2-1024-MS", "steps": 20, "guidance": 4.5, "megapixels": 0.8},
    # Added 13 Nov 2026 (after the hackathon; licences checked on Hugging Face that day). Each brings a decoder family
    # the head has not seen: Sana (DC-AE, Apache 2.0), Wuerstchen v2 (Paella VQGAN, MIT), aMUSEd (VQ-VAE token model,
    # OpenRAIL++, 512 px). Kolors (Apache 2.0) uses SDXL's VAE, so like PixArt it is a dev set only, never training.
    # "pipeline": the diffusers class; "variant": fp16 weights where they exist; "offload": CPU offload (16 GB GPUs);
    # "multiple": the size step the model needs; "call": extra call arguments; "plain_call": no guidance/steps names.
    "sana": {"repo": "Efficient-Large-Model/Sana_1600M_1024px_diffusers", "pipeline": "SanaPipeline", "variant": "fp16",
             "steps": 20, "guidance": 4.5, "megapixels": 0.8, "multiple": 32, "bf16_parts": ("vae", "text_encoder")},
    "wuerstchen": {"repo": "warp-ai/wuerstchen", "pipeline": "AutoPipelineForText2Image", "megapixels": 0.8, "multiple": 128,
                   "plain_call": True, "call": {"prior_guidance_scale": 4.0, "prior_num_inference_steps": 60,
                                                "num_inference_steps": 12, "decoder_guidance_scale": 0.0}},
    "amused": {"repo": "amused/amused-512", "pipeline": "AmusedPipeline", "variant": "fp16", "steps": 12, "guidance": 10.0,
               "megapixels": 0.26, "multiple": 16},
    "kolors": {"repo": "Kwai-Kolors/Kolors-diffusers", "pipeline": "KolorsPipeline", "variant": "fp16", "offload": True,
               "steps": 25, "guidance": 5.0, "megapixels": 0.8},
    # Kandinsky 2.2 was trained at 768 x 768, so it gets a smaller size.
    "k22": {"repo": "kandinsky-community/kandinsky-2-2-decoder", "steps": 30, "guidance": 4.0, "megapixels": 0.6},
}
FIELDS = ["id", "file", "width", "height", "seconds", "status"]


def load_pipeline(name, device):
    """fp16: Kaggle's T4 has no bf16. SD 3.5 needs CPU offload to fit 16 GB; PixArt fits as it is."""
    if name == "sd35m":
        from diffusers import StableDiffusion3Pipeline
        pipe = StableDiffusion3Pipeline.from_pretrained(MODELS[name]["repo"], torch_dtype=torch.float16)
        pipe.enable_model_cpu_offload(gpu_id=torch.device(device).index or 0)
    elif name == "k22":
        from diffusers import AutoPipelineForText2Image  # loads the prior and the decoder together
        pipe = AutoPipelineForText2Image.from_pretrained(MODELS[name]["repo"], torch_dtype=torch.float16).to(device)
    elif name == "pixart":
        from diffusers import PixArtSigmaPipeline
        pipe = PixArtSigmaPipeline.from_pretrained(MODELS[name]["repo"], torch_dtype=torch.float16).to(device)
    else:  # the generic entries (13 Nov): class name, fp16 variant, optional CPU offload, parts kept in bf16
        import diffusers
        m = MODELS[name]
        kwargs = {"torch_dtype": torch.float16}
        if m.get("variant"):
            kwargs["variant"] = m["variant"]
        pipe = getattr(diffusers, m["pipeline"]).from_pretrained(m["repo"], **kwargs)
        for part in m.get("bf16_parts", ()):  # e.g. Sana's VAE and text encoder overflow in fp16 (their own snippet)
            getattr(pipe, part).to(torch.bfloat16)
        if m.get("offload"):
            pipe.enable_model_cpu_offload(gpu_id=torch.device(device).index or 0)
        else:
            pipe.to(device)
    pipe.set_progress_bar_config(disable=True)
    return pipe


def target_size(size, megapixels, multiple=64):
    """The real photo's aspect ratio at about `megapixels`, both sides a multiple of `multiple`."""
    w, h = (int(v) for v in size.split("x"))
    scale = math.sqrt(megapixels * 1e6 / (w * h))
    return (max(256, round(w * scale / multiple) * multiple), max(256, round(h * scale / multiple) * multiple))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", choices=list(MODELS), required=True)
    parser.add_argument("--prompts", type=Path, nargs="+", required=True)
    parser.add_argument("--part", type=int, default=0, help="this model's share: rows with index %% parts == part")
    parser.add_argument("--parts", type=int, default=1)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--megapixels", type=float, help="image size; default: the model's own setting")
    parser.add_argument("--limit", type=int, help="generate at most this many new images (for a quick test)")
    args = parser.parse_args()

    rows = [r for path in args.prompts for r in csv.DictReader(open(path, newline="", encoding="utf-8"))]
    rows = [r for i, r in enumerate(rows) if i % args.parts == args.part]
    out = args.out_dir / args.model
    out.mkdir(parents=True, exist_ok=True)
    manifest = out / "manifest.csv"
    done = set()
    if manifest.exists() and manifest.stat().st_size > 0:
        with open(manifest, newline="") as f:
            done = {r["id"] for r in csv.DictReader(f) if r["status"] == "ok"}
    todo = [r for r in rows if r["id"] not in done][: args.limit]
    print(f"{args.model}: {len(rows)} prompts in this share, {len(done)} done, {len(todo)} to generate", flush=True)
    if not todo:
        return

    pipe = load_pipeline(args.model, args.device)
    settings = MODELS[args.model]
    new_file = not manifest.exists() or manifest.stat().st_size == 0
    with open(manifest, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS, lineterminator="\n")
        if new_file:
            writer.writeheader()
        for i, row in enumerate(todo, 1):
            width, height = target_size(row["size"], args.megapixels or settings["megapixels"], settings.get("multiple", 64))
            seed = zlib.crc32(row["id"].encode())  # the same image on every run
            start = time.time()
            call = dict(settings.get("call", {}))
            if not settings.get("plain_call"):
                call.update(num_inference_steps=settings["steps"], guidance_scale=settings["guidance"])
            image = pipe(row["prompt"], width=width, height=height, generator=torch.Generator("cpu").manual_seed(seed),
                         **call).images[0]
            # fp16 can overflow and give a black (or NaN) image: record it instead of using it.
            status = "ok" if np.asarray(image).mean() > 2 else "black"
            if status == "ok":
                image.save(out / f"{row['id']}.png")
            writer.writerow({"id": row["id"], "file": f"{row['id']}.png" if status == "ok" else "",
                             "width": image.width, "height": image.height,
                             "seconds": f"{time.time() - start:.1f}", "status": status})
            f.flush()
            if i <= 3 or i % 25 == 0:
                print(f"  {args.model} [{i}/{len(todo)}] {row['id']} {status} {image.width}x{image.height} "
                      f"{time.time() - start:.1f} s", flush=True)


if __name__ == "__main__":
    main()
