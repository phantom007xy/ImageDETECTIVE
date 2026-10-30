"""Generate Muse Image fakes from a prompt list. COSTS META API CREDITS: $0.01 per image.

Without --spend it only shows the plan and the maximum cost. With --spend it calls Meta's image
API in standard mode (model muse-image-1.0; Contributor mode would need a "-contributor" model ID,
and in standard mode Meta does not train on our content), saves each image unchanged and adds a
row to manifest.csv. Images already done (or refused) are skipped, so a re-run never pays twice.

The built-in search tools are switched off, so every image comes from the generator itself and
not from real photos it found online. The API key is read from MODEL_API_KEY in the environment
or in .env, and is never printed.

Usage:
  python scripts/make_muse_images.py --prompts data/prompts/muse_image_test_v1.csv --out-dir D:/Datasets/MuseImage/test_v1
  python scripts/make_muse_images.py --prompts ... --out-dir ... --limit 5 --spend
"""

import argparse
import base64
import csv
import io
import os
import time
from pathlib import Path

import httpx
from PIL import Image

API_URL = "https://api.meta.ai/v1/images/generations"
MODEL = "muse-image-1.0"
PRICE_PER_IMAGE = 0.01
FIELDS = ["id", "category", "style", "size_requested", "file", "width", "height", "seconds", "status", "error"]


def api_key():
    if os.environ.get("MODEL_API_KEY"):
        return os.environ["MODEL_API_KEY"]
    env = Path(".env")
    if env.exists():
        for line in env.read_text().splitlines():
            name, _, value = line.partition("=")
            if name.strip() == "MODEL_API_KEY":
                return value.strip().strip('"').strip("'")
    raise SystemExit("MODEL_API_KEY not found: add a line MODEL_API_KEY=... to .env")


def generate(client, key, row):
    """One API call (with retries on rate limits and server errors). Returns the HTTP response."""
    body = {
        "model": MODEL,
        "prompt": row["prompt"],
        "n": 1,
        "response_format": "b64_json",
        "output_format": "png",
        "tool_enablement": {"enable_image_search": False, "enable_web_search": False, "enable_shell": False},
        "user": "imagedetective-dataprep",
    }
    if row["size"]:
        body["size"] = row["size"]
    for attempt in range(5):
        response = client.post(API_URL, headers={"Authorization": f"Bearer {key}"}, json=body)
        if response.status_code != 429 and response.status_code < 500:
            return response
        time.sleep(15 * (attempt + 1))
    return response


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--prompts", type=Path, required=True, help="CSV with id, category, style, size, prompt")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int, help="generate at most this many new images")
    parser.add_argument("--spend", action="store_true", help="really call the API (costs credits)")
    args = parser.parse_args()

    with open(args.prompts, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    manifest = args.out_dir / "manifest.csv"
    done = set()
    if manifest.exists():
        with open(manifest, newline="", encoding="utf-8") as f:
            done = {r["id"] for r in csv.DictReader(f) if r["status"] in ("ok", "refused")}
    todo = [r for r in rows if r["id"] not in done][: args.limit]
    print(f"{len(rows)} prompts, {len(done)} already done or refused, {len(todo)} to generate "
          f"-> at most ${len(todo) * PRICE_PER_IMAGE:.2f}")
    if not args.spend:
        print("dry run: nothing was sent. Add --spend to generate.")
        return

    key = api_key()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    new_file = not manifest.exists()
    with open(manifest, "a", newline="", encoding="utf-8") as out, httpx.Client(timeout=300) as client:
        writer = csv.DictWriter(out, fieldnames=FIELDS, lineterminator="\n")
        if new_file:
            writer.writeheader()
        for i, row in enumerate(todo, 1):
            start = time.time()
            response = generate(client, key, row)
            record = {"id": row["id"], "category": row["category"], "style": row["style"],
                      "size_requested": row["size"], "seconds": f"{time.time() - start:.1f}"}
            if response.status_code == 200:
                data = base64.b64decode(response.json()["data"][0]["b64_json"])
                with Image.open(io.BytesIO(data)) as im:
                    width, height, fmt = im.width, im.height, im.format.lower()
                path = args.out_dir / f"{row['id']}.{fmt}"
                path.write_bytes(data)  # saved exactly as returned; re-encoding happens in preprocessing
                record.update(file=path.name, width=width, height=height, status="ok")
            else:
                error = response.json().get("error", {}) if "json" in response.headers.get("content-type", "") else {}
                code = error.get("code") or str(response.status_code)
                record.update(status="refused" if code == "content_policy_violation" else "error",
                              error=f"{response.status_code} {code}: {error.get('message', '')[:200]}")
            writer.writerow(record)
            out.flush()
            print(f"  [{i}/{len(todo)}] {row['id']} {record['status']} {record.get('width', '')}x"
                  f"{record.get('height', '')} {record['seconds']} s {record.get('error', '')}")
            if record["status"] == "error" and response.status_code in (400, 401, 402, 403):
                print("stopping: this error will repeat for every prompt (check the key, credits or request)")
                break


if __name__ == "__main__":
    main()
