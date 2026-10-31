"""Can Muse Spark help as an "investigator"? COSTS META API CREDITS (~$1.25 for the default sample).

For held-out Muse test edits (the edit prompt and the changed area are known) and real originals, Muse Spark
(muse-spark-1.3, standard mode) looks at the photo and answers in JSON: does it look AI-edited (yes/no + confidence),
and which one object or area was most likely added or changed (1-4 words, or "none"). SAM 3.1 then outlines that
guess (a text prompt), and the outline is compared with the true changed area of local edits.
Measured: (1) detection: AUC of Muse Spark's confidence, edits vs. real originals; (2) localisation: overlap (IoU)
of the outline with the true change, and "hit" = at least 30% of the outline inside the change; (3) the guess names
the added object of an "add a ..." prompt. Muse Spark never decides the app's verdict (plan Section 0).

Without --spend it only shows the sample and the maximum cost. Answers go to features/investigator/answers.csv.
Usage (from the repo root):
  python scripts/investigator_test.py
  python scripts/investigator_test.py --spend --workers 8
"""

import argparse
import base64
import csv
import io
import json
import random
import re
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root, so `src` can be imported
from src.eval import metrics  # noqa: E402
from src.services import meta_api  # noqa: E402
from src.services.sam_outline import output_text, parse_mask  # noqa: E402

SPARK_COST_MAX, SAM_COST = 0.004, 0.0025  # Muse Spark per photo (image + short answer, upper estimate); SAM 3.1
PROMPT = ("Look at this photo carefully. It may be an unedited camera photo, or a real photo that an AI image editor "
          "changed (for example by adding, removing or replacing an object, or by re-rendering the whole scene). "
          'Answer with JSON only, no other text: {"ai_edited": "yes" or "no", "confidence": a number from 0 to 100, '
          '"changed": "the single object or area most likely added or changed by AI, in 1 to 4 words, or none"}')
NOUNS = ["dog", "bicycle", "bike", "bird", "car", "bench", "plant", "pot"]  # the objects the "add a ..." test prompts add
OUT = Path("features/investigator/answers.csv")
FIELDS = ["key", "set", "prompt", "ai_edited", "confidence", "changed", "named_noun", "sam_iou", "sam_hit", "tokens_in", "tokens_out"]


def sample(seed):
    rng = random.Random(seed)
    items = []
    for version, folder, n_local, n_whole, n_orig in (("v2", "edits_test_v2", 100, 40, 99), ("v1", "edits_v1", 51, 0, 0)):
        rows = [r for r in csv.DictReader(open(f"D:/Datasets/MuseImage/{folder}/manifest.csv")) if r["status"] == "ok"]
        local = [r for r in rows if r["kind"] == "local"]
        whole = [r for r in rows if r["kind"] == "whole"]
        for r in rng.sample(local, min(n_local, len(local))):
            items.append({"key": f"{version}/{r['id']}", "set": f"{version} local", "prompt": r["prompt"],
                          "image": f"D:/Datasets/MuseImage/{folder}/{r['id']}_edited.png",
                          "mask": f"D:/Datasets/MuseImage/{folder}/{r['id']}_mask.png"})
        for r in rng.sample(whole, min(n_whole, len(whole))):
            items.append({"key": f"{version}/{r['id']}", "set": f"{version} whole", "prompt": r["prompt"],
                          "image": f"D:/Datasets/MuseImage/{folder}/{r['id']}_edited.png", "mask": None})
        originals = {}
        for r in rows:  # one original per photo
            originals.setdefault(r["source"], r)
        for r in rng.sample(list(originals.values()), min(n_orig, len(originals))):
            items.append({"key": f"{version}/{r['id']}_original", "set": f"{version} original", "prompt": "",
                          "image": f"D:/Datasets/MuseImage/{folder}/{r['id']}_original.jpg", "mask": None})
    return items


def jpeg_b64(image):
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=90)
    return base64.b64encode(buffer.getvalue()).decode()


def ask(key, item):
    image = Image.open(item["image"]).convert("RGB")
    image.thumbnail((1024, 1024))
    data = jpeg_b64(image)
    body = {"model": "muse-spark-1.3", "reasoning_effort": "low", "messages": [{"role": "user", "content": [
        {"type": "text", "text": PROMPT}, {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{data}"}}]}]}
    reply = meta_api.post(key, "/chat/completions", body)
    text = reply["choices"][0]["message"]["content"]
    usage = reply.get("usage", {})
    match = re.search(r"\{.*\}", text, re.S)
    answer = json.loads(match.group(0)) if match else {}
    row = {k: item[k] for k in ("key", "set", "prompt")}
    row.update({"ai_edited": str(answer.get("ai_edited", "")).lower(), "confidence": answer.get("confidence", ""),
                "changed": str(answer.get("changed", "")).strip(), "tokens_in": usage.get("prompt_tokens", ""),
                "tokens_out": usage.get("completion_tokens", "")})
    row["named_noun"] = int(any(n in row["changed"].lower() and n in item["prompt"] for n in NOUNS)) if "add" in item["prompt"] else ""
    if item["mask"] and row["changed"].lower() not in ("", "none"):
        body = {"model": "sam-3.1", "input": [{"type": "message", "role": "user", "content": [
            {"type": "input_text", "text": row["changed"]},
            {"type": "input_image", "image_url": "data:image/jpeg;base64," + data}]}]}
        found = parse_mask(output_text(meta_api.post(key, "/responses", body)), image.size)
        truth = np.asarray(Image.open(item["mask"]).convert("L").resize(image.size, Image.NEAREST)) > 0
        if found is None:
            row.update({"sam_iou": 0.0, "sam_hit": 0})
        else:
            inter = (found["mask"] & truth).sum()
            row.update({"sam_iou": round(float(inter / max(1, (found["mask"] | truth).sum())), 3),
                        "sam_hit": int(inter >= 0.3 * max(1, found["mask"].sum()))})
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--spend", action="store_true")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    items = sample(args.seed)
    n_sam = sum(1 for i in items if i["mask"])
    counts = Counter(i["set"] for i in items)
    print(f"{len(items)} photos ({', '.join(f'{s}: {n}' for s, n in sorted(counts.items()))}); at most "
          f"${len(items) * SPARK_COST_MAX + n_sam * SAM_COST:.2f} (Muse Spark ~${SPARK_COST_MAX}/photo, SAM 3.1 ${SAM_COST} on local edits)")
    if not args.spend:
        print("dry run: nothing was sent. Add --spend.")
        return
    key = meta_api.api_key()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(args.workers) as pool:
        rows = list(pool.map(lambda item: ask(key, item), items))
    with open(OUT, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    tokens_in = sum(int(r["tokens_in"] or 0) for r in rows)
    tokens_out = sum(int(r["tokens_out"] or 0) for r in rows)
    cost = tokens_in * 1.25e-6 + tokens_out * 4.25e-6 + sum(1 for r in rows if r.get("sam_iou", "") != "") * SAM_COST
    print(f"\nspent: {tokens_in} input + {tokens_out} output tokens + SAM 3.1 calls = ${cost:.2f}")

    def score(r):  # Muse Spark's AI-edited confidence on one scale: yes -> conf, no -> 100 - conf
        c = float(r["confidence"]) if str(r["confidence"]).replace(".", "", 1).isdigit() else 50.0
        return c if r["ai_edited"] == "yes" else 100 - c

    for group in ("v2", "v1"):
        orig = [score(r) for r in rows if r["set"] == f"{group} original"] or [score(r) for r in rows if r["set"] == "v2 original"]
        for kind in ("local", "whole"):
            edits = [score(r) for r in rows if r["set"] == f"{group} {kind}"]
            if edits:
                auc = metrics.auc(np.r_[orig, edits], np.r_[np.zeros(len(orig)), np.ones(len(edits))] == 1)
                said_yes = np.mean([r["ai_edited"] == "yes" for r in rows if r["set"] == f"{group} {kind}"])
                print(f"  {group} {kind:<5} n={len(edits):>3} | Muse Spark says AI-edited {said_yes:5.1%} | AUC vs real originals {auc:.3f}")
    print(f"  real originals: Muse Spark says AI-edited {np.mean([r['ai_edited'] == 'yes' for r in rows if 'original' in r['set']]):.1%}")
    local = [r for r in rows if "local" in r["set"] and r.get("sam_iou", "") != ""]
    if local:
        print(f"  local edits with a guess outlined by SAM 3.1: {len(local)} | mean IoU with the true change "
              f"{np.mean([float(r['sam_iou']) for r in local]):.2f} | hit (>=30% of the outline inside the change) "
              f"{np.mean([int(r['sam_hit']) for r in local]):.1%}")
    added = [r for r in rows if r["named_noun"] != ""]
    if added:
        print(f"  'add a ...' edits: the guess names the added object in {np.mean([int(r['named_noun']) for r in added]):.1%} of {len(added)}")


if __name__ == "__main__":
    main()
