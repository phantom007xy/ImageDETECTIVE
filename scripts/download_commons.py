"""Download real photos from Wikimedia Commons: polished/professional ones and modern phone photos.

Why: our real training photos (OpenImages) are older amateur snapshots, while Muse Image fakes,
stock and news photos and modern phone photos look polished. Without polished real photos a head
can learn "polished = AI" (results/muse_pairs_v1_heads.txt).

Only files under CC0, CC BY (any version, no SA) or public domain; JPEG; at least 2 MP; taken in
2010 or later (to skip scans of old prints); not extremely wide or tall. At most --per-author
photos per photographer, so no single style dominates. The split goes by photographer: all photos
of one photographer are either "dev" (the polished dev set, never trained on) or "train".

Writes:
  - the images (1280 px thumbnails) to --out-dir/<split>/
  - data/attribution/commons_v1.csv (in the repo): author, licence and link of every photo, for the README
  - features/lists/commons_v1_train.csv and commons_v1_dev.csv (path, label 0, source)

Usage:
  python scripts/download_commons.py --out-dir D:/Datasets/Commons/v1
"""

import argparse
import csv
import re
import time
import zlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx

API = "https://commons.wikimedia.org/w/api.php"
USER_AGENT = "ImageDETECTIVE-research/0.1 (hackathon project; https://github.com/phantom007xy/ImageDETECTIVE)"
# (category, kind, how many photos to keep at most, also look one level into its subcategories)
PLANS = {
    "v1": [
        ("Quality images", "quality", 1000, False),
        ("Quality images of people", "people", 400, False),
        ("Portrait photographs", "portrait", 300, False),
        ("Taken with Google Pixel 7", "phone", 150, False),
        ("Taken with Google Pixel 6", "phone", 150, False),
        ("Taken with Samsung Galaxy S23 Ultra", "phone", 150, False),
        ("Taken with Xiaomi Redmi Note 10 Pro", "phone", 100, False),
    ],
    # v1 had too few polished people photos (most Commons portraits are CC BY-SA): more people sources.
    "v2": [
        ("Portrait photographs of men", "portrait", 500, False),
        ("Portrait photographs of women", "portrait", 500, False),
        ("Web Summit", "people", 300, True),
        ("Featured pictures of people", "people", 200, True),
        ("Quality images of people", "people", 200, True),
    ],
}
FIELDS = ["file", "split", "kind", "category", "title", "page_url", "author", "licence", "licence_url"]


def licence_ok(name):
    name = name.strip().lower()
    return name == "cc0" or name.startswith("public domain") or (name.startswith("cc by ") and "sa" not in name)


def plain(html):
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", "", html or "")).strip()


def candidates(client, category, kind, want, per_author, seen, author_counts, scan_limit=12000, give_up_after=1500):
    """Newest files of a category that pass every filter, at most `want` of them.

    Stops early after `give_up_after` files in a row without a usable one (e.g. a category whose
    single author already reached the per-author cap), instead of scanning it to the end.
    """
    params = {"action": "query", "format": "json", "generator": "categorymembers", "gcmtitle": f"Category:{category}",
              "gcmtype": "file", "gcmsort": "timestamp", "gcmdir": "desc", "gcmlimit": 50, "prop": "imageinfo",
              "iiprop": "url|size|mime|extmetadata", "iiurlwidth": 1280,
              "iiextmetadatafilter": "LicenseShortName|LicenseUrl|Artist|DateTimeOriginal"}
    found, scanned, last_keep = [], 0, 0
    while len(found) < want and scanned < scan_limit and scanned - last_keep < give_up_after:
        data = client.get(API, params=params).json()
        pages = sorted(data.get("query", {}).get("pages", {}).values(), key=lambda p: p["title"])
        scanned += len(pages)
        for page in pages:
            info = (page.get("imageinfo") or [{}])[0]
            meta = info.get("extmetadata", {})
            licence = plain(meta.get("LicenseShortName", {}).get("value", ""))
            author = plain(meta.get("Artist", {}).get("value", "")) or "unknown"
            year = re.search(r"\b(19|20)\d{2}\b", plain(meta.get("DateTimeOriginal", {}).get("value", "")))
            w, h = info.get("width", 0), info.get("height", 0)
            if (page["title"] in seen or info.get("mime") != "image/jpeg" or not licence_ok(licence)
                    or w * h < 2e6 or max(w, h) > 2.5 * min(w, h) or not year or int(year.group()) < 2010
                    or author_counts.get(author, 0) >= per_author or not info.get("thumburl")):
                continue
            seen.add(page["title"])
            author_counts[author] = author_counts.get(author, 0) + 1
            last_keep = scanned
            found.append({"title": page["title"], "page_url": info["descriptionurl"], "author": author,
                          "licence": licence, "licence_url": plain(meta.get("LicenseUrl", {}).get("value", "")),
                          "kind": kind, "category": category, "thumb": info["thumburl"]})
            if len(found) == want:
                break
        if "continue" not in data:
            break
        params.update(data["continue"])
        time.sleep(0.2)  # be gentle with the API
    print(f"  {category}: kept {len(found)} of {scanned} scanned", flush=True)
    return found


def subcategories(client, category):
    params = {"action": "query", "format": "json", "list": "categorymembers", "cmtitle": f"Category:{category}",
              "cmtype": "subcat", "cmlimit": 500}
    data = client.get(API, params=params).json()
    return [m["title"].removeprefix("Category:") for m in data.get("query", {}).get("categorymembers", [])]


def download(client, item, path):
    if path.exists():
        return True
    for attempt in range(4):
        response = client.get(item["thumb"])
        if response.status_code == 200:
            path.write_bytes(response.content)
            return True
        time.sleep(5 * (attempt + 1))
    return False


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--plan", choices=list(PLANS), default="v1")
    parser.add_argument("--per-author", type=int, default=20, help="across all plans downloaded so far")
    parser.add_argument("--dev-percent", type=int, default=15, help="share of photographers whose photos are dev")
    args = parser.parse_args()

    # Photos of earlier plans are skipped, and their photographers count towards the per-author cap.
    # A photographer keeps the split of an earlier plan, so no style is in both train and dev.
    seen, author_counts, author_split, items = set(), {}, {}, []
    for earlier in sorted(Path("data/attribution").glob("commons_v*.csv")):
        if earlier.stem != f"commons_{args.plan}":
            with open(earlier, newline="", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    seen.add(row["title"])
                    author_counts[row["author"]] = author_counts.get(row["author"], 0) + 1
                    author_split[row["author"]] = row["split"]

    with httpx.Client(headers={"User-Agent": USER_AGENT}, timeout=60, follow_redirects=True) as client:
        for category, kind, want, with_subcats in PLANS[args.plan]:
            if not with_subcats:
                items += candidates(client, category, kind, want, args.per_author, seen, author_counts)
                continue
            subs = [category] + subcategories(client, category)
            for sub in subs:  # spread the quota over the category and its subcategories
                items += candidates(client, sub, kind, max(1, want // len(subs)), args.per_author, seen, author_counts)

        for n, item in enumerate(items):
            by_hash = "dev" if zlib.crc32(item["author"].lower().encode()) % 100 < args.dev_percent else "train"
            item["split"] = author_split.get(item["author"], by_hash)
            item["file"] = f"{item['split']}/commons_{n:05d}.jpg"
        for split in ("train", "dev"):
            (args.out_dir / split).mkdir(parents=True, exist_ok=True)
        with ThreadPoolExecutor(2) as pool:  # two downloads at a time: polite to Wikimedia's servers
            ok = list(pool.map(lambda it: download(client, it, args.out_dir / it["file"]), items))
    items = [it for it, good in zip(items, ok) if good]

    attribution = Path(f"data/attribution/commons_{args.plan}.csv")
    attribution.parent.mkdir(parents=True, exist_ok=True)
    with open(attribution, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(items)
    for split in ("train", "dev"):
        rows = [{"path": str(args.out_dir / it["file"]), "label": 0, "source": f"commons/{it['kind']}"}
                for it in items if it["split"] == split]
        with open(f"features/lists/commons_{args.plan}_{split}.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["path", "label", "source"], lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
        kinds = {k: sum(it["kind"] == k for it in items if it["split"] == split) for k in ("quality", "people", "portrait", "phone")}
        print(f"{split}: {len(rows)} photos {kinds}")
    print(f"{len(ok) - len(items)} downloads failed | {len(author_counts)} photographers | attribution: {attribution}")


if __name__ == "__main__":
    main()
