"""Download photos from OpenForensics: real photos in which some faces were replaced by AI-made faces (face swaps in
still photos), with a box and a real/fake label for every face.

OpenForensics (Le, Nguyen, Yamagishi, Echizen; ICCV 2021), Zenodo record 5528418, CC BY 4.0:
https://zenodo.org/records/5528418. Photos from Google Open Images; in each photo, faces were replaced by GAN-made
faces and blended in; the other faces stay real. So one photo gives real AND fake faces with the same camera, light
and compression: the only difference is the face itself.

Why: our face-swap check learned only from DFDC video frames (2019-20 swaps of paid actors) and is weak
(results/dfdc_face_v1.txt). Swaps in still photos may look different from video swaps.

How: the zips are big (Val 3.1 GB, Test-Dev 2 x ~4 GB, Train 5 x ~4.5 GB), Zenodo is slow per connection and allows
about 133 requests a minute. So:
  - all photos of a zip (--n 0): the zip is downloaded in 32 MB pieces over several connections, then read on disk;
  - a sample (--n N): only the zip's table of contents (at its end) and the sampled photos are fetched with HTTP range
    requests (as scripts/download_sfhq.py), at most 2 requests a second.
Every photo is checked against the zip's CRC; Zenodo sometimes cuts a transfer short.

Writes:
  - the photos to --data-dir/<split>/ (never git)
  - features/lists/openforensics_<split>.csv: path, label (1 = at least one fake face), source openforensics/<split>
    (per-face labels stay in <split>_poly.json; src/features/extract_openforensics_faces.py uses them)

Usage (from the repo root):
  python scripts/download_openforensics.py --split Val --n 0
  python scripts/download_openforensics.py --split Test-Dev --n 1500
"""

import argparse
import csv
import io
import json
import random
import threading
import time
import zipfile
import zlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx

from download_sfhq import RemoteFile, member  # same folder: HTTP range reads of a zip

RECORD = "https://zenodo.org/records/5528418/files"
PARTS = {"Val": ["Val.zip"], "Test-Dev": ["Test-Dev_part_1.zip", "Test-Dev_part_2.zip"],
         "Test-Challenge": [f"Test-Challenge_part_{i}.zip" for i in range(1, 6)],
         "Train": [f"Train_part_{i}.zip" for i in range(1, 6)]}
PIECE = 32 << 20
_lock, _next = threading.Lock(), [0.0]


GAP = [0.5]  # seconds between requests (--gap)


def pace():
    """Wait for our turn: at most one request every GAP seconds over all threads (Zenodo: ~133 a minute)."""
    with _lock:
        wait = _next[0] - time.time()
        _next[0] = max(_next[0], time.time()) + GAP[0]
    if wait > 0:
        time.sleep(wait)


def annotations(path):
    """Photo name -> number of fake faces, from the COCO-style annotation file (category 1 = Fake)."""
    data = json.loads(path.read_text())
    name_of = {img["id"]: Path(img["file_name"]).name for img in data["images"]}
    fakes = {name: 0 for name in name_of.values()}
    for a in data["annotations"]:
        fakes[name_of[a["image_id"]]] += a["category_id"] == 1
    return fakes


def download_zip(client, url, path, workers):
    """The whole zip in 32 MB pieces over several connections (one request per piece)."""
    if path.exists():
        return
    pace()
    size = int(client.get(url, headers={"Range": "bytes=0-0"}).headers["content-range"].split("/")[1])
    tmp = path.with_suffix(".part")
    with open(tmp, "wb") as f:
        f.truncate(size)
    pieces = [(a, min(a + PIECE, size) - 1) for a in range(0, size, PIECE)]

    def get(piece):
        a, b = piece
        for attempt in range(6):
            try:
                pace()
                r = client.get(url, headers={"Range": f"bytes={a}-{b}"})
                if r.status_code == 206 and len(r.content) == b - a + 1:
                    with open(tmp, "r+b") as f:
                        f.seek(a)
                        f.write(r.content)
                    return True
            except httpx.HTTPError:
                pass
            time.sleep(10 * (attempt + 1))
        return False

    start = time.time()
    with ThreadPoolExecutor(workers) as pool:
        for done, ok in enumerate(pool.map(get, pieces), 1):
            if not ok:
                raise SystemExit(f"{path.name}: a piece failed 6 times; run again")
            if done % 10 == 0:
                print(f"  {path.name}: {done}/{len(pieces)} pieces, {done * PIECE / (time.time() - start) / 1e6:.1f} MB/s",
                      flush=True)
    tmp.rename(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", choices=list(PARTS), required=True)
    parser.add_argument("--n", type=int, default=1500, help="photos to sample (0 = all: whole zips)")
    parser.add_argument("--parts", nargs="*", help="only these zips of the split (default: all)")
    parser.add_argument("--data-dir", type=Path, default=Path("D:/Datasets/OpenForensics"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--gap", type=float, default=0.5, help="seconds between requests (two runs at once: 1.0 each)")
    parser.add_argument("--list-only", action="store_true", help="only write the list of the photos already here")
    args = parser.parse_args()
    GAP[0] = args.gap

    out = args.data_dir / args.split
    out.mkdir(parents=True, exist_ok=True)
    fakes = annotations(args.data_dir / f"{args.split}_poly.json")
    print(f"{args.split}: {len(fakes)} annotated photos, {sum(v > 0 for v in fakes.values())} with a fake face", flush=True)
    if args.list_only:  # a run that stopped while writing its list: the photos are already here
        write_list(args.split, out, fakes, sorted(fakes))
        return

    with httpx.Client(timeout=300, follow_redirects=True) as client:
        where = {}  # photo name -> (zip on disk or URL, ZipInfo)
        for name in args.parts or PARTS[args.split]:
            pace()
            url = str(client.get(f"{RECORD}/{name}?download=1", headers={"Range": "bytes=0-0"}).url)
            if args.n == 0:
                download_zip(client, url, args.data_dir / name, args.workers)
                source, zf = args.data_dir / name, zipfile.ZipFile(args.data_dir / name)
            else:
                source, zf = url, zipfile.ZipFile(io.BufferedReader(RemoteFile(client, url), buffer_size=1 << 20))
            with zf:
                where.update({Path(i.filename).name: (source, i) for i in zf.infolist() if i.filename.endswith(".jpg")})
            print(f"  {name}: table of contents read ({len(where)} photos so far)", flush=True)

        names = sorted(n for n in where if n in fakes)
        if args.n:
            names = sorted(random.Random(args.seed).sample(names, min(args.n, len(names))))
        whole = lambda n: (out / n).exists() and zlib.crc32((out / n).read_bytes()) == where[n][1].CRC  # noqa: E731
        todo = [n for n in names if not whole(n)]
        print(f"{len(names)} photos picked, {len(names) - len(todo)} already here, {len(todo)} to fetch", flush=True)

        def fetch(name):
            """True when the photo arrived whole (its CRC matches the zip's)."""
            source, info = where[name]
            for attempt in range(5):
                try:
                    if isinstance(source, Path):
                        with zipfile.ZipFile(source) as zf:
                            data = zf.read(info)
                    else:
                        pace()
                        data = member(client, source, info)
                    if zlib.crc32(data) == info.CRC:
                        (out / name).write_bytes(data)
                        return True
                except (httpx.HTTPError, zlib.error):
                    pass
                time.sleep(10 * (attempt + 1))
            return False

        failed = []
        with ThreadPoolExecutor(args.workers) as pool:
            for done, (name, ok) in enumerate(zip(todo, pool.map(fetch, todo)), 1):
                failed += [] if ok else [name]
                if done % 250 == 0:
                    print(f"  {done}/{len(todo)} ({len(failed)} failed)", flush=True)
        if failed:
            print(f"{len(failed)} photos failed after 5 tries (left out; run again to retry)")

    write_list(args.split, out, fakes, names)


def write_list(split, out, fakes, names):
    names = [n for n in names if (out / n).exists()]
    rows = [{"path": str(out / n), "label": int(fakes[n] > 0), "source": f"openforensics/{split}"} for n in names]
    lst = Path(f"features/lists/openforensics_{split}.csv")
    lst.parent.mkdir(parents=True, exist_ok=True)  # features/ is not in git: missing in a fresh clone (Colab)
    with open(lst, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["path", "label", "source"], lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"{len(rows)} photos -> {lst}")


if __name__ == "__main__":
    main()
