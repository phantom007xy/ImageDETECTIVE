"""Download StyleGAN2 faces from SFHQ part 3 (Kaggle, CC0): GAN-made faces for training and testing.

Why: the detector missed every StyleGAN face it was shown (thispersondoesnotexist.com style, results/commons_ai_v1.txt):
all our AI training images come from diffusion models. SFHQ part 3 ("Synthetic Faces High Quality", David Beniaguev)
has 118,358 curated 1024 x 1024 faces made by StyleGAN2; Kaggle lists it as CC0: Public Domain.
https://www.kaggle.com/datasets/selfishgene/synthetic-faces-high-quality-sfhq-part-3

How: Kaggle allows only a burst of about 90 single-file downloads and then blocks for ~45 minutes (HTTP 429). So the
script asks Kaggle ONCE for the whole dataset's zip (a signed Google Storage link, 23 GB), reads only the zip's table of
contents (at its end) and then fetches just the picked faces with HTTP range requests: about 0.4 GB instead of 23 GB.
Picks --n faces spread evenly over the numbered files (images/images/SFHQ_pt3_00000001.jpg ...), split by number into
train (85%) and test (15%). Needs ~/.kaggle/kaggle.json (kaggle.com -> Settings -> API -> Create New Token).

Writes:
  - the faces to --out-dir/train and --out-dir/test (never git)
  - features/lists/sfhq_pt3_train.csv and sfhq_pt3_test.csv (path, label 1, source sfhq/pt3)

Usage:  python scripts/download_sfhq.py --out-dir D:/Datasets/SFHQ/pt3 --n 2400
"""

import argparse
import csv
import io
import json
import struct
import zipfile
import zlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx

DATASET = "selfishgene/synthetic-faces-high-quality-sfhq-part-3"
LAST = 118358


class RemoteFile(io.RawIOBase):
    """A read-only, seekable file over HTTP range requests (enough for zipfile to read the table of contents)."""

    def __init__(self, client, url):
        self.client, self.url, self.pos = client, url, 0
        r = client.get(url, headers={"Range": "bytes=0-0"})
        self.size = int(r.headers["content-range"].split("/")[1])

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, offset, whence=io.SEEK_SET):
        self.pos = {io.SEEK_SET: 0, io.SEEK_CUR: self.pos, io.SEEK_END: self.size}[whence] + offset
        return self.pos

    def readinto(self, buffer):
        end = min(self.pos + len(buffer), self.size) - 1
        if end < self.pos:
            return 0
        data = self.client.get(self.url, headers={"Range": f"bytes={self.pos}-{end}"}).content
        buffer[:len(data)] = data
        self.pos += len(data)
        return len(data)


def member(client, url, info):
    """One file of the zip: its local header tells where the data starts; stored or deflated."""
    start = info.header_offset
    raw = client.get(url, headers={"Range": f"bytes={start}-{start + 30 + 1024 + info.compress_size}"}).content
    name_len, extra_len = struct.unpack("<HH", raw[26:30])
    data = raw[30 + name_len + extra_len: 30 + name_len + extra_len + info.compress_size]
    return data if info.compress_type == zipfile.ZIP_STORED else zlib.decompress(data, -15)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--n", type=int, default=2400)
    args = parser.parse_args()

    token = json.loads((Path.home() / ".kaggle" / "kaggle.json").read_text())
    numbers = [1 + round(i * (LAST - 1) / (args.n - 1)) for i in range(args.n)]
    split_of = lambda number: "test" if zlib.crc32(str(number).encode()) % 100 < 15 else "train"  # noqa: E731
    for split in ("train", "test"):
        (args.out_dir / split).mkdir(parents=True, exist_ok=True)
    todo = [n for n in numbers if not (args.out_dir / split_of(n) / f"SFHQ_pt3_{n:08d}.jpg").exists()]
    print(f"{len(numbers)} faces picked, {len(numbers) - len(todo)} already here, {len(todo)} to fetch", flush=True)

    if todo:
        with httpx.Client(auth=(token["username"], token["key"]), timeout=120) as kaggle:  # the ONE Kaggle request
            r = kaggle.get(f"https://www.kaggle.com/api/v1/datasets/download/{DATASET}")
        if r.status_code != 302:
            raise SystemExit(f"Kaggle answered {r.status_code} {r.text[:120]} (429 = wait until the block lifts)")
        url = r.headers["location"]
        with httpx.Client(timeout=120) as storage:  # Google Storage: no Kaggle limits, no Kaggle token
            with zipfile.ZipFile(io.BufferedReader(RemoteFile(storage, url), buffer_size=1 << 20)) as zf:
                infos = {i.filename.rsplit("/", 1)[-1]: i for i in zf.infolist() if i.filename.startswith("images/images/")}
            print(f"zip table of contents: {len(infos)} faces in images/images/", flush=True)

            def fetch(number):
                name = f"SFHQ_pt3_{number:08d}.jpg"
                if name not in infos:
                    return None  # a gap in the numbering (curated out)
                (args.out_dir / split_of(number) / name).write_bytes(member(storage, url, infos[name]))
                return name

            with ThreadPoolExecutor(8) as pool:
                got = [g for g in pool.map(fetch, todo) if g]
        print(f"fetched {len(got)} of {len(todo)}", flush=True)

    for split in ("train", "test"):
        rows = [{"path": str(p), "label": 1, "source": "sfhq/pt3"} for p in sorted((args.out_dir / split).glob("*.jpg"))]
        with open(f"features/lists/sfhq_pt3_{split}.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["path", "label", "source"], lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
        print(f"{split}: {len(rows)} faces -> features/lists/sfhq_pt3_{split}.csv")


if __name__ == "__main__":
    main()
