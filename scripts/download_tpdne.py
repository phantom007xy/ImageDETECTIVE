"""Download faces from thispersondoesnotexist.com (StyleGAN) out of a public Hugging Face dataset (MIT licence).

Why: the detector missed every StyleGAN face it was shown (results/commons_ai_v1.txt), and thispersondoesnotexist.com
is the first site many people try. javi22/this-person-does-not-exist-10k (https://huggingface.co/datasets/
javi22/this-person-does-not-exist-10k, MIT) holds about 10,000 of its faces as "face (N).jpg". (SFHQ part 3 on Kaggle,
CC0, is the other GAN source: scripts/download_sfhq.py.)

Picks every --step-th face, splits them by number into train (85%) and test (15%), and writes the faces to
--out-dir/train and --out-dir/test (never git) and features/lists/tpdne_train.csv and tpdne_test.csv
(path, label 1, source gan/tpdne).

Usage:  python scripts/download_tpdne.py --out-dir D:/Datasets/TPDNE --step 6
"""

import argparse
import csv
import os
import shutil
import time
import zlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

from huggingface_hub import hf_hub_download  # noqa: E402

REPO, LAST = "javi22/this-person-does-not-exist-10k", 9980


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--step", type=int, default=6)
    args = parser.parse_args()

    split_of = lambda n: "test" if zlib.crc32(str(n).encode()) % 100 < 15 else "train"  # noqa: E731
    for split in ("train", "test"):
        (args.out_dir / split).mkdir(parents=True, exist_ok=True)

    def fetch(n):
        target = args.out_dir / split_of(n) / f"tpdne_{n:05d}.jpg"
        if target.exists():
            return target
        for attempt in range(5):
            try:
                shutil.copy(hf_hub_download(REPO, f"face ({n}).jpg", repo_type="dataset"), target)
                return target
            except Exception as error:  # e.g. a rate limit: wait and try again; a missing number is skipped
                if "404" in str(error) or "EntryNotFound" in type(error).__name__:
                    return None
                time.sleep(10 * (attempt + 1))
        return None

    with ThreadPoolExecutor(4) as pool:
        got = [p for p in pool.map(fetch, range(1, LAST + 1, args.step)) if p]
    for split in ("train", "test"):
        rows = [{"path": str(p), "label": 1, "source": "gan/tpdne"} for p in got if p.parent.name == split]
        with open(f"features/lists/tpdne_{split}.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["path", "label", "source"], lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
        print(f"{split}: {len(rows)} faces -> features/lists/tpdne_{split}.csv")


if __name__ == "__main__":
    main()
