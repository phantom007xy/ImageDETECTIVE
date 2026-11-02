"""Download RTFS-10k (RealTimeFaceSwap-10k, Kaggle stplusplus/rtfs-10k, CC BY-SA 4.0) for the face-swap model.

1,636 short video-call-like clips (Creative Commons YouTube videos, one person talking to the camera) and 9,772 face
swaps made from them with inswapper (the model behind Roop, FaceFusion and Deep-Live-Cam) and UniFace, about 40% of the
inswapper ones with a face enhancer; 2,000 source photos are only listed (photos.csv), not included. The whole dataset
is one 17.7 GB zip: made to run on Colab (fast network). Needs a Kaggle API token: KAGGLE_USERNAME and KAGGLE_KEY in
the environment (Colab secrets) or ~/.kaggle/kaggle.json. Unpacks to --out (original_videos/, inswapper/, uniface/ and
the CSV files) and deletes the zip. Face crops: python -m src.faces.make_crops rtfs (see there).

Usage (from the repo root):  python scripts/download_rtfs.py --out /content/data/rtfs
"""

import argparse
import json
import os
import shutil
import time
import zipfile
from pathlib import Path

import httpx

URL = "https://www.kaggle.com/api/v1/datasets/download/stplusplus/rtfs-10k"


def credentials():
    if os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"):
        return os.environ["KAGGLE_USERNAME"], os.environ["KAGGLE_KEY"]
    token = Path.home() / ".kaggle" / "kaggle.json"
    if token.exists():
        c = json.loads(token.read_text())
        return c["username"], c["key"]
    raise SystemExit("no Kaggle token: set KAGGLE_USERNAME and KAGGLE_KEY (Colab secrets) or ~/.kaggle/kaggle.json")


def download(target, auth):
    """The zip, continued where it stopped after a broken connection (Range requests; the file host supports them)."""
    for attempt in range(10):
        done = target.stat().st_size if target.exists() else 0
        try:
            with httpx.Client(auth=auth, timeout=120, follow_redirects=True) as client, \
                    client.stream("GET", URL, headers={"Range": f"bytes={done}-"} if done else {}) as r:
                if r.status_code == 416:  # nothing left to get
                    return
                if r.status_code not in (200, 206):
                    raise httpx.HTTPError(f"HTTP {r.status_code}")
                if done and r.status_code == 200:  # the range was ignored: start again
                    done = 0
                total = done + int(r.headers.get("content-length", 0))
                start, first, mark = time.time(), done, done
                with open(target, "ab" if done else "wb") as f:
                    for chunk in r.iter_bytes(1 << 22):
                        f.write(chunk)
                        done += len(chunk)
                        if done - mark >= 2 << 30:
                            mark = done
                            speed = (done - first) / (time.time() - start) / 1e6
                            print(f"  {done / 1e9:.1f} of {total / 1e9:.1f} GB, {speed:.0f} MB/s", flush=True)
            if total and done >= total:
                return
        except httpx.HTTPError as e:
            print(f"  connection lost ({e!r}); retry {attempt + 1}", flush=True)
            time.sleep(10 * (attempt + 1))
    raise SystemExit("RTFS-10k download failed 10 times")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if (args.out / "inswapper.csv").exists() and (args.out / "original_videos").exists():
        print("RTFS-10k is already here", flush=True)
        return
    for _ in range(60):  # the zip and its unpacked videos need ~36 GB for a moment; other downloads may free space first
        if shutil.disk_usage(args.out).free > 40e9:
            break
        print(f"  waiting for disk space ({shutil.disk_usage(args.out).free / 1e9:.0f} GB free)", flush=True)
        time.sleep(60)
    target = args.out / "rtfs-10k.zip"
    download(target, credentials())
    print(f"downloaded {target.stat().st_size / 1e9:.1f} GB; unpacking ...", flush=True)
    with zipfile.ZipFile(target) as z:
        z.extractall(args.out)
    target.unlink()
    videos = {d: len(list((args.out / d).glob("*.mp4"))) for d in ("original_videos", "inswapper", "uniface")}
    print("RTFS-10k: " + ", ".join(f"{d} {n}" for d, n in videos.items()), flush=True)


if __name__ == "__main__":
    main()
