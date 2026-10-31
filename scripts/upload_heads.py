"""Upload the app's trained heads to a PRIVATE Hugging Face model repo, for the hosted demo.

The heads (a few KB each) are never in the public GitHub repo or the public Space: the DFDC face-swap head
must not be published, and a public Space's files are public. The Space downloads them at start with its
token (app/app.py, environment variables HEADS_REPO and HF_TOKEN).

Uploads: the app's head, its calibration file and every head the calibration names (e.g. the DFDC swap head),
under the same paths as in weights/. Stops if the repo exists and is not private.

Usage (from the repo root, logged in with `huggingface-cli login` or with HF_TOKEN set; run it yourself):
  python scripts/upload_heads.py --repo <your HF user>/imagedetective-heads --head weights/open_gen_v1/head_F.pt
"""

import argparse
import json
from pathlib import Path

from huggingface_hub import HfApi


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo", required=True, help="model repo id, e.g. user/imagedetective-heads")
    parser.add_argument("--head", type=Path, required=True, help="the app's head (weights/...)")
    args = parser.parse_args()

    calibration = args.head.with_name(args.head.stem + "_calibration.json")
    files = [args.head, calibration]
    swap = json.loads(calibration.read_text()).get("swap_head")
    if swap:
        files.append(Path(swap.replace("\\", "/")))
    for f in files:
        if not f.exists():
            raise SystemExit(f"missing: {f}")

    api = HfApi()
    api.create_repo(args.repo, repo_type="model", private=True, exist_ok=True)
    if not api.repo_info(args.repo, repo_type="model").private:
        raise SystemExit(f"{args.repo} is NOT private: nothing uploaded. Make it private first.")
    for f in files:
        in_repo = f.relative_to("weights").as_posix()  # same layout as weights/, so the app's paths work
        api.upload_file(path_or_fileobj=str(f), path_in_repo=in_repo, repo_id=args.repo, repo_type="model")
        print(f"uploaded {f} -> {args.repo}/{in_repo}")


if __name__ == "__main__":
    main()
