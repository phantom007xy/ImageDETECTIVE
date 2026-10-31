"""Upload the app's code to a Hugging Face Space (the hosted demo), in one commit. Run it yourself, logged in.

What goes up: app/ (code, sample photos), src/ (code), main.py, LICENSE, the Space card (space/README.md as
README.md) and a requirements.txt for the Space (CPU PyTorch + requirements.txt). Nothing else: no weights (the
Space downloads them at start from the PRIVATE heads repo, scripts/upload_heads.py), no features, no data, no results,
no private notes.

The Space needs these secrets (Space -> Settings -> Variables and secrets):
  HF_TOKEN           a read token of an account with access to facebook/dinov3-vitl16-pretrain-lvd1689m,
                     facebook/sam3 and the private heads repo
  HEADS_REPO         the private heads repo, e.g. <user>/imagedetective-heads
  MODEL_API_KEY      the Meta Model API key (investigator, explanation, "Try to fool it"); without it the app
                     still gives the full verdict
  API_DAILY_CAP_USD  daily Meta API spending cap in dollars (default 3)

Usage (from the repo root, after `huggingface-cli login`):  python scripts/deploy_space.py --space <user>/ImageDETECTIVE
"""

import argparse
from pathlib import Path

from huggingface_hub import CommitOperationAdd, HfApi

# The Space has no GPU: the CPU build of PyTorch (smaller), same version as on the laptop.
TORCH_CPU = ["--extra-index-url https://download.pytorch.org/whl/cpu", "torch==2.11.0+cpu", "torchvision==0.26.0+cpu"]
CODE = ["app/*.py", "app/examples/*", "src/**/*.py", "main.py", "LICENSE"]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--space", required=True, help="Space id, e.g. user/ImageDETECTIVE")
    args = parser.parse_args()

    files = sorted({p for pattern in CODE for p in Path(".").glob(pattern) if p.is_file() and "__pycache__" not in p.parts})
    requirements = "\n".join(TORCH_CPU + Path("requirements.txt").read_text().splitlines()) + "\n"
    operations = [CommitOperationAdd(path_in_repo=p.as_posix(), path_or_fileobj=str(p)) for p in files]
    operations += [CommitOperationAdd(path_in_repo="README.md", path_or_fileobj="space/README.md"),
                   CommitOperationAdd(path_in_repo="requirements.txt", path_or_fileobj=requirements.encode())]

    api = HfApi()
    api.create_repo(args.space, repo_type="space", space_sdk="gradio", exist_ok=True)
    api.create_commit(repo_id=args.space, repo_type="space", operations=operations, commit_message="Update the app")
    print(f"uploaded {len(operations)} files to https://huggingface.co/spaces/{args.space}")


if __name__ == "__main__":
    main()
