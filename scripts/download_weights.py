"""Download only the eight required trained assets from an approved private model repository."""
import argparse
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download
from huggingface_hub.errors import HfHubHTTPError

from check_setup import ASSETS, ROOT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True, help="Approved private model repo ID supplied by the project owner")
    parser.add_argument("--revision", default="main", help="Optional approved model revision")
    args = parser.parse_args()
    try:
        info = HfApi().repo_info(args.repo, repo_type="model", revision=args.revision)
        if not set(ASSETS) <= {file.rfilename for file in info.siblings}:
            raise ValueError("Repository does not contain the complete required artifact bundle.")
        snapshot_download(args.repo, revision=info.sha, allow_patterns=list(ASSETS), local_dir=ROOT / "weights")
    except HfHubHTTPError as error:
        print(f"Download failed (HTTP {error.response.status_code}). Run hf auth login with your own approved account.")
        print("The public code repository does not grant access to the private trained artifacts.")
        return 1
    except ValueError as error:
        print(str(error))
        return 1
    if any(not (ROOT / "weights" / name).is_file() for name in ASSETS):
        print("Download incomplete. Do not start the app until all eight artifacts are present.")
        return 1
    print("Required artifacts downloaded. Next: python scripts/check_setup.py --hardware")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
