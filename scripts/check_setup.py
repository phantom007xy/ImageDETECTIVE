"""Check local installation without loading models, downloading files or calling APIs."""
import argparse
from importlib import metadata
import json
import math
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
ASSETS = (
    "google_v1/head_G3tt.pt", "google_v1/head_G3tt_calibration.json",
    "edit_check_v3/head_EC3pf.pt", "edit_check_v3/head_EC3g.pt",
    "edit_check_v3/pe_EC3.pt", "edit_check_v3/head_EC3pf_calibration.json",
    "face_pe_v4/head_facepe.pt", "face_pe_v4/pe_step14000.pt",
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checks-only", action="store_true", help="Check dependencies for synthetic tests; skip private assets")
    parser.add_argument("--weights-dir", type=Path, default=ROOT / "weights")
    parser.add_argument("--hardware", action="store_true", help="Also import PyTorch and report CUDA availability; no models loaded")
    args = parser.parse_args()
    errors = []
    print(f"Python: {sys.version.split()[0]}")
    if sys.version_info[:2] != (3, 12):
        errors.append("Use Python 3.12, the version tested by this project.")
    requirements = [("torch", "2.11.0"), ("torchvision", "0.26.0")]
    for line in (ROOT / "requirements.txt").read_text().splitlines():
        match = re.fullmatch(r"([A-Za-z0-9_-]+)==([^\s#]+)", line.strip())
        if match:
            requirements.append(match.groups())
    for package, expected in requirements:
        try:
            actual = metadata.version(package)
        except metadata.PackageNotFoundError:
            errors.append(f"Missing dependency: {package}=={expected}")
            continue
        if actual.split("+")[0] != expected:
            errors.append(f"Dependency mismatch: {package} needs {expected}; installed {actual}")
    if not args.checks_only:
        missing = [name for name in ASSETS if not (args.weights_dir / name).is_file()
                   or not (args.weights_dir / name).stat().st_size]
        errors.extend("Missing artifact: weights/" + name for name in missing)
        if not missing:
            try:
                whole = json.loads((args.weights_dir / ASSETS[1]).read_text())
                edit = json.loads((args.weights_dir / ASSETS[5]).read_text())
                references = {"head2": (edit, "edit_check_v3/head_EC3g.pt"),
                              "pe_weights": (edit, "edit_check_v3/pe_EC3.pt"),
                              "swap_head": (whole, "face_pe_v4/head_facepe.pt"),
                              "swap_pe": (whole, "face_pe_v4/pe_step14000.pt")}
                for field, (config, expected) in references.items():
                    if config.get(field, "").replace("\\", "/") != "weights/" + expected:
                        errors.append(f"Calibration references an unexpected {field}; use the complete matching artifact bundle.")
                for base, moved in (("ai_logit", "whole_ai_logit"), ("face_ai_logit", "face_ai_logit"),
                                    ("swap_ai_logit", "swap_ai_logit")):
                    if abs(whole[base] - edit[moved + "_calibrated"]) > 1e-6:
                        errors.append("Whole-image and edit calibration do not match: " + base)
                for value in (whole["temperature"], whole["real_logit"], whole["ai_logit"],
                              edit["edit_logit"], edit["edit2_logit"]):
                    if not isinstance(value, (int, float)) or not math.isfinite(value):
                        raise ValueError("Invalid calibrated value")
                if whole["temperature"] <= 0:
                    raise ValueError("Invalid temperature")
            except (OSError, ValueError, TypeError, KeyError, AttributeError):
                errors.append("Calibration JSON is invalid or incomplete; restore the matching project artifact bundle.")
    if args.hardware:
        try:
            import torch
            print(f"PyTorch: {torch.__version__}; CUDA available: {torch.cuda.is_available()}")
            if torch.cuda.is_available():
                print("GPU: " + torch.cuda.get_device_name(0))
            else:
                print("The detector will use CPU; inference is slower.")
        except Exception as error:
            errors.append("PyTorch runtime could not load: " + type(error).__name__)
    for error in errors:
        print("ERROR: " + error)
    if errors:
        print("Setup incomplete. Follow README > Run it yourself.")
        return 1
    print("Dependency setup ready for synthetic checks; trained artifacts were skipped." if args.checks_only else
          "Dependencies and trained artifact layout ready. Model access/cache is verified when app.server starts.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
