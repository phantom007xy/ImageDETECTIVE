"""Calibrate everything the app uses, in the right order, with all dev pools (public and my private ones).

  1. head F (src/calibrate/calibrate.py): AI / authentic thresholds, temperature, faces, the face-swap check
  2. the Muse-edit check (src/calibrate/calibrate_edit_check.py): its line, jointly with step 1's AI threshold and the
     joint trade (head F gives up its lowest flag in its tightest pool; results/edit_check_v3.txt)
Private pools are used if their feature files exist (made by src/eval/private_photo_test.py --edit-check ...
--pe-weights ... on my phone photos and selfies); only totals are printed. Run it yourself after the private tests.

Usage (from the repo root):  python -m src.calibrate.calibrate_app
"""

import subprocess
import sys
from pathlib import Path

HEAD = "weights/open_gen_v1/head_F.pt"
EDIT = "weights/edit_check_v3/head_EC3.pt"  # on the fine-tuned PE (src/models/tune_pe.py, step 7,000)
PE_WEIGHTS = "weights/edit_check_v3/pe_EC3.pt"
PE_DIR = "features/pe_ec3"  # its features of the public pools (src/features/extract_pe.py --pe-weights)
GIVE_UP = 1
PRIVATE = {"phone": "data/private/phone_v1", "selfies": "data/private/originals"}  # <folder name> of the private test
# Real photos used nowhere else (src/eval/leak_check.py): unseen OpenImages photos and Commons phone photos (dev part).
NEW_REAL = {"OpenImages unseen": "sid_reals_0_9_unseen", "Commons v3 dev": "commons_v3_dev"}


def run(module, args):
    print(f"\n=== python -m {module} ===", flush=True)
    subprocess.run([sys.executable, "-m", module, *args], check=True)


def main():
    private = {name: p for name, p in PRIVATE.items() if Path(p + "_app.npz").exists()}
    print("private pools found: " + (", ".join(private) or "none (public pools only)"))

    args = ["--head", HEAD, "--out", HEAD.replace(".pt", "_calibration.json"),
            "--real", "polished dev:features/commons_dev_app.npz", "--real", "OpenImages val:features/muse_pairs_val_app.npz",
            "--fake", "Muse test v1:features/muse_image_test_v1_app.npz", "--fake", "Muse pairs val:features/muse_pairs_val_app.npz",
            "--fake", "SDXL (unseen):features/sd_faces_app.npz@sd_faces/sdxl",
            "--real-faces", "polished faces:features/faces/commons_dev_eval.npz",
            "--real-faces", "OpenImages faces:features/faces/muse_pairs_val_eval.npz",
            "--fake-faces", "Muse test faces:features/faces/muse_image_test_v1_eval.npz",
            "--fake-faces", "SD faces:features/faces/sd_faces_eval.npz",
            "--swap-head", "weights/dfdc_v1/head_dfdc.pt",
            "--swap-real-faces", "DFDC p3 real:features/faces/dfdc_part3_eval.npz",
            "--swap-real-faces", "SID val real:features/faces/validation-00000-of-00034_eval.npz",
            "--swap-fakes", "DFDC p3 swaps:features/faces/dfdc_part3_eval.npz",
            "--swap-fakes", "WildDeepfake:features/faces/wilddeepfake_eval.npz"]
    for name, stem in NEW_REAL.items():
        args += ["--real", f"{name}:features/{stem}_app.npz"]
    for name, p in private.items():
        args += ["--real", f"{name}:{p}_app.npz"]
        if Path(p + "_faces_app.npz").exists():
            args += ["--real-faces", f"{name} faces:{p}_faces_app.npz"]
    run("src.calibrate.calibrate", args)

    pools = {"polished dev": "commons_dev", "OpenImages val": "muse_pairs_val", "originals v1": "muse_edits_v1",
             "originals v2": "muse_edits_test_v2", **NEW_REAL}
    pools = {name: (f"features/{stem}_app.npz", f"{PE_DIR}/{stem}_eval.npz") for name, stem in pools.items()}
    pe_name = Path(PE_WEIGHTS).stem
    pools.update({name: (p + "_app.npz", f"{p}_{pe_name}.npz") for name, p in private.items()
                  if Path(f"{p}_{pe_name}.npz").exists()})
    args = ["--head", EDIT, "--pe-weights", PE_WEIGHTS, "--whole-head", HEAD, "--give-up", str(GIVE_UP),
            "--out", EDIT.replace(".pt", "_calibration.json")]
    for name, (whole, pe) in pools.items():
        args += ["--pool", f"{name}:{whole}:{pe}"]
    fakes = [("edits v1 local", "muse_edits_v1", "muse_edits/local"), ("edits v1 whole", "muse_edits_v1", "muse_edits/whole"),
             ("edits v2 local", "muse_edits_test_v2", "muse_edits/local"),
             ("edits v2 whole", "muse_edits_test_v2", "muse_edits/whole"), ("Muse test v1", "muse_image_test_v1", ""),
             ("Muse pairs val", "muse_pairs_val", ""), ("SDXL (unseen)", "sd_faces", "sd_faces/sdxl"),
             ("SD 3.5 Medium", "open_gen_dev", "open_gen/sd35m"), ("Kandinsky 2.2", "open_gen_dev", "open_gen/k22"),
             ("PixArt-Sigma", "open_gen_dev", "open_gen/pixart")]
    for name, stem, prefix in fakes:
        args += ["--fake", f"{name}:features/{stem}_app.npz:{PE_DIR}/{stem}_eval.npz" + (f"@{prefix}" if prefix else "")]
    run("src.calibrate.calibrate_edit_check", args)


if __name__ == "__main__":
    main()
