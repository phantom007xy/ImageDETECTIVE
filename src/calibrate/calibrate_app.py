"""Calibrate everything the app uses, in the right order, with all dev pools (public and my private ones).

  1. head F (src/calibrate/calibrate.py): AI / authentic thresholds, temperature, faces, the face-swap check
  2. the Muse-edit check (src/calibrate/calibrate_edit_check.py): its line, jointly with step 1's AI threshold
Private pools are used if their feature files exist (made by src/eval/private_photo_test.py --edit-check on my phone
photos and selfies); only totals are printed. Run it yourself after the private tests.

Usage (from the repo root):  python -m src.calibrate.calibrate_app
"""

import subprocess
import sys
from pathlib import Path

HEAD = "weights/open_gen_v1/head_F.pt"
EDIT = "weights/edit_check_v1/head_EC_both.pt"
PRIVATE = {"phone": "data/private/phone_v1", "selfies": "data/private/originals"}  # <folder name> of the private test


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
    for name, p in private.items():
        args += ["--real", f"{name}:{p}_app.npz"]
        if Path(p + "_faces_app.npz").exists():
            args += ["--real-faces", f"{name} faces:{p}_faces_app.npz"]
    run("src.calibrate.calibrate", args)

    pools = {"polished dev": ("features/commons_dev_app.npz", "features/pe/commons_dev_eval.npz"),
             "OpenImages val": ("features/muse_pairs_val_app.npz", "features/pe/muse_pairs_val_eval.npz"),
             "originals v1": ("features/muse_edits_v1_app.npz", "features/pe/muse_edits_v1_eval.npz"),
             "originals v2": ("features/muse_edits_test_v2_app.npz", "features/pe/muse_edits_test_v2_eval.npz")}
    pools.update({name: (p + "_app.npz", p + "_pe.npz") for name, p in private.items() if Path(p + "_pe.npz").exists()})
    args = ["--head", EDIT, "--whole-head", HEAD, "--out", EDIT.replace(".pt", "_calibration.json")]
    for name, (whole, pe) in pools.items():
        args += ["--pool", f"{name}:{whole}:{pe}"]
    for name, stem in (("edits v1", "muse_edits_v1"), ("edits v2", "muse_edits_test_v2"), ("Muse test v1", "muse_image_test_v1"),
                       ("Muse pairs val", "muse_pairs_val")):
        args += ["--fake", f"{name}:features/{stem}_app.npz:features/pe/{stem}_eval.npz"]
    run("src.calibrate.calibrate_edit_check", args)


if __name__ == "__main__":
    main()
