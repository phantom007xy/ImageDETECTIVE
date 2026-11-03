"""Calibrate everything the app uses, in the right order, with all dev pools (public and my private ones).

  1. head F (src/calibrate/calibrate.py): AI / authentic thresholds, temperature, faces, the face-swap check
  2. the Muse-edit check (src/calibrate/calibrate_edit_check.py): its line, jointly with step 1's AI threshold and the
     joint trade (head F gives up its lowest flag in its tightest pool; results/edit_check_v3.txt); then the red face
     lines are raised where needed, so that the final verdict (whole OR edited OR a red face) stays within the budget
Private pools are used if their feature files exist (made by src/eval/private_photo_test.py --edit-check ...
--pe-weights ... on my phone photos and selfies); only totals are printed. Run it yourself after the private tests.

Usage (from the repo root):  python -m src.calibrate.calibrate_app
"""

import subprocess
import sys
from pathlib import Path

import numpy as np

from src.models.linear_head import LinearHead

HEAD = "weights/google_v1/head_G3tt.pt"  # G3 + tighter real close-ups (tight face crops looked like GAN faces; results/head_g_v1.txt)
EDIT = "weights/edit_check_v3/head_EC3pf.pt"  # EC3's PE (tune_pe step 7,000) + a head that knows person edits and
# forwarded copies (EC3p + training views of forwarded copies: src/features/extract_pe.py --forward)
# (src/eval/edit_head_cv.py --save-final, 1 Nov: from EC3's head, pull 0.1, my 30 selfies' Muse edits x5)
PE_WEIGHTS = "weights/edit_check_v3/pe_EC3.pt"
# The face-swap check: Meta's Perception Encoder fine-tuned on face crops (src/models/tune_face_pe.py, step 4,000 chosen on
# dev: DFDC + OpenForensics swaps, self-blended fakes, real faces; results/face_pe_v1.txt) and its head; one more pass per
# face. Before 2 Nov: swap_v3 (DINOv3, results/swap_head_v2.txt), swap_v2, dfdc_v1.
SWAP = "weights/face_pe_v2/head_facepe.pt"
SWAP_PE = "weights/face_pe_v2/pe_step1000.pt"
# A second head on the same PE features, trained from EC3p with Google's pictures too (Gemini 3.1 Flash, Nano Banana Pro,
# Gemini 2.5 restorations and edits): its own, higher line, so the Muse head keeps its line (results/head_g_v1.txt, 1 Nov)
EDIT2 = "weights/edit_check_v3/head_EC3g.pt"
PE_DIR = "features/pe_ec3"  # its features of the public pools (src/features/extract_pe.py --pe-weights)
GIVE_UP = 1
# The edit check alone may flag at most this share of every real pool (the whole verdict stays <= 5%). 0.5% let one
# outlier among my 100 new object photos set the line (local Muse edits 76% -> 24%; 3 Nov), so it gets 2% (local edits 71%, Gemini dev 90%; my 100 objects 3%, judge test 3 of 60 real photos).
EDIT_RATE = 0.02
# <folder name> of the private test. My first 30 selfies ("originals") are TRAINING data of head_EC3p since 1 Nov,
# so they are no calibration pool any more; new test selfies (other place, clothes, light) take their place.
PRIVATE = {"phone": "data/private/phone_v1", "new selfies": "data/private/test_selfies",
           "phone objects": "data/private/phone_objects"}  # new phone photos without people (3 Nov; calibration only)
# Real photos used nowhere else (src/eval/leak_check.py): unseen OpenImages photos and Commons phone photos (dev part).
NEW_REAL = {"OpenImages unseen": "sid_reals_0_9_unseen", "Commons v3 dev": "commons_v3_dev",
            # real tight face close-ups (like profile pictures) and real Commons selfies and portraits (1 Nov)
            "close-ups": "real_face_crops_dev_all", "selfies+portraits": "commons_v4_dev"}
# The app's face crops of each set (src/features/extract_faces.py --mode eval), so red faces count in the final verdict.
# The Muse edits' originals are people-free photos (checked with SAM 3 and by eye), so they have no face file.
FACES = {"polished dev": "commons_dev", "OpenImages val": "muse_pairs_val", "OpenImages unseen": "sid_reals_0_9_unseen",
         "Commons v3 dev": "commons_v3_dev", "Muse test v1": "muse_image_test_v1", "Muse pairs val": "muse_pairs_val",
         "SDXL (unseen)": "sd_faces", "GAN faces (TPDNE)": "tpdne_test",
         "close-ups": "real_face_crops_dev_all", "selfies+portraits": "commons_v4_dev"}


def run(module, args):
    print(f"\n=== python -m {module} ===", flush=True)
    subprocess.run([sys.executable, "-m", module, *args], check=True)


def main():
    private = {name: p for name, p in PRIVATE.items() if Path(p + "_app.npz").exists()}
    keys = LinearHead.load(SWAP).keys  # the face features the swap head reads; older private face files lack the middle layers
    for name, p in private.items():
        faces = Path(p + "_faces_app.npz")
        if faces.exists() and not set(keys) <= set(np.load(faces).files):
            raise SystemExit(f"{name}: {faces} has no {', '.join(keys)} (the face-swap head reads middle layers since 1 Nov). "
                             f"Run src.eval.private_photo_test on that folder again first (with --edit-check and --pe-weights).")
    print("private pools found: " + (", ".join(private) or "none (public pools only)"))

    args = ["--head", HEAD, "--out", HEAD.replace(".pt", "_calibration.json"),
            "--real", "polished dev:features/commons_dev_app.npz", "--real", "OpenImages val:features/muse_pairs_val_app.npz",
            "--fake", "Muse test v1:features/muse_image_test_v1_app.npz", "--fake", "Muse pairs val:features/muse_pairs_val_app.npz",
            "--fake", "SDXL (unseen):features/sd_faces_app.npz@sd_faces/sdxl",
            "--real-faces", "polished faces:features/faces/commons_dev_eval.npz",
            "--real-faces", "OpenImages faces:features/faces/muse_pairs_val_eval.npz",
            "--fake-faces", "Muse test faces:features/faces/muse_image_test_v1_eval.npz",
            "--fake-faces", "SD faces:features/faces/sd_faces_eval.npz",
            # face swaps: DFDC video swaps + OpenForensics photo swaps + real SID-Set faces (results/swap_head_v2.txt)
            "--swap-head", SWAP, "--swap-pe", SWAP_PE,
            "--swap-real-faces", "DFDC p3 real:features/faces/dfdc_part3_eval.npz",
            "--swap-real-faces", "SID val real:features/faces/validation-00000-of-00034_eval.npz",
            "--swap-fakes", "DFDC p3 swaps:features/faces/dfdc_part3_eval.npz",
            "--swap-fakes", "WildDeepfake:features/faces/wilddeepfake_eval.npz",
            "--swap-fakes", "OpenForensics test (photo swaps):features/faces/openforensics_test_dev_eval.npz"]
    for name, stem in NEW_REAL.items():
        args += ["--real", f"{name}:features/{stem}_app.npz"]
        if Path(f"features/faces/{stem}_eval.npz").exists():
            args += ["--real-faces", f"{name} faces:features/faces/{stem}_eval.npz"]
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
    args = ["--head", EDIT, "--pe-weights", PE_WEIGHTS, "--whole-head", HEAD, "--give-up", str(GIVE_UP), "--rate", str(EDIT_RATE), "--head2", EDIT2,
            "--out", EDIT.replace(".pt", "_calibration.json")]
    for name, (whole, pe) in pools.items():
        args += ["--pool", f"{name}:{whole}:{pe}"]
    faces = {name: f"features/faces/{stem}_eval.npz" for name, stem in FACES.items()}
    faces.update({name: p + "_faces_app.npz" for name, p in private.items() if name in pools})
    for name, path in faces.items():
        if Path(path).exists():
            args += ["--faces", f"{name}:{path}"]
    fakes = [("edits v1 local", "muse_edits_v1", "muse_edits/local"), ("edits v1 whole", "muse_edits_v1", "muse_edits/whole"),
             ("edits v2 local", "muse_edits_test_v2", "muse_edits/local"),
             ("edits v2 whole", "muse_edits_test_v2", "muse_edits/whole"), ("Muse test v1", "muse_image_test_v1", ""),
             ("Muse pairs val", "muse_pairs_val", ""), ("SDXL (unseen)", "sd_faces", "sd_faces/sdxl"),
             ("SD 3.5 Medium", "open_gen_dev", "open_gen/sd35m"), ("Kandinsky 2.2", "open_gen_dev", "open_gen/k22"),
             ("PixArt-Sigma", "open_gen_dev", "open_gen/pixart"), ("GAN faces (TPDNE)", "tpdne_test", ""), ("GAN faces (SFHQ)", "sfhq_pt3_test", ""),
             ("Nano Banana dev", "bitmind_nb_dev", ""), ("GEMINI dev (today's model)", "google/gemini_dev_v1", "")]
    for name, stem, prefix in fakes:
        args += ["--fake", f"{name}:features/{stem}_app.npz:{PE_DIR}/{stem}_eval.npz" + (f"@{prefix}" if prefix else "")]
    run("src.calibrate.calibrate_edit_check", args)


if __name__ == "__main__":
    main()
