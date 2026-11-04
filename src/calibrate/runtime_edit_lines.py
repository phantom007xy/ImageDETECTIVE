"""Finish the edit calibration using real-pool logits from the actual runtime.

Public scalar predictions are reused only if all model/head hashes match a frozen
baseline. Private calibration photos run offline; their per-image records stay in
data/private. No training or final judge-test photos are used to choose thresholds.
"""

import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from src.calibrate.calibrate_edit_check import joint_line
from src.eval.fair_compare import threshold_for
from src.inference import Detector

HEAD = "weights/google_v1/head_G3tt.pt"
EDIT = "weights/edit_check_v3/head_EC3pf.pt"
PUBLIC = {"polished dev": "commons_dev", "OpenImages val": "muse_pairs_val",
          "originals v1": "muse_edits_v1", "originals v2": "muse_edits_test_v2",
          "OpenImages unseen": "sid_reals_0_9_unseen", "Commons v3 dev": "commons_v3_dev",
          "close-ups": "real_face_crops_dev_all", "selfies+portraits": "commons_v4_dev"}
PRIVATE = {"phone": "data/private/release_calibration_phone.csv",
           "phone objects": "data/private/release_calibration_objects.csv"}


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for part in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(part)
    return digest.hexdigest()


def scalar(result):
    return {"whole": float(result["score"]), "edit1": float(result["edit"]["logit"]),
            "edit2": float(result["edit"]["logit2"]),
            "face": bool(any(f.get("flagged") for f in result["faces"]))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, default=Path("results/release_2026-11-04"))
    parser.add_argument("--out-dir", type=Path, default=Path("results/hardened_2026-11-04"))
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(4)
    frozen = json.loads((args.baseline / "configuration.json").read_text())
    predictions = json.loads((args.baseline / "live_predictions.json").read_text())
    if predictions["configuration_sha256"] != frozen["sha256"]:
        raise RuntimeError("Public scalar cache does not match its baseline")
    cal_path = Path(EDIT).with_name("head_EC3pf_calibration.json")
    cal = json.loads(cal_path.read_text())
    whole = json.loads(Path(HEAD).with_name("head_G3tt_calibration.json").read_text())
    for name in (HEAD, EDIT, cal["head2"], cal["pe_weights"], whole["swap_head"], whole["swap_pe"]):
        if sha(name) != frozen["artifacts_and_code"][Path(name).as_posix()]:
            raise RuntimeError("Model artifacts changed; collect a matching live baseline first")
    for key, baseline_key in (("whole_ai_logit", "whole_ai_logit"), ("face_ai_logit", "face_ai_logit"),
                              ("swap_ai_logit", "swap_ai_logit")):
        old_cal = json.loads(Path("weights/edit_check_v3/head_EC3pf_calibration_baseline.json").read_text())
        if cal[key] != old_cal[baseline_key]:
            raise RuntimeError("Fixed whole/face lines changed; public scalar flags need refreshing")
    pools = {}
    for name, stem in PUBLIC.items():
        rows = [r for r in csv.DictReader(Path(f"features/lists/{stem}.csv").open(newline="")) if r["label"] == "0"]
        records = []
        for row in rows:
            p = predictions["records"][sha(row["path"])]
            records.append({"whole": p["score"], "edit1": p["edit"]["logit"],
                            "edit2": p["edit"]["logit2"], "face": p["faces_flagged"] > 0})
        pools[name] = records
    private_path = Path("data/private/runtime_calibration_2026-11-04.json")
    cached = json.loads(private_path.read_text()) if private_path.exists() else {}
    fingerprint = {name: frozen["artifacts_and_code"][Path(name).as_posix()]
                   for name in (HEAD, EDIT, cal["head2"], cal["pe_weights"], whole["swap_head"], whole["swap_pe"])}
    private_scores = cached.get("records", {}) if cached.get("models") == fingerprint else {}
    detector = None
    for name, filename in PRIVATE.items():
        records = []
        rows = list(csv.DictReader(Path(filename).open(newline="")))
        if any(r["label"] != "0" for r in rows):
            raise RuntimeError("Only real calibration pools are allowed")
        for index, row in enumerate(rows, 1):
            key = sha(row["path"])
            if key not in private_scores:
                if detector is None:
                    detector = Detector(HEAD, edit_check=EDIT)
                data = Path(row["path"]).read_bytes()
                with Image.open(row["path"]) as image:
                    private_scores[key] = scalar(detector.analyze(image, file_bytes=data))
            records.append(private_scores[key])
            if index % 20 == 0:
                private_path.write_text(json.dumps({"models": fingerprint, "records": private_scores}))
                print(f"private calibration {name}: {index}/{len(rows)}", flush=True)
        pools[name] = records
    private_path.write_text(json.dumps({"models": fingerprint, "records": private_scores}))
    arrays = {name: {key: np.array([r[key] for r in records]) for key in ("whole", "edit1", "edit2", "face")}
              for name, records in pools.items()}
    base = {name: (p["whole"] > cal["whole_ai_logit"]) | p["face"] for name, p in arrays.items()}
    if any(int(mask.sum()) > int(np.floor(.05 * len(mask))) for mask in base.values()):
        raise RuntimeError("Fixed whole/face checks already exceed the shared budget")
    line1 = max(cal["edit_logit"], max(threshold_for(p["edit1"], cal["rate"]) for p in arrays.values()))
    line1 = joint_line([p["edit1"] for p in arrays.values()], list(base.values()), line1)
    line1 = float(np.nextafter(np.float32(line1), np.float32(np.inf)))
    first = {name: base[name] | (p["edit1"] > line1) for name, p in arrays.items()}
    line2 = max(cal["edit2_logit"], max(threshold_for(p["edit2"], cal["rate"]) for p in arrays.values()))
    line2 = joint_line([p["edit2"] for p in arrays.values()], list(first.values()), line2)
    line2 = float(np.nextafter(np.float32(line2), np.float32(np.inf)))
    report = {"scope": "Real calibration pools only; no training or test-set threshold selection",
              "old_lines": [cal["edit_logit"], cal["edit2_logit"]], "new_lines": [line1, line2],
              "private_records_exported": False, "pools": {}, "applied": False}
    for name, p in arrays.items():
        e1, e2 = p["edit1"] > line1, p["edit2"] > line2
        union = base[name] | e1 | e2
        n = len(union)
        assert int(e1.sum()) <= np.floor(cal["rate"] * n)
        assert int(e2.sum()) <= np.floor(cal["rate"] * n)
        assert int(union.sum()) <= np.floor(.05 * n)
        report["pools"][name] = {"n": n, "edit1": int(e1.sum()), "edit2": int(e2.sum()), "any_ai": int(union.sum())}
    if args.apply:
        cal.update(edit_logit=line1, edit2_logit=line2, calibration_precision="runtime_float32_logits",
                   runtime_calibration_device=str(detector.backbone.device) if detector else "cached verified runtime",
                   runtime_calibration_report=str(args.out_dir / "runtime_calibration.json"))
        cal_path.write_text(json.dumps(cal, indent=2))
        report["applied"] = True
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "runtime_calibration.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
