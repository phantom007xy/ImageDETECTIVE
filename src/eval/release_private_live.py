"""Offline full-detector checks of the two private calibration pools; aggregate output only.

Prepare without loading a model: python -m src.eval.release_private_live --prepare-only
Run after other GPU work ends: python -m src.eval.release_private_live
Private input paths stay in private CSVs; no per-photo predictions are published.
"""

import os

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import time


HEAD = "weights/google_v1/head_G3tt.pt"
EDIT = "weights/edit_check_v3/head_EC3pf.pt"
POOLS = (("private/calibration_phone", "data/private/release_calibration_phone.csv", 104),
         ("private/calibration_objects", "data/private/release_calibration_objects.csv", 100))
VERDICTS = ("AI-generated", "Likely AI-generated", "Likely AI-edited",
            "This face looks AI-made or AI-changed", "Uncertain", "Likely authentic")
NON_AI = {"Uncertain", "Likely authentic"}


def save(path, report):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def inputs():
    pools = {}
    for name, filename, expected in POOLS:
        with Path(filename).open(newline="", encoding="utf-8-sig") as stream:
            rows = list(csv.DictReader(stream))
        if len(rows) != expected or any(row.get("label") != "0" for row in rows):
            raise ValueError("Private calibration list count or labels do not match the expected pool.")
        if not all(Path(row["path"]).is_file() for row in rows):
            raise FileNotFoundError("Private calibration input files are missing.")
        pools[name] = rows
    return pools


def configuration(directory):
    frozen = json.loads((directory / "configuration.json").read_text())
    whole_path = Path(HEAD).with_name("head_G3tt_calibration.json")
    edit_path = Path(EDIT).with_name("head_EC3pf_calibration.json")
    whole, edit = json.loads(whole_path.read_text()), json.loads(edit_path.read_text())
    checked = {}
    for filename in (HEAD, EDIT, edit["head2"], whole["swap_head"], whole_path, edit_path):
        path = Path(filename)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if frozen["artifacts_and_code"].get(path.as_posix()) != digest:
            raise ValueError("Head/calibration configuration differs from the frozen release snapshot.")
        checked[path.as_posix()] = digest
    if not all(Path(filename).exists() for filename in (edit["pe_weights"], whole["swap_pe"])):
        raise FileNotFoundError("Required fine-tuned model weights are missing.")
    budgets = {"max_false_alarm_rate": whole["max_false_alarms"], "max_edit_flag_rate_per_head": edit["rate"],
               "whole_ai_logit": edit["whole_ai_logit"], "edit1_logit": edit["edit_logit"],
               "edit2_logit": edit["edit2_logit"], "strong_ai_logit": whole["strong_ai_logit"],
               "face_ai_logit": edit["face_ai_logit"], "swap_ai_logit": edit["swap_ai_logit"]}
    return frozen["sha256"], checked, budgets


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=Path("results/release_2026-11-04"))
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    destination = args.directory / "private_calibration_live.json"
    report = {"scope": "Current frozen detector on the original private calibration pools; neutral aggregate totals only.",
              "complete": False, "status": "preparing", "errors": 0, "groups": {},
              "external_apis_called": False, "backbone_weight_hashes_recomputed": False,
              "limitations": ["Calibration checks, not independent test performance.",
                              "Backbone weight hashes are taken from the existing frozen configuration; only heads/calibrations are rehashed here."]}
    try:
        pools = inputs()
        report["configuration_sha256"], report["verified_small_artifact_sha256"], report["calibration_budgets"] = configuration(args.directory)
        for name, rows in pools.items():
            report["groups"][name] = {"expected": len(rows), "processed": 0, "n": 0, "errors": 0,
                "error_types": {}, "pixel_verdict_counts": dict.fromkeys(VERDICTS, 0),
                "final_verdict_counts": dict.fromkeys(VERDICTS, 0), "pixel_ai": 0, "final_ai": 0,
                "strict_ai_generated_real": 0, "final_strict_ai_generated_real": 0,
                "component_flag_counts": dict.fromkeys(("whole", "edit1", "edit2", "any_face"), 0),
                "faces_checked": 0, "faces_red": 0, "complete": False}
        report["status"] = "prepared_not_started" if args.prepare_only else "loading_detector"
        save(destination, report)
        if args.prepare_only:
            print("Prepared private calibration checks: phone 104, objects 100; small-artifact hashes match; no model loaded.")
            return

        import torch
        from PIL import Image
        from src.inference import Detector

        torch.set_num_threads(4)
        detector = Detector(HEAD, edit_check=EDIT)
        if not (detector.calibrated and detector.edit_check and detector.finder and detector.face_pe
                and detector.swap_head and "head2" in detector.edit_check):
            raise RuntimeError("A required detector branch is missing.")
        report.update(status="running", device=str(detector.backbone.device), dtype=str(detector.backbone.dtype))
        started = time.perf_counter()
        for name, rows in pools.items():
            group = report["groups"][name]
            error_types = Counter()
            for index, row in enumerate(rows):
                try:
                    data = Path(row["path"]).read_bytes()
                    with Image.open(row["path"]) as image:
                        result = detector.analyze(image, file_bytes=data)
                    pixel, final = result["pixel_verdict"], result["verdict"]
                    group["n"] += 1
                    group["pixel_verdict_counts"][pixel] = group["pixel_verdict_counts"].get(pixel, 0) + 1
                    group["final_verdict_counts"][final] = group["final_verdict_counts"].get(final, 0) + 1
                    group["pixel_ai"] += int(pixel not in NON_AI)
                    group["final_ai"] += int(final not in NON_AI)
                    group["strict_ai_generated_real"] += int(pixel == "AI-generated")
                    group["final_strict_ai_generated_real"] += int(final == "AI-generated")
                    faces = [face for face in result["faces"] if not face["too_small"]]
                    components = {"whole": bool(result["whole_ai"]), "edit1": bool(result["edit"] and result["edit"]["flagged"]),
                                  "edit2": bool(result["edit"] and result["edit"].get("flagged2")),
                                  "any_face": any(face.get("flagged") for face in faces)}
                    assert any(components.values()) == (pixel not in NON_AI)
                    for key, flagged in components.items():
                        group["component_flag_counts"][key] += int(flagged)
                    group["faces_checked"] += len(faces)
                    group["faces_red"] += sum(bool(face.get("flagged")) for face in faces)
                except Exception as error:
                    error_types[type(error).__name__] += 1
                    group["errors"] += 1
                    report["errors"] += 1
                group["processed"] = index + 1
                group["error_types"] = dict(error_types)
                if (index + 1) % 20 == 0:
                    save(destination, report)
                    print(f"PROGRESS {name}: {index + 1}/{len(rows)}; errors={group['errors']}", flush=True)
            assert sum(group["pixel_verdict_counts"].values()) == group["n"]
            assert sum(group["final_verdict_counts"].values()) == group["n"]
            assert group["n"] + group["errors"] == group["expected"]
            group["complete"] = group["n"] == group["expected"] and group["errors"] == 0
            save(destination, report)
        report["seconds"] = time.perf_counter() - started
        report["complete"] = report["errors"] == 0 and all(group["complete"] for group in report["groups"].values())
        report["status"] = "complete" if report["complete"] else "complete_with_errors"
        save(destination, report)
        print("Private calibration run complete=" + str(report["complete"]) + "; errors=" + str(report["errors"]))
        if not report["complete"]:
            raise SystemExit(1)
    except Exception as error:
        report.update(status="blocked", complete=False, fatal_error_type=type(error).__name__)
        report["errors"] += 1
        save(destination, report)
        print("Private calibration run stopped; error type=" + type(error).__name__)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
