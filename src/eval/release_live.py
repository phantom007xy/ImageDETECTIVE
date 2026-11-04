"""Offline, resumable release evaluation using the complete detector and real upload path.

Writes aggregate public results, including neutral private-group totals. Private per-image
checkpoints stay under data/private. Never calls the investigator or any paid service.
This records a frozen configuration; it does not train or recalibrate anything.
"""

import os

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import argparse
import csv
import hashlib
import importlib
import io
import json
import statistics
import time
import warnings
from collections import Counter, defaultdict
from pathlib import Path
from unittest.mock import patch

import torch
from PIL import Image

from src.inference import Detector

HEAD = "weights/google_v1/head_G3tt.pt"
EDIT = "weights/edit_check_v3/head_EC3pf.pt"
OUT = Path("results/release_2026-11-04")
PRIVATE_OUT = Path("data/private/release_2026-11-04")
NON_AI = {"Likely authentic", "Uncertain"}
CASES = [
    ("judge_ai", "features/lists/judge_test_ai_v1.csv", "", False, "regression"),
    ("fresh_muse_v2", "features/lists/muse_image_test_v2.csv", "", False, "held-out evaluation"),
    ("private_judge", "data/private/judge_test_real_v1.csv", "", True, "reused regression"),
    ("polished_real", "features/lists/commons_dev.csv", "", False, "calibration"),
    ("openimages_real", "features/lists/sid_reals_0_9_unseen.csv", "", False, "calibration"),
    ("phone_real_public", "features/lists/commons_v3_dev.csv", "", False, "calibration"),
    ("selfies_real_public", "features/lists/commons_v4_dev.csv", "", False, "calibration"),
    ("closeups_real", "features/lists/real_face_crops_dev_all.csv", "", False, "calibration"),
    ("muse_pairs", "features/lists/muse_pairs_val.csv", "", False, "calibration/development"),
    ("muse_edits_v2", "features/lists/muse_edits_test_v2.csv", "", False, "development/reused evaluation"),
    ("muse_edits_v1", "features/lists/muse_edits_v1.csv", "", False, "development/reused evaluation"),
    ("muse_v1", "features/lists/muse_image_test_v1.csv", "", False, "calibration/development"),
    ("sdxl", "features/lists/sd_faces.csv", "sd_faces/sdxl", False, "calibration/held-out generator"),
    ("open_generators", "features/lists/open_gen_v1/open_gen_dev.csv", "", False, "development"),
    ("gan_tpdne", "features/lists/tpdne_test.csv", "", False, "reused evaluation"),
    ("gan_sfhq", "features/lists/sfhq_pt3_test.csv", "", False, "reused evaluation"),
    ("gemini_dev", "features/lists/gemini_dev_v1.csv", "", False, "development"),
    ("google_captions_dev", "features/lists/google_gen_v1/open_gen_dev.csv", "", False, "development"),
    ("commons_ai_photos", "features/lists/commons_ai_v1_photo.csv", "", False, "reused evaluation"),
]


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)


def sha_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def snapshot():
    whole = json.loads(Path(HEAD).with_name("head_G3tt_calibration.json").read_text())
    edit = json.loads(Path(EDIT).with_name("head_EC3pf_calibration.json").read_text())
    artifacts = [HEAD, EDIT, edit["head2"], edit["pe_weights"], whole["swap_head"], whole["swap_pe"],
                 "weights/google_v1/head_G3tt_calibration.json",
                 "weights/edit_check_v3/head_EC3pf_calibration.json"]
    code = sorted(str(p) for folder in ("src", "app") for p in Path(folder).rglob("*.py")
                  if "eval" not in p.parts)
    hashes = {str(Path(p).as_posix()): sha_file(p) for p in artifacts + code}
    return {"sha256": hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest(),
            "artifacts_and_code": hashes}


def compact(result):
    checked = [f for f in result["faces"] if not f["too_small"]]
    edit = result["edit"]
    return {
        "verdict": result["verdict"], "pixel_verdict": result["pixel_verdict"],
        "score": float(result["score"]), "probability_ai": float(result["probability_ai"]),
        "whole_ai": bool(result["whole_ai"]), "metadata_label": (result["metadata"] or {}).get("ai_label"),
        "edit": None if edit is None else {k: bool(v) if k.startswith("flagged") else float(v)
                                           for k, v in edit.items()},
        "faces": len(result["faces"]), "faces_checked": len(checked),
        "faces_flagged": sum(bool(f.get("flagged")) for f in checked),
        "face_levels": [f.get("level") for f in checked], "face_kinds": [f.get("kind") for f in checked],
        "seconds": float(result["seconds"]),
    }


def ui_numbers(result):
    return {"verdict": result["verdict"], "probability_ai": round(result["probability_ai"], 6),
            "edit": None if result["edit"] is None else [result["edit"]["flagged"], result["edit"].get("flagged2", False)],
            "metadata_label": result["metadata_label"], "faces_checked": result["faces_checked"],
            "faces_flagged": result["faces_flagged"]}


def response_numbers(body):
    return {"verdict": body["verdict"], "probability_ai": round(body["probability_ai"], 6),
            "edit": None if body["edit"] is None else [body["edit"]["flagged"], body["edit"]["flagged2"]],
            "metadata_label": body["metadata"]["ai_label"], "faces_checked": body["faces_checked"],
            "faces_flagged": body["faces_flagged"]}


def group_summary(entries):
    out = {}
    for source, rows in sorted(entries.items()):
        predictions = [r["prediction"] for r in rows if "prediction" in r]
        counts = Counter(r["pixel_verdict"] for r in predictions)
        out[source] = {
            "requested": len(rows), "n": len(predictions), "errors": len(rows) - len(predictions),
            "unique_files": len({r["key"] for r in rows if "key" in r}),
            "labels": dict(Counter("real" if r["label"] == 0 else "ai" for r in rows)),
            "pixel_verdict_counts": dict(counts),
            "pixel_ai": sum(r["pixel_verdict"] not in NON_AI for r in predictions),
            "final_ai": sum(r["verdict"] not in NON_AI for r in predictions),
            "ai_file_labels": sum(bool(r["metadata_label"]) for r in predictions),
            "faces_checked": sum(r["faces_checked"] for r in predictions),
            "faces_flagged": sum(r["faces_flagged"] for r in predictions),
            "median_seconds": statistics.median(r["seconds"] for r in predictions) if predictions else None,
        }
    return out


def main():
    global OUT, PRIVATE_OUT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=OUT)
    parser.add_argument("--private-out-dir", type=Path, default=PRIVATE_OUT)
    parser.add_argument("--cases", nargs="+", help="selected case names (default: all)")
    parser.add_argument("--ui", action="store_true", help="also rerun all 80 judge images through the real HTTP upload path")
    args = parser.parse_args()
    OUT, PRIVATE_OUT = args.out_dir, args.private_out_dir
    torch.set_num_threads(4)
    warnings.filterwarnings("ignore")
    OUT.mkdir(parents=True, exist_ok=True)
    frozen = snapshot()
    write_json(OUT / "configuration.json", frozen)
    caches = {}
    for private, directory in ((False, OUT), (True, PRIVATE_OUT)):
        path = directory / "live_predictions.json"
        saved = json.loads(path.read_text()) if path.exists() else {}
        caches[private] = saved.get("records", {}) if saved.get("configuration_sha256") == frozen["sha256"] else {}
    print(f"frozen configuration {frozen['sha256'][:16]}; loading full detector offline", flush=True)
    loaded = time.perf_counter()
    detector = Detector(HEAD, edit_check=EDIT)
    if not (detector.calibrated and detector.edit_check and detector.finder and detector.face_pe and detector.swap_head
            and "head2" in detector.edit_check):
        raise RuntimeError("An expected detector branch is missing; release evaluation stopped")
    report = {"configuration_sha256": frozen["sha256"], "complete": False, "cases": {},
              "device": str(detector.backbone.device), "dtype": str(detector.backbone.dtype),
              "load_seconds": time.perf_counter() - loaded, "ui": {"requested": args.ui, "complete": False}}
    lineage_file = OUT / "metadata_refresh.json"
    if lineage_file.exists():
        lineage = json.loads(lineage_file.read_text())
        if lineage["current_configuration_sha256"] == frozen["sha256"]:
            report["pixel_measurement_reuse"] = lineage
    selected = [c for c in CASES if not args.cases or c[0] in args.cases]
    if args.cases and set(args.cases) - {c[0] for c in selected}:
        parser.error("Unknown case name")
    judge = []

    def save():
        for private, directory in ((False, OUT), (True, PRIVATE_OUT)):
            write_json(directory / "live_predictions.json", {"configuration_sha256": frozen["sha256"],
                                                            "records": caches[private]})
        write_json(OUT / "live_metrics.json", report)

    for name, list_path, prefix, private, role in selected:
        print(f"START {name}", flush=True)
        if not Path(list_path).exists():
            report["cases"][name] = {"role": role, "error": "required list missing"}
            save()
            continue
        rows = list(csv.DictReader(Path(list_path).open(newline="", encoding="utf-8-sig")))
        rows = [r for r in rows if not prefix or r["source"].startswith(prefix)]
        grouped = defaultdict(list)
        for index, row in enumerate(rows):
            source = row["source"] if not private else (
                "private/selfies" if "self" in row["source"].lower() else "private/objects")
            item = {"label": int(row["label"])}
            try:
                data = Path(row["path"]).read_bytes()
                key = hashlib.sha256(data).hexdigest()
                item["key"] = key
                if key not in caches[private]:
                    with Image.open(io.BytesIO(data)) as image:
                        caches[private][key] = compact(detector.analyze(image, file_bytes=data))
                item["prediction"] = caches[private][key]
                if name == "judge_ai":
                    judge.append((row["path"], key))
            except Exception as error:
                item["error_type"] = type(error).__name__
            grouped[source].append(item)
            if (index + 1) % 20 == 0:
                report["cases"][name] = {"role": role, "complete": False, "groups": group_summary(grouped)}
                save()
                print(f"PROGRESS {name} {index + 1}/{len(rows)}", flush=True)
        report["cases"][name] = {"role": role, "complete": True, "groups": group_summary(grouped)}
        save()
        print("RESULT " + name + " " + json.dumps(report["cases"][name]), flush=True)

    if args.ui:
        from fastapi.testclient import TestClient
        # Reuse this exact loaded detector; importing a second full detector wastes laptop VRAM.
        os.environ.pop("HEADS_REPO", None)
        with patch("src.inference.Detector", return_value=detector):
            server = importlib.import_module("app.server")
        mismatches, checked = [], 0
        with TestClient(server.app) as client:
            for path, key in judge:
                response = client.post("/api/investigate", files={"file": (Path(path).name, Path(path).read_bytes())})
                expected = ui_numbers(caches[False][key])
                if response.status_code == 200:
                    body = response.json()
                    actual = response_numbers(body)
                    if expected != actual:
                        mismatches.append({"public_file": Path(path).name, "expected": expected, "actual": actual})
                else:
                    mismatches.append({"public_file": Path(path).name, "http_status": response.status_code})
                checked += 1
                # Full-resolution follow-up images are unnecessary for this equivalence test.
                with server.memory_lock:
                    server.memory.clear()
                if checked % 20 == 0:
                    print(f"PROGRESS http_upload {checked}/{len(judge)}", flush=True)
            if "commons_ai_photos" in report["cases"]:
                with Path("features/lists/commons_ai_v1_photo.csv").open(newline="", encoding="utf-8-sig") as stream:
                    fragmented = [row for row in csv.DictReader(stream) if row["source"] == "commons_ai/firefly"]
                fragment_errors = []
                for row in fragmented:
                    data = Path(row["path"]).read_bytes()
                    response = client.post("/api/investigate", files={"file": (Path(row["path"]).name, data)})
                    expected = ui_numbers(caches[False][hashlib.sha256(data).hexdigest()])
                    if response.status_code != 200 or response_numbers(response.json()) != expected:
                        fragment_errors.append({"public_file": Path(row["path"]).name, "status": response.status_code})
                    with server.memory_lock:
                        server.memory.clear()
                report["metadata_http"] = {"checked": len(fragmented), "errors": fragment_errors,
                                           "passed": len(fragmented) == 2 and not fragment_errors}
        report["ui"] = {"requested": True, "complete": True, "checked": checked,
                        "mismatches": mismatches, "passed": checked == 80 and not mismatches}
        print("RESULT http_upload " + json.dumps(report["ui"]), flush=True)
    report["complete"] = True
    if torch.cuda.is_available():
        report["gpu_peak_allocated_gib"] = torch.cuda.max_memory_allocated() / 2**30
        report["gpu_peak_reserved_gib"] = torch.cuda.max_memory_reserved() / 2**30
    report["errors"] = sum(sum(g["errors"] for g in case.get("groups", {}).values())
                           + int("error" in case) for case in report["cases"].values())
    save()
    print(f"DONE: {len(selected)} cases; errors={report['errors']}; results in {OUT.as_posix()}", flush=True)
    raise SystemExit(1 if report["errors"] or (args.ui and not report["ui"].get("passed"))
                     or report.get("metadata_http", {}).get("passed") is False else 0)


if __name__ == "__main__":
    main()
