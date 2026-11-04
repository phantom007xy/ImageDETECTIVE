"""Offline, resumable robustness evaluation of the frozen release pixel checks.

Runs DINOv3 + both calibrated PE heads with the production joint thresholds.
Faces and file metadata are explicitly excluded. No API calls or calibration.

Run after other GPU work ends:
    python -m src.eval.release_robustness
    python -m src.eval.release_robustness --groups polished_real muse_v2_local

JSON/Markdown are aggregate reports; the separate public checkpoint supports
resume. Keep the same selection/configuration when resuming an output folder.
"""

import os

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import argparse
import csv
import hashlib
import json
import random
import re
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import torch
from PIL import Image, ImageOps

from src.eval.robustness_check import VERSIONS, read_list
from src.inference import Detector

HEAD = "weights/google_v1/head_G3tt.pt"
EDIT = "weights/edit_check_v3/head_EC3pf.pt"
GROUPS = {
    "polished_real": ("REAL", "features/lists/commons_dev.csv", "", "calibration/development"),
    "openimages_real": ("REAL", "features/lists/sid_reals_0_9_unseen.csv", "", "calibration/development"),
    "muse_v2_originals": ("REAL", "features/lists/muse_edits_test_v2.csv", "muse_edits/original", "development; unique source photos"),
    "muse_v2_local": ("AI", "features/lists/muse_edits_test_v2.csv", "muse_edits/local", "development/reused evaluation"),
    "muse_v2_whole": ("AI", "features/lists/muse_edits_test_v2.csv", "muse_edits/whole", "development/reused evaluation"),
    "muse_v1": ("AI", "features/lists/muse_image_test_v1.csv", "", "calibration/development"),
    "sdxl": ("AI", "features/lists/sd_faces.csv", "sd_faces/sdxl", "calibration/held-out generator"),
    "pixart": ("AI", "features/lists/open_gen_v1/open_gen_dev.csv", "open_gen/pixart", "held-out generator; shared training captions"),
    "sd35": ("AI", "features/lists/open_gen_v1/open_gen_dev.csv", "open_gen/sd35m_devcaption", "known generator; dev captions"),
}
CATEGORIES = ("AI", "Uncertain", "Likely authentic")


def now():
    return datetime.now(timezone.utc).isoformat()


def private(path):
    value = str(path).replace("\\", "/").lower()
    return any(token in value for token in ("/private/", "/private_", "/selfedits/", "muse_self_", "/self_train_"))


def sha_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)


def select_rows(name, n, seed):
    kind, csv_path, source, role = GROUPS[name]
    spec = csv_path + ("@" + source if source else "")
    details = {"kind": kind, "list": csv_path, "source_filter": source, "role": role}
    extra_files = []
    if name == "muse_v2_originals":
        # Different edit IDs store copies of the same original source photo.
        all_rows = read_list(spec, 10**9, seed)
        source_by_copy, by_source = {}, {}
        folders = {Path(row["path"]).parent for row in all_rows}
        for folder in sorted(folders):
            manifest = folder / "manifest.csv"
            if private(manifest):
                raise ValueError("Private input excluded from this public robustness evaluation")
            extra_files.append(str(manifest))
            with manifest.open(encoding="utf-8-sig", newline="") as stream:
                for row in csv.DictReader(stream):
                    if row.get("source") and row.get("id"):
                        if private(row["source"]):
                            raise ValueError("Private source excluded from this public robustness evaluation")
                        source_by_copy[(str(folder), row["id"])] = row["source"].replace("\\", "/").lower()
        for row in sorted(all_rows, key=lambda value: value["path"]):
            path = Path(row["path"])
            match = re.match(r"(edit_\d+)_original", path.stem)
            origin = source_by_copy.get((str(path.parent), match[1])) if match else None
            if origin is None:
                raise ValueError("Original-copy source mapping incomplete; refusing path-only deduplication")
            by_source.setdefault(origin, row)
        rows = random.Random(seed).sample(list(by_source.values()), min(n, len(by_source)))
        details.update(available_original_copy_paths=len(all_rows), available_unique_sources=len(by_source),
                       deduplication="original source path from the edit manifest")
    else:
        rows = read_list(spec, n, seed)
        details["deduplication"] = "unique image path, existing robustness_check.read_list"
    if not rows:
        raise ValueError(f"Empty public evaluation group: {name}")
    if any(private(row["path"]) for row in rows):
        raise ValueError("Private input excluded from this public robustness evaluation")
    labels = {int(row["label"]) for row in rows}
    if (kind == "REAL" and labels != {0}) or (kind == "AI" and 0 in labels):
        raise ValueError(f"Unexpected label mix in public evaluation group: {name}")
    details["selected"] = len(rows)
    return rows, details, extra_files


def configuration(args, selected, extra_files):
    head, edit = Path(args.head), Path(args.edit_check)
    head_cal = head.with_name(head.stem + "_calibration.json")
    edit_cal = edit.with_name(edit.stem + "_calibration.json")
    calibration = json.loads(edit_cal.read_text())
    if not calibration.get("head2") or not calibration.get("pe_weights"):
        raise ValueError("This release robustness check requires both calibrated PE heads and tuned PE weights")
    files = {str(head), str(edit), str(head_cal), str(edit_cal), calibration["head2"], calibration["pe_weights"],
             "src/inference.py", "src/data/preprocess.py", "src/models/backbone.py", "src/models/pe_core.py",
             "src/models/linear_head.py", "src/eval/robustness_check.py", "src/eval/release_robustness.py"}
    files.update(GROUPS[name][1] for name in args.groups)
    files.update(extra_files)
    hashes = {str(Path(path).as_posix()): sha_file(path) for path in sorted(files)}
    config = {"schema_version": 1, "head": str(head.as_posix()), "edit_check": str(edit.as_posix()),
              "n": args.n, "seed": args.seed, "groups": args.groups, "versions": args.versions,
              "selected_public_paths": {name: [row["path"].replace("\\", "/") for row in rows]
                                        for name, rows in selected.items()}, "artifact_sha256": hashes,
              "scope": "whole-image DINOv3 + Muse PE head + second PE head; faces=False; no metadata; no APIs"}
    fingerprint = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    return config, fingerprint


def case_key(group, path, version):
    return hashlib.sha256(json.dumps([group, path, version]).encode()).hexdigest()


def summarize(state, selected, details, versions, runtime, started):
    groups = {}
    total_requested = total_completed = total_errors = 0
    for name, rows in selected.items():
        versions_out = {}
        for version in versions:
            keys = [case_key(name, row["path"], version) for row in rows]
            predictions = [state["cases"][key] for key in keys if key in state["cases"]]
            errors = [state["errors"][key] for key in keys if key in state["errors"]]
            counts = Counter(value["category"] for value in predictions)
            successful = len(predictions)
            versions_out[version] = {
                "requested": len(rows), "completed": successful, "errors": len(errors),
                "remaining": len(rows) - successful - len(errors),
                "counts": {category: counts[category] for category in CATEGORIES},
                "ai": counts["AI"], "uncertain": counts["Uncertain"], "authentic": counts["Likely authentic"],
                "ai_rate_on_successful": counts["AI"] / successful if successful else None,
                "uncertain_rate_on_successful": counts["Uncertain"] / successful if successful else None,
                "authentic_rate_on_successful": counts["Likely authentic"] / successful if successful else None,
                "whole_ai": sum(value["whole_ai"] for value in predictions),
                "pe_edit_flags": sum(value["pe_edit"] for value in predictions),
                "pe_second_flags": sum(value["pe_second"] for value in predictions),
                "verdict_counts": dict(Counter(value["verdict"] for value in predictions)),
                "error_examples": errors[:3],
            }
            total_requested += len(rows)
            total_completed += successful
            total_errors += len(errors)
        groups[name] = {**details[name], "versions": versions_out}
    remaining = total_requested - total_completed - total_errors
    status = "running/incomplete" if remaining else "complete_with_errors" if total_errors else "complete"
    return {"updated_at_utc": now(), "started_at_utc": started, "status": status,
            "complete": remaining == 0 and total_errors == 0,
            "configuration_fingerprint": state["configuration_fingerprint"],
            "scope": state["configuration"]["scope"],
            "configuration": state["configuration"], "runtime": runtime,
            "progress": {"requested": total_requested, "completed": total_completed,
                         "errors": total_errors, "remaining": remaining}, "groups": groups,
            "interpretation": [
                "REAL AI flags are false alarms; AI authentic verdicts are misses. Uncertain remains separate.",
                "All seven transformations reuse the existing VERSIONS implementation. WhatsApp is a JPEG/resize approximation, not an actual WhatsApp service round trip.",
                "The production joint thresholds and both PE heads are frozen. Faces and metadata are excluded, so these are not full-app verdicts.",
                "The no-face model uses thresholds calibrated for the full app, without recalibration. The constructor may load the calibrated face PE although no face is found or scored.",
                "Groups and versions overlap by source; no pooled or overall accuracy is computed.",
                "Most groups were used during development or calibration. This is a robustness regression, not a newly independent final test.",
                "Resume binds code, model/calibration bytes, source-list bytes and selected paths. Public source images are assumed unchanged in place; their bytes are not all hashed here."]}


def save_outputs(out, state, selected, details, versions, runtime, started):
    report = summarize(state, selected, details, versions, runtime, started)
    atomic_json(out / "robustness_checkpoint.json", state)
    atomic_json(out / "robustness.json", report)
    p = report["progress"]
    lines = ["# Release robustness check", "", f"Status: **{report['status']}**.", "",
             "Scope: frozen DINOv3 whole-image check + both PE heads, production joint thresholds. No faces, metadata, private photos, APIs, or recalibration.",
             f"Progress: {p['completed']}/{p['requested']} successful transformed-image checks; {p['errors']} errors; {p['remaining']} remaining.", "",
             "Counts and percentages use successful checks only; missing/error cases are shown separately. No overall accuracy is computed.", "",
             "| Group | Kind | Version | Successful/requested | AI | Uncertain | Authentic | Errors |",
             "|---|---|---|---:|---:|---:|---:|---:|"]
    for name, group in report["groups"].items():
        for version, value in group["versions"].items():
            def cell(count):
                return f"{count} ({count / value['completed']:.1%})" if value["completed"] else "0 (pending)"
            lines.append(f"| {name} | {group['kind']} | {version} | {value['completed']}/{value['requested']} | "
                         f"{cell(value['ai'])} | {cell(value['uncertain'])} | {cell(value['authentic'])} | {value['errors']} |")
    lines += ["", "## Interpretation", ""] + ["- " + line for line in report["interpretation"]]
    lines += ["", "Muse-v2 originals are deduplicated by original source photo, rather than the repeated original-copy paths in the edit list.",
              f"Configuration fingerprint: `{state['configuration_fingerprint']}`."]
    temporary = out / "robustness.md.tmp"
    temporary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    temporary.replace(out / "robustness.md")
    return report


def gate_exit_code(report):
    """A successful CLI exit means all requested cases passed accounting checks."""
    totals = Counter(requested=0, completed=0, errors=0, remaining=0)
    for group in report["groups"].values():
        for version in group["versions"].values():
            values = [version[key] for key in ("requested", "completed", "errors", "remaining")]
            if any(not isinstance(value, int) or value < 0 for value in values):
                return 1
            if version["completed"] + version["errors"] + version["remaining"] != version["requested"]:
                return 1
            if sum(version["counts"].values()) != version["completed"]:
                return 1
            if version["ai"] + version["uncertain"] + version["authentic"] != version["completed"]:
                return 1
            totals.update({key: version[key] for key in totals})
    if dict(totals) != report["progress"]:
        return 1
    return 0 if report["complete"] and not totals["errors"] and not totals["remaining"] else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--head", default=HEAD)
    parser.add_argument("--edit-check", default=EDIT)
    parser.add_argument("--n", type=int, default=150)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--groups", nargs="+", choices=list(GROUPS), default=list(GROUPS))
    parser.add_argument("--versions", nargs="+", choices=list(VERSIONS), default=list(VERSIONS))
    parser.add_argument("--out-dir", type=Path, default=Path("results/release_2026-11-04"))
    parser.add_argument("--device", default=None)
    parser.add_argument("--retry-errors", action="store_true")
    args = parser.parse_args()
    if args.n < 1:
        parser.error("--n must be positive")
    torch.set_num_threads(4)
    args.groups = list(dict.fromkeys(args.groups))
    args.versions = list(dict.fromkeys(args.versions))
    selected, details, extra_files = {}, {}, []
    for name in args.groups:
        selected[name], details[name], extras = select_rows(name, args.n, args.seed)
        extra_files.extend(extras)
    config, fingerprint = configuration(args, selected, extra_files)
    out = args.out_dir
    checkpoint = out / "robustness_checkpoint.json"
    if checkpoint.exists():
        state = json.loads(checkpoint.read_text())
        if state.get("configuration_fingerprint") != fingerprint:
            raise SystemExit("Robustness checkpoint configuration differs. Use a separate --out-dir; do not mix runs.")
        if args.retry_errors:
            state["errors"] = {}
    else:
        state = {"configuration_fingerprint": fingerprint, "configuration": config,
                 "started_at_utc": now(), "cases": {}, "errors": {}}
    started = state["started_at_utc"]
    runtime = state.get("runtime", {"device": args.device or "automatic", "models_loaded": False})
    report = save_outputs(out, state, selected, details, args.versions, runtime, started)
    if not report["progress"]["remaining"]:
        print(json.dumps({"status": report["status"], **report["progress"]}), flush=True)
        raise SystemExit(gate_exit_code(report))
    detector = Detector(args.head, device=args.device, faces=False, edit_check=args.edit_check)
    if not detector.calibrated or detector.edit_check is None or "head2" not in detector.edit_check:
        raise RuntimeError("Full calibrated whole-image + both PE heads required; refusing a degraded run")
    runtime = {"device": str(detector.backbone.device), "dtype": str(detector.backbone.dtype),
               "torch_num_threads": torch.get_num_threads(),
               "models_loaded": True, "faces": False, "metadata": False,
               "whole_ai_logit": detector.calibration["ai_logit"],
               "real_logit": detector.calibration["real_logit"],
               "strong_ai_logit": detector.calibration["strong_ai_logit"],
               "pe_edit_line": detector.edit_check["line"], "pe_second_line": detector.edit_check["line2"]}
    state["runtime"] = runtime
    save_outputs(out, state, selected, details, args.versions, runtime, started)
    touched, clock = 0, time.monotonic()
    try:
        for name, rows in selected.items():
            for index, row in enumerate(rows, 1):
                pending = [version for version in args.versions if
                           case_key(name, row["path"], version) not in state["cases"] and
                           case_key(name, row["path"], version) not in state["errors"]]
                if not pending:
                    continue
                try:
                    with Image.open(row["path"]) as opened:
                        image = ImageOps.exif_transpose(opened).convert("RGB")
                except Exception as error:
                    for version in pending:
                        state["errors"][case_key(name, row["path"], version)] = {
                            "path": row["path"].replace("\\", "/"), "version": version, "type": type(error).__name__,
                            "stage": "open_public_image"}
                else:
                    for version in pending:
                        key = case_key(name, row["path"], version)
                        try:
                            damaged = VERSIONS[version](image)
                            result = detector.analyze(damaged)  # no original bytes => no metadata check
                            if result["faces"] or result["metadata"] is not None:
                                raise RuntimeError("Unexpected face or metadata result in the pixel-only evaluation")
                            verdict = result["pixel_verdict"]
                            category = verdict if verdict in ("Uncertain", "Likely authentic") else "AI"
                            state["cases"][key] = {"category": category, "verdict": verdict,
                                                   "whole_ai": bool(result["whole_ai"]),
                                                   "pe_edit": bool(result["edit"]["flagged"]),
                                                   "pe_second": bool(result["edit"].get("flagged2")),
                                                   "seconds": float(result["seconds"])}
                        except Exception as error:
                            state["errors"][key] = {"path": row["path"].replace("\\", "/"),
                                                    "version": version, "type": type(error).__name__,
                                                    "stage": "transform_or_inference"}
                touched += 1
                if touched % 20 == 0 or index == len(rows):
                    report = save_outputs(out, state, selected, details, args.versions, runtime, started)
                    print(json.dumps({"group": name, "pictures": f"{index}/{len(rows)}",
                                      **report["progress"], "elapsed_seconds_this_session": round(time.monotonic() - clock, 1)}), flush=True)
    finally:
        report = save_outputs(out, state, selected, details, args.versions, runtime, started)
        print(json.dumps({"status": report["status"], **report["progress"]}), flush=True)
    raise SystemExit(gate_exit_code(report))


if __name__ == "__main__":
    main()
