"""Reparse metadata after a metadata-only fix, preserving measured pixel-score provenance.

Requires a completed GPU run and exact byte equality of every frozen artifact
except src/services/metadata.py. This does not rerun models or claim new pixel
measurements. Run release_live --ui afterwards for fresh HTTP/model equivalence.
"""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil

from src.eval.release_live import CASES, group_summary, snapshot, write_json
from src.services import metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=Path("results/hardened_2026-11-04"))
    parser.add_argument("--private-directory", type=Path, default=Path("data/private/hardened_2026-11-04"))
    args = parser.parse_args()
    out = args.directory
    old = json.loads((out / "configuration.json").read_text())
    current = snapshot()
    changed = [name for name in old["artifacts_and_code"]
               if old["artifacts_and_code"][name] != current["artifacts_and_code"].get(name)]
    if set(current["artifacts_and_code"]) != set(old["artifacts_and_code"]) or changed != ["src/services/metadata.py"]:
        raise RuntimeError("Metadata-only refresh requires exactly one changed production artifact: metadata.py")
    live = json.loads((out / "live_metrics.json").read_text())
    if not live["complete"] or live["errors"] or not live["ui"]["passed"]:
        raise RuntimeError("A complete, error-free GPU/HTTP measurement must precede metadata refresh")
    backup = out / "pixel_run_before_metadata_fix"
    backup.mkdir(exist_ok=True)
    for filename in ("configuration.json", "live_metrics.json", "live_predictions.json"):
        if (backup / filename).exists():
            if (out / filename).read_bytes() != (backup / filename).read_bytes():
                raise RuntimeError("Existing preserved GPU run differs; use a separate output directory")
        else:
            shutil.copyfile(out / filename, backup / filename)
    private_backup = args.private_directory / "pixel_run_before_metadata_fix"
    private_backup.mkdir(exist_ok=True)
    if (private_backup / "live_predictions.json").exists():
        if (args.private_directory / "live_predictions.json").read_bytes() != (private_backup / "live_predictions.json").read_bytes():
            raise RuntimeError("Existing preserved private run differs")
    else:
        shutil.copyfile(args.private_directory / "live_predictions.json", private_backup / "live_predictions.json")
    caches = {private: json.loads((directory / "live_predictions.json").read_text())["records"]
              for private, directory in ((False, out), (True, args.private_directory))}
    seen = set()
    labels_changed = 0
    entries = 0
    for name, filename, prefix, private, role in CASES:
        with Path(filename).open(encoding="utf-8-sig", newline="") as stream:
            rows = [row for row in csv.DictReader(stream) if not prefix or row["source"].startswith(prefix)]
        groups = {}
        for row in rows:
            data = Path(row["path"]).read_bytes()
            key = hashlib.sha256(data).hexdigest()
            prediction = caches[private][key]
            if (private, key) not in seen:
                declaration = metadata.check_file(data)["ai_label"]
                labels_changed += int(declaration != prediction["metadata_label"])
                verdict = prediction["pixel_verdict"]
                if declaration == "generated":
                    verdict = "AI-generated"
                elif declaration == "edited" and "AI-generated" not in verdict:
                    verdict = "Part of this photo looks AI-made or AI-changed"
                prediction.update(verdict=verdict, metadata_label=declaration)
                seen.add((private, key))
            source = row["source"] if not private else (
                "private/selfies" if "self" in row["source"].lower() else "private/objects")
            groups.setdefault(source, []).append({"key": key, "label": int(row["label"]), "prediction": prediction})
            entries += 1
        refreshed = group_summary(groups)
        numeric_fields = ("n", "unique_files", "labels", "pixel_verdict_counts", "pixel_ai", "faces_checked", "faces_flagged")
        if set(refreshed) != set(live["cases"][name]["groups"]) or any(
                refreshed[source][field] != live["cases"][name]["groups"][source][field]
                for source in refreshed for field in numeric_fields):
            raise RuntimeError("Input partitions/pixel measurements changed; full model rerun required")
        live["cases"][name] = {"role": role, "complete": True, "groups": refreshed}
    lineage = {"pixel_measurement_configuration_sha256": old["sha256"],
               "current_configuration_sha256": current["sha256"], "changed_production_artifacts": changed,
               "pixel_scoring_artifacts_byte_identical": True, "metadata_entries_reparsed": entries,
               "unique_file_records_reparsed": len(seen), "file_declarations_changed": labels_changed,
               "scope": "Reuse the completed GPU pixel/face/edit scores; reparse original file metadata locally. No new model measurements claimed.",
               "fresh_http_equivalence_required": True, "private_per_photo_records_exported": False}
    live.update(configuration_sha256=current["sha256"], complete=False,
                ui={"requested": True, "complete": False}, metadata_refresh=lineage)
    for private, directory in ((False, out), (True, args.private_directory)):
        write_json(directory / "live_predictions.json", {"configuration_sha256": current["sha256"], "records": caches[private]})
    write_json(out / "configuration.json", current)
    write_json(out / "live_metrics.json", live)
    write_json(out / "metadata_refresh.json", lineage)
    print(json.dumps(lineage))


if __name__ == "__main__":
    main()
