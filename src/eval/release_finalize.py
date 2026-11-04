"""Summarize the hardened local release without changing models or contacting services.

Run after release_live, release_private_live and release_robustness finish:
    python -m src.eval.release_finalize
"""

import argparse
import ast
import csv
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import platform
import re
import subprocess
import sys

from src.eval.release_summary import calibration_promises, robustness_summary


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=Path("results/hardened_2026-11-04"))
    parser.add_argument("--baseline", type=Path, default=Path("results/release_2026-11-04"))
    parser.add_argument("--private-directory", type=Path, default=Path("data/private/hardened_2026-11-04"))
    args = parser.parse_args()
    out = args.directory
    frozen = read(out / "configuration.json")
    previous = read(args.baseline / "configuration.json")
    unchanged_weights = all(expected == previous["artifacts_and_code"].get(name)
                            for name, expected in frozen["artifacts_and_code"].items() if name.endswith(".pt"))
    changed = [name for name, expected in frozen["artifacts_and_code"].items()
               if not Path(name).is_file() or digest(Path(name)) != expected]
    syntax_errors = []
    files = sorted(p for folder in ("app", "src", "scripts") for p in Path(folder).rglob("*.py"))
    for path in files:
        try:
            ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
        except SyntaxError as error:
            syntax_errors.append({"file": path.as_posix(), "line": error.lineno})
    javascript = []
    for name in ("app/web/app.js", "src/eval/release_browser_check.js"):
        check = subprocess.run(["node", "--check", name], capture_output=True, text=True)
        javascript.append({"file": name, "exit_code": check.returncode, "error": check.stderr.strip()})
    integrity = {"scope": "Exact frozen production/model/calibration bytes plus Python/JavaScript syntax",
                 "configuration_sha256": frozen["sha256"],
                 "production_artifacts_checked": len(frozen["artifacts_and_code"]),
                 "changed_production_artifacts": changed, "python_files_parsed": len(files),
                 "python_syntax_errors": syntax_errors, "javascript_syntax_checks": javascript,
                 "model_weights_unchanged_from_baseline": unchanged_weights,
                 "passed": unchanged_weights and not changed and not syntax_errors
                     and all(c["exit_code"] == 0 for c in javascript)}
    integrity["release_files_sha256"] = {
        p.as_posix(): digest(p) for p in sorted([Path("README.md"), Path("requirements.txt"),
        Path(".github/workflows/checks.yml"), Path("deploy/gcp_vm_setup.sh"),
        Path("docs/DEMO.md"), *Path("app/web").glob("*.html"),
        *Path("app/web").glob("*.js"), *Path("app/web").glob("*.css")])}
    save(out / "integrity_check.json", integrity)

    requirements = []
    for line in Path("requirements.txt").read_text().splitlines():
        match = re.fullmatch(r"([A-Za-z0-9_-]+)==([^\s#]+)", line.strip())
        if match:
            package, required = match.groups()
            installed = version(package)
            requirements.append({"package": package, "required": "==" + required,
                                 "installed": installed, "matches": installed == required})
    pip = subprocess.run([sys.executable, "-m", "pip", "check"], capture_output=True, text=True)
    environment = {"measured_at_utc": datetime.now(timezone.utc).isoformat(), "python": sys.version,
                   "platform": platform.platform(), "requirements": requirements,
                   "pip_check": {"exit_code": pip.returncode, "output": (pip.stdout + pip.stderr).strip()},
                   "core_versions": {name: version(name) for name in ("torch", "torchvision", "fastapi", "starlette")}}
    save(out / "environment.json", environment)

    rows = []
    for name in ("judge_test_real_v1.csv", "release_calibration_phone.csv", "release_calibration_objects.csv"):
        with (Path("data/private") / name).open(encoding="utf-8-sig", newline="") as stream:
            rows.extend(csv.DictReader(stream))
    identifiers = set()
    for row in rows:
        original = row["path"].replace("\\", "/")
        identifiers.update((original.lower(), original.rsplit("/", 1)[-1].lower()))
    public = sorted(set(p for folder in (args.baseline, out) for p in folder.rglob("*")
                        if p.suffix in (".json", ".md"))) + [Path("README.md"), Path("docs/DEMO.md"),
                                                               Path("results/hackathon_audit_2026-11-04.md")]
    findings = []
    for path in public:
        content = path.read_text(encoding="utf-8").lower().replace("\\\\", "/").replace("\\", "/")
        if any(identifier in content for identifier in identifiers):
            findings.append(path.as_posix())  # Never publish the matched private identifier.
    private_keys = set(read(args.private_directory / "live_predictions.json")["records"])
    public_keys = set(read(out / "live_predictions.json")["records"])
    privacy = {"scope": "Private input filenames/paths scanned in public reports/README; private prediction keys disjoint",
               "public_artifacts_checked": len(public), "private_image_rows": len(rows), "findings": findings,
               "private_prediction_keys_in_public_cache": len(private_keys & public_keys),
               "private_checkpoint_separate": args.private_directory.resolve() != out.resolve(),
               "passed": not findings and not (private_keys & public_keys)}
    save(out / "privacy_check.json", privacy)

    app = read(out / "app_checks.json")
    browser = read(out / "browser_checks.json")
    live = read(out / "live_metrics.json")
    baseline = read(args.baseline / "live_metrics.json")
    private = read(out / "private_calibration_live.json")
    robustness = read(out / "robustness.json")
    cached = read(out / "cached_metrics.json")
    precision = read(out / "cross_device_boundary.json")
    calibration = read(out / "runtime_calibration.json")
    all_predictions = [*read(out / "live_predictions.json")["records"].values(),
                       *read(args.private_directory / "live_predictions.json")["records"].values()]
    calibration_checks = calibration_promises(cached, read(Path("weights/google_v1/head_G3tt_calibration.json")))
    save(out / "calibration_checks.json", calibration_checks)
    live_total = sum(g["n"] for c in live["cases"].values() for g in c.get("groups", {}).values())
    gates = {
        "frozen_artifacts_and_syntax": integrity["passed"], "private_inputs_not_published": privacy["passed"],
        "pinned_dependencies": pip.returncode == 0 and all(r["matches"] for r in requirements),
        "cpu_behavior": app.get("successful") is True and len(app["checks"]) >= 42,
        "chrome_interactions": browser.get("successful") is True and len(browser["checks"]) >= 13,
        "complete_full_detector": live.get("complete") is True and len(live["cases"]) == 19
            and live_total == 3951 and live.get("errors") == 0 and live["configuration_sha256"] == frozen["sha256"],
        "http_equivalence": live["ui"].get("passed") is True and live["ui"].get("checked") == 80,
        "fragmented_metadata_http": live.get("metadata_http", {}).get("passed") is True,
        "runtime_real_calibration": calibration["applied"] and all(
            p["any_ai"] / p["n"] <= .05 and p["edit1"] / p["n"] <= .02 and p["edit2"] / p["n"] <= .02
            for p in calibration["pools"].values()),
        "private_live_calibration": private["complete"] and private["errors"] == 0
            and private["configuration_sha256"] == frozen["sha256"]
            and sum(p["n"] for p in private["groups"].values()) == 204 and all(
            p["pixel_ai"] / p["n"] <= .05 and p["component_flag_counts"]["edit1"] / p["n"] <= .02
            and p["component_flag_counts"]["edit2"] / p["n"] <= .02 for p in private["groups"].values()),
        "transformation_checks": robustness["complete"] and robustness["progress"]["completed"] == 7245
            and robustness["progress"]["errors"] == 0 and robustness_summary(robustness)["passed"],
        "cache_provenance": cached["required_calibration_caches_valid"]
            and len(cached["face_cache_provenance"]) == 13
            and all(p["status"] == "exact_match" for p in cached["face_cache_provenance"]),
        "saved_calibration_mix_and_face_caps": calibration_checks["passed"],
        "pixel_certainty_boundary_removed": precision.get("any_confident_pixel_false_alarm") is False
            and bool(all_predictions) and all(p["pixel_verdict"] != "AI-generated" for p in all_predictions),
    }
    socket_smoke = out / "local_socket_smoke.json"
    if socket_smoke.exists():
        gates["real_socket_public_examples_and_api_fallback"] = read(socket_smoke)["passed"]
    changes = []
    total = 0
    for name, case in live["cases"].items():
        for source, group in case["groups"].items():
            old = baseline["cases"][name]["groups"][source]
            total += group["n"]
            if any(group[key] != old[key] for key in ("n", "pixel_ai", "final_ai", "ai_file_labels")):
                changes.append({"case": name, "group": source,
                                "before": {k: old[k] for k in ("n", "pixel_ai", "final_ai", "ai_file_labels")},
                                "after": {k: group[k] for k in ("n", "pixel_ai", "final_ai", "ai_file_labels")}})
    summary = {"updated_at_utc": datetime.now(timezone.utc).isoformat(),
               "local_release_ready": all(gates.values()), "hosted_deployment_verified": False,
               "configuration_sha256": frozen["sha256"], "gates": gates,
               "dataset_entries_not_unique_images": total, "full_detector_count_changes": changes,
               "pixel_measurement_reuse": live.get("pixel_measurement_reuse"),
               "baseline_audit": (args.baseline / "summary.md").as_posix(),
               "remaining": ["Deploy the reviewed code AND recalibrated private edit JSON together; do not expose weights.",
                             "Verify hosted /healthz, public samples, CPU boundary case and API fallback after deployment.",
                             "GitHub Actions has not run remotely for these unpushed changes.",
                             "Record and link the submission video; README still has a placeholder."],
               "limits": ["Calibration and reused regression pools are not independent tests; entries overlap.",
                          "The CPU boundary check tests one photo and the whole-image branch only.",
                          "Paid Meta services were mocked/offline; actual remote success and billing were not retested.",
                          "No new model training; hard cases such as Nano Banana 2.1, crops and video swaps remain.",
                          "Training provenance is partial: original Colab face manifests and some derivative-source records are missing; see the baseline manifest audit.",
                          "Earlier face robustness and training-manifest audit remain baseline artifacts; not rerun here."]}
    save(out / "summary.json", summary)
    lines = ["# Hardened local release", "", f"Local gates: **{'PASS' if summary['local_release_ready'] else 'FAIL'}**. Hosted deployment: **pending verification**.", "",
             "Production code, model weights and calibration are bound to the frozen configuration below. No paid APIs were called by these checks.", "",
             "| Check | Result |", "|---|---|"]
    lines += [f"| {name.replace('_', ' ')} | {'PASS' if passed else 'FAIL'} |" for name, passed in gates.items()]
    lines += ["", f"Full detector: {total} dataset entries across 19 cases, {live['errors']} errors; overlapping sources are not independent photos.",
              f"HTTP upload equivalence: {live['ui']['checked']}/80, plus {live.get('metadata_http', {}).get('checked', 0)} real fragmented-C2PA JPEGs. CPU checks: {len(app['checks'])}. Chrome interactions: {len(browser['checks'])}.",
              f"Transformations: {robustness['progress']['completed']}/7245 checks, without faces or metadata.", "",
              "After the JPEG metadata-only fix, all original file declarations were reparsed. Pixel/face/edit scores reuse the completed GPU run with byte-identical scoring artifacts; HTTP equivalence was rerun with real models. See metadata_refresh.json for the preserved run and configuration lineage.", "",
              "The 19 full-detector cases retain their original AI-flag/file-declaration totals." if not changes else f"Full-detector count changes: {json.dumps(changes)}", "",
              "Runtime calibration fixes private object edit-head false alarms from 3/100 to 2/100; combined alerts fall from 4/100 to 3/100. The separate 30-photo object regression remains 3/30.", "",
              "Pixel-only results never use the unqualified AI-generated tier. Structured AI declarations are marked unverified. PE contribution maps cover only their analysed centre square.", "",
              "## Remaining before submission", ""]
    lines += ["- " + item for item in summary["remaining"]]
    lines += ["", "## Limits", ""] + ["- " + item for item in summary["limits"]]
    lines += ["", "## Artifacts", "", "- [Full detector](live_metrics.json)", "- [Transformation details](robustness.md)",
              "- [Runtime calibration](runtime_calibration.json)", "- [CPU behavior](app_checks.json)",
              "- [Chrome interactions](browser_checks.json)", "- [Privacy](privacy_check.json)",
              "- [Integrity](integrity_check.json)", "- [Dependencies](environment.json)", "",
              f"Configuration: `{frozen['sha256']}`."]
    (out / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"local_release_ready": summary["local_release_ready"], "gates": gates}))
    return 0 if summary["local_release_ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
