"""Merge local release artifacts into a concise baseline summary; no model or API calls.

Rerun while/after release_live is running:
    python -m src.eval.release_summary --directory results/release_2026-11-04
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path


CASE_CACHE = {
    "judge_ai": None, "fresh_muse_v2": None, "private_judge": None,
    "polished_real": "commons_dev", "openimages_real": "sid_reals_0_9_unseen",
    "phone_real_public": "commons_v3_dev", "selfies_real_public": "commons_v4_dev",
    "closeups_real": "real_face_crops_dev_all", "muse_pairs": "muse_pairs_val",
    "muse_edits_v2": "muse_edits_test_v2", "muse_edits_v1": "muse_edits_v1",
    "muse_v1": "muse_image_test_v1", "sdxl": "sd_faces", "open_generators": "open_gen_dev",
    "gan_tpdne": "tpdne_test", "gan_sfhq": "sfhq_pt3_test", "gemini_dev": "google/gemini_dev_v1",
    "google_captions_dev": "google/open_gen_dev", "commons_ai_photos": "commons_ai_v1_photo",
}
ARTIFACTS = ("app_checks", "cached_metrics", "manifest_checks", "environment", "face_robustness", "live_metrics")
OPTIONAL_ARTIFACTS = ("precision_differences", "strong_threshold_check", "private_calibration_live",
                      "cross_device_boundary", "hosted_boundary")
REQUIRED_EXTRA_ARTIFACTS = ("browser_checks", "privacy_check", "integrity_check")
ROBUST_VERSIONS = {"original", "jpeg70", "whatsapp", "whatsapp2", "small", "screenshot", "crop"}
REAL_POOL_CACHE = {"polished dev": "commons_dev", "OpenImages val": "muse_pairs_val",
    "OpenImages unseen": "sid_reals_0_9_unseen", "Commons v3 dev": "commons_v3_dev",
    "close-ups": "real_face_crops_dev_all", "selfies+portraits": "commons_v4_dev",
    "phone": "private: phone", "phone objects": "private: phone objects", "new selfies": "private: new selfies"}
FACE_POOL_CACHE = {"polished dev": "commons_dev_eval", "OpenImages val": "muse_pairs_val_eval",
    "OpenImages unseen": "sid_reals_0_9_unseen_eval", "Commons v3 dev": "commons_v3_dev_eval",
    "close-ups": "real_face_crops_dev_all_eval", "selfies+portraits": "commons_v4_dev_eval",
    "phone": "private: phone", "phone objects": "private: phone objects", "new selfies": "private: new selfies"}


def read_artifacts(directory):
    loaded, fingerprints, missing = {}, {}, []
    for name in (*ARTIFACTS, "configuration", *REQUIRED_EXTRA_ARTIFACTS, *OPTIONAL_ARTIFACTS):
        path = directory / (name + ".json")
        if not path.exists():
            loaded[name] = {}
            if name not in OPTIONAL_ARTIFACTS:
                missing.append(name)
            continue
        content = path.read_bytes()
        try:
            loaded[name] = json.loads(content)
        except (ValueError, UnicodeDecodeError):
            loaded[name] = {}
            missing.append(name + " (invalid JSON)")
        fingerprints[name] = hashlib.sha256(content).hexdigest()
    robust_path = next((directory / name for name in ("robustness.json", "whole_image_robustness.json")
                        if (directory / name).exists()), None)
    if robust_path is None:
        missing.append("whole-image robustness (expected; pending)")
        loaded["whole_image_robustness"] = {}
    else:
        content = robust_path.read_bytes()
        try:
            loaded["whole_image_robustness"] = json.loads(content)
        except (ValueError, UnicodeDecodeError):
            loaded["whole_image_robustness"] = {}
            missing.append("whole-image robustness (invalid JSON)")
        fingerprints[robust_path.stem] = hashlib.sha256(content).hexdigest()
    head = loaded.get("cached_metrics", {}).get("models", {}).get("whole_head")
    if head:
        path = Path(head).with_name(Path(head).stem + "_calibration.json")
        if path.exists():
            content = path.read_bytes()
            loaded["whole_calibration"] = json.loads(content)
            fingerprints["whole_calibration"] = hashlib.sha256(content).hexdigest()
        else:
            missing.append("saved whole-image calibration")
    return loaded, fingerprints, missing


def rate(ai, n):
    return round(100 * ai / n, 4) if n else None


def live_counts(group):
    verdicts = group.get("pixel_verdict_counts", {})
    return {"n": group.get("n", 0), "requested_so_far": group.get("requested", 0),
            "ai": group.get("pixel_ai", 0), "uncertain": verdicts.get("Uncertain", 0),
            "authentic": verdicts.get("Likely authentic", 0), "errors": group.get("errors", 0),
            "final_ai_including_metadata": group.get("final_ai", 0),
            "ai_file_labels": group.get("ai_file_labels", 0), "verdict_counts": verdicts,
            "ai_percent": rate(group.get("pixel_ai", 0), group.get("n", 0))}


def compare_cached(cached, cache_name, source, live, complete):
    dataset = next((entry for entry in cached.get("image_sets", []) if entry["name"] == cache_name), None)
    if dataset is None:
        return {"status": "no_corresponding_current_cache"}
    buckets = [bucket for key, bucket in dataset.get("by_source", {}).items() if key.rsplit(" | ", 1)[0] == source]
    if not buckets:
        return {"status": "no_matching_cached_source"}
    counts = {key: sum(bucket[key] for bucket in buckets) for key in ("n", "ai", "uncertain", "authentic")}
    result = {"cached_scope": dataset["status"], "cached": counts}
    if complete and live["n"] == counts["n"] and not live["errors"]:
        result.update(status="compared_equal_denominators",
                      live_minus_cached={key: live[key] - counts[key] for key in counts})
    else:
        result["status"] = "pending_or_unequal_denominators; no count deltas asserted"
    return result


def calibration_promises(cached, whole):
    """Check the saved calibration's own row-weighted mix, not an arbitrary evaluation union."""
    if not whole:
        return {"available": False, "passed": False, "reason": "Saved whole-image calibration unavailable."}
    datasets = {entry["name"]: entry for entry in cached.get("image_sets", [])}
    faces = {entry["name"]: entry for entry in cached.get("face_sets", [])}
    real_checks, fake_mix, face_checks = [], [], []
    for name, expected in whole.get("real_pools", {}).items():
        dataset = datasets.get(REAL_POOL_CACHE.get(name), {})
        c = dataset.get("by_label", {}).get("real", {})
        valid = dataset.get("status") == "complete_cached_pipeline" and c.get("n") == expected
        allowed = math.floor(whole["max_false_alarms"] * expected)
        real_checks.append({"pool": name, "n": expected, "ai": c.get("ai"), "uncertain": c.get("uncertain"),
            "authentic": c.get("authentic"), "allowed_ai": allowed,
            "passed": valid and c["ai"] <= allowed})
    fake_specs = {"Muse test v1": ("muse_image_test_v1", None), "Muse pairs val": ("muse_pairs_val", None),
                  "SDXL (unseen)": ("sd_faces", "sd_faces/sdxl | ai")}
    for name, expected in whole.get("fake_sets", {}).items():
        stem, source = fake_specs.get(name, (None, None))
        dataset = datasets.get(stem, {})
        c = dataset.get("by_source", {}).get(source, {}) if source else dataset.get("by_label", {}).get("ai", {})
        fake_mix.append({"set": name, "n": expected, "uncertain": c.get("uncertain"),
                         "valid": dataset.get("status") == "complete_cached_pipeline" and c.get("n") == expected})
    for pool in whole.get("real_pools", {}):
        face = faces.get(FACE_POOL_CACHE.get(pool), {})
        for component in ("ai_made", "swap"):
            c = face.get("components", {}).get(component, {}).get("real", {})
            n, flagged = c.get("n_faces"), c.get("strict")
            allowed = math.floor(.005 * n) if n is not None else None
            face_checks.append({"pool": pool, "check": component, "n_faces": n, "strict": flagged,
                "allowed_strict": allowed, "passed": n is not None and flagged is not None and flagged <= allowed})
    for name, stem in (("DFDC p3 real", "dfdc_part3_eval"), ("SID val real", "validation-00000-of-00034_eval")):
        c = faces.get(stem, {}).get("components", {}).get("swap", {}).get("real", {})
        n, flagged = c.get("n_faces"), c.get("strict")
        allowed = math.floor(.005 * n) if n is not None else None
        face_checks.append({"pool": name, "check": "swap", "n_faces": n, "strict": flagged,
            "allowed_strict": allowed, "passed": n is not None and flagged is not None and flagged <= allowed})
    valid_mix = bool(real_checks and fake_mix) and all(c["uncertain"] is not None for c in real_checks) \
        and all(c["valid"] for c in fake_mix)
    real_n, fake_n = sum(c["n"] for c in real_checks), sum(c["n"] for c in fake_mix)
    real_unc = sum(c["uncertain"] for c in real_checks) if valid_mix else None
    fake_unc = sum(c["uncertain"] for c in fake_mix) if valid_mix else None
    uncertainty = .5 * (real_unc / real_n + fake_unc / fake_n) if valid_mix and real_n and fake_n else None
    uncertainty_ok = uncertainty is not None and uncertainty <= whole["max_uncertain"]
    return {"available": True, "scope": "Final cached ensemble on exactly the saved whole-calibration real_pools and fake_sets; calibration-mix check only, not independent performance.",
        "real_photo_union_checks": real_checks, "fake_mix_components": fake_mix, "real_face_line_checks": face_checks,
        "class_balanced_uncertainty": {"real_n": real_n, "real_uncertain": real_unc, "fake_n": fake_n,
            "fake_uncertain": fake_unc, "percent": round(100 * uncertainty, 6) if uncertainty is not None else None,
            "cap_percent": 100 * whole["max_uncertain"], "formula": "0.5 * real_uncertain/real_n + 0.5 * fake_uncertain/fake_n",
            "passed": uncertainty_ok},
        "face_check_limit": "The 0.5% checks apply to each calibrated face head on its specified pools. Separate head caps do not mathematically guarantee a 0.5% union of both heads; off-domain test pools are not guaranteed.",
        "passed": uncertainty_ok and all(c["passed"] for c in real_checks + face_checks)}


def robustness_summary(raw):
    """Normalize aggregate group/version counts and reject incomplete count partitions."""
    groups = raw.get("groups", raw.get("cases", []))
    if isinstance(groups, dict):
        groups = [{"name": name, **group} for name, group in groups.items()]
    if not isinstance(groups, list):
        groups = []
    normalized = []
    valid = bool(groups)
    row_error_total = 0
    reported_errors = raw.get("errors", raw.get("progress", {}).get("errors", 0))
    valid &= isinstance(reported_errors, int) and reported_errors >= 0
    for group in groups:
        versions = {}
        version_data = group.get("versions", {})
        if not isinstance(version_data, dict):
            version_data = {}
        for version, row in version_data.items():
            verdicts = row.get("pixel_verdict_counts", row.get("verdict_counts", {}))
            n = row.get("n", row.get("completed", group.get("n")))
            ai = row.get("pixel_ai", row.get("ai"))
            if ai is None and verdicts:
                ai = sum(count for verdict, count in verdicts.items() if verdict not in ("Uncertain", "Likely authentic"))
            uncertain = row.get("uncertain", verdicts.get("Uncertain", 0))
            authentic = row.get("authentic", verdicts.get("Likely authentic", 0))
            row_errors = row.get("errors", 0)
            good = all(isinstance(value, int) and value >= 0 for value in (n, ai, uncertain, authentic, row_errors))
            good = good and n > 0 and ai + uncertain + authentic == n and row_errors == 0
            if "requested" in row:
                good &= n == row["requested"]
            valid &= good
            row_error_total += row_errors if isinstance(row_errors, int) and row_errors >= 0 else 0
            versions[version] = {"n": n, "ai": ai, "uncertain": uncertain, "authentic": authentic,
                                 "errors": row_errors, "counts_valid": good}
        valid &= ROBUST_VERSIONS <= versions.keys()
        normalized.append({"name": group.get("name", "unnamed group"), "versions": versions})
    errors = max(reported_errors if isinstance(reported_errors, int) else 0, row_error_total)
    passed = raw.get("complete") is True and valid and errors == 0
    status = "complete" if passed else "pending" if not raw else "running_or_incomplete" if not raw.get("complete") else "invalid_counts_or_missing_versions"
    return {"status": status, "passed": passed, "scope": raw.get("scope"), "complete": raw.get("complete", False),
            "errors": errors, "groups": normalized, "group_count": len(normalized), "required_versions": sorted(ROBUST_VERSIONS)}


def build(data, fingerprints, missing):
    app, cached, manifest, environment, robustness, live = (data[name] for name in ARTIFACTS)
    cases = live.get("cases", {})
    failures = [{key: entry.get(key) for key in ("id", "status", "message", "measurements") if key in entry}
                for entry in app.get("checks", []) if entry.get("status") != "passed"]
    completed_cases = [name for name in CASE_CACHE if cases.get(name, {}).get("complete") is True]
    live_error_count = sum(group.get("errors", 0) for case in cases.values() for group in case.get("groups", {}).values())
    live_error_count += sum("error" in case for case in cases.values())
    live_error_count = max(live_error_count, live.get("errors", 0))
    ui = live.get("ui", {})
    ui_ok = ui.get("complete") is True and ui.get("checked") == 80 and ui.get("passed") is True and not ui.get("mismatches")
    live_ok = live.get("complete") is True and len(completed_cases) == len(CASE_CACHE) and not live_error_count
    app_ok = bool(app) and app.get("successful") is True and not failures and not app.get("failures") and not app.get("errors")
    environment_ok = bool(environment) and environment.get("pip_check", {}).get("exit_code") == 0 \
        and all(entry.get("matches") is True for entry in environment.get("requirements", []))
    provenance = cached.get("face_cache_provenance", [])
    cache_ok = cached.get("required_calibration_caches_valid") is True and bool(provenance) \
        and all(entry["status"] == "exact_match" for entry in provenance)
    configuration = data.get("configuration", {})
    config_ok = bool(configuration.get("sha256")) and live.get("configuration_sha256") == configuration.get("sha256")
    head_hashes = cached.get("head_sha256", {})
    artifact_hashes = configuration.get("artifacts_and_code", {})
    for name, path in (("whole", "whole_head"), ("edit", "edit_head"), ("second", "second_head"), ("swap", "swap_head")):
        key = cached.get("models", {}).get(path, "").replace("\\", "/")
        config_ok &= bool(key) and head_hashes.get(name) == artifact_hashes.get(key)
    calibration_checks = calibration_promises(cached, data.get("whole_calibration", {}))
    browser, privacy = data.get("browser_checks", {}), data.get("privacy_check", {})
    browser_checks = browser.get("checks", [])
    browser_ok = browser.get("successful") is True and len(browser_checks) >= 12 \
        and all(check.get("status") == "passed" for check in browser_checks)
    privacy_ok = privacy.get("passed") is True and not privacy.get("findings") \
        and privacy.get("private_prediction_keys_in_public_cache") == 0 and privacy.get("private_checkpoint_separate") is True
    whole_robustness = robustness_summary(data.get("whole_image_robustness", {}))
    private_calibration = data.get("private_calibration_live", {})
    private_groups = private_calibration.get("groups", {})
    private_budgets = private_calibration.get("calibration_budgets", {})
    private_budget_checks = []
    expected_private = {"private/calibration_phone": 104, "private/calibration_objects": 100}
    private_ok = private_calibration.get("complete") is True and private_calibration.get("errors") == 0 \
        and private_calibration.get("configuration_sha256") == configuration.get("sha256")
    for name, expected in expected_private.items():
        group = private_groups.get(name, {})
        private_ok &= group.get("complete") is True and group.get("n") == expected and group.get("errors") == 0
        complete = group.get("complete") is True and group.get("n") == expected and group.get("errors") == 0
        components = group.get("component_flag_counts", {})
        for component, count, cap_key in (("final_pixel_union", group.get("pixel_ai"), "max_false_alarm_rate"),
                                         ("edit1", components.get("edit1"), "max_edit_flag_rate_per_head"),
                                         ("edit2", components.get("edit2"), "max_edit_flag_rate_per_head")):
            cap = private_budgets.get(cap_key)
            allowed = math.floor(cap * expected) if cap is not None else None
            private_budget_checks.append({"pool": name, "component": component, "n": expected, "flagged": count,
                "allowed": allowed, "cap_percent": cap * 100 if cap is not None else None,
                "passed": complete and count is not None and allowed is not None and count <= allowed})
    private_caps_ok = bool(private_budget_checks) and all(check["passed"] for check in private_budget_checks)
    gates = {"all_required_artifacts_present": not missing, "all_app_checks_pass": app_ok,
             "all_19_live_cases_complete_without_errors": live_ok, "real_upload_ui_80_match": ui_ok,
             "environment_requirements_match": environment_ok, "required_calibration_and_v4_cache_checks_pass": cache_ok,
             "live_and_cached_configuration_match": bool(config_ok),
             "final_ensemble_calibration_mix_and_per_pool_caps_pass": calibration_checks["passed"],
             "desktop_mobile_browser_checks_pass": browser_ok, "scoped_private_report_scan_passes": privacy_ok,
             "whole_image_robustness_complete_without_errors": whole_robustness["passed"],
             "private_calibration_live_complete_without_errors": bool(private_ok),
             "private_live_union_5pct_and_each_edit_head_2pct_caps_pass": private_caps_ok}
    gates["frozen_production_artifacts_and_python_syntax_valid"] = data.get("integrity_check", {}).get("passed") is True
    hosted_boundary = data.get("hosted_boundary", {})
    if hosted_boundary:
        gates["hosted_public_real_boundary_avoids_confident_false_alarm"] = (
            hosted_boundary.get("status") == "complete"
            and hosted_boundary.get("confident_false_alarm") is False)
    sources = []
    for case_name, case in cases.items():
        for source, group in case.get("groups", {}).items():
            counts = live_counts(group)
            sources.append({"case": case_name, "source": source, "role": case.get("role"),
                "case_complete": case.get("complete") is True, "counts": counts,
                "cached_comparison": compare_cached(cached, CASE_CACHE.get(case_name), source, counts, case.get("complete") is True)})
    completed_totals = {}
    for name in completed_cases:
        rows = [row["counts"] for row in sources if row["case"] == name]
        completed_totals[name] = {key: sum(row[key] for row in rows)
                                 for key in ("n", "ai", "uncertain", "authentic", "errors", "final_ai_including_metadata")}
    partial_caches = [{"name": entry["name"], "status": entry["status"],
                       **({"missing_components": entry["missing_components"]} if "missing_components" in entry else {})}
                      for entry in cached.get("image_sets", []) if entry["status"] != "complete_cached_pipeline"]
    manifest_checks = manifest.get("checks", [])
    claims = []
    for name, old, case_name, cache_name, source in (
        ("SDXL pixel detection", "92%", "sdxl", "sd_faces", "sd_faces/sdxl"),
        ("OpenImages unseen real false alarms", "1.7%", "openimages_real", "sid_reals_0_9_unseen", None),
        ("SD 3.5 full-app pixel detection", "about half", "open_generators", "open_gen_dev", "open_gen/sd35m_devcaption")):
        ds = next((entry for entry in cached.get("image_sets", []) if entry["name"] == cache_name), {})
        if source:
            bucket = next((v for k, v in ds.get("by_source", {}).items() if k == source + " | ai"), {})
        else:
            bucket = ds.get("by_label", {}).get("real", {})
        complete = cases.get(case_name, {}).get("complete") is True
        groups = [g for name, g in cases.get(case_name, {}).get("groups", {}).items()
                  if not source or name == source] if complete else []
        n, ai = sum(g.get("n", 0) for g in groups), sum(g.get("pixel_ai", 0) for g in groups)
        claims.append({"claim": name, "readme_value": old, "current_cached": {key: bucket.get(key) for key in ("n", "ai", "ai_percent")},
                       "current_live": {"n": n, "ai": ai, "ai_percent": rate(ai, n)} if complete else None,
                       "status": "live_confirmed_difference_from_README" if complete else "cached_difference; live_pending"})
    if whole_robustness["passed"]:
        local = next((g for g in whole_robustness["groups"] if g["name"] == "muse_v2_local"), None)
        if local:
            for version in ("small", "crop"):
                c = local["versions"][version]
                claims.append({"claim": f"Local Muse edits, {version} (whole-image + both PE heads; no faces/metadata)",
                               "readme_value": "about 60%", "current_cached": {},
                               "current_live": {"n": c["n"], "ai": c["ai"], "ai_percent": rate(c["ai"], c["n"])},
                               "status": "current_robustness_replaces_historical_figure"})
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Current baseline before final commit; no improved-model claim and no new independent final-test claim.",
        "release_ready": all(gates.values()), "gates": gates,
        "blockers": [name for name, passed in gates.items() if not passed], "source_artifact_sha256": fingerprints,
        "app_checks": {"tests_run": app.get("tests_run"), "failures": app.get("failures"), "errors": app.get("errors"),
                       "failed_checks": failures, "scope": app.get("scope")},
        "live_progress": {"status": "complete" if live_ok else "running_or_incomplete", "artifact_complete": live.get("complete", False),
            "completed_cases": len(completed_cases), "expected_cases": len(CASE_CACHE),
            "missing_or_incomplete_cases": [name for name in CASE_CACHE if name not in completed_cases],
            "processed_rows_including_overlapping_sets": sum(s["counts"]["n"] for s in sources), "errors": live_error_count},
        "ui_equivalence": {key: ui.get(key) for key in ("requested", "complete", "checked", "passed")}
            | {"mismatch_count": len(ui.get("mismatches", [])), "gate_passed": ui_ok},
        "live_sources": sources, "completed_case_totals_not_pooled_across_cases": completed_totals,
        "partial_cached_coverage": bool(partial_caches), "partial_or_missing_cached_sets": partial_caches,
        "extra_cache_gap": "muse_edits_v1_all_app.npz lacks pe_ec3/muse_edits_v1_all_eval.npz; canonical v1 pair exists and this extra cache is not required by saved calibration.",
        "cache_success_meaning": "Exit zero verifies required calibration/provenance checks; it does not establish complete coverage of every optional cache or live inference.",
        "readme_claims_to_review": claims, "face_robustness": robustness,
        "calibration_promise_checks": calibration_checks,
        "precision_boundary_checks": {"interpretation": data.get("precision_differences", {}).get("interpretation"),
            "differences": [{key: row.get(key) for key in ("set", "cached_verdict", "live_verdict", "cached_whole", "live_whole",
                "whole_delta", "cached_whole_ai_margin", "live_whole_ai_margin", "cached_whole_real_margin", "live_whole_real_margin",
                "within_mean_fp16_rounding_bound")} for row in data.get("precision_differences", {}).get("differences", [])]},
        "strong_threshold_check": data.get("strong_threshold_check", {}),
        "cross_device_boundary": data.get("cross_device_boundary", {}),
        "hosted_boundary": hosted_boundary,
        "integrity_check": data.get("integrity_check", {}),
        "browser_checks": {"scope": browser.get("scope"), "tests_run": len(browser_checks),
            "passed": sum(check.get("status") == "passed" for check in browser_checks), "gate_passed": browser_ok,
            "failed_checks": [{"id": check.get("id"), "status": check.get("status")} for check in browser_checks if check.get("status") != "passed"]},
        "privacy_check": {key: privacy.get(key) for key in ("scope", "public_artifacts_checked", "private_image_rows",
            "private_prediction_keys_in_public_cache", "private_checkpoint_separate", "passed")}
            | {"finding_count": len(privacy.get("findings", [])), "gate_passed": privacy_ok},
        "whole_image_robustness": whole_robustness,
        "private_calibration_live": {"status": private_calibration.get("status", "pending"), "gate_passed": bool(private_ok),
            "complete": private_calibration.get("complete", False), "errors": private_calibration.get("errors", 0),
            "groups": private_groups, "scope": private_calibration.get("scope"),
            "backbone_weight_hashes_recomputed": private_calibration.get("backbone_weight_hashes_recomputed"),
            "calibration_budgets": private_budgets, "live_component_budget_checks": private_budget_checks},
        "manifest_audit": {"conclusion": manifest.get("conclusion"), "hash_check": {key: manifest.get("hash_check", {}).get(key)
                for key in ("candidates_hashed", "matching_training_file_count", "complete_for_accessible_public_files", "limitations",
                            "unavailable_training_files_or_nonfile_ids", "unavailable_evaluation_files_or_nonfile_ids")},
            "comparison_rows_with_exact_identity_overlap": sum(entry.get("exact_identity_overlap", 0) > 0 for entry in manifest_checks),
            "comparison_rows_with_source_family_overlap": sum(entry.get("source_family_overlap", 0) > 0 for entry in manifest_checks),
            "missing_final_face_training_manifest_names": manifest.get("missing_final_face_training_manifest_names", []),
            "limitations": manifest.get("limitations", [])},
        "environment": {"python": environment.get("python"), "gpu": environment.get("gpu"), "pip_check": environment.get("pip_check"),
                        "requirements_match": environment_ok, "core_versions": environment.get("core_versions", {})},
        "validation_limits": ["No aggregate accuracy is computed across overlapping sets.",
            "Live-versus-cached differences may include float16 boundary effects; partial caches also omit runtime face checks.",
            "Private groups contain neutral totals only, never private per-image names, IDs, paths, or hashes.",
            "Meta paid APIs were mocked or disabled. Only one public real-photo basic-detection request was tested on the hosted demo; hosted-wide accuracy, concurrency, and hardware provenance were not validated.",
            "Manifest coverage is partial; accessible-file exact-byte results do not prove universal absence of leakage.",
            "Face robustness uses 30 photos per group and simulated phone processing, not a population guarantee."],
    }


def render(report):
    progress = report["live_progress"]
    lines = ["# Current baseline before final commit", "",
        f"**Release ready: {str(report['release_ready']).lower()}.** Live evaluation: **{progress['status']}** "
        f"({progress['completed_cases']}/{progress['expected_cases']} cases complete; {progress['errors']} errors).", "",
        report["scope"], "", "## Blocking app failures", "",
        f"**{report['app_checks']['failures']} failures / {report['app_checks']['tests_run']} checks; {report['app_checks']['errors']} test errors.**", ""]
    for failure in report["app_checks"]["failed_checks"]:
        lines.append(f"- **{failure['id']}**: {failure.get('message', failure['status'])}")
    hosted = report.get("hosted_boundary", {})
    if hosted:
        lines += ["", "## Hosted confidence boundary check", "",
                  f"Status: **{hosted.get('status')}**. One known-real public OpenImages calibration photo returned "
                  f"**{hosted.get('verdict', 'unavailable')}**; confident false alarm: **{hosted.get('confident_false_alarm')}**.",
                  "The laptop GPU kept this photo below the strongest AI threshold. A separate laptop CPU bf16 whole-image check crossed it, "
                  "and the actual hosted endpoint also returned the strongest AI label. This is a confirmed label failure on one real photo, "
                  "not an estimate of the hosted false-alarm rate. No private photo or paid investigator API was used."]
    lines += ["", "## Release gates", ""]
    for name, passed in report["gates"].items():
        lines.append(f"- {'PASS' if passed else 'BLOCKED'}: {name.replace('_', ' ')}")
    lines += ["", "Completed named checks (these cases are not pooled into one accuracy):", ""]
    for name in ("judge_ai", "fresh_muse_v2", "private_judge"):
        c = report["completed_case_totals_not_pooled_across_cases"].get(name)
        if c:
            lines.append(f"- {name}: {c['n']} images; {c['ai']} pixel-AI, {c['uncertain']} uncertain, {c['authentic']} authentic; "
                         f"{c['final_ai_including_metadata']} AI including file metadata; {c['errors']} errors.")
    lines += ["", "## Live source counts", "", "AI counts below use pixels; original-file metadata flags are kept separate in JSON. Partial cases are still running.", "",
        "| Case / source | Case complete | n | Pixel AI | Uncertain | Authentic | Errors |", "|---|---|---:|---:|---:|---:|---:|"]
    for row in report["live_sources"]:
        c = row["counts"]
        lines.append(f"| {row['case']} / {row['source']} | {row['case_complete']} | {c['n']} | {c['ai']} | {c['uncertain']} | {c['authentic']} | {c['errors']} |")
    lines += ["", "## Live versus cached", "", "Deltas are live minus cached and are emitted only for complete cases with equal source denominators. Missing runtime faces make some cached rows partial.", "",
              "| Case / source | Cached scope | n | Δ AI | Δ uncertain | Δ authentic |", "|---|---|---:|---:|---:|---:|"]
    for row in report["live_sources"]:
        comparison = row["cached_comparison"]
        if "live_minus_cached" in comparison:
            delta = comparison["live_minus_cached"]
            lines.append(f"| {row['case']} / {row['source']} | {comparison['cached_scope']} | {row['counts']['n']} | {delta['ai']:+d} | {delta['uncertain']:+d} | {delta['authentic']:+d} |")
    lines += ["", "## README figures to review", ""]
    for claim in report["readme_claims_to_review"]:
        c, live = claim["current_cached"], claim["current_live"]
        value = f"{live['ai']}/{live['n']} ({live['ai_percent']:.2f}%) live" if live and live.get("ai_percent") is not None else "live pending"
        cache_value = f"{c['ai']}/{c['n']} ({c['ai_percent']:.2f}%)" if c.get("ai_percent") is not None else "unavailable"
        lines.append(f"- {claim['claim']}: README {claim['readme_value']}; cached {cache_value}; {value}.")
    lines += ["", "## Robustness of the current face branch", "", "Counts are photos with a strict face flag, not full-detector recall.", "",
              "| Group | n | Original | WhatsApp | Small | Simulated phone processing |", "|---|---:|---:|---:|---:|---:|"]
    for group in report["face_robustness"].get("groups", []):
        f = group["flagged"]
        lines.append(f"| {group['name']} | {group['n']} | {f['original']} | {f['whatsapp']} | {f['small']} | {f['phone']} |")
    calibration = report["calibration_promise_checks"]
    lines += ["", "## Calibration promises on the saved mix", ""]
    if calibration.get("available"):
        u = calibration["class_balanced_uncertainty"]
        percentage = f"{u['percent']:.4f}%" if u["percent"] is not None else "unavailable"
        lines += [calibration["scope"], "",
            f"Class-balanced Uncertain: **{percentage}** (cap {u['cap_percent']:g}%): "
            f"0.5 × ({u['real_uncertain']}/{u['real_n']}) + 0.5 × ({u['fake_uncertain']}/{u['fake_n']}).", "",
            "| Required real-photo pool | n | Full-ensemble AI | Allowed at 5% | Passed |", "|---|---:|---:|---:|---|"]
        for c in calibration["real_photo_union_checks"]:
            lines.append(f"| {c['pool']} | {c['n']} | {c['ai']} | {c['allowed_ai']} | {c['passed']} |")
        lines += ["", "| Required real-face pool / head | n faces | Strict | Allowed at 0.5% | Passed |", "|---|---:|---:|---:|---|"]
        for c in calibration["real_face_line_checks"]:
            lines.append(f"| {c['pool']} / {c['check']} | {c['n_faces']} | {c['strict']} | {c['allowed_strict']} | {c['passed']} |")
        lines += ["", calibration["face_check_limit"]]
    else:
        lines.append(calibration["reason"])
    precision = report["precision_boundary_checks"]
    lines += ["", "## Threshold precision checks", ""]
    if precision["differences"]:
        for c in precision["differences"]:
            lines.append(f"- {c['set']}: {c['cached_verdict']} cached → {c['live_verdict']} live; whole-score delta {c['whole_delta']:+.9f}. "
                         f"Within the cached mean-feature float16 rounding bound: {c['within_mean_fp16_rounding_bound']}.")
        lines.append(precision["interpretation"])
    maximum = report["strong_threshold_check"].get("live_max_row_check", {})
    if maximum.get("live_prediction_available"):
        lines += ["", f"The single real calibration maximum came from {maximum['pool']}. Cached score {maximum['cached_whole']:.9f} "
                  f"equals the strong-AI line; recorded live score {maximum['live_whole']:.9f} has margin {maximum['live_margin_to_strong']:+.9f}. "
                  f"Live verdict: {maximum['live_pixel_verdict']}; above strong line: {maximum['live_above_strong_line']}."]
    browser, privacy = report["browser_checks"], report["privacy_check"]
    lines += ["", "## Browser and scoped privacy checks", "",
        f"Browser: **{browser['passed']}/{browser['tests_run']} passed**. {browser['scope']}. Actual model inference and paid APIs were mocked for these interactions.", "",
        f"Privacy scan: passed={privacy['gate_passed']}; {privacy['finding_count']} findings, "
        f"{privacy['public_artifacts_checked']} public artifacts checked against {privacy['private_image_rows']} private judge-image rows; "
        f"{privacy['private_prediction_keys_in_public_cache']} private prediction keys in the public cache. This is the scan's stated scope, not an assertion about all possible private data."]
    robust = report["whole_image_robustness"]
    lines += ["", "## Whole-image robustness", "",
        f"Status: **{robust['status']}**; {robust['group_count']} groups; {robust['errors']} errors. Seven versions per group are required for this gate."]
    if robust["groups"]:
        lines += ["", "| Group / version | n | AI | Uncertain | Authentic | Errors |", "|---|---:|---:|---:|---:|---:|"]
        for group in robust["groups"]:
            for version, c in group["versions"].items():
                lines.append(f"| {group['name']} / {version} | {c['n']} | {c['ai']} | {c['uncertain']} | {c['authentic']} | {c['errors']} |")
    private_calibration = report["private_calibration_live"]
    lines += ["", "## Original private calibration pools", "",
        f"Status: **{private_calibration['status']}**; complete={private_calibration['complete']}; errors={private_calibration['errors']}. "
        "These are calibration checks, not independent test results.", "",
        "| Neutral group | n / expected | Pixel AI | Strict AI-generated real | Faces checked / red | Errors |", "|---|---:|---:|---:|---:|---:|"]
    for name, c in private_calibration["groups"].items():
        lines.append(f"| {name} | {c['n']} / {c['expected']} | {c['pixel_ai']} | {c['strict_ai_generated_real']} | "
                     f"{c['faces_checked']} / {c['faces_red']} | {c['errors']} |")
    lines += ["", "Private calibration budgets apply to the pixel union (5%) and to each edit head separately (2%); the two edit heads' union is not given a separate 2% promise.", "",
              "| Neutral group / component | n | Flagged | Allowed | Passed after complete run |", "|---|---:|---:|---:|---|"]
    for c in private_calibration["live_component_budget_checks"]:
        lines.append(f"| {c['pool']} / {c['component']} | {c['n']} | {c['flagged']} | {c['allowed']} | {c['passed']} |")
    lines += ["", "## Coverage limits", "", report["cache_success_meaning"], "", report["extra_cache_gap"], "",
              f"Partial cached coverage: **{report['partial_cached_coverage']}**. See cached_metrics.md for each missing/partial set.", "",
              report["manifest_audit"].get("conclusion") or "Manifest report unavailable.", ""]
    h = report["manifest_audit"]["hash_check"]
    lines.append(f"Accessible public-file hash check: {h.get('matching_training_file_count')} matching training files among "
                 f"{h.get('candidates_hashed')} equal-size candidates. This does not cover inaccessible files, decoded/perceptual duplicates, or all Colab training manifests.")
    lines += ["", *["- " + limit for limit in report["validation_limits"]], "",
              "Rerun: `python -m src.eval.release_summary --directory results/release_2026-11-04`", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=Path("results/release_2026-11-04"))
    args = parser.parse_args()
    data, fingerprints, missing = read_artifacts(args.directory)
    report = build(data, fingerprints, missing)
    args.directory.mkdir(parents=True, exist_ok=True)
    for suffix, content in (("json", json.dumps(report, indent=2, ensure_ascii=False) + "\n"), ("md", render(report))):
        path = args.directory / ("summary." + suffix)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(content, encoding="utf-8")
        temporary.replace(path)
    print(f"release_ready={report['release_ready']}; live={report['live_progress']['status']}; "
          f"app_failures={report['app_checks']['failures']}; ui_pass={report['ui_equivalence']['gate_passed']}")


if __name__ == "__main__":
    main()
