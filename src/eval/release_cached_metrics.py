"""Reconstruct release metrics from existing CPU feature caches, without recalibrating.

Run from the repository root:
    python -m src.eval.release_cached_metrics --include-private \
        --out-dir results/release_2026-11-04

No image is opened, no model backbone is loaded, and no API/network/GPU is used.
Reports contain aggregate counts only. Missing face caches are explicitly partial:
they are never interpreted as evidence that an image has no faces.
"""

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from src.features.add_face_pe import PAIRS, norm
from src.inference import Detector, load_calibration, load_edit_check
from src.models.linear_head import LinearHead


VERDICTS = ["AI-generated", "Likely AI-generated", "Likely AI-edited",
            "This face looks AI-made or AI-changed", "Uncertain", "Likely authentic"]
POOL_STEMS = {
    "polished dev": "commons_dev", "OpenImages val": "muse_pairs_val",
    "originals v1": "muse_edits_v1", "originals v2": "muse_edits_test_v2",
    "OpenImages unseen": "sid_reals_0_9_unseen", "Commons v3 dev": "commons_v3_dev",
    "close-ups": "real_face_crops_dev_all", "selfies+portraits": "commons_v4_dev",
}
PRIVATE_STEMS = {"phone": "phone_v1", "phone objects": "phone_objects", "new selfies": "test_selfies"}
INDEPENDENT_EXPECTED = {
    "judge_test_ai_v1": "release regression; observed during earlier development",
    "muse_image_test_v2": "fresh Muse evaluation; use live inference if caches absent",
    "chameleon_random": "external evaluation; previously inspected during development",
    "chameleon_size_matched": "external size-matched evaluation; previously inspected",
    "rtfs_test": "unseen identities; original Colab results are not a reconstruction at the final app line",
}


class CacheError(Exception):
    """A cache failed an aggregate-only validation."""


def arrays(path):
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def score(head, data):
    if not set(head.keys) <= data.keys():
        raise CacheError("required current-model feature keys missing")
    x = head.inputs(data)
    if not torch.isfinite(x).all():
        raise CacheError("nonfinite cached features")
    with torch.no_grad():
        out = head.logits(x).cpu().numpy()
    if not np.isfinite(out).all():
        raise CacheError("nonfinite head scores")
    return out


def pe_score(head, data):
    if "features" not in data:
        raise CacheError("PE features missing")
    x = data["features"].reshape(len(data["features"]), -1)
    return score(LinearHead(head.weight, head.bias), {"features": x})


def aligned(a, b):
    for key in ("labels", "img_ids"):
        if key not in a or key not in b or not np.array_equal(a[key], b[key]):
            raise CacheError("whole-image and PE row identities/labels do not align")


def counts(mask, verdicts, ids, stages):
    n = int(mask.sum())
    values = Counter(verdicts[mask].tolist())
    ai = sum(values[v] for v in VERDICTS[:4])
    unique = len(set(ids[mask].astype(str)))
    return {
        "n": n, "unique_image_ids": unique, "repeated_id_rows": n - unique,
        "ai": ai, "uncertain": values["Uncertain"], "authentic": values["Likely authentic"],
        "ai_percent": round(100 * ai / n, 4) if n else None,
        "verdict_counts": {v: values[v] for v in VERDICTS},
        "component_flag_counts": {key: int(value[mask].sum()) for key, value in stages.items()},
    }


def source_name(value):
    value = str(value)
    return "redacted_source" if ":" in value or value.startswith(("/", "\\")) else value


def face_provenance(source_dir):
    results, invalid = [], set()
    for crop_stem, face_stem in PAIRS.items():
        source = source_dir / (crop_stem + ".npz")
        dest = Path("features/faces") / (face_stem + ".npz")
        entry = {"source_set": crop_stem, "face_set": face_stem}
        if not source.exists() or not dest.exists():
            entry["status"] = "missing_source_or_destination"
            invalid.add(face_stem)
        else:
            a, b = arrays(source), arrays(dest)
            x = a["features"].reshape(len(a["features"]), -1)
            original_ids = norm(b["groups"] if "groups" in b else b["img_ids"])
            crop_ids = norm(a["img_ids"] if crop_stem == "wilddeepfake" else a["groups"])
            rows = np.arange(len(original_ids)) if len(crop_ids) == len(original_ids) else np.flatnonzero(b["labels"] == 0)
            matches = ("features_facepe" in b and len(rows) == len(crop_ids)
                       and [original_ids[i] for i in rows] == crop_ids
                       and np.array_equal(b["labels"][rows].astype(int), a["labels"].astype(int))
                       and np.array_equal(b["features_facepe"][rows], x))
            entry.update(status="exact_match" if matches else "mismatch", source_rows=len(x),
                         destination_rows=len(original_ids), matched_rows=len(rows) if matches else 0)
            if not matches:
                invalid.add(face_stem)
        results.append(entry)
    return results, invalid


def reconstruct(name, whole_path, pe_path, face_path, models, cal, check, invalid_faces, private=False, coverage=None):
    entry = {"name": name, "private": private}
    missing = [component for component, path in (("whole_image", whole_path), ("edit_features", pe_path))
               if path is None or not path.exists()]
    if missing:
        return {**entry, "status": "missing_required_cache", "missing_components": missing}
    try:
        a, b = arrays(whole_path), arrays(pe_path)
        aligned(a, b)
        ids = a["img_ids"].astype(str)
        labels = a["labels"]
        w, e, e2 = score(models["whole"], a), pe_score(models["edit"], b), pe_score(models["second"], b)
        best = np.full((2, len(ids)), -np.inf)
        has_faces = face_path is not None and face_path.exists()
        face_cache_status = "available" if has_faces else "missing"
        if has_faces:
            if not private and face_path.stem in invalid_faces:
                raise CacheError("current v4 face cache provenance failed")
            f = arrays(face_path)
            if not (set(models["whole"].keys) | set(models["swap"].keys)) <= f.keys():
                has_faces = False
                face_cache_status = "missing_current_model_features"
        if has_faces:
            rows = {}
            for row, photo in enumerate(ids):
                rows.setdefault(photo, []).append(row)
            for h, head in enumerate((models["whole"], models["swap"])):
                scores = score(head, f)
                for photo, value in zip(f["img_ids"].astype(str), scores):
                    for row in rows.get(photo, ()):
                        best[h, row] = max(best[h, row], value)
        red_ai, red_swap = best[0] > cal["face_ai_logit"], best[1] > cal["swap_ai_logit"]
        stages = {"whole": w > cal["ai_logit"], "edit": e > check["line"],
                  "second": e2 > check["line2"], "face_ai": red_ai, "face_swap": red_swap}
        runtime = Detector.__new__(Detector)
        runtime.calibration = cal
        verdicts = np.array([runtime.verdict(float(w[i]), [{"flagged": bool(red_ai[i] or red_swap[i])}],
                                             {"flagged": bool(stages["edit"][i]), "flagged2": bool(stages["second"][i])})
                             for i in range(len(ids))])
        entry.update(status="complete_cached_pipeline" if has_faces else "partial_missing_face_cache",
                     scope="pixels_only_no_metadata", available_components=["whole", "edit", "second"]
                     + (["face_ai", "face_swap"] if has_faces else []), face_cache_status=face_cache_status)
        entry["totals"] = counts(np.ones(len(ids), bool), verdicts, ids, stages)
        entry["full_verdict_counts"] = entry["totals"]["verdict_counts"] if has_faces else None
        entry["by_label"] = {kind: counts(mask, verdicts, ids, stages)
                             for kind, mask in (("real", labels == 0), ("ai", labels > 0)) if mask.any()}
        sources = np.full(len(ids), "private_aggregate") if private else a.get("sources", np.full(len(ids), "unspecified")).astype(str)
        entry["by_source"] = {}
        for source in np.unique(sources):
            for kind, label_mask in (("real", labels == 0), ("ai", labels > 0)):
                mask = (sources == source) & label_mask
                if mask.any():
                    entry["by_source"][source_name(source) + " | " + kind] = counts(mask, verdicts, ids, stages)
        entry["near_threshold_rows_within_0_001"] = {
            "whole": int((np.abs(w - cal["ai_logit"]) <= .001).sum()),
            "edit": int((np.abs(e - check["line"]) <= .001).sum()),
            "second": int((np.abs(e2 - check["line2"]) <= .001).sum()),
        }
        if private:
            entry["note"] = "Only neutral group totals; no private identities, paths, source tags, or per-image scores."
        if coverage is not None:
            coverage["rows"] += len(ids)
            coverage["ids"].update(ids.tolist())
        return entry
    except (CacheError, KeyError, ValueError, OSError):
        return {**entry, "status": "invalid_cache", "error": "Missing current-model keys, nonfinite values, or cache alignment/provenance failure; no metrics emitted."}


def face_metrics(path, models, cal, invalid_faces, name=None):
    entry = {"name": name or path.stem}
    if path.stem in invalid_faces:
        return {**entry, "status": "invalid_v4_provenance"}
    data = arrays(path)
    components = {}
    for component, head, strict, leans in (("ai_made", models["whole"], cal["face_ai_logit"], cal["face_leans_logit"]),
                                           ("swap", models["swap"], cal["swap_ai_logit"], cal["swap_leans_logit"])):
        try:
            s = score(head, data)
            components[component] = {}
            for label, mask in (("real", data["labels"] == 0), ("ai_or_swap", data["labels"] > 0)):
                if mask.any():
                    components[component][label] = {"n_faces": int(mask.sum()),
                        "strict": int((s[mask] > strict).sum()), "leans_or_strict": int((s[mask] > leans).sum()),
                        "leans_only": int(((s[mask] > leans) & (s[mask] <= strict)).sum()),
                        "unique_image_ids": len(set(data["img_ids"][mask].astype(str)))}
        except CacheError:
            components[component] = {"status": "missing_current_model_features"}
    return {**entry, "status": "complete" if "real" in components.get("swap", {})
            or "ai_or_swap" in components.get("swap", {}) else "partial_missing_swap_features", "components": components}


def source_face_metrics(path, head, cal):
    data = arrays(path)
    n = len(data["features"])
    s = score(head, {"features_facepe": data["features"].reshape(n, -1)})
    groups = {}
    for label, mask in (("real", data["labels"] == 0), ("ai_or_swap", data["labels"] > 0)):
        if mask.any():
            groups[label] = {"n_faces": int(mask.sum()), "strict": int((s[mask] > cal["swap_ai_logit"]).sum()),
                "leans_or_strict": int((s[mask] > cal["swap_leans_logit"]).sum()),
                "leans_only": int(((s[mask] > cal["swap_leans_logit"]) & (s[mask] <= cal["swap_ai_logit"])).sum()),
                "unique_image_ids": len(set(data["img_ids"][mask].astype(str))),
                "unique_groups": len(set(data["groups"][mask].astype(str))) if "groups" in data else None}
    return {"name": "v4 crops: " + path.stem, "status": "current_source_crop_features", "components": {"swap": groups}}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def markdown(report):
    lines = ["# Cached release metrics — 4 November 2026", "", report["method"], "",
             "These are cached reconstructions, not fresh live inference. Existing float16 caches can differ near thresholds. "
             "Calibration/dev figures are not independent test guarantees. Rows from overlapping sets must not be pooled into an accuracy.", "",
             "## Effective fixed thresholds", "", "```json", json.dumps(report["effective_thresholds"], indent=2), "```", "",
             "## Image counts", "", "AI counts include every AI verdict. Partial rows cover whole-image and both edit heads only.", "",
             "Unique counts refer to cached image IDs, not perceptually distinct photos. Different IDs may hold copies of the same original.", "",
             "| Set | Label | Scope | n / unique IDs | AI | Uncertain | Authentic |", "|---|---|---|---:|---:|---:|---:|"]
    for entry in report["image_sets"]:
        if "by_label" not in entry:
            lines.append(f"| {entry['name']} | — | {entry['status']} | — | — | — | — |")
            continue
        scope = "full cached" if entry["status"] == "complete_cached_pipeline" else "partial; faces unavailable"
        for label, c in entry["by_label"].items():
            lines.append(f"| {entry['name']} | {label} | {scope} | {c['n']} / {c['unique_image_ids']} | {c['ai']} ({c['ai_percent']:.2f}%) | {c['uncertain']} | {c['authentic']} |")
    lines += ["", "## Public source counts", "", "| Set / source | n | AI | Uncertain | Authentic |", "|---|---:|---:|---:|---:|"]
    for entry in report["image_sets"]:
        if entry.get("private"):
            continue
        for source, c in entry.get("by_source", {}).items():
            lines.append(f"| {entry['name']} / {source.replace('|', '/')} | {c['n']} | {c['ai']} ({c['ai_percent']:.2f}%) | {c['uncertain']} | {c['authentic']} |")
    lines += ["", "## Per-face counts at the app thresholds", "", "Leans-or-strict includes red faces; leans-only is orange. Face labels can mean fully generated faces or swaps, depending on the source.", "",
              "| Face set | Check / label | n faces | Strict | Leans or strict | Unique image IDs |", "|---|---|---:|---:|---:|---:|"]
    for entry in report["face_sets"] + report["v4_source_face_sets"]:
        for component, groups in entry.get("components", {}).items():
            for label, c in groups.items():
                if isinstance(c, dict):
                    lines.append(f"| {entry['name']} | {component} / {label} | {c['n_faces']} | {c['strict']} | {c['leans_or_strict']} | {c['unique_image_ids']} |")
    lines += ["", "## Calibration coverage", ""]
    for entry in report["calibration_coverage"]:
        lines.append(f"- {entry['pool']}: {entry['status']}; expected {entry['expected_real_rows']} real rows, observed {entry.get('observed_real_rows', 'unavailable')}.")
    lines += ["", "## Missing or non-independent evaluations", ""]
    for entry in report["evaluation_coverage"]:
        lines.append(f"- {entry['name']}: {entry['status']}. {entry['role']}")
    lines += ["", "## Face-cache provenance", ""]
    for entry in report["face_cache_provenance"]:
        lines.append(f"- {entry['face_set']}: {entry['status']}; current-v4 source {entry['source_set']}.")
    lines += ["", "No metadata labels, API output, uploaded pixels, or private per-photo identifiers appear in this report.", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--head", type=Path, default=Path("weights/google_v1/head_G3tt.pt"))
    parser.add_argument("--edit-check", type=Path, default=Path("weights/edit_check_v3/head_EC3pf.pt"))
    parser.add_argument("--face-source", type=Path, default=Path("features/face_pe_v4"))
    parser.add_argument("--include-private", action="store_true")
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    raw = json.loads(args.head.with_name(args.head.stem + "_calibration.json").read_text())
    edit_raw = json.loads(args.edit_check.with_name(args.edit_check.stem + "_calibration.json").read_text())
    cal = load_calibration(args.head)
    check = load_edit_check(args.edit_check, cal)
    if check is None or "head2" not in check:
        raise SystemExit("Current runtime joint calibration or second head unavailable; no report generated.")
    models = {"whole": LinearHead.load(args.head), "edit": check["head"], "second": check["head2"],
              "swap": LinearHead.load(cal["swap_head"])}
    if any(head.weight.device.type != "cpu" for head in models.values()):
        raise SystemExit("CPU-only cached evaluation requires CPU head tensors.")
    provenance, invalid = face_provenance(args.face_source)
    report = {
        "schema_version": 1, "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "method": "CPU-only scoring of existing feature caches with the runtime's fixed joint calibration and Detector.verdict; no recalibration, backbone loading, image reads, network, APIs, or GPU.",
        "models": {"whole_head": args.head.as_posix(), "edit_head": args.edit_check.as_posix(),
                   "second_head": edit_raw["head2"], "swap_head": cal["swap_head"], "swap_pe": cal["swap_pe"]},
        "head_sha256": {name: digest(path) for name, path in (("whole", args.head), ("edit", args.edit_check),
                                                               ("second", edit_raw["head2"]), ("swap", cal["swap_head"]))},
        "effective_thresholds": {key: cal[key] for key in ("ai_logit", "real_logit", "strong_ai_logit", "face_ai_logit",
                                                            "face_leans_logit", "swap_ai_logit", "swap_leans_logit", "temperature")},
        "face_cache_provenance": provenance, "image_sets": [], "face_sets": [], "v4_source_face_sets": [], "calibration_coverage": [],
        "evaluation_coverage": [],
        "limitations": ["Feature caches are float16; live inference can differ at strict threshold boundaries.",
                        "Missing face caches produce explicit partial metrics, never a full-verdict claim.",
                        "Private source categories and per-photo identifiers are suppressed.",
                        "Unique-ID counts are not perceptual deduplication; separately named copies of one original count as distinct IDs.",
                        "Current v4 copied face features are checked byte-for-value against the designated source caches; those caches do not store backbone checkpoint hashes."],
    }
    coverage = {"rows": 0, "ids": set()}
    report["effective_thresholds"].update(edit_logit=check["line"], edit2_logit=check["line2"])
    public = sorted(Path("features").glob("*_app.npz")) + sorted(Path("features/google").glob("*_app.npz"))
    for path in public:
        stem = path.stem.removesuffix("_app")
        relative = ("google/" if path.parent.name == "google" else "") + stem
        pe = Path("features/pe_ec3") / (relative + "_eval.npz")
        if stem == "commons_ai_v1_photo":
            pe = Path("features/pe_ec3/google/commons_ai_v1_photo_eval.npz")
        report["image_sets"].append(reconstruct(relative, path, pe, Path("features/faces") / (stem + "_eval.npz"),
                                                models, cal, check, invalid, coverage=coverage))
    private_paths = {}
    if args.include_private:
        private_paths = {label: stem for label, stem in PRIVATE_STEMS.items()
                         if Path(f"data/private/{stem}_app.npz").exists()}
        private_paths.update({"training selfies (reused)": "originals", "training person edits (reused)": "self_edits_v1"})
        for label, stem in private_paths.items():
            base = Path("data/private")
            pe = base / (stem + "_pe_EC3.npz")
            if stem == "self_edits_v1":
                pe = base / (stem + "_app_pe_EC3.npz")
            entry = reconstruct("private: " + label, base / (stem + "_app.npz"), pe,
                                base / (stem + "_faces_app.npz"), models, cal, check, invalid, private=True, coverage=coverage)
            report["image_sets"].append(entry)
            faces = base / (stem + "_faces_app.npz")
            if faces.exists():
                report["face_sets"].append(face_metrics(faces, models, cal, invalid, name="private: " + label))
    for pool, expected in edit_raw["real_pools"].items():
        stem = POOL_STEMS.get(pool)
        name = stem if stem else "private: " + pool
        dataset = next((entry for entry in report["image_sets"] if entry["name"] == name), {})
        observed = dataset.get("by_label", {}).get("real", {}).get("n")
        # People-free edit originals have no face cache in the calibrated pipeline.
        allowed_status = ("complete_cached_pipeline", "partial_missing_face_cache") if pool in ("originals v1", "originals v2") else ("complete_cached_pipeline",)
        valid = observed == expected and dataset.get("status") in allowed_status
        entry = {"pool": pool, "expected_real_rows": expected, "observed_real_rows": observed,
                 "status": "verified" if valid else ("private_not_requested" if pool in PRIVATE_STEMS and not args.include_private else "missing_or_invalid")}
        report["calibration_coverage"].append(entry)
    for path in sorted(Path("features/faces").glob("*_eval.npz")):
        report["face_sets"].append(face_metrics(path, models, cal, invalid))
    for path in sorted(args.face_source.glob("*.npz")):
        report["v4_source_face_sets"].append(source_face_metrics(path, models["swap"], cal))
    for name, role in INDEPENDENT_EXPECTED.items():
        entry = next((entry for entry in report["image_sets"] if entry["name"] == name), {})
        report["evaluation_coverage"].append({"name": name, "role": role,
            "status": entry.get("status", "no_current_full_pipeline_caches; run live inference")})
    report["evaluation_coverage"].extend([
        {"name": "person-edit cross-validation", "status": "training-reused person cache is not an independent test", "role": "Earlier folds used the same person's sessions; fresh edited selfies are still needed."},
        {"name": "forwarding and phone processing", "status": "not represented by original-view caches", "role": "Use live release robustness artifacts; cached clean metrics cannot establish transformation robustness."},
    ])
    report["required_calibration_caches_valid"] = all(e["status"] in ("verified", "private_not_requested") for e in report["calibration_coverage"])
    report["coverage_counts_not_an_accuracy_denominator"] = {"rows": coverage["rows"], "unique_image_ids": len(coverage["ids"]),
        "repeated_id_rows": coverage["rows"] - len(coverage["ids"])}
    for entry in report["image_sets"]:
        if "totals" not in entry:
            continue
        for c in [entry["totals"], *entry["by_label"].values(), *entry["by_source"].values()]:
            assert c["ai"] + c["uncertain"] + c["authentic"] == c["n"]
            assert sum(c["verdict_counts"].values()) == c["n"]
            assert c["unique_image_ids"] + c["repeated_id_rows"] == c["n"]
        assert sum(c["n"] for c in entry["by_label"].values()) == entry["totals"]["n"]
        assert sum(c["n"] for c in entry["by_source"].values()) == entry["totals"]["n"]
        assert (entry["full_verdict_counts"] is not None) == (entry["status"] == "complete_cached_pipeline")
    report["validation"] = {"count_partitions_consistent": True, "verdict_implementation": "src.inference.Detector.verdict",
                            "runtime_calibration_loaded_without_recalibration": True, "torch_threads": 4,
                            "cuda_visible_devices": "-1"}
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "cached_metrics.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (args.out_dir / "cached_metrics.md").write_text(markdown(report), encoding="utf-8")
    print(f"Cached report: {len(report['image_sets'])} image sets; {len(report['face_sets'])} face sets; "
          f"calibration caches valid: {report['required_calibration_caches_valid']}; "
          f"v4 exact matches: {sum(p['status'] == 'exact_match' for p in provenance)}/{len(provenance)}.")
    if not report["required_calibration_caches_valid"] or invalid:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
