"""Audit release data identities without models, downloads, or private identifiers.

This reconstructs *reported* training source families from result notes; it is not
proof of the exact data that produced a weight file. NPZ reads access metadata
arrays only. Private inputs contribute aggregate counts and never examples.

Run: python -m src.eval.release_manifest_check
"""

from __future__ import annotations

import argparse
import csv
import glob
import hashlib
import json
import ntpath
import re
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


def normalized(value):
    return ntpath.normpath(str(value)).replace("\\", "/").lower()


def private(value):
    value = normalized(value)
    return any(token in value for token in
               ("/private/", "/private_", "/selfedits/", "muse_self_", "/self_train_"))


def csv_rows(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def metadata_rows(path):
    """Do not materialize feature tensors (some archives contain many GB)."""
    path = Path(path)
    if path.suffix == ".csv":
        return [dict(row, artifact=str(path)) for row in csv_rows(path) if row.get("path")]
    with np.load(path, allow_pickle=False) as data:
        if "img_ids" not in data.files:
            return []
        ids = data["img_ids"].astype(str)
        labels = data["labels"] if "labels" in data.files else np.full(len(ids), -1)
        sources = data["sources"].astype(str) if "sources" in data.files else [""] * len(ids)
        # Augmented copies are one identity for this audit.
        return [{"path": value, "label": str(label), "source": source, "artifact": str(path)}
                for value, label, source in dict.fromkeys(zip(ids, labels, sources))]


class Identities:
    def __init__(self, dataset_root, all_rows):
        self.root = Path(dataset_root)
        self.aliases = {}
        self.faces = {}
        self.pairs = {}
        self.ambiguous_crops = set()
        self.unmapped_derivatives = set()
        self.manifests = []
        self.by_stem = defaultdict(set)
        for row in all_rows:
            value = normalized(row["path"])
            if not private(value) and "/realfacecrops/" not in value:
                self.by_stem[Path(value).stem].add(value)
        for pattern in ("MuseImage/edits*/manifest.csv", "GoogleGen/*/manifest.csv"):
            for manifest in sorted(self.root.glob(pattern)):
                if private(manifest):
                    continue
                self.manifests.append(str(manifest))
                for row in csv_rows(manifest):
                    if row.get("source") and row.get("id"):
                        self.aliases[(normalized(manifest.parent), row["id"].lower())] = row["source"]
        for manifest in sorted(self.root.glob("MuseImage/pairs_v*/pairs.csv")):
            version = manifest.parent.name.removeprefix("pairs_")
            for row in csv_rows(manifest):
                key = f"{version}_{row['pair_id']}"
                img_id = row["img_id"]
                if re.fullmatch(r"[0-9a-f]{16}", img_id):
                    family = "sid:" + img_id
                else:
                    match = re.fullmatch(r"(v\d+)_(commons_\d+)", img_id)
                    family = (normalized(self.root / "Commons" / match[1] / row["split"] / (match[2] + ".jpg"))
                              if match else "pair-real:" + img_id)
                self.pairs[key] = family
                real = normalized(manifest.parent / row["real_file"])
                self.aliases[(real, "")] = family
                for folder in ("fake", "fake_matched"):
                    self.aliases[(normalized(manifest.parent / folder / (row["pair_id"] + ".png")), "")] = family
        for manifest in sorted((self.root / "FaceCrops").glob("*.csv")):
            if "before_dedup" in manifest.stem:
                continue
            for row in csv_rows(manifest):
                if row.get("path") and row.get("group"):
                    source = row.get("source", "").split("/")[0]
                    self.faces[normalized(row["path"])] = "face-group:" + source + ":" + row["group"].lower()

    def family(self, value, source="", seen=None):
        value = normalized(value)
        seen = set() if seen is None else seen
        if value in seen:
            return value
        seen.add(value)
        if private(value):
            # The caller must suppress this identity in all serialized results.
            return value
        alias = self.aliases.get((value, ""))
        if alias:
            return self.family(alias, seen=seen)
        if value in self.faces:
            return self.faces[value]
        path = Path(value)
        if "/realfacecrops/" in value:
            stem = re.sub(r"_\d+$", "", path.stem)
            if re.fullmatch(r"[0-9a-f]{16}", stem):
                return "sid:" + stem
            options = self.by_stem.get(stem, set())
            options = {self.family(option, seen=set(seen)) for option in options}
            if len(options) == 1:
                return options.pop()
            self.ambiguous_crops.add(value)
            return "unresolved-crop:" + stem
        if "/sid_set/" in value or source.startswith("sid_set/"):
            return "sid:" + path.stem
        if re.fullmatch(r"[0-9a-f]{16}|full_synthetic_\d+", value):
            return "sid:" + value
        pair = re.search(r"(v\d+_pair_\d+)", path.stem)
        if pair and pair[1] in self.pairs:
            return self.pairs[pair[1]]
        derivative = re.match(r"((?:edit|restore)_\d+)", path.stem)
        if derivative:
            origin = self.aliases.get((normalized(path.parent), derivative[1]))
            if origin:
                return self.family(origin, seen=seen)
            self.unmapped_derivatives.add(value)
            return "unresolved-derivative:" + normalized(path.parent / derivative[1])
        return value


def expand(patterns):
    return sorted({Path(path) for pattern in patterns for path in glob.glob(str(pattern), recursive=True)})


def indexes(rows, identities):
    exact, families = defaultdict(list), defaultdict(list)
    private_count = 0
    for row in rows:
        if private(row["path"]) or private(row["artifact"]):
            private_count += 1
            continue
        key = scoped_identity(row)
        family = identities.family(row["path"], row.get("source", ""))
        if family.startswith("sid:full_synthetic_"):
            family = key
        exact[key].append(row)
        families[family].append(row)
    return exact, families, private_count


def examples(keys, left, right, limit=3):
    return [{"identity": key,
             "training_artifact": left[key][0]["artifact"].replace("\\", "/"),
             "evaluation_artifact": right[key][0]["artifact"].replace("\\", "/")}
            for key in sorted(keys)[:limit]]


def scoped_identity(row):
    """SID's synthetic numbering restarts in validation: IDs need split scope."""
    value = normalized(row["path"])
    if re.fullmatch(r"full_synthetic_\d+", value):
        match = re.search(r"(train|validation|test)-\d+-of-\d+", normalized(row["artifact"]))
        return "sid-" + (match[1] if match else "unknown-split") + ":" + value
    return value


def verify_synthetic_numbering(train_rows, eval_rows, dataset_root, seconds=45):
    """Stream small Parquet batches to test raw-ID collisions across SID splits."""
    def by_id(rows):
        found = defaultdict(set)
        for row in rows:
            value = normalized(row["path"])
            shard = re.search(r"(?:train|validation|test)-\d+-of-\d+", normalized(row["artifact"]))
            if re.fullmatch(r"full_synthetic_\d+", value) and shard:
                found[value].add(shard[0])
        return found
    train, evaluation = by_id(train_rows), by_id(eval_rows)
    common = set(train) & set(evaluation)
    requested = defaultdict(set)
    for value in common:
        for shard in train[value] | evaluation[value]:
            requested[shard].add(value)
    result = {"raw_synthetic_ids_shared_across_splits": len(common), "records_hashed": 0,
              "verified_different_bytes_ids": 0, "verified_same_bytes_ids": 0,
              "unverified_ids": len(common), "examples": [], "batch_size": 8}
    try:
        import pyarrow.parquet as pq
    except ImportError:
        result["limitation"] = "pyarrow unavailable; byte verification skipped"
        return result
    started, hashes = time.monotonic(), {}
    for shard, targets in sorted(requested.items()):
        if time.monotonic() - started > seconds:
            break
        path = dataset_root / "SID_Set" / "data" / (shard + ".parquet")
        if not path.exists():
            continue
        remaining = set(targets)
        for batch in pq.ParquetFile(path).iter_batches(batch_size=8, columns=["img_id", "image"]):
            for row in batch.to_pylist():
                if row["img_id"] in remaining:
                    data = row["image"]["bytes"]
                    hashes[(shard, row["img_id"])] = (hashlib.sha256(data).hexdigest(), len(data))
                    remaining.remove(row["img_id"])
            if not remaining or time.monotonic() - started > seconds:
                break
    for value in sorted(common):
        left = [(shard, hashes[(shard, value)]) for shard in train[value] if (shard, value) in hashes]
        right = [(shard, hashes[(shard, value)]) for shard in evaluation[value] if (shard, value) in hashes]
        if not left or not right:
            continue
        same = any(a[1][0] == b[1][0] for a in left for b in right)
        result["verified_same_bytes_ids" if same else "verified_different_bytes_ids"] += 1
        if len(result["examples"]) < 3:
            result["examples"].append({"public_id": value, "same_bytes": same,
                                       "training_shard": left[0][0], "training_sha256": left[0][1][0],
                                       "evaluation_shard": right[0][0], "evaluation_sha256": right[0][1][0]})
    result["records_hashed"] = len(hashes)
    result["unverified_ids"] = len(common) - result["verified_same_bytes_ids"] - result["verified_different_bytes_ids"]
    result["elapsed_seconds"] = round(time.monotonic() - started, 2)
    result["interpretation"] = "Split-local IDs are scoped separately. Equal ID strings alone do not prove shared image content."
    return result


def file_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def hash_check(train_rows, eval_rows, seconds):
    """Exact bytes only. Size-filtering cannot miss a byte-identical file."""
    started = time.monotonic()
    train_paths = {Path(row["path"]) for row in train_rows
                   if not private(row["path"]) and not private(row["artifact"])}
    eval_paths = {Path(row["path"]) for row in eval_rows
                  if not private(row["path"]) and not private(row["artifact"])}
    by_size = defaultdict(list)
    missing_eval = 0
    for path in eval_paths:
        if path.is_file():
            by_size[path.stat().st_size].append(path)
        else:
            missing_eval += 1
    candidates, missing_train = [], 0
    for path in train_paths:
        if path.is_file():
            if path.stat().st_size in by_size:
                candidates.append(path)
        else:
            missing_train += 1
    digests, collisions, matched_train = {}, [], set()
    considered = 0
    for path in sorted(candidates):
        if time.monotonic() - started > seconds:
            break
        considered += 1
        digest = file_hash(path)
        for other in by_size[path.stat().st_size]:
            if other not in digests:
                digests[other] = file_hash(other)
            if digest == digests[other]:
                matched_train.add(str(path))
                if len(collisions) < 10:
                    collisions.append({"training": str(path).replace("\\", "/"),
                                       "evaluation": str(other).replace("\\", "/")})
    return {"method": "SHA-256 exact bytes; public files only; candidate selection by equal byte size",
            "training_public_references": len(train_paths), "evaluation_public_references": len(eval_paths),
            "unavailable_training_files_or_nonfile_ids": missing_train,
            "unavailable_evaluation_files_or_nonfile_ids": missing_eval,
            "equal_size_training_candidates": len(candidates), "candidates_hashed": considered,
            "complete_for_accessible_public_files": considered == len(candidates),
            "matching_training_file_count": len(matched_train), "examples": collisions,
            "elapsed_seconds": round(time.monotonic() - started, 2),
            "limitations": "Does not test decoded-pixel equality, recompression, perceptual duplicates, or inaccessible files."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=Path("D:/Datasets"))
    parser.add_argument("--out-dir", type=Path, default=Path("results/release_2026-11-04"))
    parser.add_argument("--hash-seconds", type=float, default=45)
    args = parser.parse_args()
    lists = Path("features/lists")
    files = set(lists.rglob("*.csv"))
    # G3tt = reported C->D->E->F->G3->G3tt lineage, rather than every cached experiment.
    whole_patterns = ["features/aug_v2/*.npz", "features/aug_v2/pairs_v2/*.npz",
                      "features/aug_v2/open_gen/*.npz", "features/aug_v2/gan/*.npz",
                      "features/aug_v2/google/bitmind_nb_train_train.npz"]
    pe_patterns = ["features/pe_ec3_train/*_train.npz", "features/pe_ec3_train/pairs_v2/*_train.npz",
                   "features/pe_ec3_train_fwd/**/*_train.npz"]
    profiles = {"whole_G3tt_reported_lineage": expand(whole_patterns),
                "PE_EC3_backbone_and_forward_head_reported_sources": expand(pe_patterns),
                "Google_head_candidate_sources_not_exact_provenance": expand([
                    "features/pe_ec3_train/google/*_train.npz",
                    "features/pe_ec3_train/google_v1_backup/*_train.npz"])}
    face_train = ("of_val_train of_train_train of_test_challenge_train dfdc_train rtfs_train real_sid real_commons "
                  "real_fairface real_openimages real_faceid550_train real_indian").split()
    face_root = args.dataset_root / "FaceCrops"
    profiles["face_v4_available_local_training_manifests"] = [face_root / (stem + ".csv")
                                                               for stem in face_train if (face_root / (stem + ".csv")).exists()]
    for paths in profiles.values():
        files.update(paths)
    eval_csv = [path for path in lists.rglob("*.csv") if
                any(token in path.stem.lower() for token in ("dev", "test", "val", "chameleon", "sd_faces", "muse_edits_v1", "commons_ai"))
                and "train" not in path.stem.lower() and not private(path)]
    eval_npz = expand(["features/*_app.npz", "features/*_eval.npz", "features/google/*_app.npz",
                       "features/face_pe_v4/*.npz", "features/faces/*_eval.npz"])
    eval_npz = [path for path in eval_npz if "_train" not in path.stem]
    face_eval = [path for path in face_root.glob("*.csv") if
                 (path.stem.startswith("pool_") or path.stem in ("of_val_dev", "of_test_dev", "dfdc_p3_dev", "rtfs_dev", "rtfs_test"))
                 and "before_dedup" not in path.stem]
    evaluations = sorted(set(eval_csv + eval_npz + face_eval))
    files.update(evaluations)
    # Private artifacts are read only for aggregate counts; no paths/IDs are serialized.
    private_files = list(Path("data/private").rglob("*.npz"))
    loaded, errors = {}, []
    for path in sorted(files):
        try:
            loaded[path] = metadata_rows(path)
        except (OSError, ValueError, KeyError, csv.Error) as error:
            errors.append({"artifact": str(path).replace("\\", "/"), "error_type": type(error).__name__})
    identities = Identities(args.dataset_root, [row for rows in loaded.values() for row in rows])
    profile_indexes, profile_stats = {}, {}
    for name, paths in profiles.items():
        rows = [row for path in paths for row in loaded.get(path, [])]
        exact, family, priv = indexes(rows, identities)
        profile_indexes[name] = (exact, family)
        profile_stats[name] = {"artifacts": [str(path).replace("\\", "/") for path in paths],
                               "metadata_rows_after_within_file_copy_dedup": len(rows),
                               "public_exact_identities": len(exact), "public_source_families": len(family),
                               "private_rows_suppressed": priv}
    checks = []
    for path in evaluations:
        exact, family, priv = indexes(loaded.get(path, []), identities)
        for name, (train_exact, train_family) in profile_indexes.items():
            common_exact, common_family = set(exact) & set(train_exact), set(family) & set(train_family)
            unresolved_family = {key for key in common_family if key.startswith("unresolved-")}
            common_family -= unresolved_family
            checks.append({"evaluation": str(path).replace("\\", "/"), "training_profile": name,
                           "public_identities": len(exact), "private_rows_suppressed": priv,
                           "exact_identity_overlap": len(common_exact), "source_family_overlap": len(common_family),
                           "unresolved_family_naming_collisions": len(unresolved_family),
                           "exact_examples": examples(common_exact, train_exact, exact),
                           "source_family_examples": examples(common_family, train_family, family)})
    broad = expand(["features/pe_ec3_train/**/*_train.npz"])
    backup = [path for path in broad if "backup" in str(path).lower()]
    backup_shared = []
    for path in backup:
        current = Path(str(path).replace("google_v1_backup", "google"))
        if current in loaded:
            a = {normalized(row["path"]) for row in loaded.get(path, []) if not private(row["path"])}
            b = {normalized(row["path"]) for row in loaded.get(current, []) if not private(row["path"])}
            backup_shared.append({"backup": path.as_posix(), "current": current.as_posix(),
                                  "shared_public_ids": len(a & b), "backup_public_ids": len(a), "current_public_ids": len(b)})
    private_counts = Counter(artifacts=len(private_files), readable_artifacts=0, metadata_rows=0, unreadable_artifacts=0)
    for path in private_files:
        try:
            private_counts["metadata_rows"] += len(metadata_rows(path))
            private_counts["readable_artifacts"] += 1
        except (OSError, ValueError, KeyError):
            private_counts["unreadable_artifacts"] += 1
    active_rows = [row for name, paths in profiles.items() if "candidate" not in name
                   for path in paths for row in loaded.get(path, [])]
    eval_rows = [row for path in evaluations for row in loaded.get(path, [])]
    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Local public CSV/NPZ metadata; reported active source families; candidate provenance kept separate.",
        "conclusion": "Provenance incomplete: overlap counts are findings within documented/available sources, not a clean bill for the exact deployed weights.",
        "provenance_evidence": ["results/muse_pairs_v1_heads.txt", "results/commons_v1_heads.txt", "results/generator_variety_v1.txt",
                                "results/open_gen_v1_heads.txt", "results/head_g_v1.txt", "results/edit_check_v3.txt",
                                "results/face_pe_v4.txt", "src/eval/edit_head_cv.py", "scripts/make_colab_notebook.py"],
        "training_profiles": profile_stats, "checks": checks,
        "recursive_glob_check": {"pattern": "features/pe_ec3_train/**/*_train.npz", "matched_artifacts": len(broad),
                                 "backup_artifacts_included": [path.as_posix() for path in backup],
                                 "backup_current_identity_intersections": backup_shared,
                                 "interpretation": "A recursive training glob includes historical backups and current Google data; this is possible accidental reweighting, not proof the deployed head used that glob."},
        "source_resolution": {"public_manifest_files_with_edit_source_lookup": len(identities.manifests),
                              "ambiguous_crop_source_identities": len(identities.ambiguous_crops),
                              "unmapped_derivative_public_paths": len(identities.unmapped_derivatives)},
        "private_aggregate_only": dict(private_counts),
        "synthetic_id_collision_verification": verify_synthetic_numbering(active_rows, eval_rows, args.dataset_root),
        "missing_final_face_training_manifest_names": [stem for stem in face_train if not (face_root / (stem + ".csv")).exists()],
        "hash_check": hash_check(active_rows, eval_rows, args.hash_seconds), "read_errors": errors,
        "limitations": [
            "Weight files do not serialize the exact training manifest list, feature-file hashes, run command, or data revision; profiles are reconstructed from checked-in result notes.",
            "Current versus backup Google features include data generated after the active EC3g head; neither candidate directory is assumed to be its exact training set.",
            "Source-family sharing can be intentional: PixArt generator holdout uses training-pair captions. Shared caption content is not an identical training image, but is not an independent-content test.",
            "Google edit/restore manifests omit original source paths; those derivatives cannot be traced fully to their source photos from manifests alone.",
            "Ambiguous numbered Commons crops cannot always be traced to v1/v2 originals because the crop filenames omit the original download round.",
            "Only locally present face-v4 manifests are checked. Missing Colab-only train/RTFS manifests prevent reproducing the complete face identity/near-duplicate exclusion audit.",
            "Private feature artifacts are counted without serializing paths or IDs. No private cross-artifact or train/eval source-identity leakage assertion is made.",
            "Many reported dev pools set calibration thresholds or informed model selection; absence of training overlap does not make them independent final tests.",
            "Exact hashes cover accessible public image files only. No perceptual or identity similarity model is run."]}
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "manifest_checks.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    positive = [check for check in checks if check["exact_identity_overlap"] or check["source_family_overlap"]]
    lines = ["# Release manifest audit", "", report["conclusion"], "",
             f"Inspected {len(loaded)} public/local metadata artifacts and {len(evaluations)} evaluation artifacts; {len(checks)} profile comparisons.",
             f"Private artifacts: {private_counts['artifacts']} total, {private_counts['readable_artifacts']} readable; aggregate rows {private_counts['metadata_rows']}.", "",
             "## Reconstructed training profiles", ""]
    for name, stat in profile_stats.items():
        lines.append(f"- {name}: {len(stat['artifacts'])} artifacts, {stat['public_exact_identities']} public exact identities, {stat['public_source_families']} source families.")
    lines += ["", "## Overlaps requiring interpretation", "",
              "Source-family overlap includes intentionally shared captions across generator tests. These counts alone do not establish training-image leakage.", "",
              "| Evaluation artifact | Training profile | Exact identity overlap | Shared source families |",
              "|---|---|---:|---:|"]
    for check in positive:
        lines.append(f"| {check['evaluation']} | {check['training_profile']} | {check['exact_identity_overlap']} | {check['source_family_overlap']} |")
    if not positive:
        lines.append("| No overlaps detected in available sources | — | 0 | 0 |")
    h = report["hash_check"]
    sid = report["synthetic_id_collision_verification"]
    lines += ["", "## Exact byte check", "",
              f"SHA-256: {h['matching_training_file_count']} matching training files; {h['candidates_hashed']}/{h['equal_size_training_candidates']} equal-size candidates checked. Complete for accessible public files: {h['complete_for_accessible_public_files']}.",
              f"Unavailable/non-file references: training {h['unavailable_training_files_or_nonfile_ids']}, evaluation {h['unavailable_evaluation_files_or_nonfile_ids']}.",
              "", "## Identifier collision verification", "",
              f"SID synthetic IDs repeated across train/validation: {sid['raw_synthetic_ids_shared_across_splits']}; verified different image bytes: {sid['verified_different_bytes_ids']}; verified identical bytes: {sid['verified_same_bytes_ids']}; unverified: {sid['unverified_ids']}. Split-local synthetic IDs are compared with split scope.",
              f"Unresolved crop/derivative name matches are reported separately in JSON: {sum(c['unresolved_family_naming_collisions'] for c in checks)} comparison-level matches. They are not counted as resolved source-family overlap.",
              "", "## Recursive backup inclusion", "",
              f"The recursive PE training glob matches {len(broad)} archives, including {len(backup)} backups. This is a reproducibility/reweighting risk; exact active-run inclusion is unproven.", "",
              "## Limits", ""]
    lines += ["- " + limitation for limitation in report["limitations"]]
    lines += ["", "Full per-artifact counts, public examples, source-resolution coverage and provenance references are in manifest_checks.json."]
    (args.out_dir / "manifest_checks.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"metadata_artifacts": len(loaded), "evaluation_artifacts": len(evaluations),
                      "comparisons": len(checks), "nonzero_overlap_comparisons": len(positive),
                      "backup_artifacts_in_recursive_glob": len(backup),
                      "hash_matching_training_files": h["matching_training_file_count"],
                      "conclusion": report["conclusion"]}, indent=2))


if __name__ == "__main__":
    main()
