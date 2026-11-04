"""Verify an explicitly supplied deployment using existing public images only; no paid endpoints.

Optional --judge-list reruns the 80 known public judge files against their saved
public byte hashes. Private photos are never accepted by that regression path.
"""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import httpx

BOUNDARY_SHA = "1e913aa9bad9d28a8f7303cf995493409b1324db5d6690fec789790337db67e3"
EXAMPLES = (("real", "1_real_phone_photo.jpg", "Likely authentic"),
            ("edit", "2_ai_edit_red_bicycle.jpg", "Likely AI-edited"),
            ("group", "3_ai_group_photo.jpg", "Likely AI-edited"),
            ("original", "4_ai_selfie_original_file.png", "AI-generated"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--output", type=Path, default=Path("results/deployment_2026-11-04/hosted_checks.json"))
    parser.add_argument("--boundary", type=Path)
    parser.add_argument("--judge-list", type=Path)
    args = parser.parse_args()
    report = {"checked_at_utc": datetime.now(timezone.utc).isoformat(), "base_url": args.base_url,
              "private_photos_used": False, "paid_endpoints_called": False,
              "checks": [], "judge_files": [], "pixel_component_differences": []}
    def save():
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    def check(name, passed, **details):
        report["checks"].append({"name": name, "passed": bool(passed), **details})
        save()
    with httpx.Client(base_url=args.base_url.rstrip("/"), timeout=180, follow_redirects=False) as client:
        health = client.get("/healthz")
        check("detector_ready", health.status_code == 200 and all(health.json().get("checks", {}).values()),
              status=health.status_code, body=health.json())
        home = client.get("/")
        check("four_clickable_public_examples", home.status_code == 200 and all(
            f'data-example="{name}"' in home.text for name, _, _ in EXAMPLES))
        for name, filename, expected in EXAMPLES:
            data = (Path("app/examples") / filename).read_bytes()
            sample = client.get("/examples/" + name)
            check("public_example_bytes_" + name, sample.status_code == 200 and sample.content == data)
            started = time.perf_counter()
            response = client.post("/api/investigate", files={"file": (filename, data)})
            body = response.json()
            check("real_inference_" + name, response.status_code == 200 and body.get("verdict") == expected,
                  status=response.status_code, verdict=body.get("verdict"), seconds=round(time.perf_counter()-started, 3),
                  faces_checked=body.get("faces_checked"), faces_flagged=body.get("faces_flagged"),
                  evidence_source=body.get("evidence_source"))
        bad = client.post("/api/investigate", files={"file": ("broken.jpg", b"not an image")})
        check("invalid_image_rejected", bad.status_code == 400, status=bad.status_code)
        expired = client.post("/api/investigate/not-a-session/explain")
        check("expired_explanation_rejected_before_api", expired.status_code == 410, status=expired.status_code)
        refused = client.post("/api/fool", json={"prompt": "a celebrity lookalike"})
        check("real_person_prompt_refused_before_api", refused.status_code == 400, status=refused.status_code)
        if args.boundary:
            data = args.boundary.read_bytes()
            if hashlib.sha256(data).hexdigest() != BOUNDARY_SHA:
                raise ValueError("Boundary check accepts only the previously audited PUBLIC image bytes.")
            response = client.post("/api/investigate", files={"file": (args.boundary.name, data)})
            body = response.json()
            check("public_real_boundary_keeps_likely", response.status_code == 200
                  and body.get("verdict") == "Likely AI-generated" and body.get("metadata", {}).get("ai_label") is None,
                  verdict=body.get("verdict"), public_image_sha256=BOUNDARY_SHA)
        if args.judge_list:
            reference = json.loads(Path("results/hardened_2026-11-04/live_predictions.json").read_text())["records"]
            with args.judge_list.open(encoding="utf-8-sig", newline="") as stream:
                rows = list(csv.DictReader(stream))
            allowed = {"judge/chatgpt", "judge/metaai", "judge/nanobanana_pro", "judge/nanobanana_2_1"}
            if len(rows) != 80 or any(r["source"] not in allowed or r["label"] != "1" for r in rows):
                raise ValueError("Only the known 80 public judge AI files are accepted.")
            # Validate the complete byte whitelist before making any requests.
            for row in rows:
                sha = hashlib.sha256(Path(row["path"]).read_bytes()).hexdigest()
                if sha not in reference or reference[sha]["metadata_label"] != "generated":
                    raise ValueError("Input does not match the known public judge byte whitelist.")
            for index, row in enumerate(rows, 1):
                data = Path(row["path"]).read_bytes()
                baseline = reference[hashlib.sha256(data).hexdigest()]
                response = client.post("/api/investigate", files={"file": (Path(row["path"]).name, data)})
                body = response.json()
                passed = response.status_code == 200 and body.get("verdict") == baseline["verdict"] \
                    and body.get("metadata", {}).get("ai_label") == baseline["metadata_label"]
                report["judge_files"].append({"public_file": Path(row["path"]).name, "source": row["source"],
                                              "passed": passed, "verdict": body.get("verdict")})
                changed = [key for key in ("faces_checked", "faces_flagged") if body.get(key) != baseline[key]]
                if body.get("edit") != {"flagged": baseline["edit"]["flagged"], "flagged2": baseline["edit"]["flagged2"]}:
                    changed.append("edit_flags")
                if changed:
                    report["pixel_component_differences"].append({"public_file": Path(row["path"]).name, "fields": changed})
                save()
                if index % 10 == 0:
                    print(f"Hosted judge checks: {index}/80", flush=True)
            check("judge_original_file_verdicts", all(r["passed"] for r in report["judge_files"]))
    report["passed"] = all(c["passed"] for c in report["checks"])
    report["limitations"] = ["Public examples and known regressions, not a hosted population-level accuracy estimate.",
                             "CPU/GPU floating-point scores need not match exactly; component differences are recorded.",
                             "No live Muse/SAM API success or billing test; those endpoints were not invoked."]
    save()
    print(json.dumps({"passed": report["passed"], "checks": len(report["checks"]),
                      "judge_files": len(report["judge_files"]), "pixel_component_differences": len(report["pixel_component_differences"])}))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
