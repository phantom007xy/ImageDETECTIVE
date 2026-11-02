"""Same-numbers check: does the web app (app/server.py) give exactly the detector's own numbers for the same files?

Every picture of a list goes (1) straight to src/inference.Detector.analyze, the way the evaluation scripts call it
(PIL open, turned upright, the file's bytes for the metadata check), and (2) through the web app's real upload path
(an HTTP multipart upload to /api/investigate, via FastAPI's test client). Verdict, calibrated AI evidence, AI-edit
flags, metadata label and face results must match exactly. With --save the numbers are written to a JSON file;
with --compare FILE they are also compared with an earlier run (e.g. before and after a UI change).
Public pictures only (the numbers per file are printed and saved).

Usage (from the repo root):
  python -m src.eval.ui_same_numbers --list features/lists/judge_test_ai_v1.csv --save results/ui_numbers_judge_ai.json
  python -m src.eval.ui_same_numbers --list features/lists/judge_test_ai_v1.csv --compare results/ui_numbers_judge_ai.json
"""

import argparse
import csv
import io
import json
from pathlib import Path

from fastapi.testclient import TestClient
from PIL import Image, ImageOps

from app import server


def direct(data):
    with Image.open(io.BytesIO(data)) as im:
        image = ImageOps.exif_transpose(im).convert("RGB")
    r = server.detector.analyze(image, file_bytes=data)
    checked = [f for f in r["faces"] if not f["too_small"]]
    return {"verdict": r["verdict"], "probability_ai": round(r["probability_ai"], 6),
            "edit": None if r["edit"] is None else [bool(r["edit"]["flagged"]), bool(r["edit"].get("flagged2"))],
            "ai_label": (r["metadata"] or {}).get("ai_label"), "faces": len(r["faces"]), "faces_checked": len(checked),
            "faces_flagged": sum(1 for f in checked if f.get("flagged"))}


def via_app(client, path, data):
    response = client.post("/api/investigate", files={"file": (Path(path).name, data)})
    if response.status_code != 200:
        return {"error": response.json().get("detail")}
    r = response.json()
    return {"verdict": r["verdict"], "probability_ai": round(r["probability_ai"], 6),
            "edit": None if r["edit"] is None else [r["edit"]["flagged"], r["edit"]["flagged2"]],
            "ai_label": r["metadata"]["ai_label"], "faces": len(r["faces"]), "faces_checked": r["faces_checked"],
            "faces_flagged": r["faces_flagged"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--list", type=Path, required=True, help="CSV image list (path, label, source); public pictures")
    parser.add_argument("--save", type=Path, help="write the numbers here")
    parser.add_argument("--compare", type=Path, help="compare with an earlier --save file")
    args = parser.parse_args()

    client = TestClient(server.app)
    rows = list(csv.DictReader(open(args.list, newline="")))
    numbers, mismatches = {}, 0
    for row in rows:
        data = Path(row["path"]).read_bytes()
        a, b = direct(data), via_app(client, row["path"], data)
        numbers[row["path"]] = b
        if a != b:
            mismatches += 1
            print(f"DIFFERENT: {Path(row['path']).name}\n  detector: {a}\n  web app:  {b}")
    print(f"{len(rows)} pictures: the web app matches the detector exactly on {len(rows) - mismatches}")
    if args.compare:
        before = json.loads(args.compare.read_text())
        changed = [p for p in numbers if p in before and before[p] != numbers[p]]
        print(f"compared with {args.compare.name}: {len(numbers) - len(changed)} of {len(numbers)} identical"
              + "".join(f"\n  changed: {Path(p).name}: {before[p]} -> {numbers[p]}" for p in changed))
    if args.save:
        args.save.write_text(json.dumps(numbers, indent=1))
        print(f"saved {args.save}")


if __name__ == "__main__":
    main()
