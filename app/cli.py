"""Check photos from the command line, with exactly the web app's detector, verdict logic and wording.

  python -m app.cli photo.jpg more.png folder/            one line per photo, then the details
  python -m app.cli photo.jpg --json                      the web app's report as JSON lines (one per photo)
  python -m app.cli photo.jpg --evidence out/             also save the evidence picture (map + face boxes) as JPEG

Needs the trained weights (see README, "Run it yourself"). Runs fully locally: no Meta API call is made (the optional
investigator and explanation belong to the web app). Every photo gets the same result as an upload to the web app,
because the same functions do the work (app/server.py: open_image, analyze, report).
"""

import argparse
import json
import sys
from pathlib import Path

from fastapi import HTTPException

from app import server

SUFFIXES = (".jpg", ".jpeg", ".png", ".webp")


def photos_of(inputs):
    for item in inputs:
        p = Path(item)
        if p.is_dir():
            yield from sorted(q for q in p.iterdir() if q.suffix.lower() in SUFFIXES)
        else:
            yield p


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("inputs", nargs="+", help="photos (JPG, PNG, WEBP) or folders of photos")
    parser.add_argument("--json", action="store_true", help="print the web app's report as JSON, one line per photo")
    parser.add_argument("--evidence", type=Path, help="folder for the evidence pictures (<name>_evidence.jpg)")
    args = parser.parse_args()
    if args.evidence:
        args.evidence.mkdir(parents=True, exist_ok=True)

    failed = 0
    for path in photos_of(args.inputs):
        try:
            image, result = server.analyze(path.read_bytes())
        except HTTPException as e:
            failed += 1
            print(f"{path}: {e.detail}" if not args.json else json.dumps({"file": str(path), "error": e.detail}))
            continue
        found, base = server.report(image, result)
        if args.evidence:
            server.draw_faces(base, result["faces"]).convert("RGB").save(args.evidence / f"{path.stem}_evidence.jpg", quality=90)
        if args.json:
            found = {k: v for k, v in found.items() if k != "evidence"}  # the picture is a file, not a data URL, here
            print(json.dumps({"file": str(path), **found}))
            continue
        faces = found["faces_checked"]
        face_text = (f"{faces} face{'s' if faces != 1 else ''} checked, {found['faces_flagged']} flagged"
                     if found["faces"] else "no face found")
        edit = found["edit"]
        edit_text = ("AI-edit check: flagged" if edit and (edit["flagged"] or edit["flagged2"]) else
                     "AI-edit check: no clear signs" if edit else "AI-edit check: off")
        meta_text = found["metadata"]["text"] or "no AI label in the file"
        print(f"{path}: {found['verdict']}")
        print(f"    {found['reason']}")
        print(f"    AI evidence {found['probability_ai']:.0%} | {edit_text} | {face_text} | {meta_text} | {found['seconds']} s")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
