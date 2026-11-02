"""ImageDETECTIVE web app: a small FastAPI server for the investigation page, "Try to fool it" and About.

The page itself (app/web/: index.html, style.css, app.js, logo.png) is plain HTML, CSS and JavaScript. It talks to
this server's JSON API, which runs the same detector as the earlier Gradio app (src/inference.Detector: the DINOv3
whole-image check, the Muse-edit check on Meta's Perception Encoder, SAM 3 faces with the AI-made-face and face-swap
checks, the file's metadata). A second step runs Meta's investigator (Muse Spark + SAM 3.1) and the plain-language
explanation; the verdict never waits for it and never comes from it.

Uploads: JPG, PNG or WEBP, at most 20 MB, checked as uploaded (the original bytes, metadata included). The picture is
kept in this server's memory only, for at most 30 minutes (for the investigator step), and never written to disk.
For pictures that are not clearly authentic, the investigator sends a copy at most 1,024 px long to Meta's Model API
(standard mode). One check runs at a time (the models share one GPU or CPU).

Run from the repo root:  python -m app.server   (then open http://127.0.0.1:7860)
"""

import base64
import io
import os
import threading
import time
import uuid
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from huggingface_hub import snapshot_download
from PIL import Image, ImageOps
from pydantic import BaseModel

from src.inference import Detector, draw_faces, draw_outline
from src.services import explain, investigator, metadata, muse_image, sam_outline

HEAD_PATH = "weights/google_v1/head_G3tt.pt"
EDIT_CHECK_PATH = "weights/edit_check_v3/head_EC3pf.pt"
WEB = Path(__file__).parent / "web"
MAX_BYTES = 20 * 1024 * 1024
KEEP_SECONDS = 30 * 60
FORMATS = {"JPEG": "image/jpeg", "PNG": "image/png", "WEBP": "image/webp"}

# On a hosted demo the trained heads are not in the public code: they come from a PRIVATE Hugging Face model repo
# (scripts/upload_heads.py), read with the host's token (DFDC rule: the weights are never published).
if os.environ.get("HEADS_REPO") and not Path(HEAD_PATH).exists():
    snapshot_download(os.environ["HEADS_REPO"], local_dir="weights", token=os.environ.get("HF_TOKEN"))
detector = Detector(HEAD_PATH, edit_check=EDIT_CHECK_PATH)
lock = threading.Lock()  # one check at a time
memory = {}  # id -> (time, data); pictures for the follow-up steps, in memory only

TONE = {"AI-generated": "red", "Likely AI-generated": "red", "Likely AI-edited": "amber",
        "Part of this photo looks AI-made or AI-changed": "amber", "This face looks AI-made or AI-changed": "orange",
        "Likely authentic": "green", "Uncertain": "slate"}
CONTEXT = "This photo doesn't look AI-made or AI-changed. It doesn't confirm when or where it was taken."

app = FastAPI(title="ImageDETECTIVE", docs_url=None, redoc_url=None, openapi_url=None)
app.mount("/static", StaticFiles(directory=WEB), name="static")


@app.middleware("http")
async def limit_upload(request: Request, call_next):
    """Refuse bodies far above the 20 MB limit before reading them."""
    if int(request.headers.get("content-length") or 0) > MAX_BYTES + 1024 * 1024:
        return JSONResponse({"detail": "This file is larger than 20 MB."}, status_code=413)
    return await call_next(request)


def forget_old():
    for key in [k for k, (t, _) in memory.items() if time.time() - t > KEEP_SECONDS]:
        memory.pop(key, None)


def data_url(image, long_side=1600):
    """A display copy (JPEG, at most long_side px) as a data URL; the checks always use the original."""
    copy = image.copy()
    copy.thumbnail((long_side, long_side))
    buffer = io.BytesIO()
    copy.convert("RGB").save(buffer, format="JPEG", quality=90)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode()


def evidence_map(image, patch_logits, scale=4.0, weak=0.25):
    """The whole-image check's own patch scores over the photo: red = points to AI, pale green = points to a real photo,
    dark blue = a weak or mixed signal. The image score is exactly the average of these patch scores. Display only."""
    strength = np.tanh(patch_logits / scale)
    heat = np.zeros(strength.shape + (4,), dtype=np.uint8)
    ai, real = strength >= weak, strength <= -weak
    heat[ai] = (239, 68, 68, 0)
    heat[real] = (134, 239, 172, 0)
    heat[~(ai | real)] = (30, 58, 138, 0)
    alpha = np.where(ai, 70 + 130 * strength, np.where(real, 30 + 80 * -strength, 45))
    heat[..., 3] = np.clip(alpha, 0, 255).astype(np.uint8)
    layer = Image.fromarray(heat, "RGBA").resize(image.size, Image.BILINEAR)
    return Image.alpha_composite(image.convert("RGBA"), layer).convert("RGB")


def reason_text(result):
    """The short, evidence-based explanation of the verdict (the wording of the earlier app, app/ui.py)."""
    v, p = result["verdict"], result["probability_ai"]
    if result["verdict_reason"] == "metadata":
        return (f"{metadata.describe(result['metadata'])} From the pixels alone: {result['pixel_verdict']} "
                f"({p:.0%} AI evidence in the whole-image check). A forwarded copy loses this label; the pixel checks "
                "still work then.")
    if v == "AI-generated":
        return f"The whole image looks AI-made ({p:.0%} AI evidence); no real test photo scored this high."
    if v == "Likely AI-generated" and not result.get("whole_ai", True):
        return ("Meta's Perception Encoder sees how the pixels were rendered: this picture looks made by an AI image "
                f"model, although the whole-image check alone says {p:.0%}.")
    if v == "Likely AI-generated":
        return f"The whole image looks AI-made ({p:.0%} AI evidence)."
    if v == "Likely AI-edited":
        return ("Meta's Perception Encoder, fine-tuned on AI edits, sees the traces an AI image model leaves when it "
                "re-renders a photo; a fully AI-made picture can show them too.")
    if v == "This face looks AI-made or AI-changed":
        return "At least one face looks AI-made or swapped, although the picture as a whole does not."
    if v == "Likely authentic":
        return f"No clear signs of AI in the pixels ({1 - p:.0%} real in the whole-image check)."
    return "The checks don't point clearly either way, so ImageDETECTIVE doesn't decide."


def face_rows(faces):
    out = []
    for n, f in enumerate(faces, 1):
        out.append({"n": n, "too_small": bool(f["too_small"]), "level": f.get("level"), "kind": f.get("kind"),
                    "probability_ai": f.get("probability_ai")})
    return out


def report(image, result):
    """Everything the page shows, from the detector's real result (no invented scores)."""
    meta, edit = result["metadata"], result["edit"]
    checked = [f for f in result["faces"] if not f["too_small"]]
    base = evidence_map(image, result["patch_logits"])
    return {
        "verdict": result["verdict"], "tone": TONE.get(result["verdict"], "slate"), "reason": reason_text(result),
        "context": CONTEXT if result["verdict"] == "Likely authentic" else "",
        "probability_ai": result["probability_ai"], "calibrated": detector.calibrated,
        "edit": None if edit is None else {"flagged": bool(edit["flagged"]), "flagged2": bool(edit.get("flagged2"))},
        "faces": face_rows(result["faces"]), "faces_checked": len(checked),
        "faces_flagged": sum(1 for f in checked if f.get("flagged")),
        "metadata": {"read": meta is not None, "ai_label": (meta or {}).get("ai_label"),
                     "text": (metadata.describe(meta) or "") if meta else ""},
        "seconds": round(result["seconds"], 1),
        "evidence": data_url(draw_faces(base, result["faces"])),
    }, base


def open_image(data):
    """The upload's format (JPEG, PNG, WEBP only) and the upright picture, or an HTTP error the page can show."""
    try:
        with Image.open(io.BytesIO(data)) as im:
            if im.format not in FORMATS:
                raise HTTPException(415, "Please use a JPG, PNG or WEBP image.")
            image = ImageOps.exif_transpose(im)
            image.load()
            return image.convert("RGB")
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(400, "This file could not be read as an image. It may be damaged; please try another file.")


def analyze(data):
    image = open_image(data)
    with lock:
        result = detector.analyze(image, file_bytes=data)  # the original bytes: the metadata check reads the file as is
    return image, result


@app.get("/")
def index():
    return FileResponse(WEB / "index.html", headers={"Cache-Control": "no-cache"})


@app.post("/api/investigate")
def investigate(file: UploadFile):
    data = file.file.read(MAX_BYTES + 1)
    if len(data) > MAX_BYTES:
        raise HTTPException(413, "This file is larger than 20 MB.")
    if not data:
        raise HTTPException(400, "The file is empty.")
    image, result = analyze(data)
    forget_old()
    key = uuid.uuid4().hex
    body, base = report(image, result)
    memory[key] = (time.time(), {"image": image, "result": result, "base": base})
    body["id"] = key
    return body


@app.post("/api/investigate/{key}/explain")
def investigate_more(key: str):
    """The second step: Meta's investigator (not for pictures called authentic) and the plain-language explanation."""
    forget_old()
    if key not in memory:
        raise HTTPException(410, "This investigation has expired. Please start it again.")
    item = memory[key][1]
    image, result, base = item["image"], item["result"], item["base"]
    out = {"investigator": "not run", "named": None, "outline": None, "evidence": None, "explanation": None}
    outline, evidence, guess = None, None, None
    with lock:
        if result["verdict"] in ("Likely AI-edited", "Uncertain"):
            # Muse Spark names what an AI editor most likely changed; SAM 3.1 outlines it (never decides the verdict)
            found = investigator.investigate(image)
            if found is None:
                out["investigator"] = "unavailable"
            else:
                out["investigator"], out["named"] = "ran", found["guess"] or ""
                guess = found["guess"]
                if found["guess"] and found["mask"] is not None:
                    out["outline"] = "named object"
                    out["evidence"] = data_url(draw_faces(draw_outline(base, found["mask"]), result["faces"]))
        elif result["verdict"] != "Likely authentic":  # outlining the "hottest spot" of a real photo would mislead
            outline = sam_outline.outline(image, result["patch_logits"])
            if outline is None:
                out["investigator"] = "unavailable"
            else:
                out["investigator"] = "ran"
                if outline["mask"] is not None:
                    inside = sam_outline.evidence_inside(result["patch_logits"], outline["mask"])
                    evidence = tuple(detector.probability(v) for v in inside) if inside else None
                    out["outline"] = "strongest object"
                    out["evidence"] = data_url(draw_faces(draw_outline(base, outline["mask"]), result["faces"]))
        words = explain.explain(result, outline, evidence, guess)  # numbers only, never the picture
    out["explanation"] = words or None
    return out


class Prompt(BaseModel):
    prompt: str


@app.post("/api/fool")
def fool(body: Prompt):
    prompt = (body.prompt or "").strip()
    if not prompt:
        raise HTTPException(400, "Please type a prompt first.")
    if muse_image.asks_for_real_people(prompt):
        raise HTTPException(400, "This demo doesn't make pictures of real or famous people, or lookalikes of them: that "
                                 "is exactly the kind of fake ImageDETECTIVE is built against. Try a made-up person or a scene.")
    data = muse_image.generate(prompt)
    if data is None:
        raise HTTPException(503, "Muse Image couldn't make this picture: Meta's safety filter refused it, the daily limit "
                                 "is reached, or the API is unavailable. No picture was substituted.")
    image, result = analyze(data)
    forget_old()
    key = uuid.uuid4().hex
    memory[key] = (time.time(), {"data": data})
    found, _ = report(image, result)
    return {"id": key, "picture": data_url(image), "report": found}


@app.post("/api/fool/{key}/crop")
def fool_crop(key: str):
    forget_old()
    if key not in memory or "data" not in memory[key][1]:
        raise HTTPException(410, "This picture has expired. Please make a new one.")
    data = muse_image.crop_and_resave(memory[key][1]["data"])
    image, result = analyze(data)
    found, _ = report(image, result)
    return {"picture": data_url(image), "report": found}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=os.environ.get("HOST", "127.0.0.1"), port=int(os.environ.get("PORT", "7860")))
