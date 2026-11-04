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
import contextlib
import io
import os
import threading
import time
import uuid
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from huggingface_hub import snapshot_download
from PIL import Image, ImageDraw, ImageOps
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool
from starlette.formparsers import MultiPartException, MultiPartParser
from python_multipart.exceptions import MultipartParseError

from src.inference import Detector, draw_faces, draw_outline
from src.services import explain, investigator, metadata, muse_image, sam_outline

HEAD_PATH = "weights/google_v1/head_G3tt.pt"
EDIT_CHECK_PATH = "weights/edit_check_v3/head_EC3pf.pt"
WEB = Path(__file__).parent / "web"
MAX_BYTES = 20 * 1024 * 1024
MAX_PIXELS = 40_000_000
KEEP_SECONDS = 29 * 60  # + a sweep every 30 s: an upload is gone at most 29.5 min after its check ("at most 30 minutes")
SWEEP_SECONDS = 30
FORMATS = {"JPEG": "image/jpeg", "PNG": "image/png", "WEBP": "image/webp"}
DISPLAY_LONG_SIDE = 1600
MAX_SESSIONS = 32
MAX_SESSION_BYTES = 128 * 1024 * 1024
EXAMPLES = {"real": "1_real_phone_photo.jpg", "edit": "2_ai_edit_red_bicycle.jpg",
            "group": "3_ai_group_photo.jpg", "original": "4_ai_selfie_original_file.png"}

# On a hosted demo the trained heads are not in the public code: they come from a PRIVATE Hugging Face model repo
# (scripts/upload_heads.py), read with the host's token (DFDC rule: the weights are never published).
if os.environ.get("HEADS_REPO"):  # hosted: our private heads repo (only changed files are downloaded again)
    snapshot_download(os.environ["HEADS_REPO"], local_dir="weights", token=os.environ.get("HF_TOKEN"))
detector = Detector(HEAD_PATH, edit_check=EDIT_CHECK_PATH)
if not all(detector.readiness().values()):
    raise RuntimeError("Required detector assets are missing or uncalibrated; restore all configured model files before serving.")
lock = threading.Lock()  # one check at a time
memory = {}  # id -> (time, data); pictures for the follow-up steps, in memory only
memory_lock = threading.Lock()  # memory is used by request threads and the cleaner

TONE = {"AI-generated": "red", "Likely AI-generated": "red", "Likely AI-edited": "amber",
        "Part of this photo looks AI-made or AI-changed": "amber", "This face looks AI-made or AI-changed": "orange",
        "Likely authentic": "green", "Uncertain": "slate"}
CONTEXT = "This photo doesn't look AI-made or AI-changed. It doesn't confirm when or where it was taken."

def cleaner(stop):
    """Deletes expired uploads every SWEEP_SECONDS, also when nobody uses the app."""
    while not stop.wait(SWEEP_SECONDS):
        forget_old()


@contextlib.asynccontextmanager
async def lifespan(_app):
    stop = threading.Event()
    thread = threading.Thread(target=cleaner, args=(stop,), daemon=True, name="upload-cleaner")
    thread.start()
    yield
    stop.set()
    thread.join(timeout=5)
    with memory_lock:
        memory.clear()  # nothing outlives the server


app = FastAPI(title="ImageDETECTIVE", docs_url=None, redoc_url=None, openapi_url=None, lifespan=lifespan)
app.mount("/static", StaticFiles(directory=WEB), name="static")


@app.middleware("http")
async def limit_upload(request: Request, call_next):
    """Refuse bodies far above the 20 MB limit before reading them."""
    try:
        length = int(request.headers.get("content-length") or 0)
    except ValueError:
        return JSONResponse({"detail": "Invalid upload size."}, status_code=400)
    if length < 0:
        return JSONResponse({"detail": "Invalid upload size."}, status_code=400)
    if length > MAX_BYTES + 1024 * 1024:
        return JSONResponse({"detail": "This file is larger than 20 MB."}, status_code=413)
    return await call_next(request)


def forget_old():
    with memory_lock:
        for key in [k for k, (t, _) in memory.items() if time.time() - t > KEEP_SECONDS]:
            memory.pop(key, None)


def remember(data):
    key = uuid.uuid4().hex
    with memory_lock:
        for old in [k for k, (t, _) in memory.items() if time.time() - t > KEEP_SECONDS]:
            memory.pop(old, None)
        size = retained_size(data)
        if size > MAX_SESSION_BYTES:
            raise HTTPException(413, "This investigation is too large to keep. Please use a smaller image.")
        while memory and (len(memory) >= MAX_SESSIONS or
                          sum(retained_size(item) for _, item in memory.values()) + size > MAX_SESSION_BYTES):
            memory.pop(next(iter(memory)))
        memory[key] = (time.time(), data)
    return key


def retained_size(value):
    """Conservative pixel storage estimate; no full-resolution upload is retained."""
    if isinstance(value, Image.Image):
        return value.width * value.height * 4
    if isinstance(value, (bytes, str)):
        return len(value)
    if isinstance(value, dict):
        return 256 + sum(retained_size(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return 64 + sum(retained_size(v) for v in value)
    return int(getattr(value, "nbytes", 64))


def display_copy(image, result):
    """Keep a display-sized copy after original-resolution inference; scale boxes with it."""
    copy = image.copy()
    copy.thumbnail((DISPLAY_LONG_SIDE, DISPLAY_LONG_SIDE))
    sx, sy = copy.width / image.width, copy.height / image.height
    faces = [dict(f, box=[f["box"][0] * sx, f["box"][1] * sy,
                         f["box"][2] * sx, f["box"][3] * sy]) for f in result["faces"]]
    return copy, dict(result, faces=faces)


def recall(key):
    """The stored data of an upload, or None once it has expired."""
    with memory_lock:
        entry = memory.get(key)
    return None if entry is None or time.time() - entry[0] > KEEP_SECONDS else entry[1]


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
        return f"The image has strong AI evidence ({p:.0%} in the whole-image check)."
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
    source = result.get("evidence_source", "whole")
    if source in ("edit", "render"):
        bounds = tuple(round(v * (image.width if i % 2 == 0 else image.height))
                       for i, v in enumerate(result["evidence_bounds"]))
        base = image.copy()
        region = image.crop(bounds)
        base.paste(evidence_map(region, result["evidence_logits"]), bounds[:2])
        ImageDraw.Draw(base).rectangle(bounds, outline=(225, 225, 225), width=max(1, image.width // 800))
        evidence_note = ("Perception Encoder evidence from the check that raised this finding. "
                         "It analyzes the outlined center square; the outer area is not scored by this check. "
                         "Colors show model contributions, not a proven edit boundary.")
    else:
        base = evidence_map(image, result["patch_logits"])
        evidence_note = ("DINOv3 whole-image evidence. Colors show the model's contributions, not proof of authenticity "
                         "or exact edit boundaries. Face boxes show the separate face checks.")
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
        "evidence_source": "Perception Encoder · AI-edit check" if source == "edit" else
                           "Perception Encoder · AI-render check" if source == "render" else "DINOv3 · Whole-image check",
        "evidence_note": evidence_note,
        "evidence": data_url(draw_faces(base, result["faces"])),
    }, base


def open_image(data):
    """The upload's format (JPEG, PNG, WEBP only) and the upright picture, or an HTTP error the page can show."""
    try:
        with Image.open(io.BytesIO(data)) as im:
            if im.format not in FORMATS:
                raise HTTPException(415, "Please use a JPG, PNG or WEBP image.")
            if im.width * im.height > MAX_PIXELS:
                raise HTTPException(413, "Please resize this image to at most 40 megapixels.")
            image = ImageOps.exif_transpose(im)
            image.load()
            return image.convert("RGB")
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(400, "This file could not be read as an image. It may be damaged; please try another file.")


def analyze(data):
    with lock:
        image = open_image(data)
        result = detector.analyze(image, file_bytes=data)  # the original bytes: the metadata check reads the file as is
    return image, result


@app.get("/")
def index():
    return FileResponse(WEB / "index.html", headers={"Cache-Control": "no-cache"})


@app.get("/examples/{name}")
def example(name: str):
    if name not in EXAMPLES:
        raise HTTPException(404, "This example was not found.")
    return FileResponse(Path(__file__).parent / "examples" / EXAMPLES[name])


@app.get("/healthz")
def health():
    checks = detector.readiness()
    ready = all(checks.values())
    return JSONResponse({"status": "ready" if ready else "unavailable", "checks": checks}, status_code=200 if ready else 503)


@app.post("/api/investigate")
async def investigate(request: Request):
    """Bound multipart data before its spool can ever roll to a temporary disk file."""
    if request.headers.get("content-type", "").split(";", 1)[0].strip().lower() != "multipart/form-data":
        raise HTTPException(422, "Please choose an image file.")

    async def bounded_stream():
        total = 0
        async for chunk in request.stream():
            total += len(chunk)
            if total > MAX_BYTES + 1024 * 1024:
                raise MultiPartException("This file is larger than 20 MB.")
            yield chunk

    parser = MemoryUploadParser(request.headers, bounded_stream(), max_files=1, max_fields=0)
    try:
        form = await parser.parse()
    except (MultiPartException, MultipartParseError) as error:
        status = 413 if "larger than" in str(error) else 400
        raise HTTPException(status, str(error))
    try:
        file = form.get("file")
        if file is None or not hasattr(file, "file"):
            raise HTTPException(422, "Please choose an image file.")
        data = await file.read()
        if not data:
            raise HTTPException(400, "The file is empty.")
    finally:
        await form.close()
    return await run_in_threadpool(investigate_data, data)


class MemoryUploadParser(MultiPartParser):
    spool_max_size = MAX_BYTES + 1

    async def parse(self):
        try:
            return await super().parse()
        except BaseException:
            # Starlette closes multipart-policy failures itself; also close malformed
            # parser input and cancelled streams. Closing an already closed spool is safe.
            for file in self._files_to_close_on_error:
                file.close()
            raise

    def on_part_begin(self):
        super().on_part_begin()
        self.upload_size = 0

    def on_part_data(self, data, start, end):
        if self._current_part.file is not None:
            self.upload_size += end - start
            if self.upload_size > MAX_BYTES:
                raise MultiPartException("This file is larger than 20 MB.")
        super().on_part_data(data, start, end)


def investigate_data(data):
    image, result = analyze(data)
    image, result = display_copy(image, result)
    body, base = report(image, result)
    key = remember({"image": image, "result": result, "base": base})
    body["id"] = key
    return body


@app.post("/api/investigate/{key}/explain")
def investigate_more(key: str):
    """The second step: Meta's investigator (not for pictures called authentic) and the plain-language explanation."""
    item = recall(key)
    if item is None or not all(k in item for k in ("image", "result", "base")):
        raise HTTPException(410, "This investigation has expired. Please start it again.")
    image, result, base = item["image"], item["result"], item["base"]
    out = {"investigator": "not run", "named": None, "outline": None, "evidence": None, "explanation": None}
    outline, evidence, guess = None, None, None
    if result["verdict"] in ("Likely AI-edited", "Uncertain"):
        # API work has its own budget lock; it must never hold the model-inference lock.
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
        pe_evidence = result.get("evidence_source") in ("edit", "render")
        outline = sam_outline.outline(image, result.get("evidence_logits", result["patch_logits"]),
                                      result.get("evidence_bounds")) if pe_evidence else sam_outline.outline(image, result["patch_logits"])
        if outline is None:
            out["investigator"] = "unavailable"
        else:
            out["investigator"] = "ran"
            if outline["mask"] is not None:
                inside = None if pe_evidence else sam_outline.evidence_inside(result["patch_logits"], outline["mask"])
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
    if len(prompt) > 1000:
        raise HTTPException(400, "Please keep the prompt under 1,000 characters.")
    if muse_image.asks_for_real_people(prompt):
        raise HTTPException(400, "This demo doesn't make pictures of real or famous people, or lookalikes of them: that "
                                 "is exactly the kind of fake ImageDETECTIVE is built against. Try a made-up person or a scene.")
    data = muse_image.generate(prompt)
    if data is None:
        raise HTTPException(503, "Muse Image couldn't make this picture: Meta's safety filter refused it, the daily limit "
                                 "is reached, or the API is unavailable. No picture was substituted.")
    image, result = analyze(data)
    key = remember({"data": data})
    found, _ = report(image, result)
    return {"id": key, "picture": data_url(image), "report": found}


@app.post("/api/fool/{key}/crop")
def fool_crop(key: str):
    item = recall(key)
    if item is None or "data" not in item:
        raise HTTPException(410, "This picture has expired. Please make a new one.")
    data = muse_image.crop_and_resave(item["data"])
    image, result = analyze(data)
    found, _ = report(image, result)
    return {"picture": data_url(image), "report": found}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=os.environ.get("HOST", "127.0.0.1"), port=int(os.environ.get("PORT", "7860")))
