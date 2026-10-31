"""ImageDETECTIVE: check a photo -> verdict, confidence, evidence map, faces, SAM 3.1 outline, explanation.

Tab 1 "Check a photo": our detector (DINOv3 + calibrated head, SAM 3 faces, metadata check) gives the
verdict first; then, if switched on, Meta's SAM 3.1 outlines the object with the strongest AI evidence
and Muse Spark writes 2-3 plain sentences from the numbers. The verdict never waits on an API.
Tab 2 "Try to fool it": Muse Image makes a picture from a text prompt, the detector checks it, and a
crop + re-save shows what happens when the file's AI label is lost.

Run from the repo root:  python -m app.app   (then open http://127.0.0.1:7860)
"""

import io
from pathlib import Path

import gradio as gr
from PIL import Image, ImageOps

from src.inference import Detector, draw_faces, draw_outline, evidence_overlay
from src.services import explain, metadata, muse_image, sam_outline

# Provisional (31 Oct): head F = SID-Set + Muse pairs + Commons real photos and pairs + SD 1.5/2.1 faces
# + SD 3.5 Medium and Kandinsky 2.2 fakes (results/open_gen_v1_heads.txt, calibration_F_v1.txt).
# Before: head E (weights/commons_v1/head_E.pt), C (weights/muse_pairs_v1/head_C.pt), weights/head_path_a_sidset30.pt.
HEAD_PATH = "weights/open_gen_v1/head_F.pt"
detector = Detector(HEAD_PATH)


def analyze_bytes(data):
    """The file's bytes -> (upright RGB image, detector result with the metadata check)."""
    with Image.open(io.BytesIO(data)) as im:
        image = ImageOps.exif_transpose(im).convert("RGB")  # upright, as the phone shows it
    return image, detector.analyze(image, file_bytes=data)


def face_lines(faces):
    """One line per face, numbered like the boxes in the evidence map."""
    if not faces:
        return ["**Faces:** none found."]
    lines = [f"**Faces:** {len(faces)} found (box numbers in the evidence map)"]
    for n, face in enumerate(faces, 1):
        if face["too_small"]:
            lines.append(f"- Face {n}: too small to check")
        else:
            label = {"ai": "🔴 looks AI-made or AI-changed", "leans_ai": "🟠 shows some AI signs (not enough on its own)",
                     "clear": "🟢 no clear AI signs in this face"}[face["level"]]
            p = face["probability_ai"]
            lines.append(f"- Face {n}: {label}" + (f" (face AI probability {p:.0%})" if p is not None else ""))
    lines.append("<small>The face check is weaker than the whole-image check: green means no clear AI signs, "
                 "not proof that the face is real.</small>")
    return lines


def verdict_lines(result):
    p = result["probability_ai"]
    if result["verdict_reason"] == "metadata":
        confidence = f"**Reason:** {metadata.describe(result['metadata'])}"
    else:
        confidence = {"AI-generated": f"**Confidence:** {p:.0%} AI (no real test photo scored this high)",
                      "Likely AI-generated": f"**Confidence:** {p:.0%} AI",
                      "Likely authentic": f"**Confidence:** {1 - p:.0%} real"}.get(
            result["verdict"], f"**Whole-image AI probability:** {p:.0%}")
    lines = [f"## {result['verdict']}", confidence]
    if result["verdict_reason"] == "metadata":
        lines.append(f"*The detector alone, from the pixels: {result['pixel_verdict']} ({p:.0%} AI). "
                     "A forwarded or edited copy loses this label, so the pixel check is what still works then.*")
    elif metadata.describe(result["metadata"]):
        lines.append(metadata.describe(result["metadata"]))
    if result["verdict"] == "Likely authentic":
        lines.append("*This photo doesn't look AI-made or AI-changed. It doesn't confirm when or where it was taken.*")
    return lines + [""] + face_lines(result["faces"])


def footer(result):
    status = ("Thresholds calibrated on development photos: at most 5% of real dev photos are called AI, and at "
              "most 15% of a real/fake mix is Uncertain." if detector.calibrated else
              "**Preview:** thresholds are not calibrated yet.")
    return f"\n\n<small>Checked in {result['seconds']:.1f} s. {status} Edited regions are not checked yet.</small>"


def check(path, use_meta):
    """Yields the result in steps: our own verdict first, then the optional Meta extras."""
    if path is None:
        yield "Upload a photo first.", None, ""
        return
    image, result = analyze_bytes(Path(path).read_bytes())  # the original upload, with its metadata
    lines = verdict_lines(result)
    base_map = evidence_overlay(image, result["patch_logits"])
    yield "\n".join(lines) + footer(result), draw_faces(base_map, result["faces"]), (
        "*Asking Meta's SAM 3.1 and Muse Spark ...*" if use_meta else "")
    if not use_meta:
        return

    outline, evidence, evidence_map = None, None, draw_faces(base_map, result["faces"])
    if result["verdict"] != "Likely authentic":  # outlining the "hottest spot" of a real photo would mislead
        outline = sam_outline.outline(image, result["patch_logits"])
        if outline is None:
            lines.append("\n*SAM 3.1 outline not available right now; the coloured map shows the evidence.*")
        elif outline["mask"] is None:
            lines.append("\n*SAM 3.1 found no single object at the spot with the strongest evidence.*")
        else:
            inside = sam_outline.evidence_inside(result["patch_logits"], outline["mask"])
            evidence = tuple(detector.probability(v) for v in inside) if inside else None
            evidence_map = draw_faces(draw_outline(base_map, outline["mask"]), result["faces"])
            text = "**Outline (SAM 3.1):** the yellow outline marks the object with the strongest AI evidence"
            if evidence:
                text += f": inside it {evidence[0]:.0%} AI evidence, the rest of the photo {evidence[1]:.0%}"
            lines += ["", text + "."]
    yield "\n".join(lines) + footer(result), evidence_map, "*Muse Spark is writing a short explanation ...*"

    words = explain.explain(result, outline, evidence)
    yield "\n".join(lines) + footer(result), evidence_map, (
        f"**In plain words (Muse Spark):** {words}" if words else
        "*No explanation right now (daily limit reached or the API is unavailable). Everything above comes from "
        "our own detector.*")


def fool(prompt):
    if muse_image.asks_for_real_people(prompt or ""):
        return None, ("This demo doesn't make pictures of real or famous people, or lookalikes of them: that is "
                      "exactly the kind of fake ImageDETECTIVE is built against. Try a made-up person or a scene."), None
    data = muse_image.generate(prompt or "")
    if data is None:
        return None, ("Muse Image couldn't make this image: Meta's safety filter refused it, the daily limit is "
                      "reached, or the API is unavailable."), None
    image, result = analyze_bytes(data)
    text = ["### Muse Image made this picture. What does ImageDETECTIVE say?"] + verdict_lines(result)
    return draw_faces(evidence_overlay(image, result["patch_logits"]), result["faces"]), "\n".join(text), data


def crop(data):
    if data is None:
        return None, "Make a picture first."
    cropped = muse_image.crop_and_resave(data)
    image, result = analyze_bytes(cropped)
    text = ["### After cropping and re-saving (as often happens when a picture is forwarded)",
            "The file's AI label is gone now, so only the pixels can tell."] + verdict_lines(result)
    return draw_faces(evidence_overlay(image, result["patch_logits"]), result["faces"]), "\n".join(text)


# analytics_enabled=False: Gradio sends no usage statistics.
with gr.Blocks(title="ImageDETECTIVE", analytics_enabled=False) as demo:
    gr.Markdown("# ImageDETECTIVE\nIs this photo real, made by AI, or changed by AI? And **where** is the evidence?")
    with gr.Tab("Check a photo"):
        with gr.Row():
            # filepath + image_mode=None: Gradio hands over the uploaded file unchanged (with its metadata).
            photo = gr.Image(type="filepath", image_mode=None, label="Photo")
            evidence = gr.Image(type="pil", label="Evidence map (red = points to AI, blue = points to real), faces "
                                                  "(red = looks AI-made, orange = some AI signs, green = no clear AI "
                                                  "signs, grey = too small) and the SAM 3.1 outline (yellow)")
        use_meta = gr.Checkbox(value=True, label="Use Meta's SAM 3.1 outline and Muse Spark explanation (a smaller "
                                                 "copy of the photo goes to Meta's SAM 3.1 API; Muse Spark only gets "
                                                 "the numbers; standard mode: Meta doesn't train on them)")
        button = gr.Button("Check", variant="primary")
        result_text = gr.Markdown()
        explanation = gr.Markdown()
        button.click(check, inputs=[photo, use_meta], outputs=[result_text, evidence, explanation])
    with gr.Tab("Try to fool it (Muse Image)"):
        gr.Markdown("Type a prompt, Meta's **Muse Image** makes a picture, and ImageDETECTIVE checks it right away. "
                    "Then crop and re-save it, like a forwarded copy, and check again. Text prompts only: this app "
                    "never edits uploaded photos, and it doesn't make pictures of real or famous people.")
        prompt = gr.Textbox(label="Prompt", placeholder="e.g. a selfie of a young woman at a train station")
        made = gr.State(None)
        with gr.Row():
            make = gr.Button("Make it and check it", variant="primary")
            cut = gr.Button("Crop and re-save it, then check again")
        with gr.Row():
            fake_map = gr.Image(type="pil", label="Muse Image picture with the evidence map")
            fake_text = gr.Markdown()
        make.click(fool, inputs=prompt, outputs=[fake_map, fake_text, made])
        cut.click(crop, inputs=made, outputs=[fake_map, fake_text])
    gr.Markdown("<small>Built with DINOv3 · SAM 3 and SAM 3.1 · Muse Image · Muse Spark (Meta). A result is evidence, "
                "not proof; it can't check when or where a photo was taken, or classic manual edits.</small>")

if __name__ == "__main__":
    demo.launch()
