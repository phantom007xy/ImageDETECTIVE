"""ImageDETECTIVE: check a photo -> verdict, confidence, where the evidence is, faces, Meta's investigator.

A plain, tool-like page: a menu on the left (light/dark switch, "Check a photo", "Try to fool it", "About") and the
chosen page on the right.
"Check a photo": our detectors give the verdict first (DINOv3 + calibrated head for the whole image, the Muse-edit
check on Meta's Perception Encoder, SAM 3 faces, the file's metadata). Then Meta's investigator runs: for AI-edited or
uncertain photos Muse Spark names what was most likely changed and SAM 3.1 outlines it; for AI-made photos SAM 3.1
outlines the object with the strongest evidence; Muse Spark then writes 2-3 plain sentences from the numbers. Photos
called authentic never go to the API. The verdict never waits on an API and never comes from Muse Spark.
"Try to fool it": Muse Image makes a picture from a text prompt, the detector checks it, and a crop + re-save shows
what happens when the file's AI label is lost.
The look (colours, verdict card, tables, About text) is in app/ui.py.

Run from the repo root:  python -m app.app   (then open http://127.0.0.1:7860)
"""

import io
import os
from functools import partial
from pathlib import Path

import gradio as gr
from huggingface_hub import snapshot_download
from PIL import Image, ImageOps

from app import ui
from src.inference import Detector, draw_faces, draw_outline, evidence_overlay
from src.services import explain, investigator, muse_image, sam_outline

# Whole-image head F (results/open_gen_v1_heads.txt, calibration_F_v1.txt) and the Muse-edit check on Meta's
# Perception Encoder, fine-tuned (results/edit_check_v3.txt), its line calibrated jointly with head F.
HEAD_PATH = "weights/google_v1/head_G3tt.pt"
EDIT_CHECK_PATH = "weights/edit_check_v3/head_EC3pf.pt"
# On the hosted demo the trained heads are not in the public code: they come from a PRIVATE Hugging Face model
# repo (scripts/upload_heads.py), read with the Space's token (DFDC: the weights are never published).
if os.environ.get("HEADS_REPO") and not Path(HEAD_PATH).exists():
    snapshot_download(os.environ["HEADS_REPO"], local_dir="weights", token=os.environ.get("HF_TOKEN"))
detector = Detector(HEAD_PATH, edit_check=EDIT_CHECK_PATH)
EXAMPLES = sorted(str(p) for p in Path("app/examples").glob("*.*") if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp"))


def analyze_bytes(data):
    """The file's bytes -> (upright RGB image, detector result with the metadata check)."""
    with Image.open(io.BytesIO(data)) as im:
        image = ImageOps.exif_transpose(im).convert("RGB")  # upright, as the phone shows it
    return image, detector.analyze(image, file_bytes=data)


def check(path):
    """Yields the page in steps: our own verdict first, then Meta's investigator and the explanation.
    Outputs: verdict card, evidence image, original image, checks table, faces, explanation."""
    if path is None:
        yield ui.placeholder_card(), None, None, "", "", ""
        return
    yield ui.placeholder_card("Checking the pixels, the faces and the file ..."), None, None, "", "", ""
    image, result = analyze_bytes(Path(path).read_bytes())  # the original upload, with its metadata
    base_map = evidence_overlay(image, result["patch_logits"])
    evidence_map = draw_faces(base_map, result["faces"])
    card = ui.verdict_card(result) + ui.footer_note(result["seconds"], detector.calibrated)
    yield card, evidence_map, image, ui.checks_html(result), ui.faces_html(result["faces"]), ""

    outline, evidence, guess, named = None, None, None, None
    if result["verdict"] in ("Likely AI-edited", "Uncertain"):
        # Muse Spark names what an AI editor most likely changed, SAM 3.1 outlines it
        # (results/investigator_v1.txt: rarely wrong when it names something; never decides the verdict).
        found = investigator.investigate(image)
        if found is not None:  # None: API unavailable or daily limit reached; everything else still stands
            named = found["guess"] or ""
            if found["guess"]:
                guess = found["guess"]
                if found["mask"] is not None:
                    evidence_map = draw_faces(draw_outline(base_map, found["mask"]), result["faces"])
    elif result["verdict"] != "Likely authentic":  # outlining the "hottest spot" of a real photo would mislead
        outline = sam_outline.outline(image, result["patch_logits"])
        if outline is not None and outline["mask"] is not None:
            inside = sam_outline.evidence_inside(result["patch_logits"], outline["mask"])
            evidence = tuple(detector.probability(v) for v in inside) if inside else None
            evidence_map = draw_faces(draw_outline(base_map, outline["mask"]), result["faces"])
    checks = ui.checks_html(result, named)
    yield card, evidence_map, image, checks, ui.faces_html(result["faces"]), ui.explanation_html(waiting=True)

    words = explain.explain(result, outline, evidence, guess)
    yield (card, evidence_map, image, checks, ui.faces_html(result["faces"]),
           ui.explanation_html(words) if words else '<div class="note">No explanation right now (daily limit reached or '
                                                    'the API is unavailable). Everything above comes from our own detectors.</div>')


def fool(prompt):
    if muse_image.asks_for_real_people(prompt or ""):
        return None, ui.placeholder_card("This demo doesn't make pictures of real or famous people, or lookalikes of "
                                         "them: that is exactly the kind of fake ImageDETECTIVE is built against. Try a "
                                         "made-up person or a scene."), "", None
    data = muse_image.generate(prompt or "")
    if data is None:
        return None, ui.placeholder_card("Muse Image couldn't make this picture: Meta's safety filter refused it, the "
                                         "daily limit is reached, or the API is unavailable."), "", None
    image, result = analyze_bytes(data)
    return (draw_faces(evidence_overlay(image, result["patch_logits"]), result["faces"]), ui.verdict_card(result),
            ui.checks_html(result), data)


def crop(data):
    if data is None:
        return None, ui.placeholder_card("Make a picture first."), ""
    image, result = analyze_bytes(muse_image.crop_and_resave(data))
    note = ('<div class="note">Cropped and re-saved, like a forwarded copy: the file\'s AI label is gone, so only the '
            'pixels can tell.</div>')
    return (draw_faces(evidence_overlay(image, result["patch_logits"]), result["faces"]), ui.verdict_card(result) + note,
            ui.checks_html(result))


PAGES = ("Check a photo", "Try to fool it", "About")


def show(page):
    """Menu click: show that page, mark its menu entry."""
    return ([gr.Column(visible=p == page) for p in PAGES]
            + [gr.Button(variant="primary" if p == page else "secondary") for p in PAGES])


# analytics_enabled=False: Gradio sends no usage statistics. delete_cache: every 30 minutes, uploaded and generated
# files older than 30 minutes are deleted from the server.
with gr.Blocks(title="ImageDETECTIVE", analytics_enabled=False, delete_cache=(1800, 1800)) as demo:
    with gr.Sidebar(width=230):
        with gr.Row(equal_height=True):
            gr.HTML('<div class="nav-label">Menu</div>')
            theme = gr.Button("\u25d0", size="sm", scale=0, min_width=36, elem_id="theme-toggle")  # light/dark switch
        menu = [gr.Button(p, variant="primary" if p == PAGES[0] else "secondary", size="sm", elem_classes="nav")
                for p in PAGES]
    gr.HTML(ui.TITLE)

    with gr.Column(visible=True) as page_check:
        with gr.Row(equal_height=False):
            with gr.Column(scale=5):
                # filepath + image_mode=None: Gradio hands over the uploaded file unchanged (with its metadata).
                photo = gr.Image(type="filepath", image_mode=None, label="Photo", sources=["upload", "clipboard"],
                                 height=360)
                button = gr.Button("Check this photo", variant="primary", size="lg")
                if EXAMPLES:
                    gr.Examples(examples=EXAMPLES, inputs=photo, label="Or try a sample")
            with gr.Column(scale=7):
                verdict = gr.HTML(ui.placeholder_card())
                with gr.Tabs():
                    with gr.Tab("Evidence"):
                        evidence = gr.Image(type="pil", show_label=False, height=440, interactive=False)
                    with gr.Tab("Original"):
                        original = gr.Image(type="pil", show_label=False, height=440, interactive=False)
                checks = gr.HTML()
                faces = gr.HTML()
                explanation = gr.HTML()
        button.click(check, inputs=photo, outputs=[verdict, evidence, original, checks, faces, explanation])

    with gr.Column(visible=False) as page_fool:
        gr.HTML(ui.FOOL_INTRO)
        with gr.Row():
            prompt = gr.Textbox(label="Prompt", placeholder="e.g. a selfie of a young woman at a train station", scale=4)
            make = gr.Button("Make it and check it", variant="primary", scale=1)
            cut = gr.Button("Crop, re-save, check again", scale=1)
        made = gr.State(None)
        with gr.Row(equal_height=False):
            with gr.Column(scale=7):
                fake_map = gr.Image(type="pil", label="Muse Image picture with the evidence map", height=460,
                                    interactive=False)
            with gr.Column(scale=5):
                fake_verdict = gr.HTML(ui.placeholder_card("No picture yet."))
                fake_checks = gr.HTML()
        make.click(fool, inputs=prompt, outputs=[fake_map, fake_verdict, fake_checks, made])
        cut.click(crop, inputs=made, outputs=[fake_map, fake_verdict, fake_checks])

    with gr.Column(visible=False) as page_about:
        gr.HTML(ui.ABOUT)

    gr.HTML(ui.FOOTER)
    pages = [page_check, page_fool, page_about]
    for page, entry in zip(PAGES, menu):
        entry.click(partial(show, page), outputs=pages + menu, queue=False, show_progress="hidden")
        entry.click(None, js=ui.CLOSE_MENU_ON_PHONE_JS)
    theme.click(None, js=ui.TOGGLE_JS)

if __name__ == "__main__":
    demo.launch(theme=ui.THEME, css=ui.CSS, js=ui.JS)
