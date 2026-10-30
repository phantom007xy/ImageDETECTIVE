"""ImageDETECTIVE, thin preview: upload a photo -> verdict, AI probability, evidence map.

Run from the repo root:  python -m app.app   (then open http://127.0.0.1:7860)
"""

import gradio as gr
from PIL import ImageOps

from src.inference import Detector, evidence_overlay

# Provisional (31 Oct): head E = SID-Set + Muse pairs + Commons real photos and pairs + SD 1.5/2.1 faces
# (results/generator_variety_v1.txt). Before: head C (weights/muse_pairs_v1/head_C.pt), and before that
# weights/head_path_a_sidset30.pt.
HEAD_PATH = "weights/commons_v1/head_E.pt"
detector = Detector(HEAD_PATH)


def check(image):
    if image is None:
        return "Upload a photo first.", None
    image = ImageOps.exif_transpose(image)  # upright, as the phone shows it
    result = detector.analyze(image)
    p = result["probability_ai"]
    confidence = {"Likely AI-generated": f"**Confidence:** {p:.0%} AI",
                  "Likely authentic": f"**Confidence:** {1 - p:.0%} real"}.get(result["verdict"], f"**AI probability:** {p:.0%}")
    lines = [f"## {result['verdict']}", confidence]
    if result["verdict"] == "Likely authentic":
        lines.append("*This photo doesn't look AI-made or AI-changed. It doesn't confirm when or where it was taken.*")
    if detector.calibrated:
        status = ("Thresholds calibrated on development photos: at most 5% of real dev photos are called AI, "
                  "and at most 15% of a real/fake mix is Uncertain.")
    else:
        status = "**Preview:** thresholds are not calibrated yet."
    lines += ["", f"Checked in {result['seconds']:.1f} s. {status} Only the whole image is checked so far "
              "(no faces, no edited regions)."]
    return "\n".join(lines), evidence_overlay(image, result["patch_logits"])


# analytics_enabled=False: Gradio sends no usage statistics (photos never leave this machine anyway).
with gr.Blocks(title="ImageDETECTIVE", analytics_enabled=False) as demo:
    gr.Markdown("# ImageDETECTIVE (thin preview)\nUpload a photo to check whether it looks real or AI-generated.")
    with gr.Row():
        photo = gr.Image(type="pil", label="Photo")
        evidence = gr.Image(type="pil", label="Evidence map (red = points to AI, blue = points to real)")
    result_text = gr.Markdown()
    gr.Button("Check", variant="primary").click(check, inputs=photo, outputs=[result_text, evidence])

if __name__ == "__main__":
    demo.launch()
