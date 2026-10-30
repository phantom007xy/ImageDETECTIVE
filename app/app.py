"""ImageDETECTIVE, thin preview: upload a photo -> verdict, AI probability, evidence map.

Run from the repo root:  python -m app.app   (then open http://127.0.0.1:7860)
"""

import gradio as gr
from PIL import ImageOps

from src.inference import Detector, evidence_overlay

# Provisional (31 Oct): head C = SID-Set + Muse Image pairs, results/muse_pairs_v1_heads.txt.
# The previous head was weights/head_path_a_sidset30.pt.
HEAD_PATH = "weights/muse_pairs_v1/head_C.pt"
detector = Detector(HEAD_PATH)


def check(image):
    if image is None:
        return "Upload a photo first.", None
    image = ImageOps.exif_transpose(image)  # upright, as the phone shows it
    result = detector.analyze(image)
    lines = [
        f"## {result['verdict']}",
        f"**AI probability:** {result['probability_ai']:.0%}",
    ]
    if result["verdict"] == "Likely authentic":
        lines.append("*This photo doesn't look AI-made or AI-changed. It doesn't confirm when or where it was taken.*")
    lines += [
        "",
        f"Checked in {result['seconds']:.1f} s. **Preview:** thresholds are not calibrated yet, and only the "
        "whole image is checked (no faces, no edited regions).",
    ]
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
