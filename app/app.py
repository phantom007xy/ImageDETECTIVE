"""ImageDETECTIVE, thin preview: upload a photo -> verdict, confidence, evidence map, face boxes.

Run from the repo root:  python -m app.app   (then open http://127.0.0.1:7860)
"""

import gradio as gr
from PIL import ImageOps

from src.inference import Detector, draw_faces, evidence_overlay

# Provisional (31 Oct): head F = SID-Set + Muse pairs + Commons real photos and pairs + SD 1.5/2.1 faces
# + SD 3.5 Medium and Kandinsky 2.2 fakes (results/open_gen_v1_heads.txt, calibration_F_v1.txt).
# Before: head E (weights/commons_v1/head_E.pt), C (weights/muse_pairs_v1/head_C.pt), weights/head_path_a_sidset30.pt.
HEAD_PATH = "weights/open_gen_v1/head_F.pt"
detector = Detector(HEAD_PATH)


def face_lines(faces):
    """One line per face, numbered like the boxes in the evidence map."""
    if not faces:
        return ["**Faces:** none found."]
    lines = [f"**Faces:** {len(faces)} found (box numbers in the evidence map)"]
    for n, face in enumerate(faces, 1):
        if face["too_small"]:
            lines.append(f"- Face {n}: too small to check")
        else:
            label = {"ai": "🔴 looks AI-made or AI-changed", "leans_ai": "🟠 leans AI-made (not enough on its own)",
                     "real": "🟢 looks real"}[face["level"]]
            lines.append(f"- Face {n}: {label} (AI probability {face['probability_ai']:.0%})")
    return lines


def check(image):
    if image is None:
        return "Upload a photo first.", None
    image = ImageOps.exif_transpose(image).convert("RGB")  # upright, as the phone shows it
    result = detector.analyze(image)
    p = result["probability_ai"]
    confidence = {"Likely AI-generated": f"**Confidence:** {p:.0%} AI",
                  "Likely authentic": f"**Confidence:** {1 - p:.0%} real"}.get(result["verdict"],
                                                                               f"**Whole-image AI probability:** {p:.0%}")
    lines = [f"## {result['verdict']}", confidence]
    if result["verdict"] == "Likely authentic":
        lines.append("*This photo doesn't look AI-made or AI-changed. It doesn't confirm when or where it was taken.*")
    lines += [""] + face_lines(result["faces"])
    if detector.calibrated:
        status = ("Thresholds calibrated on development photos: at most 5% of real dev photos are called AI, "
                  "and at most 15% of a real/fake mix is Uncertain.")
    else:
        status = "**Preview:** thresholds are not calibrated yet."
    lines += ["", f"Checked in {result['seconds']:.1f} s. {status} Edited regions are not checked yet."]
    evidence = draw_faces(evidence_overlay(image, result["patch_logits"]), result["faces"])
    return "\n".join(lines), evidence


# analytics_enabled=False: Gradio sends no usage statistics (photos never leave this machine anyway).
with gr.Blocks(title="ImageDETECTIVE", analytics_enabled=False) as demo:
    gr.Markdown("# ImageDETECTIVE (thin preview)\nUpload a photo to check whether it looks real or AI-generated.")
    with gr.Row():
        photo = gr.Image(type="pil", label="Photo")
        evidence = gr.Image(type="pil", label="Evidence map (red = points to AI, blue = points to real) and faces "
                                              "(box: red = looks AI-made, orange = leans AI, green = looks real, "
                                              "grey = too small)")
    result_text = gr.Markdown()
    gr.Button("Check", variant="primary").click(check, inputs=photo, outputs=[result_text, evidence])

if __name__ == "__main__":
    demo.launch()
