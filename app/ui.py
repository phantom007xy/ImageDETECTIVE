"""The look of the ImageDETECTIVE app: a dark, "investigative" theme and the HTML pieces of the result page.

Only presentation lives here (no models): app/app.py calls these functions with the detector's result.
Wording follows the careful verdicts decided on 31 Oct (CLAUDE.local.md, UI decisions).
"""

import html

import gradio as gr

from src.services import metadata

# Colours (one place, so the whole page stays consistent)
BG, PANEL, BORDER, TEXT, MUTED, ACCENT = "#0b0f14", "#111823", "#1f2a37", "#e6edf3", "#8b98a5", "#22d3ee"
VERDICT_COLOUR = {"red": "#ef4444", "amber": "#f59e0b", "orange": "#f97316", "green": "#22c55e", "slate": "#94a3b8"}
VERDICT_STYLE = {
    "AI-generated": "red", "Likely AI-generated": "red",
    "Likely AI-edited": "amber", "Part of this photo looks AI-made or AI-changed": "amber",
    "This face looks AI-made or AI-changed": "orange",
    "Likely authentic": "green", "Uncertain": "slate",
}

# The same dark palette for light and dark mode, so the app always looks the same.
_dark = {"body_background_fill": BG, "body_text_color": TEXT, "body_text_color_subdued": MUTED,
         "background_fill_primary": PANEL, "background_fill_secondary": BG, "block_background_fill": PANEL,
         "block_border_color": BORDER, "border_color_primary": BORDER, "block_label_background_fill": PANEL,
         "block_label_text_color": MUTED, "block_title_text_color": TEXT, "input_background_fill": BG,
         "button_primary_background_fill": ACCENT, "button_primary_background_fill_hover": "#67e8f9",
         "button_primary_text_color": "#04121a", "button_secondary_background_fill": "#1a2330",
         "button_secondary_text_color": TEXT, "checkbox_background_color": BG}
THEME = gr.themes.Base(
    primary_hue=gr.themes.colors.cyan, neutral_hue=gr.themes.colors.slate,
    font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"],
    font_mono=[gr.themes.GoogleFont("JetBrains Mono"), "ui-monospace", "monospace"],
).set(**_dark, **{k + "_dark": v for k, v in _dark.items()}, color_accent=ACCENT)

CSS = f"""
.gradio-container {{ max-width: 1240px !important; margin: 0 auto !important; }}
footer {{ display: none !important; }}
.hero {{ padding: 22px 4px 8px; }}
.hero h1 {{ font-size: 34px; font-weight: 800; letter-spacing: -0.02em; margin: 0; color: {TEXT}; }}
.hero h1 span {{ color: {ACCENT}; }}
.hero p {{ color: {MUTED}; font-size: 16px; margin: 6px 0 12px; }}
.chips span {{ display: inline-block; border: 1px solid {BORDER}; border-radius: 999px; padding: 3px 11px;
  margin: 0 6px 6px 0; font-size: 12px; color: #9fb3c8; background: {PANEL}; }}
.card {{ border-radius: 16px; padding: 20px 22px; border: 1px solid {BORDER}; background: {PANEL}; }}
.verdict .label {{ font-size: 30px; font-weight: 800; letter-spacing: -0.01em; line-height: 1.15; }}
.verdict .reason {{ color: #c9d4df; margin-top: 8px; font-size: 15px; line-height: 1.45; }}
.verdict .context {{ color: {MUTED}; margin-top: 10px; font-size: 13px; font-style: italic; }}
.meter {{ height: 10px; background: {BORDER}; border-radius: 999px; overflow: hidden; margin-top: 16px; }}
.meter > div {{ height: 100%; border-radius: 999px; }}
.meter-label {{ display: flex; justify-content: space-between; color: {MUTED}; font-size: 12px; margin-top: 6px;
  font-family: 'JetBrains Mono', monospace; }}
.steps {{ list-style: none; padding: 0; margin: 0; display: grid; gap: 7px; }}
.steps li {{ display: flex; align-items: center; gap: 10px; padding: 8px 12px; border-radius: 10px;
  background: {BG}; border: 1px solid {BORDER}; font-size: 14px; color: {TEXT}; }}
.steps li small {{ color: {MUTED}; margin-left: auto; font-size: 12px; }}
.steps .dot {{ width: 10px; height: 10px; border-radius: 50%; background: #334155; flex: none; }}
.steps li.done .dot {{ background: #22c55e; }}
.steps li.active .dot {{ background: {ACCENT}; animation: pulse 0.9s ease-in-out infinite; }}
.steps li.off {{ opacity: 0.45; }}
.steps li.warn .dot {{ background: #f59e0b; }}
@keyframes pulse {{ 0%, 100% {{ opacity: 1; }} 50% {{ opacity: 0.25; }} }}
.section-title {{ font-size: 12px; font-weight: 700; letter-spacing: 0.08em; text-transform: uppercase;
  color: {MUTED}; margin: 0 0 10px; }}
.checks {{ width: 100%; border-collapse: collapse; font-size: 14px; }}
.checks td {{ padding: 9px 8px; border-bottom: 1px solid {BORDER}; vertical-align: top; }}
.checks tr:last-child td {{ border-bottom: none; }}
.checks td.k {{ color: {MUTED}; width: 42%; }}
.checks td.k b {{ color: {TEXT}; font-weight: 600; }}
.tag {{ display: inline-block; padding: 1px 9px; border-radius: 999px; font-size: 12px; font-weight: 600;
  margin-right: 6px; }}
.faces li {{ margin: 4px 0; }}
.faces {{ padding-left: 18px; margin: 0; font-size: 14px; }}
.note {{ color: {MUTED}; font-size: 12.5px; margin-top: 10px; line-height: 1.45; }}
.legend {{ display: flex; flex-wrap: wrap; gap: 14px; color: {MUTED}; font-size: 12.5px; padding: 6px 2px 0; }}
.legend i {{ display: inline-block; width: 11px; height: 11px; border-radius: 3px; margin-right: 6px;
  vertical-align: -1px; }}
.explain {{ font-size: 15px; line-height: 1.55; color: #d7e0e9; }}
.footer-strip {{ color: {MUTED}; font-size: 12.5px; border-top: 1px solid {BORDER}; padding: 14px 4px; margin-top: 8px;
  line-height: 1.6; }}
"""

HERO = """
<div class="hero">
  <h1>Image<span>DETECTIVE</span></h1>
  <p>Is this photo real, made by AI, or changed by AI? And <b>where</b> is the evidence?</p>
  <div class="chips"><span>DINOv3 · whole image</span><span>Perception Encoder · AI edits</span>
    <span>SAM 3 · faces</span><span>Muse Spark + SAM 3.1 · investigator</span><span>Muse Image · try to fool it</span></div>
</div>"""

LEGEND = """<div class="legend"><span><i style="background:#ef4444"></i>points to AI</span>
<span><i style="background:#3b82f6"></i>points to a real photo</span><span><i style="background:#facc15"></i>outline (SAM 3.1)</span>
<span><i style="background:#dc2626"></i>face looks AI-made</span><span><i style="background:#f59e0b"></i>face: some AI signs</span>
<span><i style="background:#16a34a"></i>face: no clear AI signs</span><span><i style="background:#94a3b8"></i>face too small</span></div>"""

FOOTER = """<div class="footer-strip">A result is evidence, not proof. ImageDETECTIVE checks whether the pixels look
AI-made or AI-edited; it can't tell when or where a photo was taken, and it doesn't check classic manual edits
(e.g. old-style Photoshop cut-and-paste). Face swaps and small pasted-in AI edits are only partly covered.<br>
Built with DINOv3 · Perception Encoder · SAM 3 and SAM 3.1 · Muse Image · Muse Spark (Meta).</div>"""

STEPS = [("pixels", "Whole image", "DINOv3"), ("edit", "Edit check", "Perception Encoder"),
         ("faces", "Faces", "SAM 3 + DINOv3"), ("label", "File label", "metadata"),
         ("investigate", "Investigator", "Muse Spark + SAM 3.1"), ("explain", "Explanation", "Muse Spark")]


def steps_html(state):
    """state: {step key: "done" | "active" | "off" | "warn"}; missing keys are waiting."""
    items = "".join(f'<li class="{state.get(k, "")}"><span class="dot"></span>{name}<small>{model}</small></li>'
                    for k, name, model in STEPS)
    return f'<div class="section-title">Steps</div><ul class="steps">{items}</ul>'


def placeholder_card(text="Upload a photo, or pick a sample, and press Check."):
    return f'<div class="card verdict"><div class="label" style="color:{MUTED}">No photo yet</div>' \
           f'<div class="reason">{html.escape(text)}</div></div>'


def verdict_card(result):
    v = result["verdict"]
    colour = VERDICT_COLOUR[VERDICT_STYLE.get(v, "slate")]
    p = result["probability_ai"]
    if result["verdict_reason"] == "metadata":
        reason = (f"{metadata.describe(result['metadata'])} From the pixels alone: {result['pixel_verdict']} "
                  f"({p:.0%} AI evidence). A forwarded copy loses this label; the pixel checks still work then.")
    elif v == "AI-generated":
        reason = f"The whole image looks AI-made ({p:.0%} AI evidence); no real test photo scored this high."
    elif v == "Likely AI-generated" and not result.get("whole_ai", True):
        reason = ("Meta's Perception Encoder sees how the pixels were rendered: this picture looks made by an AI image "
                  f"model (a check also trained on Google Gemini pictures), although the whole-image check alone says {p:.0%}.")
    elif v == "Likely AI-generated":
        reason = f"The whole image looks AI-made ({p:.0%} AI evidence)."
    elif v == "Likely AI-edited":
        reason = ("Meta's Perception Encoder, fine-tuned on Muse edits, sees the traces an AI image model leaves when it "
                  "re-renders a photo; a fully AI-made picture can show them too. Very few real test photos reach this level.")
    elif v == "This face looks AI-made or AI-changed":
        reason = "At least one face looks AI-made or swapped, although the photo as a whole does not (see Faces)."
    elif v == "Likely authentic":
        reason = f"No clear signs of AI in the pixels ({1 - p:.0%} real)."
    else:
        reason = "The checks don't point clearly either way, so ImageDETECTIVE doesn't decide."
    context = ('<div class="context">This photo doesn\'t look AI-made or AI-changed. It doesn\'t confirm when or '
               'where it was taken.</div>' if v == "Likely authentic" else "")
    return (f'<div class="card verdict" style="border-color:{colour}55;background:linear-gradient(135deg,{colour}14,'
            f'{PANEL} 55%)"><div class="label" style="color:{colour}">{html.escape(v)}</div>'
            f'<div class="reason">{html.escape(reason)}</div>{context}'
            f'<div class="meter"><div style="width:{p * 100:.0f}%;background:{colour}"></div></div>'
            f'<div class="meter-label"><span>whole-image AI evidence</span><span>{p:.0%}</span></div></div>')


def _tag(text, colour):
    return f'<span class="tag" style="background:{colour}22;color:{colour}">{html.escape(text)}</span>'


def checks_html(result, investigation=None):
    """The 'what we checked' table. investigation: None (not run), "" (nothing named) or Muse Spark's guess."""
    p, edit, meta = result["probability_ai"], result["edit"], result["metadata"]
    checked = [f for f in result["faces"] if not f["too_small"]]
    flagged = [f for f in checked if f.get("flagged")]
    rows = [("Whole image", "DINOv3", _tag(f"{p:.0%} AI evidence", VERDICT_COLOUR["red" if p > 0.5 else "green"]))]
    if edit is None:
        rows.append(("Edit check", "Perception Encoder", _tag("off", MUTED)))
    else:
        rows.append(("Edit check", "Perception Encoder",
                     _tag("looks re-rendered by an AI editor", VERDICT_COLOUR["amber"]) if edit["flagged"]
                     else _tag("looks rendered by an AI image model", VERDICT_COLOUR["red"]) if edit.get("flagged2")
                     else _tag("no clear signs of an AI edit", VERDICT_COLOUR["green"])))
    faces_text = (_tag(f"{len(flagged)} of {len(checked)} look AI-made or swapped", VERDICT_COLOUR["orange"]) if flagged
                  else _tag(f"{len(checked)} checked, none flagged", VERDICT_COLOUR["green"]) if checked
                  else _tag("no face big enough to check", MUTED))
    rows.append(("Faces", "SAM 3 + DINOv3", faces_text))
    label = metadata.describe(meta) if meta else ""
    rows.append(("File label", "metadata", _tag(label, VERDICT_COLOUR["amber"]) if label and meta.get("ai_label")
                 else _tag("no AI label (forwarded copies lose it, so this proves nothing)", MUTED)))
    if investigation is not None:
        rows.append(("Investigator", "Muse Spark + SAM 3.1",
                     _tag(f"guess of what was changed: {investigation}", "#facc15") if investigation
                     else _tag("named nothing", MUTED)))
    body = "".join(f'<tr><td class="k"><b>{a}</b> · {b}</td><td>{c}</td></tr>' for a, b, c in rows)
    return f'<div class="card"><div class="section-title">What we checked</div><table class="checks">{body}</table></div>'


def faces_html(faces):
    if not faces:
        return ""
    names = {"ai": ("looks AI-made or AI-changed", "#dc2626"), "leans_ai": ("some AI signs (not enough on its own)", "#f59e0b"),
             "clear": ("no clear AI signs", "#16a34a")}
    items = []
    for n, face in enumerate(faces, 1):
        if face["too_small"]:
            items.append(f'<li><span class="tag" style="background:#94a3b822;color:#94a3b8">{n}</span>too small to check</li>')
            continue
        text, colour = names[face["level"]]
        if face.get("kind") == "swapped":
            text += " (face-swap check)"
        p = face["probability_ai"]
        items.append(f'<li>{_tag(str(n), colour)}{text}' + (f' <span class="note">face AI probability {p:.0%}</span>'
                                                           if p is not None else "") + "</li>")
    return (f'<div class="card"><div class="section-title">Faces (numbers match the boxes)</div>'
            f'<ul class="faces">{"".join(items)}</ul><div class="note">The face checks are weaker than the whole-image '
            f'check: "no clear AI signs" is not proof that a face is real.</div></div>')


def explanation_html(text=None, waiting=False):
    if waiting:
        return '<div class="card"><div class="section-title">In plain words · Muse Spark</div><div class="note">Writing a short explanation …</div></div>'
    if not text:
        return ""
    return (f'<div class="card"><div class="section-title">In plain words · Muse Spark</div>'
            f'<div class="explain">{html.escape(text)}</div><div class="note">Written from our detectors\' numbers; '
            f'Muse Spark doesn\'t decide the verdict.</div></div>')


def footer_note(seconds, calibrated):
    status = ("Thresholds calibrated on development photos: at most 5% of real test photos are called AI."
              if calibrated else "Preview: thresholds are not calibrated yet.")
    return f'<div class="note">Checked in {seconds:.1f} s. {status}</div>'
