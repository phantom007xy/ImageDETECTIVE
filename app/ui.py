"""The look of the ImageDETECTIVE app: a plain, tool-like page (black by default, light mode on a switch) and the HTML
pieces of the result page.

Only presentation lives here (no models): app/app.py calls these functions with the detector's result.
Wording follows the careful verdicts decided on 31 Oct (CLAUDE.local.md, UI decisions). Colours only carry meaning
(the verdict); everything else is black, white and grey. The HTML pieces use Gradio's theme variables
(--body-text-color, --border-color-primary, ...), so they follow the light/dark switch by themselves.
"""

import html

import gradio as gr

from src.services import metadata

# Verdict colours: readable on black and on white
VERDICT_COLOUR = {"red": "#ef4444", "amber": "#d97706", "orange": "#ea580c", "green": "#16a34a", "slate": "#64748b"}
VERDICT_STYLE = {
    "AI-generated": "red", "Likely AI-generated": "red",
    "Likely AI-edited": "amber", "Part of this photo looks AI-made or AI-changed": "amber",
    "This face looks AI-made or AI-changed": "orange",
    "Likely authentic": "green", "Uncertain": "slate",
}
MUTED = "#8b8b8b"

LIGHT = {"bg": "#ffffff", "panel": "#f4f4f5", "border": "#d4d4d8", "text": "#111111", "muted": "#5b5b63",
         "button": "#111111", "button_hover": "#3f3f46", "button_text": "#ffffff"}
DARK = {"bg": "#000000", "panel": "#0b0b0b", "border": "#262626", "text": "#e5e5e5", "muted": "#8b8b8b",
        "button": "#e5e5e5", "button_hover": "#a3a3a3", "button_text": "#000000"}


def _palette(c):
    return {"body_background_fill": c["bg"], "body_text_color": c["text"], "body_text_color_subdued": c["muted"],
            "background_fill_primary": c["bg"], "background_fill_secondary": c["panel"],
            "block_background_fill": c["panel"], "block_border_color": c["border"], "border_color_primary": c["border"],
            "block_label_background_fill": c["panel"], "block_label_text_color": c["muted"],
            "block_title_text_color": c["text"], "input_background_fill": c["bg"],
            "button_primary_background_fill": c["button"], "button_primary_background_fill_hover": c["button_hover"],
            "button_primary_text_color": c["button_text"], "button_primary_border_color": c["button"],
            "button_secondary_background_fill": c["panel"], "button_secondary_background_fill_hover": c["border"],
            "button_secondary_text_color": c["text"], "button_secondary_border_color": c["border"],
            "checkbox_background_color": c["bg"], "color_accent_soft": c["panel"], "link_text_color": c["text"],
            "table_even_background_fill": c["bg"], "table_odd_background_fill": c["panel"]}


_mono = [gr.themes.GoogleFont("JetBrains Mono"), "ui-monospace", "Consolas", "monospace"]
THEME = gr.themes.Base(neutral_hue=gr.themes.colors.neutral, radius_size=gr.themes.sizes.radius_sm,
                       text_size=gr.themes.sizes.text_md, font=_mono, font_mono=_mono).set(
    **_palette(LIGHT), **{k + "_dark": v for k, v in _palette(DARK).items()}, color_accent="#737373")

# Black unless the link asks for light (?__theme=light) or the visitor switched to light (remembered in this browser).
# On a phone the menu covers the whole screen, so there it starts closed (its arrow at the top left opens it).
JS = """() => {
  let saved = null;
  try { saved = localStorage.getItem('idetective-theme'); } catch (e) {}
  const asked = new URLSearchParams(window.location.search).get('__theme');
  document.body.classList.toggle('dark', (asked || saved || 'dark') !== 'light');
  if (window.innerWidth <= 768) {
    let tries = 0;
    const close = () => {
      const arrow = document.querySelector('.sidebar.open .toggle-button');
      if (arrow) arrow.click(); else if (tries++ < 40) setTimeout(close, 150);
    };
    close();
  }
}"""
TOGGLE_JS = """() => {
  const dark = document.body.classList.toggle('dark');
  try { localStorage.setItem('idetective-theme', dark ? 'dark' : 'light'); } catch (e) {}
}"""
# On a phone: close the menu after a page is chosen, so the page is visible.
CLOSE_MENU_ON_PHONE_JS = """() => {
  if (window.innerWidth <= 768) { const arrow = document.querySelector('.sidebar.open .toggle-button'); if (arrow) arrow.click(); }
}"""

CSS = """
footer { display: none !important; }
.sidebar-parent, .sidebar { transition: none !important; }  /* the menu opens and closes without sliding */
.idt-title { font-size: 18px; font-weight: 700; letter-spacing: 0.02em; padding: 2px 2px 10px;
  border-bottom: 1px solid var(--border-color-primary); }
.nav-label { font-size: 11px; letter-spacing: 0.1em; text-transform: uppercase; color: var(--body-text-color-subdued);
  padding-top: 10px; }
button.nav { justify-content: flex-start !important; text-align: left !important; }
#theme-toggle { min-width: 36px !important; max-width: 40px !important; padding: 2px 6px !important; }
.card { border: 1px solid var(--border-color-primary); background: var(--block-background-fill); border-radius: 4px;
  padding: 14px 16px; }
.verdict .label { font-size: 22px; font-weight: 700; line-height: 1.2; }
.verdict .reason { margin-top: 6px; font-size: 13.5px; line-height: 1.5; color: var(--body-text-color); }
.verdict .context { margin-top: 8px; font-size: 12.5px; color: var(--body-text-color-subdued); }
.meter { height: 6px; background: var(--border-color-primary); margin-top: 12px; }
.meter > div { height: 100%; }
.meter-label { display: flex; justify-content: space-between; color: var(--body-text-color-subdued); font-size: 12px;
  margin-top: 4px; }
.section-title { font-size: 11px; font-weight: 700; letter-spacing: 0.1em; text-transform: uppercase;
  color: var(--body-text-color-subdued); margin: 0 0 8px; }
.checks { width: 100%; border-collapse: collapse; font-size: 13px; }
.checks td { padding: 7px 6px; border-bottom: 1px solid var(--border-color-primary); vertical-align: top; }
.checks tr:last-child td { border-bottom: none; }
.checks td.k { color: var(--body-text-color-subdued); width: 40%; }
.checks td.k b { color: var(--body-text-color); font-weight: 600; }
.tag { display: inline-block; padding: 0 6px; border: 1px solid; border-radius: 2px; font-size: 12px; font-weight: 600;
  margin-right: 6px; }
.faces { padding-left: 18px; margin: 0; font-size: 13px; }
.faces li { margin: 3px 0; }
.note { color: var(--body-text-color-subdued); font-size: 12px; margin-top: 8px; line-height: 1.5; }
.explain { font-size: 13.5px; line-height: 1.6; }
.about { max-width: 820px; }
.about h3 { font-size: 12px; letter-spacing: 0.1em; text-transform: uppercase; color: var(--body-text-color-subdued);
  margin: 18px 0 6px; }
.about p, .about li { font-size: 13.5px; line-height: 1.6; margin: 0 0 4px; }
.about ul { padding-left: 18px; margin: 0; }
.footer-strip { color: var(--body-text-color-subdued); font-size: 12px; border-top: 1px solid var(--border-color-primary);
  padding: 10px 2px; margin-top: 10px; line-height: 1.6; }
"""

TITLE = '<div class="idt-title">ImageDETECTIVE</div>'

FOOTER = """<div class="footer-strip">A result is evidence, not proof. Limits and privacy: see About.
Built with DINOv3 · Perception Encoder · SAM 3 and SAM 3.1 · Muse Image · Muse Spark (Meta).</div>"""

FOOL_INTRO = ('<div class="note" style="font-size:13px;margin:0 0 6px">Type a prompt: Meta\'s Muse Image makes a picture '
              'and ImageDETECTIVE checks it right away. Then crop and re-save it, like a forwarded copy, and check again. '
              'Text prompts only: this app never edits uploaded photos, and it doesn\'t make pictures of real or famous '
              'people.</div>')

ABOUT = """<div class="about">
<h3>What it does</h3>
<p>ImageDETECTIVE checks one photo: does it look real, made by AI, or changed by AI? It shows where the evidence is:
an evidence map over the image, a box and a result for every face, and an outline of the most suspicious object.</p>
<h3>What it checks</h3>
<ul>
<li><b>Whole image</b> · DINOv3 (Meta) with a calibrated linear head. The evidence map shows which parts of the image
pushed the score: red points to AI, blue to a real photo.</li>
<li><b>AI edits</b> · Perception Encoder (Meta), fine-tuned on Muse Image edits. It also notices pictures rendered by
other AI image models.</li>
<li><b>Faces</b> · SAM 3 (Meta) finds every face; each face big enough is checked on its own (AI-made faces, face
swaps). Box colours: red = looks AI-made or swapped, orange = some AI signs, green = no clear AI signs, grey = too
small to check.</li>
<li><b>File label</b> · AI tags in the file's metadata. Forwarded copies lose them, so a missing label proves nothing.</li>
<li><b>Investigator</b> · Muse Spark names what was most likely changed and SAM 3.1 outlines it (yellow outline);
Muse Spark writes a short explanation from the numbers. Neither decides the verdict.</li>
<li><b>Try to fool it</b> · Muse Image makes a picture from your prompt, and the detector checks it.</li>
</ul>
<h3>How the verdict is set</h3>
<p>The thresholds are set on development photos so that at most 5% of real photos get an AI verdict. When the checks
don't point clearly either way, the answer is "Uncertain" instead of a guess.</p>
<h3>Limits</h3>
<ul>
<li>A result is evidence, not proof. "No clear AI signs" is not proof that a photo is real.</li>
<li>It can't tell when or where a photo was taken: a real photo forwarded with a false story still looks real.</li>
<li>Classic manual edits (old-style Photoshop cut-and-paste) are not covered; face swaps and small pasted-in AI edits
only partly.</li>
<li>Built for photos: drawings, documents, chat screenshots and videos are out of scope.</li>
</ul>
<h3>Privacy</h3>
<ul>
<li>Uploaded photos are deleted from the server within an hour.</li>
<li>For photos that are not clearly authentic, a smaller copy goes to Meta's API in standard mode (Meta doesn't
train on it), so SAM 3.1 and Muse Spark can outline and name the evidence. The verdict never waits for this step.</li>
<li>Try to fool it takes text prompts only and never edits uploaded photos.</li>
</ul>
<h3>Built with</h3>
<p>DINOv3 · Perception Encoder · SAM 3 · SAM 3.1 · Muse Image · Muse Spark (Meta). Datasets and licences: see the
README on GitHub.</p>
</div>"""


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
        reason = f"The whole image has strong AI evidence ({p:.0%}); this is evidence, not proof."
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
    return (f'<div class="card verdict" style="border-left:3px solid {colour}">'
            f'<div class="label" style="color:{colour}">{html.escape(v)}</div>'
            f'<div class="reason">{html.escape(reason)}</div>{context}'
            f'<div class="meter"><div style="width:{p * 100:.0f}%;background:{colour}"></div></div>'
            f'<div class="meter-label"><span>whole-image AI evidence</span><span>{p:.0%}</span></div></div>')


def _tag(text, colour):
    return f'<span class="tag" style="color:{colour};border-color:{colour}88">{html.escape(text)}</span>'


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
                     _tag(f"guess of what was changed: {investigation}", "#ca8a04") if investigation
                     else _tag("named nothing", MUTED)))
    body = "".join(f'<tr><td class="k"><b>{a}</b> · {b}</td><td>{c}</td></tr>' for a, b, c in rows)
    return f'<div class="card"><div class="section-title">What we checked</div><table class="checks">{body}</table></div>'


def faces_html(faces):
    if not faces:
        return ""
    names = {"ai": ("looks AI-made or AI-changed", "#dc2626"), "leans_ai": ("some AI signs (not enough on its own)", "#d97706"),
             "clear": ("no clear AI signs", "#16a34a")}
    items = []
    for n, face in enumerate(faces, 1):
        if face["too_small"]:
            items.append(f'<li>{_tag(str(n), MUTED)}too small to check</li>')
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
