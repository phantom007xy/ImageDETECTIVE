"""Muse Image "Try to fool it" (Meta Model API): make an image from a visitor's text prompt, then check it.

Safety rule (plan Section 5.7): only text prompts, never edits of uploaded photos, so the app cannot
be used to make fakes of real people; prompts that ask for real or famous people (or lookalikes) are
refused before any call. Meta's own safety filter also refuses some prompts (not billed).
Standard mode (muse-image-1.0), search tools off. $0.01 per image; it goes through
src/services/meta_api.py (daily cap, cache: the same prompt returns the same image without a new cost).
"""

import base64
import io

from PIL import Image

from src.services import meta_api

MODEL = "muse-image-1.0"
COST = 0.01
# Our own rule, before Meta's filter: no pictures of real or famous people, and no lookalikes. A word
# list can't catch every name, but it refuses the obvious requests (and Meta's filter is a second layer).
REAL_PEOPLE_WORDS = ["looks like", "look like", "lookalike", "look-alike", "resembl", "celebrity", "famous",
                     "actor", "actress", "singer", "cricketer", "footballer", "politician", "president",
                     "prime minister", "minister", "influencer", "youtuber", "real person"]


def asks_for_real_people(prompt):
    text = prompt.lower()
    return any(word in text for word in REAL_PEOPLE_WORDS)


def generate(prompt):
    """PNG bytes of a new Muse Image picture (with Muse's own AI label in its metadata), or None."""
    prompt = " ".join(prompt.split())[:500]
    if not prompt or asks_for_real_people(prompt):
        return None

    def call(key):
        body = {"model": MODEL, "prompt": prompt, "n": 1, "response_format": "b64_json", "output_format": "png",
                "tool_enablement": {"enable_image_search": False, "enable_web_search": False, "enable_shell": False},
                "user": "imagedetective-demo"}
        response = meta_api.post(key, "/images/generations", body)
        return base64.b64decode(response["data"][0]["b64_json"]), COST

    return meta_api.paid_call((MODEL, prompt), COST, call)


def crop_and_resave(png_bytes, keep=0.7, quality=85):
    """What often happens to a forwarded fake: cut to the middle part and saved again as JPEG.

    The re-save drops the file's AI label, so only the pixel check can still catch it.
    """
    with Image.open(io.BytesIO(png_bytes)) as im:
        image = im.convert("RGB")
    w, h = image.size
    cw, ch = round(w * keep), round(h * keep)
    left, top = (w - cw) // 2, (h - ch) // 2
    buffer = io.BytesIO()
    image.crop((left, top, left + cw, top + ch)).save(buffer, format="JPEG", quality=quality)
    return buffer.getvalue()
