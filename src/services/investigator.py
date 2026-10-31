"""Muse Spark investigator (Meta Model API): after our verdict, name what an AI editor most likely changed;
SAM 3.1 outlines it.

Measured on held-out Muse test edits (results/investigator_v1.txt): as a judge Muse Spark is weak (AUC 0.61-0.69), so
it never decides the verdict (plan Section 0); but it says "AI-edited" for only 1% of real photos, and when it names a
changed object on a local edit, SAM 3.1's outline of that text lands on the truly changed area 96% of the time. Our
pixel checks can't localise Muse edits (results/path_b_v1.txt), so this adds the "where".

Runs only when the app asks (after our verdict) and returns a guess only if Muse Spark says the photo looks AI-edited
and names something. Standard mode (Meta doesn't train on it); a smaller copy is sent (long side 1024).
About $0.004 (Muse Spark) + $0.0025 (SAM 3.1); through src/services/meta_api.py (daily cap, cache, fallback).
"""

import base64
import hashlib
import io
import json
import re

import numpy as np
from PIL import Image

from src.services import meta_api
from src.services.sam_outline import output_text, parse_mask

MODEL = "muse-spark-1.3"
PRICE_IN, PRICE_OUT = 1.25 / 1e6, 4.25 / 1e6  # $ per token (standard mode)
SAM_COST = 0.0025
MAX_COST = 0.02  # a generous upper bound for one investigation, for the daily cap
SEND_LONG_SIDE = 1024
PROMPT = ("Look at this photo carefully. It may be an unedited camera photo, or a real photo that an AI image editor "
          "changed (for example by adding, removing or replacing an object, or by re-rendering the whole scene). "
          'Answer with JSON only, no other text: {"ai_edited": "yes" or "no", "confidence": a number from 0 to 100, '
          '"changed": "the single object or area most likely added or changed by AI, in 1 to 4 words, or none"}')


def investigate(image):
    """image: the upright PIL RGB photo. Returns {"guess", "confidence", "mask" (image size) or None} if Muse Spark
    names a likely changed object; {"guess": None} if it sees no edit; None if the API is unavailable or capped."""
    small = image.copy()
    small.thumbnail((SEND_LONG_SIDE, SEND_LONG_SIDE))
    buffer = io.BytesIO()
    small.save(buffer, format="JPEG", quality=90)
    data = base64.b64encode(buffer.getvalue()).decode()

    def call(key):
        body = {"model": MODEL, "reasoning_effort": "low", "messages": [{"role": "user", "content": [
            {"type": "text", "text": PROMPT}, {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{data}"}}]}]}
        reply = meta_api.post(key, "/chat/completions", body)
        usage = reply.get("usage", {})
        cost = usage.get("prompt_tokens", 0) * PRICE_IN + usage.get("completion_tokens", 0) * PRICE_OUT
        match = re.search(r"\{.*\}", reply["choices"][0]["message"]["content"], re.S)
        answer = json.loads(match.group(0)) if match else {}
        guess = str(answer.get("changed", "")).strip()
        if str(answer.get("ai_edited", "")).lower() != "yes" or guess.lower() in ("", "none"):
            return {"guess": None}, cost  # it sees no edit, or can't say what: show nothing
        body = {"model": "sam-3.1", "input": [{"type": "message", "role": "user", "content": [
            {"type": "input_text", "text": guess}, {"type": "input_image", "image_url": "data:image/jpeg;base64," + data}]}]}
        found = parse_mask(output_text(meta_api.post(key, "/responses", body)), small.size)
        mask = None
        if found is not None:  # back to the photo's own size, so the app can draw it on the full image
            mask = np.asarray(Image.fromarray(found["mask"].astype(np.uint8) * 255).resize(image.size, Image.NEAREST)) > 0
        return {"guess": guess, "confidence": answer.get("confidence"), "mask": mask}, cost + SAM_COST

    return meta_api.paid_call((MODEL, "investigate", hashlib.sha256(data.encode()).hexdigest()), MAX_COST, call)
