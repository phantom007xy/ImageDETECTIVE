"""Muse Spark explanation (Meta Model API): 2-3 plain sentences written from the numbers only.

Muse Spark gets the facts our detector produced (verdict, confidence, faces, metadata, the SAM 3.1
outline), never the photo itself, and is told not to judge the photo on its own: the verdict always
comes from our detector (plan Section 5.7). Standard mode (muse-spark-1.3, not "-contributor"), so
Meta does not train on it. About $0.001-0.002 per explanation; it goes through
src/services/meta_api.py (daily cap, cache, fallback: without it the app shows everything else).
"""

import json

from src.services import meta_api

MODEL = "muse-spark-1.3"
PRICE_IN, PRICE_OUT = 1.25 / 1e6, 4.25 / 1e6  # $ per token (standard mode)
MAX_COST = 0.01  # a generous upper bound for one explanation, for the daily cap
INSTRUCTION = (
    "You explain the result of an image authenticity checker to a non-expert, e.g. someone who received "
    "this photo in a WhatsApp group. Write 2 or 3 short, plain sentences. Use ONLY the facts below. "
    "Do not judge the photo yourself, do not add certainty beyond the given confidence, and do not "
    "contradict the verdict. If the verdict is 'Uncertain', say clearly that the checker could not decide. "
    "If it is 'Likely authentic', mention that this does not confirm when or where the photo was taken. "
    "No lists, no technical words like logit or patch.\n\nFacts:\n"
)


def facts(result, outline=None, evidence=None):
    """The detector's result as a small, plain dictionary (no pixels, no file names)."""
    faces = [f for f in result["faces"] if not f["too_small"]]
    out = {
        "verdict": result["verdict"],
        "verdict_based_on": "the file's own metadata label" if result["verdict_reason"] == "metadata" else "the pixels",
        "ai_probability_from_pixels": round(result["probability_ai"], 2),
        "faces_checked": len(faces),
        "faces_that_look_ai_made": sum(f["level"] == "ai" for f in faces),
        "faces_that_lean_ai": sum(f["level"] == "leans_ai" for f in faces),
        "faces_too_small_to_check": sum(f["too_small"] for f in result["faces"]),
    }
    if result["metadata"] and result["metadata"]["ai_label"]:
        out["file_metadata"] = f"the file says it was {'made' if result['metadata']['ai_label'] == 'generated' else 'partly changed'} by AI"
    if outline and outline.get("mask") is not None and evidence:
        out["strongest_evidence"] = (f"one object was outlined; its AI evidence is about {evidence[0]:.0%}, "
                                     f"the rest of the photo about {evidence[1]:.0%}")
    return out


def explain(result, outline=None, evidence=None):
    """2-3 sentences, or None (cap reached, no key, API error)."""
    text = INSTRUCTION + json.dumps(facts(result, outline, evidence), indent=1)

    def call(key):
        body = {"model": MODEL, "reasoning_effort": "minimal", "messages": [{"role": "user", "content": text}]}
        response = meta_api.post(key, "/chat/completions", body)
        usage = response.get("usage", {})
        cost = usage.get("prompt_tokens", 0) * PRICE_IN + usage.get("completion_tokens", 0) * PRICE_OUT
        return " ".join(response["choices"][0]["message"]["content"].split()), cost

    return meta_api.paid_call((MODEL, text), MAX_COST, call)
