"""Metadata check: does the file itself say that it was made or changed by AI? (plan Section 5.7)

Generators and editing tools can write an IPTC "digital source type" into the file:
  trainedAlgorithmicMedia              -> made by a trained AI model (e.g. every Muse Image PNG, in XMP)
  compositeWithTrainedAlgorithmicMedia -> a picture partly made or changed by AI
These values sit either in the XMP block or inside a C2PA manifest (a signed "content credentials"
record), so a plain search of the file's bytes finds them in both. The check also notes whether a C2PA
manifest is present at all.

Only evidence TOWARDS AI. Missing metadata proves nothing: WhatsApp, screenshots and many editors
strip it. Camera details can be faked. So metadata never pushes the verdict towards "authentic".
Limits: labels inside compressed blocks are not found, and a label could be added to a real photo
on purpose (pointless for a faker, but possible).
"""

# The longer value first: it is the "partly AI" one.
LABELS = [
    (b"compositeWithTrainedAlgorithmicMedia", "edited"),
    (b"trainedAlgorithmicMedia", "generated"),
]


def check_file(data):
    """data: the file's bytes -> {"ai_label": "generated" | "edited" | None, "c2pa": bool}."""
    label = next((name for term, name in LABELS if term in data), None)
    # C2PA manifests live in JUMBF boxes labelled "c2pa" (JPEG APP11 segments, a PNG "caBX" chunk, ...).
    c2pa = b"c2pa" in data and (b"jumb" in data or b"caBX" in data)
    return {"ai_label": label, "c2pa": c2pa}


def describe(result):
    """One plain sentence for the app, or None if the file says nothing about AI."""
    if result["ai_label"] == "generated":
        return "The file itself says it was made by AI (IPTC label in its metadata)."
    if result["ai_label"] == "edited":
        return "The file itself says it was partly made or changed by AI (IPTC label in its metadata)."
    if result["c2pa"]:
        return "The file carries content credentials (C2PA), but they don't mark it as AI-made."
    return None
