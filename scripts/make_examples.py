"""Make the app's sample photos (app/examples/) and their credits file, from shareable sources only.

Only our own Muse Image pictures and edits (made for this project; nobody in them is real) and a Wikimedia Commons
photo under CC0 (credited anyway); never private photos. "forward" = what a WhatsApp forward does (long side at most
1600 px, JPEG quality 75, no metadata), so only the pixels can tell; "copy" keeps the file byte for byte (the Muse
original keeps its AI label, to show the metadata check).

Usage (from the repo root):  python scripts/make_examples.py
"""

import io
from pathlib import Path

from PIL import Image, ImageOps

OUT = Path("app/examples")
SAMPLES = [  # (file name, source, how, credit)
    ("1_real_phone_photo.jpg", "D:/Datasets/Commons/v1/dev/commons_00782.jpg", "copy",
     'A real phone photo (Google Pixel 7): "2025-01-07 SNCF Class X 73500 driver cabin" by RAIL P (RAIL.PHOTOGRAPHY), '
     "CC0, https://commons.wikimedia.org/wiki/File:2025-01-07_SNCF_Class_X_73500_driver_cabin.jpg"),
    ("2_ai_edit_red_bicycle.jpg", "D:/Datasets/MuseImage/edits_v1/edit_0060_edited.png", "forward",
     "The same photo, edited with Meta's Muse Image for this project (prompt: \"add a red bicycle somewhere in the "
     'scene"); a forwarded copy without metadata'),
    ("3_ai_group_photo.jpg", "D:/Datasets/MuseImage/test_v1/mi_test_014.png", "forward",
     "Made with Meta's Muse Image for this project (prompt: \"group photo of eight colleagues standing in front of an "
     'office building in Bangalore, taken by a passerby on a phone"); nobody in it is real; a forwarded copy without '
     "metadata"),
    ("4_ai_selfie_original_file.png", "D:/Datasets/MuseImage/test_v1/mi_test_003.png", "copy",
     "Made with Meta's Muse Image for this project (prompt: \"selfie of two friends laughing in a college canteen in "
     'India, front camera, slightly blurry background"); nobody in it is real; the original file, with the AI label '
     "Muse Image writes into its metadata"),
]


def forward(path):
    with Image.open(path) as im:
        image = ImageOps.exif_transpose(im).convert("RGB")
    image.thumbnail((1600, 1600), Image.BICUBIC)  # only ever smaller
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=75)  # Pillow writes no metadata unless asked
    return buffer.getvalue()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    lines = ["# Sample photos", "", "Shareable sources only: our own Muse Image pictures (nobody in them is real) and a "
             "CC0 photo from Wikimedia Commons. Made with `python scripts/make_examples.py`.", ""]
    for name, source, how, credit in SAMPLES:
        data = Path(source).read_bytes() if how == "copy" else forward(source)
        (OUT / name).write_bytes(data)
        lines.append(f"- `{name}`: {credit}.")
        print(f"{name}: {len(data) / 1e6:.1f} MB")
    (OUT / "CREDITS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
