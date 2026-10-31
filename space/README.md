---
title: ImageDETECTIVE
emoji: 🔍
colorFrom: gray
colorTo: red
sdk: gradio
sdk_version: 6.31.0
python_version: "3.12"
app_file: main.py
pinned: false
license: mit
short_description: Real, AI-made or AI-edited? One verdict, with evidence.
---

# ImageDETECTIVE

Is this photo real, made by AI, or changed by AI? Upload one photo: you get one clear verdict, a calibrated
confidence and where the evidence is (an evidence map, a score for every face, and Meta's investigator: Muse Spark
names what was most likely changed and SAM 3.1 outlines it).

Built for the Meta Global AI Developer Hackathon 2026. **Built with DINOv3**, Meta's Perception Encoder, SAM 3 /
SAM 3.1, Muse Image and Muse Spark. Code, method, data and honest test numbers:
[github.com/phantom007xy/ImageDETECTIVE](https://github.com/phantom007xy/ImageDETECTIVE).

"Likely authentic" means no AI traces were found; it doesn't confirm when or where a photo was taken. Uploaded photos
are never used for training and are deleted from the server within an hour.
