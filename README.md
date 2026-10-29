# ImageDETECTIVE

An authenticity checker for photos. It checks a single photo and gives one clear verdict: is it real, fully AI-generated, or partly changed by AI? It also shows a calibrated confidence and **where** the evidence is (the edited region, a suspicious face, or where the AI traces are strongest).

Built for the **Meta Global AI Developer Hackathon 2026** (Indie Developer track, theme "Build for Your World"), 26 Oct – 4 Nov 2026.

> **Status:** work in progress. Day 4: data checks done, DINOv3 feature extraction working; no detector or app yet.

## Why

Fake and AI-edited images spread fast through forwarded messages (for example on WhatsApp). People who receive them, fact-checkers and journalists need a quick, honest answer, and one that rarely calls a real photo fake.

## How it works

*Coming soon.* Planned: one Meta DINOv3 backbone for the whole image and for each face, an edited-region map, SAM 3.1 to name regions, and Muse Spark for a short plain-language explanation. The verdict never depends on an external API call.

## Data used

| Dataset | Link | Licence / terms | Used for |
|---|---|---|---|
| WildDeepfake | [Hugging Face](https://huggingface.co/datasets/xingjunm/WildDeepfake) | Hugging Face lists Apache 2.0; the authors' [GitHub](https://github.com/OpenTAI/wild-deepfake) states research-only use | Evaluation only (test split) |
| Chameleon | [AIDE GitHub](https://github.com/shilinyan99/AIDE); the official release is a list of image URLs, and the images were obtained via the copy the authors pointed to ([issue #1](https://github.com/shilinyan99/AIDE/issues/1)) | "Only used for academic research. Commercial use in any form is prohibited." (authors' terms). The images belong to their original owners and are not shared in this repository | Evaluation only |

*More datasets will be added here as they are used.*

### References

- **Chameleon** comes from *A Sanity Check for AI-generated Image Detection* by Shilin Yan, Ouxiang Li, Jiayin Cai, Yanbin Hao, Xiaolong Jiang, Yao Hu and Weidi Xie ([arXiv:2406.19435](https://arxiv.org/abs/2406.19435), [project page](https://shilinyan99.github.io/AIDE)). The paper asks whether AI-image detectors still work on today's most realistic AI images. Its Chameleon set (26,033 images: 14,863 real, 11,170 AI-generated) keeps only AI images that human annotators took for real photos. The paper also proposes AIDE, a detector that combines low-level pixel traces with high-level image meaning.

```bibtex
@article{yan2024sanity,
  title={A Sanity Check for AI-generated Image Detection},
  author={Yan, Shilin and Li, Ouxiang and Cai, Jiayin and Hao, Yanbin and Jiang, Xiaolong and Hu, Yao and Xie, Weidi},
  journal={arXiv preprint arXiv:2406.19435},
  year={2024}
}
```

## Background

I worked on the same problem in ImageCLEF 2026 ([paper](https://ceur-ws.org/Vol-4283/paper289.pdf)). This project was built from scratch during the hackathon, with different models and a different architecture.

## Licence

The code in this repository is released under the [MIT Licence](LICENSE). It covers only the code written for this project. Pretrained models (e.g. DINOv3, under the [DINOv3 Licence](https://ai.meta.com/resources/models-and-libraries/dinov3-license)) and datasets are not part of this repository and keep their own licences (see "Data used" above).
