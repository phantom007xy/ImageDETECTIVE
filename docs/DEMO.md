# Submission walkthrough

The strongest story is an everyday forwarded photo, a visible investigation, and measured limits. The demo examples are public assets; never show the owner's private test photos or filenames.

## Three-minute recording

| Time | Show | Say |
|---|---|---|
| 0:00–0:20 | Upload screen and four example buttons | "A forwarded photo can look convincing. ImageDETECTIVE checks AI generation, AI edits and individual faces, then shows the evidence." |
| 0:20–0:40 | Real phone-photo example | "This example has no clear AI signs. That does not verify the story, date or location." |
| 0:40–1:05 | Red-bicycle edit example; evidence and checks | "The whole-image check misses this subtle edit. Our fine-tuned Perception Encoder raises the alert. The map explains that check, inside the centre square it analysed; it is not an edit mask." |
| 1:05–1:25 | Group-photo example; numbered face boxes | "Faces are checked separately for AI generation and swaps. Small faces can remain unchecked; each box shows its result." |
| 1:25–1:40 | Original-file example; metadata row | "This file declares AI generation. We show the declaration as unverified, because metadata can be forged or stripped." |
| 1:40–2:05 | Model diagram and optional investigator result | "DINOv3 checks the image; two separately fine-tuned Perception Encoders check rendering and faces; local SAM 3 finds faces. Muse Spark and SAM 3.1 provide an optional explanation and outline. API failure cannot change the detector's verdict." |
| 2:05–2:35 | Final local results report | "Fresh Muse images: 100 of 100. The four judge-like generator groups: 20/20, 20/20, 19/20, 12/20 from pixels. We also check false alarms, compression, screenshots and crops. Calibration and test results are labelled separately." |
| 2:35–3:00 | Known limits and repo | "Crops, Nano Banana 2.1 and tight video swaps remain hard. Three of 30 real object photos were falsely flagged as likely AI. The repo includes the pipeline, checks and reports; trained weights are private. No pixel score is presented as proof." |

Record the working app after the hosted checks below pass. Use the report's final figures. Optional remote stages can be slow or unavailable: keep the complete local evidence visible. A paid generation or investigator demonstration needs an available API budget; local regression tests do not establish current remote availability.

## Hosted checks after the reviewed release is deployed

1. Publish code and the updated private edit calibration JSON together through the existing deployment process. The code, weights and calibration in the local report must match the deployed assets. Keep weights, keys, `.env`, private photos and private per-image results out of GitHub.
2. Confirm `/healthz` returns HTTP 200 and every required detector component is ready. A running web server alone does not establish detector readiness.
3. Open all four public samples. Confirm the same verdicts as `results/hardened_2026-11-04/gpu_samples.json`, allowing documented CPU/GPU score differences. Inspect face boxes and the PE centre-square map on mobile as well as desktop.
4. Recheck the public OpenImages boundary case identified in `cross_device_boundary.json`. It must no longer receive an unqualified pixel-only `AI-generated` verdict. This single check does not establish a hosted-wide false-alarm rate.
5. Confirm an invalid image and an oversized upload produce a clear error. Confirm local results remain available when the optional Meta API is unavailable or over budget.
6. After pushing the reviewed changes, check the actual GitHub Actions result. A workflow file alone is not a passing CI run.
7. Replace the README video placeholder with the recording link, check it without account access, and submit before the deadline.

The existing private-weights uploader accepts the current model paths below. This command publishes private model artifacts; run it only as part of the approved release, with the intended private repository ID:

```bash
python scripts/upload_heads.py --repo <private-model-repo> --head weights/google_v1/head_G3tt.pt --edit-check weights/edit_check_v3/head_EC3pf.pt
```

It also uploads the referenced face weights, second edit head and recalibrated JSON. Then deploy the reviewed code with `deploy/gcp_vm_setup.sh`; the setup now checks dependency consistency and waits for detector readiness. These instructions do not mean either upload or deployment has been performed.

## Explain the scores accurately

- The UI's AI evidence percentage is a model score, not the probability that a particular photo is fake.
- The 5% shared false-alarm budget applies to the calibration pools. It is not a guarantee on new photos.
- The 3,951 full-detector entries overlap; do not call them 3,951 independent unseen photos or combine every dataset into one accuracy.
- The 7,245 transformation checks reuse source photos and exclude face checks and metadata.
- Face-swap crop recall and whole-photo AI detection are different measures. Show their denominators.
- Green face boxes mean no clear signs were found in those crops. A fully AI-generated group can have green face boxes while the PE image check raises an alert; the checks do not prove individual faces authentic.
- No new model was trained for these hardening changes. The remaining model weaknesses are documented rather than hidden by presentation changes.

JPEG metadata implementation reference: the [official C2PA SDK's JPEG handler](https://github.com/contentauth/c2pa-rs/blob/main/sdk/src/asset_handlers/jpeg_io.rs) describes repeated headers in continuation packets. Our parser reads bounded structured declarations; it does not verify signatures or trust chains.
