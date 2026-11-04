ImageDETECTIVE review, 4 November 2026

This review assesses improvements feasible on the existing laptop before the user's 23:00 IST deadline. It covers the model and inference paths, training history, evaluation and calibration, local data availability, web app, API integration, deployment scripts, and repository state. No application code, weights, calibration, or private photos were changed. No paid API calls were made.

The laptop checkout is at b5e70cf2f6aff239b2117176b93ecac84bc14dfd, which matched GitHub's HEAD and main during the review. Existing local changes were preserved. The browsing tool initially could not open the hosted website; a subsequent HTTP test of its real basic-detection endpoint succeeded. One known-real public calibration image received the strongest AI-generated label there, while remaining below that strongest cutoff on the laptop GPU. This confirms a confidence-label failure on that hosted example; it does not establish hosted-wide error rates or concurrency behavior.

The subsequent full baseline and reusable checks are recorded in `results/release_2026-11-04/summary.md` and the corresponding JSON artifacts. The live baseline completed 3,951 list entries with no processing errors and matched the real HTTP upload path on all 80 judge images for the compared result fields. The full baseline peaked at 4.177 GiB allocated / 6.031 GiB reserved GPU memory. Original private calibration pools were also rerun offline: phone 0/104 AI, objects 4/100 AI, with no strongest AI-generated labels. The object pool's primary edit head flagged 3/100, exceeding its configured 2% allowance despite the final union remaining within 5%. These live findings supersede cached counts where precision changes a threshold crossing. No production code, model weights, calibration, or private photos were changed; no paid Meta APIs were called.

**What was verified live**

- NVIDIA RTX 5050 Laptop GPU, approximately 8 GB VRAM; PyTorch 2.11.0+cu128 supports its sm_120 architecture. bf16 matrix multiplication succeeded.
- System RAM: 15.7 GiB, with only 3.4 GiB available when sampled; free disk: 152.9 GiB.
- The complete current detector loaded offline in 12.73 seconds, including DINOv3, SAM 3, the edit PE and the face-swap PE. All expected branches were active.
- Four public example files were each checked twice. Warm runs took 0.594-1.111 seconds; peak allocated GPU memory was 3.686 GiB and peak reserved memory was 4.186 GiB. The real sample was authentic; the bicycle edit and AI group photo were flagged by PE; the original AI selfie was flagged from pixels and its metadata.
- All 97 Python files under app, src and scripts parsed successfully. JavaScript syntax checking passed. These are syntax checks, not a comprehensive behavioral test suite.
- All image references in 82 CSV lists under features/lists resolved to existing local files: 85,927 references, including repeated images across lists. This is an availability check, not proof of unique examples or absence of leakage.

An initial Python launcher failure was caused by sandbox access to its base interpreter. Running the existing environment with the required access worked; rebuilding the environment is unnecessary on this evidence.

**Fresh robustness check of the current v4 face model**

Ran `src.eval.face_swap_robustness` offline with the current face PE, head, and app threshold 0.22257614135742188. Each of three groups contained 30 public photos, tested in four versions. A photo counts as flagged if any sufficiently large detected face crosses the swap threshold; these numbers are not per-face recall or the whole ensemble verdict.

| Group | Original | WhatsApp-like copy | Small copy | Simulated phone processing |
|---|---|---|---|---|
| Real polished photos, 30 | 0/30 flagged | 0/30 | 0/30 | 0/30 |
| Real OpenImages photos, 30 | 0/30 flagged | 0/30 | 0/30 | 0/30 |
| OpenForensics swap photos, 30 | 30/30 detected | 29/30, 96.7% | 23/30, 76.7% | 14/30, 46.7% |

WhatsApp-like means a simulated 1,600-pixel/JPEG-75 copy, and small means an 800-pixel/JPEG-60 copy. Phone processing is the script's deterministic simulated beauty, tone, or portrait processing, not photos collected from actual phone hardware. This small check is encouraging for real-photo robustness but cannot establish a population false-alarm bound. It exposes a specific remaining swap weakness under strong reduction and processing. A bounded experiment could match these augmentations on TRAINING photos, with source-balanced real/fake sampling and fresh calibration. Do not train on the evaluated OpenForensics test photos. No gain from that experiment has yet been demonstrated.

**Model weaknesses and realistic work**

| Area | Evidence | Useful action tonight |
|---|---|---|
| Real object/scene false alarms | Stored final judge results report 3/30 real object photos called AI, versus 0/30 selfies. | Protect real-photo performance before increasing fake recall. Use independent public real objects/scenes for any new training and a fresh held-out check. Keep private calibration/test photos out of training. |
| New Google generators | Stored judge results: Nano Banana 2.1, 12/20 detected, 7 uncertain, 1 authentic. | At most one bounded specialist-head experiment on frozen cached features, with source-balanced training and the same joint false-alarm budget. A gain is not established. |
| Local AI edits | Current cached reconstruction: 94/132 local Muse edits detected; 31/132 called authentic. Historical robustness results lose further recall after cropping/downsizing and predate the last recalibration. | Validate current thresholds on transformed copies. A three-crop PE ablation is possible, but requires matching calibration and may not improve Muse edits, which re-render the whole image. |
| Face swaps outside the familiar photo domain | Stored current app results: OpenForensics 96.9%, DFDC 38.5%, tight WildDeepfake crops 13.7%. | Keep photo-first scope explicit. Broad cross-domain generalization is unlikely to be solved reliably before tonight. |

The current detector sees only the center square through its edit PE (`src/inference.py:196`, `src/models/pe_core.py:56`). Three-view support exists but is not active at inference. This is a coverage limitation and a testable hypothesis, not evidence that enabling three crops will help.

Full PE fine-tuning is technically feasible: `results/edit_check_v3.txt:4-15` records 7,000 steps at approximately 1.95 seconds/step, about 3.8 hours on this same GPU. The validation curve had already plateaued around 4,000 steps. `results/head_g_v1.txt:127-140` also records that adding more Google pictures did not help. Another broad fine-tune has a poor time-to-evidence tradeoff tonight. Head-only experiments and feature extraction are much cheaper. Training should run separately from the app, with enough system RAM freed for data loading.

**Concrete implementation improvements, in recommended order**

1. Connect the evidence map to the model that supplied the verdict. The map always displays DINO patch scores (`src/inference.py:191`, `app/server.py:169`), even when PE triggers the verdict. In the live bicycle sample, DINO scored -7.738 while the edit PE scored +11.868. PE's six-layer linear head permits a contribution map whose mean reproduces the existing PE score, without changing the classifier or thresholds. Show it only over the center square PE actually analyzed; identify the model and call this evidence, not a proven edit boundary. Estimated implementation/verification: 1-2 hours.
2. Remove external API work from the detector lock. `app/server.py:201` and `:234` share a lock; optional investigator and explanation calls therefore block subsequent basic checks. The API client timeout is 120 seconds (`src/services/meta_api.py:66`). Separate inference synchronization from API concurrency. Estimated work: 20-40 minutes.
3. Supply both PE heads to explanations. `src/services/explain.py:44-45` ignores `edit.flagged2`. An offline reproduction produced a Likely AI-generated verdict alongside `ai_edit_check: no clear signs`. Include the triggering branch and label the DINO probability as whole-image evidence. Estimated work: 10-20 minutes.
4. Bound retained image memory. `app/server.py:95-99,220` retains full-resolution image and evidence buffers without a global cache bound. Two RGB buffers for a 12 MP photo occupy about 72 MB before overhead; 50 such sessions approach 3.6 GB. Retain smaller follow-up copies and bound cached bytes/sessions. Estimated work: 30-45 minutes.
5. Make upload handling match the privacy promise. The HTML promises never saved to disk (`app/web/index.html:130-132,279`), but the FastAPI UploadFile path uses Starlette's spooled temporary files, whose threshold in the installed version is 1 MiB. Use bounded memory upload handling or accurately disclose temporary buffering. Estimated work: 20-45 minutes.
6. Add one-click public samples using the normal analysis path. The four useful examples already exist, but the current HTML page exposes no sample buttons. Estimated work: 20-40 minutes.

Additional correctness issues: the API budget check does not reserve in-flight spending and simultaneous identical calls can duplicate charges (`src/services/meta_api.py:72-86`). Metadata is detected by an unrestricted byte search (`src/services/metadata.py:25`), not a structured provenance parser or verified C2PA signature. An ordinary PNG Comment mentioning trainedAlgorithmicMedia was classified as an AI label in an offline reproduction. Treat this as an unverified file declaration; parse the actual metadata field before using it to override the pixel verdict.

**Evaluation and submission work**

Current cached reconstruction, using existing heads and final joint thresholds, gave:

| Pool | Count flagged AI | Other result |
|---|---|---|
| Polished real photos | 16/330, 4.85% | 79 uncertain |
| OpenImages real photos | 11/576, 1.91% | 82 uncertain |
| Real close-ups | 6/150, 4.00% | 49 uncertain; 149 unique IDs |
| Public selfies/portraits | 2/62, 3.23% | 13 uncertain |
| Muse v1 generated | 100/100 | 0 authentic |
| Local Muse edits | 94/132, 71.21% | 31 authentic |
| Whole-photo Muse edits | 58/66, 87.88% | 7 authentic |
| SDXL | 317/332, 95.48% | 0 authentic |
| SD 3.5 | 20/38, 52.63% | 9 authentic |
| Gemini dev | 27/30, 90% | 3 authentic |

These are reconstructions from cached features, not fresh independent tests. Cached fp16 feature storage can affect borderline scores relative to live inference. README SDXL 92% appears historical; OpenImages 1.7% differs by one image from the current cache calculation. Generate the final result table once from the frozen configuration rather than mixing versions.

The existing leakage checker omits later training sources and newer test manifests (`src/eval/leak_check.py:14-20`). That is incomplete verification, not proof that leakage occurred. Extend it to the final training manifest and source identities. Also distinguish the reused judge regression set from an untouched final evaluation: its first object-photo errors influenced new calibration data and the final edit allowance (`results/judge_test_v1.txt:32-37`).

The README still has a placeholder for the three-minute video (`README.md:7`). Reserve the last three hours for a frozen-model regression, the video, final documentation, submission, and buffer. A practical cutoff is 20:00 IST for model changes. Prefer fixes with predictable behavior gains, and at most one controlled model experiment with a separate checkpoint and a clear rollback decision.
