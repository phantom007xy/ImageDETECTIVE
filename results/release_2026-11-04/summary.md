# Current baseline before final commit

**Release ready: false.** Live evaluation: **complete** (19/19 cases complete; 0 errors).

Current baseline before final commit; no improved-model claim and no new independent final-test claim.

## Blocking app failures

**5 failures / 26 checks; 0 test errors.**

- **test_budget_reserves_inflight_cost_and_deduplicates_same_request**: 0.02 not less than or equal to 0.01 : Concurrent identical requests charged more than the remaining simulated daily cap
- **test_comment_mention_is_not_an_actual_ai_source_declaration**: 'generated' is not None : Ordinary PNG Comment text was mistaken for an AI source field
- **test_memory_only_upload_does_not_spool_to_disk**: 1 != 0 : Synthetic upload reached temporary disk storage despite memory-only promise
- **test_optional_api_does_not_block_another_detector_verdict**: False is not true : A blocked optional API call held up the next basic detector verdict
- **test_secondary_head_facts_describe_positive_detection**: 'no clear signs' == 'no clear signs' : Secondary PE head triggered the verdict but explanation facts say no clear signs

## Hosted confidence boundary check

Status: **complete**. One known-real public OpenImages calibration photo returned **AI-generated**; confident false alarm: **True**.
The laptop GPU kept this photo below the strongest AI threshold. A separate laptop CPU bf16 whole-image check crossed it, and the actual hosted endpoint also returned the strongest AI label. This is a confirmed label failure on one real photo, not an estimate of the hosted false-alarm rate. No private photo or paid investigator API was used.

## Release gates

- PASS: all required artifacts present
- BLOCKED: all app checks pass
- PASS: all 19 live cases complete without errors
- PASS: real upload ui 80 match
- PASS: environment requirements match
- PASS: required calibration and v4 cache checks pass
- PASS: live and cached configuration match
- PASS: final ensemble calibration mix and per pool caps pass
- PASS: desktop mobile browser checks pass
- PASS: scoped private report scan passes
- PASS: whole image robustness complete without errors
- PASS: private calibration live complete without errors
- BLOCKED: private live union 5pct and each edit head 2pct caps pass
- PASS: frozen production artifacts and python syntax valid
- BLOCKED: hosted public real boundary avoids confident false alarm

Completed named checks (these cases are not pooled into one accuracy):

- judge_ai: 80 images; 71 pixel-AI, 8 uncertain, 1 authentic; 80 AI including file metadata; 0 errors.
- fresh_muse_v2: 100 images; 100 pixel-AI, 0 uncertain, 0 authentic; 100 AI including file metadata; 0 errors.
- private_judge: 60 images; 3 pixel-AI, 0 uncertain, 57 authentic; 3 AI including file metadata; 0 errors.

## Live source counts

AI counts below use pixels; original-file metadata flags are kept separate in JSON. Partial cases are still running.

| Case / source | Case complete | n | Pixel AI | Uncertain | Authentic | Errors |
|---|---|---:|---:|---:|---:|---:|
| judge_ai / judge/chatgpt | True | 20 | 20 | 0 | 0 | 0 |
| judge_ai / judge/metaai | True | 20 | 20 | 0 | 0 | 0 |
| judge_ai / judge/nanobanana_2_1 | True | 20 | 12 | 7 | 1 | 0 |
| judge_ai / judge/nanobanana_pro | True | 20 | 19 | 1 | 0 | 0 |
| fresh_muse_v2 / muse_test_v2/animal | True | 14 | 14 | 0 | 0 | 0 |
| fresh_muse_v2 / muse_test_v2/object_food | True | 13 | 13 | 0 | 0 | 0 |
| fresh_muse_v2 / muse_test_v2/people | True | 60 | 60 | 0 | 0 | 0 |
| fresh_muse_v2 / muse_test_v2/scene | True | 13 | 13 | 0 | 0 | 0 |
| private_judge / private/objects | True | 30 | 3 | 0 | 27 | 0 |
| private_judge / private/selfies | True | 30 | 0 | 0 | 30 | 0 |
| polished_real / commons/people | True | 30 | 1 | 8 | 21 | 0 |
| polished_real / commons/phone | True | 54 | 3 | 11 | 40 | 0 |
| polished_real / commons/portrait | True | 209 | 10 | 53 | 146 | 0 |
| polished_real / commons/quality | True | 37 | 2 | 8 | 27 | 0 |
| openimages_real / sid_set/openimages | True | 576 | 11 | 82 | 483 | 0 |
| phone_real_public / commons/phone | True | 31 | 0 | 5 | 26 | 0 |
| phone_real_public / commons/quality | True | 1 | 0 | 0 | 1 | 0 |
| selfies_real_public / commons/portrait | True | 48 | 2 | 12 | 34 | 0 |
| selfies_real_public / commons/selfie | True | 14 | 0 | 1 | 13 | 0 |
| closeups_real / real_face_crops/commons/people | True | 8 | 1 | 3 | 4 | 0 |
| closeups_real / real_face_crops/commons/phone | True | 1 | 0 | 1 | 0 | 0 |
| closeups_real / real_face_crops/commons/portrait | True | 91 | 6 | 28 | 57 | 0 |
| closeups_real / real_face_crops/commons/quality | True | 5 | 0 | 3 | 2 | 0 |
| closeups_real / real_face_crops/commons/selfie | True | 7 | 0 | 2 | 5 | 0 |
| closeups_real / real_face_crops/sid_set/openimages | True | 38 | 0 | 11 | 27 | 0 |
| muse_pairs / muse_pairs/muse | True | 76 | 75 | 0 | 1 | 0 |
| muse_pairs / muse_pairs/real | True | 76 | 1 | 12 | 63 | 0 |
| muse_edits_v2 / muse_edits/local | True | 132 | 94 | 7 | 31 | 0 |
| muse_edits_v2 / muse_edits/original | True | 198 | 4 | 36 | 158 | 0 |
| muse_edits_v2 / muse_edits/whole | True | 66 | 58 | 1 | 7 | 0 |
| muse_edits_v1 / muse_edits/local | True | 51 | 30 | 6 | 15 | 0 |
| muse_edits_v1 / muse_edits/original | True | 76 | 2 | 18 | 56 | 0 |
| muse_edits_v1 / muse_edits/whole | True | 25 | 23 | 0 | 2 | 0 |
| muse_v1 / muse_image/animal | True | 14 | 14 | 0 | 0 | 0 |
| muse_v1 / muse_image/object_food | True | 13 | 13 | 0 | 0 | 0 |
| muse_v1 / muse_image/people/candid | True | 12 | 12 | 0 | 0 | 0 |
| muse_v1 / muse_image/people/group | True | 12 | 12 | 0 | 0 | 0 |
| muse_v1 / muse_image/people/indoor_lowlight | True | 12 | 12 | 0 | 0 | 0 |
| muse_v1 / muse_image/people/portrait | True | 12 | 12 | 0 | 0 | 0 |
| muse_v1 / muse_image/people/selfie | True | 12 | 12 | 0 | 0 | 0 |
| muse_v1 / muse_image/scene | True | 13 | 13 | 0 | 0 | 0 |
| sdxl / sd_faces/sdxl | True | 332 | 317 | 15 | 0 | 0 |
| open_generators / open_gen/k22_devcaption | True | 77 | 75 | 2 | 0 | 0 |
| open_generators / open_gen/pixart | True | 396 | 388 | 7 | 1 | 0 |
| open_generators / open_gen/sd35m_devcaption | True | 38 | 26 | 5 | 7 | 0 |
| gan_tpdne / gan/tpdne | True | 244 | 233 | 11 | 0 | 0 |
| gan_sfhq / sfhq/pt3 | True | 375 | 345 | 30 | 0 | 0 |
| gemini_dev / gemini_api/3.1-flash-image | True | 30 | 27 | 0 | 3 | 0 |
| google_captions_dev / open_gen/gemini-2.5-flash-image_devcaption | True | 16 | 13 | 3 | 0 | 0 |
| google_captions_dev / open_gen/gemini-3-pro-image_devcaption | True | 14 | 10 | 1 | 3 | 0 |
| google_captions_dev / open_gen/gemini-3.1-flash-image_devcaption | True | 37 | 25 | 10 | 2 | 0 |
| commons_ai_photos / commons_ai/bing_dalle3 | True | 14 | 13 | 1 | 0 | 0 |
| commons_ai_photos / commons_ai/chatgpt | True | 19 | 18 | 1 | 0 | 0 |
| commons_ai_photos / commons_ai/dalle | True | 32 | 25 | 3 | 4 | 0 |
| commons_ai_photos / commons_ai/firefly | True | 2 | 1 | 1 | 0 | 0 |
| commons_ai_photos / commons_ai/flux | True | 7 | 6 | 0 | 1 | 0 |
| commons_ai_photos / commons_ai/flux/ai_restored | True | 1 | 0 | 0 | 1 | 0 |
| commons_ai_photos / commons_ai/gemini | True | 19 | 16 | 1 | 2 | 0 |
| commons_ai_photos / commons_ai/gemini/ai_restored | True | 30 | 8 | 2 | 20 | 0 |
| commons_ai_photos / commons_ai/grok | True | 11 | 11 | 0 | 0 | 0 |
| commons_ai_photos / commons_ai/ideogram | True | 2 | 1 | 1 | 0 | 0 |
| commons_ai_photos / commons_ai/imagen | True | 2 | 2 | 0 | 0 | 0 |
| commons_ai_photos / commons_ai/mai_image | True | 1 | 1 | 0 | 0 | 0 |
| commons_ai_photos / commons_ai/meta_ai | True | 3 | 3 | 0 | 0 | 0 |
| commons_ai_photos / commons_ai/midjourney | True | 18 | 14 | 3 | 1 | 0 |
| commons_ai_photos / commons_ai/nano_banana | True | 4 | 4 | 0 | 0 | 0 |
| commons_ai_photos / commons_ai/reve | True | 1 | 1 | 0 | 0 | 0 |
| commons_ai_photos / commons_ai/stable_diffusion | True | 33 | 33 | 0 | 0 | 0 |
| commons_ai_photos / commons_ai/stylegan | True | 3 | 3 | 0 | 0 | 0 |

## Live versus cached

Deltas are live minus cached and are emitted only for complete cases with equal source denominators. Missing runtime faces make some cached rows partial.

| Case / source | Cached scope | n | Δ AI | Δ uncertain | Δ authentic |
|---|---|---:|---:|---:|---:|
| polished_real / commons/people | complete_cached_pipeline | 30 | +0 | +0 | +0 |
| polished_real / commons/phone | complete_cached_pipeline | 54 | +0 | +0 | +0 |
| polished_real / commons/portrait | complete_cached_pipeline | 209 | +0 | +1 | -1 |
| polished_real / commons/quality | complete_cached_pipeline | 37 | +0 | +0 | +0 |
| openimages_real / sid_set/openimages | complete_cached_pipeline | 576 | +0 | +0 | +0 |
| phone_real_public / commons/phone | complete_cached_pipeline | 31 | +0 | +0 | +0 |
| phone_real_public / commons/quality | complete_cached_pipeline | 1 | +0 | +0 | +0 |
| selfies_real_public / commons/portrait | complete_cached_pipeline | 48 | +0 | +0 | +0 |
| selfies_real_public / commons/selfie | complete_cached_pipeline | 14 | +0 | +0 | +0 |
| closeups_real / real_face_crops/commons/people | complete_cached_pipeline | 8 | +0 | +0 | +0 |
| closeups_real / real_face_crops/commons/phone | complete_cached_pipeline | 1 | +0 | +0 | +0 |
| closeups_real / real_face_crops/commons/portrait | complete_cached_pipeline | 91 | +1 | -1 | +0 |
| closeups_real / real_face_crops/commons/quality | complete_cached_pipeline | 5 | +0 | +0 | +0 |
| closeups_real / real_face_crops/commons/selfie | complete_cached_pipeline | 7 | +0 | +0 | +0 |
| closeups_real / real_face_crops/sid_set/openimages | complete_cached_pipeline | 38 | +0 | +0 | +0 |
| muse_pairs / muse_pairs/muse | complete_cached_pipeline | 76 | +0 | +0 | +0 |
| muse_pairs / muse_pairs/real | complete_cached_pipeline | 76 | +0 | +0 | +0 |
| muse_edits_v2 / muse_edits/local | partial_missing_face_cache | 132 | +0 | +0 | +0 |
| muse_edits_v2 / muse_edits/original | partial_missing_face_cache | 198 | +0 | +0 | +0 |
| muse_edits_v2 / muse_edits/whole | partial_missing_face_cache | 66 | +0 | +0 | +0 |
| muse_edits_v1 / muse_edits/local | partial_missing_face_cache | 51 | +0 | +0 | +0 |
| muse_edits_v1 / muse_edits/original | partial_missing_face_cache | 76 | +0 | +0 | +0 |
| muse_edits_v1 / muse_edits/whole | partial_missing_face_cache | 25 | +0 | +0 | +0 |
| muse_v1 / muse_image/animal | complete_cached_pipeline | 14 | +0 | +0 | +0 |
| muse_v1 / muse_image/object_food | complete_cached_pipeline | 13 | +0 | +0 | +0 |
| muse_v1 / muse_image/people/candid | complete_cached_pipeline | 12 | +0 | +0 | +0 |
| muse_v1 / muse_image/people/group | complete_cached_pipeline | 12 | +0 | +0 | +0 |
| muse_v1 / muse_image/people/indoor_lowlight | complete_cached_pipeline | 12 | +0 | +0 | +0 |
| muse_v1 / muse_image/people/portrait | complete_cached_pipeline | 12 | +0 | +0 | +0 |
| muse_v1 / muse_image/people/selfie | complete_cached_pipeline | 12 | +0 | +0 | +0 |
| muse_v1 / muse_image/scene | complete_cached_pipeline | 13 | +0 | +0 | +0 |
| sdxl / sd_faces/sdxl | complete_cached_pipeline | 332 | +0 | +0 | +0 |
| open_generators / open_gen/k22_devcaption | partial_missing_face_cache | 77 | +0 | +0 | +0 |
| open_generators / open_gen/pixart | partial_missing_face_cache | 396 | +0 | +0 | +0 |
| open_generators / open_gen/sd35m_devcaption | partial_missing_face_cache | 38 | +6 | -4 | -2 |
| gan_tpdne / gan/tpdne | complete_cached_pipeline | 244 | +0 | +0 | +0 |
| gan_sfhq / sfhq/pt3 | partial_missing_face_cache | 375 | +0 | +0 | +0 |
| gemini_dev / gemini_api/3.1-flash-image | partial_missing_face_cache | 30 | +0 | +0 | +0 |
| google_captions_dev / open_gen/gemini-2.5-flash-image_devcaption | partial_missing_face_cache | 16 | +0 | +0 | +0 |
| google_captions_dev / open_gen/gemini-3-pro-image_devcaption | partial_missing_face_cache | 14 | +0 | +0 | +0 |
| google_captions_dev / open_gen/gemini-3.1-flash-image_devcaption | partial_missing_face_cache | 37 | +0 | +0 | +0 |
| commons_ai_photos / commons_ai/bing_dalle3 | partial_missing_face_cache | 14 | +0 | +0 | +0 |
| commons_ai_photos / commons_ai/chatgpt | partial_missing_face_cache | 19 | +0 | +0 | +0 |
| commons_ai_photos / commons_ai/dalle | partial_missing_face_cache | 32 | +0 | +0 | +0 |
| commons_ai_photos / commons_ai/firefly | partial_missing_face_cache | 2 | +0 | +0 | +0 |
| commons_ai_photos / commons_ai/flux | partial_missing_face_cache | 7 | +0 | +0 | +0 |
| commons_ai_photos / commons_ai/flux/ai_restored | partial_missing_face_cache | 1 | +0 | +0 | +0 |
| commons_ai_photos / commons_ai/gemini | partial_missing_face_cache | 19 | +0 | +0 | +0 |
| commons_ai_photos / commons_ai/gemini/ai_restored | partial_missing_face_cache | 30 | +0 | +0 | +0 |
| commons_ai_photos / commons_ai/grok | partial_missing_face_cache | 11 | +0 | +0 | +0 |
| commons_ai_photos / commons_ai/ideogram | partial_missing_face_cache | 2 | +0 | +0 | +0 |
| commons_ai_photos / commons_ai/imagen | partial_missing_face_cache | 2 | +0 | +0 | +0 |
| commons_ai_photos / commons_ai/mai_image | partial_missing_face_cache | 1 | +0 | +0 | +0 |
| commons_ai_photos / commons_ai/meta_ai | partial_missing_face_cache | 3 | +0 | +0 | +0 |
| commons_ai_photos / commons_ai/midjourney | partial_missing_face_cache | 18 | +0 | +0 | +0 |
| commons_ai_photos / commons_ai/nano_banana | partial_missing_face_cache | 4 | +0 | +0 | +0 |
| commons_ai_photos / commons_ai/reve | partial_missing_face_cache | 1 | +0 | +0 | +0 |
| commons_ai_photos / commons_ai/stable_diffusion | partial_missing_face_cache | 33 | +0 | +0 | +0 |
| commons_ai_photos / commons_ai/stylegan | partial_missing_face_cache | 3 | +0 | +0 | +0 |

## README figures to review

- SDXL pixel detection: README 92%; cached 317/332 (95.48%); 317/332 (95.48%) live.
- OpenImages unseen real false alarms: README 1.7%; cached 11/576 (1.91%); 11/576 (1.91%) live.
- SD 3.5 full-app pixel detection: README about half; cached 20/38 (52.63%); 26/38 (68.42%) live.
- Local Muse edits, small (whole-image + both PE heads; no faces/metadata): README about 60%; cached unavailable; 72/132 (54.55%) live.
- Local Muse edits, crop (whole-image + both PE heads; no faces/metadata): README about 60%; cached unavailable; 70/132 (53.03%) live.

## Robustness of the current face branch

Counts are photos with a strict face flag, not full-detector recall.

| Group | n | Original | WhatsApp | Small | Simulated phone processing |
|---|---:|---:|---:|---:|---:|
| REAL polished dev | 30 | 0 | 0 | 0 | 0 |
| REAL OpenImages unseen | 30 | 0 | 0 | 0 | 0 |
| SWAPS OpenForensics test | 30 | 30 | 29 | 23 | 14 |

## Calibration promises on the saved mix

Final cached ensemble on exactly the saved whole-calibration real_pools and fake_sets; calibration-mix check only, not independent performance.

Class-balanced Uncertain: **10.3925%** (cap 15%): 0.5 × (255/1430) + 0.5 × (15/508).

| Required real-photo pool | n | Full-ensemble AI | Allowed at 5% | Passed |
|---|---:|---:|---:|---|
| polished dev | 330 | 16 | 16 | True |
| OpenImages val | 76 | 1 | 3 | True |
| OpenImages unseen | 576 | 11 | 28 | True |
| Commons v3 dev | 32 | 0 | 1 | True |
| close-ups | 150 | 6 | 7 | True |
| selfies+portraits | 62 | 2 | 3 | True |
| phone | 104 | 0 | 5 | True |
| phone objects | 100 | 3 | 5 | True |

| Required real-face pool / head | n faces | Strict | Allowed at 0.5% | Passed |
|---|---:|---:|---:|---|
| polished dev / ai_made | 285 | 1 | 1 | True |
| polished dev / swap | 285 | 0 | 1 | True |
| OpenImages val / ai_made | 81 | 0 | 0 | True |
| OpenImages val / swap | 81 | 0 | 0 | True |
| OpenImages unseen / ai_made | 557 | 0 | 2 | True |
| OpenImages unseen / swap | 557 | 0 | 2 | True |
| Commons v3 dev / ai_made | 15 | 0 | 0 | True |
| Commons v3 dev / swap | 15 | 0 | 0 | True |
| close-ups / ai_made | 174 | 0 | 0 | True |
| close-ups / swap | 174 | 0 | 0 | True |
| selfies+portraits / ai_made | 78 | 0 | 0 | True |
| selfies+portraits / swap | 78 | 0 | 0 | True |
| phone / ai_made | 148 | 0 | 0 | True |
| phone / swap | 148 | 0 | 0 | True |
| phone objects / ai_made | 20 | 0 | 0 | True |
| phone objects / swap | 20 | 0 | 0 | True |
| DFDC p3 real / swap | 1452 | 7 | 7 | True |
| SID val real / swap | 292 | 1 | 1 | True |

The 0.5% checks apply to each calibrated face head on its specified pools. Separate head caps do not mathematically guarantee a 0.5% union of both heads; off-domain test pools are not guaranteed.

## Threshold precision checks

- real_face_crops_dev_all: Uncertain cached → Likely AI-generated live; whole-score delta +0.002010345. Within the cached mean-feature float16 rounding bound: True.
- commons_dev: Likely authentic cached → Uncertain live; whole-score delta +0.001902819. Within the cached mean-feature float16 rounding bound: True.
Small threshold crossings within the mean-vector float16 rounding bound are consistent with cache quantization; no evidence of stale weights from these cases.

The single real calibration maximum came from OpenImages unseen. Cached score 5.520974636 equals the strong-AI line; recorded live score 5.520281792 has margin -0.000692844. Live verdict: Likely AI-generated; above strong line: False.

## Browser and scoped privacy checks

Browser: **12/12 passed**. Real Chrome interactions; actual frontend/ASGI; synthetic uploads; fake CPU detector. Actual model inference and paid APIs were mocked for these interactions.

Privacy scan: passed=True; 0 findings, 23 public artifacts checked against 264 private judge-image rows; 0 private prediction keys in the public cache. This is the scan's stated scope, not an assertion about all possible private data.

## Whole-image robustness

Status: **complete**; 9 groups; 0 errors. Seven versions per group are required for this gate.

| Group / version | n | AI | Uncertain | Authentic | Errors |
|---|---:|---:|---:|---:|---:|
| polished_real / original | 150 | 4 | 35 | 111 | 0 |
| polished_real / jpeg70 | 150 | 3 | 34 | 113 | 0 |
| polished_real / whatsapp | 150 | 3 | 34 | 113 | 0 |
| polished_real / whatsapp2 | 150 | 3 | 35 | 112 | 0 |
| polished_real / small | 150 | 4 | 31 | 115 | 0 |
| polished_real / screenshot | 150 | 4 | 35 | 111 | 0 |
| polished_real / crop | 150 | 7 | 40 | 103 | 0 |
| openimages_real / original | 150 | 1 | 23 | 126 | 0 |
| openimages_real / jpeg70 | 150 | 0 | 18 | 132 | 0 |
| openimages_real / whatsapp | 150 | 0 | 18 | 132 | 0 |
| openimages_real / whatsapp2 | 150 | 0 | 18 | 132 | 0 |
| openimages_real / small | 150 | 0 | 14 | 136 | 0 |
| openimages_real / screenshot | 150 | 1 | 23 | 126 | 0 |
| openimages_real / crop | 150 | 1 | 14 | 135 | 0 |
| muse_v2_originals / original | 99 | 2 | 18 | 79 | 0 |
| muse_v2_originals / jpeg70 | 99 | 1 | 14 | 84 | 0 |
| muse_v2_originals / whatsapp | 99 | 1 | 16 | 82 | 0 |
| muse_v2_originals / whatsapp2 | 99 | 1 | 17 | 81 | 0 |
| muse_v2_originals / small | 99 | 0 | 15 | 84 | 0 |
| muse_v2_originals / screenshot | 99 | 1 | 20 | 78 | 0 |
| muse_v2_originals / crop | 99 | 1 | 19 | 79 | 0 |
| muse_v2_local / original | 132 | 94 | 7 | 31 | 0 |
| muse_v2_local / jpeg70 | 132 | 82 | 10 | 40 | 0 |
| muse_v2_local / whatsapp | 132 | 84 | 11 | 37 | 0 |
| muse_v2_local / whatsapp2 | 132 | 82 | 11 | 39 | 0 |
| muse_v2_local / small | 132 | 72 | 12 | 48 | 0 |
| muse_v2_local / screenshot | 132 | 93 | 8 | 31 | 0 |
| muse_v2_local / crop | 132 | 70 | 12 | 50 | 0 |
| muse_v2_whole / original | 66 | 58 | 1 | 7 | 0 |
| muse_v2_whole / jpeg70 | 66 | 59 | 1 | 6 | 0 |
| muse_v2_whole / whatsapp | 66 | 58 | 2 | 6 | 0 |
| muse_v2_whole / whatsapp2 | 66 | 58 | 2 | 6 | 0 |
| muse_v2_whole / small | 66 | 57 | 3 | 6 | 0 |
| muse_v2_whole / screenshot | 66 | 58 | 2 | 6 | 0 |
| muse_v2_whole / crop | 66 | 58 | 1 | 7 | 0 |
| muse_v1 / original | 100 | 100 | 0 | 0 | 0 |
| muse_v1 / jpeg70 | 100 | 100 | 0 | 0 | 0 |
| muse_v1 / whatsapp | 100 | 100 | 0 | 0 | 0 |
| muse_v1 / whatsapp2 | 100 | 100 | 0 | 0 | 0 |
| muse_v1 / small | 100 | 100 | 0 | 0 | 0 |
| muse_v1 / screenshot | 100 | 100 | 0 | 0 | 0 |
| muse_v1 / crop | 100 | 100 | 0 | 0 | 0 |
| sdxl / original | 150 | 142 | 8 | 0 | 0 |
| sdxl / jpeg70 | 150 | 132 | 18 | 0 | 0 |
| sdxl / whatsapp | 150 | 135 | 15 | 0 | 0 |
| sdxl / whatsapp2 | 150 | 135 | 15 | 0 | 0 |
| sdxl / small | 150 | 120 | 30 | 0 | 0 |
| sdxl / screenshot | 150 | 140 | 10 | 0 | 0 |
| sdxl / crop | 150 | 129 | 21 | 0 | 0 |
| pixart / original | 150 | 148 | 2 | 0 | 0 |
| pixart / jpeg70 | 150 | 143 | 7 | 0 | 0 |
| pixart / whatsapp | 150 | 146 | 4 | 0 | 0 |
| pixart / whatsapp2 | 150 | 146 | 4 | 0 | 0 |
| pixart / small | 150 | 141 | 9 | 0 | 0 |
| pixart / screenshot | 150 | 148 | 2 | 0 | 0 |
| pixart / crop | 150 | 147 | 2 | 1 | 0 |
| sd35 / original | 38 | 20 | 9 | 9 | 0 |
| sd35 / jpeg70 | 38 | 16 | 12 | 10 | 0 |
| sd35 / whatsapp | 38 | 15 | 13 | 10 | 0 |
| sd35 / whatsapp2 | 38 | 16 | 13 | 9 | 0 |
| sd35 / small | 38 | 12 | 14 | 12 | 0 |
| sd35 / screenshot | 38 | 19 | 10 | 9 | 0 |
| sd35 / crop | 38 | 17 | 11 | 10 | 0 |

## Original private calibration pools

Status: **complete**; complete=True; errors=0. These are calibration checks, not independent test results.

| Neutral group | n / expected | Pixel AI | Strict AI-generated real | Faces checked / red | Errors |
|---|---:|---:|---:|---:|---:|
| private/calibration_phone | 104 / 104 | 0 | 0 | 148 / 0 | 0 |
| private/calibration_objects | 100 / 100 | 4 | 0 | 20 / 0 | 0 |

Private calibration budgets apply to the pixel union (5%) and to each edit head separately (2%); the two edit heads' union is not given a separate 2% promise.

| Neutral group / component | n | Flagged | Allowed | Passed after complete run |
|---|---:|---:|---:|---|
| private/calibration_phone / final_pixel_union | 104 | 0 | 5 | True |
| private/calibration_phone / edit1 | 104 | 0 | 2 | True |
| private/calibration_phone / edit2 | 104 | 0 | 2 | True |
| private/calibration_objects / final_pixel_union | 100 | 4 | 5 | True |
| private/calibration_objects / edit1 | 100 | 3 | 2 | False |
| private/calibration_objects / edit2 | 100 | 2 | 2 | True |

## Coverage limits

Exit zero verifies required calibration/provenance checks; it does not establish complete coverage of every optional cache or live inference.

muse_edits_v1_all_app.npz lacks pe_ec3/muse_edits_v1_all_eval.npz; canonical v1 pair exists and this extra cache is not required by saved calibration.

Partial cached coverage: **True**. See cached_metrics.md for each missing/partial set.

Provenance incomplete: overlap counts are findings within documented/available sources, not a clean bill for the exact deployed weights.

Accessible public-file hash check: 0 matching training files among 6064 equal-size candidates. This does not cover inaccessible files, decoded/perceptual duplicates, or all Colab training manifests.

- No aggregate accuracy is computed across overlapping sets.
- Live-versus-cached differences may include float16 boundary effects; partial caches also omit runtime face checks.
- Private groups contain neutral totals only, never private per-image names, IDs, paths, or hashes.
- Meta paid APIs were mocked or disabled. Only one public real-photo basic-detection request was tested on the hosted demo; hosted-wide accuracy, concurrency, and hardware provenance were not validated.
- Manifest coverage is partial; accessible-file exact-byte results do not prove universal absence of leakage.
- Face robustness uses 30 photos per group and simulated phone processing, not a population guarantee.

Rerun: `python -m src.eval.release_summary --directory results/release_2026-11-04`
