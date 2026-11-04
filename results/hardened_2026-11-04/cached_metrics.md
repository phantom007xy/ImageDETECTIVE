# Cached release metrics — 4 November 2026

CPU-only scoring of existing feature caches with the runtime's fixed joint calibration and Detector.verdict; no recalibration, backbone loading, image reads, network, APIs, or GPU.

These are cached reconstructions, not fresh live inference. Existing float16 caches can differ near thresholds. Calibration/dev figures are not independent test guarantees. Rows from overlapping sets must not be pooled into an accuracy.

## Effective fixed thresholds

```json
{
  "ai_logit": 0.6646256446838379,
  "real_logit": -2.7401700019836426,
  "strong_ai_logit": 5.520974636077881,
  "face_ai_logit": 4.044705390930176,
  "face_leans_logit": -1.1909427642822266,
  "swap_ai_logit": 0.22257614135742188,
  "swap_leans_logit": -2.4891319274902344,
  "temperature": 1.0710187451791922,
  "edit_logit": 5.4107747077941895,
  "edit2_logit": 5.403400897979736
}
```

## Image counts

AI counts include every AI verdict. Partial rows cover whole-image and both edit heads only.

Unique counts refer to cached image IDs, not perceptually distinct photos. Different IDs may hold copies of the same original.

| Set | Label | Scope | n / unique IDs | AI | Uncertain | Authentic |
|---|---|---|---:|---:|---:|---:|
| bitmind_nb_dev | ai | partial; faces unavailable | 610 / 610 | 604 (99.02%) | 5 | 1 |
| commons_ai_v1_photo | ai | partial; faces unavailable | 202 / 202 | 160 (79.21%) | 13 | 29 |
| commons_dev | real | full cached | 330 / 330 | 16 (4.85%) | 79 | 235 |
| commons_v3_dev | real | full cached | 32 / 32 | 0 (0.00%) | 5 | 27 |
| commons_v4_dev | real | full cached | 62 / 62 | 2 (3.23%) | 13 | 47 |
| muse_edits_test_v2 | real | partial; faces unavailable | 198 / 198 | 4 (2.02%) | 36 | 158 |
| muse_edits_test_v2 | ai | partial; faces unavailable | 198 / 198 | 152 (76.77%) | 8 | 38 |
| muse_edits_v1_all | — | missing_required_cache | — | — | — | — |
| muse_edits_v1 | real | partial; faces unavailable | 76 / 76 | 2 (2.63%) | 18 | 56 |
| muse_edits_v1 | ai | partial; faces unavailable | 76 / 76 | 53 (69.74%) | 6 | 17 |
| muse_image_test_v1 | ai | full cached | 100 / 100 | 100 (100.00%) | 0 | 0 |
| muse_pairs_val | real | full cached | 76 / 76 | 1 (1.32%) | 12 | 63 |
| muse_pairs_val | ai | full cached | 76 / 76 | 75 (98.68%) | 0 | 1 |
| open_gen_dev | ai | partial; faces unavailable | 511 / 511 | 483 (94.52%) | 18 | 10 |
| real_face_crops_dev_all | real | full cached | 150 / 149 | 6 (4.00%) | 49 | 95 |
| real_face_crops_dev | real | partial; faces unavailable | 129 / 128 | 6 (4.65%) | 44 | 79 |
| real_face_crops_dev_v4 | real | partial; faces unavailable | 21 / 21 | 0 (0.00%) | 5 | 16 |
| sd_faces | ai | full cached | 996 / 996 | 974 (97.79%) | 22 | 0 |
| sfhq_pt3_test | ai | partial; faces unavailable | 375 / 375 | 345 (92.00%) | 30 | 0 |
| sid_reals_0_9_unseen | real | full cached | 576 / 576 | 11 (1.91%) | 82 | 483 |
| tpdne_test | ai | full cached | 244 / 244 | 233 (95.49%) | 11 | 0 |
| google/gemini_dev_v1 | ai | partial; faces unavailable | 30 / 30 | 27 (90.00%) | 0 | 3 |
| google/open_gen_dev | ai | partial; faces unavailable | 67 / 67 | 48 (71.64%) | 14 | 5 |
| private: phone | real | full cached | 104 / 104 | 0 (0.00%) | 10 | 94 |
| private: phone objects | real | full cached | 100 / 100 | 3 (3.00%) | 5 | 92 |
| private: training selfies (reused) | real | partial; faces unavailable | 30 / 30 | 0 (0.00%) | 0 | 30 |
| private: training person edits (reused) | real | partial; faces unavailable | 60 / 60 | 0 (0.00%) | 0 | 60 |
| private: training person edits (reused) | ai | partial; faces unavailable | 60 / 60 | 56 (93.33%) | 0 | 4 |

## Public source counts

| Set / source | n | AI | Uncertain | Authentic |
|---|---:|---:|---:|---:|
| bitmind_nb_dev / bitmind/nano_banana / ai | 610 | 604 (99.02%) | 5 | 1 |
| commons_ai_v1_photo / commons_ai/bing_dalle3 / ai | 14 | 13 (92.86%) | 1 | 0 |
| commons_ai_v1_photo / commons_ai/chatgpt / ai | 19 | 18 (94.74%) | 1 | 0 |
| commons_ai_v1_photo / commons_ai/dalle / ai | 32 | 25 (78.12%) | 3 | 4 |
| commons_ai_v1_photo / commons_ai/firefly / ai | 2 | 1 (50.00%) | 1 | 0 |
| commons_ai_v1_photo / commons_ai/flux / ai | 7 | 6 (85.71%) | 0 | 1 |
| commons_ai_v1_photo / commons_ai/flux/ai_restored / ai | 1 | 0 (0.00%) | 0 | 1 |
| commons_ai_v1_photo / commons_ai/gemini / ai | 19 | 16 (84.21%) | 1 | 2 |
| commons_ai_v1_photo / commons_ai/gemini/ai_restored / ai | 30 | 8 (26.67%) | 2 | 20 |
| commons_ai_v1_photo / commons_ai/grok / ai | 11 | 11 (100.00%) | 0 | 0 |
| commons_ai_v1_photo / commons_ai/ideogram / ai | 2 | 1 (50.00%) | 1 | 0 |
| commons_ai_v1_photo / commons_ai/imagen / ai | 2 | 2 (100.00%) | 0 | 0 |
| commons_ai_v1_photo / commons_ai/mai_image / ai | 1 | 1 (100.00%) | 0 | 0 |
| commons_ai_v1_photo / commons_ai/meta_ai / ai | 3 | 3 (100.00%) | 0 | 0 |
| commons_ai_v1_photo / commons_ai/midjourney / ai | 18 | 14 (77.78%) | 3 | 1 |
| commons_ai_v1_photo / commons_ai/nano_banana / ai | 4 | 4 (100.00%) | 0 | 0 |
| commons_ai_v1_photo / commons_ai/reve / ai | 1 | 1 (100.00%) | 0 | 0 |
| commons_ai_v1_photo / commons_ai/stable_diffusion / ai | 33 | 33 (100.00%) | 0 | 0 |
| commons_ai_v1_photo / commons_ai/stylegan / ai | 3 | 3 (100.00%) | 0 | 0 |
| commons_dev / commons/people / real | 30 | 1 (3.33%) | 8 | 21 |
| commons_dev / commons/phone / real | 54 | 3 (5.56%) | 11 | 40 |
| commons_dev / commons/portrait / real | 209 | 10 (4.78%) | 52 | 147 |
| commons_dev / commons/quality / real | 37 | 2 (5.41%) | 8 | 27 |
| commons_v3_dev / commons/phone / real | 31 | 0 (0.00%) | 5 | 26 |
| commons_v3_dev / commons/quality / real | 1 | 0 (0.00%) | 0 | 1 |
| commons_v4_dev / commons/portrait / real | 48 | 2 (4.17%) | 12 | 34 |
| commons_v4_dev / commons/selfie / real | 14 | 0 (0.00%) | 1 | 13 |
| muse_edits_test_v2 / muse_edits/local / ai | 132 | 94 (71.21%) | 7 | 31 |
| muse_edits_test_v2 / muse_edits/original / real | 198 | 4 (2.02%) | 36 | 158 |
| muse_edits_test_v2 / muse_edits/whole / ai | 66 | 58 (87.88%) | 1 | 7 |
| muse_edits_v1 / muse_edits/local / ai | 51 | 30 (58.82%) | 6 | 15 |
| muse_edits_v1 / muse_edits/original / real | 76 | 2 (2.63%) | 18 | 56 |
| muse_edits_v1 / muse_edits/whole / ai | 25 | 23 (92.00%) | 0 | 2 |
| muse_image_test_v1 / muse_image/animal / ai | 14 | 14 (100.00%) | 0 | 0 |
| muse_image_test_v1 / muse_image/object_food / ai | 13 | 13 (100.00%) | 0 | 0 |
| muse_image_test_v1 / muse_image/people/candid / ai | 12 | 12 (100.00%) | 0 | 0 |
| muse_image_test_v1 / muse_image/people/group / ai | 12 | 12 (100.00%) | 0 | 0 |
| muse_image_test_v1 / muse_image/people/indoor_lowlight / ai | 12 | 12 (100.00%) | 0 | 0 |
| muse_image_test_v1 / muse_image/people/portrait / ai | 12 | 12 (100.00%) | 0 | 0 |
| muse_image_test_v1 / muse_image/people/selfie / ai | 12 | 12 (100.00%) | 0 | 0 |
| muse_image_test_v1 / muse_image/scene / ai | 13 | 13 (100.00%) | 0 | 0 |
| muse_pairs_val / muse_pairs/muse / ai | 76 | 75 (98.68%) | 0 | 1 |
| muse_pairs_val / muse_pairs/real / real | 76 | 1 (1.32%) | 12 | 63 |
| open_gen_dev / open_gen/k22_devcaption / ai | 77 | 75 (97.40%) | 2 | 0 |
| open_gen_dev / open_gen/pixart / ai | 396 | 388 (97.98%) | 7 | 1 |
| open_gen_dev / open_gen/sd35m_devcaption / ai | 38 | 20 (52.63%) | 9 | 9 |
| real_face_crops_dev_all / real_face_crops/commons/people / real | 8 | 1 (12.50%) | 3 | 4 |
| real_face_crops_dev_all / real_face_crops/commons/phone / real | 1 | 0 (0.00%) | 1 | 0 |
| real_face_crops_dev_all / real_face_crops/commons/portrait / real | 91 | 5 (5.49%) | 29 | 57 |
| real_face_crops_dev_all / real_face_crops/commons/quality / real | 5 | 0 (0.00%) | 3 | 2 |
| real_face_crops_dev_all / real_face_crops/commons/selfie / real | 7 | 0 (0.00%) | 2 | 5 |
| real_face_crops_dev_all / real_face_crops/sid_set/openimages / real | 38 | 0 (0.00%) | 11 | 27 |
| real_face_crops_dev / real_face_crops/commons/people / real | 8 | 1 (12.50%) | 3 | 4 |
| real_face_crops_dev / real_face_crops/commons/phone / real | 1 | 0 (0.00%) | 1 | 0 |
| real_face_crops_dev / real_face_crops/commons/portrait / real | 77 | 5 (6.49%) | 26 | 46 |
| real_face_crops_dev / real_face_crops/commons/quality / real | 5 | 0 (0.00%) | 3 | 2 |
| real_face_crops_dev / real_face_crops/sid_set/openimages / real | 38 | 0 (0.00%) | 11 | 27 |
| real_face_crops_dev_v4 / real_face_crops/commons/portrait / real | 14 | 0 (0.00%) | 3 | 11 |
| real_face_crops_dev_v4 / real_face_crops/commons/selfie / real | 7 | 0 (0.00%) | 2 | 5 |
| sd_faces / sd_faces/sd1.5 / ai | 332 | 328 (98.80%) | 4 | 0 |
| sd_faces / sd_faces/sd2.1 / ai | 332 | 329 (99.10%) | 3 | 0 |
| sd_faces / sd_faces/sdxl / ai | 332 | 317 (95.48%) | 15 | 0 |
| sfhq_pt3_test / sfhq/pt3 / ai | 375 | 345 (92.00%) | 30 | 0 |
| sid_reals_0_9_unseen / sid_set/openimages / real | 576 | 11 (1.91%) | 82 | 483 |
| tpdne_test / gan/tpdne / ai | 244 | 233 (95.49%) | 11 | 0 |
| google/gemini_dev_v1 / gemini_api/3.1-flash-image / ai | 30 | 27 (90.00%) | 0 | 3 |
| google/open_gen_dev / open_gen/gemini-2.5-flash-image_devcaption / ai | 16 | 13 (81.25%) | 3 | 0 |
| google/open_gen_dev / open_gen/gemini-3-pro-image_devcaption / ai | 14 | 10 (71.43%) | 1 | 3 |
| google/open_gen_dev / open_gen/gemini-3.1-flash-image_devcaption / ai | 37 | 25 (67.57%) | 10 | 2 |

## Per-face counts at the app thresholds

Leans-or-strict includes red faces; leans-only is orange. Face labels can mean fully generated faces or swaps, depending on the source.

| Face set | Check / label | n faces | Strict | Leans or strict | Unique image IDs |
|---|---|---:|---:|---:|---:|
| private: phone | ai_made / real | 148 | 0 | 4 | 95 |
| private: phone | swap / real | 148 | 0 | 2 | 95 |
| private: phone objects | ai_made / real | 20 | 0 | 1 | 14 |
| private: phone objects | swap / real | 20 | 0 | 0 | 14 |
| private: training selfies (reused) | ai_made / real | 34 | 0 | 0 | 30 |
| chameleon_random_eval | ai_made / real | 398 | 3 | 35 | 310 |
| chameleon_random_eval | ai_made / ai_or_swap | 607 | 173 | 341 | 575 |
| chameleon_size_matched_eval | ai_made / real | 381 | 9 | 47 | 308 |
| chameleon_size_matched_eval | ai_made / ai_or_swap | 277 | 112 | 172 | 237 |
| commons_dev_eval | ai_made / real | 285 | 1 | 14 | 248 |
| commons_dev_eval | swap / real | 285 | 0 | 1 | 248 |
| commons_v3_dev_eval | ai_made / real | 15 | 0 | 0 | 11 |
| commons_v3_dev_eval | swap / real | 15 | 0 | 0 | 11 |
| commons_v4_dev_eval | ai_made / real | 78 | 0 | 3 | 55 |
| commons_v4_dev_eval | swap / real | 78 | 0 | 0 | 55 |
| dfdc_part3_eval | ai_made / real | 1452 | 0 | 0 | 1452 |
| dfdc_part3_eval | ai_made / ai_or_swap | 1455 | 0 | 0 | 1455 |
| dfdc_part3_eval | swap / real | 1452 | 7 | 72 | 1452 |
| dfdc_part3_eval | swap / ai_or_swap | 1455 | 560 | 995 | 1455 |
| muse_image_test_v1_eval | ai_made / ai_or_swap | 184 | 31 | 105 | 75 |
| muse_image_test_v1_eval | swap / ai_or_swap | 184 | 0 | 2 | 75 |
| muse_pairs_val_eval | ai_made / real | 81 | 0 | 3 | 41 |
| muse_pairs_val_eval | ai_made / ai_or_swap | 98 | 14 | 47 | 49 |
| muse_pairs_val_eval | swap / real | 81 | 0 | 0 | 41 |
| muse_pairs_val_eval | swap / ai_or_swap | 98 | 0 | 0 | 49 |
| openforensics_test_dev_eval | ai_made / real | 1531 | 1 | 69 | 1531 |
| openforensics_test_dev_eval | ai_made / ai_or_swap | 2270 | 3 | 149 | 2270 |
| openforensics_test_dev_eval | swap / real | 1531 | 4 | 12 | 1531 |
| openforensics_test_dev_eval | swap / ai_or_swap | 2270 | 2199 | 2251 | 2270 |
| real_face_crops_dev_all_eval | ai_made / real | 174 | 0 | 8 | 147 |
| real_face_crops_dev_all_eval | swap / real | 174 | 0 | 1 | 147 |
| real_face_crops_dev_eval | ai_made / real | 149 | 0 | 8 | 126 |
| real_face_crops_dev_v4_eval | ai_made / real | 25 | 0 | 0 | 21 |
| sd_faces_eval | ai_made / ai_or_swap | 1006 | 305 | 747 | 996 |
| sd_faces_eval | swap / ai_or_swap | 1006 | 45 | 146 | 996 |
| sid_reals_0_9_unseen_eval | ai_made / real | 557 | 0 | 8 | 290 |
| sid_reals_0_9_unseen_eval | swap / real | 557 | 0 | 1 | 290 |
| tpdne_test_eval | ai_made / ai_or_swap | 267 | 36 | 185 | 244 |
| tpdne_test_eval | swap / ai_or_swap | 267 | 0 | 1 | 244 |
| validation-00000-of-00034_eval | ai_made / real | 292 | 1 | 3 | 148 |
| validation-00000-of-00034_eval | ai_made / ai_or_swap | 252 | 172 | 204 | 141 |
| validation-00000-of-00034_eval | swap / real | 292 | 1 | 1 | 148 |
| validation-00000-of-00034_eval | swap / ai_or_swap | 252 | 0 | 0 | 141 |
| wilddeepfake_eval | ai_made / real | 3168 | 4 | 191 | 3168 |
| wilddeepfake_eval | ai_made / ai_or_swap | 3280 | 17 | 191 | 3280 |
| wilddeepfake_eval | swap / real | 3168 | 19 | 83 | 3168 |
| wilddeepfake_eval | swap / ai_or_swap | 3280 | 450 | 1062 | 3280 |
| v4 crops: dfdc_p3_dev | swap / real | 1452 | 7 | 72 | 1452 |
| v4 crops: dfdc_p3_dev | swap / ai_or_swap | 1455 | 560 | 995 | 1455 |
| v4 crops: of_test_dev | swap / real | 1531 | 4 | 12 | 1531 |
| v4 crops: of_test_dev | swap / ai_or_swap | 2270 | 2199 | 2251 | 2270 |
| v4 crops: of_val_dev | swap / real | 656 | 2 | 10 | 656 |
| v4 crops: of_val_dev | swap / ai_or_swap | 1459 | 1410 | 1442 | 1459 |
| v4 crops: pool_commons_dev | swap / real | 285 | 0 | 1 | 285 |
| v4 crops: pool_commons_v3_dev | swap / real | 15 | 0 | 0 | 15 |
| v4 crops: pool_commons_v4_dev | swap / real | 78 | 0 | 0 | 78 |
| v4 crops: pool_muse_image_test_v1 | swap / ai_or_swap | 184 | 0 | 2 | 184 |
| v4 crops: pool_muse_pairs_val | swap / real | 81 | 0 | 0 | 81 |
| v4 crops: pool_muse_pairs_val | swap / ai_or_swap | 98 | 0 | 0 | 98 |
| v4 crops: pool_real_face_crops_dev_all | swap / real | 174 | 0 | 1 | 174 |
| v4 crops: pool_sd_faces | swap / ai_or_swap | 1006 | 45 | 146 | 1006 |
| v4 crops: pool_sid_reals_0_9_unseen | swap / real | 557 | 0 | 1 | 557 |
| v4 crops: pool_sid_val | swap / real | 292 | 1 | 1 | 292 |
| v4 crops: pool_tpdne_test | swap / ai_or_swap | 267 | 0 | 1 | 267 |
| v4 crops: wilddeepfake | swap / real | 3168 | 19 | 83 | 3168 |
| v4 crops: wilddeepfake | swap / ai_or_swap | 3280 | 450 | 1062 | 3280 |

## Calibration coverage

- polished dev: verified; expected 330 real rows, observed 330.
- OpenImages val: verified; expected 76 real rows, observed 76.
- originals v1: verified; expected 76 real rows, observed 76.
- originals v2: verified; expected 198 real rows, observed 198.
- OpenImages unseen: verified; expected 576 real rows, observed 576.
- Commons v3 dev: verified; expected 32 real rows, observed 32.
- close-ups: verified; expected 150 real rows, observed 150.
- selfies+portraits: verified; expected 62 real rows, observed 62.
- phone: verified; expected 104 real rows, observed 104.
- phone objects: verified; expected 100 real rows, observed 100.

## Missing or non-independent evaluations

- judge_test_ai_v1: no_current_full_pipeline_caches; run live inference. release regression; observed during earlier development
- muse_image_test_v2: no_current_full_pipeline_caches; run live inference. fresh Muse evaluation; use live inference if caches absent
- chameleon_random: no_current_full_pipeline_caches; run live inference. external evaluation; previously inspected during development
- chameleon_size_matched: no_current_full_pipeline_caches; run live inference. external size-matched evaluation; previously inspected
- rtfs_test: no_current_full_pipeline_caches; run live inference. unseen identities; original Colab results are not a reconstruction at the final app line
- person-edit cross-validation: training-reused person cache is not an independent test. Earlier folds used the same person's sessions; fresh edited selfies are still needed.
- forwarding and phone processing: not represented by original-view caches. Use live release robustness artifacts; cached clean metrics cannot establish transformation robustness.

## Face-cache provenance

- commons_dev_eval: exact_match; current-v4 source pool_commons_dev.
- muse_pairs_val_eval: exact_match; current-v4 source pool_muse_pairs_val.
- sid_reals_0_9_unseen_eval: exact_match; current-v4 source pool_sid_reals_0_9_unseen.
- commons_v3_dev_eval: exact_match; current-v4 source pool_commons_v3_dev.
- real_face_crops_dev_all_eval: exact_match; current-v4 source pool_real_face_crops_dev_all.
- commons_v4_dev_eval: exact_match; current-v4 source pool_commons_v4_dev.
- validation-00000-of-00034_eval: exact_match; current-v4 source pool_sid_val.
- dfdc_part3_eval: exact_match; current-v4 source dfdc_p3_dev.
- openforensics_test_dev_eval: exact_match; current-v4 source of_test_dev.
- wilddeepfake_eval: exact_match; current-v4 source wilddeepfake.
- muse_image_test_v1_eval: exact_match; current-v4 source pool_muse_image_test_v1.
- sd_faces_eval: exact_match; current-v4 source pool_sd_faces.
- tpdne_test_eval: exact_match; current-v4 source pool_tpdne_test.

No metadata labels, API output, uploaded pixels, or private per-photo identifiers appear in this report.
