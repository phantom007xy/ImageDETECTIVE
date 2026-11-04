# Release manifest audit

Provenance incomplete: overlap counts are findings within documented/available sources, not a clean bill for the exact deployed weights.

Inspected 220 public/local metadata artifacts and 101 evaluation artifacts; 404 profile comparisons.
Private artifacts: 25 total, 25 readable; aggregate rows 2343.

## Reconstructed training profiles

- whole_G3tt_reported_lineage: 34 artifacts, 24380 public exact identities, 20125 source families.
- PE_EC3_backbone_and_forward_head_reported_sources: 24 artifacts, 22307 public exact identities, 7769 source families.
- Google_head_candidate_sources_not_exact_provenance: 6 artifacts, 915 public exact identities, 623 source families.
- face_v4_available_local_training_manifests: 4 artifacts, 24085 public exact identities, 9327 source families.

## Overlaps requiring interpretation

Source-family overlap includes intentionally shared captions across generator tests. These counts alone do not establish training-image leakage.

| Evaluation artifact | Training profile | Exact identity overlap | Shared source families |
|---|---|---:|---:|
| features/faces/sid_reals_0_9_unseen_eval.npz | whole_G3tt_reported_lineage | 0 | 3 |
| features/faces/sid_reals_0_9_unseen_eval.npz | Google_head_candidate_sources_not_exact_provenance | 0 | 2 |
| features/lists/open_gen_v1/open_gen_dev.csv | whole_G3tt_reported_lineage | 0 | 357 |
| features/lists/open_gen_v1/open_gen_dev.csv | PE_EC3_backbone_and_forward_head_reported_sources | 0 | 354 |
| features/lists/open_gen_v1/open_gen_dev.csv | Google_head_candidate_sources_not_exact_provenance | 0 | 222 |
| features/open_gen_dev_app.npz | whole_G3tt_reported_lineage | 0 | 357 |
| features/open_gen_dev_app.npz | PE_EC3_backbone_and_forward_head_reported_sources | 0 | 354 |
| features/open_gen_dev_app.npz | Google_head_candidate_sources_not_exact_provenance | 0 | 222 |
| features/sid_reals_0_9_unseen_app.npz | whole_G3tt_reported_lineage | 0 | 3 |
| features/sid_reals_0_9_unseen_app.npz | Google_head_candidate_sources_not_exact_provenance | 0 | 2 |

## Exact byte check

SHA-256: 0 matching training files; 6064/6064 equal-size candidates checked. Complete for accessible public files: True.
Unavailable/non-file references: training 11229, evaluation 7294.

## Identifier collision verification

SID synthetic IDs repeated across train/validation: 18; verified different image bytes: 18; verified identical bytes: 0; unverified: 0. Split-local synthetic IDs are compared with split scope.
Unresolved crop/derivative name matches are reported separately in JSON: 180 comparison-level matches. They are not counted as resolved source-family overlap.

## Recursive backup inclusion

The recursive PE training glob matches 18 archives, including 3 backups. This is a reproducibility/reweighting risk; exact active-run inclusion is unproven.

## Limits

- Weight files do not serialize the exact training manifest list, feature-file hashes, run command, or data revision; profiles are reconstructed from checked-in result notes.
- Current versus backup Google features include data generated after the active EC3g head; neither candidate directory is assumed to be its exact training set.
- Source-family sharing can be intentional: PixArt generator holdout uses training-pair captions. Shared caption content is not an identical training image, but is not an independent-content test.
- Google edit/restore manifests omit original source paths; those derivatives cannot be traced fully to their source photos from manifests alone.
- Ambiguous numbered Commons crops cannot always be traced to v1/v2 originals because the crop filenames omit the original download round.
- Only locally present face-v4 manifests are checked. Missing Colab-only train/RTFS manifests prevent reproducing the complete face identity/near-duplicate exclusion audit.
- Private feature artifacts are counted without serializing paths or IDs. No private cross-artifact or train/eval source-identity leakage assertion is made.
- Many reported dev pools set calibration thresholds or informed model selection; absence of training overlap does not make them independent final tests.
- Exact hashes cover accessible public image files only. No perceptual or identity similarity model is run.

Full per-artifact counts, public examples, source-resolution coverage and provenance references are in manifest_checks.json.
