# Release robustness check

Status: **complete**.

Scope: frozen DINOv3 whole-image check + both PE heads, production joint thresholds. No faces, metadata, private photos, APIs, or recalibration.
Progress: 7245/7245 successful transformed-image checks; 0 errors; 0 remaining.

Counts and percentages use successful checks only; missing/error cases are shown separately. No overall accuracy is computed.

| Group | Kind | Version | Successful/requested | AI | Uncertain | Authentic | Errors |
|---|---|---|---:|---:|---:|---:|---:|
| polished_real | REAL | original | 150/150 | 4 (2.7%) | 35 (23.3%) | 111 (74.0%) | 0 |
| polished_real | REAL | jpeg70 | 150/150 | 3 (2.0%) | 34 (22.7%) | 113 (75.3%) | 0 |
| polished_real | REAL | whatsapp | 150/150 | 3 (2.0%) | 34 (22.7%) | 113 (75.3%) | 0 |
| polished_real | REAL | whatsapp2 | 150/150 | 3 (2.0%) | 35 (23.3%) | 112 (74.7%) | 0 |
| polished_real | REAL | small | 150/150 | 4 (2.7%) | 31 (20.7%) | 115 (76.7%) | 0 |
| polished_real | REAL | screenshot | 150/150 | 4 (2.7%) | 35 (23.3%) | 111 (74.0%) | 0 |
| polished_real | REAL | crop | 150/150 | 7 (4.7%) | 40 (26.7%) | 103 (68.7%) | 0 |
| openimages_real | REAL | original | 150/150 | 1 (0.7%) | 23 (15.3%) | 126 (84.0%) | 0 |
| openimages_real | REAL | jpeg70 | 150/150 | 0 (0.0%) | 18 (12.0%) | 132 (88.0%) | 0 |
| openimages_real | REAL | whatsapp | 150/150 | 0 (0.0%) | 18 (12.0%) | 132 (88.0%) | 0 |
| openimages_real | REAL | whatsapp2 | 150/150 | 0 (0.0%) | 18 (12.0%) | 132 (88.0%) | 0 |
| openimages_real | REAL | small | 150/150 | 0 (0.0%) | 14 (9.3%) | 136 (90.7%) | 0 |
| openimages_real | REAL | screenshot | 150/150 | 1 (0.7%) | 23 (15.3%) | 126 (84.0%) | 0 |
| openimages_real | REAL | crop | 150/150 | 1 (0.7%) | 14 (9.3%) | 135 (90.0%) | 0 |
| muse_v2_originals | REAL | original | 99/99 | 2 (2.0%) | 18 (18.2%) | 79 (79.8%) | 0 |
| muse_v2_originals | REAL | jpeg70 | 99/99 | 1 (1.0%) | 14 (14.1%) | 84 (84.8%) | 0 |
| muse_v2_originals | REAL | whatsapp | 99/99 | 1 (1.0%) | 16 (16.2%) | 82 (82.8%) | 0 |
| muse_v2_originals | REAL | whatsapp2 | 99/99 | 1 (1.0%) | 17 (17.2%) | 81 (81.8%) | 0 |
| muse_v2_originals | REAL | small | 99/99 | 0 (0.0%) | 15 (15.2%) | 84 (84.8%) | 0 |
| muse_v2_originals | REAL | screenshot | 99/99 | 1 (1.0%) | 20 (20.2%) | 78 (78.8%) | 0 |
| muse_v2_originals | REAL | crop | 99/99 | 1 (1.0%) | 19 (19.2%) | 79 (79.8%) | 0 |
| muse_v2_local | AI | original | 132/132 | 94 (71.2%) | 7 (5.3%) | 31 (23.5%) | 0 |
| muse_v2_local | AI | jpeg70 | 132/132 | 82 (62.1%) | 10 (7.6%) | 40 (30.3%) | 0 |
| muse_v2_local | AI | whatsapp | 132/132 | 84 (63.6%) | 11 (8.3%) | 37 (28.0%) | 0 |
| muse_v2_local | AI | whatsapp2 | 132/132 | 82 (62.1%) | 11 (8.3%) | 39 (29.5%) | 0 |
| muse_v2_local | AI | small | 132/132 | 72 (54.5%) | 12 (9.1%) | 48 (36.4%) | 0 |
| muse_v2_local | AI | screenshot | 132/132 | 93 (70.5%) | 8 (6.1%) | 31 (23.5%) | 0 |
| muse_v2_local | AI | crop | 132/132 | 70 (53.0%) | 12 (9.1%) | 50 (37.9%) | 0 |
| muse_v2_whole | AI | original | 66/66 | 58 (87.9%) | 1 (1.5%) | 7 (10.6%) | 0 |
| muse_v2_whole | AI | jpeg70 | 66/66 | 59 (89.4%) | 1 (1.5%) | 6 (9.1%) | 0 |
| muse_v2_whole | AI | whatsapp | 66/66 | 58 (87.9%) | 2 (3.0%) | 6 (9.1%) | 0 |
| muse_v2_whole | AI | whatsapp2 | 66/66 | 58 (87.9%) | 2 (3.0%) | 6 (9.1%) | 0 |
| muse_v2_whole | AI | small | 66/66 | 57 (86.4%) | 3 (4.5%) | 6 (9.1%) | 0 |
| muse_v2_whole | AI | screenshot | 66/66 | 58 (87.9%) | 2 (3.0%) | 6 (9.1%) | 0 |
| muse_v2_whole | AI | crop | 66/66 | 58 (87.9%) | 1 (1.5%) | 7 (10.6%) | 0 |
| muse_v1 | AI | original | 100/100 | 100 (100.0%) | 0 (0.0%) | 0 (0.0%) | 0 |
| muse_v1 | AI | jpeg70 | 100/100 | 100 (100.0%) | 0 (0.0%) | 0 (0.0%) | 0 |
| muse_v1 | AI | whatsapp | 100/100 | 100 (100.0%) | 0 (0.0%) | 0 (0.0%) | 0 |
| muse_v1 | AI | whatsapp2 | 100/100 | 100 (100.0%) | 0 (0.0%) | 0 (0.0%) | 0 |
| muse_v1 | AI | small | 100/100 | 100 (100.0%) | 0 (0.0%) | 0 (0.0%) | 0 |
| muse_v1 | AI | screenshot | 100/100 | 100 (100.0%) | 0 (0.0%) | 0 (0.0%) | 0 |
| muse_v1 | AI | crop | 100/100 | 100 (100.0%) | 0 (0.0%) | 0 (0.0%) | 0 |
| sdxl | AI | original | 150/150 | 142 (94.7%) | 8 (5.3%) | 0 (0.0%) | 0 |
| sdxl | AI | jpeg70 | 150/150 | 132 (88.0%) | 18 (12.0%) | 0 (0.0%) | 0 |
| sdxl | AI | whatsapp | 150/150 | 135 (90.0%) | 15 (10.0%) | 0 (0.0%) | 0 |
| sdxl | AI | whatsapp2 | 150/150 | 135 (90.0%) | 15 (10.0%) | 0 (0.0%) | 0 |
| sdxl | AI | small | 150/150 | 120 (80.0%) | 30 (20.0%) | 0 (0.0%) | 0 |
| sdxl | AI | screenshot | 150/150 | 140 (93.3%) | 10 (6.7%) | 0 (0.0%) | 0 |
| sdxl | AI | crop | 150/150 | 129 (86.0%) | 21 (14.0%) | 0 (0.0%) | 0 |
| pixart | AI | original | 150/150 | 148 (98.7%) | 2 (1.3%) | 0 (0.0%) | 0 |
| pixart | AI | jpeg70 | 150/150 | 143 (95.3%) | 7 (4.7%) | 0 (0.0%) | 0 |
| pixart | AI | whatsapp | 150/150 | 146 (97.3%) | 4 (2.7%) | 0 (0.0%) | 0 |
| pixart | AI | whatsapp2 | 150/150 | 146 (97.3%) | 4 (2.7%) | 0 (0.0%) | 0 |
| pixart | AI | small | 150/150 | 141 (94.0%) | 9 (6.0%) | 0 (0.0%) | 0 |
| pixart | AI | screenshot | 150/150 | 148 (98.7%) | 2 (1.3%) | 0 (0.0%) | 0 |
| pixart | AI | crop | 150/150 | 147 (98.0%) | 2 (1.3%) | 1 (0.7%) | 0 |
| sd35 | AI | original | 38/38 | 20 (52.6%) | 9 (23.7%) | 9 (23.7%) | 0 |
| sd35 | AI | jpeg70 | 38/38 | 16 (42.1%) | 12 (31.6%) | 10 (26.3%) | 0 |
| sd35 | AI | whatsapp | 38/38 | 15 (39.5%) | 13 (34.2%) | 10 (26.3%) | 0 |
| sd35 | AI | whatsapp2 | 38/38 | 16 (42.1%) | 13 (34.2%) | 9 (23.7%) | 0 |
| sd35 | AI | small | 38/38 | 12 (31.6%) | 14 (36.8%) | 12 (31.6%) | 0 |
| sd35 | AI | screenshot | 38/38 | 19 (50.0%) | 10 (26.3%) | 9 (23.7%) | 0 |
| sd35 | AI | crop | 38/38 | 17 (44.7%) | 11 (28.9%) | 10 (26.3%) | 0 |

## Interpretation

- REAL AI flags are false alarms; AI authentic verdicts are misses. Uncertain remains separate.
- All seven transformations reuse the existing VERSIONS implementation. WhatsApp is a JPEG/resize approximation, not an actual WhatsApp service round trip.
- The production joint thresholds and both PE heads are frozen. Faces and metadata are excluded, so these are not full-app verdicts.
- The no-face model uses thresholds calibrated for the full app, without recalibration. The constructor may load the calibrated face PE although no face is found or scored.
- Groups and versions overlap by source; no pooled or overall accuracy is computed.
- Most groups were used during development or calibration. This is a robustness regression, not a newly independent final test.
- Resume binds code, model/calibration bytes, source-list bytes and selected paths. Public source images are assumed unchanged in place; their bytes are not all hashed here.

Muse-v2 originals are deduplicated by original source photo, rather than the repeated original-copy paths in the edit list.
Configuration fingerprint: `d6b4380ebaac9dc9e531a6938e0c25a4dd1ff10574b4f90c17f60fb9a33538a1`.
