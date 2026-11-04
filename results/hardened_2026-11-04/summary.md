# Hardened local release

Local gates: **PASS**. Hosted deployment: **pending verification**.

Production code, model weights and calibration are bound to the frozen configuration below. No paid APIs were called by these checks.

| Check | Result |
|---|---|
| frozen artifacts and syntax | PASS |
| private inputs not published | PASS |
| pinned dependencies | PASS |
| cpu behavior | PASS |
| chrome interactions | PASS |
| complete full detector | PASS |
| http equivalence | PASS |
| fragmented metadata http | PASS |
| runtime real calibration | PASS |
| private live calibration | PASS |
| transformation checks | PASS |
| cache provenance | PASS |
| saved calibration mix and face caps | PASS |
| pixel certainty boundary removed | PASS |
| real socket public examples and api fallback | PASS |

Full detector: 3951 dataset entries across 19 cases, 0 errors; overlapping sources are not independent photos.
HTTP upload equivalence: 80/80, plus 2 real fragmented-C2PA JPEGs. CPU checks: 42. Chrome interactions: 13.
Transformations: 7245/7245 checks, without faces or metadata.

After the JPEG metadata-only fix, all original file declarations were reparsed. Pixel/face/edit scores reuse the completed GPU run with byte-identical scoring artifacts; HTTP equivalence was rerun with real models. See metadata_refresh.json for the preserved run and configuration lineage.

The 19 full-detector cases retain their original AI-flag/file-declaration totals.

Runtime calibration fixes private object edit-head false alarms from 3/100 to 2/100; combined alerts fall from 4/100 to 3/100. The separate 30-photo object regression remains 3/30.

Pixel-only results never use the unqualified AI-generated tier. Structured AI declarations are marked unverified. PE contribution maps cover only their analysed centre square.

## Remaining before submission

- Deploy the reviewed code AND recalibrated private edit JSON together; do not expose weights.
- Verify hosted /healthz, public samples, CPU boundary case and API fallback after deployment.
- GitHub Actions has not run remotely for these unpushed changes.
- Record and link the submission video; README still has a placeholder.

## Limits

- Calibration and reused regression pools are not independent tests; entries overlap.
- The CPU boundary check tests one photo and the whole-image branch only.
- Paid Meta services were mocked/offline; actual remote success and billing were not retested.
- No new model training; hard cases such as Nano Banana 2.1, crops and video swaps remain.
- Training provenance is partial: original Colab face manifests and some derivative-source records are missing; see the baseline manifest audit.
- Earlier face robustness and training-manifest audit remain baseline artifacts; not rerun here.

## Artifacts

- [Full detector](live_metrics.json)
- [Transformation details](robustness.md)
- [Runtime calibration](runtime_calibration.json)
- [CPU behavior](app_checks.json)
- [Chrome interactions](browser_checks.json)
- [Privacy](privacy_check.json)
- [Integrity](integrity_check.json)
- [Dependencies](environment.json)

Configuration: `3bd80a815a518a8ece21e75002249c172406c289f8c4ed4db6bb84182b89d412`.
