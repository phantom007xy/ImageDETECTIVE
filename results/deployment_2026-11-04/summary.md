# Verified hosted release

App: https://136-114-230-97.sslip.io. Hosted checks: **PASS**. Application CI: **PASS**.

The VM matches all 51 frozen source/model/calibration artifacts. The six trained weight files remain private and unchanged; the live-float32 edit calibration was updated through the existing private model repository.

- 15 hosted checks passed: model readiness, original example bytes, four real-model examples, malformed uploads, expired sessions, prompt refusal and the known public real-photo confidence boundary.
- 80/80 known public original judge-file verdicts matched the GPU reference, including their structured AI declarations. This does not mean 100% detection from pixels.
- 4 CPU/GPU face-count/flag differences were observed; see [the per-file record](hosted_checks.json). Final verdicts matched; exact pixel-level score equality is not claimed.
- The known real boundary photo now remains Likely AI-generated. It is still a false alert; the unqualified certainty tier has been removed for pixels.
- No private photos were uploaded and no paid Meta API endpoints were invoked. Live provider success/billing is not established by these checks.
- Local setup passed with the approved artifact bundle, in no-weights check mode, and with the expected clear failure for missing artifacts.

## Evidence

- [Hosted checks](hosted_checks.json)
- [VM artifact verification](vm_verified.json)
- [Setup checks](setup_checks.json)
- [Passing application CI](https://github.com/phantom007xy/ImageDETECTIVE/actions/runs/41836585609)
- [Current workflow runs](https://github.com/phantom007xy/ImageDETECTIVE/actions/workflows/checks.yml)

## Local installation

Follow [Run it yourself](../../README.md#run-it-yourself). The public clone supports synthetic checks; full inference requires approved trained artifacts and gated Meta backbone access. No shared token is included.

## Submission handoff

The creator is handling the video. Record/upload it, add the link, and submit before the deadline. The earlier local report's pending-deployment/CI items describe its original checkpoint and are superseded by this report.

Application revision at verification: `8f64ca6c6350bcbd0d59c4a4923f312bf26f0ab8`. Later documentation-only commits do not change these frozen inference artifacts.
