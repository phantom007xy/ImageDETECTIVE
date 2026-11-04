# Section index verification ? 5 November 2026

**24/24 Chrome checks passed.** Synthetic inputs and mocked CPU detection; no private photos, paid API calls, model downloads or GPU initialization.

About and Image Investigation use the same route-scoped section index. Its 216 px desktop column reserves layout space; narrow shells use an above-content disclosure. The active dot follows measured marker coordinates and the most recently crossed heading. Result-only details are conditional, clicks expand the disclosure, and navigation preserves the route and investigation state.

Covered: both themes, desktop/mobile/zoom widths, left-menu reflow, first/last headings, upward/fast scrolling, marker alignment with wrapped/indented labels, native keyboard access, reduced motion, scroll interruption, independent index scrolling, dynamic entries, loading/replacement states and repeated-route listener/API-call checks.

## Screenshots

- [About, desktop dark](verified/index-about-desktop-dark.png)
- [About, desktop light](verified/index-about-desktop-light.png)
- [About with left navigation open](verified/index-about-sidebar-open.png)
- [About, mobile light](verified/index-about-mobile-light.png)
- [Investigation, empty state](verified/index-investigation-empty-desktop.png)
- [Investigation, completed details](verified/index-investigation-result-desktop.png)
- [Investigation, mobile dark](verified/index-investigation-mobile-dark.png)

[Full measurements](verified/browser_checks.json) ? [Component notes](../../docs/SECTION_INDEX.md)

## Limits

The reference page returned a loading shell through the available reader; implementation follows the provided behavior. Mobile testing uses Chrome viewport emulation. Detector accuracy is not remeasured by these UI checks; previous measured results remain unchanged.
