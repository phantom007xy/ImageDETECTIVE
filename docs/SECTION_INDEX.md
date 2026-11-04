# Section navigation

Added 5 November 2026 to the existing HTML/CSS/JavaScript frontend. Detector models, API requests and investigation data are unchanged.

`app/web/section-index.js` is a reusable route-scoped component. It indexes the active page's actual h2/h3 headings and the existing Investigation details summary. About's heading IDs are explicit and stable. The result-only entry is added after a report exists and removed on image replacement.

The document is the scrolling container. Selection follows headings crossing a reading line beneath the sticky header, with a small dead band to prevent fractional-pixel flicker. First-section and page-bottom cases are handled separately. The circle and connecting path use measured rendered dot coordinates, including wrapped labels and indented subsections. Circle movement takes 180 ms and is disabled with reduced motion.

The right column reserves 216 px. A container query switches to a compact disclosure when the shell has less than 1040 px available, including when the left menu opens. The list can scroll independently. Revealing its active entry changes only its own scrollTop.

Clicking an entry expands a collapsed target disclosure and scrolls without changing the router hash or history. User wheel, touch and scroll-key input can interrupt the movement. Ordinary scroll updates do not move keyboard focus. Each active route owns an AbortController and ResizeObserver; leaving the route removes its listeners and observers. There is no polling or broad mutation observer. Result rendering, image loads, native disclosures, font loads and resizing request targeted refreshes through requestAnimationFrame.

## Verification

Run `python -m src.eval.release_browser_check --output-dir results/section_index_2026-11-05/verified` with Chrome and Node installed. This uses synthetic images, mocked CPU detection and blocked paid/external services; it does not allocate GPU model memory.

The Chrome suite covers existing uploads/image fitting plus both routes, both themes, desktop/mobile/zoom-sized layouts, left-menu reflow, heading order, first/last selection, fast/upward scrolling, wrapped markers, dynamic result entries, collapsed details, preserved upload/report state, keyboard access, reduced motion, interruption and repeated-route listener cleanup. Screenshots and measurements are saved in the output directory.

The reference event page returned only a loading shell through the available reader, so the implementation follows the supplied behavior. Mobile verification uses Chrome viewport emulation rather than physical devices. Original detector accuracy reports remain the earlier measured results.
