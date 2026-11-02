// ImageDETECTIVE page logic: navigation, theme, upload, the investigation and its report. No build step.
(() => {
  "use strict";
  const $ = (id) => document.getElementById(id);
  const MAX_BYTES = 20 * 1024 * 1024;
  const TYPES = ["image/jpeg", "image/png", "image/webp"];
  const EXTENSIONS = /\.(jpe?g|png|webp)$/i;
  const root = document.documentElement;

  // ---------- theme (dark for first-time visitors; the choice is remembered in this browser) ----------
  const themeButton = $("theme-toggle"), themeLabel = $("theme-label");
  function applyTheme(theme, save) {
    root.dataset.theme = theme;
    const action = theme === "dark" ? "Light mode" : "Dark mode"; // the label says what a click will do
    themeButton.title = action;
    themeButton.setAttribute("aria-label", action);
    themeLabel.textContent = action;
    if (save) { try { localStorage.setItem("idetective-theme", theme); } catch (e) { /* private window: not saved */ } }
  }
  applyTheme(root.dataset.theme === "light" ? "light" : "dark", false);
  themeButton.addEventListener("click", () => applyTheme(root.dataset.theme === "dark" ? "light" : "dark", true));

  // ---------- sidebar ----------
  const sidebar = $("sidebar"), toggle = $("sidebar-toggle"), scrim = $("scrim");
  function setSidebar(open) {
    sidebar.classList.toggle("open", open);
    toggle.setAttribute("aria-expanded", String(open));
    scrim.hidden = !open;
  }
  toggle.addEventListener("click", () => setSidebar(!sidebar.classList.contains("open")));
  scrim.addEventListener("click", () => setSidebar(false));
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape" && sidebar.classList.contains("open")) { setSidebar(false); toggle.focus(); }
  });

  // ---------- pages (#/investigate, #/fool, #/about) ----------
  const pages = document.querySelectorAll(".page");
  const navItems = document.querySelectorAll(".side-item[data-page]");
  const TITLES = { investigate: "ImageDETECTIVE", fool: "Try to fool it · ImageDETECTIVE", about: "About · ImageDETECTIVE" };
  function showPage() {
    const match = location.hash.match(/^#\/(investigate|fool|about)\b/);
    const page = match ? match[1] : "investigate";
    pages.forEach((p) => { p.hidden = p.dataset.page !== page; });
    navItems.forEach((a) => {
      if (a.dataset.page === page) a.setAttribute("aria-current", "page"); else a.removeAttribute("aria-current");
    });
    document.title = TITLES[page];
    setSidebar(false);
    return page;
  }
  window.addEventListener("hashchange", () => { showPage(); window.scrollTo(0, 0); $("main").focus({ preventScroll: true }); });
  showPage();

  // ---------- steps ----------
  document.querySelectorAll(".step-toggle").forEach((button) => {
    button.addEventListener("click", () => {
      const open = button.getAttribute("aria-expanded") !== "true";
      button.setAttribute("aria-expanded", String(open));
      $(button.getAttribute("aria-controls")).hidden = !open;
    });
  });

  // ---------- helpers ----------
  function showError(el, text) { el.textContent = text; el.hidden = false; }
  function hideError(el) { el.textContent = ""; el.hidden = true; }
  function pct(x) { return `${Math.round(x * 100)}%`; }
  async function postJSON(url, options) {
    let response;
    try { response = await fetch(url, Object.assign({ method: "POST" }, options)); }
    catch (e) {
      if (e.name === "AbortError") throw e;
      throw new Error("Could not reach the ImageDETECTIVE server. Please check your connection and try again.");
    }
    const body = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(body.detail || "Something went wrong on the server. Please try again.");
    return body;
  }
  function setTabs(evidenceTab, originalTab, img, sources, which) {
    const evidence = which === "evidence";
    evidenceTab.setAttribute("aria-selected", String(evidence));
    originalTab.setAttribute("aria-selected", String(!evidence));
    evidenceTab.tabIndex = evidence ? 0 : -1;
    originalTab.tabIndex = evidence ? -1 : 0;
    img.src = evidence ? sources.evidence : sources.original;
    img.alt = evidence ? "Evidence view: the picture with the detector's signals and face boxes"
                       : "The original picture, unchanged";
  }
  function wireTabs(evidenceTab, originalTab, img, sources) {
    evidenceTab.addEventListener("click", () => setTabs(evidenceTab, originalTab, img, sources, "evidence"));
    originalTab.addEventListener("click", () => setTabs(evidenceTab, originalTab, img, sources, "original"));
    [evidenceTab, originalTab].forEach((tab) => tab.addEventListener("keydown", (e) => {
      if (e.key === "ArrowLeft" || e.key === "ArrowRight") {
        const next = tab === evidenceTab ? originalTab : evidenceTab;
        next.click();
        next.focus();
      }
    }));
  }
  function fillVerdict(card, report) {
    card.className = `verdict tone-${report.tone}`;
    card.querySelector(".verdict-title").textContent = report.verdict;
    card.querySelector(".verdict-reason").textContent = report.reason;
    const context = card.querySelector(".verdict-context");
    context.textContent = report.context;
    context.hidden = !report.context;
    card.querySelector(".meter > div").style.width = pct(report.probability_ai);
    card.querySelector(".meter-label").textContent = `Whole-image check: ${pct(report.probability_ai)} AI evidence`
      + (report.calibrated ? " (calibrated)" : " (not calibrated yet)");
  }

  // ---------- Image Investigation ----------
  const drop = $("drop"), dropEmpty = $("drop-empty"), preview = $("preview"), input = $("file-input");
  const fileRow = $("file-row"), fileName = $("file-name"), uploadError = $("upload-error");
  const go = $("go"), goText = $("go-text");
  const reportEmpty = $("report-empty"), reportError = $("report-error"), reportBody = $("report-body");
  const viewerImg = $("viewer-img"), outlineNote = $("outline-note"), detailsList = $("details-list");
  const sources = { evidence: "", original: "" };
  let file = null, previewUrl = null, token = 0, controller = null;
  wireTabs($("tab-evidence"), $("tab-original"), viewerImg, sources);

  function setGo(text, busy) {
    goText.textContent = text;
    go.disabled = busy || !file;
    if (busy) go.setAttribute("aria-busy", "true"); else go.removeAttribute("aria-busy");
  }
  function clearReport() {
    reportBody.hidden = true;
    hideError(reportError);
    reportEmpty.hidden = false;
    outlineNote.hidden = true;
    detailsList.innerHTML = "";
  }
  function choose(f) {
    hideError(uploadError);
    if (!f) return;
    if (!(TYPES.includes(f.type) || EXTENSIONS.test(f.name))) { showError(uploadError, "Please choose a JPG, PNG or WEBP image."); return; }
    if (f.size > MAX_BYTES) { showError(uploadError, "This file is larger than 20 MB. Please choose a smaller image."); return; }
    if (f.size === 0) { showError(uploadError, "This file is empty."); return; }
    token += 1; // a new picture: any running or finished investigation belongs to the old one
    if (controller) controller.abort();
    clearReport();
    file = f;
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    previewUrl = URL.createObjectURL(f); // the original file itself: nothing is re-encoded in the browser
    preview.onerror = () => {
      showError(uploadError, "This browser can't display this image; it may be damaged. Please choose another file.");
      file = null;
      setGo("Start investigation", false);
      preview.hidden = true;
      dropEmpty.hidden = false;
    };
    preview.src = previewUrl;
    preview.hidden = false;
    dropEmpty.hidden = true;
    fileName.textContent = f.name;
    fileRow.hidden = false;
    setGo("Start investigation", false);
  }
  drop.addEventListener("click", () => input.click());
  drop.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); input.click(); } });
  $("replace").addEventListener("click", () => input.click());
  input.addEventListener("change", () => { choose(input.files[0]); input.value = ""; });
  ["dragenter", "dragover"].forEach((t) => drop.addEventListener(t, (e) => { e.preventDefault(); drop.classList.add("dragging"); }));
  ["dragleave", "dragend"].forEach((t) => drop.addEventListener(t, () => drop.classList.remove("dragging")));
  drop.addEventListener("drop", (e) => {
    e.preventDefault();
    drop.classList.remove("dragging");
    if (e.dataTransfer.files.length) choose(e.dataTransfer.files[0]);
  });
  // a file dropped outside the box must not open in the tab (and lose the page)
  window.addEventListener("dragover", (e) => e.preventDefault());
  window.addEventListener("drop", (e) => e.preventDefault());

  function detail(term, state, text) {
    const dt = document.createElement("dt"), dd = document.createElement("dd"), tag = document.createElement("span");
    dt.textContent = term;
    tag.className = `state state-${state.toLowerCase().replace(/[^a-z]+/g, "-")}`;
    tag.textContent = state;
    dd.append(tag, document.createTextNode(text));
    detailsList.append(dt, dd);
    return dd;
  }
  const FACE_WORDS = { ai: "looks AI-made or swapped", leans_ai: "some signs, not enough on its own", clear: "no clear signs" };
  function renderDetails(r, more) {
    detailsList.innerHTML = "";
    detail("Whole-image analysis · DINOv3", "Ran", `${pct(r.probability_ai)} AI evidence` + (r.calibrated ? " (calibrated)." : "."));
    if (!r.edit) detail("AI-edit analysis · Perception Encoder", "Unavailable", "This check is switched off on this server.");
    else detail("AI-edit analysis · Perception Encoder", "Ran", r.edit.flagged ? "Signs that an AI image model re-rendered (edited) the picture."
      : r.edit.flagged2 ? "The pixels look rendered by an AI image model." : "No clear signs of an AI edit.");
    const found = r.faces.length, checked = r.faces_checked;
    if (!found) detail("Face analysis · SAM 3", "Not applicable", "No face found.");
    else if (!checked) detail("Face analysis · SAM 3", "Not applicable", `${found} face${found > 1 ? "s" : ""} found, all too small to check (under 64 px).`);
    else {
      const parts = r.faces.map((f) => `${f.n}: ${f.too_small ? "too small to check" : (FACE_WORDS[f.level] || "checked")
        + (f.kind === "swapped" ? " (face-swap check)" : "")}`);
      detail("Face analysis · SAM 3", "Ran", `${checked} of ${found} checked; ${r.faces_flagged} flagged. ${parts.join("; ")}.`);
    }
    if (!r.metadata.read) detail("File metadata", "Unavailable", "The file's metadata could not be read.");
    else if (r.metadata.ai_label) detail("File metadata", "Ran", r.metadata.text);
    else if (r.metadata.text) detail("File metadata", "Ran", `${r.metadata.text} No AI label.`);
    else detail("File metadata", "Ran", "No AI label in the file (forwarded copies lose it, so this proves nothing).");
    const investigatorRow = detail("Investigator · Muse Spark + SAM 3.1", more ? more.investigatorState : "Running",
      more ? more.investigatorText : "Asking Meta's investigator…");
    const explanationRow = detail("Explanation · Muse Spark", more ? more.explanationState : "Running",
      more ? more.explanationText : "Writing a short explanation from the numbers…");
    detail("Processing time", "Ran", `${r.seconds} s for the detectors.`);
    return { investigatorRow, explanationRow };
  }
  function moreText(m) {
    let investigatorState = "Ran", investigatorText;
    if (m.investigator === "not run") { investigatorState = "Not applicable"; investigatorText = "Not run: the picture looks authentic, and outlining a real photo would mislead."; }
    else if (m.investigator === "unavailable") { investigatorState = "Unavailable"; investigatorText = "Meta's API is unavailable or today's limit is reached. Everything else comes from our own detectors."; }
    else if (m.named) investigatorText = `Points to: ${m.named}.` + (m.outline ? " Outlined in yellow in the evidence view." : "");
    else if (m.outline === "strongest object") investigatorText = "Outlined the object with the strongest AI evidence (yellow, in the evidence view).";
    else investigatorText = "Named nothing specific.";
    const explanationState = m.explanation ? "Ran" : "Unavailable";
    const explanationText = m.explanation || "No explanation right now (the API is unavailable or today's limit is reached).";
    return { investigatorState, investigatorText, explanationState, explanationText };
  }
  function render(r) {
    fillVerdict($("verdict"), r);
    sources.evidence = r.evidence;
    sources.original = previewUrl;
    setTabs($("tab-evidence"), $("tab-original"), viewerImg, sources, "evidence");
    outlineNote.hidden = true;
    reportEmpty.hidden = true;
    reportBody.hidden = false;
    return renderDetails(r, null);
  }
  async function investigateMore(r, rows, mine) {
    try {
      const m = await postJSON(`/api/investigate/${r.id}/explain`, { signal: controller.signal });
      if (mine !== token) return;
      const t = moreText(m);
      renderDetails(r, t);
      if (m.evidence) {
        sources.evidence = m.evidence;
        if ($("tab-evidence").getAttribute("aria-selected") === "true") viewerImg.src = m.evidence;
        outlineNote.textContent = m.outline === "named object"
          ? `Yellow outline: the object Meta's investigator named ("${m.named}"). It shows where to look, not a proven edit boundary.`
          : "Yellow outline: the object with the strongest AI evidence, outlined by SAM 3.1. It shows where to look, not a proven edit boundary.";
        outlineNote.hidden = false;
      }
    } catch (e) {
      if (mine !== token || e.name === "AbortError") return;
      renderDetails(r, { investigatorState: "Failed", investigatorText: e.message, explanationState: "Failed", explanationText: "Not available." });
    }
  }
  go.addEventListener("click", async () => {
    if (!file || go.disabled) return;
    token += 1;
    const mine = token;
    if (controller) controller.abort();
    controller = new AbortController();
    hideError(reportError);
    setGo("Investigating...", true);
    const form = new FormData();
    form.append("file", file, file.name); // the original file, unchanged
    try {
      const r = await postJSON("/api/investigate", { body: form, signal: controller.signal });
      if (mine !== token) return;
      const rows = render(r);
      setGo("Investigate again", false);
      investigateMore(r, rows, mine);
    } catch (e) {
      if (mine !== token || e.name === "AbortError") return;
      clearReport();
      reportEmpty.hidden = true;
      showError(reportError, e.message);
      setGo("Start investigation", false);
    }
  });

  // ---------- Try to fool it ----------
  const prompt = $("prompt"), foolGo = $("fool-go"), foolGoText = $("fool-go-text"), foolError = $("fool-error");
  const foolResult = $("fool-result"), foolImg = $("fool-img"), foolNote = $("fool-note"), foolCrop = $("fool-crop");
  const foolSources = { evidence: "", original: "" };
  let foolId = null, foolBusy = false;
  wireTabs($("fool-tab-evidence"), $("fool-tab-original"), foolImg, foolSources);
  function foolState(busy, text) {
    foolBusy = busy;
    foolGo.disabled = busy;
    foolCrop.disabled = busy || !foolId;
    foolGoText.textContent = text;
    if (busy) foolGo.setAttribute("aria-busy", "true"); else foolGo.removeAttribute("aria-busy");
  }
  function showFool(result, note) {
    fillVerdict($("fool-verdict"), result.report);
    foolSources.evidence = result.report.evidence;
    foolSources.original = result.picture;
    setTabs($("fool-tab-evidence"), $("fool-tab-original"), foolImg, foolSources, "evidence");
    foolNote.textContent = note || "";
    foolNote.hidden = !note;
    foolResult.hidden = false;
  }
  foolGo.addEventListener("click", async () => {
    if (foolBusy) return;
    hideError(foolError);
    if (!prompt.value.trim()) { showError(foolError, "Please type a prompt first."); prompt.focus(); return; }
    foolState(true, "Making the picture and investigating...");
    try {
      const result = await postJSON("/api/fool", { headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ prompt: prompt.value }) });
      foolId = result.id;
      showFool(result, "");
    } catch (e) { showError(foolError, e.message); }
    foolState(false, "Make it and investigate");
  });
  foolCrop.addEventListener("click", async () => {
    if (foolBusy || !foolId) return;
    hideError(foolError);
    foolState(true, "Make it and investigate");
    foolCrop.textContent = "Investigating the cropped copy...";
    try {
      const result = await postJSON(`/api/fool/${foolId}/crop`, {});
      showFool(result, "Cropped and re-saved, like a forwarded copy: the file's AI label is gone, so only the pixels can tell.");
    } catch (e) { showError(foolError, e.message); }
    foolCrop.textContent = "Crop, re-save and investigate again";
    foolState(false, "Make it and investigate");
  });
})();
