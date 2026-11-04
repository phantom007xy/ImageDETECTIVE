// Node built-ins only. Real Chrome/CDP interactions; no browser packages or installs.
"use strict";
const fs = require("node:fs");
const path = require("node:path");
const [websocketUrl, baseUrl, fixture, corrupt, output] = process.argv.slice(2);
const report = { successful: false, checks: [], screenshots: [], javascript_errors: [],
  blocked_external_resources: [], local_requests: 0, local_request_paths: [] };
const pending = new Map();
let nextId = 1, sessionId;
const socket = new WebSocket(websocketUrl);
const delay = (ms) => new Promise((resolve) => setTimeout(resolve, ms));
function call(method, params = {}, browser = false) {
  const id = nextId++;
  return new Promise((resolve, reject) => {
    const timer = setTimeout(() => { pending.delete(id); reject(new Error(`CDP timeout: ${method}`)); }, 10000);
    pending.set(id, { resolve, reject, timer });
    socket.send(JSON.stringify({ id, method, params, ...(!browser && sessionId ? { sessionId } : {}) }));
  });
}
socket.addEventListener("message", (event) => {
  const message = JSON.parse(String(event.data));
  if (message.id) {
    const entry = pending.get(message.id);
    if (entry) { clearTimeout(entry.timer); pending.delete(message.id);
      if (message.error) entry.reject(new Error(message.error.message)); else entry.resolve(message.result); }
    return;
  }
  if (message.method === "Runtime.exceptionThrown") {
    const detail = message.params.exceptionDetails;
    report.javascript_errors.push(detail.exception?.description || detail.text);
  }
  if (message.method === "Runtime.consoleAPICalled" && message.params.type === "error") {
    report.javascript_errors.push(message.params.args.map((arg) => arg.value || arg.description || "error").join(" "));
  }
  if (message.method === "Fetch.requestPaused") {
    const { requestId, request } = message.params;
    let local = false;
    try { const url = new URL(request.url); local = ["127.0.0.1", "localhost", "[::1]"].includes(url.hostname); } catch (_) {}
    if (local) { report.local_requests++; report.local_request_paths.push(new URL(request.url).pathname); call("Fetch.continueRequest", { requestId }).catch(() => {}); }
    else { report.blocked_external_resources.push(request.url);
      call("Fetch.failRequest", { requestId, errorReason: "BlockedByClient" }).catch(() => {}); }
  }
});
function assert(condition, message) { if (!condition) throw new Error(message); }
async function evaluate(expression) {
  const result = await call("Runtime.evaluate", { expression, awaitPromise: true, returnByValue: true });
  if (result.exceptionDetails) throw new Error(result.exceptionDetails.exception?.description || result.exceptionDetails.text);
  return result.result.value;
}
async function waitFor(expression, message, timeout = 8000) {
  const end = Date.now() + timeout;
  while (Date.now() < end) { if (await evaluate(expression)) return; await delay(75); }
  throw new Error(message || `Condition timed out: ${expression}`);
}
async function check(name, callback) {
  const started = Date.now();
  try { const measurements = await callback(); report.checks.push({ id: name, status: "passed", seconds: (Date.now() - started) / 1000, ...(measurements ? { measurements } : {}) }); }
  catch (error) { report.checks.push({ id: name, status: "failed", seconds: (Date.now() - started) / 1000, message: error.message }); }
}
async function click(selector) { await evaluate(`document.querySelector(${JSON.stringify(selector)}).click()`); }
async function upload(file) {
  const { root } = await call("DOM.getDocument");
  const { nodeId } = await call("DOM.querySelector", { nodeId: root.nodeId, selector: "#file-input" });
  await call("DOM.setFileInputFiles", { nodeId, files: [file] });
}
async function screenshot(name) {
  const result = await call("Page.captureScreenshot", { format: "png", captureBeyondViewport: false });
  fs.writeFileSync(path.join(output, name), Buffer.from(result.data, "base64"));
  report.screenshots.push(name);
}
async function viewport(width, height, mobile) {
  await call("Emulation.setDeviceMetricsOverride", { width, height, deviceScaleFactor: 1, mobile });
  await delay(100);
}
async function navigate(page) {
  await click(`a[data-page="${page}"]`);
  await waitFor(`!document.querySelector('[data-page="${page}"].page').hidden`, `Navigation to ${page} failed`);
}
async function investigateFixture() {
  await upload(fixture);
  await waitFor("!document.getElementById('go').disabled", "Valid upload did not enable investigation");
  await click("#go");
  await waitFor("!document.getElementById('report-body').hidden && document.querySelector('#verdict .verdict-title').textContent === 'Likely authentic'", "No rendered authentic verdict");
  await waitFor("document.getElementById('details-list').textContent.includes('No clear AI signs were found')", "Mocked explanation did not render");
}
(async () => {
  try {
    await new Promise((resolve, reject) => { socket.addEventListener("open", resolve, { once: true }); socket.addEventListener("error", reject, { once: true }); });
    const { targetId } = await call("Target.createTarget", { url: "about:blank" }, true);
    ({ sessionId } = await call("Target.attachToTarget", { targetId, flatten: true }, true));
    for (const domain of ["Page", "Runtime", "DOM", "Network"]) await call(domain + ".enable");
    await call("Fetch.enable", { patterns: [{ urlPattern: "http*", requestStage: "Request" }] });
    await viewport(1280, 900, false);
    await call("Page.navigate", { url: baseUrl + "/#/investigate" });
    await waitFor("document.readyState === 'complete' && !!document.getElementById('go')", "Frontend did not load");

    await check("desktop_home_and_initial_state", async () => {
      const state = await evaluate("({title: document.title, disabled: document.getElementById('go').disabled, width: innerWidth, overflow: document.documentElement.scrollWidth > innerWidth + 1})");
      assert(state.title === "ImageDETECTIVE" && state.disabled && !state.overflow, "Unexpected home state or horizontal overflow");
      await screenshot("browser-desktop-home.png"); return state;
    });
    await check("navigation_fool_about_investigate", async () => {
      await navigate("fool"); assert(await evaluate("!!document.getElementById('prompt')"), "Missing generation prompt");
      await navigate("about"); assert(await evaluate("document.querySelector('.about').textContent.includes('What happens to your upload')"), "About content missing");
      await navigate("investigate");
    });
    await check("theme_toggle_roundtrip", async () => {
      const initial = await evaluate("document.documentElement.dataset.theme");
      await click("#theme-toggle"); assert(await evaluate("document.documentElement.dataset.theme") !== initial, "Theme did not toggle");
      await click("#theme-toggle"); assert(await evaluate("document.documentElement.dataset.theme") === initial, "Theme did not return");
    });
    await check("desktop_real_upload_verdict_evidence_details", async () => {
      await investigateFixture();
      const state = await evaluate("({verdict: document.querySelector('#verdict .verdict-title').textContent, evidence: document.getElementById('viewer-img').src.startsWith('data:image/jpeg;base64,'), context: document.querySelector('#verdict .verdict-context').textContent, details: document.getElementById('details-list').textContent.includes('Processing time')})");
      assert(state.evidence && state.details && state.context.includes("when or where"), "Rendered evidence/details/context missing");
      await evaluate("document.getElementById('report-heading').scrollIntoView({block:'start'})");
      await screenshot("browser-desktop-result.png"); return state;
    });
    await check("evidence_original_tabs", async () => {
      await click("#tab-original"); assert(await evaluate("document.getElementById('viewer-img').src.startsWith('blob:') && document.getElementById('tab-original').getAttribute('aria-selected') === 'true'"), "Original tab did not show original file");
      await click("#tab-evidence"); assert(await evaluate("document.getElementById('viewer-img').src.startsWith('data:image/jpeg;base64,') && document.getElementById('tab-evidence').getAttribute('aria-selected') === 'true'"), "Evidence tab did not restore evidence");
    });
    await check("upload_interface_has_no_example_gallery", async () => {
      assert(await evaluate("!document.querySelector('[data-example], .examples') && !document.body.textContent.includes('Try an example, or upload your own image below')"), "Removed example gallery is still visible");
      assert(await evaluate("!!document.getElementById('drop') && !!document.getElementById('file-input')"), "Upload controls are missing");
      assert(await evaluate("document.getElementById('evidence-source').textContent.includes('DINOv3') && document.getElementById('evidence-description').textContent.length > 20"), "Evidence source or coverage note missing");
    });
    await check("corrupt_replacement_clears_stale_result", async () => {
      await upload(corrupt);
      await waitFor("!document.getElementById('upload-error').hidden && document.getElementById('go').disabled", "Corrupt preview did not show recoverable error");
      assert(await evaluate("document.getElementById('report-body').hidden && !document.getElementById('report-empty').hidden"), "Previous verdict stayed visible after invalid replacement");
    });
    await check("browser_fetch_corrupt_server_upload_rejected", async () => {
      const result = await evaluate("(async()=>{const form=new FormData();form.append('file',new Blob(['synthetic corrupt image'],{type:'image/png'}),'corrupt.png');const response=await fetch('/api/investigate',{method:'POST',body:form});return {status:response.status,detail:(await response.json()).detail};})()");
      assert(result.status === 400 && result.detail.includes("could not be read"), "Server did not reject corrupt image through browser request"); return result;
    });
    await check("valid_upload_recovers_after_corrupt_file", async () => {
      await investigateFixture(); assert(await evaluate("document.getElementById('upload-error').hidden"), "Upload error persisted after valid replacement");
    });
    await check("generation_api_unavailable_shows_recoverable_error", async () => {
      await navigate("fool"); await evaluate("document.getElementById('prompt').value='a fictional blue bicycle'");
      await click("#fool-go");
      await waitFor("!document.getElementById('fool-error').hidden && !document.getElementById('fool-go').disabled", "Generation fallback missing or button stuck");
      assert(await evaluate("document.getElementById('fool-error').textContent.includes('unavailable') && document.getElementById('fool-result').hidden"), "Generation failure showed substituted image or unclear fallback");
      await navigate("investigate");
    });
    await check("mobile_layout_sidebar_and_navigation", async () => {
      await viewport(375, 812, true);
      await evaluate("window.scrollTo(0,0)");
      const state = await evaluate("({width: innerWidth, scrollWidth: document.documentElement.scrollWidth, mainWidth: document.getElementById('main').getBoundingClientRect().width, sidebarOpen: document.getElementById('sidebar').classList.contains('open')})");
      assert(state.width === 375 && state.scrollWidth <= 376 && state.mainWidth > 200, "Mobile layout overflows or usable content is too narrow");
      await click("#sidebar-toggle"); assert(await evaluate("document.getElementById('sidebar-toggle').getAttribute('aria-expanded') === 'true'"), "Mobile menu did not expand");
      await navigate("about"); assert(await evaluate("!document.getElementById('sidebar').classList.contains('open')"), "Mobile menu did not close after navigation");
      await navigate("investigate"); await screenshot("browser-mobile-home.png"); return state;
    });
    await check("mobile_upload_and_report", async () => {
      await investigateFixture();
      assert(await evaluate("document.documentElement.scrollWidth <= innerWidth + 1"), "Mobile report has horizontal overflow");
      await evaluate("document.getElementById('report-heading').scrollIntoView({block:'start'})");
      await screenshot("browser-mobile-result.png");
    });
    await check("frontend_has_no_javascript_exceptions", async () => {
      assert(report.javascript_errors.length === 0, report.javascript_errors.join("\n"));
      return { blocked_external_resources: report.blocked_external_resources.length, local_requests: report.local_requests };
    });
    report.successful = report.checks.length > 0 && report.checks.every((check) => check.status === "passed");
  } catch (error) {
    report.limitation = error.message;
    try { report.page_diagnostic = await evaluate("({url:location.href,ready:document.readyState,title:document.title,text:document.body?.textContent.slice(0,500),html:document.documentElement.outerHTML.slice(0,500)})"); await screenshot("browser-diagnostic.png"); } catch (_) {}
  }
  finally {
    fs.writeFileSync(path.join(output, "browser_checks.json"), JSON.stringify(report, null, 2) + "\n");
    try { await call("Browser.close", {}, true); } catch (_) {}
    socket.close();
    for (const entry of pending.values()) { clearTimeout(entry.timer); entry.reject(new Error("Browser closed")); }
    pending.clear();
  }
  console.log(JSON.stringify({successful:report.successful, checks:report.checks.length, failures:report.checks.filter((check)=>check.status!=="passed").length}));
  process.exitCode = report.successful ? 0 : 1;
})().catch((error) => { console.error(error.message); process.exitCode = 2; socket.close(); });
