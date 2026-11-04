// Node built-ins only. Real Chrome/CDP interactions; no browser packages or installs.
"use strict";
const fs = require("node:fs");
const path = require("node:path");
const [websocketUrl, baseUrl, fixture, corrupt, output] = process.argv.slice(2);
const report = { successful: false, checks: [], screenshots: [], javascript_errors: [],
  blocked_external_resources: [], local_requests: 0, local_request_paths: [] };
const pending = new Map();
let nextId = 1, sessionId;
let holdInvestigations = false;
const heldInvestigations = [];
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
    if (local) {
      report.local_requests++; report.local_request_paths.push(new URL(request.url).pathname);
      if (holdInvestigations && new URL(request.url).pathname === "/api/investigate") heldInvestigations.push(requestId);
      else call("Fetch.continueRequest", { requestId }).catch(() => {});
    }
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
  if (page === "about" || page === "investigate") {
    await waitFor("document.querySelectorAll('.section-index-link[aria-current=location]').length===1", "Route outline has not finished its animation-frame update");
  }
}
async function investigateFixture() {
  await upload(fixture);
  await waitFor("!document.getElementById('go').disabled", "Valid upload did not enable investigation");
  await click("#go");
  await waitFor("!document.getElementById('report-body').hidden && document.querySelector('#verdict .verdict-title').textContent === 'Likely authentic'", "No rendered authentic verdict");
  await waitFor("document.getElementById('details-list').textContent.includes('No clear AI signs were found')", "Mocked explanation did not render");
}
async function checkFullImageFit(label) {
  const measurements = [];
  for (const [width, height] of [[400, 1600], [1800, 300], [1200, 1200]]) {
    await navigate("investigate");
    await evaluate(`(async()=>{
      const canvas=document.createElement('canvas');canvas.width=${width};canvas.height=${height};
      const ctx=canvas.getContext('2d');ctx.fillStyle='#ddd';ctx.fillRect(0,0,canvas.width,canvas.height);
      ctx.fillStyle='red';ctx.fillRect(0,0,canvas.width,24);
      ctx.fillStyle='lime';ctx.fillRect(0,canvas.height-24,canvas.width,24);
      ctx.fillStyle='blue';ctx.fillRect(0,0,24,canvas.height);
      ctx.fillStyle='yellow';ctx.fillRect(canvas.width-24,0,24,canvas.height);
      const blob=await new Promise(resolve=>canvas.toBlob(resolve,'image/png'));
      const transfer=new DataTransfer();transfer.items.add(new File([blob],'fit-${width}x${height}.png',{type:'image/png'}));
      const input=document.getElementById('file-input');input.files=transfer.files;input.dispatchEvent(new Event('change',{bubbles:true}));
      window.fitFixture=canvas.toDataURL('image/png');
    })()`);
    await waitFor(`document.getElementById('preview').complete && document.getElementById('preview').naturalWidth===${width}`, "Synthetic preview did not load");
    async function contained(selector, parent) {
      const state = await evaluate(`(()=>{const image=document.querySelector('${selector}'),box=document.querySelector('${parent}');
        const i=image.getBoundingClientRect(),b=box.getBoundingClientRect();
        return {imageWidth:i.width,imageHeight:i.height,boxWidth:b.width,boxHeight:b.height,fit:getComputedStyle(image).objectFit,
          inside:i.width>0 && i.height>0 && i.left>=b.left-1 && i.top>=b.top-1 && i.right<=b.right+1 && i.bottom<=b.bottom+1};})()`);
      assert(state.inside && state.fit === "contain", `${label} ${width}x${height} ${selector} clips the image: ${JSON.stringify(state)}`);
      return state;
    }
    const preview = await contained("#preview", "#drop");
    await click("#go");
    await waitFor("!document.getElementById('report-body').hidden && !document.getElementById('go').disabled && document.getElementById('viewer-img').complete", "Synthetic report did not load");
    const evidence = await contained("#viewer-img", "#viewer");
    await click("#tab-original");
    await waitFor(`document.getElementById('viewer-img').complete && document.getElementById('viewer-img').naturalWidth===${width}`, "Original tab did not load");
    const original = await contained("#viewer-img", "#viewer");
    if (height > width) {
      await evaluate("document.getElementById('drop').scrollIntoView({block:'center'})");
      await screenshot(`browser-${label}-portrait-upload.png`);
      await evaluate("document.getElementById('viewer').scrollIntoView({block:'center'})");
      await screenshot(`browser-${label}-portrait-result.png`);
    }
    await navigate("fool");
    await evaluate("document.getElementById('fool-result').hidden=false;document.getElementById('fool-img').src=window.fitFixture");
    await waitFor("document.getElementById('fool-img').complete && document.getElementById('fool-img').naturalWidth>0", "Synthetic generated-image view did not load");
    const generated = await contained("#fool-img", "#fool-viewer");
    measurements.push({width,height,preview,evidence,original,generated});
    await evaluate("document.getElementById('fool-result').hidden=true");
  }
  await navigate("investigate");
  return measurements;
}
function apiCount() { return report.local_request_paths.filter((p) => p.startsWith("/api/")).length; }
async function activeSection() {
  return await evaluate("document.querySelector('.section-index-link[aria-current=location]')?.dataset.target");
}
async function jumpToHeading(id) {
  await evaluate(`(()=>{const h=document.getElementById(${JSON.stringify(id)}),bar=document.querySelector('.topbar');window.scrollTo({top:h.getBoundingClientRect().top+scrollY-bar.getBoundingClientRect().bottom-16,behavior:'instant'});})()`);
}
async function assertMarkerAligned() {
  await delay(240);
  const state = await evaluate(`(()=>{
    const a=document.querySelector('.section-index-link[aria-current=location] .section-index-dot').getBoundingClientRect();
    const b=document.querySelector('.section-index-active').getBoundingClientRect();
    return {dx:Math.abs(a.x+a.width/2-b.x-b.width/2),dy:Math.abs(a.y+a.height/2-b.y-b.height/2)};
  })()`);
  assert(state.dx < 1.5 && state.dy < 1.5, "Active marker does not follow the actual rendered dot: " + JSON.stringify(state));
  return state;
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
    await check("desktop_portrait_wide_square_images_fit", async () => {
      await viewport(1280, 900, false);
      return await checkFullImageFit("desktop");
    });
    await check("mobile_portrait_wide_square_images_fit", async () => {
      await viewport(375, 812, true);
      return await checkFullImageFit("mobile");
    });
    await check("section_index_about_structure_layout_and_themes", async () => {
      await viewport(1440, 900, false);
      await navigate("about");
      await evaluate("window.scrollTo({top:0,behavior:'instant'})");
      await waitFor("document.querySelectorAll('.section-index-link').length===6 && document.getElementById('section-index-disclosure').open", "About index not initialized");
      const state = await evaluate(`(()=>{
        const main=document.getElementById('main').getBoundingClientRect(),nav=document.getElementById('section-index').getBoundingClientRect();
        return {headings:[...document.querySelectorAll('#page-about h2,#page-about h3')].map(h=>h.textContent.trim()),
          labels:[...document.querySelectorAll('.section-index-link')].map(b=>b.textContent.trim()),
          gap:nav.left-main.right,width:nav.width,active:document.querySelectorAll('.section-index-link[aria-current=location]').length,
          overflow:document.documentElement.scrollWidth>innerWidth+1};})()`);
      assert(JSON.stringify(state.headings)===JSON.stringify(state.labels), "About outline differs from actual headings");
      assert(state.gap>=20 && state.width>=200 && state.width<=230 && state.active===1 && !state.overflow, "Invalid desktop outline layout: "+JSON.stringify(state));
      assert(await activeSection()==="about-how-it-works", "First entry not active above the first section");
      await assertMarkerAligned();
      await screenshot("index-about-desktop-dark.png");
      await click("#theme-toggle");
      await screenshot("index-about-desktop-light.png");
      await click("#theme-toggle");
      return state;
    });
    await check("section_index_tracks_up_down_fast_scroll_and_bottom", async () => {
      const before=apiCount();
      for (const id of ["about-how-it-works","about-evidence","about-upload-privacy","about-limitations","about-verdict","about-how-it-works"]) {
        await jumpToHeading(id);
        const expected=await evaluate("document.scrollingElement.scrollHeight-(scrollY+innerHeight)<=3") ? "about-source-code" : id;
        await waitFor(`document.querySelector('.section-index-link[aria-current=location]')?.dataset.target==='${expected}'`, "Incorrect scroll selection: "+id);
        assert(await evaluate("document.querySelectorAll('.section-index-link[aria-current=location]').length===1"), "Outline has multiple active entries");
      }
      await evaluate("window.scrollTo({top:document.scrollingElement.scrollHeight,behavior:'instant'})");
      await waitFor("document.querySelector('.section-index-link[aria-current=location]')?.dataset.target==='about-source-code'", "Last entry not active at page bottom");
      await assertMarkerAligned();
      await jumpToHeading("about-how-it-works");
      await waitFor("document.querySelector('.section-index-link[aria-current=location]')?.dataset.target==='about-how-it-works'", "Fast upward scroll did not restore first entry");
      const focus=await evaluate("document.activeElement.id");
      await evaluate("window.scrollBy({top:35,behavior:'instant'})");
      await delay(220);
      assert(await activeSection()==="about-how-it-works", "Marker moved while reading within one section");
      assert(await evaluate("document.activeElement.id")===focus, "Ordinary scrolling stole keyboard focus");
      assert(apiCount()===before, "Outline scrolling issued API calls");
    });
    await check("section_index_click_offset_hash_history_and_wrapped_markers", async () => {
      const before=apiCount(), history=await evaluate("history.length"), hash=await evaluate("location.hash");
      await evaluate("document.querySelector('.section-index-link[data-target=about-how-it-works] span:last-child').style.maxWidth='100px'");
      await click('.section-index-link[data-target="about-how-it-works"]');
      await waitFor("Math.abs(document.getElementById('about-how-it-works').getBoundingClientRect().top-document.querySelector('.topbar').getBoundingClientRect().bottom-20)<3", "Heading did not clear the sticky header");
      await waitFor("document.querySelector('.section-index-link[aria-current=location]')?.dataset.target==='about-how-it-works'", "Click did not update active entry");
      const marker=await assertMarkerAligned();
      const row=await evaluate("document.querySelector('.section-index-link[data-target=about-how-it-works]').getBoundingClientRect().height");
      assert(row>45, "Wrapped-label case did not actually wrap");
      assert(await evaluate("location.hash")===hash && await evaluate("history.length")===history, "Section navigation changed routing/history");
      assert(apiCount()===before, "Clicking the outline issued API calls");
      await evaluate("document.querySelector('.section-index-link[data-target=about-how-it-works] span:last-child').style.maxWidth=''");
      return {wrappedRowHeight:row,marker};
    });
    await check("section_index_sidebar_reflows_without_overlap", async () => {
      await viewport(1180,900,false);
      await click("#sidebar-toggle");
      await waitFor("getComputedStyle(document.getElementById('section-index')).getPropertyValue('--index-wide').trim()==='0'", "Open sidebar did not switch narrow content to compact outline");
      assert(await evaluate("document.documentElement.scrollWidth<=innerWidth+1"), "Sidebar/index caused overflow");
      await click("#sidebar-toggle");
      await waitFor("getComputedStyle(document.getElementById('section-index')).getPropertyValue('--index-wide').trim()==='1'", "Closed sidebar did not restore wide outline");
      await viewport(1440,900,false);
      await click("#sidebar-toggle");
      await delay(240);
      const state=await evaluate("({mainLeft:document.getElementById('main').getBoundingClientRect().left,sideRight:document.getElementById('sidebar').getBoundingClientRect().right,scrim:getComputedStyle(document.getElementById('scrim')).display,overflow:document.documentElement.scrollWidth>innerWidth+1})");
      assert(state.mainLeft>=state.sideRight && state.scrim==='none' && !state.overflow, "Open desktop navigation overlaps or blocks the content");
      await screenshot("index-about-sidebar-open.png");
      await click("#sidebar-toggle");
      return state;
    });
    await check("section_index_investigation_empty_loading_results_and_details", async () => {
      await viewport(1440,900,false); await navigate("investigate");
      await upload(fixture);
      await waitFor("document.querySelectorAll('.section-index-link').length===3", "Details link persisted after replacing the image");
      await screenshot("index-investigation-empty-desktop.png");
      holdInvestigations=true;
      await click("#go");
      await waitFor("document.getElementById('go').getAttribute('aria-busy')==='true'", "Could not observe investigating state");
      assert(await evaluate("!document.querySelector('.section-index-link[data-target=investigation-details-heading]')"), "Nonexistent results are linked during investigation");
      holdInvestigations=false;
      for (const requestId of heldInvestigations.splice(0)) await call("Fetch.continueRequest",{requestId});
      await waitFor("document.querySelectorAll('.section-index-link').length===4 && !document.getElementById('go').disabled", "Results did not add the details subsection");
      await waitFor("document.getElementById('details-list').textContent.includes('No clear AI signs were found')", "Mocked optional result did not settle");
      const before=apiCount();
      assert(await evaluate("!document.querySelector('#report-body details').open"), "Details started open unexpectedly");
      await click('.section-index-link[data-target="investigation-details-heading"]');
      await waitFor("document.querySelector('#report-body details').open && document.querySelector('.section-index-link[aria-current=location]')?.dataset.target==='investigation-details-heading'", "Outline click did not expand/select details");
      assert(await evaluate("document.getElementById('investigation-details-heading').tabIndex===0"), "Native disclosure keyboard access was removed");
      await evaluate("document.querySelector('.section-index-link[data-target=investigation-details-heading] span:last-child').style.maxWidth='85px'");
      await assertMarkerAligned();
      const nested=await evaluate("document.querySelector('.section-index-link[data-level=\"3\"] .section-index-dot').getBoundingClientRect().left-document.querySelector('.section-index-link[data-target=report-heading] .section-index-dot').getBoundingClientRect().left");
      assert(nested>=16,"Subsection marker is not indented");
      await screenshot("index-investigation-result-desktop.png");
      await evaluate("document.querySelector('.section-index-link[data-target=investigation-details-heading] span:last-child').style.maxWidth='';document.querySelector('#report-body details').open=false");
      const state=await evaluate("({preview:document.getElementById('preview').src,name:document.getElementById('file-name').textContent,verdict:document.querySelector('#verdict .verdict-title').textContent})");
      for(let i=0;i<5;i++){await navigate("about");await navigate("fool");await navigate("investigate");}
      assert(await evaluate(`document.getElementById('preview').src===${JSON.stringify(state.preview)} && document.getElementById('file-name').textContent===${JSON.stringify(state.name)} && document.querySelector('#verdict .verdict-title').textContent===${JSON.stringify(state.verdict)} && !document.getElementById('report-body').hidden`), "Routing lost the upload or report state");
      assert(apiCount()===before,"Outline/route navigation duplicated investigation/API calls");
      return {subsectionIndent:nested,apiCallsDuringNavigation:apiCount()-before};
    });
    await check("section_index_mobile_disclosure_own_scroll_and_light_theme", async () => {
      await viewport(375,500,true); await navigate("about");
      await waitFor("!document.getElementById('section-index-disclosure').open", "Compact outline did not start collapsed");
      await click("#section-index-disclosure > summary");
      await waitFor("document.getElementById('section-index-disclosure').open", "Compact outline would not open");
      const overflow=await evaluate("(()=>{const n=document.querySelector('.section-index-scroll');return n.scrollHeight>n.clientHeight+5;})()");
      assert(overflow,"Short-viewport outline did not exercise independent scrolling");
      await evaluate("window.scrollTo({top:document.scrollingElement.scrollHeight,behavior:'instant'})");
      await waitFor("document.querySelector('.section-index-link[aria-current=location]')?.dataset.target==='about-source-code'", "Mobile bottom selection is wrong");
      assert(await evaluate("(()=>{const a=document.querySelector('.section-index-link[aria-current=location]').getBoundingClientRect(),b=document.querySelector('.section-index-scroll').getBoundingClientRect();return a.top>=b.top-1 && a.bottom<=b.bottom+1;})()"),"The active entry was not revealed inside the outline's own scroll area");
      const before=await evaluate("scrollY");
      await evaluate("document.querySelector('.section-index-scroll').scrollTop=0");
      await delay(120);
      assert(Math.abs(await evaluate("scrollY")-before)<2,"Scrolling the outline moved the main document");
      await click('.section-index-link[data-target="about-how-it-works"]');
      await waitFor("!document.getElementById('section-index-disclosure').open && document.activeElement.id==='about-how-it-works'", "Mobile click did not close the outline and focus its target");
      await waitFor("Math.abs(document.getElementById('about-how-it-works').getBoundingClientRect().top-document.querySelector('.topbar').getBoundingClientRect().bottom-20)<3", "Mobile target did not settle below the sticky header");
      assert(await evaluate("location.hash")==="#/about", "Mobile outline broke hash routing");
      await viewport(375,812,true);
      await evaluate("window.scrollTo({top:0,behavior:'instant'})");
      await click("#section-index-disclosure > summary");
      await click("#theme-toggle");
      await evaluate("document.activeElement.blur();window.scrollTo({top:0,behavior:'instant'})");
      await delay(150);
      await screenshot("index-about-mobile-light.png");
      assert(await evaluate("document.documentElement.scrollWidth<=innerWidth+1"),"Mobile outline overflows horizontally");
      await click("#theme-toggle");
      await navigate("investigate");
      await screenshot("index-investigation-mobile-dark.png");
    });
    await check("section_index_keyboard_reduced_motion_and_scroll_interruption", async () => {
      await viewport(1440,900,false);await navigate("about");
      await call("Emulation.setEmulatedMedia",{features:[{name:"prefers-reduced-motion",value:"reduce"}]});
      await waitFor("document.getElementById('section-index-disclosure').open && getComputedStyle(document.getElementById('section-index')).getPropertyValue('--index-wide').trim()==='1'", "Desktop outline did not finish route layout");
      await evaluate("window.scrollTo({top:0,behavior:'instant'});document.querySelector('.section-index-link[data-target=about-how-it-works]').focus()");
      assert(await evaluate("document.activeElement.dataset.target==='about-how-it-works'"),"Visible outline button could not receive keyboard focus");
      await call("Input.dispatchKeyEvent",{type:"keyDown",key:"Enter",code:"Enter",windowsVirtualKeyCode:13,nativeVirtualKeyCode:13,text:"\r",unmodifiedText:"\r"});
      await call("Input.dispatchKeyEvent",{type:"keyUp",key:"Enter",code:"Enter",windowsVirtualKeyCode:13,nativeVirtualKeyCode:13});
      await waitFor("document.activeElement.id==='about-how-it-works'", "Keyboard could not activate the outline");
      const state=await evaluate("({reduced:matchMedia('(prefers-reduced-motion: reduce)').matches,duration:getComputedStyle(document.querySelector('.section-index-active')).transitionDuration,allButtonsTabbable:[...document.querySelectorAll('.section-index-link')].every(b=>b.tabIndex===0),offset:document.getElementById('about-how-it-works').getBoundingClientRect().top-document.querySelector('.topbar').getBoundingClientRect().bottom})");
      assert(state.reduced && state.duration==='0s' && state.allButtonsTabbable && Math.abs(state.offset-20)<3,"Reduced-motion or keyboard/header behavior is incorrect");
      await call("Emulation.setEmulatedMedia",{features:[{name:"prefers-reduced-motion",value:"no-preference"}]});
      await evaluate("window.scrollTo({top:0,behavior:'instant'})");await delay(100);
      await click('.section-index-link[data-target="about-source-code"]');await delay(60);
      await call("Input.dispatchMouseEvent",{type:"mouseWheel",x:400,y:400,deltaX:0,deltaY:-450});
      await delay(500);
      assert(await evaluate("scrollY<document.scrollingElement.scrollHeight-innerHeight-20"),"User wheel input did not interrupt smooth navigation");
      return state;
    });
    await check("section_index_repeated_routes_have_no_duplicate_click_handlers", async () => {
      const before=apiCount();
      for(let i=0;i<6;i++){await navigate("investigate");await navigate("about");}
      await evaluate("window.originalScrollForIndexTest=window.scrollTo;window.indexScrollCalls=0;window.scrollTo=function(...args){window.indexScrollCalls++;return window.originalScrollForIndexTest.apply(window,args)}");
      await click('.section-index-link[data-target="about-how-it-works"]');
      await delay(550);
      const calls=await evaluate("window.indexScrollCalls");
      await evaluate("window.scrollTo=window.originalScrollForIndexTest;delete window.originalScrollForIndexTest");
      assert(calls===1,"Duplicate click handlers after route changes: "+calls);
      assert(apiCount()===before,"Repeated outline navigation caused API work");
      assert(await evaluate("document.querySelectorAll('.section-index-link[aria-current=location]').length===1"),"Repeated routing lost the single active entry");
      return {navigationScrollCalls:calls,apiCalls:apiCount()-before};
    });
    await check("section_index_short_page_zoom_and_dynamic_compact_layout", async () => {
      await viewport(1440,2200,false);await navigate("investigate");await upload(fixture);
      await waitFor("document.querySelectorAll('.section-index-link').length===3", "Replacement did not remove result-only entries");
      await evaluate("window.scrollTo({top:0,behavior:'instant'})");
      await waitFor("document.querySelector('.section-index-link[aria-current=location]')?.dataset.target==='steps-heading'", "A page fitting the viewport did not select its first section");
      await viewport(640,450,false);
      await waitFor("getComputedStyle(document.getElementById('section-index')).getPropertyValue('--index-wide').trim()==='0'", "Zoom-sized viewport did not use compact navigation");
      assert(await evaluate("document.documentElement.scrollWidth<=innerWidth+1"),"Zoom-sized viewport overflows");
      if (!await evaluate("document.getElementById('section-index-disclosure').open")) await click("#section-index-disclosure > summary");
      await waitFor("document.getElementById('section-index-disclosure').open", "Compact navigation must be expanded for the scroll-preservation check");
      await evaluate("document.getElementById('examine-heading').scrollIntoView({block:'start',behavior:'instant'})");
      const before=await evaluate("document.getElementById('examine-heading').getBoundingClientRect().top");
      // Isolate the outline's added row from the app's existing intentional report auto-scroll.
      await evaluate("document.getElementById('report-heading').scrollIntoView=()=>{}");
      await click("#go");
      await waitFor("document.querySelectorAll('.section-index-link').length===4 && !document.getElementById('go').disabled", "Compact index did not add the completed details entry");
      await delay(150);
      await evaluate("delete document.getElementById('report-heading').scrollIntoView");
      assert(await evaluate("document.querySelectorAll('.section-index-link[aria-current=location]').length===1"),"Layout refresh produced multiple active entries");
      const after=await evaluate("document.getElementById('examine-heading').getBoundingClientRect().top");
      assert(Math.abs(after-before)<3,"Adding the compact outline entry moved the reading position unexpectedly: "+JSON.stringify({before,after}));
      await viewport(1440,900,false);await navigate("investigate");
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
