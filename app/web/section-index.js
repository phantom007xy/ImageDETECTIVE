// A route-scoped outline. Tracks actual document headings; never touches investigation data or URLs.
(() => {
  "use strict";
  class SectionIndex {
    constructor(element, layout) {
      this.element = element;
      this.layout = layout;
      this.disclosure = element.querySelector("details");
      this.scroller = element.querySelector(".section-index-scroll");
      this.inner = element.querySelector(".section-index-inner");
      this.list = element.querySelector("ol");
      this.track = element.querySelector("path");
      this.indicator = element.querySelector(".section-index-active");
      this.entries = [];
      this.active = null;
      this.page = null;
      this.frame = 0;
      this.wide = null;
      this.compactOpen = false;
    }

    activate(page) {
      this.destroy();
      if (!page || !["about", "investigate"].includes(page.dataset.page)) return;
      this.page = page;
      this.controller = new AbortController();
      const signal = this.controller.signal;
      this.element.hidden = false;
      this.layout.classList.add("has-section-index");
      this.disclosure.addEventListener("click", (event) => {
        if (this.wide && event.target.closest("summary")) event.preventDefault();
      }, { signal });
      this.disclosure.addEventListener("toggle", () => {
        if (!this.wide) this.compactOpen = this.disclosure.open;
        this.schedule(true);
      }, { signal });
      this.list.addEventListener("click", (event) => {
        const button = event.target.closest("button[data-target]");
        if (button) this.goTo(button.dataset.target);
      }, { signal });
      window.addEventListener("scroll", () => this.schedule(), { passive: true, signal });
      window.addEventListener("resize", () => this.schedule(true), { passive: true, signal });
      const interrupt = () => {
        if (!this.scrollingToTarget) return;
        this.scrollingToTarget = false;
        window.scrollTo({ top: window.scrollY, behavior: "instant" });
      };
      window.addEventListener("wheel", interrupt, { passive: true, signal });
      window.addEventListener("touchstart", interrupt, { passive: true, signal });
      window.addEventListener("keydown", (event) => {
        if (["ArrowUp", "ArrowDown", "PageUp", "PageDown", "Home", "End", " "].includes(event.key)) interrupt();
      }, { signal });
      window.addEventListener("scrollend", () => {
        if (Math.abs(window.scrollY - this.scrollDestination) < 3) this.scrollingToTarget = false;
      }, { signal });
      page.addEventListener("load", () => this.schedule(true), { capture: true, signal });
      page.addEventListener("toggle", () => this.schedule(true), { capture: true, signal });
      if (document.fonts) document.fonts.addEventListener("loadingdone", () => this.schedule(true), { signal });
      this.resizeObserver = new ResizeObserver(() => this.schedule(true));
      for (const target of [page, this.layout, this.inner, document.querySelector(".topbar")]) {
        this.resizeObserver.observe(target);
      }
      this.refresh();
    }

    destroy() {
      this.controller?.abort();
      this.resizeObserver?.disconnect();
      cancelAnimationFrame(this.frame);
      this.frame = 0;
      this.page = null;
      this.active = null;
      this.entries = [];
      this.wide = null;
      this.positions = [];
      this.scrollingToTarget = false;
      this.element.hidden = true;
      this.indicator.hidden = true;
      this.layout.classList.remove("has-section-index");
      this.list.replaceChildren();
    }

    available(target) {
      if (target.closest("[hidden]")) return false;
      for (let node = target; node && node !== this.page; node = node.parentElement) {
        if (getComputedStyle(node).display === "none") return false;
      }
      // Headings inside a closed disclosure can still be navigated to; goTo expands it.
      return target.getClientRects().length > 0 || !!target.closest("details:not([open])");
    }

    refresh() {
      if (!this.page) return;
      const targets = [...this.page.querySelectorAll("h2, h3, [data-section-index]")]
        .filter((target) => this.available(target));
      if (targets.length === this.entries.length && targets.every((target, i) => target === this.entries[i].target
          && target.textContent.trim() === this.entries[i].button.textContent.trim())) {
        this.schedule(true);
        return;
      }
      const pageTop = this.page.getBoundingClientRect().top;
      const keepPosition = !this.wide && this.disclosure.open && scrollY > 0 && pageTop < this.readingLine();
      const fragment = document.createDocumentFragment();
      this.entries = targets.map((target) => {
        if (!target.id) {
          const stem = this.page.id + "-" + target.textContent.trim().toLowerCase().replace(/[^a-z0-9]+/g, "-");
          let id = stem, number = 2;
          while (document.getElementById(id)) id = stem + "-" + number++;
          target.id = id;
        }
        const item = document.createElement("li");
        const button = document.createElement("button");
        button.type = "button";
        button.className = "section-index-link";
        button.dataset.target = target.id;
        button.dataset.level = target.dataset.indexLevel || (target.tagName === "H3" ? "3" : "2");
        const dot = document.createElement("span");
        dot.className = "section-index-dot";
        dot.setAttribute("aria-hidden", "true");
        const label = document.createElement("span");
        label.textContent = target.textContent.trim();
        button.append(dot, label);
        item.append(button);
        fragment.append(item);
        return { target, button, dot };
      });
      this.list.replaceChildren(fragment);
      this.active = null;
      this.indicator.hidden = true;
      if (keepPosition) {
        const delta = this.page.getBoundingClientRect().top - pageTop;
        if (Math.abs(delta) > 1) window.scrollBy({ top: delta, behavior: "instant" });
      }
      this.schedule(true);
    }

    readingLine() {
      return Math.max(0, document.querySelector(".topbar").getBoundingClientRect().bottom) + 24;
    }

    schedule(layout = false) {
      if (!this.page) return;
      this.needsLayout ||= layout;
      if (this.frame) return;
      this.frame = requestAnimationFrame(() => {
        this.frame = 0;
        if (!this.page) return;
        const wide = getComputedStyle(this.element).getPropertyValue("--index-wide").trim() === "1";
        if (wide !== this.wide) {
          this.wide = wide;
          this.disclosure.open = wide || this.compactOpen;
          this.disclosure.querySelector("summary").tabIndex = wide ? -1 : 0;
          this.needsLayout = true;
        }
        this.update();
        this.needsLayout = false;
      });
    }

    update() {
      const visible = this.entries.filter(({ target }) => target.getClientRects().length > 0);
      if (!visible.length) return;
      const line = this.readingLine();
      let selected = visible[0];
      for (const entry of visible) {
        if (entry.target.getBoundingClientRect().top <= line) selected = entry;
      }
      const root = document.scrollingElement;
      const bottom = window.scrollY > 1 && root.scrollHeight - (window.scrollY + window.innerHeight) <= 3;
      if (bottom) selected = visible[visible.length - 1];
      if (this.active && !bottom && selected !== this.active && visible.includes(this.active)) {
        const oldIndex = visible.indexOf(this.active), newIndex = visible.indexOf(selected);
        // A tiny dead band avoids a fractional-pixel oscillation at the reading line.
        if (newIndex > oldIndex && selected.target.getBoundingClientRect().top > line - 2) selected = this.active;
        if (newIndex < oldIndex && this.active.target.getBoundingClientRect().top < line + 2) selected = this.active;
      }
      const changed = selected !== this.active;
      if (changed) {
        this.active = selected;
        for (const entry of this.entries) {
          if (entry === selected) entry.button.setAttribute("aria-current", "location");
          else entry.button.removeAttribute("aria-current");
        }
      }
      if (!this.disclosure.open) return;
      if (this.needsLayout || changed || !this.positions.length) {
        const bounds = this.inner.getBoundingClientRect();
        this.positions = this.entries.map(({ dot }) => {
          const rect = dot.getBoundingClientRect();
          return { x: rect.left - bounds.left + rect.width / 2, y: rect.top - bounds.top + rect.height / 2 };
        });
        this.track.setAttribute("d", this.positions.map((point, i) => i
          ? `V ${point.y} H ${point.x}` : `M ${point.x} ${point.y}`).join(" "));
      }
      const point = this.positions[this.entries.indexOf(selected)];
      if (point) {
        const first = this.indicator.hidden;
        if (first) this.indicator.style.transition = "none";
        this.indicator.style.transform = `translate(${point.x - 5}px, ${point.y - 5}px)`;
        this.indicator.hidden = false;
        if (first) {
          this.indicator.getBoundingClientRect();
          this.indicator.style.transition = "";
        }
      }
      if (changed || this.needsLayout) {
        const item = selected.button.getBoundingClientRect(), area = this.scroller.getBoundingClientRect();
        if (item.top < area.top) this.scroller.scrollTop -= area.top - item.top + 6;
        else if (item.bottom > area.bottom) this.scroller.scrollTop += item.bottom - area.bottom + 6;
      }
    }

    goTo(id) {
      const target = this.entries.find((entry) => entry.target.id === id)?.target;
      if (!target) return;
      for (let details = target.closest("details"); details; details = details.parentElement?.closest("details")) {
        details.open = true;
      }
      if (!this.wide) {
        this.compactOpen = false;
        this.disclosure.open = false;
      }
      const signal = this.controller.signal;
      requestAnimationFrame(() => {
        if (signal.aborted || !this.page || !this.page.contains(target)) return;
        const top = target.getBoundingClientRect().top + window.scrollY - this.readingLine() + 4;
        this.scrollDestination = Math.min(Math.max(0, top), Math.max(0, document.scrollingElement.scrollHeight - innerHeight));
        this.scrollingToTarget = !matchMedia("(prefers-reduced-motion: reduce)").matches;
        window.scrollTo({ top: this.scrollDestination, behavior: this.scrollingToTarget ? "smooth" : "instant" });
        if (target.tabIndex < 0 && !target.hasAttribute("tabindex")) target.tabIndex = -1;
        target.focus({ preventScroll: true });
        this.schedule(true);
      });
    }
  }
  window.SectionIndex = SectionIndex;
})();
