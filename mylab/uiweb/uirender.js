/* mylab app renderer: draws the component tree of mylab.ui as HTML controls.
 *
 * One file, no dependencies, no build step. It is used by the mylab web app (web/js/uirender.js
 * is a copy of mylab/uiweb/uirender.js; scripts/build_web.py keeps it in sync), by the local
 * server behind `mylab app.m` and by static HTML snapshots.
 *
 *   const view = MylabUI.mount(container, {
 *     send(event) -> Promise<{ error, update, output, figures }>,   // deliver one event to the engine
 *     onOutput(text), onError(text), onResult(answer), onEmpty(),   // all optional
 *     readOnly: false,                                              // snapshot: controls do nothing
 *   });
 *   view.setTree(tree);     // { figures: [...], dialogs: [...] }  (mylab.ui.snapshot)
 *   view.apply(update);     // the "update" part of an answer to an event
 *   view.destroy();
 *
 * The tree and event formats are described at the top of mylab/ui.py.
 */
(function (global) {
  "use strict";

  const SVG_NS = "http://www.w3.org/2000/svg";

  function h(tag, attrs, ...children) {
    const e = document.createElement(tag);
    for (const [k, v] of Object.entries(attrs || {})) {
      if (v === false || v == null) continue;
      if (k === "class") e.className = v;
      else if (k.startsWith("on")) e.addEventListener(k.slice(2), v);
      else e.setAttribute(k, v === true ? "" : v);
    }
    for (const c of children.flat()) if (c != null && c !== false) e.append(c);
    return e;
  }

  function closeIcon() {
    const svg = document.createElementNS(SVG_NS, "svg");
    svg.setAttribute("viewBox", "0 0 16 16");
    svg.setAttribute("width", "14");
    svg.setAttribute("height", "14");
    svg.setAttribute("aria-hidden", "true");
    const p = document.createElementNS(SVG_NS, "path");
    p.setAttribute("d", "M3.5 3.5l9 9m0-9l-9 9");
    p.setAttribute("stroke", "currentColor");
    p.setAttribute("stroke-width", "1.6");
    p.setAttribute("stroke-linecap", "round");
    p.setAttribute("fill", "none");
    svg.append(p);
    return svg;
  }

  /** Black or white, whichever reads better on the background color "#rrggbb". */
  function contrast(hex) {
    const m = /^#([0-9a-f]{2})([0-9a-f]{2})([0-9a-f]{2})$/i.exec(hex || "");
    if (!m) return null;
    const [r, g, b] = m.slice(1).map((x) => parseInt(x, 16) / 255).map((c) => (c <= 0.03928 ? c / 12.92 : Math.pow((c + 0.055) / 1.055, 2.4)));
    return 0.2126 * r + 0.7152 * g + 0.0722 * b > 0.4 ? "#111111" : "#ffffff";
  }

  const trackSize = (s) => (typeof s === "number" ? `${s}px` : s === "fit" ? "auto" : `minmax(0, ${parseFloat(s) || 1}fr)`);
  const fmt = (x) => (Math.abs(x) >= 1000 || Number.isInteger(x) ? String(Math.round(x * 100) / 100) : String(Math.round(x * 100) / 100));
  const safeUrl = (u) => (/^(https?:|mailto:)/i.test(u || "") ? u : null);

  const FORM = "button, input, select, textarea";
  let groupCounter = 0;

  /* ------------------------------------------------------------------ view */

  class View {
    constructor(container, opts) {
      this.root = h("div", { class: "mui-root" });
      this.windows = h("div", { class: "mui-windows" });
      this.root.append(this.windows);
      container.append(this.root);
      this.opts = opts || {};
      this.items = new Map(); // id -> { node, el, content, parent }
      this.dialogEls = new Map();
      this.queue = Promise.resolve();
      this.pending = 0;
      this.live = new Map(); // id -> latest "changing" event that waits for its turn
      this.observers = [];
      this.sizeTimer = null;
      this.destroyed = false;
    }

    destroy() {
      this.destroyed = true;
      for (const o of this.observers) o.disconnect();
      this.observers = [];
      clearTimeout(this.sizeTimer);
      this.root.remove();
    }

    /* ---------- talking to the engine ---------- */

    send(event) {
      if (this.opts.readOnly || this.destroyed || !this.opts.send) return Promise.resolve();
      this.busy(1);
      const run = () => this.deliver(event);
      this.queue = this.queue.then(run, run);
      return this.queue;
    }

    /** Events while dragging or typing: only the newest one per component waits in line. */
    sendLive(event) {
      if (this.opts.readOnly || this.destroyed || !this.opts.send) return;
      const waiting = this.live.has(event.id);
      this.live.set(event.id, event);
      if (waiting) return;
      this.busy(1);
      const run = () => {
        const latest = this.live.get(event.id);
        this.live.delete(event.id);
        return this.deliver(latest);
      };
      this.queue = this.queue.then(run, run);
    }

    busy(delta) {
      this.pending += delta;
      this.root.classList.toggle("mui-busy", this.pending > 0);
    }

    deliver(event) {
      const done = (async () => {
        let res = null;
        try {
          res = await this.opts.send(event);
        } catch (e) {
          if (this.opts.onError) this.opts.onError(String((e && e.message) || e));
        }
        this.busy(-1);
        if (!res || this.destroyed) return;
        if (res.output && this.opts.onOutput) this.opts.onOutput(res.output);
        if (res.error && this.opts.onError) this.opts.onError(res.error);
        if (this.opts.onResult) this.opts.onResult(res);
        if (res.update) this.apply(res.update);
      })();
      // With "patience", an answer that takes long (a callback that waits for the user) does not
      // hold back the events that follow it.
      if (!this.opts.patience) return done;
      return Promise.race([done, new Promise((resolve) => setTimeout(resolve, this.opts.patience))]);
    }

    /* ---------- trees and updates ---------- */

    setTree(tree) {
      const focusId = this.focusedId();
      for (const o of this.observers) o.disconnect();
      this.observers = [];
      this.items.clear();
      this.windows.textContent = "";
      const figures = (tree && tree.figures) || [];
      for (const fig of figures) this.windows.append(this.build(fig, null));
      this.setDialogs((tree && tree.dialogs) || []);
      if (focusId != null) this.refocus(focusId);
      if (!figures.length && !((tree && tree.dialogs) || []).length && this.opts.onEmpty) this.opts.onEmpty();
      this.measureSoon();
    }

    apply(update) {
      if (!update) return;
      if (update.full) return this.setTree(update);
      for (const [id, changes] of Object.entries(update.patch || {})) {
        const item = this.items.get(Number(id));
        if (!item) continue;
        for (const [k, v] of Object.entries(changes)) {
          if (v === null) delete item.node.props[k];
          else item.node.props[k] = v;
        }
        this.sync(item);
      }
      this.setDialogs(update.dialogs || []);
      this.measureSoon();
    }

    focusedId() {
      const a = document.activeElement;
      const box = a && this.root.contains(a) ? a.closest("[data-mui-id]") : null;
      return box ? Number(box.dataset.muiId) : null;
    }

    refocus(id) {
      const item = this.items.get(id);
      const target = item && (item.el.matches("button, input, select, textarea") ? item.el : item.el.querySelector("button, input, select, textarea, [tabindex]"));
      if (target) target.focus({ preventScroll: true });
    }

    /* ---------- building ---------- */

    build(node, parent) {
      const maker = MAKERS[node.type] || MAKERS.unknown;
      const item = { node, parent, el: null, content: null, view: this };
      maker.create(item, this);
      item.el.dataset.muiId = node.id;
      item.el.classList.add("mui-" + node.type);
      this.items.set(node.id, item);
      const kids = node.children || [];
      if (node.type === "figure") this.buildFigureChildren(item, kids);
      else if (node.type === "tabgroup") this.buildTabs(item, kids);
      else if (node.type === "menu") this.buildMenuChildren(item, kids);
      else if (item.content) for (const c of kids) item.content.append(this.build(c, item));
      this.sync(item);
      return item.el;
    }

    buildFigureChildren(item, kids) {
      const menus = kids.filter((c) => c.type === "menu");
      if (menus.length) {
        const bar = h("div", { class: "mui-menubar", role: "menubar", "aria-label": "Menu" });
        for (const m of menus) bar.append(this.build(m, item));
        item.el.insertBefore(bar, item.frame);
      }
      for (const c of kids) if (c.type !== "menu") item.content.append(this.build(c, item));
    }

    buildMenuChildren(item, kids) {
      if (!kids.length) return;
      item.list = h("div", { class: "mui-menu-list", role: "menu", hidden: true });
      for (const c of kids) item.list.append(this.build(c, item));
      item.el.append(item.list);
      item.button.setAttribute("aria-haspopup", "menu");
      item.button.setAttribute("aria-expanded", "false");
    }

    buildTabs(item, kids) {
      item.tabs = [];
      kids.forEach((tab, i) => {
        const panel = this.build(tab, item);
        const tabItem = this.items.get(tab.id);
        const btn = h("button", {
          type: "button", role: "tab", class: "mui-tab-button", id: `mui-tab-${tab.id}`, "aria-controls": `mui-tabpanel-${tab.id}`,
          onclick: () => this.pickTab(item, i, true),
          onkeydown: (e) => {
            const d = e.key === "ArrowRight" ? 1 : e.key === "ArrowLeft" ? -1 : 0;
            if (!d) return;
            e.preventDefault();
            const n = (i + d + item.tabs.length) % item.tabs.length;
            this.pickTab(item, n, true);
            item.tabs[n].btn.focus();
          },
        });
        panel.id = `mui-tabpanel-${tab.id}`;
        panel.setAttribute("role", "tabpanel");
        panel.setAttribute("aria-labelledby", btn.id);
        tabItem.tabButton = btn;
        this.sync(tabItem);
        item.tabs.push({ btn, panel, id: tab.id });
        item.tablist.append(btn);
        item.content.append(panel);
      });
    }

    pickTab(item, index, notify) {
      if (item.node.props.selected === index && !notify) return;
      const before = item.node.props.selected;
      item.node.props.selected = index;
      this.sync(item);
      if (notify && before !== index) this.send({ id: item.node.id, event: "select", value: index });
      this.measureSoon();
    }

    /* ---------- keeping an element in step with its props ---------- */

    sync(item) {
      const { node, el } = item;
      const p = node.props || {};
      el.hidden = p.visible === false;
      const disabled = p.enabled === false;
      el.classList.toggle("mui-disabled", disabled);
      if (p.tooltip) el.title = p.tooltip;
      else el.removeAttribute("title");
      this.place(item);
      this.paint(item);
      const maker = MAKERS[node.type] || MAKERS.unknown;
      maker.update(item, p, this);
      if (!item.content && node.type !== "table") {
        const controls = el.matches(FORM) ? [el] : el.querySelectorAll(FORM);
        for (const c of controls) c.disabled = disabled;
      }
    }

    place(item) {
      const { node, el, parent } = item;
      const p = node.props || {};
      const s = el.style;
      if (node.type === "figure" || node.type === "tab" || node.type === "menu") return;
      s.gridRow = s.gridColumn = s.left = s.bottom = s.width = s.height = s.position = s.inset = "";
      el.classList.remove("mui-fit-h", "mui-fit-w");
      if (p.grid) {
        const [r1, r2, c1, c2] = p.grid;
        s.gridRow = `${r1} / ${r2 + 1}`;
        s.gridColumn = `${c1} / ${c2 + 1}`;
        const spec = parent && parent.node.props;
        if (spec) {
          const rows = (spec.rows || []).slice(r1 - 1, r2);
          const cols = (spec.cols || []).slice(c1 - 1, c2);
          if (rows.length && rows.every((x) => x === "fit")) el.classList.add("mui-fit-h");
          if (cols.length && cols.every((x) => x === "fit")) el.classList.add("mui-fit-w");
        }
      } else if (p.npos) {
        const [x, y, w, ht] = p.npos;
        s.position = "absolute";
        s.left = `${x * 100}%`;
        s.bottom = `${y * 100}%`;
        s.width = `${w * 100}%`;
        s.height = `${ht * 100}%`;
      } else if (p.pos) {
        let [x, y, w, ht] = p.pos;
        if (node.type === "slider" && !p.vertical && ht < 22) {
          y -= (22 - ht) / 2; // a slider's Position is its thin track
          ht = 22;
        }
        if (node.type === "slider" && p.vertical && w < 22) {
          x -= (22 - w) / 2;
          w = 22;
        }
        s.position = "absolute";
        s.left = `${x}px`;
        s.bottom = `${y}px`;
        s.width = `${w}px`;
        s.height = `${ht}px`;
      } else {
        s.position = "absolute";
        s.inset = "0";
      }
    }

    paint(item) {
      const st = (item.node.props && item.node.props.style) || {};
      const target = item.paintTarget || item.el;
      const s = target.style;
      s.color = st.color || (st.background ? contrast(st.background) || "" : "");
      s.background = st.background || "";
      s.fontSize = st.fontSize ? `${st.fontSize}px` : "";
      s.fontWeight = st.bold ? "700" : "";
      s.fontStyle = st.italic ? "italic" : "";
      s.fontFamily = st.font && !/^helvetica$/i.test(st.font) ? `"${st.font.replace(/["\\]/g, "")}", var(--mui-font)` : "";
      target.dataset.align = st.align || "";
      target.dataset.valign = st.valign || "";
    }

    /* ---------- axes sizes ---------- */

    measureSoon() {
      if (this.opts.readOnly) return;
      clearTimeout(this.sizeTimer);
      this.sizeTimer = setTimeout(() => this.measure(), 120);
    }

    measure() {
      if (this.destroyed) return;
      for (const item of this.items.values()) {
        if (item.node.type !== "axes" || item.el.hidden) continue;
        const r = item.el.getBoundingClientRect();
        const scale = item.el.offsetWidth ? r.width / item.el.offsetWidth : 1;
        const w = Math.round(item.el.offsetWidth);
        const ht = Math.round(item.el.offsetHeight);
        if (w < 40 || ht < 40 || !scale) continue;
        const p = item.node.props;
        if (Math.abs(w - p.width) > 6 || Math.abs(ht - p.height) > 6) {
          if (item.asked && item.asked[0] === w && item.asked[1] === ht) continue;
          item.asked = [w, ht];
          this.send({ id: item.node.id, event: "size", width: w, height: ht });
        }
      }
    }

    watch(el, fn) {
      if (typeof ResizeObserver === "undefined") return;
      const o = new ResizeObserver(fn);
      o.observe(el);
      this.observers.push(o);
    }

    /* ---------- dialogs ---------- */

    setDialogs(list) {
      const seen = new Set();
      for (const d of list) {
        seen.add(d.id);
        let entry = this.dialogEls.get(d.id);
        if (!entry) {
          entry = this.buildDialog(d);
          this.dialogEls.set(d.id, entry);
          this.root.append(entry.el);
          const first = entry.el.querySelector("button.mui-default") || entry.el.querySelector("button");
          if (first && d.kind !== "progress") first.focus({ preventScroll: true });
        }
        entry.update(d);
      }
      for (const [id, entry] of this.dialogEls) {
        if (!seen.has(id)) {
          entry.el.remove();
          this.dialogEls.delete(id);
        }
      }
    }

    buildDialog(d) {
      const title = h("h2", { class: "mui-dialog-title", id: `mui-dlg-title-${d.id}` });
      const message = h("p", { class: "mui-dialog-message", id: `mui-dlg-msg-${d.id}` });
      const body = h("div", { class: "mui-dialog-body" });
      const buttons = h("div", { class: "mui-dialog-buttons" });
      const box = h("div", {
        class: `mui-dialog mui-dialog-${d.kind} mui-icon-${d.icon || "none"}`,
        role: d.kind === "progress" ? "dialog" : "alertdialog", "aria-modal": "true",
        "aria-labelledby": title.id, "aria-describedby": message.id,
      }, title, message, body, buttons);
      const el = h("div", { class: "mui-overlay" }, box);
      const answer = (value) => this.send({ id: d.id, event: "dialog", value });
      let bar = null;
      let pct = null;
      if (d.kind === "progress") {
        bar = h("progress", { class: "mui-progress", max: "1", "aria-labelledby": title.id });
        pct = h("span", { class: "mui-progress-pct" });
        body.append(bar, pct);
      } else {
        for (const opt of d.options || ["OK"]) {
          buttons.append(h("button", { type: "button", class: "mui-button" + (opt === (d.default || (d.options || ["OK"])[0]) ? " mui-default" : ""), onclick: () => answer(opt) }, opt));
        }
        el.addEventListener("keydown", (e) => {
          if (e.key === "Escape") {
            e.preventDefault();
            const opts = d.options || ["OK"];
            answer(opts[opts.length - 1]);
          }
        });
      }
      const update = (now) => {
        title.textContent = now.title || (now.kind === "progress" ? "Working" : "Message");
        message.textContent = now.message || "";
        if (now.kind !== "progress") return;
        if (now.indeterminate) bar.removeAttribute("value");
        else bar.value = now.value || 0;
        pct.textContent = now.percent && !now.indeterminate ? `${Math.round((now.value || 0) * 100)}%` : "";
        buttons.textContent = "";
        if (now.cancelable) {
          buttons.append(h("button", { type: "button", class: "mui-button", disabled: now.cancelRequested, onclick: () => answer("Cancel") }, now.cancelRequested ? "Canceling…" : now.cancelText || "Cancel"));
        }
      };
      return { el, update };
    }
  }

  /* ------------------------------------------------------------------ components */

  const text = (el, s) => {
    if (el.textContent !== (s || "")) el.textContent = s || "";
  };
  const setValue = (input, v) => {
    if (input.value !== String(v)) input.value = String(v);
  };

  function fillOptions(select, items, placeholder) {
    const key = JSON.stringify([items, placeholder || ""]);
    if (select.dataset.items === key) return;
    select.dataset.items = key;
    select.textContent = "";
    if (placeholder != null) select.append(h("option", { value: "-1", disabled: true, hidden: true }, placeholder));
    items.forEach((s, i) => select.append(h("option", { value: String(i) }, s)));
  }

  const MAKERS = {
    unknown: {
      create(item) {
        item.el = h("div", { class: "mui-unknown" }, `(${item.node.type})`);
      },
      update() {},
    },

    figure: {
      create(item, view) {
        const id = item.node.id;
        item.title = h("span", { class: "mui-window-name", id: `mui-win-${id}` });
        const close = h("button", { type: "button", class: "mui-window-close", "aria-label": "Close this window", title: "Close this window", "data-mui-keep": "1", onclick: () => view.send({ id, event: "close" }) }, closeIcon());
        item.image = h("img", { class: "mui-window-image", alt: "", hidden: true });
        item.content = h("div", { class: "mui-window-body" }, item.image);
        item.frame = h("div", { class: "mui-window-frame" }, item.content);
        item.el = h("section", { class: "mui-window", role: "group", "aria-labelledby": item.title.id }, h("header", { class: "mui-window-bar" }, item.title, view.opts.readOnly ? null : close), item.frame);
        item.paintTarget = item.content;
        view.watch(item.el, () => MAKERS.figure.fit(item));
      },
      update(item, p) {
        text(item.title, p.name || "App");
        const kids = (item.node.children || []).filter((c) => c.type !== "menu");
        item.fluid = kids.length > 0 && kids.every((c) => c.type === "gridlayout" && !c.props.pos && !c.props.npos);
        item.el.classList.toggle("mui-fluid", item.fluid);
        item.el.classList.toggle("mui-classic", !!p.classic);
        item.el.style.setProperty("--mui-w", `${p.width}px`);
        item.el.style.setProperty("--mui-h", `${p.height}px`);
        if (p.image) {
          if (item.image.src !== p.image) item.image.src = p.image;
          item.image.hidden = false;
        } else {
          item.image.hidden = true;
        }
        MAKERS.figure.fit(item);
      },
      /** Windows laid out in pixels are scaled down when the page is narrower than they are. */
      fit(item) {
        const p = item.node.props;
        const body = item.content;
        if (item.fluid) {
          body.style.transform = "";
          item.frame.style.height = item.frame.style.width = "";
          return;
        }
        const avail = item.el.parentElement ? item.el.parentElement.clientWidth - 2 : p.width;
        const scale = avail > 0 ? Math.min(1, avail / p.width) : 1;
        body.style.transform = scale < 1 ? `scale(${scale})` : "";
        item.frame.style.width = `${Math.floor(p.width * scale)}px`;
        item.frame.style.height = `${Math.floor(p.height * scale)}px`;
      },
    },

    gridlayout: {
      create(item) {
        item.el = item.content = h("div", { class: "mui-grid" });
      },
      update(item, p) {
        const s = item.el.style;
        s.gridTemplateRows = (p.rows || []).map(trackSize).join(" ");
        s.gridTemplateColumns = (p.cols || []).map(trackSize).join(" ");
        const [l, b, r, t] = p.padding || [10, 10, 10, 10];
        s.padding = `${t}px ${r}px ${b}px ${l}px`;
        s.rowGap = `${p.rowGap ?? 10}px`;
        s.columnGap = `${p.colGap ?? 10}px`;
        s.overflow = p.scroll ? "auto" : "";
      },
    },

    panel: {
      create(item) {
        item.caption = h("div", { class: "mui-panel-title" });
        item.content = h("div", { class: "mui-panel-body" });
        item.el = h("div", { class: "mui-panel", role: "group" }, item.caption, item.content);
      },
      update(item, p) {
        text(item.caption, p.title);
        item.caption.hidden = !p.title;
        if (p.title) item.el.setAttribute("aria-label", p.title);
        else item.el.removeAttribute("aria-label");
        item.el.classList.toggle("mui-borderless", p.border === false);
        item.content.style.overflow = p.scroll ? "auto" : "";
      },
    },

    tabgroup: {
      create(item) {
        item.tablist = h("div", { class: "mui-tablist", role: "tablist" });
        item.content = h("div", { class: "mui-tabpanels" });
        item.el = h("div", { class: "mui-tabs" }, item.tablist, item.content);
      },
      update(item, p) {
        (item.tabs || []).forEach((t, i) => {
          const on = i === (p.selected || 0);
          t.btn.setAttribute("aria-selected", String(on));
          t.btn.tabIndex = on ? 0 : -1;
          t.panel.classList.toggle("mui-tab-hidden", !on);
        });
      },
    },

    tab: {
      create(item) {
        item.el = item.content = h("div", { class: "mui-tabpanel" });
      },
      update(item, p) {
        if (item.tabButton) text(item.tabButton, p.title || "Tab");
        item.el.hidden = false;
        item.el.style.overflow = p.scroll ? "auto" : "";
      },
    },

    label: {
      create(item) {
        item.el = h("div", { class: "mui-label" });
      },
      update(item, p) {
        text(item.el, p.text);
        item.el.classList.toggle("mui-wrap", !!p.wrap);
      },
    },

    button: {
      create(item, view) {
        item.icon = h("img", { class: "mui-button-icon", alt: "", hidden: true });
        item.caption = h("span");
        item.el = h("button", { type: "button", class: "mui-button", onclick: () => view.send({ id: item.node.id, event: "push" }) }, item.icon, item.caption);
      },
      update(item, p) {
        text(item.caption, p.text);
        item.icon.hidden = !p.icon;
        if (p.icon && item.icon.src !== p.icon) item.icon.src = p.icon;
        if (!p.text && p.tooltip) item.el.setAttribute("aria-label", p.tooltip);
      },
    },

    statebutton: {
      create(item, view) {
        item.el = h("button", {
          type: "button", class: "mui-button mui-state",
          onclick: () => {
            const next = !(item.node.props.value === true);
            if (item.node.props.toggle && !next) return; // one button of a group stays pressed
            item.node.props.value = next;
            MAKERS.statebutton.update(item, item.node.props);
            view.send({ id: item.node.id, event: "value", value: next });
          },
        });
      },
      update(item, p) {
        text(item.el, p.text);
        item.el.setAttribute("aria-pressed", String(!!p.value));
      },
    },

    editfield: {
      create(item, view) {
        const id = item.node.id;
        const input = h("input", {
          type: "text", class: "mui-input", autocomplete: "off", spellcheck: "false",
          onchange: () => view.send({ id, event: "value", value: input.value }),
          oninput: () => item.node.props.live && view.sendLive({ id, event: "value", value: input.value, changing: true }),
          onkeydown: (e) => {
            if (e.key === "Enter") input.blur(), input.focus();
          },
        });
        item.el = input;
      },
      update(item, p) {
        const input = item.el;
        if (document.activeElement !== input || !item.node.props.live) setValue(input, p.value ?? "");
        input.placeholder = p.placeholder || "";
        input.readOnly = p.editable === false;
        input.inputMode = p.numeric ? "decimal" : "text";
        input.classList.toggle("mui-numeric", !!p.numeric);
        if (!input.getAttribute("aria-label") || input.dataset.autoLabel) labelFromNeighbor(item);
      },
    },

    textarea: {
      create(item, view) {
        const id = item.node.id;
        const area = h("textarea", {
          class: "mui-input mui-textarea-input", spellcheck: "false",
          onchange: () => view.send({ id, event: "value", value: area.value }),
          oninput: () => item.node.props.live && view.sendLive({ id, event: "value", value: area.value, changing: true }),
        });
        item.el = area;
      },
      update(item, p) {
        if (document.activeElement !== item.el || !item.node.props.live) setValue(item.el, p.value ?? "");
        item.el.placeholder = p.placeholder || "";
        item.el.readOnly = p.editable === false;
        labelFromNeighbor(item);
      },
    },

    dropdown: {
      create(item, view) {
        const id = item.node.id;
        const select = h("select", { class: "mui-input mui-select", onchange: () => view.send({ id, event: "value", value: Number(select.value) }) });
        const input = h("input", {
          type: "text", class: "mui-input", hidden: true, autocomplete: "off",
          onchange: () => {
            const i = (item.node.props.items || []).indexOf(input.value);
            view.send({ id, event: "value", value: i >= 0 ? i : input.value });
          },
        });
        item.select = select;
        item.input = input;
        item.list = h("datalist", { id: `mui-list-${id}` });
        input.setAttribute("list", item.list.id);
        item.el = h("div", { class: "mui-dropdown" }, select, input, item.list);
      },
      update(item, p) {
        const items = p.items || [];
        if (p.editable) {
          item.select.hidden = true;
          item.input.hidden = false;
          const key = JSON.stringify(items);
          if (item.list.dataset.items !== key) {
            item.list.dataset.items = key;
            item.list.textContent = "";
            for (const s of items) item.list.append(h("option", { value: s }));
          }
          setValue(item.input, p.value ?? (items[p.selected] || ""));
        } else {
          item.select.hidden = false;
          item.input.hidden = true;
          fillOptions(item.select, items, p.selected < 0 ? p.placeholder || "" : null);
          setValue(item.select, p.selected);
        }
        labelFromNeighbor(item, p.editable ? item.input : item.select);
      },
    },

    listbox: {
      create(item, view) {
        const id = item.node.id;
        const select = h("select", {
          class: "mui-input mui-listbox-input", size: "4",
          onchange: () => {
            const picked = [...select.selectedOptions].map((o) => Number(o.value));
            view.send({ id, event: "value", value: item.node.props.multi ? picked : picked[0] ?? -1 });
          },
        });
        item.el = select;
      },
      update(item, p) {
        const select = item.el;
        select.multiple = !!p.multi;
        fillOptions(select, p.items || [], null);
        select.size = Math.max(2, Math.min((p.items || []).length, 8));
        const picked = new Set(p.selected || []);
        [...select.options].forEach((o, i) => (o.selected = picked.has(i)));
        labelFromNeighbor(item);
      },
    },

    checkbox: {
      create(item, view) {
        item.input = h("input", { type: "checkbox", onchange: () => view.send({ id: item.node.id, event: "value", value: item.input.checked }) });
        item.caption = h("span");
        item.el = h("label", { class: "mui-check" }, item.input, item.caption);
      },
      update(item, p) {
        item.input.checked = !!p.value;
        text(item.caption, p.text);
      },
    },

    radiobutton: {
      create(item, view) {
        const group = item.parent && item.parent.node.type === "panel" ? `mui-group-${item.parent.node.id}` : `mui-radio-${++groupCounter}`;
        item.input = h("input", { type: "radio", name: group, onchange: () => view.send({ id: item.node.id, event: "value", value: item.input.checked }) });
        item.caption = h("span");
        item.el = h("label", { class: "mui-check" }, item.input, item.caption);
      },
      update(item, p) {
        item.input.checked = !!p.value;
        text(item.caption, p.text);
      },
    },

    switch: {
      create(item, view) {
        item.off = h("span", { class: "mui-switch-text" });
        item.on = h("span", { class: "mui-switch-text" });
        item.button = h("button", {
          type: "button", role: "switch", class: "mui-switch-button",
          onclick: () => {
            const next = item.node.props.selected ? 0 : 1;
            item.node.props.selected = next;
            MAKERS.switch.update(item, item.node.props);
            view.send({ id: item.node.id, event: "value", value: next });
          },
        }, h("span", { class: "mui-switch-knob" }));
        item.el = h("div", { class: "mui-switch-box" }, item.off, item.button, item.on);
      },
      update(item, p) {
        const items = p.items || ["Off", "On"];
        text(item.off, items[0]);
        text(item.on, items[1]);
        item.button.setAttribute("aria-checked", String(p.selected === 1));
        item.button.setAttribute("aria-label", `${items[0]} / ${items[1]}`);
      },
    },

    slider: {
      create(item, view) {
        const id = item.node.id;
        const input = h("input", {
          type: "range", class: "mui-range",
          oninput: () => {
            item.out.textContent = fmt(Number(input.value));
            if (item.node.props.live) view.sendLive({ id, event: "value", value: Number(input.value), changing: true });
          },
          onchange: () => view.send({ id, event: "value", value: Number(input.value) }),
        });
        item.input = input;
        item.out = h("output", { class: "mui-range-value" });
        item.el = h("div", { class: "mui-slider-box" }, input, item.out);
      },
      update(item, p) {
        const input = item.input;
        input.min = p.min;
        input.max = p.max;
        input.step = p.step ? String(p.step) : "any";
        if (document.activeElement !== input || Math.abs(Number(input.value) - p.value) > 1e-9) input.value = p.value;
        item.out.textContent = fmt(Number(p.value));
        item.out.hidden = !!p.classic;
        item.el.classList.toggle("mui-vertical", !!p.vertical);
        labelFromNeighbor(item, input);
      },
    },

    spinner: {
      create(item, view) {
        const id = item.node.id;
        const input = h("input", {
          type: "number", class: "mui-input mui-numeric",
          onchange: () => view.send({ id, event: "value", value: input.value }),
          oninput: () => item.node.props.live && view.sendLive({ id, event: "value", value: Number(input.value), changing: true }),
        });
        item.el = input;
      },
      update(item, p) {
        const input = item.el;
        if (Math.abs(p.min) < 1e300) input.min = p.min;
        else input.removeAttribute("min");
        if (Math.abs(p.max) < 1e300) input.max = p.max;
        else input.removeAttribute("max");
        input.step = p.step || 1;
        if (document.activeElement !== input || Number(input.value) !== p.value) setValue(input, p.value);
        input.readOnly = p.editable === false;
        labelFromNeighbor(item);
      },
    },

    gauge: {
      create(item) {
        item.fill = h("div", { class: "mui-gauge-fill" });
        item.out = h("span", { class: "mui-gauge-value" });
        item.el = h("div", { class: "mui-gauge-box", role: "meter" }, h("div", { class: "mui-gauge-track" }, item.fill), item.out);
      },
      update(item, p) {
        const span = p.max - p.min || 1;
        const frac = Math.max(0, Math.min(1, (p.value - p.min) / span));
        item.fill.style.width = `${frac * 100}%`;
        text(item.out, fmt(p.value));
        item.el.setAttribute("aria-valuemin", p.min);
        item.el.setAttribute("aria-valuemax", p.max);
        item.el.setAttribute("aria-valuenow", p.value);
        if (!item.el.getAttribute("aria-label")) item.el.setAttribute("aria-label", p.tooltip || "Gauge");
      },
    },

    lamp: {
      create(item) {
        item.dot = h("span", { class: "mui-lamp-dot" });
        item.el = h("div", { class: "mui-lamp-box", role: "img" }, item.dot);
      },
      update(item, p) {
        item.dot.style.background = p.color;
        item.el.setAttribute("aria-label", `${p.tooltip || "Lamp"}: ${p.color}`);
      },
    },

    table: {
      create(item) {
        item.table = h("table", { class: "mui-table-grid" });
        item.el = h("div", { class: "mui-table", tabindex: "0", role: "region", "aria-label": "Table" }, item.table);
      },
      update(item, p, view) {
        const key = JSON.stringify([p.columns, p.rows, p.rowNames, p.enabled]);
        if (item.key === key) return;
        const active = document.activeElement && item.el.contains(document.activeElement) ? document.activeElement.dataset.cell : null;
        item.key = key;
        const id = item.node.id;
        const cols = p.columns || [];
        const named = !!p.rowNames;
        const head = h("tr", {}, named ? h("th", { scope: "col" }) : null, cols.map((c) => h("th", { scope: "col" }, c.name)));
        const body = h("tbody");
        (p.rows || []).forEach((row, r) => {
          const tr = h("tr", {}, named ? h("th", { scope: "row" }, p.rowNames[r] ?? "") : null);
          row.forEach((value, c) => {
            const col = cols[c] || { type: "text" };
            const commit = (v) => view.send({ id, event: "cell", row: r, col: c, value: v });
            const label = `${col.name}, row ${r + 1}`;
            let content;
            if (col.type === "bool") {
              content = h("input", { type: "checkbox", "aria-label": label, "data-cell": `${r},${c}`, disabled: !col.editable || p.enabled === false, onchange: (e) => commit(e.target.checked) });
              content.checked = value === true;
            } else if (col.editable && col.options) {
              content = h("select", { class: "mui-cell-input", "aria-label": label, "data-cell": `${r},${c}`, onchange: (e) => commit(e.target.value) }, col.options.map((o) => h("option", { value: o }, o)));
              content.value = value;
            } else if (col.editable && p.enabled !== false) {
              content = h("input", {
                type: "text", class: "mui-cell-input" + (col.type === "num" ? " mui-numeric" : ""), "aria-label": label, "data-cell": `${r},${c}`,
                onchange: (e) => commit(e.target.value),
                onkeydown: (e) => {
                  if (e.key === "Enter") e.target.blur(), e.target.focus();
                },
              });
              content.value = value;
            } else {
              content = String(value);
            }
            const td = h("td", { class: col.type === "num" ? "mui-numeric" : col.type === "bool" ? "mui-center" : "" }, content);
            if (p.selectable) td.addEventListener("click", () => view.send({ id, event: "select", row: r, col: c }));
            tr.append(td);
          });
          body.append(tr);
        });
        item.table.textContent = "";
        item.table.append(h("thead", {}, head), body);
        if (active) {
          const again = item.table.querySelector(`[data-cell="${active}"]`);
          if (again) again.focus({ preventScroll: true });
        }
      },
    },

    axes: {
      create(item, view) {
        item.img = h("img", { class: "mui-axes-image", draggable: "false" });
        item.el = h("div", { class: "mui-axes-box" }, item.img);
        view.watch(item.el, () => view.measureSoon());
      },
      update(item, p) {
        if (item.img.src !== p.image) item.img.src = p.image;
        item.img.alt = p.alt || "Plot";
        item.el.style.setProperty("--mui-ax-w", `${p.width}px`);
        item.el.style.setProperty("--mui-ax-h", `${p.height}px`);
        item.el.style.aspectRatio = "";
      },
    },

    image: {
      create(item, view) {
        item.img = h("img", { class: "mui-image-img", draggable: "false" });
        item.el = h("div", { class: "mui-image-box", onclick: () => item.node.props.clickable && view.send({ id: item.node.id, event: "push" }) }, item.img);
      },
      update(item, p) {
        if (p.src) {
          if (item.img.src !== p.src) item.img.src = p.src;
          item.img.hidden = false;
        } else {
          item.img.hidden = true;
        }
        item.img.alt = p.alt || p.tooltip || "";
        item.img.style.objectFit = { fit: "contain", fill: "cover", none: "none", stretch: "fill", scaledown: "scale-down", scaleup: "contain" }[p.scale] || "contain";
      },
    },

    datepicker: {
      create(item, view) {
        const input = h("input", { type: "date", class: "mui-input", onchange: () => view.send({ id: item.node.id, event: "value", value: input.value }) });
        item.el = input;
      },
      update(item, p) {
        setValue(item.el, p.value || "");
        labelFromNeighbor(item);
      },
    },

    hyperlink: {
      create(item, view) {
        item.el = h("a", { class: "mui-link", target: "_blank", rel: "noopener noreferrer", onclick: () => view.send({ id: item.node.id, event: "push" }) });
      },
      update(item, p) {
        text(item.el, p.text);
        const url = safeUrl(p.url);
        if (url) item.el.href = url;
        else item.el.removeAttribute("href");
        if (!url) item.el.tabIndex = 0;
      },
    },

    menu: {
      create(item, view) {
        const id = item.node.id;
        const top = !item.parent || item.parent.node.type !== "menu";
        item.button = h("button", {
          type: "button", role: "menuitem", class: "mui-menu-button",
          onclick: () => {
            if (item.list) {
              const open = item.list.hidden;
              closeMenus(view, item);
              item.list.hidden = !open;
              item.button.setAttribute("aria-expanded", String(open));
            } else {
              closeMenus(view, null);
              view.send({ id, event: "push" });
            }
          },
        });
        item.el = h("div", { class: "mui-menu" + (top ? " mui-menu-top" : "") }, item.button);
        if (top) {
          item.el.addEventListener("keydown", (e) => {
            if (e.key === "Escape") closeMenus(view, null), item.button.focus();
          });
          item.el.addEventListener("focusout", (e) => {
            if (!item.el.contains(e.relatedTarget)) closeMenus(view, null);
          });
        }
      },
      update(item, p) {
        text(item.button, (p.checked ? "✓ " : "") + (p.text || "Menu"));
        item.el.classList.toggle("mui-separator", !!p.separator);
        if (p.checked) item.button.setAttribute("aria-checked", "true");
        else item.button.removeAttribute("aria-checked");
      },
    },
  };

  MAKERS.knob = MAKERS.slider; // a knob is shown as a slider

  function closeMenus(view, except) {
    for (const it of view.items.values()) {
      if (it.node.type !== "menu" || !it.list) continue;
      if (except && (it === except || it.el.contains(except.el))) continue;
      it.list.hidden = true;
      it.button.setAttribute("aria-expanded", "false");
    }
  }

  /** Give a field without a label of its own the text of the label that sits just before it. */
  function labelFromNeighbor(item, target) {
    const el = target || item.el;
    const p = item.node.props;
    const siblings = item.parent ? item.parent.node.children || [] : [];
    const i = siblings.indexOf(item.node);
    let name = p.tooltip || "";
    const prev = i > 0 ? siblings[i - 1] : null;
    if (prev && prev.type === "label" && prev.props.text) name = prev.props.text;
    else if (!name && p.placeholder) name = p.placeholder;
    else if (!name && p.tag) name = p.tag;
    if (name) {
      el.setAttribute("aria-label", name.replace(/[:\s]+$/, ""));
      el.dataset.autoLabel = "1";
    }
  }

  /* ------------------------------------------------------------------ public */

  global.MylabUI = {
    mount(container, opts) {
      return new View(container, opts);
    },
    /** Short text for a callback error (first line only). */
    firstLine: (s) => String(s || "").split("\n")[0],
    version: 1,
  };
})(typeof globalThis !== "undefined" ? globalThis : window);
