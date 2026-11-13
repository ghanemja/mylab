// Scope windows: time plots (one axes per input port) and XY plots, drawn on a canvas.
// Data arrives in chunks while a simulation runs, so the curves grow live.

import { el, icon, downloadBlob } from "../ui.js";

const LIGHT = ["#0b6bcb", "#c2410c", "#0f766e", "#9333ea", "#b45309", "#be185d", "#4d7c0f"];
const DARK = ["#7cc4ff", "#fb923c", "#5eead4", "#c4a3ff", "#facc15", "#f9a8d4", "#bef264"];

function niceTicks(lo, hi, target) {
  if (!(hi > lo)) return [lo];
  const raw = (hi - lo) / Math.max(target, 1);
  const mag = Math.pow(10, Math.floor(Math.log10(raw)));
  const step = [1, 2, 5, 10].map((m) => m * mag).find((s) => s >= raw) || raw;
  const out = [];
  for (let v = Math.ceil(lo / step - 1e-9) * step; v <= hi + step * 1e-9; v += step) out.push(Math.abs(v) < step * 1e-9 ? 0 : v);
  return out;
}

export function formatNumber(v) {
  if (v === null || v === undefined || Number.isNaN(v)) return "NaN";
  if (v === 0) return "0";
  const a = Math.abs(v);
  if (a >= 1e5 || a < 1e-3) return v.toExponential(2).replace(/\.?0+e/, "e");
  return String(Number(v.toPrecision(5)));
}

class ScopePanel {
  constructor(dock, meta) {
    this.dock = dock;
    this.meta = meta;
    this.t = [];
    this.sig = (meta.names || [""]).map(() => []); // per port: list of columns
    this.view = null; // { x0, x1 } while zoomed
    this.cursor = null; // index into t
    this.legend = true;
    this.dirty = true;
    this.running = false;
    this.span = null; // [t0, tf] of the run, so the axis does not jump while data arrives
    this.canvas = el("canvas", { class: "scope-canvas", tabindex: "0", role: "img", "aria-label": `Plot of ${meta.path}` });
    this.readout = el("div", { class: "scope-readout small muted", "aria-live": "off" }, "Run the model to see data here.");
    const btn = (title, ic, fn, extra = {}) => el("button", { class: "btn icon sm ghost", type: "button", title, "aria-label": title, onclick: fn, ...extra }, icon(ic));
    this.legendBtn = btn("Show or hide the legend", "outline", () => this.toggleLegend(), { "aria-pressed": "true" });
    this.el = el(
      "section",
      { class: "scope-panel", "data-scope": meta.path, "aria-label": `Scope ${meta.path}` },
      el(
        "header",
        { class: "scope-head" },
        el("strong", { class: "scope-title" }, meta.title || meta.path),
        el("span", { class: "spacer" }),
        this.legendBtn,
        btn("Show everything (Home)", "expand", () => this.resetZoom()),
        btn("Save the plot as PNG", "image", () => this.exportPng()),
        btn("Save the data as CSV", "table", () => this.exportCsv()),
        btn("Close", "close", () => dock.close(meta.path))
      ),
      this.canvas,
      this.readout
    );
    this.bind();
    this.observer = new ResizeObserver(() => this.redrawSoon());
    this.observer.observe(this.canvas);
  }

  get isXY() {
    return this.meta.kind === "xy";
  }

  reset(meta, span) {
    this.meta = meta;
    this.t = [];
    this.sig = (meta.names || [""]).map(() => []);
    this.view = null;
    this.cursor = null;
    this.span = span || null;
    this.running = true;
    this.el.querySelector(".scope-title").textContent = meta.title || meta.path;
    this.readout.textContent = "Running…";
    this.redrawSoon();
  }

  append(chunk) {
    const n0 = this.t.length;
    for (const v of chunk.t) this.t.push(v);
    chunk.signals.forEach((rows, p) => {
      const cols = (this.sig[p] = this.sig[p] || []);
      const width = rows.reduce((w, r) => Math.max(w, Array.isArray(r) ? r.length : 1), 0);
      while (cols.length < width) cols.push(new Array(n0).fill(null));
      rows.forEach((r, i) => {
        for (let c = 0; c < cols.length; c++) cols[c][n0 + i] = Array.isArray(r) ? (r[c] ?? r[0] ?? null) : r;
      });
    });
    this.redrawSoon();
  }

  finish() {
    this.running = false;
    this.span = null;
    this.updateFinal();
    this.showReadout();
    this.redrawSoon();
  }

  /** Last value of every curve: also put on the element, where tests and screen readers can read it. */
  updateFinal() {
    const last = this.sig.map((cols) => cols.map((c) => c[c.length - 1]));
    this.el.dataset.points = String(this.t.length);
    this.el.dataset.final = JSON.stringify(last);
    this.el.dataset.tEnd = String(this.t[this.t.length - 1] ?? "");
    this.canvas.setAttribute("aria-label", `Plot of ${this.meta.path}: ${this.t.length} points, final values ${last.flat().map(formatNumber).join(", ")}`);
  }

  names() {
    const out = [];
    this.sig.forEach((cols, p) => {
      const base = (this.meta.names && this.meta.names[p]) || (this.sig.length > 1 ? `input ${p + 1}` : "signal");
      out.push(cols.map((_, c) => (cols.length > 1 ? `${base}(${c + 1})` : base)));
    });
    return out;
  }

  toggleLegend() {
    this.legend = !this.legend;
    this.legendBtn.setAttribute("aria-pressed", String(this.legend));
    this.redrawSoon();
  }

  resetZoom() {
    this.view = null;
    this.redrawSoon();
  }

  redrawSoon() {
    if (this.pending) return;
    this.pending = requestAnimationFrame(() => {
      this.pending = 0;
      this.draw();
    });
  }

  /* ---------- drawing ---------- */

  xRange() {
    if (this.isXY) return this.range(this.sig[0] && this.sig[0][0]);
    if (this.view) return [this.view.x0, this.view.x1];
    if (this.span && this.running) return this.span;
    const t = this.t;
    return t.length ? [t[0], t[t.length - 1] > t[0] ? t[t.length - 1] : t[0] + 1] : [0, 1];
  }

  range(arr, i0 = 0, i1 = arr ? arr.length : 0) {
    let lo = Infinity, hi = -Infinity;
    for (let i = i0; i < i1; i++) {
      const v = arr[i];
      if (v === null || v === undefined || !Number.isFinite(v)) continue;
      if (v < lo) lo = v;
      if (v > hi) hi = v;
    }
    if (lo === Infinity) return [0, 1];
    if (hi === lo) {
      const pad = Math.abs(lo) * 0.1 || 0.5;
      return [lo - pad, hi + pad];
    }
    const pad = (hi - lo) * 0.06;
    return [lo - pad, hi + pad];
  }

  visibleIndices() {
    const [x0, x1] = this.xRange();
    const t = this.t;
    let i0 = 0, i1 = t.length;
    if (this.view) {
      i0 = Math.max(lowerBound(t, x0) - 1, 0);
      i1 = Math.min(lowerBound(t, x1) + 1, t.length);
    }
    return [i0, i1];
  }

  layout() {
    const w = this.canvas.clientWidth, h = this.canvas.clientHeight;
    const n = this.isXY ? 1 : Math.max(this.sig.length, 1);
    const left = 54, right = 12, top = 8, bottom = 24, gap = 10;
    const each = Math.max((h - top - bottom - gap * (n - 1)) / n, 20);
    const axes = [];
    for (let k = 0; k < n; k++) axes.push({ x: left, y: top + k * (each + gap), w: Math.max(w - left - right, 20), h: each });
    return { w, h, axes };
  }

  draw() {
    const cv = this.canvas;
    const dpr = window.devicePixelRatio || 1;
    const w = cv.clientWidth, h = cv.clientHeight;
    if (!w || !h) return;
    if (cv.width !== Math.round(w * dpr) || cv.height !== Math.round(h * dpr)) {
      cv.width = Math.round(w * dpr);
      cv.height = Math.round(h * dpr);
    }
    const g = cv.getContext("2d");
    g.setTransform(dpr, 0, 0, dpr, 0, 0);
    const css = getComputedStyle(cv);
    const col = { bg: css.getPropertyValue("--panel").trim() || "#fff", text: css.getPropertyValue("--text").trim() || "#000",
      muted: css.getPropertyValue("--muted").trim() || "#666", grid: css.getPropertyValue("--border").trim() || "#ddd",
      frame: css.getPropertyValue("--border-strong").trim() || "#bbb" };
    const dark = luminance(col.bg) < 0.4;
    const palette = dark ? DARK : LIGHT;
    const contrast = document.documentElement.dataset.theme === "hc";
    g.fillStyle = col.bg;
    g.fillRect(0, 0, w, h);
    g.font = "11px system-ui, sans-serif";
    const lay = this.layout();
    const [x0, x1] = this.xRange();
    const [i0, i1] = this.isXY ? [0, this.t.length] : this.visibleIndices();
    const names = this.names();
    this.geom = [];
    lay.axes.forEach((ax, k) => {
      const series = this.isXY ? [(this.sig[1] && this.sig[1][0]) || []] : this.sig[k] || [];
      let lo = Infinity, hi = -Infinity;
      for (const s of series) {
        const [a, b] = this.range(s, i0, i1);
        lo = Math.min(lo, a);
        hi = Math.max(hi, b);
      }
      if (lo === Infinity) [lo, hi] = [0, 1];
      const X = (v) => ax.x + ((v - x0) / (x1 - x0)) * ax.w;
      const Y = (v) => ax.y + ax.h - ((v - lo) / (hi - lo)) * ax.h;
      this.geom.push({ ax, X, Y, lo, hi, x0, x1 });
      // grid and tick labels
      g.strokeStyle = col.grid;
      g.fillStyle = col.muted;
      g.lineWidth = 1;
      g.textAlign = "right";
      g.textBaseline = "middle";
      for (const v of niceTicks(lo, hi, Math.max(Math.floor(ax.h / 28), 2))) {
        const y = Math.round(Y(v)) + 0.5;
        g.beginPath();
        g.moveTo(ax.x, y);
        g.lineTo(ax.x + ax.w, y);
        g.stroke();
        g.fillText(formatNumber(v), ax.x - 6, y);
      }
      g.textAlign = "center";
      g.textBaseline = "top";
      const lastAxes = k === lay.axes.length - 1;
      for (const v of niceTicks(x0, x1, Math.max(Math.floor(ax.w / 70), 2))) {
        const x = Math.round(X(v)) + 0.5;
        g.beginPath();
        g.moveTo(x, ax.y);
        g.lineTo(x, ax.y + ax.h);
        g.stroke();
        if (lastAxes) g.fillText(formatNumber(v), x, ax.y + ax.h + 5);
      }
      g.strokeStyle = col.frame;
      g.strokeRect(ax.x + 0.5, ax.y + 0.5, ax.w, ax.h);
      // curves
      g.save();
      g.beginPath();
      g.rect(ax.x, ax.y, ax.w, ax.h);
      g.clip();
      g.lineWidth = contrast ? 2.4 : 1.6;
      g.lineJoin = "round";
      series.forEach((s, c) => {
        g.strokeStyle = palette[c % palette.length];
        g.beginPath();
        const xs = this.isXY ? (this.sig[0] && this.sig[0][0]) || [] : this.t;
        this.trace(g, xs, s, i0, i1, X, Y, ax.w, !this.isXY && this.meta.stairs);
        g.stroke();
      });
      // cursor
      if (this.cursor !== null && !this.isXY && this.t.length) {
        const tc = this.t[this.cursor];
        const x = Math.round(X(tc)) + 0.5;
        g.strokeStyle = col.text;
        g.setLineDash([4, 3]);
        g.lineWidth = 1;
        g.beginPath();
        g.moveTo(x, ax.y);
        g.lineTo(x, ax.y + ax.h);
        g.stroke();
        g.setLineDash([]);
        series.forEach((s, c) => {
          const v = s[this.cursor];
          if (v === null || v === undefined) return;
          g.fillStyle = palette[c % palette.length];
          g.beginPath();
          g.arc(X(tc), Y(v), 3.5, 0, 2 * Math.PI);
          g.fill();
        });
      }
      // zoom rubber band
      if (this.drag && this.drag.moved) {
        g.fillStyle = dark ? "rgba(255,255,255,.14)" : "rgba(0,0,0,.10)";
        g.fillRect(Math.min(this.drag.x0, this.drag.x1), ax.y, Math.abs(this.drag.x1 - this.drag.x0), ax.h);
      }
      g.restore();
      // legend
      if (this.legend && series.length) {
        const labels = this.isXY ? [] : names[k] || [];
        if (labels.length && (labels.length > 1 || (this.meta.names && this.meta.names[k]))) {
          g.textAlign = "left";
          g.textBaseline = "middle";
          const widths = labels.map((s) => g.measureText(s).width);
          const bw = Math.max(...widths) + 30, bh = labels.length * 15 + 6;
          const bx = ax.x + ax.w - bw - 6, by = ax.y + 6;
          g.fillStyle = col.bg;
          g.globalAlpha = 0.88;
          g.fillRect(bx, by, bw, bh);
          g.globalAlpha = 1;
          g.strokeStyle = col.grid;
          g.strokeRect(bx + 0.5, by + 0.5, bw, bh);
          labels.forEach((s, c) => {
            g.strokeStyle = palette[c % palette.length];
            g.lineWidth = 2;
            g.beginPath();
            g.moveTo(bx + 6, by + 10 + c * 15);
            g.lineTo(bx + 22, by + 10 + c * 15);
            g.stroke();
            g.fillStyle = col.text;
            g.fillText(s, bx + 26, by + 10 + c * 15);
          });
        }
      }
    });
    if (!this.t.length) {
      g.fillStyle = col.muted;
      g.textAlign = "center";
      g.textBaseline = "middle";
      g.fillText(this.running ? "Waiting for data…" : "No data yet. Run the model.", w / 2, h / 2);
    }
  }

  trace(g, xs, ys, i0, i1, X, Y, width, stairs) {
    const n = i1 - i0;
    let started = false, prevY = 0;
    const move = (x, y) => {
      if (!started) g.moveTo(x, y);
      else if (stairs) {
        g.lineTo(x, prevY);
        g.lineTo(x, y);
      } else g.lineTo(x, y);
      started = true;
      prevY = y;
    };
    if (n > width * 4 && !this.isXY) {
      // many points per pixel: draw the lowest and highest value of each pixel column
      const per = n / width;
      for (let px = 0; px < width; px++) {
        const a = i0 + Math.floor(px * per), b = Math.min(i0 + Math.floor((px + 1) * per), i1);
        let lo = Infinity, hi = -Infinity;
        for (let i = a; i < b; i++) {
          const v = ys[i];
          if (v === null || v === undefined) continue;
          if (v < lo) lo = v;
          if (v > hi) hi = v;
        }
        if (lo === Infinity) continue;
        const x = X(xs[a]);
        move(x, Y(lo));
        if (hi !== lo) g.lineTo(x, Y(hi));
      }
      return;
    }
    for (let i = i0; i < i1; i++) {
      const v = ys[i], x = xs[i];
      if (v === null || v === undefined || x === null || x === undefined) {
        started = false;
        continue;
      }
      move(X(x), Y(v));
    }
  }

  /* ---------- interaction ---------- */

  bind() {
    const cv = this.canvas;
    const pos = (e) => {
      const r = cv.getBoundingClientRect();
      return { x: e.clientX - r.left, y: e.clientY - r.top };
    };
    cv.addEventListener("pointerdown", (e) => {
      if (this.isXY || !this.geom || !this.t.length) return;
      cv.setPointerCapture(e.pointerId);
      const p = pos(e);
      this.drag = { x0: p.x, x1: p.x, moved: false };
    });
    cv.addEventListener("pointermove", (e) => {
      if (this.isXY || !this.geom || !this.t.length) return;
      const p = pos(e);
      if (this.drag) {
        this.drag.x1 = p.x;
        if (Math.abs(this.drag.x1 - this.drag.x0) > 4) this.drag.moved = true;
      }
      this.setCursorAt(p.x);
    });
    const up = () => {
      const d = this.drag;
      this.drag = null;
      if (d && d.moved && this.geom) {
        const ge = this.geom[0];
        const inv = (x) => ge.x0 + ((x - ge.ax.x) / ge.ax.w) * (ge.x1 - ge.x0);
        const a = inv(Math.min(d.x0, d.x1)), b = inv(Math.max(d.x0, d.x1));
        if (b > a) this.view = { x0: Math.max(a, this.t[0]), x1: Math.min(b, this.t[this.t.length - 1]) };
      }
      this.redrawSoon();
    };
    cv.addEventListener("pointerup", up);
    cv.addEventListener("pointercancel", up);
    cv.addEventListener("pointerleave", () => {
      if (!this.drag && document.activeElement !== cv) {
        this.cursor = null;
        this.showReadout();
        this.redrawSoon();
      }
    });
    cv.addEventListener("dblclick", () => this.resetZoom());
    cv.addEventListener(
      "wheel",
      (e) => {
        if (this.isXY || !this.geom || !this.t.length) return;
        e.preventDefault();
        const ge = this.geom[0];
        const at = ge.x0 + ((pos(e).x - ge.ax.x) / ge.ax.w) * (ge.x1 - ge.x0);
        this.zoom(e.deltaY < 0 ? 0.8 : 1.25, at);
      },
      { passive: false }
    );
    cv.addEventListener("keydown", (e) => {
      if (this.isXY || !this.t.length) return;
      const step = e.shiftKey ? 10 : 1;
      if (e.key === "ArrowLeft" || e.key === "ArrowRight") {
        const cur = this.cursor === null ? (e.key === "ArrowRight" ? -1 : this.t.length) : this.cursor;
        this.cursor = Math.max(0, Math.min(this.t.length - 1, cur + (e.key === "ArrowRight" ? step : -step)));
      } else if (e.key === "Home") this.resetZoom();
      else if (e.key === "+" || e.key === "=") this.zoom(0.8);
      else if (e.key === "-") this.zoom(1.25);
      else if (e.key === "Escape") this.cursor = null;
      else return;
      e.preventDefault();
      e.stopPropagation();
      this.showReadout(true);
      this.redrawSoon();
    });
  }

  zoom(factor, at) {
    const [x0, x1] = this.xRange();
    const lo = this.t[0], hi = this.t[this.t.length - 1];
    const c = at === undefined ? (this.cursor !== null ? this.t[this.cursor] : (x0 + x1) / 2) : at;
    let a = c - (c - x0) * factor, b = c + (x1 - c) * factor;
    a = Math.max(a, lo);
    b = Math.min(b, hi);
    this.view = b - a >= (hi - lo) * 0.999 ? null : { x0: a, x1: b };
    this.redrawSoon();
  }

  setCursorAt(px) {
    const ge = this.geom && this.geom[0];
    if (!ge) return;
    const tq = ge.x0 + ((px - ge.ax.x) / ge.ax.w) * (ge.x1 - ge.x0);
    const t = this.t;
    let i = lowerBound(t, tq);
    if (i > 0 && (i >= t.length || tq - t[i - 1] < t[i] - tq)) i--;
    this.cursor = Math.max(0, Math.min(t.length - 1, i));
    this.showReadout();
    this.redrawSoon();
  }

  showReadout(announce = false) {
    if (!this.t.length) return;
    const names = this.names();
    if (this.isXY) {
      this.readout.textContent = `${this.t.length} points`;
      return;
    }
    const i = this.cursor === null ? this.t.length - 1 : this.cursor;
    const parts = [`t = ${formatNumber(this.t[i])}`];
    this.sig.forEach((cols, p) => cols.forEach((c, k) => parts.push(`${names[p][k]} = ${formatNumber(c[i])}`)));
    this.readout.textContent = (this.cursor === null ? "End: " : "") + parts.join("   ");
    this.readout.setAttribute("aria-live", announce ? "polite" : "off");
  }

  /* ---------- export ---------- */

  fileBase() {
    return this.meta.path.replace(/[^\w.-]+/g, "_");
  }

  exportPng() {
    this.canvas.toBlob((blob) => blob && downloadBlob(blob, this.fileBase() + ".png"), "image/png");
  }

  csvText() {
    const names = this.names();
    const head = ["t"], cols = [];
    this.sig.forEach((list, p) => list.forEach((c, k) => {
      head.push(/[",\n]/.test(names[p][k]) ? `"${names[p][k].replace(/"/g, '""')}"` : names[p][k]);
      cols.push(c);
    }));
    const lines = [head.join(",")];
    for (let i = 0; i < this.t.length; i++) lines.push([this.t[i], ...cols.map((c) => (c[i] === null || c[i] === undefined ? "NaN" : c[i]))].join(","));
    return lines.join("\n") + "\n";
  }

  exportCsv() {
    downloadBlob(new Blob([this.csvText()], { type: "text/csv" }), this.fileBase() + ".csv");
  }

  destroy() {
    this.observer.disconnect();
    if (this.pending) cancelAnimationFrame(this.pending);
    this.el.remove();
  }
}

function lowerBound(arr, v) {
  let lo = 0, hi = arr.length;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (arr[mid] < v) lo = mid + 1;
    else hi = mid;
  }
  return lo;
}

function luminance(color) {
  const m = /^#?([0-9a-f]{2})([0-9a-f]{2})([0-9a-f]{2})$/i.exec(color) || /^#?([0-9a-f])([0-9a-f])([0-9a-f])$/i.exec(color);
  if (m) {
    const [r, g, b] = m.slice(1).map((h) => parseInt(h.length === 1 ? h + h : h, 16) / 255);
    return 0.2126 * r + 0.7152 * g + 0.0722 * b;
  }
  const n = /rgba?\(([^)]+)\)/.exec(color);
  if (n) {
    const [r, g, b] = n[1].split(",").map((s) => parseFloat(s) / 255);
    return 0.2126 * r + 0.7152 * g + 0.0722 * b;
  }
  return 1;
}

/** The strip of scope windows under the canvas. */
export class ScopeDock {
  constructor(host, { onChange } = {}) {
    this.host = host;
    this.onChange = onChange || (() => {});
    this.panels = new Map();
    this.store = new Map(); // path -> { meta, chunks } of the last run, for scopes opened later
    host.hidden = true;
  }

  has(path) {
    return this.panels.has(path);
  }

  open(path, meta) {
    let p = this.panels.get(path);
    if (!p) {
      const saved = this.store.get(path);
      p = new ScopePanel(this, meta || (saved && saved.meta) || { path, kind: "scope", title: "", names: [""] });
      this.panels.set(path, p);
      this.host.appendChild(p.el);
      if (saved) {
        for (const c of saved.chunks) p.append(c);
        if (!saved.running) p.finish();
        else p.running = true;
      }
    }
    this.host.hidden = false;
    this.onChange();
    return p;
  }

  close(path) {
    const p = this.panels.get(path);
    if (!p) return;
    p.destroy();
    this.panels.delete(path);
    this.host.hidden = this.panels.size === 0;
    this.onChange();
  }

  closeAll() {
    for (const path of [...this.panels.keys()]) this.close(path);
  }

  /** A run starts: forget the old data; open the scopes of the model (up to max). */
  start(metas, span, { autoOpen = 3 } = {}) {
    this.store.clear();
    for (const m of metas) this.store.set(m.path, { meta: m, chunks: [], running: true });
    for (const [path, p] of this.panels) {
      const m = metas.find((x) => x.path === path);
      if (m) p.reset(m, span);
      else this.close(path);
    }
    for (const m of metas.slice(0, autoOpen)) if (!this.panels.has(m.path)) this.open(m.path, m).reset(m, span);
  }

  append(chunks) {
    for (const c of chunks || []) {
      const s = this.store.get(c.path);
      if (s) s.chunks.push(c);
      const p = this.panels.get(c.path);
      if (p) p.append(c);
    }
  }

  finish() {
    for (const s of this.store.values()) s.running = false;
    for (const p of this.panels.values()) p.finish();
  }

  redraw() {
    for (const p of this.panels.values()) p.redrawSoon();
  }
}
