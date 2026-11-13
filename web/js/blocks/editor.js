// The diagram editor: an SVG canvas with blocks and lines, edited by mouse, touch or keyboard.
// It knows nothing about the rest of the app; index.js connects it to files and the engine.

import {
  GRID, parseModel, serializeModel, systemAt, portCounts, portPoint, findType, makeBlock, uniqueName, pruneLines, renumberPorts,
  unconnectedInputs, bounds, needsLayout, defaultSize,
} from "./model.js";
import { routeAll, routeLine, branchPoints, pathData, autoLayout, alignBlocks } from "./route.js";
import { svg, drawBlock, drawLine } from "./draw.js";

const CLIP_KEY = "mylab.blocks.clipboard";
let clipboard = null;
const snap = (v, g = GRID / 2) => Math.round(v / g) * g;
const clone = (v) => JSON.parse(JSON.stringify(v));

export class DiagramEditor {
  /**
   * host: element to fill. hooks: { onChange(text), toast(msg, kind), editParams(block, sys), quickInsert(point, from),
   * editLabel(line), rename(block), contextMenu(event, items), announce(text), onNavigate(), openScope(path) }
   */
  constructor(host, hooks) {
    this.hooks = hooks;
    this.model = null;
    this.path = [];
    this.sel = new Set();
    this.selLines = new Set();
    this.view = { x: 40, y: 30, k: 1 };
    this.undoStack = [];
    this.redoStack = [];
    this.diagnostics = [];
    this.displays = new Map();
    this.pointers = new Map();
    this.drag = null;
    this.wire = null;
    this.readOnly = false;

    this.gWorld = svg("g", { class: "bd-world" });
    this.gLines = svg("g", { class: "bd-lines" });
    this.gBlocks = svg("g", { class: "bd-blocks" });
    this.gOver = svg("g", { class: "bd-overlay" });
    this.gWorld.append(this.gLines, this.gBlocks, this.gOver);
    this.pattern = svg("pattern", { id: "bd-grid", width: 20, height: 20, patternUnits: "userSpaceOnUse" }, svg("circle", { class: "bd-grid-dot", cx: 1, cy: 1, r: 1 }));
    this.svg = svg(
      "svg",
      { class: "bd-svg", tabindex: "0", role: "application", "aria-label": "Block diagram" },
      svg("defs", {}, this.pattern),
      svg("rect", { class: "bd-bg", x: 0, y: 0, width: "100%", height: "100%", fill: "url(#bd-grid)" }),
      this.gWorld
    );
    host.append(this.svg);
    this.bind();
  }

  /* ---------- model ---------- */

  /** Load model text. Returns an error message, or null. */
  setText(text, { keepView = false } = {}) {
    let model;
    try {
      model = parseModel(text);
    } catch (e) {
      return e.message;
    }
    this.model = model;
    this.lastText = text;
    if (!systemAt(model, this.path)) this.path = [];
    this.sel.clear();
    this.selLines.clear();
    this.undoStack = [];
    this.redoStack = [];
    if (needsLayout(model)) {
      this.eachSystem(model, (s) => autoLayout(s));
      this.emitChange();
    }
    this.render();
    if (!keepView) requestAnimationFrame(() => this.fit());
    return null;
  }

  eachSystem(sys, fn) {
    fn(sys);
    for (const b of sys.blocks) if (Array.isArray(b.blocks)) this.eachSystem(b, fn);
  }

  text() {
    return serializeModel(this.model);
  }

  sys() {
    return systemAt(this.model, this.path) || this.model;
  }

  prefix() {
    return this.path.length ? this.path.join("/") + "/" : "";
  }

  block(name) {
    return this.sys().blocks.find((b) => b.name === name) || null;
  }

  selectedBlocks() {
    return this.sys().blocks.filter((b) => this.sel.has(b.name));
  }

  snapshot() {
    return JSON.stringify({ m: this.model, p: this.path });
  }

  /** Call before changing the model; pair with commit(). */
  begin() {
    this.pending = this.snapshot();
  }

  /** The model changed: remember the old state for undo, save, redraw. */
  commit() {
    const sys = this.sys();
    pruneLines(sys);
    if (this.pending && this.pending !== this.snapshot()) {
      this.undoStack.push(this.pending);
      if (this.undoStack.length > 200) this.undoStack.shift();
      this.redoStack = [];
    }
    this.pending = null;
    this.emitChange();
    this.render();
  }

  /** Run fn as one undoable change. */
  change(fn) {
    if (this.readOnly) return;
    this.begin();
    fn(this.sys());
    this.commit();
  }

  emitChange() {
    this.lastText = this.text();
    this.hooks.onChange && this.hooks.onChange(this.lastText);
  }

  restore(snapshot) {
    const s = JSON.parse(snapshot);
    this.model = s.m;
    this.path = systemAt(s.m, s.p) ? s.p : [];
    this.sel = new Set([...this.sel].filter((n) => this.block(n)));
    this.selLines.clear();
    this.emitChange();
    this.render();
    this.hooks.onNavigate && this.hooks.onNavigate();
  }

  undo() {
    if (!this.undoStack.length) return this.say("Nothing to undo.");
    this.redoStack.push(this.snapshot());
    this.restore(this.undoStack.pop());
    this.say("Undone.");
  }

  redo() {
    if (!this.redoStack.length) return this.say("Nothing to redo.");
    this.undoStack.push(this.snapshot());
    this.restore(this.redoStack.pop());
    this.say("Redone.");
  }

  say(text) {
    this.hooks.announce && this.hooks.announce(text);
  }

  /* ---------- drawing ---------- */

  setDiagnostics(list) {
    this.diagnostics = list || [];
    this.render();
  }

  setDisplays(map) {
    this.displays = new Map(Object.entries(map || {}));
    this.render();
  }

  /** Problems of the current system by block name: engine diagnostics plus unconnected inputs found here. */
  markers() {
    const out = new Map();
    const open = new Map();
    const prefix = this.prefix();
    const rank = { error: 3, warning: 2, info: 1 };
    const put = (name, level, message) => {
      const cur = out.get(name);
      if (!cur || rank[level] > rank[cur.level]) out.set(name, { level, message, more: cur ? [cur.message, ...(cur.more || [])] : [] });
      else cur.more = [...(cur.more || []), message];
    };
    for (const u of unconnectedInputs(this.sys())) {
      if (!open.has(u.block)) open.set(u.block, new Set());
      open.get(u.block).add("in:" + u.port);
      put(u.block, "warning", `Input ${u.port} is not connected (it is read as zero).`);
    }
    for (const d of this.diagnostics) {
      if (d.level === "info" || d.kind === "unconnected-input") continue;
      for (const p of d.paths && d.paths.length ? d.paths : [d.path]) {
        if (!p || !p.startsWith(prefix)) continue;
        const name = this.sys().blocks.map((b) => b.name).filter((n) => p === prefix + n || p.startsWith(prefix + n + "/")).sort((a, b) => b.length - a.length)[0];
        if (name) put(name, d.level === "error" ? "error" : "warning", d.message);
      }
    }
    // unconnected inputs deeper down show on the subsystem block
    const deep = (b) => Array.isArray(b.blocks) && (unconnectedInputs(b).length > 0 || b.blocks.some(deep));
    for (const b of this.sys().blocks) if (deep(b) && !out.has(b.name)) put(b.name, "warning", "A block inside has an input that is not connected.");
    return { markers: out, open };
  }

  render() {
    if (!this.model) return;
    const sys = this.sys();
    const { markers, open } = this.markers();
    this.lastMarkers = markers;
    const prefix = this.prefix();
    this.gBlocks.replaceChildren(
      ...sys.blocks.map((b) =>
        drawBlock(b, {
          selected: this.sel.has(b.name), handles: this.sel.size === 1 && !this.readOnly, marker: markers.get(b.name), openPorts: open.get(b.name),
          display: this.displays.has(prefix + b.name) ? this.displays.get(prefix + b.name) : undefined,
        })
      )
    );
    this.routes = routeAll(sys);
    const lines = this.routes.map((r) => drawLine(r, sys.lines.indexOf(r.line), { selected: this.selLines.has(r.line) }));
    const dots = branchPoints(this.routes).map((p) => svg("circle", { class: "bd-branch", cx: p[0], cy: p[1], r: 3.2 }));
    this.gLines.replaceChildren(...lines, ...dots);
    this.applyView();
    const n = sys.blocks.length;
    const problems = [...markers.values()].filter((m) => m.level !== "info").length;
    this.svg.setAttribute(
      "aria-label",
      `Block diagram ${[this.model.name, ...this.path].join(" / ")}: ${n} block${n === 1 ? "" : "s"}, ${sys.lines.length} line${sys.lines.length === 1 ? "" : "s"}` +
        (problems ? `, ${problems} with problems` : "") + ". Tab moves between blocks; press ? for the keys."
    );
    this.hooks.onRender && this.hooks.onRender();
  }

  applyView() {
    const { x, y, k } = this.view;
    this.gWorld.setAttribute("transform", `translate(${x} ${y}) scale(${k})`);
    const g = 20 * k;
    this.pattern.setAttribute("width", g);
    this.pattern.setAttribute("height", g);
    this.pattern.setAttribute("patternTransform", `translate(${x % g} ${y % g})`);
    this.svg.classList.toggle("bd-far", k < 0.45);
  }

  toWorld(e) {
    const r = this.svg.getBoundingClientRect();
    return { x: (e.clientX - r.left - this.view.x) / this.view.k, y: (e.clientY - r.top - this.view.y) / this.view.k };
  }

  center() {
    const r = this.svg.getBoundingClientRect();
    return { x: (r.width / 2 - this.view.x) / this.view.k, y: (r.height / 2 - this.view.y) / this.view.k };
  }

  zoomBy(factor, at) {
    const r = this.svg.getBoundingClientRect();
    const px = at ? at.x : r.width / 2, py = at ? at.y : r.height / 2;
    const k = Math.max(0.15, Math.min(4, this.view.k * factor));
    this.view.x = px - ((px - this.view.x) / this.view.k) * k;
    this.view.y = py - ((py - this.view.y) / this.view.k) * k;
    this.view.k = k;
    this.applyView();
    this.hooks.onRender && this.hooks.onRender();
  }

  fit() {
    const r = this.svg.getBoundingClientRect();
    if (!r.width || !r.height || !this.model) return;
    const pts = [];
    for (const b of this.sys().blocks) pts.push([b.position[0], b.position[1]], [b.position[0] + b.position[2], b.position[1] + b.position[3] + 18]);
    for (const rt of this.routes || []) pts.push(...rt.pts);
    if (!pts.length) {
      this.view = { x: 40, y: 30, k: 1 };
    } else {
      const x0 = Math.min(...pts.map((p) => p[0])), x1 = Math.max(...pts.map((p) => p[0]));
      const y0 = Math.min(...pts.map((p) => p[1])), y1 = Math.max(...pts.map((p) => p[1]));
      const k = Math.max(0.15, Math.min(1.5, (r.width - 60) / Math.max(x1 - x0, 1), (r.height - 60) / Math.max(y1 - y0, 1)));
      this.view = { k, x: (r.width - (x1 - x0) * k) / 2 - x0 * k, y: (r.height - (y1 - y0) * k) / 2 - y0 * k };
    }
    this.applyView();
    this.hooks.onRender && this.hooks.onRender();
  }

  /** Scroll so a block is visible. */
  reveal(name) {
    const b = this.block(name);
    if (!b) return;
    const r = this.svg.getBoundingClientRect();
    const { k } = this.view;
    const sx = b.position[0] * k + this.view.x, sy = b.position[1] * k + this.view.y;
    const w = b.position[2] * k, h = b.position[3] * k + 20;
    if (sx < 20) this.view.x += 20 - sx;
    else if (sx + w > r.width - 20) this.view.x -= sx + w - (r.width - 20);
    if (sy < 20) this.view.y += 20 - sy;
    else if (sy + h > r.height - 20) this.view.y -= sy + h - (r.height - 20);
    this.applyView();
  }

  /* ---------- selection and navigation ---------- */

  select(names, { add = false } = {}) {
    if (!add) {
      this.sel.clear();
      this.selLines.clear();
    }
    for (const n of names) this.sel.add(n);
    this.render();
    this.describeSelection();
  }

  describeSelection() {
    const bs = this.selectedBlocks();
    if (bs.length === 1) {
      const b = bs[0], t = findType(b.type), c = portCounts(b), m = this.lastMarkers && this.lastMarkers.get(b.name);
      this.say(`${b.name}, ${t ? t.title : b.type}, ${c.in} input${c.in === 1 ? "" : "s"}, ${c.out} output${c.out === 1 ? "" : "s"}${m ? ". " + m.message : ""}`);
    } else if (bs.length > 1) this.say(`${bs.length} blocks selected.`);
  }

  selectAll() {
    this.select(this.sys().blocks.map((b) => b.name));
  }

  /** Blocks in reading order: left to right, then top to bottom. */
  ordered() {
    return this.sys().blocks.slice().sort((a, b) => a.position[0] - b.position[0] || a.position[1] - b.position[1]);
  }

  /** Tab order. Returns false when the selection would leave the list (so focus can move on). */
  cycle(dir) {
    const list = this.ordered();
    if (!list.length) return false;
    const cur = list.findIndex((b) => this.sel.has(b.name));
    const next = cur < 0 ? (dir > 0 ? 0 : list.length - 1) : cur + dir;
    if (next < 0 || next >= list.length) {
      this.select([]);
      return false;
    }
    this.select([list[next].name]);
    this.reveal(list[next].name);
    return true;
  }

  /** Select the nearest block in a direction (Alt+arrow). */
  neighbour(dx, dy) {
    const cur = this.selectedBlocks()[0];
    if (!cur) return this.cycle(1);
    const c = [cur.position[0] + cur.position[2] / 2, cur.position[1] + cur.position[3] / 2];
    let best = null, score = Infinity;
    for (const b of this.sys().blocks) {
      if (b === cur) continue;
      const ox = b.position[0] + b.position[2] / 2 - c[0], oy = b.position[1] + b.position[3] / 2 - c[1];
      const along = ox * dx + oy * dy, across = Math.abs(ox * dy) + Math.abs(oy * dx);
      if (along <= 1) continue;
      const s = along + 2.5 * across;
      if (s < score) [best, score] = [b, s];
    }
    if (best) {
      this.select([best.name]);
      this.reveal(best.name);
    }
    return !!best;
  }

  enter(name) {
    const b = this.block(name);
    if (!b || !Array.isArray(b.blocks)) return;
    this.path.push(name);
    this.sel.clear();
    this.selLines.clear();
    this.render();
    this.fit();
    this.say(`Inside subsystem ${name}. Escape goes back up.`);
    this.hooks.onNavigate && this.hooks.onNavigate();
  }

  leave(levels = 1) {
    if (!this.path.length) return false;
    let from = null;
    for (let i = 0; i < levels && this.path.length; i++) from = this.path.pop();
    this.sel = new Set(from ? [from] : []);
    this.selLines.clear();
    this.render();
    this.fit();
    this.say(`Back in ${this.path.length ? this.path[this.path.length - 1] : this.model.name}.`);
    this.hooks.onNavigate && this.hooks.onNavigate();
    return true;
  }

  goTo(path) {
    if (!systemAt(this.model, path)) return;
    this.path = path.slice();
    this.sel.clear();
    this.selLines.clear();
    this.render();
    this.fit();
    this.hooks.onNavigate && this.hooks.onNavigate();
  }

  /** Open the system that holds a block (engine path "Sub/Block") and select the block. */
  showBlock(fullPath) {
    const parts = String(fullPath || "").split("/");
    for (let cut = parts.length - 1; cut >= 0; cut--) {
      const sysPath = parts.slice(0, cut), name = parts.slice(cut).join("/");
      const sys = systemAt(this.model, sysPath);
      if (sys && sys.blocks.some((b) => b.name === name)) {
        this.path = sysPath;
        this.sel = new Set([name]);
        this.selLines.clear();
        this.render();
        this.fit();
        this.hooks.onNavigate && this.hooks.onNavigate();
        return true;
      }
    }
    return false;
  }

  /* ---------- editing ---------- */

  addBlock(typeName, at, { select = true } = {}) {
    let added = null;
    const p = at || this.freeSpot();
    this.change((sys) => {
      added = makeBlock(sys, typeName, p.x, p.y);
      if (added) sys.blocks.push(added);
    });
    if (added && select) {
      this.select([added.name]);
      this.reveal(added.name);
      this.say(`Added ${added.name}.`);
    }
    return added;
  }

  /** A place for a new block: right of the selection, or the middle of the view, moved until it is free. */
  freeSpot() {
    const sys = this.sys();
    const cur = this.selectedBlocks()[0];
    const p = cur ? { x: cur.position[0] + cur.position[2] + 90, y: cur.position[1] + cur.position[3] / 2 } : this.center();
    const hits = (q) => sys.blocks.some((b) => Math.abs(b.position[0] + b.position[2] / 2 - q.x) < 70 && Math.abs(b.position[1] + b.position[3] / 2 - q.y) < 50);
    for (let i = 0; i < 30 && hits(p); i++) p.y += 70;
    return p;
  }

  deleteSelection() {
    if (!this.sel.size && !this.selLines.size) return;
    const n = this.sel.size, m = this.selLines.size;
    this.change((sys) => {
      sys.lines = sys.lines.filter((l) => !this.selLines.has(l) && !this.sel.has(l.from[0]) && !this.sel.has(l.to[0]));
      sys.blocks = sys.blocks.filter((b) => !this.sel.has(b.name));
      renumberPorts(sys);
      this.sel.clear();
      this.selLines.clear();
    });
    this.say(`Deleted ${n ? n + " block" + (n === 1 ? "" : "s") : ""}${n && m ? " and " : ""}${m ? m + " line" + (m === 1 ? "" : "s") : ""}.`);
  }

  copy() {
    const blocks = this.selectedBlocks();
    if (!blocks.length) return false;
    const lines = this.sys().lines.filter((l) => this.sel.has(l.from[0]) && this.sel.has(l.to[0]));
    clipboard = clone({ blocks, lines });
    try {
      localStorage.setItem(CLIP_KEY, JSON.stringify(clipboard));
    } catch {
      /* private mode: copying still works inside this tab */
    }
    this.say(`Copied ${blocks.length} block${blocks.length === 1 ? "" : "s"}.`);
    return true;
  }

  cut() {
    if (this.copy()) this.deleteSelection();
  }

  paste(at) {
    let clip = clipboard;
    try {
      clip = clip || JSON.parse(localStorage.getItem(CLIP_KEY) || "null");
    } catch {
      /* nothing stored */
    }
    if (!clip || !clip.blocks || !clip.blocks.length) return this.say("Nothing to paste.");
    this.insert(clone(clip), at);
  }

  duplicate() {
    const blocks = this.selectedBlocks();
    if (!blocks.length) return;
    this.insert(clone({ blocks, lines: this.sys().lines.filter((l) => this.sel.has(l.from[0]) && this.sel.has(l.to[0])) }));
  }

  insert(part, at) {
    const names = new Map();
    this.change((sys) => {
      const bb = bounds(part.blocks);
      const overlaps = sys.blocks.some((b) => part.blocks.some((p) => p.position[0] === b.position[0] && p.position[1] === b.position[1]));
      const dx = at ? snap(at.x - bb.x - bb.w / 2, GRID) : overlaps ? 2 * GRID : 0;
      const dy = at ? snap(at.y - bb.y - bb.h / 2, GRID) : overlaps ? 2 * GRID : 0;
      for (const b of part.blocks) {
        const name = uniqueName(sys, b.name);
        names.set(b.name, name);
        b.name = name;
        b.position[0] += dx;
        b.position[1] += dy;
        sys.blocks.push(b);
      }
      for (const l of part.lines) {
        l.from[0] = names.get(l.from[0]);
        l.to[0] = names.get(l.to[0]);
        if (l.points) l.points = l.points.map((p) => [p[0] + dx, p[1] + dy]);
        sys.lines.push(l);
      }
      renumberPorts(sys);
    });
    this.select([...names.values()]);
    this.say(`Pasted ${names.size} block${names.size === 1 ? "" : "s"}.`);
  }

  rotate(dir = 1) {
    if (!this.sel.size) return;
    this.change((sys) => {
      for (const b of this.selectedBlocks()) {
        b.rotation = (((b.rotation || 0) + dir) % 4 + 4) % 4;
        const [x, y, w, h] = b.position;
        if (w !== h) b.position = [snap(x + w / 2 - h / 2), snap(y + h / 2 - w / 2), h, w];
        this.clearRoutes(sys, b.name);
      }
    });
  }

  flip() {
    if (!this.sel.size) return;
    this.change((sys) => {
      for (const b of this.selectedBlocks()) {
        b.flip = !b.flip;
        this.clearRoutes(sys, b.name);
      }
    });
  }

  clearRoutes(sys, name) {
    for (const l of sys.lines) if (l.from[0] === name || l.to[0] === name) delete l.points;
  }

  moveSelection(dx, dy) {
    if (!this.sel.size) return;
    this.change((sys) => {
      for (const b of this.selectedBlocks()) {
        b.position[0] += dx;
        b.position[1] += dy;
      }
      this.fixRoutesAfterMove(sys, dx, dy);
    });
    const first = this.selectedBlocks()[0];
    if (first) this.reveal(first.name);
  }

  /** Hand-made routes: move them with their blocks when both ends moved, otherwise route afresh. */
  fixRoutesAfterMove(sys, dx, dy) {
    for (const l of sys.lines) {
      const a = this.sel.has(l.from[0]), b = this.sel.has(l.to[0]);
      if (!l.points || !l.points.length || (!a && !b)) continue;
      if (a && b) l.points = l.points.map((p) => [p[0] + dx, p[1] + dy]);
      else delete l.points;
    }
  }

  renameBlock(oldName, newName) {
    newName = String(newName).trim().replace(/\s+/g, " ");
    if (!newName || newName === oldName) return null;
    if (this.block(newName)) return `There is already a block named "${newName}" here.`;
    this.change((sys) => {
      const b = sys.blocks.find((x) => x.name === oldName);
      b.name = newName;
      for (const l of sys.lines) {
        if (l.from[0] === oldName) l.from[0] = newName;
        if (l.to[0] === oldName) l.to[0] = newName;
      }
      if (this.sel.delete(oldName)) this.sel.add(newName);
    });
    return null;
  }

  /** Free input ports of a block: [port]. */
  freeInputs(block) {
    const c = portCounts(block);
    const used = new Set(this.sys().lines.filter((l) => l.to[0] === block.name).map((l) => String(l.to[1])));
    const out = [];
    for (let k = 1; k <= c.in; k++) if (!used.has(String(k))) out.push(k);
    for (const ctl of ["enable", "trigger"]) if (c[ctl] && !used.has(ctl)) out.push(ctl);
    return out;
  }

  /** Connect an output to an input. Returns an error text or null. */
  connect(from, to) {
    const sys = this.sys();
    const s = this.block(from.block), d = this.block(to.block);
    if (!s || !d) return "That block is gone.";
    const sc = portCounts(s), dc = portCounts(d);
    if (from.port > sc.out) return `${s.name} has no output ${from.port}.`;
    if (typeof to.port === "number" ? to.port > dc.in : !dc[to.port]) return `${d.name} has no input ${to.port}.`;
    if (sys.lines.some((l) => l.from[0] === from.block && l.from[1] === from.port && l.to[0] === to.block && l.to[1] === to.port)) return "These ports are already connected.";
    let replaced = false;
    this.change((sy) => {
      const before = sy.lines.length;
      sy.lines = sy.lines.filter((l) => !(l.to[0] === to.block && l.to[1] === to.port));
      replaced = sy.lines.length < before;
      sy.lines.push({ from: [from.block, from.port], to: [to.block, to.port] });
    });
    this.say(`Connected ${from.block} output ${from.port} to ${to.block} input ${to.port}.` + (replaced ? " The line that was there was replaced." : ""));
    if (replaced) this.hooks.toast && this.hooks.toast(`Input ${to.port} of ${to.block} takes one line; the old one was replaced. Ctrl+Z brings it back.`);
    return null;
  }

  /**
   * Connect the selected blocks with the keyboard: with two or more selected, each one's first free output
   * goes to the next one's first free input (in the order left to right). Returns an error text or null.
   */
  connectSelected() {
    const list = this.ordered().filter((b) => this.sel.has(b.name));
    if (list.length < 2) return "Select two blocks (Shift+click, or Tab then Shift+Tab with Shift held) to connect them.";
    let made = 0;
    for (let i = 0; i < list.length - 1; i++) {
      const a = list[i], b = list[i + 1];
      const free = this.freeInputs(b);
      const outs = portCounts(a).out;
      if (!outs || !free.length) continue;
      const usedOut = new Set(this.sys().lines.filter((l) => l.from[0] === a.name).map((l) => l.from[1]));
      let port = 1;
      for (let k = 1; k <= outs; k++) if (!usedOut.has(k)) { port = k; break; }
      if (!this.connect({ block: a.name, port }, { block: b.name, port: free[0] })) made++;
    }
    return made ? null : "No free ports to connect: the first block needs an output and the second a free input.";
  }

  layout() {
    this.change((sys) => autoLayout(sys));
    this.fit();
    this.say("Blocks arranged.");
  }

  align(how) {
    if (this.sel.size < 2) return this.hooks.toast && this.hooks.toast("Select two or more blocks to align them.");
    this.change((sys) => {
      alignBlocks(this.selectedBlocks(), how);
      for (const n of this.sel) this.clearRoutes(sys, n);
    });
  }

  /** Put the selected blocks into a new subsystem; lines that cross the border become ports. */
  group() {
    const picked = this.selectedBlocks();
    if (!picked.length) return this.hooks.toast && this.hooks.toast("Select the blocks to put into a subsystem.");
    let name = null;
    this.change((sys) => {
      const inside = new Set(picked.map((b) => b.name));
      const bb = bounds(picked);
      const sub = { name: uniqueName(sys, "Subsystem"), type: "Subsystem", position: [0, 0, 0, 0], params: {}, blocks: [], lines: [] };
      name = sub.name;
      const shift = [100 - bb.x, 40 - bb.y];
      for (const b of picked) {
        b.position[0] += shift[0];
        b.position[1] += shift[1];
        sub.blocks.push(b);
      }
      const outer = [];
      const ins = new Map(), outs = new Map();
      const innerName = (base) => uniqueName(sub, base);
      for (const l of sys.lines) {
        const a = inside.has(l.from[0]), b = inside.has(l.to[0]);
        if (a && b) {
          if (l.points) l.points = l.points.map((p) => [p[0] + shift[0], p[1] + shift[1]]);
          sub.lines.push(l);
        } else if (!a && b) {
          const k = l.from[0] + "\n" + l.from[1];
          if (!ins.has(k)) {
            const port = ins.size + 1;
            const target = sub.blocks.find((x) => x.name === l.to[0]);
            const y = portPoint(target, "in", l.to[1]).y;
            const blk = { name: innerName(l.name && !sub.blocks.some((x) => x.name === l.name) ? l.name : "In" + port), type: "Inport", position: [20, snap(y - 12, GRID), 40, 24], params: { Port: String(port) } };
            sub.blocks.push(blk);
            ins.set(k, { port, name: blk.name });
            outer.push({ from: l.from.slice(), to: [sub.name, port], name: l.name });
          }
          sub.lines.push({ from: [ins.get(k).name, 1], to: l.to.slice() });
        } else if (a && !b) {
          const k = l.from[0] + "\n" + l.from[1];
          if (!outs.has(k)) {
            const port = outs.size + 1;
            const source = sub.blocks.find((x) => x.name === l.from[0]);
            const y = portPoint(source, "out", l.from[1]).y;
            const blk = { name: innerName(l.name && !sub.blocks.some((x) => x.name === l.name) ? l.name : "Out" + port), type: "Outport", position: [bb.w + 180, snap(y - 12, GRID), 40, 24], params: { Port: String(port) } };
            sub.blocks.push(blk);
            outs.set(k, { port, name: blk.name });
            sub.lines.push({ from: l.from.slice(), to: [blk.name, 1], name: l.name });
          }
          outer.push({ from: [sub.name, outs.get(k).port], to: l.to.slice(), name: l.name });
        } else outer.push(l);
      }
      sys.blocks = sys.blocks.filter((b) => !inside.has(b.name));
      const [w, h] = defaultSize(sub);
      sub.position = [snap(bb.x + bb.w / 2 - w / 2, GRID), snap(bb.y + bb.h / 2 - h / 2, GRID), w, h];
      sys.blocks.push(sub);
      sys.lines = outer;
      this.sel = new Set([sub.name]);
    });
    this.say(`Made subsystem ${name} from ${picked.length} block${picked.length === 1 ? "" : "s"}. Enter opens it.`);
  }

  /* ---------- pointer input ---------- */

  bind() {
    const s = this.svg;
    s.addEventListener("pointerdown", (e) => this.onDown(e));
    s.addEventListener("pointermove", (e) => this.onMove(e));
    s.addEventListener("pointerup", (e) => this.onUp(e));
    s.addEventListener("pointercancel", (e) => this.onUp(e, true));
    s.addEventListener("contextmenu", (e) => {
      e.preventDefault();
      this.openContext(e);
    });
    s.addEventListener(
      "wheel",
      (e) => {
        e.preventDefault();
        const r = s.getBoundingClientRect();
        if (e.ctrlKey || e.metaKey) this.zoomBy(Math.exp(-e.deltaY * 0.01), { x: e.clientX - r.left, y: e.clientY - r.top });
        else {
          this.view.x -= e.shiftKey ? e.deltaY : e.deltaX;
          this.view.y -= e.shiftKey ? 0 : e.deltaY;
          this.applyView();
        }
      },
      { passive: false }
    );
    // Keys go through the window first (capture), so the diagram's keys win over the app-wide shortcuts
    // while the canvas has the focus; keys the diagram does not use fall through to the app.
    window.addEventListener(
      "keydown",
      (e) => {
        if (e.target !== s || e.isComposing) return;
        this.onKey(e);
        if (e.defaultPrevented) e.stopImmediatePropagation();
      },
      true
    );
    this.resizeObs = new ResizeObserver(() => this.applyView());
    this.resizeObs.observe(s);
  }

  hit(e) {
    const t = e.target instanceof Element ? e.target : null;
    if (!t) return { kind: "bg" };
    const port = t.closest(".bd-port");
    if (port) {
      const p = port.dataset.port;
      return { kind: "port", block: port.dataset.block, io: port.dataset.kind, port: /^\d+$/.test(p) ? Number(p) : p };
    }
    const handle = t.closest(".bd-handle");
    if (handle) return { kind: "handle", corner: handle.dataset.corner, block: handle.closest(".bd-block").dataset.name };
    const blk = t.closest(".bd-block");
    if (blk) return { kind: "block", block: blk.dataset.name, onName: t.classList.contains("bd-name") };
    const line = t.closest(".bd-line");
    if (line) return { kind: "line", line: this.sys().lines[Number(line.dataset.line)] };
    return { kind: "bg" };
  }

  onDown(e) {
    this.svg.focus({ preventScroll: true });
    this.pointers.set(e.pointerId, { x: e.clientX, y: e.clientY });
    if (this.pointers.size === 2) {
      // second finger: pinch to zoom
      const [a, b] = [...this.pointers.values()];
      this.cancelDrag();
      this.drag = { type: "pinch", dist: Math.hypot(a.x - b.x, a.y - b.y), k: this.view.k, mid: { x: (a.x + b.x) / 2, y: (a.y + b.y) / 2 }, view: { ...this.view } };
      return;
    }
    if (e.button === 2) return;
    const h = this.hit(e);
    const w = this.toWorld(e);
    // Double click or double tap. (The browser's own dblclick event is no help here: the first click
    // redraws the block, and with the pointer captured the event would name the canvas, not the block.)
    const now = performance.now(), last = this.lastDown;
    const same = last && now - last.t < 420 && Math.hypot(e.clientX - last.x, e.clientY - last.y) < 8 && last.kind === h.kind && last.block === h.block;
    this.lastDown = same ? null : { t: now, x: e.clientX, y: e.clientY, kind: h.kind, block: h.block };
    if (same && e.button === 0 && h.kind !== "port" && h.kind !== "handle" && !(this.wire && this.wire.sticky)) {
      this.cancelDrag();
      this.pointers.delete(e.pointerId);
      this.onDouble(e, h);
      return;
    }
    try {
      this.svg.setPointerCapture(e.pointerId);
    } catch {
      /* synthetic events have no active pointer */
    }
    if (this.wire && this.wire.sticky) {
      this.wire.x = w.x;
      this.wire.y = w.y;
      this.finishWire(e);
      return;
    }
    if (e.button === 1 || this.spaceDown || (h.kind === "bg" && e.pointerType === "touch")) {
      this.drag = { type: "pan", sx: e.clientX, sy: e.clientY, view: { ...this.view }, moved: false, clear: h.kind === "bg" };
      return;
    }
    if (this.readOnly && h.kind !== "block") {
      if (h.kind === "bg") this.drag = { type: "pan", sx: e.clientX, sy: e.clientY, view: { ...this.view }, moved: false, clear: true };
      return;
    }
    if (h.kind === "port") {
      this.wire = { block: h.block, io: h.io, port: h.port, x: w.x, y: w.y, sx: e.clientX, sy: e.clientY, moved: false, sticky: false };
      this.drawWire();
      return;
    }
    if (h.kind === "handle") {
      const b = this.block(h.block);
      this.begin();
      this.drag = { type: "resize", block: b, corner: h.corner, start: w, orig: b.position.slice(), moved: false };
      return;
    }
    if (h.kind === "block") {
      if (e.shiftKey || e.ctrlKey || e.metaKey) {
        if (this.sel.has(h.block)) this.sel.delete(h.block);
        else this.sel.add(h.block);
        this.render();
        this.describeSelection();
        return;
      }
      if (!this.sel.has(h.block)) this.select([h.block]);
      if (this.readOnly) return;
      this.begin();
      this.drag = { type: "move", start: w, orig: new Map(this.selectedBlocks().map((b) => [b.name, b.position.slice()])), moved: false, last: [0, 0] };
      return;
    }
    if (h.kind === "line") {
      if (e.ctrlKey || e.metaKey) {
        // branch: a new line from the same output
        this.wire = { block: h.line.from[0], io: "out", port: h.line.from[1], x: w.x, y: w.y, sx: e.clientX, sy: e.clientY, moved: false, sticky: false };
        this.drawWire();
        return;
      }
      if (!e.shiftKey) {
        this.sel.clear();
        this.selLines.clear();
      }
      this.selLines.add(h.line);
      this.render();
      const route = this.routes.find((r) => r.line === h.line);
      const seg = route ? nearestSegment(route.pts, w) : -1;
      if (route && seg >= 1 && seg <= route.pts.length - 3) {
        this.begin();
        this.drag = { type: "segment", line: h.line, pts: route.pts.map((p) => p.slice()), seg, start: w, moved: false };
      }
      return;
    }
    if (!e.shiftKey) {
      this.sel.clear();
      this.selLines.clear();
      this.render();
    }
    this.drag = { type: "band", start: w, cur: w, moved: false };
  }

  onMove(e) {
    if (this.pointers.has(e.pointerId)) this.pointers.set(e.pointerId, { x: e.clientX, y: e.clientY });
    const w = this.toWorld(e);
    if (this.wire) {
      this.wire.x = w.x;
      this.wire.y = w.y;
      if (Math.hypot(e.clientX - this.wire.sx, e.clientY - this.wire.sy) > 5) this.wire.moved = true;
      this.drawWire(e);
      return;
    }
    const d = this.drag;
    if (!d) return;
    if (d.type === "pinch" && this.pointers.size >= 2) {
      const [a, b] = [...this.pointers.values()];
      const dist = Math.hypot(a.x - b.x, a.y - b.y);
      const r = this.svg.getBoundingClientRect();
      const mid = { x: (a.x + b.x) / 2 - r.left, y: (a.y + b.y) / 2 - r.top };
      const k = Math.max(0.15, Math.min(4, (d.k * dist) / Math.max(d.dist, 1)));
      const m0 = { x: d.mid.x - r.left, y: d.mid.y - r.top };
      this.view.k = k;
      this.view.x = mid.x - ((m0.x - d.view.x) / d.view.k) * k;
      this.view.y = mid.y - ((m0.y - d.view.y) / d.view.k) * k;
      this.applyView();
      return;
    }
    if (d.type === "pan") {
      if (Math.hypot(e.clientX - d.sx, e.clientY - d.sy) > 3) d.moved = true;
      this.view.x = d.view.x + e.clientX - d.sx;
      this.view.y = d.view.y + e.clientY - d.sy;
      this.applyView();
    } else if (d.type === "move") {
      const dx = snap(w.x - d.start.x), dy = snap(w.y - d.start.y);
      if (dx === d.last[0] && dy === d.last[1]) return;
      d.moved = d.moved || dx !== 0 || dy !== 0;
      d.last = [dx, dy];
      for (const b of this.selectedBlocks()) {
        const o = d.orig.get(b.name);
        b.position[0] = o[0] + dx;
        b.position[1] = o[1] + dy;
      }
      this.render();
    } else if (d.type === "resize") {
      const [x, y, bw, bh] = d.orig;
      const dx = snap(w.x - d.start.x), dy = snap(w.y - d.start.y);
      let nx = x, ny = y, nw = bw, nh = bh;
      if (d.corner.includes("e")) nw = Math.max(10, bw + dx);
      if (d.corner.includes("s")) nh = Math.max(20, bh + dy);
      if (d.corner.includes("w")) {
        nw = Math.max(10, bw - dx);
        nx = x + bw - nw;
      }
      if (d.corner.includes("n")) {
        nh = Math.max(20, bh - dy);
        ny = y + bh - nh;
      }
      d.block.position = [nx, ny, nw, nh];
      d.moved = true;
      this.clearRoutes(this.sys(), d.block.name);
      this.render();
    } else if (d.type === "segment") {
      const a = d.pts[d.seg], b = d.pts[d.seg + 1];
      const horizontal = Math.abs(a[1] - b[1]) < 0.01;
      const pts = d.pts.map((p) => p.slice());
      if (horizontal) pts[d.seg][1] = pts[d.seg + 1][1] = snap(a[1] + w.y - d.start.y);
      else pts[d.seg][0] = pts[d.seg + 1][0] = snap(a[0] + w.x - d.start.x);
      d.line.points = pts.slice(1, -1);
      d.moved = true;
      this.render();
    } else if (d.type === "band") {
      d.cur = w;
      d.moved = d.moved || Math.hypot(w.x - d.start.x, w.y - d.start.y) > 4;
      const x = Math.min(d.start.x, w.x), y = Math.min(d.start.y, w.y);
      this.gOver.replaceChildren(svg("rect", { class: "bd-band-rect", x, y, width: Math.abs(w.x - d.start.x), height: Math.abs(w.y - d.start.y) }));
    }
  }

  onUp(e, cancelled = false) {
    this.pointers.delete(e.pointerId);
    if (this.wire && !this.wire.sticky) {
      if (cancelled) this.cancelWire();
      else if (!this.wire.moved) {
        // a click on a port: keep the line attached to the pointer until the next click
        this.wire.sticky = true;
        this.say(`Line from ${this.wire.block}. Click the port to connect to, or press Escape.`);
      } else this.finishWire(e);
      return;
    }
    const d = this.drag;
    this.drag = null;
    if (!d) return;
    if (d.type === "pinch") return;
    if (d.type === "pan") {
      if (!d.moved && d.clear) this.select([]);
      return;
    }
    if (d.type === "move") {
      if (d.moved && !cancelled) {
        this.fixRoutesAfterMove(this.sys(), d.last[0], d.last[1]);
        this.commit();
      } else {
        this.pending = null;
        if (cancelled) for (const b of this.selectedBlocks()) b.position = d.orig.get(b.name);
        this.render();
      }
    } else if (d.type === "resize" || d.type === "segment") {
      if (d.moved) this.commit();
      else this.pending = null;
    } else if (d.type === "band") {
      this.gOver.replaceChildren();
      if (d.moved) {
        const x0 = Math.min(d.start.x, d.cur.x), x1 = Math.max(d.start.x, d.cur.x), y0 = Math.min(d.start.y, d.cur.y), y1 = Math.max(d.start.y, d.cur.y);
        const inside = this.sys().blocks.filter((b) => b.position[0] < x1 && b.position[0] + b.position[2] > x0 && b.position[1] < y1 && b.position[1] + b.position[3] > y0);
        this.select(inside.map((b) => b.name), { add: e.shiftKey });
      }
    }
  }

  cancelDrag() {
    if (this.drag && this.drag.type === "move") for (const b of this.selectedBlocks()) b.position = this.drag.orig.get(b.name) || b.position;
    this.drag = null;
    this.pending = null;
    this.cancelWire();
    this.gOver.replaceChildren();
  }

  drawWire(e) {
    const wr = this.wire;
    const b = this.block(wr.block);
    if (!b) return this.cancelWire();
    const p = portPoint(b, wr.io, wr.port);
    const target = e ? this.portUnder(e) : null;
    let end = { x: wr.x, y: wr.y, dx: -p.dx, dy: -p.dy };
    let ok = false;
    if (target && target.io !== wr.io && this.block(target.block)) {
      end = portPoint(this.block(target.block), target.io, target.port);
      ok = true;
    }
    const pts = wr.io === "out" ? routeLine(p, end, [b.position, [end.x, end.y, 0, 0]]) : routeLine(end, p, [[end.x, end.y, 0, 0], b.position]);
    this.gOver.replaceChildren(svg("path", { class: "bd-wire" + (ok ? " ok" : ""), d: pathData(pts) }));
    this.svg.classList.add("bd-wiring");
    this.svg.dataset.wire = wr.io;
  }

  portUnder(e) {
    const t = document.elementFromPoint(e.clientX, e.clientY);
    const port = t && t.closest && t.closest(".bd-port");
    if (!port || !this.svg.contains(port)) return null;
    const p = port.dataset.port;
    return { block: port.dataset.block, io: port.dataset.kind, port: /^\d+$/.test(p) ? Number(p) : p };
  }

  finishWire(e) {
    const wr = this.wire;
    const el = document.elementFromPoint(e.clientX, e.clientY);
    let target = this.portUnder(e);
    let err = null;
    if (!target && el && el.closest) {
      const blk = el.closest(".bd-block");
      const line = el.closest(".bd-line");
      if (blk && this.svg.contains(blk) && blk.dataset.name !== wr.block) {
        // dropped on a block: take its first free port of the right kind
        const b = this.block(blk.dataset.name);
        if (wr.io === "out") {
          const free = this.freeInputs(b);
          if (free.length) target = { block: b.name, io: "in", port: free[0] };
          else err = portCounts(b).in ? `All inputs of ${b.name} already have a line.` : `${b.name} has no inputs.`;
        } else if (portCounts(b).out) target = { block: b.name, io: "out", port: 1 };
        else err = `${b.name} has no outputs.`;
      } else if (line && wr.io === "in") {
        const l = this.sys().lines[Number(line.dataset.line)];
        if (l) target = { block: l.from[0], io: "out", port: l.from[1] };
      }
    }
    this.cancelWire();
    if (target) {
      if (target.io === wr.io) err = wr.io === "out" ? "A line goes from an output to an input. Both of these are outputs." : "A line goes from an output to an input. Both of these are inputs.";
      else err = wr.io === "out" ? this.connect({ block: wr.block, port: wr.port }, target) : this.connect(target, { block: wr.block, port: wr.port });
    } else if (!err && wr.moved && this.hooks.quickInsert && el === this.svg.querySelector(".bd-bg")) {
      // dropped on empty canvas: offer to add a block there and connect it
      this.hooks.quickInsert(this.toWorld(e), { block: wr.block, io: wr.io, port: wr.port });
    }
    if (err) this.hooks.toast && this.hooks.toast(err, "error");
  }

  cancelWire() {
    this.wire = null;
    this.gOver.replaceChildren();
    this.svg.classList.remove("bd-wiring");
    delete this.svg.dataset.wire;
  }

  onDouble(e, h) {
    if (h.kind === "block") {
      const b = this.block(h.block);
      if (h.onName && !this.readOnly) this.hooks.rename && this.hooks.rename(b);
      else this.activate(b);
    } else if (h.kind === "line" && !this.readOnly) this.hooks.editLabel && this.hooks.editLabel(h.line);
    else if (h.kind === "bg" && !this.readOnly) this.hooks.quickInsert && this.hooks.quickInsert(this.toWorld(e), null);
  }

  /** Enter on a block: look inside a subsystem, open a scope, or edit parameters. */
  activate(b) {
    if (!b) return;
    const t = findType(b.type);
    if (Array.isArray(b.blocks)) return this.enter(b.name);
    if (t && (t.type === "Scope" || t.type === "XYGraph") && this.hooks.openScope) return this.hooks.openScope(this.prefix() + b.name, b);
    if (!this.readOnly && this.hooks.editParams) this.hooks.editParams(b, this.sys());
  }

  openContext(e) {
    const h = this.hit(e);
    if (h.kind === "block" && !this.sel.has(h.block)) this.select([h.block]);
    if (h.kind === "line") {
      this.sel.clear();
      this.selLines = new Set([h.line]);
      this.render();
    }
    this.hooks.contextMenu && this.hooks.contextMenu(e, h, this.toWorld(e));
  }

  /* ---------- keyboard ---------- */

  onKey(e) {
    const mod = e.ctrlKey || e.metaKey;
    const k = e.key;
    const done = () => {
      e.preventDefault();
      e.stopPropagation();
    };
    if (k === " " && !mod) {
      this.spaceDown = true;
      const up = (ev) => {
        if (ev.key === " ") {
          this.spaceDown = false;
          window.removeEventListener("keyup", up);
        }
      };
      window.addEventListener("keyup", up);
      return done();
    }
    if (k === "Escape") {
      if (this.wire || this.drag) this.cancelDrag();
      else if (this.sel.size || this.selLines.size) this.select([]);
      else if (!this.leave()) return;
      return done();
    }
    if (k === "Tab" && !mod && !e.altKey) {
      if (this.cycle(e.shiftKey ? -1 : 1)) done();
      return;
    }
    const arrows = { ArrowLeft: [-1, 0], ArrowRight: [1, 0], ArrowUp: [0, -1], ArrowDown: [0, 1] };
    if (arrows[k]) {
      const [dx, dy] = arrows[k];
      if (e.altKey && k === "ArrowUp" && !this.sel.size) this.leave();
      else if (e.altKey) this.neighbour(dx, dy);
      else if (this.sel.size && !this.readOnly) this.moveSelection(dx * (e.shiftKey ? 1 : GRID), dy * (e.shiftKey ? 1 : GRID));
      else {
        this.view.x -= dx * 40;
        this.view.y -= dy * 40;
        this.applyView();
      }
      return done();
    }
    if (k === "Enter" && !mod && !e.altKey) {
      const b = this.selectedBlocks()[0];
      if (b && this.sel.size === 1) this.activate(b);
      else if (this.selLines.size === 1 && !this.readOnly) this.hooks.editLabel && this.hooks.editLabel([...this.selLines][0]);
      else return;
      return done();
    }
    if (mod && !e.altKey) {
      const key = k.toLowerCase();
      if (key === "a") this.selectAll();
      else if (key === "0") this.fit();
      else if (key === "=" || key === "+") this.zoomBy(1.25);
      else if (key === "-") this.zoomBy(0.8);
      else if (key === "c") this.copy();
      else if (this.readOnly) return;
      else if (key === "z" && !e.shiftKey) this.undo();
      else if (key === "y" || (key === "z" && e.shiftKey)) this.redo();
      else if (key === "x") this.cut();
      else if (key === "v") this.paste();
      else if (key === "d") this.duplicate();
      else if (key === "g") this.group();
      else if (key === "l" && e.shiftKey) this.layout();
      else return;
      return done();
    }
    if (mod || e.altKey) return;
    if (k === "+" || k === "=") this.zoomBy(1.25);
    else if (k === "-") this.zoomBy(0.8);
    else if (k === "0") this.fit();
    else if (k === "1") {
      this.zoomBy(1 / this.view.k);
    } else if (k === "?") this.hooks.showKeys && this.hooks.showKeys();
    else if (this.readOnly) return;
    else if (k === "Delete" || k === "Backspace") this.deleteSelection();
    else if (k === "r" || k === "R") this.rotate(e.shiftKey ? -1 : 1);
    else if (k === "f" || k === "F") this.flip();
    else if (k === "F2") {
      const b = this.selectedBlocks()[0];
      if (b) this.hooks.rename && this.hooks.rename(b);
    } else if (k === "c" || k === "C") {
      const err = this.sel.size >= 2 ? this.connectSelected() : "pick";
      if (err === "pick") this.hooks.connectDialog && this.hooks.connectDialog(this.selectedBlocks()[0] || null);
      else if (err) this.hooks.toast && this.hooks.toast(err);
    } else if (k === "/" || k === "Insert" || k === "a" || k === "A") this.hooks.quickInsert && this.hooks.quickInsert(null, null);
    else if (k === "l" || k === "L") {
      const line = [...this.selLines][0];
      if (line) this.hooks.editLabel && this.hooks.editLabel(line);
    } else return;
    done();
  }
}

/** Index of the segment of a polyline that is closest to a point. */
function nearestSegment(pts, p) {
  let best = -1, dist = Infinity;
  for (let i = 0; i < pts.length - 1; i++) {
    const a = pts[i], b = pts[i + 1];
    const len2 = (b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2 || 1;
    const u = Math.max(0, Math.min(1, ((p.x - a[0]) * (b[0] - a[0]) + (p.y - a[1]) * (b[1] - a[1])) / len2));
    const d = Math.hypot(p.x - (a[0] + u * (b[0] - a[0])), p.y - (a[1] + u * (b[1] - a[1])));
    if (d < dist) [best, dist] = [i, d];
  }
  return best;
}
