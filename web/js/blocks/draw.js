// Drawing blocks and lines as SVG. The shapes are mylab's own: rounded boxes with a coloured
// band for the block family, and small pictograms drawn from a few line segments.

import { findType, portCounts, portLabels, portPoint, paramsOf, blockText, facing } from "./model.js";
import { pathData } from "./route.js";

const NS = "http://www.w3.org/2000/svg";

export function svg(tag, attrs = {}, ...children) {
  const e = document.createElementNS(NS, tag);
  for (const [k, v] of Object.entries(attrs)) if (v !== null && v !== undefined && v !== false) e.setAttribute(k, v === true ? "" : v);
  for (const c of children.flat()) if (c !== null && c !== undefined && c !== false) e.append(c);
  return e;
}

export const CATEGORY_CLASS = {
  Sources: "src", Sinks: "snk", Continuous: "con", Discrete: "dis", Math: "mat", Logic: "log", Nonlinear: "non", Routing: "rou",
  "User-defined": "usr",
};

// Pictograms on a 0..1 square (y down). Each is a list of polylines.
const PICTO = {
  step: [[[0.1, 0.8], [0.5, 0.8], [0.5, 0.2], [0.9, 0.2]]],
  ramp: [[[0.1, 0.8], [0.35, 0.8], [0.9, 0.2]]],
  sine: [wave(0.1, 0.9, 1, 0.3)],
  wave: [wave(0.1, 0.9, 2, 0.28)],
  chirp: [chirp()],
  pulse: [[[0.08, 0.8], [0.25, 0.8], [0.25, 0.2], [0.45, 0.2], [0.45, 0.8], [0.65, 0.8], [0.65, 0.2], [0.85, 0.2], [0.85, 0.8], [0.94, 0.8]]],
  saw: [[[0.1, 0.8], [0.45, 0.2], [0.45, 0.8], [0.8, 0.2], [0.8, 0.8], [0.9, 0.63]]],
  stair: [[[0.1, 0.8], [0.3, 0.8], [0.3, 0.6], [0.5, 0.6], [0.5, 0.4], [0.7, 0.4], [0.7, 0.2], [0.9, 0.2]]],
  rand: [[[0.08, 0.5], [0.18, 0.3], [0.26, 0.7], [0.36, 0.2], [0.46, 0.62], [0.56, 0.42], [0.66, 0.8], [0.76, 0.3], [0.84, 0.55], [0.93, 0.4]]],
  noise: [[[0.08, 0.5], [0.15, 0.25], [0.22, 0.72], [0.3, 0.3], [0.37, 0.8], [0.45, 0.2], [0.52, 0.6], [0.6, 0.35], [0.68, 0.78], [0.76, 0.28], [0.84, 0.66], [0.92, 0.45]]],
  clock: [circle(0.5, 0.5, 0.34), [[0.5, 0.28], [0.5, 0.5], [0.66, 0.6]]],
  sat: [[[0.1, 0.78], [0.32, 0.78], [0.68, 0.22], [0.9, 0.22]]],
  dz: [[[0.1, 0.8], [0.35, 0.5], [0.65, 0.5], [0.9, 0.2]]],
  relay: [[[0.1, 0.75], [0.62, 0.75], [0.62, 0.25], [0.9, 0.25]], [[0.38, 0.75], [0.38, 0.25], [0.62, 0.25]]],
  quant: [[[0.1, 0.8], [0.3, 0.8], [0.3, 0.6], [0.5, 0.6], [0.5, 0.4], [0.7, 0.4], [0.7, 0.2], [0.9, 0.2]], [[0.1, 0.85], [0.9, 0.15]]],
  rate: [[[0.1, 0.8], [0.3, 0.8], [0.7, 0.25], [0.9, 0.25]]],
  lash: [[[0.15, 0.75], [0.55, 0.75], [0.85, 0.25], [0.45, 0.25], [0.15, 0.75]]],
  fric: [[[0.1, 0.9], [0.5, 0.62], [0.5, 0.38], [0.9, 0.1]]],
  lut: [[[0.1, 0.8], [0.3, 0.45], [0.5, 0.55], [0.7, 0.25], [0.9, 0.3]]],
  lut2: [[[0.1, 0.8], [0.3, 0.45], [0.5, 0.55], [0.7, 0.25], [0.9, 0.3]], [[0.1, 0.6], [0.3, 0.3], [0.5, 0.38], [0.7, 0.14], [0.9, 0.16]]],
  delay: [[[0.1, 0.8], [0.3, 0.8], [0.3, 0.25], [0.5, 0.25]], [[0.5, 0.8], [0.7, 0.8], [0.7, 0.25], [0.9, 0.25]]],
  zoh: [[[0.1, 0.8], [0.1, 0.62], [0.36, 0.62], [0.36, 0.38], [0.62, 0.38], [0.62, 0.2], [0.9, 0.2]]],
  rt: [[[0.1, 0.8], [0.1, 0.62], [0.36, 0.62], [0.36, 0.38], [0.62, 0.38], [0.62, 0.2], [0.9, 0.2]]],
  hit: [wave(0.1, 0.9, 1, 0.3), [[0.1, 0.5], [0.9, 0.5]]],
  scope: [wave(0.14, 0.86, 1.5, 0.22)],
  xy: [circle(0.5, 0.5, 0.26)],
  stop: [[[0.3, 0.3], [0.7, 0.3], [0.7, 0.7], [0.3, 0.7], [0.3, 0.3]]],
  term: [[[0.15, 0.5], [0.6, 0.5]], [[0.6, 0.25], [0.6, 0.75]]],
  sw: [[[0.1, 0.25], [0.4, 0.25], [0.75, 0.5], [0.9, 0.5]], [[0.1, 0.75], [0.4, 0.75]]],
  msw: [[[0.1, 0.25], [0.4, 0.25], [0.75, 0.5], [0.9, 0.5]], [[0.1, 0.5], [0.4, 0.5]], [[0.1, 0.75], [0.4, 0.75]]],
  man: [[[0.1, 0.25], [0.4, 0.25], [0.75, 0.5], [0.9, 0.5]], [[0.1, 0.75], [0.4, 0.75]]],
  mrg: [[[0.1, 0.25], [0.5, 0.5], [0.9, 0.5]], [[0.1, 0.75], [0.5, 0.5]]],
  en: [[[0.15, 0.7], [0.4, 0.7], [0.4, 0.3], [0.85, 0.3]]],
  trg: [[[0.15, 0.7], [0.5, 0.7], [0.5, 0.3], [0.85, 0.3]], [[0.4, 0.45], [0.5, 0.33], [0.6, 0.45]]],
  ws: [[[0.2, 0.25], [0.8, 0.25], [0.8, 0.75], [0.2, 0.75], [0.2, 0.25]], [[0.2, 0.42], [0.8, 0.42]], [[0.4, 0.25], [0.4, 0.75]]],
  disp: [[[0.15, 0.3], [0.85, 0.3], [0.85, 0.7], [0.15, 0.7], [0.15, 0.3]]],
};

function wave(x0, x1, periods, amp) {
  const pts = [];
  for (let i = 0; i <= 32; i++) {
    const u = i / 32;
    pts.push([x0 + (x1 - x0) * u, 0.5 - amp * Math.sin(2 * Math.PI * periods * u)]);
  }
  return pts;
}

function chirp() {
  const pts = [];
  for (let i = 0; i <= 60; i++) {
    const u = i / 60;
    pts.push([0.1 + 0.8 * u, 0.5 - 0.28 * Math.sin(2 * Math.PI * (0.8 * u + 2.2 * u * u))]);
  }
  return pts;
}

function circle(cx, cy, r) {
  const pts = [];
  for (let i = 0; i <= 28; i++) pts.push([cx + r * Math.cos((2 * Math.PI * i) / 28), cy + r * Math.sin((2 * Math.PI * i) / 28)]);
  return pts;
}

function pictogram(name, w, h) {
  const lines = PICTO[name];
  if (!lines) return null;
  const s = Math.min(w, h) - 8, ox = (w - s) / 2, oy = (h - s) / 2;
  const d = lines.map((pts) => "M" + pts.map((p) => `${(ox + p[0] * s).toFixed(1)} ${(oy + p[1] * s).toFixed(1)}`).join(" L")).join(" ");
  return svg("path", { class: "bd-picto", d });
}

function fitText(text, w, max = 13) {
  const size = Math.max(8, Math.min(max, (w - 8) / Math.max(text.length * 0.58, 1)));
  return size.toFixed(1);
}

/** The <g> for one block. opts: { selected, marker: {level, message}, openPorts: Set("in:1"), display } */
export function drawBlock(block, opts = {}) {
  const t = findType(block.type);
  const [x, y, w, h] = block.position;
  const counts = portCounts(block);
  const cat = t ? CATEGORY_CLASS[t.category] : "unk";
  const type = t ? t.type : "";
  const g = svg("g", {
    class: `bd-block cat-${cat}${opts.selected ? " sel" : ""}${t ? "" : " unknown"}${opts.marker ? " has-" + opts.marker.level : ""}`,
    "data-name": block.name, "data-type": block.type, transform: `translate(${x} ${y})`,
  });
  const label = `${block.name}, ${t ? t.title : block.type + " (not supported)"}${opts.marker ? ". " + opts.marker.message : ""}`;
  g.append(svg("title", {}, label));
  // body
  const bar = ["Mux", "Demux", "BusCreator", "BusSelector"].includes(type);
  const pill = ["Inport", "Outport", "Goto", "From", "Sum"].includes(type);
  const rx = pill ? Math.min(w, h) / 2 : bar ? 2 : 7;
  g.append(svg("rect", { class: "bd-body" + (bar ? " bar" : ""), x: 0, y: 0, width: w, height: h, rx }));
  if (!bar && !pill && t) {
    // family band along the input side
    const f = facing(block);
    const band = f === 0 ? { x: 0, y: 0, width: 5, height: h } : f === 2 ? { x: w - 5, y: 0, width: 5, height: h } : f === 1 ? { x: 0, y: 0, width: w, height: 5 } : { x: 0, y: h - 5, width: w, height: 5 };
    g.append(svg("rect", { class: "bd-band", ...band, rx: 2.5 }));
  }
  if (type === "Subsystem") g.append(svg("rect", { class: "bd-inner", x: 4, y: 4, width: Math.max(w - 8, 2), height: Math.max(h - 8, 2), rx: 4 }));
  if (type === "Scope" || type === "XYGraph" || type === "Display") {
    g.append(svg("rect", { class: "bd-screen", x: 8, y: 7, width: Math.max(w - 14, 4), height: Math.max(h - 14, 4), rx: 3 }));
  }
  // content
  if (!bar) {
    const p = paramsOf(block);
    const text = opts.display !== undefined ? opts.display : blockText(block);
    // port names take room on their side: move the text out of their way
    const named = portLabels(block);
    const lead = named.ins.some((s) => s && s.length > 1) ? 16 : 0, trail = named.outs.some((s) => s && s.length > 1) ? 16 : 0;
    const f0 = facing(block);
    const cx = w / 2 + (f0 === 0 ? (lead - trail) / 2 : f0 === 2 ? (trail - lead) / 2 : 0), cy = h / 2;
    if (["TransferFcn", "DiscreteTransferFcn", "DiscreteFilter"].includes(type) && opts.display === undefined) {
      const num = String(p.Numerator), den = String(p.Denominator);
      const size = fitText(num.length > den.length ? num : den, w, 11);
      g.append(
        svg("text", { class: "bd-text", x: cx, y: cy - 6, "font-size": size }, num),
        svg("line", { class: "bd-frac", x1: 9, y1: cy, x2: w - 6, y2: cy }),
        svg("text", { class: "bd-text", x: cx, y: cy + 9, "font-size": size }, den)
      );
    } else if (type === "Subsystem" || type === "MATLABFunction" || !t) {
      const cap = !t ? "?" : type === "MATLABFunction" ? "fcn" : "";
      if (cap) g.append(svg("text", { class: "bd-text big", x: cx, y: cy, "font-size": 15 }, cap));
    } else if (text) {
      g.append(svg("text", { class: "bd-text" + (type === "Display" ? " mono" : ""), x: cx, y: cy, "font-size": fitText(text, w, type === "Integrator" ? 20 : 13) }, text));
    } else {
      const pic = t && pictogram(t.icon, w, h);
      if (pic) g.append(pic);
      else if (t) g.append(svg("text", { class: "bd-text", x: cx, y: cy, "font-size": fitText(t.icon || t.title, w, 12) }, t.icon || t.title));
    }
  }
  // ports
  const labels = portLabels(block);
  const local = (pt) => ({ x: pt.x - x, y: pt.y - y });
  const addPort = (kind, port, text) => {
    const pt = portPoint(block, kind, port, counts);
    const q = local(pt);
    const open = opts.openPorts && opts.openPorts.has(kind + ":" + port);
    const pg = svg("g", {
      class: `bd-port ${kind}${open ? " open" : ""}${typeof port === "string" ? " ctl" : ""}`, "data-block": block.name, "data-kind": kind, "data-port": port,
      transform: `translate(${q.x} ${q.y})`,
    });
    const what = typeof port === "string" ? `${port} input` : `${kind === "in" ? "input" : "output"} ${port}`;
    pg.append(svg("title", {}, `${block.name}: ${what}${text ? " (" + text + ")" : ""}${open ? ", not connected" : ""}`));
    pg.append(svg("circle", { class: "bd-port-hit", r: 10 }));
    if (kind === "out") pg.append(svg("path", { class: "bd-port-dot", d: arrow(pt.dx, pt.dy, 5) }));
    else if (typeof port === "string") pg.append(svg("rect", { class: "bd-port-dot", x: -4, y: -4, width: 8, height: 8, rx: 1.5 }));
    else pg.append(svg("circle", { class: "bd-port-dot", r: 3.6 }));
    g.append(pg);
    if (text && !bar) {
      const tx = q.x - pt.dx * 9, ty = q.y - pt.dy * 9;
      g.append(svg("text", { class: "bd-portlabel", x: tx, y: ty, "text-anchor": pt.dx < 0 ? "start" : pt.dx > 0 ? "end" : "middle" }, text.length > 9 ? text.slice(0, 8) + "…" : text));
    }
  };
  for (let k = 1; k <= counts.in; k++) addPort("in", k, labels.ins[k - 1]);
  for (let k = 1; k <= counts.out; k++) addPort("out", k, labels.outs[k - 1]);
  if (counts.enable) addPort("in", "enable", "");
  if (counts.trigger) addPort("in", "trigger", "");
  // name under the block
  if (!(block.other && block.other.ShowName === "off")) {
    g.append(svg("text", { class: "bd-name", x: w / 2, y: h + 14 }, block.name));
  }
  if (opts.marker) {
    const m = svg("g", { class: `bd-marker ${opts.marker.level}`, transform: `translate(${w - 2} 2)` });
    m.append(svg("title", {}, opts.marker.message), svg("circle", { r: 8 }), svg("text", { x: 0, y: 0.5 }, "!"));
    g.append(m);
  }
  if (opts.selected && opts.handles) {
    for (const [cx, cy, name] of [[0, 0, "nw"], [w, 0, "ne"], [0, h, "sw"], [w, h, "se"]]) {
      g.append(svg("rect", { class: "bd-handle", "data-corner": name, x: cx - 4.5, y: cy - 4.5, width: 9, height: 9, rx: 2 }));
    }
  }
  return g;
}

function arrow(dx, dy, s) {
  // a small triangle pointing out of the block
  if (dx > 0) return `M0 ${-s} L${s + 2} 0 L0 ${s} Z`;
  if (dx < 0) return `M0 ${-s} L${-s - 2} 0 L0 ${s} Z`;
  if (dy > 0) return `M${-s} 0 L0 ${s + 2} L${s} 0 Z`;
  return `M${-s} 0 L0 ${-s - 2} L${s} 0 Z`;
}

/** The <g> for one line. route: { line, pts }. */
export function drawLine(route, index, opts = {}) {
  const { line, pts } = route;
  const d = pathData(pts);
  const g = svg("g", { class: "bd-line" + (opts.selected ? " sel" : "") + (opts.error ? " err" : ""), "data-line": index });
  g.append(svg("title", {}, `Line from ${line.from[0]} output ${line.from[1]} to ${line.to[0]} input ${line.to[1]}${line.name ? ", signal " + line.name : ""}`));
  g.append(svg("path", { class: "bd-line-hit", d }), svg("path", { class: "bd-line-path", d }));
  // arrow head at the end
  const a = pts[pts.length - 2], b = pts[pts.length - 1];
  const len = Math.hypot(b[0] - a[0], b[1] - a[1]) || 1;
  const ux = (b[0] - a[0]) / len, uy = (b[1] - a[1]) / len;
  const tip = [b[0] - ux * 4, b[1] - uy * 4];
  const base = [tip[0] - ux * 8, tip[1] - uy * 8];
  g.append(svg("path", { class: "bd-line-head", d: `M${tip[0]} ${tip[1]} L${base[0] - uy * 4} ${base[1] + ux * 4} L${base[0] + uy * 4} ${base[1] - ux * 4} Z` }));
  if (line.name) {
    // label on the longest of the first two segments
    let best = 0;
    for (let i = 0; i < pts.length - 1; i++) {
      if (Math.hypot(pts[i + 1][0] - pts[i][0], pts[i + 1][1] - pts[i][1]) > Math.hypot(pts[best + 1][0] - pts[best][0], pts[best + 1][1] - pts[best][1]) + 12) best = i;
    }
    const p = pts[best], q = pts[best + 1];
    const horizontal = Math.abs(q[1] - p[1]) < 0.5;
    g.append(svg("text", { class: "bd-line-label", x: (p[0] + q[0]) / 2 + (horizontal ? 0 : 6), y: (p[1] + q[1]) / 2 - (horizontal ? 6 : 0), "text-anchor": horizontal ? "middle" : "start" }, line.name + (line.log ? " ●" : "")));
  }
  return g;
}

/** A small picture of a block type for the palette. */
export function paletteIcon(typeName) {
  const t = findType(typeName);
  const block = { name: "", type: typeName, position: [0, 0, 44, 30], params: {} };
  if (t && t.type === "Subsystem") block.blocks = [{ name: "In1", type: "Inport", position: [0, 0, 1, 1], params: {} }, { name: "Out1", type: "Outport", position: [0, 0, 1, 1], params: {} }];
  const bar = t && ["Mux", "Demux", "BusCreator", "BusSelector"].includes(t.type);
  if (bar) block.position = [17, 0, 10, 30];
  const g = drawBlock(block, {});
  g.querySelectorAll(".bd-name, .bd-port-hit, title, .bd-portlabel").forEach((n) => n.remove());
  const s = svg("svg", { class: "bd-pal-icon", viewBox: "-8 -4 60 38", width: 52, height: 33, "aria-hidden": "true" });
  s.append(g);
  return s;
}
