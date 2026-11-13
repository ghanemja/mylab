// Block-diagram model helpers: the JSON structure, port counts, sizes and port positions.
// The model is kept exactly as it is stored in a .mlb.json file (see docs/BLOCKS.md), so
// saving is JSON.stringify and undo is a snapshot.

import { CATALOG } from "./catalog.js";

export const FORMAT = "mylab-blocks-1";
export const GRID = 10;
export const EXT = ".mlb.json";

export const TYPES = new Map(CATALOG.blocks.map((b) => [b.type, b]));
const key = (s) => String(s).toLowerCase().replace(/[^a-z0-9]/g, "");
const BY_KEY = new Map();
for (const b of CATALOG.blocks) {
  BY_KEY.set(key(b.type), b);
  if (!BY_KEY.has(key(b.title))) BY_KEY.set(key(b.title), b);
}

/** Look a block type up by type name or title, ignoring case, spaces and punctuation. */
export function findType(name) {
  return TYPES.get(name) || BY_KEY.get(key(name)) || null;
}

export const isModelPath = (path) => typeof path === "string" && path.toLowerCase().endsWith(EXT);

export const DEFAULT_CONFIG = {
  StartTime: "0", StopTime: "10", Solver: "ode45", MaxStep: "auto", RelTol: "1e-3", AbsTol: "1e-6", FixedStep: "auto", ZeroCrossings: "on",
};

export function newModel(name = "untitled") {
  return { format: FORMAT, name, config: { ...DEFAULT_CONFIG }, blocks: [], lines: [] };
}

/** Parse model text. Throws an Error with a readable message. */
export function parseModel(text) {
  let m;
  try {
    m = JSON.parse(text);
  } catch (e) {
    throw new Error("This file is not valid JSON: " + e.message);
  }
  if (!m || typeof m !== "object" || !Array.isArray(m.blocks)) throw new Error('This is not a mylab Blocks model (no "blocks" list).');
  if (m.format && m.format !== FORMAT) throw new Error(`Unknown model format "${m.format}".`);
  m.format = FORMAT;
  m.config = { ...DEFAULT_CONFIG, ...(m.config || {}) };
  normalizeSystem(m);
  return m;
}

function normalizeSystem(sys) {
  sys.blocks = Array.isArray(sys.blocks) ? sys.blocks : [];
  sys.lines = Array.isArray(sys.lines) ? sys.lines : [];
  for (const b of sys.blocks) {
    b.params = b.params || {};
    if (b.type === "Subsystem" || b.blocks || b.lines) normalizeSystem(b);
  }
  // Blocks without a size (models built from code) get the default size for their type.
  sys.blocks.forEach((b) => {
    const p = Array.isArray(b.position) ? b.position.map(Number) : [0, 0, 0, 0];
    const [w, h] = defaultSize(b);
    b.position = [p[0] || 0, p[1] || 0, p[2] > 0 ? p[2] : w, p[3] > 0 ? p[3] : h];
  });
  sys.lines = sys.lines.filter((l) => Array.isArray(l.from) && Array.isArray(l.to));
}

/** True when every block sits on the default grid that add_block uses: the model was never laid out. */
export function needsLayout(sys) {
  if (sys.blocks.length < 2) return false;
  const seen = new Set();
  for (const b of sys.blocks) {
    const [x, y] = b.position;
    if ((x - 60) % 110 || (y - 60) % 90 || x > 60 + 110 * 5) return false;
    seen.add(x + "," + y);
  }
  return seen.size === sys.blocks.length;
}

/** Model text with short lists on one line, the same layout the Python side writes. */
export function serializeModel(model) {
  const plain = (x) => !x || typeof x !== "object";
  const flat = (v) => plain(v) || (Array.isArray(v) && v.every(plain));
  const inline = (v) => "[" + v.map((x) => (Array.isArray(x) ? inline(x) : JSON.stringify(x))).join(", ") + "]";
  const dump = (v, pad) => {
    if (Array.isArray(v)) {
      if (v.every(flat)) return inline(v);
      return "[\n" + v.map((x) => pad + "  " + dump(x, pad + "  ")).join(",\n") + "\n" + pad + "]";
    }
    if (v && typeof v === "object") {
      const keys = Object.keys(v);
      if (!keys.length) return "{}";
      return "{\n" + keys.map((k) => `${pad}  ${JSON.stringify(k)}: ${dump(v[k], pad + "  ")}`).join(",\n") + "\n" + pad + "}";
    }
    return JSON.stringify(v);
  };
  return dump(clean(model), "") + "\n";
}

function clean(sys) {
  const out = {};
  for (const [k, v] of Object.entries(sys)) {
    if (k === "blocks") out.blocks = v.map(cleanBlock);
    else if (k === "lines") out.lines = v.map(cleanLine);
    else out[k] = v;
  }
  return out;
}

function cleanBlock(b) {
  const out = { name: b.name, type: b.type, position: b.position.map((v) => Math.round(v * 100) / 100) };
  if (b.rotation) out.rotation = b.rotation;
  if (b.flip) out.flip = true;
  if (b.params && Object.keys(b.params).length) out.params = b.params;
  for (const [k, v] of Object.entries(b)) {
    if (["name", "type", "position", "rotation", "flip", "params", "blocks", "lines"].includes(k)) continue;
    out[k] = v;
  }
  if (b.blocks || b.type === "Subsystem") {
    out.blocks = (b.blocks || []).map(cleanBlock);
    out.lines = (b.lines || []).map(cleanLine);
  }
  return out;
}

function cleanLine(l) {
  const out = { from: l.from, to: l.to };
  if (l.name) out.name = l.name;
  if (l.log) out.log = true;
  if (l.points && l.points.length) out.points = l.points.map((p) => [Math.round(p[0] * 100) / 100, Math.round(p[1] * 100) / 100]);
  return out;
}

/* ---------- ports ---------- */

/** Input and output names of the first "function" line of a script. */
export function signature(code) {
  const m = /^\s*function\s+(?:\[([^\]]*)\]\s*=|(\w+)\s*=)?\s*\w+\s*(?:\(([^)]*)\))?/m.exec(code || "");
  if (!m) return { ins: ["u"], outs: ["y"] };
  const list = (s) => (s || "").trim().split(/[,\s]+/).filter(Boolean);
  return { ins: list(m[3]), outs: list(m[1] || m[2]) };
}

/** Number of ports from a rule of the catalog (the same rules as mylab/blocks/library.py). */
export function countPorts(rule, params) {
  if (typeof rule === "number") return rule;
  if (typeof rule === "string") return 0;
  const kind = rule[0];
  if (kind === "sum") return rule.slice(1).reduce((n, r) => n + countPorts(r, params), 0);
  if (kind === "if") {
    const want = (Array.isArray(rule[2]) ? rule[2] : [rule[2]]).map(key);
    return countPorts(want.includes(key(params[rule[1]] ?? "")) ? rule[3] : rule[4], params);
  }
  const val = String(params[rule[1]] ?? "").trim();
  if (kind === "int") {
    const n = Math.floor(Number(val));
    return Number.isFinite(n) && val !== "" ? Math.max(n, 0) : 1;
  }
  if (kind === "signs") {
    if (/^\d+$/.test(val)) return Number(val);
    if (val.startsWith("[")) return Math.max(val.replace(/[[\]]/g, "").replace(/,/g, " ").trim().split(/\s+/).filter(Boolean).length, 1);
    return Math.max(val.replace(/[|\s]/g, "").length, 1);
  }
  if (kind === "csv") return Math.max(val.split(",").filter((s) => s.trim()).length, 1);
  if (kind === "sig") {
    const s = signature(val);
    return rule[2] === "in" ? s.ins.length : s.outs.length;
  }
  return 1;
}

export function paramsOf(block) {
  const t = findType(block.type);
  const p = {};
  if (t) for (const q of t.params) p[q.name] = q.default;
  return Object.assign(p, block.params || {});
}

const innerOfType = (block, type) => (block.blocks || []).filter((b) => (findType(b.type) || {}).type === type);

/** { in, out, enable, trigger } for a block as drawn. */
export function portCounts(block) {
  const t = findType(block.type);
  if (!t) {
    const p = Array.isArray(block.ports) ? block.ports : [1, 1];
    return { in: p[0] || 0, out: p[1] || 0, enable: false, trigger: false };
  }
  if (t.type === "Subsystem") {
    return {
      in: innerOfType(block, "Inport").length, out: innerOfType(block, "Outport").length,
      enable: innerOfType(block, "EnablePort").length > 0, trigger: innerOfType(block, "TriggerPort").length > 0,
    };
  }
  const p = paramsOf(block);
  return { in: countPorts(t.in, p), out: countPorts(t.out, p), enable: false, trigger: false };
}

/** Names shown beside the ports of a block (subsystem port names, function arguments, signs). */
export function portLabels(block) {
  const t = findType(block.type);
  const ins = [], outs = [];
  if (!t) return { ins, outs };
  const p = paramsOf(block);
  if (t.type === "Subsystem") {
    const byPort = (list) => list.slice().sort((a, b) => Number((a.params || {}).Port || 0) - Number((b.params || {}).Port || 0)).map((b) => b.name);
    return { ins: byPort(innerOfType(block, "Inport")), outs: byPort(innerOfType(block, "Outport")) };
  }
  if (t.type === "MATLABFunction") {
    const s = signature(p.Script);
    return { ins: s.ins, outs: s.outs };
  }
  if (t.type === "Sum" || t.type === "Product" || t.type === "Divide") {
    const s = String(p.Inputs || "").replace(/[|\s]/g, "");
    if (!/^\d+$/.test(s)) for (const c of s) ins.push(c === "*" ? "×" : c === "/" ? "÷" : c === "-" ? "−" : "+");
  } else if (t.type === "Integrator") {
    if (portCounts(block).in > 1) {
      ins.push("");
      if (key(p.ExternalReset) !== "none") ins.push("reset");
      if (key(p.InitialConditionSource) === "external") ins.push("x0");
    }
  } else if (t.type === "Switch") ins.push("", "?", "");
  else if (t.type === "BusSelector") for (const s of String(p.OutputSignals || "").split(",")) outs.push(s.trim());
  return { ins, outs };
}

/* ---------- geometry ---------- */

export function defaultSize(block) {
  const t = findType(block.type);
  const c = portCounts(block);
  const n = Math.max(c.in, c.out, 1);
  const tall = (min, per = 20, pad = 20) => Math.max(min, n * per + pad);
  const type = t ? t.type : "";
  if (["Mux", "Demux", "BusCreator", "BusSelector"].includes(type)) return [10, tall(40, 20, 0)];
  if (["Inport", "Outport"].includes(type)) return [40, 24];
  if (["Goto", "From"].includes(type)) return [60, 28];
  if (["EnablePort", "TriggerPort"].includes(type)) return [40, 40];
  if (type === "Subsystem" || !t) return [110, tall(60, 24, 16)];
  if (["TransferFcn", "ZeroPole", "DiscreteTransferFcn", "DiscreteFilter", "DiscreteFir"].includes(type)) return [90, 50];
  if (["StateSpace", "DiscreteStateSpace", "PIDController", "DiscreteTimeIntegrator", "ToWorkspace", "FromWorkspace"].includes(type)) return [80, 48];
  if (["Fcn", "InterpretedFunction"].includes(type)) return [130, 40];
  if (type === "MATLABFunction") return [110, tall(50, 22, 14)];
  if (["Scope", "XYGraph"].includes(type)) return [50, tall(50, 20, 10)];
  if (["Constant", "Ground", "Terminator", "Clock"].includes(type)) return [50, 36];
  if (["Sum", "Product", "Divide"].includes(type)) return [40, tall(40, 20, 0)];
  if (type === "Integrator") return [c.in > 1 ? 80 : 50, tall(44, 22, 4)];
  return [60, tall(40, 20, 0)];
}

/** Which side the inputs are on: 0 left, 1 top, 2 right, 3 bottom (outputs are opposite). */
export function facing(block) {
  return (((block.rotation || 0) + (block.flip ? 2 : 0)) % 4 + 4) % 4;
}

const DIRS = [[-1, 0], [0, -1], [1, 0], [0, 1]]; // outward normal of side 0..3

/**
 * Position and outward direction of a port. kind: "in" | "out"; port: 1-based number, or
 * "enable" / "trigger" (drawn on the top edge, or the bottom edge of a block turned upside down).
 */
export function portPoint(block, kind, port, counts = portCounts(block)) {
  const [x, y, w, h] = block.position;
  const f = facing(block);
  if (typeof port === "string") {
    const both = counts.enable && counts.trigger;
    const frac = both ? (port === "enable" ? 1 / 3 : 2 / 3) : 0.5;
    const side = f === 1 ? 0 : f === 3 ? 2 : 1; // the control ports avoid the input side
    return sidePoint(x, y, w, h, side, frac);
  }
  const n = Math.max(kind === "in" ? counts.in : counts.out, 1);
  const frac = (Math.min(port, n) - 0.5) / n;
  const side = kind === "in" ? f : (f + 2) % 4;
  return sidePoint(x, y, w, h, side, frac);
}

function sidePoint(x, y, w, h, side, frac) {
  const d = DIRS[side];
  if (side === 0) return { x, y: y + h * frac, dx: d[0], dy: d[1] };
  if (side === 2) return { x: x + w, y: y + h * frac, dx: d[0], dy: d[1] };
  if (side === 1) return { x: x + w * frac, y, dx: d[0], dy: d[1] };
  return { x: x + w * frac, y: y + h, dx: d[0], dy: d[1] };
}

/* ---------- systems ---------- */

/** The system (object with blocks and lines) at a path of subsystem names; null when a name is missing. */
export function systemAt(model, path) {
  let sys = model;
  for (const name of path) {
    const b = sys.blocks.find((x) => x.name === name);
    if (!b || !Array.isArray(b.blocks)) return null;
    sys = b;
  }
  return sys;
}

export function uniqueName(sys, base) {
  const names = new Set(sys.blocks.map((b) => b.name));
  if (!names.has(base)) return base;
  const stem = base.replace(/\d+$/, "") || base;
  for (let n = 1; ; n++) if (!names.has(stem + n)) return stem + n;
}

/** A new block of a catalog type, ready to drop at (x, y) (its centre). */
export function makeBlock(sys, typeName, x, y) {
  const t = findType(typeName);
  if (!t) return null;
  const b = { name: uniqueName(sys, t.title), type: t.type, position: [0, 0, 0, 0], params: {} };
  if (t.type === "Subsystem") {
    b.blocks = [
      { name: "In1", type: "Inport", position: [40, 60, 40, 24], params: { Port: "1" } },
      { name: "Out1", type: "Outport", position: [220, 60, 40, 24], params: { Port: "1" } },
    ];
    b.lines = [{ from: ["In1", 1], to: ["Out1", 1] }];
  }
  if (t.type === "Inport" || t.type === "Outport") {
    const same = sys.blocks.filter((o) => (findType(o.type) || {}).type === t.type);
    b.params.Port = String(same.length + 1);
    b.name = uniqueName(sys, (t.type === "Inport" ? "In" : "Out") + (same.length + 1));
  }
  const [w, h] = defaultSize(b);
  const snap = (v) => Math.round(v / GRID) * GRID;
  b.position = [snap(x - w / 2), snap(y - h / 2), w, h];
  return b;
}

/** Keep the Port numbers of In and Out blocks 1..n without gaps (after deleting or pasting). */
export function renumberPorts(sys) {
  for (const type of ["Inport", "Outport"]) {
    const list = sys.blocks.filter((b) => (findType(b.type) || {}).type === type);
    list.sort((a, b) => Number(a.params.Port || 1e9) - Number(b.params.Port || 1e9));
    list.forEach((b, i) => (b.params.Port = String(i + 1)));
  }
}

/** Drop lines whose blocks or ports no longer exist. Returns the number of lines removed. */
export function pruneLines(sys) {
  const byName = new Map(sys.blocks.map((b) => [b.name, b]));
  const counts = new Map();
  const c = (b) => counts.get(b.name) || counts.set(b.name, portCounts(b)).get(b.name);
  const seen = new Set();
  const before = sys.lines.length;
  sys.lines = sys.lines.filter((l) => {
    const s = byName.get(l.from[0]), d = byName.get(l.to[0]);
    if (!s || !d) return false;
    if (typeof l.from[1] !== "number" || l.from[1] < 1 || l.from[1] > c(s).out) return false;
    const dc = c(d);
    if (typeof l.to[1] === "number" ? l.to[1] < 1 || l.to[1] > dc.in : !dc[l.to[1]]) return false;
    const k = l.to[0] + "\n" + l.to[1];
    if (seen.has(k)) return false;
    seen.add(k);
    return true;
  });
  return before - sys.lines.length;
}

/** Inputs without a line: [{ block, port }] (found without the engine, for the markers on the canvas). */
export function unconnectedInputs(sys) {
  const fed = new Set(sys.lines.map((l) => l.to[0] + "\n" + l.to[1]));
  const out = [];
  for (const b of sys.blocks) {
    const c = portCounts(b);
    for (let k = 1; k <= c.in; k++) if (!fed.has(b.name + "\n" + k)) out.push({ block: b.name, port: k });
    for (const ctl of ["enable", "trigger"]) if (c[ctl] && !fed.has(b.name + "\n" + ctl)) out.push({ block: b.name, port: ctl });
  }
  return out;
}

export function bounds(blocks) {
  if (!blocks.length) return { x: 0, y: 0, w: 0, h: 0 };
  let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
  for (const b of blocks) {
    const [x, y, w, h] = b.position;
    x0 = Math.min(x0, x);
    y0 = Math.min(y0, y);
    x1 = Math.max(x1, x + w);
    y1 = Math.max(y1, y + h);
  }
  return { x: x0, y: y0, w: x1 - x0, h: y1 - y0 };
}

/** Short text drawn inside a block: its main parameter. */
export function blockText(block) {
  const t = findType(block.type);
  if (!t) return "?";
  const p = paramsOf(block);
  const cut = (s, n = 14) => (String(s).length > n ? String(s).slice(0, n - 1) + "…" : String(s));
  switch (t.type) {
    case "Gain": return cut(p.Gain, 9);
    case "Constant": return cut(p.Value, 8);
    case "Bias": return "u+" + cut(p.Bias, 6);
    case "Fcn": return cut(p.Expr, 22);
    case "InterpretedFunction": return cut(p.MATLABFcn, 22);
    case "Goto": case "From": return "[" + cut(p.GotoTag, 8) + "]";
    case "Inport": case "Outport": return cut(p.Port, 3);
    case "ToWorkspace": case "FromWorkspace": return cut(p.VariableName, 11);
    case "Saturation": return "";
    case "MathFunction": case "TrigonometricFunction": case "Rounding": case "Sqrt": return cut(p.Operator, 8);
    case "MinMax": return p.Function;
    case "RelationalOperator": return p.Operator;
    case "LogicalOperator": case "BitwiseOperator": return p.Operator || p.logicop;
    case "CompareToConstant": return `${p.relop} ${cut(p.const, 5)}`;
    case "CompareToZero": return `${p.relop} 0`;
    case "PIDController": return p.Controller || "PID";
    case "UnitDelay": return "1/z";
    case "Delay": return "z⁻" + cut(p.DelayLength, 2);
    case "TransportDelay": return "τ=" + cut(p.DelayTime, 6);
    case "ZeroOrderHold": case "RateTransition": return "";
    case "MATLABFunction": return "";
    case "Subsystem": return "";
    case "StateSpace": return "ẋ=Ax+Bu";
    case "DiscreteStateSpace": return "x⁺=Ax+Bu";
    default: return t.icon && t.icon.length <= 9 && !/^[a-z0-9]+$/.test(t.icon) ? t.icon : "";
  }
}
