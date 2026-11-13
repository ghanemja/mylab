// Orthogonal routing of signal lines, and the automatic layout of a system.

import { portPoint, portCounts, defaultSize, GRID } from "./model.js";

const STUB = 14;

/** Remove repeated points and points in the middle of a straight run. */
export function simplify(pts) {
  const out = [];
  for (const p of pts) {
    const a = out[out.length - 1];
    if (a && Math.abs(a[0] - p[0]) < 0.01 && Math.abs(a[1] - p[1]) < 0.01) continue;
    const b = out[out.length - 2];
    if (b && ((Math.abs(b[0] - a[0]) < 0.01 && Math.abs(a[0] - p[0]) < 0.01) || (Math.abs(b[1] - a[1]) < 0.01 && Math.abs(a[1] - p[1]) < 0.01))) {
      // collinear: keep going unless the path doubles back on itself
      const fwd = (a[0] - b[0]) * (p[0] - a[0]) + (a[1] - b[1]) * (p[1] - a[1]);
      if (fwd >= 0) {
        out[out.length - 1] = p;
        continue;
      }
    }
    out.push(p);
  }
  return out;
}

/**
 * Path of a line as a list of [x, y] corner points.
 * src, dst: port points { x, y, dx, dy } (dx, dy = outward direction of the port).
 * boxes: the two block rectangles [x, y, w, h], used to go around the blocks for feedback lines.
 * via: corner points the user (or an imported file) fixed; the path passes through them.
 */
export function routeLine(src, dst, boxes, via) {
  const p0 = [src.x, src.y], p1 = [dst.x, dst.y];
  if (via && via.length) return simplify(throughPoints(p0, src, via, p1, dst));
  const srcH = src.dx !== 0, dstH = dst.dx !== 0;
  // blocks that sit close together, facing each other, get shorter stubs instead of a detour
  let stub = STUB;
  if (srcH && dstH && src.dx === -dst.dx && (dst.x - src.x) * src.dx > 0) stub = Math.min(STUB, ((dst.x - src.x) * src.dx) / 2);
  if (!srcH && !dstH && src.dy === -dst.dy && (dst.y - src.y) * src.dy > 0) stub = Math.min(STUB, ((dst.y - src.y) * src.dy) / 2);
  const a = [src.x + src.dx * stub, src.y + src.dy * stub];
  const b = [dst.x + dst.dx * stub, dst.y + dst.dy * stub];
  const mid = [];
  const [bs, bd] = boxes;
  // a free coordinate between the two blocks when they do not overlap on that axis, otherwise beyond both
  const lane = (lo, hi) =>
    lo.max + 2 * GRID <= hi.min ? (lo.max + hi.min) / 2 : hi.max + 2 * GRID <= lo.min ? (hi.max + lo.min) / 2 : Math.max(lo.max, hi.max) + 3 * GRID;
  if (srcH && dstH) {
    const forward = (b[0] - a[0]) * src.dx >= 0;
    if (src.dx === -dst.dx && forward) {
      // the usual case: output on one side, input facing it
      if (Math.abs(a[1] - b[1]) > 0.01) {
        const mx = (a[0] + b[0]) / 2;
        mid.push([mx, a[1]], [mx, b[1]]);
      }
    } else if (src.dx === -dst.dx) {
      // feedback: go around, between the blocks if there is room, otherwise below both
      const ly = lane({ min: bs[1], max: bs[1] + bs[3] }, { min: bd[1], max: bd[1] + bd[3] });
      mid.push([a[0], ly], [b[0], ly]);
    } else {
      // both ports face the same way: a U shape on that side
      const x = src.dx > 0 ? Math.max(a[0], b[0]) : Math.min(a[0], b[0]);
      mid.push([x, a[1]], [x, b[1]]);
    }
  } else if (!srcH && !dstH) {
    const forward = (b[1] - a[1]) * src.dy >= 0;
    if (src.dy === -dst.dy && forward) {
      if (Math.abs(a[0] - b[0]) > 0.01) {
        const my = (a[1] + b[1]) / 2;
        mid.push([a[0], my], [b[0], my]);
      }
    } else if (src.dy === -dst.dy) {
      const lx = lane({ min: bs[0], max: bs[0] + bs[2] }, { min: bd[0], max: bd[0] + bd[2] });
      mid.push([lx, a[1]], [lx, b[1]]);
    } else {
      const y = src.dy > 0 ? Math.max(a[1], b[1]) : Math.min(a[1], b[1]);
      mid.push([a[0], y], [b[0], y]);
    }
  } else if (srcH) {
    // leaves sideways, arrives from above or below
    const okX = (b[0] - a[0]) * src.dx >= 0, okY = (a[1] - b[1]) * dst.dy >= 0;
    if (okX && okY) mid.push([b[0], a[1]]);
    else mid.push([a[0], b[1]]);
  } else {
    const okY = (b[1] - a[1]) * src.dy >= 0, okX = (a[0] - b[0]) * dst.dx >= 0;
    if (okY && okX) mid.push([a[0], b[1]]);
    else mid.push([b[0], a[1]]);
  }
  return simplify([p0, a, ...mid, b, p1]);
}

function throughPoints(p0, src, via, p1, dst) {
  const pts = [p0];
  let horizontal = src.dx !== 0; // direction of the segment being drawn
  const fixed = via.map((p) => [p[0], p[1]]);
  // Corners that were made for slightly different port positions (imported files, resized blocks) are
  // moved onto the port's line, so the path does not end in a small jog.
  const pull = (list, port, axis) => {
    const first = list[0];
    if (!first || Math.abs(first[axis] - port[axis]) > 16 || first[axis] === port[axis]) return;
    const from = first[axis];
    for (const q of list) {
      if (q[axis] !== from) break;
      q[axis] = port[axis];
    }
  };
  pull(fixed, p0, src.dx !== 0 ? 1 : 0);
  pull(fixed.slice().reverse(), p1, dst.dx !== 0 ? 1 : 0);
  const all = [...fixed, null];
  for (let i = 0; i < all.length; i++) {
    const cur = pts[pts.length - 1];
    const last = all[i] === null;
    const next = last ? p1 : all[i];
    const dx = Math.abs(next[0] - cur[0]), dy = Math.abs(next[1] - cur[1]);
    if (dx > 0.01 && dy > 0.01) {
      // a corner is needed; the final corner makes the line enter the port head-on
      const enterH = dst.dx !== 0;
      if (last ? !enterH : horizontal) pts.push([next[0], cur[1]]);
      else pts.push([cur[0], next[1]]);
      horizontal = last ? enterH : !horizontal;
    } else if (dx > 0.01) horizontal = false;
    else if (dy > 0.01) horizontal = true;
    else horizontal = !horizontal;
    pts.push(next);
  }
  return pts;
}

/** SVG path data with rounded corners. */
export function pathData(pts, r = 5) {
  if (pts.length < 2) return "";
  let d = `M${pts[0][0]} ${pts[0][1]}`;
  for (let i = 1; i < pts.length - 1; i++) {
    const [a, b, c] = [pts[i - 1], pts[i], pts[i + 1]];
    const l1 = Math.hypot(b[0] - a[0], b[1] - a[1]), l2 = Math.hypot(c[0] - b[0], c[1] - b[1]);
    const rr = Math.min(r, l1 / 2, l2 / 2);
    if (rr < 0.5) {
      d += ` L${b[0]} ${b[1]}`;
      continue;
    }
    const p = [b[0] - ((b[0] - a[0]) / l1) * rr, b[1] - ((b[1] - a[1]) / l1) * rr];
    const q = [b[0] + ((c[0] - b[0]) / l2) * rr, b[1] + ((c[1] - b[1]) / l2) * rr];
    d += ` L${p[0]} ${p[1]} Q${b[0]} ${b[1]} ${q[0]} ${q[1]}`;
  }
  const e = pts[pts.length - 1];
  return d + ` L${e[0]} ${e[1]}`;
}

/** Routes of every line of a system: [{ line, pts }]. */
export function routeAll(sys) {
  const byName = new Map(sys.blocks.map((b) => [b.name, b]));
  const counts = new Map(sys.blocks.map((b) => [b.name, portCounts(b)]));
  const out = [];
  for (const line of sys.lines) {
    const s = byName.get(line.from[0]), d = byName.get(line.to[0]);
    if (!s || !d) continue;
    const sp = portPoint(s, "out", line.from[1], counts.get(s.name));
    const dp = portPoint(d, "in", line.to[1], counts.get(d.name));
    out.push({ line, pts: routeLine(sp, dp, [s.position, d.position], line.points) });
  }
  return out;
}

/** Where lines from one output part ways: the dots drawn on branches. */
export function branchPoints(routes) {
  const groups = new Map();
  for (const r of routes) {
    const k = r.line.from[0] + "\n" + r.line.from[1];
    if (!groups.has(k)) groups.set(k, []);
    groups.get(k).push(r.pts);
  }
  const dots = [];
  const onSeg = (p, a, b) =>
    (Math.abs(a[0] - b[0]) < 0.01 && Math.abs(p[0] - a[0]) < 0.01 && p[1] >= Math.min(a[1], b[1]) - 0.01 && p[1] <= Math.max(a[1], b[1]) + 0.01) ||
    (Math.abs(a[1] - b[1]) < 0.01 && Math.abs(p[1] - a[1]) < 0.01 && p[0] >= Math.min(a[0], b[0]) - 0.01 && p[0] <= Math.max(a[0], b[0]) + 0.01);
  for (const paths of groups.values()) {
    if (paths.length < 2) continue;
    for (let i = 0; i < paths.length; i++) {
      for (let j = 0; j < paths.length; j++) {
        if (i === j) continue;
        // the last point of path i that still lies on path j is where they split
        const A = paths[i], B = paths[j];
        let last = null;
        for (let k = 0; k < A.length - 1; k++) {
          let on = false;
          for (let m = 0; m < B.length - 1 && !on; m++) on = onSeg(A[k], B[m], B[m + 1]);
          if (!on) break;
          last = A[k];
        }
        if (last && !(last[0] === A[0][0] && last[1] === A[0][1]) && !dots.some((d) => d[0] === last[0] && d[1] === last[1])) dots.push(last);
      }
    }
  }
  return dots;
}

/* ---------- automatic layout ---------- */

/**
 * Arrange the blocks of a system in columns from sources to sinks. Feedback lines (found by a
 * depth-first search) are ignored when the columns are worked out. Clears all hand-made routes.
 */
export function autoLayout(sys) {
  const blocks = sys.blocks;
  if (!blocks.length) return;
  const idx = new Map(blocks.map((b, i) => [b.name, i]));
  const succ = blocks.map(() => []), pred = blocks.map(() => []);
  for (const l of sys.lines) {
    const a = idx.get(l.from[0]), b = idx.get(l.to[0]);
    if (a === undefined || b === undefined || a === b) continue;
    succ[a].push({ to: b, port: typeof l.to[1] === "number" ? l.to[1] : 0, from: l.from[1] });
  }
  // mark back edges
  const state = blocks.map(() => 0);
  const back = new Set();
  const order = blocks.map((_, i) => i).sort((a, b) => (pred[a].length - pred[b].length) || a - b);
  const indeg = blocks.map(() => 0);
  for (const list of succ) for (const e of list) indeg[e.to]++;
  const roots = order.filter((i) => indeg[i] === 0);
  const visit = (start) => {
    const stack = [[start, 0]];
    state[start] = 1;
    while (stack.length) {
      const top = stack[stack.length - 1];
      const [v, k] = top;
      if (k >= succ[v].length) {
        state[v] = 2;
        stack.pop();
        continue;
      }
      top[1]++;
      const w = succ[v][k].to;
      if (state[w] === 1) back.add(v + ">" + w);
      else if (state[w] === 0) {
        state[w] = 1;
        stack.push([w, 0]);
      }
    }
  };
  for (const r of roots) visit(r);
  for (let i = 0; i < blocks.length; i++) if (state[i] === 0) visit(i);
  // longest-path layering over the forward edges
  const layer = blocks.map(() => 0);
  const fwd = succ.map((list, v) => list.filter((e) => !back.has(v + ">" + e.to)));
  const deg = blocks.map(() => 0);
  for (const list of fwd) for (const e of list) deg[e.to]++;
  const queue = blocks.map((_, i) => i).filter((i) => deg[i] === 0);
  const topo = [];
  while (queue.length) {
    const v = queue.shift();
    topo.push(v);
    for (const e of fwd[v]) {
      layer[e.to] = Math.max(layer[e.to], layer[v] + 1);
      if (--deg[e.to] === 0) queue.push(e.to);
    }
  }
  for (const v of topo) for (const e of fwd[v]) pred[e.to].push({ from: v, port: e.port });
  // sinks go to the last column they can, so scopes line up on the right
  const maxLayer = Math.max(...layer);
  for (const v of topo.slice().reverse()) {
    if (!fwd[v].length) continue;
    const next = Math.min(...fwd[v].map((e) => layer[e.to]));
    if (!pred[v].length) layer[v] = next - 1; // sources sit right before what they feed
  }
  const columns = [];
  for (let i = 0; i < blocks.length; i++) (columns[layer[i]] = columns[layer[i]] || []).push(i);
  // order within a column by the average position of the neighbours in the column before
  const pos = blocks.map(() => 0);
  for (let c = 0; c <= maxLayer; c++) {
    const col = columns[c] || [];
    const score = (v) => {
      const ps = pred[v];
      if (!ps.length) return pos[v] || blocks[v].position[1] / 1000;
      return ps.reduce((s, p) => s + pos[p.from] + p.port * 0.01, 0) / ps.length;
    };
    col.sort((a, b) => score(a) - score(b) || a - b);
    col.forEach((v, k) => (pos[v] = k));
  }
  let x = 40;
  const snap = (v) => Math.round(v / GRID) * GRID;
  const centers = blocks.map(() => 0);
  for (let c = 0; c <= maxLayer; c++) {
    const col = columns[c] || [];
    let width = 0;
    for (const v of col) {
      const b = blocks[v];
      b.rotation = 0;
      b.flip = false;
      const [w, h] = b.position[2] > 0 ? [b.position[2], b.position[3]] : defaultSize(b);
      b.position[2] = w;
      b.position[3] = h;
      width = Math.max(width, w);
    }
    let y = 40;
    for (const v of col) {
      const b = blocks[v];
      // try to sit level with the first block that feeds this one
      const ps = pred[v];
      const want = ps.length ? centers[ps[0].from] - b.position[3] / 2 : y;
      const top = Math.max(y, snap(want));
      b.position[0] = snap(x + (width - b.position[2]) / 2);
      b.position[1] = top;
      centers[v] = top + b.position[3] / 2;
      y = top + b.position[3] + 40;
    }
    x += width + 70;
  }
  for (const l of sys.lines) delete l.points;
}

/** Align or spread the given blocks. how: left right top bottom center middle hspace vspace. */
export function alignBlocks(blocks, how) {
  if (blocks.length < 2) return;
  const xs = blocks.map((b) => b.position[0]), ys = blocks.map((b) => b.position[1]);
  const x2 = blocks.map((b) => b.position[0] + b.position[2]), y2 = blocks.map((b) => b.position[1] + b.position[3]);
  const snap = (v) => Math.round(v / (GRID / 2)) * (GRID / 2);
  if (how === "left") blocks.forEach((b) => (b.position[0] = Math.min(...xs)));
  else if (how === "right") blocks.forEach((b) => (b.position[0] = Math.max(...x2) - b.position[2]));
  else if (how === "top") blocks.forEach((b) => (b.position[1] = Math.min(...ys)));
  else if (how === "bottom") blocks.forEach((b) => (b.position[1] = Math.max(...y2) - b.position[3]));
  else if (how === "center") {
    const c = (Math.min(...xs) + Math.max(...x2)) / 2;
    blocks.forEach((b) => (b.position[0] = snap(c - b.position[2] / 2)));
  } else if (how === "middle") {
    const c = (Math.min(...ys) + Math.max(...y2)) / 2;
    blocks.forEach((b) => (b.position[1] = snap(c - b.position[3] / 2)));
  } else if (how === "hspace" || how === "vspace") {
    const k = how === "hspace" ? 0 : 1;
    const sorted = blocks.slice().sort((a, b) => a.position[k] - b.position[k]);
    const total = sorted.reduce((s, b) => s + b.position[k + 2], 0);
    const span = sorted[sorted.length - 1].position[k] + sorted[sorted.length - 1].position[k + 2] - sorted[0].position[k];
    const gap = (span - total) / (sorted.length - 1);
    let at = sorted[0].position[k];
    for (const b of sorted) {
      b.position[k] = snap(at);
      at += b.position[k + 2] + gap;
    }
  }
}
