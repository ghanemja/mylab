// Line-by-line comparison of two texts, and a view that shows it.
// Used for the version history and for edits the assistant proposes.

import { el } from "./ui.js";

/**
 * Compare two texts line by line.
 * Returns [{ type: " " | "+" | "-", text, a, b }] where a and b are 1-based line numbers
 * in the old and new text (null on the side a line is missing from).
 */
export function diffLines(oldText, newText) {
  const A = String(oldText).split("\n");
  const B = String(newText).split("\n");
  let start = 0;
  while (start < A.length && start < B.length && A[start] === B[start]) start++;
  let endA = A.length;
  let endB = B.length;
  while (endA > start && endB > start && A[endA - 1] === B[endB - 1]) {
    endA--;
    endB--;
  }
  const out = [];
  for (let i = 0; i < start; i++) out.push({ type: " ", text: A[i], a: i + 1, b: i + 1 });
  const n = endA - start;
  const m = endB - start;
  if (n * m > 12e6) {
    // Too big to compare in detail: show the middle as replaced.
    for (let i = 0; i < n; i++) out.push({ type: "-", text: A[start + i], a: start + i + 1, b: null });
    for (let j = 0; j < m; j++) out.push({ type: "+", text: B[start + j], a: null, b: start + j + 1 });
  } else if (n || m) {
    // Longest common subsequence table, filled from the end.
    const w = m + 1;
    const t = new Uint32Array((n + 1) * w);
    for (let i = n - 1; i >= 0; i--) {
      for (let j = m - 1; j >= 0; j--) {
        t[i * w + j] = A[start + i] === B[start + j] ? t[(i + 1) * w + j + 1] + 1 : Math.max(t[(i + 1) * w + j], t[i * w + j + 1]);
      }
    }
    let i = 0;
    let j = 0;
    while (i < n || j < m) {
      if (i < n && j < m && A[start + i] === B[start + j]) {
        out.push({ type: " ", text: A[start + i], a: start + i + 1, b: start + j + 1 });
        i++;
        j++;
      } else if (j < m && (i === n || t[i * w + j + 1] > t[(i + 1) * w + j])) {
        out.push({ type: "+", text: B[start + j], a: null, b: start + j + 1 });
        j++;
      } else {
        out.push({ type: "-", text: A[start + i], a: start + i + 1, b: null });
        i++;
      }
    }
  }
  for (let i = endA, j = endB; i < A.length; i++, j++) out.push({ type: " ", text: A[i], a: i + 1, b: j + 1 });
  return out;
}

export function diffStats(rows) {
  let added = 0;
  let removed = 0;
  for (const r of rows) {
    if (r.type === "+") added++;
    else if (r.type === "-") removed++;
  }
  return { added, removed };
}

/** Fill `box` with a unified view of the changes. Long unchanged runs are folded away. */
export function renderDiff(box, oldText, newText, { context = 3 } = {}) {
  const rows = diffLines(oldText, newText);
  const stats = diffStats(rows);
  box.innerHTML = "";
  box.classList.add("diff");
  if (!stats.added && !stats.removed) {
    box.append(el("p", { class: "muted diff-same" }, "No differences. The two versions are the same."));
    return stats;
  }
  const keep = new Uint8Array(rows.length);
  rows.forEach((r, i) => {
    if (r.type === " ") return;
    for (let k = Math.max(0, i - context); k <= Math.min(rows.length - 1, i + context); k++) keep[k] = 1;
  });
  const table = el("table", { class: "diff-table", role: "presentation" });
  const body = el("tbody");
  table.append(body);
  let i = 0;
  while (i < rows.length) {
    if (!keep[i]) {
      let j = i;
      while (j < rows.length && !keep[j]) j++;
      body.append(el("tr", { class: "diff-skip" }, el("td", { colspan: "4" }, `⋯ ${j - i} unchanged line${j - i === 1 ? "" : "s"}`)));
      i = j;
      continue;
    }
    const r = rows[i];
    const cls = r.type === "+" ? "diff-add" : r.type === "-" ? "diff-del" : "diff-ctx";
    const label = r.type === "+" ? "added" : r.type === "-" ? "removed" : "";
    body.append(
      el(
        "tr",
        { class: cls },
        el("td", { class: "diff-num" }, r.a === null ? "" : String(r.a)),
        el("td", { class: "diff-num" }, r.b === null ? "" : String(r.b)),
        el("td", { class: "diff-sign", "aria-label": label || null }, r.type === " " ? "" : r.type),
        el("td", { class: "diff-text" }, r.text || " ")
      )
    );
    i++;
  }
  box.append(table);
  return stats;
}

export function statsText(stats) {
  if (!stats.added && !stats.removed) return "No changes";
  const parts = [];
  if (stats.added) parts.push(`${stats.added} line${stats.added === 1 ? "" : "s"} added`);
  if (stats.removed) parts.push(`${stats.removed} line${stats.removed === 1 ? "" : "s"} removed`);
  return parts.join(", ");
}
