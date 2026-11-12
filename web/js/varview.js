// Variable viewer: a grid for matrices, tables, cell arrays and structs (one page at a time
// for large values), with "copy as CSV" and quick plots. The pages come from the engine
// worker ("inspect" and "csv" requests); nothing is copied out of the browser.

import { $, el, toast, copyText } from "./ui.js";

const PAGE_ROWS = 100;
const PAGE_COLS = 25;

const PLOTS = {
  plot: (x) => `figure; plot(${x}); grid on`,
  bar: (x) => `figure; bar(${x})`,
  histogram: (x) => `figure; histogram(${x}(:))`,
  imagesc: (x) => `figure; imagesc(${x}); colorbar`,
};

export function initVarView(app) {
  let cur = null; // { name, path: [steps], labels: [text], row, col, data, view }

  /** The MATLAB expression for the value being shown, e.g. s.data{2,1}. */
  function expression() {
    return cur.name + cur.labels.join("");
  }

  function stepLabel(kind, data, r, c) {
    if (kind === "struct") return "." + data.rownames[r - data.r0];
    if (kind === "structarray") return `(${r + 1}).${data.colnames[c - data.c0]}`;
    if (kind === "cell") return `{${r + 1},${c + 1}}`;
    return "";
  }

  function stepPath(kind, data, r, c) {
    if (kind === "struct") return ["f", data.rownames[r - data.r0], 0];
    if (kind === "structarray") return ["f", data.colnames[c - data.c0], r];
    return ["c", r, c];
  }

  function renderCrumbs() {
    const box = $("var-crumbs");
    box.innerHTML = "";
    box.hidden = !cur.path.length;
    if (!cur.path.length) return;
    const go = (depth) => () => {
      cur.path = cur.path.slice(0, depth);
      cur.labels = cur.labels.slice(0, depth);
      cur.row = cur.col = 0;
      load();
    };
    box.append(el("button", { type: "button", class: "linklike", onclick: go(0) }, cur.name));
    cur.labels.forEach((label, i) => {
      if (i === cur.labels.length - 1) box.append(el("span", {}, label));
      else box.append(el("button", { type: "button", class: "linklike", onclick: go(i + 1) }, label));
    });
  }

  function renderGrid(d) {
    const table = $("var-grid");
    table.innerHTML = "";
    const numeric = d.kind === "matrix";
    table.className = "data-grid" + (numeric ? " numeric" : "");
    const head = el("tr", {}, el("th", { class: "corner", scope: "col" }, ""));
    for (let c = 0; c < d.cells[0].length; c++) head.append(el("th", { scope: "col" }, d.colnames ? d.colnames[c] : String(d.c0 + c + 1)));
    table.append(el("thead", {}, head));
    const body = el("tbody");
    const opens = new Set((d.opens || []).map((o) => o[0] + "," + o[1]));
    d.cells.forEach((row, i) => {
      const r = d.r0 + i;
      const tr = el("tr", {}, el("th", { scope: "row" }, d.rownames ? d.rownames[i] : String(r + 1)));
      row.forEach((text, j) => {
        const c = d.c0 + j;
        if (opens.has(r + "," + c)) {
          tr.append(
            el(
              "td",
              { class: "open" },
              el(
                "button",
                {
                  type: "button",
                  class: "linklike",
                  title: "Open this value",
                  onclick: () => {
                    cur.labels.push(stepLabel(d.kind, d, r, c));
                    cur.path.push(stepPath(d.kind, d, r, c));
                    cur.row = cur.col = 0;
                    load();
                  },
                },
                text
              )
            )
          );
        } else tr.append(el("td", { title: text.length > 14 ? text : null }, text));
      });
      body.append(tr);
    });
    table.append(body);
  }

  function renderPager(d) {
    const rowsEnd = d.r0 + d.cells.length;
    const colsEnd = d.c0 + (d.cells[0] ? d.cells[0].length : 0);
    const moreRows = d.rows > PAGE_ROWS;
    const moreCols = d.cols > PAGE_COLS;
    $("var-pager").hidden = !moreRows && !moreCols;
    $("var-rows").hidden = !moreRows;
    $("var-cols").hidden = !moreCols;
    $("var-rows-text").textContent = `Rows ${(d.r0 + 1).toLocaleString()}–${rowsEnd.toLocaleString()} of ${d.rows.toLocaleString()}`;
    $("var-cols-text").textContent = `Columns ${d.c0 + 1}–${colsEnd} of ${d.cols.toLocaleString()}`;
    $("var-rows-first").disabled = $("var-rows-prev").disabled = d.r0 === 0;
    $("var-rows-last").disabled = $("var-rows-next").disabled = rowsEnd >= d.rows;
    $("var-cols-prev").disabled = d.c0 === 0;
    $("var-cols-next").disabled = colsEnd >= d.cols;
  }

  function setView(view) {
    cur.view = view;
    const grid = view === "grid" && cur.data && cur.data.kind !== "text";
    $("var-grid-wrap").hidden = !grid;
    $("var-body").hidden = grid;
    $("var-pager").hidden = !grid || !(cur.data.rows > PAGE_ROWS || cur.data.cols > PAGE_COLS);
    $("var-view-grid").setAttribute("aria-pressed", String(grid));
    $("var-view-text").setAttribute("aria-pressed", String(!grid));
    $("var-view").hidden = !cur.data || cur.data.kind === "text" || cur.path.length > 0;
  }

  async function load() {
    const engine = app.engine;
    let d;
    try {
      d = await engine.inspect(cur.name, { path: cur.path, row: cur.row, rows: PAGE_ROWS, col: cur.col, cols: PAGE_COLS });
    } catch (e) {
      toast(e.message, "error");
      return false;
    }
    cur.data = d;
    const dlg = $("dlg-var");
    dlg.dataset.kind = d.kind;
    dlg.dataset.expr = expression();
    $("var-title").textContent = expression();
    $("var-meta").textContent = `${d.size} ${d.class}`;
    $("var-note").hidden = !d.note;
    $("var-note").textContent = d.note || "";
    $("var-plots").hidden = !d.plottable;
    $("var-copy-csv").hidden = d.kind === "text";
    renderCrumbs();
    if (d.kind === "text") {
      $("var-body").textContent = d.text.replace(/\n+$/, "");
      setView("text");
    } else {
      renderGrid(d);
      renderPager(d);
      setView(cur.path.length ? "grid" : cur.view || "grid");
    }
    return true;
  }

  async function showVariable(v) {
    const engine = app.engine;
    if (!engine.ready || engine.busy) {
      toast("Wait for the running code to finish.");
      return;
    }
    cur = { name: v.name, path: [], labels: [], row: 0, col: 0, data: null, view: "grid" };
    $("var-body").textContent = "";
    // The printed form (what the command window would show) is the "Text" view.
    const text = engine.showVariable(v.name).then(
      (t) => t.replace(/\n+$/, ""),
      () => ""
    );
    if (!(await load())) return;
    if (cur.data.kind !== "text") $("var-body").textContent = await text;
    if (!$("dlg-var").open) $("dlg-var").showModal();
  }

  const page = (fn) => () => {
    if (!cur || !cur.data) return;
    fn(cur.data);
    load();
  };
  $("var-rows-first").addEventListener("click", page(() => (cur.row = 0)));
  $("var-rows-prev").addEventListener("click", page(() => (cur.row = Math.max(0, cur.row - PAGE_ROWS))));
  $("var-rows-next").addEventListener("click", page((d) => (cur.row = Math.min(d.rows - 1, cur.row + PAGE_ROWS))));
  $("var-rows-last").addEventListener("click", page((d) => (cur.row = Math.floor((d.rows - 1) / PAGE_ROWS) * PAGE_ROWS)));
  $("var-cols-prev").addEventListener("click", page(() => (cur.col = Math.max(0, cur.col - PAGE_COLS))));
  $("var-cols-next").addEventListener("click", page((d) => (cur.col = Math.min(d.cols - 1, cur.col + PAGE_COLS))));
  $("var-view-grid").addEventListener("click", () => cur && setView("grid"));
  $("var-view-text").addEventListener("click", () => cur && setView("text"));

  $("var-copy").addEventListener("click", async () => {
    const text = cur && cur.view === "grid" && cur.data.kind !== "text" ? $("var-grid").innerText : $("var-body").textContent;
    const ok = await copyText(text);
    toast(ok ? "Copied." : "Couldn't access the clipboard.", ok ? "" : "error");
  });

  $("var-copy-csv").addEventListener("click", async () => {
    if (!cur) return;
    try {
      const res = await app.engine.csv(cur.name, { path: cur.path, maxCells: 500000 });
      if (!(await copyText(res.csv))) throw new Error("Couldn't access the clipboard.");
      toast(res.truncated ? `Copied the first ${res.rows.toLocaleString()} rows as CSV (the value is very large).` : "Copied as CSV. Paste it into a spreadsheet or a text file.");
    } catch (e) {
      toast(e.message, "error");
    }
  });

  document.querySelectorAll("#var-plots [data-plot]").forEach((b) =>
    b.addEventListener("click", () => {
      if (!cur) return;
      const code = PLOTS[b.dataset.plot](expression());
      $("dlg-var").close();
      app.runCode(code, { kind: "command" });
    })
  );

  Object.assign(app, { showVariable });
}
