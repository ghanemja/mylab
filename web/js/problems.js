// Problems as you type: the engine's checker (mylab.lint.check_source) runs in the worker a
// moment after each edit. Its findings are underlined in the editor, marked beside the line
// numbers and listed in the Problems tab. Nothing is run, so this is safe for any code.

import { $, el, icon } from "./ui.js";

/* global CodeMirror */

const DELAY = 500;
const MAX_SIZE = 300000;

function friendly(d) {
  if (d.name) return `mylab does not know a function or variable called “${d.name}”. Check the spelling, or create ${d.name}.m.`;
  let msg = String(d.message || "Problem").trim();
  msg = msg.charAt(0).toUpperCase() + msg.slice(1);
  return /[.!?]$/.test(msg) ? msg : msg + ".";
}

export function initProblems(app) {
  const { state } = app;
  const cm = app.editor;
  const found = new Map(); // path -> [{ line, from, to, severity, message }]
  const painted = new WeakMap(); // doc -> { marks, lines }
  let timer = null;
  let waiting = false;

  const enabled = () => app.settings.get("lint");

  /** Names the checker should accept: every .m file of the project can be called. */
  function knownNames() {
    const names = new Set();
    for (const p of state.files.keys()) if (p.endsWith(".m")) names.add(app.baseName(p).slice(0, -2));
    return [...names];
  }

  function toProblem(doc, d) {
    const last = doc.lastLine();
    let line = Math.max(0, Math.min(last, (d.line || 1) - 1));
    // An error "at the end of the file" is shown on the last line that has code.
    while (line > 0 && !doc.getLine(line).trim()) line--;
    const text = doc.getLine(line);
    let from = 0;
    let to = text.length;
    if (d.name) {
      const m = new RegExp("(^|[^\\w.])(" + d.name + ")\\b").exec(text);
      if (m) {
        from = m.index + m[1].length;
        to = from + d.name.length;
      }
    } else if (line === (d.line || 1) - 1 && d.column > 1 && d.column - 1 < text.length) {
      from = d.column - 1;
      const m = /^(\w+|\S)/.exec(text.slice(from));
      to = from + (m ? m[0].length : 1);
    } else {
      from = text.length - text.trimStart().length;
    }
    return { line, from, to, severity: d.severity === "error" ? "error" : "warning", message: friendly(d), name: d.name || null };
  }

  function unpaint(doc) {
    const p = painted.get(doc);
    if (!p) return;
    for (const m of p.marks) m.clear();
    for (const h of p.lines) {
      try {
        doc.setGutterMarker(h, "lint-gutter", null);
      } catch {
        /* the line is gone */
      }
    }
    painted.delete(doc);
  }

  function paint(path) {
    const doc = state.docs.get(path);
    if (!doc) return;
    unpaint(doc);
    const list = found.get(path) || [];
    if (!list.length) return;
    const p = { marks: [], lines: [] };
    const perLine = new Map();
    for (const d of list) {
      if (d.to > d.from) {
        p.marks.push(
          doc.markText(CodeMirror.Pos(d.line, d.from), CodeMirror.Pos(d.line, d.to), { className: "cm-lint cm-lint-" + d.severity, attributes: { "data-problem": d.message } })
        );
      }
      const prev = perLine.get(d.line);
      if (!prev || (prev.severity !== "error" && d.severity === "error")) perLine.set(d.line, d);
    }
    for (const [line, d] of perLine) {
      const dot = el("div", { class: "lint-mark " + d.severity, title: d.message, role: "img", "aria-label": (d.severity === "error" ? "Error: " : "Warning: ") + d.message });
      p.lines.push(doc.setGutterMarker(line, "lint-gutter", dot));
    }
    painted.set(doc, p);
  }

  function jump(path, d) {
    app.openFile(path);
    cm.setSelection(CodeMirror.Pos(d.line, d.from), CodeMirror.Pos(d.line, d.to), { scroll: false });
    cm.scrollIntoView({ line: d.line, ch: d.from }, 90);
    cm.focus();
  }

  function all() {
    const out = [];
    const err = state.lastError;
    if (err && err.file && err.line && state.files.has(err.file)) {
      out.push({ path: err.file, run: true, line: err.line - 1, from: 0, to: 0, severity: "error", message: err.message });
    }
    const paths = [...found.keys()].filter((p) => state.tabs.includes(p));
    paths.sort((a, b) => (a === state.active ? -1 : b === state.active ? 1 : a.localeCompare(b)));
    for (const path of paths) for (const d of found.get(path)) out.push({ path, ...d });
    return out;
  }

  function renderList() {
    const list = all();
    const box = $("problems-list");
    box.innerHTML = "";
    const count = $("problems-count");
    count.textContent = String(list.length);
    count.hidden = !list.length;
    count.classList.toggle("has-errors", list.some((d) => d.severity === "error"));
    $("tab-problems").setAttribute("aria-label", list.length ? `Problems (${list.length})` : "Problems");
    const empty = $("problems-empty");
    empty.hidden = list.length > 0;
    if (!list.length) {
      empty.textContent = !enabled()
        ? "Checking as you type is turned off. Turn it on in Settings."
        : app.engine && app.engine.ready
          ? "No problems found in the open files."
          : "Problems will show up here once the engine has loaded.";
    }
    for (const d of list) {
      box.append(
        el(
          "button",
          {
            type: "button",
            class: "problem-item " + d.severity,
            "data-path": d.path,
            "data-line": String(d.line + 1),
            "data-kind": d.run ? "run" : "check",
            onclick: () => (d.run ? app.gotoError({ file: d.path, line: d.line + 1 }) : jump(d.path, d)),
          },
          icon(d.severity === "error" ? "error" : "warn", "problem-ic"),
          el("span", { class: "problem-msg" }, d.run ? el("strong", {}, "When it ran: ") : null, d.message),
          el("span", { class: "problem-where muted" }, `${app.baseName(d.path)}:${d.line + 1}`)
        )
      );
    }
    app.emit("problems", list);
  }

  async function check(path) {
    if (!path || !path.endsWith(".m") || !app.isText(path)) return;
    if (!enabled()) return;
    const engine = app.engine;
    if (!engine || !engine.ready) return;
    if (engine.busy) {
      waiting = true;
      return;
    }
    const doc = app.getDoc(path);
    const gen = doc.changeGeneration();
    const text = doc.getValue();
    if (text.length > MAX_SIZE) return;
    let list;
    try {
      list = await engine.lint(text, knownNames());
    } catch {
      return;
    }
    if (state.docs.get(path) !== doc || doc.changeGeneration() !== gen) return; // the file changed meanwhile
    found.set(path, list.map((d) => toProblem(doc, d)));
    paint(path);
    renderList();
  }

  function schedule() {
    clearTimeout(timer);
    timer = setTimeout(() => check(state.active), DELAY);
  }

  function clearAll() {
    for (const doc of state.docs.values()) unpaint(doc);
    found.clear();
    renderList();
  }

  function refreshAll() {
    if (!enabled()) return clearAll();
    for (const p of state.tabs) check(p);
    return undefined;
  }

  /** The problem under a position of the active file, if any (for the hover text). */
  function problemAt(pos) {
    for (const d of found.get(state.active) || []) {
      if (d.line === pos.line && pos.ch >= d.from && pos.ch <= d.to) return d;
    }
    return null;
  }

  function step(dir) {
    const list = (found.get(state.active) || []).slice().sort((a, b) => a.line - b.line || a.from - b.from);
    if (!list.length) return;
    const cur = cm.getCursor();
    let i = dir > 0 ? list.findIndex((d) => d.line > cur.line || (d.line === cur.line && d.from > cur.ch)) : -1;
    if (dir < 0) {
      for (let k = list.length - 1; k >= 0; k--) {
        if (list[k].line < cur.line || (list[k].line === cur.line && list[k].to < cur.ch)) {
          i = k;
          break;
        }
      }
    }
    if (i < 0) i = dir > 0 ? 0 : list.length - 1;
    jump(state.active, list[i]);
    app.showTip && app.showTip(CodeMirror.Pos(list[i].line, list[i].from), { problem: list[i] });
  }

  cm.on("changes", () => state.active && schedule());
  app.on("fileOpened", (path) => {
    if (!found.has(path)) check(path);
    else renderList();
  });
  app.on("tabsChanged", () => {
    for (const p of [...found.keys()]) if (!state.tabs.includes(p)) found.delete(p);
    renderList();
  });
  app.on("projectChanged", () => {
    found.clear();
    renderList();
  });
  app.on("engineReady", refreshAll);
  app.on("runFinished", () => {
    renderList();
    check(state.active);
  });
  app.on("settingsChanged", (key) => key === "lint" && refreshAll());
  app.engine.addEventListener("idle", () => {
    if (!waiting) return;
    waiting = false;
    schedule();
  });

  app.commands.register([
    { id: "problems.show", group: "View", title: "Show the Problems list", keys: "Mod-Shift-M", run: () => app.showBottom("problems") },
    { id: "problems.next", group: "Go", title: "Go to the next problem", keys: "F8", run: () => step(1), when: () => !!state.active },
    { id: "problems.prev", group: "Go", title: "Go to the previous problem", keys: "Shift-F8", run: () => step(-1), when: () => !!state.active },
  ]);

  renderList();
  Object.assign(app, { checkProblems: check, problemAt, problemsOf: (path) => found.get(path) || [], allProblems: all });
}
