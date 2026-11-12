// Find and replace in the current file, and search across all files of the project.

import { $, el, toast, isMac } from "./ui.js";

/* global CodeMirror */

const MAX_MARKS = 1500;
const MAX_RESULTS = 1000;

const escapeRegex = (s) => s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");

/** Build the search expression. Returns { re } or { error }. */
export function buildQuery(query, { regex = false, matchCase = false, word = false } = {}) {
  if (!query) return { re: null };
  let src = regex ? query : escapeRegex(query);
  if (word) src = `\\b(?:${src})\\b`;
  try {
    return { re: new RegExp(src, "gm" + (matchCase ? "" : "i")), src };
  } catch (e) {
    return { error: "This is not a valid regular expression: " + String(e.message).replace(/^Invalid regular expression: /, "") };
  }
}

/** All matches of re in text: [{ index, length, groups }]. Empty matches are skipped. */
export function findAll(text, re, limit = Infinity) {
  const out = [];
  re.lastIndex = 0;
  let m;
  while ((m = re.exec(text)) && out.length < limit) {
    if (m[0].length === 0) {
      re.lastIndex++;
      continue;
    }
    out.push({ index: m.index, length: m[0].length, text: m[0] });
  }
  return out;
}

function toggleButton(id, onChange) {
  const b = $(id);
  b.addEventListener("click", () => {
    b.setAttribute("aria-pressed", b.getAttribute("aria-pressed") === "true" ? "false" : "true");
    onChange();
  });
  return () => b.getAttribute("aria-pressed") === "true";
}

export function initSearch(app) {
  const { state } = app;
  const cm = app.editor;

  /* ================================================================ find in file */

  const bar = $("find-bar");
  const input = $("find-input");
  const replaceInput = $("replace-input");
  let matches = []; // [{ from, to, text }]
  let marks = [];
  let currentMark = null;
  let current = -1;
  let query = { re: null };
  let timer = null;

  const isCase = toggleButton("find-case", () => refresh(true));
  const isWord = toggleButton("find-word", () => refresh(true));
  const isRegex = toggleButton("find-regex", () => refresh(true));
  const options = () => ({ regex: isRegex(), matchCase: isCase(), word: isWord() });

  function clearMarks() {
    for (const m of marks) m.clear();
    marks = [];
    if (currentMark) currentMark.clear();
    currentMark = null;
  }

  function showCount() {
    const box = $("find-count");
    input.classList.toggle("invalid", !!query.error);
    if (query.error) box.textContent = "Not valid";
    else if (!input.value) box.textContent = "";
    else if (!matches.length) box.textContent = "No results";
    else box.textContent = `${current >= 0 ? current + 1 : "?"} of ${matches.length.toLocaleString()}`;
    box.title = query.error || "";
    bar.dataset.count = String(matches.length);
    bar.dataset.current = String(current + 1);
    for (const id of ["find-prev", "find-next", "replace-one", "replace-all"]) $(id).disabled = !matches.length;
  }

  /** Recompute the matches. When `jump` is set, move to the first match at or after the cursor. */
  function refresh(jump = false) {
    clearMarks();
    matches = [];
    current = -1;
    query = bar.hidden ? { re: null } : buildQuery(input.value, options());
    if (query.re && state.active) {
      const doc = cm.getDoc();
      const found = findAll(doc.getValue(), query.re);
      matches = found.map((f) => ({ from: doc.posFromIndex(f.index), to: doc.posFromIndex(f.index + f.length), text: f.text }));
      cm.operation(() => {
        for (const m of matches.slice(0, MAX_MARKS)) marks.push(doc.markText(m.from, m.to, { className: "cm-find-match" }));
      });
      if (matches.length) {
        const cur = cm.getCursor("from");
        let i = matches.findIndex((m) => CodeMirror.cmpPos(m.from, cur) >= 0);
        if (i < 0) i = 0;
        if (jump) select(i, false);
        else {
          // Keep pointing at the match that is selected, if there is one.
          const sel = matches.findIndex((m) => CodeMirror.cmpPos(m.from, cm.getCursor("from")) === 0 && CodeMirror.cmpPos(m.to, cm.getCursor("to")) === 0);
          current = sel;
        }
      }
    }
    showCount();
  }

  function select(i, focusEditor) {
    if (!matches.length) return;
    current = (i + matches.length) % matches.length;
    const m = matches[current];
    if (currentMark) currentMark.clear();
    currentMark = cm.getDoc().markText(m.from, m.to, { className: "cm-find-current" });
    cm.setSelection(m.from, m.to, { scroll: false });
    cm.scrollIntoView({ from: m.from, to: m.to }, 80);
    if (focusEditor) cm.focus();
    showCount();
  }

  function step(dir) {
    if (bar.hidden) return openFind(false);
    if (!matches.length) return undefined;
    if (current < 0) {
      const cur = cm.getCursor(dir > 0 ? "to" : "from");
      let i = matches.findIndex((m) => CodeMirror.cmpPos(m.from, cur) >= 0);
      if (i < 0) i = 0;
      return select(dir > 0 ? i : i - 1, false);
    }
    return select(current + dir, false);
  }

  function replacementFor(m) {
    const text = replaceInput.value;
    if (!isRegex()) return text;
    // Let $1, $2, $& … refer to the groups of this match.
    const one = new RegExp(query.src, query.re.flags.replace("g", ""));
    return m.text.replace(one, text);
  }

  function replaceOne() {
    if (!matches.length || cm.getOption("readOnly")) return;
    if (current < 0) return void select(0, false);
    const m = matches[current];
    const at = cm.getDoc().indexFromPos(m.from);
    const rep = replacementFor(m);
    cm.replaceRange(rep, m.from, m.to, "+replace");
    refreshNow();
    if (matches.length) {
      const doc = cm.getDoc();
      const after = doc.posFromIndex(at + rep.length);
      let i = matches.findIndex((x) => CodeMirror.cmpPos(x.from, after) >= 0);
      if (i < 0) i = 0;
      select(i, false);
    }
  }

  function replaceAll() {
    if (!matches.length || cm.getOption("readOnly")) return;
    const n = matches.length;
    cm.operation(() => {
      for (let i = matches.length - 1; i >= 0; i--) cm.replaceRange(replacementFor(matches[i]), matches[i].from, matches[i].to, "+replaceAll");
    });
    refreshNow();
    toast(`Replaced ${n.toLocaleString()} match${n === 1 ? "" : "es"}. Undo with Ctrl+Z.`);
  }

  function refreshNow() {
    clearTimeout(timer);
    refresh(false);
  }

  function showReplace(show) {
    $("replace-row").hidden = !show;
    $("find-toggle-replace").setAttribute("aria-expanded", String(show));
    $("find-toggle-replace").title = show ? "Hide replace" : "Show replace";
  }

  function openFind(withReplace) {
    if (!state.active) {
      toast("Open a file first.");
      return;
    }
    const wasHidden = bar.hidden;
    // Start from the selection, but only when coming from the editor (not while typing in the bar).
    const sel = wasHidden || cm.hasFocus() ? cm.getSelection() : "";
    bar.hidden = false;
    if (withReplace || wasHidden) showReplace(!!withReplace);
    if (sel && !sel.includes("\n") && sel.length < 200) input.value = isRegex() ? escapeRegex(sel) : sel;
    else if (!input.value && wasHidden) {
      const w = cm.findWordAt(cm.getCursor());
      const word = cm.getRange(w.anchor, w.head);
      if (/^\w+$/.test(word)) input.value = word;
    }
    cm.refresh();
    refresh(true);
    const field = withReplace && !wasHidden && input.value ? replaceInput : input;
    field.focus();
    field.select();
  }

  function closeFind() {
    if (bar.hidden) return;
    bar.hidden = true;
    clearMarks();
    matches = [];
    cm.refresh();
    cm.focus();
  }

  input.addEventListener("input", () => refresh(true));
  const onKey = (e) => {
    if (e.key === "Enter") {
      e.preventDefault();
      if (e.target === replaceInput) (e.ctrlKey || e.metaKey) && e.altKey ? replaceAll() : replaceOne();
      else step(e.shiftKey ? -1 : 1);
    } else if (e.key === "Escape") {
      e.preventDefault();
      e.stopPropagation();
      closeFind();
    }
  };
  input.addEventListener("keydown", onKey);
  replaceInput.addEventListener("keydown", onKey);
  bar.addEventListener("keydown", (e) => e.key === "Escape" && (e.preventDefault(), e.stopPropagation(), closeFind()));
  $("find-next").addEventListener("click", () => step(1));
  $("find-prev").addEventListener("click", () => step(-1));
  $("find-close").addEventListener("click", closeFind);
  $("find-toggle-replace").addEventListener("click", () => showReplace($("replace-row").hidden));
  $("replace-one").addEventListener("click", replaceOne);
  $("replace-all").addEventListener("click", replaceAll);
  cm.on("changes", () => {
    if (bar.hidden) return;
    clearTimeout(timer);
    timer = setTimeout(() => refresh(false), 150);
  });
  cm.on("swapDoc", () => !bar.hidden && refresh(false));

  /* ================================================================ search in all files */

  const sInput = $("search-input");
  let sTimer = null;
  const sCase = toggleButton("search-case", () => runSearch());
  const sWord = toggleButton("search-word", () => runSearch());
  const sRegex = toggleButton("search-regex", () => runSearch());

  function searchProject(text, opts) {
    const q = buildQuery(text, opts);
    if (q.error) return { error: q.error, files: [], total: 0 };
    if (!q.re) return { files: [], total: 0 };
    const files = [];
    let total = 0;
    let capped = false;
    for (const path of app.sortedPaths()) {
      if (!app.isText(path)) continue;
      const content = app.liveContent(path);
      if (content.length > 2e6) continue;
      const found = findAll(content, q.re, MAX_RESULTS - total + 1);
      if (!found.length) continue;
      // Line starts, to turn offsets into line numbers.
      const starts = [0];
      for (let i = content.indexOf("\n"); i >= 0; i = content.indexOf("\n", i + 1)) starts.push(i + 1);
      const hits = [];
      let li = 0;
      for (const f of found) {
        if (total >= MAX_RESULTS) {
          capped = true;
          break;
        }
        while (li + 1 < starts.length && starts[li + 1] <= f.index) li++;
        const lineEnd = li + 1 < starts.length ? starts[li + 1] - 1 : content.length;
        const lineText = content.slice(starts[li], lineEnd);
        const ch = f.index - starts[li];
        hits.push({ line: li, ch, length: Math.min(f.length, lineText.length - ch), full: f.length, text: lineText, index: f.index });
        total++;
      }
      files.push({ path, hits });
      if (capped) break;
    }
    return { files, total, capped };
  }

  function preview(hit) {
    // Show the match with some text around it, trimmed on long lines.
    const start = Math.max(0, hit.ch - 30);
    const lead = hit.text.slice(start, hit.ch).replace(/^\s+/, "");
    const mid = hit.text.slice(hit.ch, hit.ch + hit.length);
    const tail = hit.text.slice(hit.ch + hit.length, hit.ch + hit.length + 80);
    return [start > 0 && lead ? "…" : "", lead, el("mark", {}, mid), tail];
  }

  function jumpTo(path, hit) {
    app.openFile(path);
    const doc = cm.getDoc();
    const from = doc.posFromIndex(hit.index);
    const to = doc.posFromIndex(hit.index + hit.full);
    cm.setSelection(from, to, { scroll: false });
    cm.scrollIntoView({ from, to }, 100);
    cm.focus();
  }

  function runSearch() {
    const box = $("search-results");
    const status = $("search-status");
    const res = searchProject(sInput.value, { regex: sRegex(), matchCase: sCase(), word: sWord() });
    box.innerHTML = "";
    sInput.classList.toggle("invalid", !!res.error);
    if (res.error) status.textContent = res.error;
    else if (!sInput.value) status.textContent = "Type something to search every text file in this project.";
    else if (!res.total) status.textContent = "No results.";
    else {
      status.textContent =
        `${res.total.toLocaleString()} result${res.total === 1 ? "" : "s"} in ${res.files.length} file${res.files.length === 1 ? "" : "s"}` +
        (res.capped ? ` (showing the first ${MAX_RESULTS.toLocaleString()})` : "");
    }
    box.dataset.total = String(res.total);
    for (const f of res.files) {
      const group = el("div", { class: "search-file", "data-path": f.path });
      group.append(el("div", { class: "search-file-name" }, f.path, el("span", { class: "pill small" }, String(f.hits.length))));
      for (const hit of f.hits) {
        group.append(
          el(
            "button",
            { type: "button", class: "search-hit", "data-line": String(hit.line + 1), title: `${f.path}, line ${hit.line + 1}`, onclick: () => jumpTo(f.path, hit) },
            el("span", { class: "search-line" }, String(hit.line + 1)),
            el("span", { class: "search-text" }, preview(hit))
          )
        );
      }
      box.append(group);
    }
    return res;
  }

  function openSearch() {
    const sel = state.active ? cm.getSelection() : "";
    if (sel && !sel.includes("\n") && sel.length < 200) sInput.value = sel;
    app.showBottom("search");
    runSearch();
    sInput.focus();
    sInput.select();
  }

  sInput.addEventListener("input", () => {
    clearTimeout(sTimer);
    sTimer = setTimeout(runSearch, 180);
  });
  sInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter") {
      e.preventDefault();
      clearTimeout(sTimer);
      runSearch();
      const first = $("search-results").querySelector(".search-hit");
      if (first) first.focus();
    }
  });
  $("search-results").addEventListener("keydown", (e) => {
    if (e.key !== "ArrowDown" && e.key !== "ArrowUp") return;
    const hits = [...$("search-results").querySelectorAll(".search-hit")];
    const i = hits.indexOf(document.activeElement);
    if (i < 0) return;
    e.preventDefault();
    const next = hits[i + (e.key === "ArrowDown" ? 1 : -1)];
    if (next) next.focus();
    else if (e.key === "ArrowUp") sInput.focus();
  });

  app.commands.register([
    { id: "find.open", group: "Find", title: "Find in this file", keys: "Mod-F", run: () => openFind(false), when: () => !!state.active },
    { id: "find.replace", group: "Find", title: "Find and replace in this file", keys: isMac ? "Mod-Alt-F" : "Mod-H", run: () => openFind(true), when: () => !!state.active },
    { id: "find.next", group: "Find", title: "Next match", keys: isMac ? ["Mod-G", "F3"] : "F3", run: () => step(1), when: () => !!state.active },
    { id: "find.prev", group: "Find", title: "Previous match", keys: isMac ? ["Mod-Shift-G", "Shift-F3"] : "Shift-F3", run: () => step(-1), when: () => !!state.active },
    { id: "find.inFiles", group: "Find", title: "Search in all files", keys: "Mod-Shift-F", run: openSearch },
  ]);

  Object.assign(app, { openFind, closeFind, openSearch, searchProject, runSearch });
}
