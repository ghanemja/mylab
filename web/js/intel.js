// Code help in the editor: hover help, usage hints while typing a call, F1 help, richer
// autocomplete, go to definition, the outline and renaming a name within a file.
//
// The help comes from the engine (mylab.help_data, fetched once when the engine is ready)
// and from the project's own files (function lines and the comments under them).

import { $, el, toast, isMac, promptDialog } from "./ui.js";
import { findFunctions, helpComments, KEYWORDS } from "./blocks.js";
import { splitSections } from "./sections.js";
import { fuzzy } from "./commands.js";

/* global CodeMirror */

const HOVER_DELAY = 380;
const IDENT = /^[A-Za-z_]\w*$/;

export function initIntel(app) {
  const { state } = app;
  const cm = app.editor;
  let helpIndex = null; // { name: [usage lines, description] }
  const setting = (k) => app.settings.get(k);

  /* ================================================================ what is this name? */

  const fileCache = new Map(); // path -> { content, functions }
  function functionsIn(path) {
    const content = app.liveContent(path);
    if (typeof content !== "string") return [];
    const c = fileCache.get(path);
    if (c && c.content === content) return c.functions;
    const functions = findFunctions(content);
    fileCache.set(path, { content, functions });
    return functions;
  }

  /** The project file that defines the function `name` for code in the active file, or null. */
  function fileFor(name) {
    const cands = [];
    for (const p of state.files.keys()) if (p.endsWith(".m") && app.baseName(p) === name + ".m") cands.push(p);
    if (!cands.length) return null;
    const here = state.active ? app.dirName(state.active) : "";
    return cands.find((p) => app.dirName(p) === here) || cands.sort((a, b) => a.length - b.length)[0];
  }

  /** Everything known about a name: { kind, name, usage, text, path, line, col, variable }. */
  function lookup(name) {
    if (!IDENT.test(name)) return null;
    const variable = state.variables.find((v) => v.name === name) || null;
    if (state.active && state.active.endsWith(".m")) {
      const local = functionsIn(state.active).find((f) => f.name === name);
      if (local) return { kind: "function", name, usage: [local.signature], text: local.doc, path: state.active, line: local.line, col: local.col, local: true, variable };
    }
    const path = fileFor(name);
    if (path) {
      const fns = functionsIn(path);
      const main = fns.find((f) => f.name === name) || (app.isFunctionFile(app.liveContent(path)) ? fns[0] : null);
      if (main) return { kind: "function", name, usage: [main.signature], text: main.doc, path, line: main.line, col: main.col, variable };
      return { kind: "script", name, usage: [name], text: "A script in this project.", path, line: 0, col: 0, variable };
    }
    if (helpIndex && Object.prototype.hasOwnProperty.call(helpIndex, name)) {
      const [usage, text] = helpIndex[name];
      return { kind: "builtin", name, usage, text, variable };
    }
    if (KEYWORDS.includes(name)) return variable ? { kind: "variable", name, variable } : null;
    return variable ? { kind: "variable", name, variable } : null;
  }

  /** The identifier at a position: { word, from, to, line } or null (also null in comments and strings). */
  function wordAt(pos, { anywhere = false } = {}) {
    const text = cm.getLine(pos.line);
    if (text == null) return null;
    let from = pos.ch;
    let to = pos.ch;
    while (from > 0 && /\w/.test(text[from - 1])) from--;
    while (to < text.length && /\w/.test(text[to])) to++;
    const word = text.slice(from, to);
    if (!IDENT.test(word)) return null;
    if (!anywhere) {
      const type = cm.getTokenTypeAt(CodeMirror.Pos(pos.line, from + 1)) || "";
      if (/comment|string/.test(type)) return null;
    }
    return { word, from, to, line: pos.line, field: text[from - 1] === "." };
  }

  /* ================================================================ tooltips */

  const tip = el("div", { id: "code-tip", class: "code-tip", role: "tooltip", hidden: true });
  document.body.appendChild(tip);
  let tipKind = null; // "hover" | "sig" | "pinned"

  function hideTip(kind) {
    if (kind && tipKind !== kind) return;
    tip.hidden = true;
    tipKind = null;
  }

  function place(pos, above) {
    const c = cm.charCoords(pos, "window");
    tip.hidden = false;
    tip.style.maxWidth = Math.min(460, window.innerWidth - 16) + "px";
    const w = tip.offsetWidth;
    const h = tip.offsetHeight;
    const left = Math.max(8, Math.min(c.left, window.innerWidth - w - 8));
    let top = above ? c.top - h - 6 : c.bottom + 6;
    if (top < 8) top = c.bottom + 6;
    if (top + h > window.innerHeight - 8) top = Math.max(8, c.top - h - 6);
    tip.style.left = left + "px";
    tip.style.top = top + "px";
  }

  function infoNodes(info, { more = true } = {}) {
    const nodes = [];
    if (info.variable) {
      const v = info.variable;
      nodes.push(el("div", { class: "tip-var" }, el("code", {}, v.name), ` is a variable: ${v.size} ${v.class}`, v.preview ? el("span", { class: "muted" }, "  " + v.preview) : null));
      if (info.kind === "variable") return nodes;
    }
    if (info.usage && info.usage.length) {
      nodes.push(el("div", { class: "tip-usage" }, info.usage.slice(0, 4).map((u) => el("code", {}, u)), info.usage.length > 4 ? el("span", { class: "muted small" }, "…") : null));
    }
    if (info.text) nodes.push(el("div", { class: "tip-text" }, info.text));
    if (info.path) {
      nodes.push(el("div", { class: "tip-foot muted" }, info.local ? `Defined in this file, line ${info.line + 1}.` : `Defined in ${info.path}.`, ` ${isMac ? "⌘" : "Ctrl"}+click or ${app.commands.keyLabel("go.definition") || "F12"} to go there.`));
    } else if (more && info.kind === "builtin") {
      nodes.push(el("div", { class: "tip-foot muted" }, `Press ${app.commands.keyLabel("help.word") || "F1"} for the full help.`));
    }
    return nodes;
  }

  /** Show the tooltip for a position. what: { problem } and/or { info }. */
  function showTip(pos, { problem = null, info = null, kind = "hover", above = false } = {}) {
    tip.innerHTML = "";
    if (problem) tip.append(el("div", { class: "tip-problem " + problem.severity }, problem.message));
    if (info) tip.append(...infoNodes(info));
    if (!tip.childNodes.length) return hideTip();
    tip.dataset.kind = kind;
    tip.dataset.name = info ? info.name : "";
    tipKind = kind;
    place(pos, above);
    return undefined;
  }

  /* ---------- hover ---------- */

  let hoverTimer = null;
  let hoverKey = "";
  const wrapper = cm.getWrapperElement();

  function hoverAt(x, y) {
    if (document.querySelector("dialog[open]")) return;
    const pos = cm.coordsChar({ left: x, top: y }, "window");
    if (pos.outside) return hideTip("hover");
    const problem = app.problemAt ? app.problemAt(pos) : null;
    const w = setting("hover") && state.active && state.active.endsWith(".m") ? wordAt(pos) : null;
    const info = w && !w.field ? lookup(w.word) : null;
    if (!problem && !info) return hideTip("hover");
    // Only when the mouse is really over the word (not far to the right of a short line).
    const start = cm.charCoords(CodeMirror.Pos(pos.line, w ? w.from : problem.from), "window");
    const end = cm.charCoords(CodeMirror.Pos(pos.line, w ? w.to : problem.to), "window");
    if (x < start.left - 4 || x > end.right + 4) return hideTip("hover");
    if (tipKind === "sig") return undefined;
    return showTip(CodeMirror.Pos(pos.line, w ? w.from : problem.from), { problem, info });
  }

  wrapper.addEventListener("mousemove", (e) => {
    if (e.buttons) return;
    const key = e.clientX + "," + e.clientY;
    if (key === hoverKey) return;
    hoverKey = key;
    clearTimeout(hoverTimer);
    hoverTimer = setTimeout(() => hoverAt(e.clientX, e.clientY), HOVER_DELAY);
  });
  wrapper.addEventListener("mouseleave", () => {
    clearTimeout(hoverTimer);
    hideTip("hover");
  });
  wrapper.addEventListener("mousedown", () => {
    clearTimeout(hoverTimer);
    hideTip();
  });
  cm.on("scroll", () => hideTip());
  cm.on("blur", () => hideTip());
  cm.on("swapDoc", () => hideTip());

  /* ---------- usage hint while typing a call ---------- */

  /** The call the cursor is inside: { name, open } or null. */
  function callAt(cur) {
    const text = cm.getLine(cur.line).slice(0, cur.ch);
    let depth = 0;
    for (let i = text.length - 1; i >= 0; i--) {
      const c = text[i];
      if (c === ")" || c === "]" || c === "}") depth++;
      else if (c === "[" || c === "{") {
        if (depth === 0) return null;
        depth--;
      } else if (c === "(") {
        if (depth > 0) {
          depth--;
          continue;
        }
        const m = /([A-Za-z_]\w*)$/.exec(text.slice(0, i));
        if (!m) return null;
        if (text[i - m[1].length - 1] === ".") return null;
        const type = cm.getTokenTypeAt(CodeMirror.Pos(cur.line, i)) || "";
        if (/comment|string/.test(type)) return null;
        return { name: m[1], open: i, start: i - m[1].length };
      } else if (c === "%") {
        const type = cm.getTokenTypeAt(CodeMirror.Pos(cur.line, i + 1)) || "";
        if (/comment/.test(type)) return null;
      }
    }
    return null;
  }

  let typed = false;
  function updateSignature() {
    const wanted = typed || tipKind === "sig";
    typed = false;
    if (!wanted) return;
    if (!setting("hover") || !state.active || !state.active.endsWith(".m") || cm.somethingSelected() || !cm.hasFocus()) return hideTip("sig");
    const cur = cm.getCursor();
    const call = callAt(cur);
    const info = call ? lookup(call.name) : null;
    if (!info || info.kind === "variable" || info.variable || !info.usage || !info.usage.length) return hideTip("sig");
    // A name that the file assigns to is a variable being indexed, not a call.
    if (info.kind === "builtin" && new RegExp("(^|[^\\w.])" + call.name + "\\s*(\\([^=\\n]*\\))?\\s*=(?!=)", "m").test(cm.getValue())) return hideTip("sig");
    tip.innerHTML = "";
    tip.append(el("div", { class: "tip-usage" }, info.usage.slice(0, 5).map((u) => el("code", {}, u))));
    if (info.text) tip.append(el("div", { class: "tip-text muted" }, info.text));
    tip.dataset.kind = "sig";
    tip.dataset.name = info.name;
    tipKind = "sig";
    place(CodeMirror.Pos(cur.line, call.start), true);
    return undefined;
  }
  let sigTimer = null;
  cm.on("inputRead", () => (typed = true));
  cm.on("cursorActivity", () => {
    if (tipKind === "hover") hideTip("hover");
    clearTimeout(sigTimer);
    sigTimer = setTimeout(updateSignature, 60);
  });
  cm.on("keydown", (editor, e) => {
    if (e.key === "Escape" && !tip.hidden) hideTip();
  });

  /* ================================================================ help (F1) */

  async function helpText(name) {
    const info = lookup(name);
    if (info && info.path) {
      const doc = info.kind === "script" ? helpComments("\n" + app.liveContent(info.path), 0) : helpComments(app.liveContent(info.path), info.line);
      return { title: name, where: `Defined in ${info.path}`, text: `  ${info.usage[0]}\n\n${doc || "This function has no help comments yet. Comment lines right under the function line show up here."}\n`, info };
    }
    const engine = app.engine;
    if (engine && engine.ready && (!helpIndex || info)) {
      try {
        const text = await engine.help(name);
        if (!/ not found\.\s*$/.test(text)) return { title: name, where: "Built into mylab", text, info };
      } catch {
        /* fall through */
      }
    }
    if (info && info.kind === "builtin") return { title: name, where: "Built into mylab", text: info.usage.map((u) => "  " + u).join("\n") + "\n\n  " + info.text + "\n", info };
    return null;
  }

  /** How many single-letter edits turn a into b (stops counting above `max`). */
  function editDistance(a, b, max) {
    if (Math.abs(a.length - b.length) > max) return max + 1;
    let prev = Array.from({ length: b.length + 1 }, (_, i) => i);
    for (let i = 1; i <= a.length; i++) {
      const row = [i];
      for (let j = 1; j <= b.length; j++) row[j] = Math.min(prev[j] + 1, row[j - 1] + 1, prev[j - 1] + (a[i - 1] === b[j - 1] ? 0 : 1));
      prev = row;
    }
    return prev[b.length];
  }

  /** Names that look like a misspelled or half-remembered name. */
  function suggestions(name) {
    const names = new Set(helpIndex ? Object.keys(helpIndex) : []);
    for (const p of state.files.keys()) if (p.endsWith(".m")) names.add(app.baseName(p).slice(0, -2));
    const lower = name.toLowerCase();
    const scored = [];
    for (const n of names) {
      const d = editDistance(lower, n.toLowerCase(), 2);
      if (d <= 2) scored.push([d, n]);
      else if (lower.length >= 3 && n.toLowerCase().startsWith(lower)) scored.push([3, n]);
    }
    return scored.sort((a, b) => a[0] - b[0] || a[1].length - b[1].length).slice(0, 8).map((s) => s[1]);
  }

  async function showHelp(name) {
    const dlg = $("dlg-help");
    const body = $("help-body");
    const also = $("help-also");
    $("help-name").value = name || "";
    also.innerHTML = "";
    if (!name) {
      $("help-where").textContent = "";
      body.textContent = "Type the name of a function, for example plot, zeros or fprintf, and press Enter.";
    } else {
      const res = await helpText(name);
      if (res) {
        $("help-where").textContent = res.where;
        body.textContent = res.text.replace(/\s+$/, "");
      } else {
        $("help-where").textContent = "";
        body.textContent = app.engine && app.engine.ready ? `No help was found for “${name}”.` : "Help for built-in functions is ready once the engine has loaded.";
        const close = suggestions(name);
        if (close.length) {
          also.append("Did you mean: ");
          close.forEach((n, i) => {
            if (i) also.append(", ");
            also.append(el("button", { type: "button", class: "linklike", onclick: () => showHelp(n) }, n));
          });
        }
      }
    }
    dlg.dataset.name = name || "";
    if (!dlg.open) dlg.showModal();
    if (!name) $("help-name").focus();
  }

  $("help-form").addEventListener("submit", (e) => {
    e.preventDefault();
    showHelp($("help-name").value.trim());
  });

  function helpForWord() {
    const w = state.active && cm.hasFocus() ? wordAt(cm.getCursor(), { anywhere: true }) : null;
    showHelp(w ? w.word : "");
  }

  /* ================================================================ autocomplete */

  // RANK orders the list (names from your own code first); INFO picks the richer description
  // when a name is known in two ways (a built-in that the file already uses, for example).
  const RANK = { variable: 0, function: 1, script: 1, word: 2, keyword: 3, builtin: 4 };
  const INFO = { variable: 5, function: 4, script: 4, builtin: 3, keyword: 2, word: 1 };
  const KIND_LABEL = { variable: "variable", function: "function", script: "script", word: "", builtin: "", keyword: "keyword" };

  function renderHint(elt, data, item) {
    elt.append(el("span", { class: "hint-kind " + item.kind, "aria-hidden": "true" }, item.kind === "variable" ? "x" : item.kind === "keyword" ? "k" : item.kind === "word" ? "·" : "ƒ"));
    elt.append(el("span", { class: "hint-name" }, item.text));
    if (item.detail) elt.append(el("span", { class: "hint-detail" }, item.detail));
  }

  function completions(prefix) {
    const items = new Map();
    const lower = prefix.toLowerCase();
    const add = (name, kind, detail) => {
      if (name === prefix || !name.toLowerCase().startsWith(lower)) return;
      const old = items.get(name);
      const item = { text: name, kind, detail: detail || KIND_LABEL[kind], render: renderHint, exact: name.startsWith(prefix), rank: RANK[kind] };
      if (!old) return void items.set(name, item);
      old.rank = Math.min(old.rank, item.rank);
      if (INFO[kind] > INFO[old.kind]) Object.assign(old, { kind, detail: item.detail });
    };
    for (const v of state.variables) add(v.name, "variable", `${v.size} ${v.class}`);
    if (state.active && state.active.endsWith(".m")) for (const f of functionsIn(state.active)) add(f.name, "function", f.doc || f.signature);
    for (const p of state.files.keys()) {
      if (!p.endsWith(".m") || p === state.active) continue;
      const name = app.baseName(p).slice(0, -2);
      if (!name.toLowerCase().startsWith(lower) || !IDENT.test(name)) continue;
      const info = lookup(name);
      add(name, info && info.kind === "script" ? "script" : "function", info && info.kind !== "script" ? info.text || info.usage[0] : p);
    }
    const re = /[A-Za-z_]\w*/g;
    const text = cm.getValue();
    let m;
    while ((m = re.exec(text))) if (m[0].length > 1 && !/^\d/.test(m[0])) add(m[0], "word", "");
    if (helpIndex) {
      for (const name of Object.keys(helpIndex)) {
        if (name.length >= prefix.length && name.toLowerCase().startsWith(lower)) add(name, "builtin", helpIndex[name][1]);
      }
    }
    for (const k of KEYWORDS) add(k, "keyword");
    return [...items.values()].sort((a, b) => b.exact - a.exact || a.rank - b.rank || a.text.length - b.text.length || a.text.localeCompare(b.text));
  }

  function hint(editor) {
    const cur = editor.getCursor();
    const line = editor.getLine(cur.line);
    let start = cur.ch;
    while (start > 0 && /\w/.test(line.charAt(start - 1))) start--;
    const prefix = line.slice(start, cur.ch);
    if (!prefix || /^\d/.test(prefix) || line[start - 1] === ".") return null;
    const type = editor.getTokenTypeAt(CodeMirror.Pos(cur.line, start + 1)) || "";
    if (/comment|string/.test(type)) return null;
    const from = CodeMirror.Pos(cur.line, start);
    const to = CodeMirror.Pos(cur.line, cur.ch);
    const finish = (extra) => {
      const list = completions(prefix);
      const have = new Set(list.map((i) => i.text));
      for (const n of extra) if (!have.has(n) && n !== prefix) list.push({ text: n, kind: "builtin", detail: "", render: renderHint });
      return list.length ? { list: list.slice(0, 80), from, to } : null;
    };
    // Until the help index has arrived, ask the engine for names.
    const engine = app.engine;
    if (!helpIndex && engine && engine.ready && !engine.busy) return engine.complete(prefix).then(finish, () => finish([]));
    return finish([]);
  }

  /**
   * The suggestion list of CodeMirror briefly swaps its keys while it refreshes after a key
   * press: Enter then means "take the only suggestion left". For suggestions that pop up by
   * themselves that would turn "end" + Enter into "endsWith". Here Enter always makes a new
   * line in that case, and Tab takes the first suggestion.
   */
  function tameHintKeys() {
    const c = cm.state.completionActive;
    const proto = c && c.widget && Object.getPrototypeOf(c.widget);
    if (!proto || proto.mylabPatched) return;
    const disable = proto.disable;
    proto.disable = function () {
      disable.call(this);
      const completion = this.completion;
      const editor = completion.cm;
      const keys = {
        Tab: () => {
          if (completion.debounce) cancelAnimationFrame(completion.debounce);
          completion.debounce = 0;
          completion.update();
          if (completion.widget) completion.widget.pick();
        },
      };
      if (completion.options.popsUp) {
        keys.Enter = () => {
          completion.close();
          app.newline ? app.newline(editor) : editor.execCommand("newlineAndIndent");
        };
      } else keys.Enter = this.keyMap.Enter;
      editor.removeKeyMap(this.keyMap);
      this.keyMap = keys;
      editor.addKeyMap(this.keyMap);
    };
    proto.mylabPatched = true;
  }

  function suggest(auto) {
    if (!state.active || cm.getOption("readOnly")) return;
    setTimeout(tameHintKeys, 0);
    cm.showHint({
      hint,
      popsUp: !!auto,
      completeSingle: !auto,
      closeCharacters: /[\s()[\]{};:>,+\-*/^'"=<~&|%]/,
      extraKeys: auto
        ? {
            // While suggestions pop up by themselves, Enter still makes a new line. Tab accepts.
            Enter: (editor, handle) => {
              handle.close();
              app.newline ? app.newline(editor) : editor.execCommand("newlineAndIndent");
            },
          }
        : undefined,
    });
    tameHintKeys();
  }

  cm.on("inputRead", (editor, change) => {
    if (!setting("suggest") || editor.state.completionActive || change.origin !== "+input") return;
    if (!state.active || !state.active.endsWith(".m") || editor.listSelections().length > 1) return;
    const ch = change.text[change.text.length - 1];
    if (change.text.length !== 1 || !/^\w$/.test(ch)) return;
    const cur = editor.getCursor();
    const before = editor.getLine(cur.line).slice(0, cur.ch);
    const m = /[A-Za-z_]\w*$/.exec(before);
    if (!m || m[0].length < 2) return;
    suggest(true);
  });

  /* ================================================================ go to definition */

  function assignmentLine(name) {
    const lines = cm.getValue().split("\n");
    const assign = new RegExp("(^|[^\\w.])" + name + "\\s*(\\([^=]*\\)|\\{[^=]*\\}|\\.[\\w.]+)?\\s*=(?!=)");
    const loop = new RegExp("^\\s*(par)?for\\s*\\(?\\s*" + name + "\\s*=");
    const multi = new RegExp("^\\s*\\[[^\\]]*\\b" + name + "\\b[^\\]]*\\]\\s*=(?!=)");
    const param = new RegExp("^\\s*function\\b[^%]*[(,=\\[\\s]" + name + "\\s*[,)\\]=]");
    for (let i = 0; i < lines.length; i++) {
      const code = lines[i].replace(/%.*$/, "");
      if (assign.test(code) || loop.test(code) || multi.test(code) || param.test(code)) return { line: i, col: Math.max(0, code.search(new RegExp("\\b" + name + "\\b"))) };
    }
    return null;
  }

  /** Where a name is defined: { path, line, col, what } or null. */
  function definitionOf(name) {
    const info = lookup(name);
    if (info && info.path) return { path: info.path, line: info.line, col: Math.max(0, info.col || 0), what: info.kind };
    if (state.active && state.active.endsWith(".m")) {
      const a = assignmentLine(name);
      if (a) return { path: state.active, line: a.line, col: a.col, what: "variable" };
    }
    return null;
  }

  function gotoDefinition(pos = cm.getCursor()) {
    const w = state.active ? wordAt(pos) : null;
    if (!w) {
      toast("Put the cursor on a name first.");
      return false;
    }
    const def = definitionOf(w.word);
    if (!def) {
      const info = lookup(w.word);
      toast(info && info.kind === "builtin" ? `${w.word} is built into mylab. Press ${app.commands.keyLabel("help.word") || "F1"} for its help.` : `No definition was found for ${w.word}.`);
      return false;
    }
    if (def.path === state.active && def.line === pos.line) {
      toast(def.what === "variable" ? `This is where ${w.word} is first set.` : `This is where ${w.word} is defined.`);
      return true;
    }
    app.openFile(def.path);
    app.gotoLine(def.line, def.col);
    return true;
  }

  // Ctrl/Cmd+click on a function goes to its definition.
  wrapper.addEventListener(
    "mousedown",
    (e) => {
      if (e.button !== 0 || !(isMac ? e.metaKey : e.ctrlKey) || e.altKey || e.shiftKey) return;
      const pos = cm.coordsChar({ left: e.clientX, top: e.clientY }, "window");
      if (pos.outside || !state.active) return;
      const w = wordAt(pos);
      const def = w ? definitionOf(w.word) : null;
      if (!def || def.what === "variable") return;
      e.preventDefault();
      e.stopPropagation();
      cm.setCursor(pos);
      gotoDefinition(pos);
    },
    true
  );

  /* ================================================================ outline */

  /** Functions and %% sections of the active file: [{ kind, name, line, detail }]. */
  function outline() {
    if (!state.active || !state.active.endsWith(".m")) return [];
    const text = cm.getValue();
    const out = [];
    const split = splitSections(text);
    if (split.hasSections) {
      split.sections.forEach((s, i) => {
        if (s.header) out.push({ kind: "section", name: s.title || `Section ${i + 1}`, line: s.start });
      });
    }
    for (const f of findFunctions(text)) out.push({ kind: "function", name: f.name, line: f.line, col: f.col, detail: f.signature });
    return out.sort((a, b) => a.line - b.line);
  }

  function symbolItems(query) {
    const list = outline();
    if (!list.length) return [{ label: state.active ? "This file has no functions or %% sections." : "Open a file first.", disabled: true }];
    const out = [];
    for (const s of list) {
      const label = s.kind === "section" ? `%% ${s.name}` : s.detail || s.name;
      const m = fuzzy(query, label);
      if (!m) continue;
      out.push({ label, hits: m.hits, score: m.score, detail: `${s.kind} · line ${s.line + 1}`, run: () => app.gotoLine(s.line, s.col || 0) });
    }
    if (query) out.sort((a, b) => b.score - a.score);
    return out;
  }

  /* ================================================================ rename */

  /** Every use of a name in the active file, outside comments and text in quotes. */
  function occurrences(name) {
    const out = [];
    const doc = cm.getDoc();
    doc.eachLine((handle) => {
      if (!handle.text.includes(name)) return;
      const line = doc.getLineNumber(handle);
      for (const t of cm.getLineTokens(line, true)) {
        if (t.string !== name || /comment|string/.test(t.type || "")) continue;
        if (handle.text[t.start - 1] === ".") continue; // a field of a struct, not this name
        out.push({ line, from: t.start, to: t.end });
      }
    });
    return out;
  }

  function renameIn(name, next) {
    const places = occurrences(name);
    cm.operation(() => {
      for (let i = places.length - 1; i >= 0; i--) {
        const p = places[i];
        cm.replaceRange(next, CodeMirror.Pos(p.line, p.from), CodeMirror.Pos(p.line, p.to), "+rename");
      }
    });
    return places.length;
  }

  async function renameSymbol() {
    if (!state.active || cm.getOption("readOnly")) return;
    const w = wordAt(cm.getCursor());
    if (!w || KEYWORDS.includes(w.word) || w.field) {
      toast("Put the cursor on the name of a variable or function first.");
      return;
    }
    const count = occurrences(w.word).length;
    const next = await promptDialog({
      title: `Rename ${w.word}`,
      text: `Changes ${count} place${count === 1 ? "" : "s"} in this file. Comments and text in quotes are left alone.`,
      value: w.word,
      okLabel: "Rename",
      selectAll: true,
      validate: (v) => {
        const n = v.trim();
        if (!IDENT.test(n)) return { error: "A name starts with a letter and uses only letters, digits and _." };
        if (KEYWORDS.includes(n)) return { error: `${n} is a word of the language, so it can't be a name.` };
        if (n !== w.word && occurrences(n).length) return { name: n, warn: `${n} is already used in this file. The two names would become one.` };
        return { name: n };
      },
    });
    if (!next || next === w.word) return;
    const n = renameIn(w.word, next);
    cm.focus();
    const isMain = app.baseName(state.active) === w.word + ".m" && functionsIn(state.active).some((f) => f.name === next);
    toast(`Renamed ${n} place${n === 1 ? "" : "s"}.` + (isMain ? ` Also rename the file to ${next}.m so it can still be called.` : " Undo with Ctrl+Z."));
  }

  /* ================================================================ wiring */

  function loadHelpIndex() {
    const engine = app.engine;
    if (!engine || !engine.ready) return;
    engine.helpIndex().then(
      (idx) => {
        helpIndex = idx;
        const dl = $("help-names");
        if (dl && !dl.childElementCount) dl.append(...Object.keys(idx).sort().map((n) => el("option", { value: n })));
        app.emit("helpIndex", idx);
      },
      () => {}
    );
  }
  app.on("engineReady", loadHelpIndex);

  app.commands.register([
    { id: "help.word", group: "Help", title: "Help for the function at the cursor", keys: "F1", run: helpForWord, hint: "Open the full help for the word at the cursor" },
    { id: "help.search", group: "Help", title: "Look up a function…", run: () => showHelp("") },
    { id: "go.definition", group: "Go", title: "Go to definition", keys: "F12", matlab: "Mod-F12", scope: "editor", run: () => gotoDefinition(), hint: "Go to where the function or variable at the cursor is defined" },
    { id: "edit.rename", group: "Code", title: "Rename a variable or function in this file", keys: "F2", scope: "editor", run: renameSymbol },
    { id: "edit.suggest", group: "Edit", title: "Complete the name at the cursor", keys: "Ctrl-Space", scope: "editor", run: () => suggest(false) },
  ]);

  Object.assign(app, { lookupName: lookup, showTip, hideTip, showHelp, gotoDefinition, definitionOf, outline, symbolItems, renameSymbol, renameIn, occurrences, editorHint: hint, hoverAt, helpIndex: () => helpIndex });
}
