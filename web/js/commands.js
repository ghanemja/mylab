// Commands: every action of the app in one list, with its keyboard shortcut.
// The command palette, the quick file switcher, the shortcut cheat sheet and the key
// handler are all built from this list, so they can never disagree.
//
// A command is { id, title, group, keys, matlab, scope, when, run, hidden }.
//   keys    "Mod-Shift-P" or a list of such strings. Mod is Ctrl (Cmd on a Mac).
//   matlab  other keys used by the "MATLAB-style keys" preset (leave out to keep `keys`).
//   scope   "global" (default), "editor" (only while typing in the editor) or
//           "noField" (anywhere except the command window and assistant inputs).
//   when    optional () => bool; the key is ignored while it returns false.

import { $, el, isMac, toast } from "./ui.js";

const CODE_KEYS = {
  Slash: "/", BracketLeft: "[", BracketRight: "]", Comma: ",", Period: ".", Backquote: "`", Equal: "=", Backslash: "\\", Semicolon: ";", Space: "Space",
};

/** "Ctrl-G" means the Ctrl key everywhere; off the Mac that is the same key as Mod. */
function normalize(spec) {
  const parts = spec.split("-");
  const key = parts.pop() || "-";
  const mods = new Set(parts.map((p) => (p === "Ctrl" && !isMac ? "Mod" : p === "Cmd" ? "Mod" : p)));
  return ["Mod", "Ctrl", "Alt", "Shift"].filter((m) => mods.has(m)).concat(key.length === 1 ? key.toUpperCase() : key).join("-");
}

/** The ways a key press can be written, most exact first. */
function eventKeys(e) {
  const mods = [];
  if (isMac ? e.metaKey : e.ctrlKey) mods.push("Mod");
  if (isMac && e.ctrlKey) mods.push("Ctrl");
  if (e.altKey) mods.push("Alt");
  const withShift = e.shiftKey ? [...mods, "Shift"] : mods;
  const out = [];
  const add = (m, k) => {
    const s = [...m, k].join("-");
    if (!out.includes(s)) out.push(s);
  };
  const k = e.key || "";
  const code = e.code || "";
  if (/^[a-z]$/i.test(k)) add(withShift, k.toUpperCase());
  else if (k.length > 1 && k !== "Dead" && k !== "Unidentified" && k !== "Process") add(withShift, k);
  if (/^Key[A-Z]$/.test(code)) add(withShift, code.slice(3));
  if (/^Digit\d$/.test(code)) add(withShift, code.slice(5));
  if (CODE_KEYS[code]) add(withShift, CODE_KEYS[code]);
  if (k.length === 1 && !/[a-z0-9 ]/i.test(k)) {
    add(withShift, k);
    add(mods, k); // the character may need Shift on this keyboard layout
  }
  if (k === " ") add(withShift, "Space");
  return out;
}

const MAC_SYMBOL = { Mod: "⌘", Ctrl: "⌃", Alt: "⌥", Shift: "⇧", Enter: "↩" };
const KEY_LABEL = { ArrowUp: "↑", ArrowDown: "↓", ArrowLeft: "←", ArrowRight: "→", PageUp: "PgUp", PageDown: "PgDn", Escape: "Esc" };

/** "Mod-Shift-P" -> ["Ctrl", "Shift", "P"] (or the Mac symbols). */
export function keyParts(spec) {
  const parts = normalize(spec).split(/-(?=.)/);
  const key = parts.pop();
  // The usual order for showing keys: Ctrl+Shift+Alt+key (on a Mac: ⌃⌥⇧⌘key).
  const order = isMac ? ["Ctrl", "Alt", "Shift", "Mod"] : ["Mod", "Ctrl", "Shift", "Alt"];
  return order
    .filter((m) => parts.includes(m))
    .concat(key)
    .map((p) => (isMac && MAC_SYMBOL[p]) || (p === "Mod" ? "Ctrl" : KEY_LABEL[p] || p));
}

export function keyText(spec) {
  return keyParts(spec).join(isMac ? "" : "+");
}

/* ------------------------------------------------------------------ fuzzy search */

/**
 * Match `query` against `text` as a subsequence. Returns null, or { score, hits } where hits
 * are the matched positions. Matches at word starts and runs of letters score higher.
 */
export function fuzzy(query, text) {
  const q = query.toLowerCase().replace(/\s+/g, "");
  if (!q) return { score: 0, hits: [] };
  const t = text.toLowerCase();
  const at = t.indexOf(q);
  if (at >= 0) {
    const hits = [];
    for (let i = 0; i < q.length; i++) hits.push(at + i);
    const wordStart = at === 0 || /[\s/_.:-]/.test(t[at - 1]);
    return { score: 1000 - at + (wordStart ? 200 : 0) - (t.length - q.length) * 0.5, hits };
  }
  const hits = [];
  let score = 0;
  let ti = 0;
  let last = -2;
  for (let qi = 0; qi < q.length; qi++) {
    // Prefer the next word start that matches; otherwise the next occurrence.
    let found = -1;
    for (let k = ti; k < t.length; k++) {
      if (t[k] !== q[qi]) continue;
      const start = k === 0 || /[\s/_.:-]/.test(t[k - 1]) || (text[k] !== t[k] && text[k - 1] === t[k - 1]);
      if (start || k === last + 1) {
        found = k;
        score += start ? 30 : 20;
        break;
      }
      if (found < 0) found = k;
    }
    if (found < 0) return null;
    if (found === last + 1) score += 10;
    score += 5 - Math.min(5, (found - ti) * 0.2);
    hits.push(found);
    last = found;
    ti = found + 1;
  }
  return { score: score - t.length * 0.2, hits };
}

function highlight(text, hits) {
  const span = el("span", { class: "pal-label" });
  if (!hits || !hits.length) {
    span.textContent = text;
    return span;
  }
  const set = new Set(hits);
  let run = "";
  let marked = false;
  const flush = () => {
    if (!run) return;
    span.append(marked ? el("b", {}, run) : run);
    run = "";
  };
  for (let i = 0; i < text.length; i++) {
    if (set.has(i) !== marked) {
      flush();
      marked = set.has(i);
    }
    run += text[i];
  }
  flush();
  return span;
}

/* ------------------------------------------------------------------ registry */

export const GROUPS = ["Run", "File", "Edit", "Find", "Go", "Code", "View", "Assistant", "Project", "Help"];

export function initCommands(app) {
  const { prefs } = app;
  const commands = new Map();
  let bindings = new Map(); // normalized key -> [command]
  let preset = prefs.get("keyPreset", "default") === "matlab" ? "matlab" : "default";

  const keysOf = (cmd) => {
    const k = preset === "matlab" && cmd.matlab !== undefined ? cmd.matlab : cmd.keys;
    return k ? [].concat(k) : [];
  };

  function rebuild() {
    bindings = new Map();
    for (const cmd of commands.values()) {
      for (const k of keysOf(cmd)) {
        const n = normalize(k);
        if (!bindings.has(n)) bindings.set(n, []);
        bindings.get(n).push(cmd);
      }
    }
  }

  function register(list) {
    for (const cmd of list) commands.set(cmd.id, { scope: "global", group: "View", ...cmd });
    rebuild();
  }

  function available(cmd) {
    try {
      return !cmd.when || !!cmd.when();
    } catch {
      return false;
    }
  }

  function run(id, ...args) {
    const cmd = commands.get(id);
    if (!cmd) return undefined;
    try {
      return cmd.run(...args);
    } catch (e) {
      console.error(e);
      toast(e.message || String(e), "error");
      return undefined;
    }
  }

  /** The first shortcut of a command as text ("Ctrl+Shift+P"), or "". */
  function keyLabel(id) {
    const cmd = commands.get(id);
    const k = cmd ? keysOf(cmd) : [];
    return k.length ? keyText(k[0]) : "";
  }

  function setPreset(name) {
    preset = name === "matlab" ? "matlab" : "default";
    prefs.set("keyPreset", preset);
    rebuild();
    app.emit("keysChanged", preset);
  }

  function inField() {
    const a = document.activeElement;
    if (!a) return false;
    if (a.closest && a.closest(".CodeMirror")) return false;
    return a.tagName === "INPUT" || a.tagName === "TEXTAREA" || a.tagName === "SELECT" || a.isContentEditable;
  }

  document.addEventListener(
    "keydown",
    (e) => {
      if (e.isComposing || e.key === "Shift" || e.key === "Control" || e.key === "Alt" || e.key === "Meta") return;
      const dialog = document.querySelector("dialog[open]");
      const editorFocus = !!app.editor && app.editor.hasFocus();
      for (const key of eventKeys(e)) {
        const list = bindings.get(key);
        if (!list) continue;
        for (const cmd of list) {
          if (dialog && !(cmd.inDialog && dialog.id === cmd.inDialog)) continue;
          if (cmd.scope === "editor" && !editorFocus) continue;
          if (cmd.scope === "noField" && inField()) continue;
          if (!available(cmd)) continue;
          e.preventDefault();
          e.stopPropagation();
          run(cmd.id);
          return;
        }
      }
    },
    true
  );

  /* ---------------------------------------------------------------- palette */

  const dlg = $("dlg-palette");
  const input = $("palette-input");
  const listEl = $("palette-list");
  let items = [];
  let active = 0;
  let source = null; // (query) => [{ label, detail, hint, hits, run, icon }]
  const MODES = {
    ">": { placeholder: "Type the name of a command", label: "Commands" },
    "": { placeholder: "Type a file name. Start with > for commands, @ for the outline, : for a line number", label: "Files" },
    "@": { placeholder: "Type the name of a function or section", label: "Outline" },
    ":": { placeholder: "Type a line number and press Enter", label: "Go to line" },
  };

  function commandItems(query) {
    const out = [];
    for (const cmd of commands.values()) {
      if (cmd.hidden || !available(cmd)) continue;
      const label = `${cmd.group}: ${cmd.title}`;
      const m = fuzzy(query, label);
      if (!m) continue;
      out.push({ label, hits: m.hits, score: m.score + (cmd.group === "Run" && !query ? 1 : 0), keys: keysOf(cmd)[0], hint: cmd.hint, run: () => run(cmd.id), id: cmd.id });
    }
    if (query) out.sort((a, b) => b.score - a.score);
    else out.sort((a, b) => GROUPS.indexOf(a.label.split(":")[0]) - GROUPS.indexOf(b.label.split(":")[0]));
    return out;
  }

  function fileItems(query) {
    const { state } = app;
    const recent = [...state.tabs].reverse();
    const out = [];
    for (const path of app.sortedPaths()) {
      const m = fuzzy(query, path);
      if (!m) continue;
      const base = app.baseName(path);
      const mb = query ? fuzzy(query, base) : null;
      const r = recent.indexOf(path);
      out.push({
        label: path,
        hits: m.hits,
        score: m.score + (mb ? mb.score * 0.5 + 50 : 0) + (r >= 0 ? 20 - r : 0) + (path === state.active ? -5 : 0),
        detail: path === state.active ? "open now" : r >= 0 ? "open tab" : "",
        icon: app.isText(path) ? "file" : "data",
        run: () => app.openFile(path),
      });
    }
    out.sort((a, b) => b.score - a.score || a.label.localeCompare(b.label));
    return out;
  }

  function lineItems(query) {
    const cm = app.editor;
    const n = parseInt(query, 10);
    const count = app.state.active ? cm.lineCount() : 0;
    if (!count) return [{ label: "Open a file first.", disabled: true }];
    if (!Number.isFinite(n)) return [{ label: `Type a line number from 1 to ${count}.`, disabled: true }];
    const line = Math.max(1, Math.min(count, n));
    return [{ label: `Go to line ${line}`, run: () => app.gotoLine(line - 1, 0) }];
  }

  function render() {
    const raw = input.value;
    const mode = source ? "custom" : raw[0] === ">" || raw[0] === "@" || raw[0] === ":" ? raw[0] : "";
    const query = (mode && mode !== "custom" ? raw.slice(1) : raw).trim();
    if (source) items = source(query);
    else if (mode === ">") items = commandItems(query);
    else if (mode === "@") items = app.symbolItems ? app.symbolItems(query) : [];
    else if (mode === ":") items = lineItems(query);
    else items = fileItems(query);
    if (!source) input.placeholder = MODES[mode].placeholder;
    items = items.slice(0, 200);
    active = Math.max(
      0,
      items.findIndex((it) => !it.disabled)
    );
    listEl.innerHTML = "";
    if (!items.length) listEl.append(el("li", { class: "pal-empty muted", role: "presentation" }, "Nothing matches."));
    items.forEach((it, i) => {
      const li = el(
        "li",
        {
          id: "pal-item-" + i,
          class: "pal-item" + (it.disabled ? " disabled" : ""),
          role: "option",
          "aria-selected": "false",
          "aria-disabled": it.disabled ? "true" : null,
          "data-id": it.id || null,
          title: it.hint || null,
          onmousemove: () => i !== active && !it.disabled && setActive(i, false),
          onclick: () => choose(i),
        },
        highlight(it.label, it.hits),
        it.detail ? el("span", { class: "pal-detail muted" }, it.detail) : null,
        it.keys ? el("span", { class: "pal-keys" }, keyParts(it.keys).map((p) => el("kbd", {}, p))) : null
      );
      listEl.append(li);
    });
    setActive(active, true);
  }

  function setActive(i, scroll) {
    active = i;
    [...listEl.children].forEach((li, k) => {
      li.classList.toggle("active", k === i);
      if (li.getAttribute("role") === "option") li.setAttribute("aria-selected", k === i ? "true" : "false");
    });
    const cur = listEl.children[i];
    if (cur && cur.id) input.setAttribute("aria-activedescendant", cur.id);
    else input.removeAttribute("aria-activedescendant");
    if (scroll && cur) cur.scrollIntoView({ block: "nearest" });
  }

  function move(delta) {
    if (!items.length) return;
    let i = active;
    for (let n = 0; n < items.length; n++) {
      i = (i + delta + items.length) % items.length;
      if (!items[i].disabled) break;
    }
    setActive(i, true);
  }

  function choose(i) {
    const it = items[i];
    if (!it || it.disabled) return;
    dlg.close();
    // Let the dialog give the focus back before the command moves it.
    setTimeout(() => {
      try {
        it.run && it.run();
      } catch (e) {
        console.error(e);
        toast(e.message || String(e), "error");
      }
    }, 0);
  }

  /**
   * Open the palette. mode: ">" commands, "" files, "@" outline, ":" go to line.
   * custom: { placeholder, items(query) } shows another list (for example, themes).
   */
  function openPalette(mode = ">", custom = null) {
    source = custom ? custom.items : null;
    input.value = custom ? "" : mode;
    if (custom) input.placeholder = custom.placeholder || "";
    $("palette-title").textContent = custom ? custom.title || "Choose" : "Command palette";
    if (!dlg.open) dlg.showModal();
    render();
    input.focus();
    input.setSelectionRange(input.value.length, input.value.length);
  }

  input.addEventListener("input", render);
  input.addEventListener("keydown", (e) => {
    if (e.key === "ArrowDown") (e.preventDefault(), move(1));
    else if (e.key === "ArrowUp") (e.preventDefault(), move(-1));
    else if (e.key === "PageDown") (e.preventDefault(), setActive(Math.min(items.length - 1, active + 8), true));
    else if (e.key === "PageUp") (e.preventDefault(), setActive(Math.max(0, active - 8), true));
    else if (e.key === "Enter") (e.preventDefault(), choose(active));
  });
  dlg.addEventListener("close", () => (source = null));

  /* ---------------------------------------------------------------- cheat sheet */

  function renderCheatSheet() {
    const box = $("keys-body");
    box.innerHTML = "";
    const kbds = (spec) => {
      const span = el("span", { class: "key-combo" });
      keyParts(spec).forEach((p, i) => {
        if (i && !isMac) span.append("+");
        span.append(el("kbd", {}, p));
      });
      return span;
    };
    const section = (title, rows) => {
      if (!rows.length) return;
      const table = el("table", { class: "keys" });
      for (const [keys, text] of rows) table.append(el("tr", {}, el("td", {}, keys), el("td", {}, text)));
      box.append(el("h3", {}, title), table);
    };
    for (const group of GROUPS) {
      const rows = [];
      for (const cmd of commands.values()) {
        const ks = keysOf(cmd);
        if (cmd.group !== group || !ks.length || cmd.hideKeys) continue;
        const cell = el("span", {});
        ks.slice(0, 2).forEach((k, i) => {
          if (i) cell.append(" or ");
          cell.append(kbds(k));
        });
        rows.push([cell, cmd.hint || cmd.title]);
      }
      section(group, rows);
    }
    const k = (...parts) => {
      const span = el("span", {});
      parts.forEach((p, i) => {
        if (i) span.append(" / ");
        span.append(el("kbd", {}, p));
      });
      return span;
    };
    section("Step through", [
      [k("←", "→"), "Step back / forward"],
      [k("Home", "End"), "First / last step"],
      [k("Esc"), "Leave step-through"],
      ["Click beside a line number", "Set or remove a breakpoint"],
    ]);
    section("Command window", [
      [k("Enter"), "Run the command"],
      [k("Shift+Enter"), "New line"],
      [k("↑", "↓"), "Previous / next command"],
      [k("Tab"), "Complete a name"],
    ]);
    section("Editor extras", [
      [k("Tab", "Shift+Tab"), "Indent / outdent the selected lines"],
      [k(isMac ? "⌘ click" : "Ctrl+click"), "Go to the definition of a function"],
      [k(isMac ? "⌥ click" : "Alt+click"), "Add another cursor"],
      [k(isMac ? "⌘Z" : "Ctrl+Z", isMac ? "⌘⇧Z" : "Ctrl+Y"), "Undo / redo"],
    ]);
    $("keys-preset").value = preset;
  }

  $("keys-preset").addEventListener("change", (e) => {
    setPreset(e.target.value);
    renderCheatSheet();
  });

  function showCheatSheet() {
    renderCheatSheet();
    if (!$("dlg-keys").open) $("dlg-keys").showModal();
  }

  app.commands = {
    register,
    run,
    keyLabel,
    setPreset,
    get preset() {
      return preset;
    },
    all: () => [...commands.values()],
    get: (id) => commands.get(id),
    openPalette,
    showCheatSheet,
  };

  register([
    { id: "palette.commands", group: "View", title: "Show all commands", keys: ["Mod-Shift-P", "Mod-Alt-P"], run: () => openPalette(">"), hint: "Command palette: every action in one list" },
    { id: "palette.files", group: "Go", title: "Go to file…", keys: "Mod-P", run: () => openPalette(""), hint: "Quick file switcher" },
    { id: "palette.outline", group: "Go", title: "Go to function or section…", keys: "Mod-Shift-O", run: () => openPalette("@"), hint: "Outline of the current file" },
    { id: "palette.line", group: "Go", title: "Go to line…", keys: "Ctrl-G", run: () => openPalette(":") },
    { id: "help.keys", group: "Help", title: "Keyboard shortcuts", run: showCheatSheet },
  ]);
}
