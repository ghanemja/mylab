// mylab web app: wires the editor, files, command window, workspace, plots and assistant together.

import { Engine } from "./engine.js";
import { FileStore, prefs } from "./storage.js";
import { EXAMPLES } from "./examples.js";
import { Assistant } from "./ai.js";

const $ = (id) => document.getElementById(id);
const isMac = /Mac|iPhone|iPad/.test(navigator.platform || "");

/* ================================================================== state */

const state = {
  files: new Map(), // name -> { content, modified }
  docs: new Map(), // name -> CodeMirror.Doc
  tabs: [],
  active: null,
  saveTimers: new Map(),
  lastError: null, // { message, line, where, code }
  variables: [],
  history: prefs.get("history", []),
  historyPos: -1,
  historyDraft: "",
  figureCount: 0,
  errorMark: null,
  runStarted: 0,
  runTimer: null,
};

const store = new FileStore();
let editor;
let engine;
let assistant;

/* ================================================================== helpers */

function toast(msg, kind = "") {
  const t = document.createElement("div");
  t.className = "toast" + (kind ? " " + kind : "");
  t.textContent = msg;
  $("toasts").appendChild(t);
  setTimeout(() => t.remove(), kind === "error" ? 5000 : 2600);
}

function el(tag, attrs = {}, ...children) {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "class") e.className = v;
    else if (k.startsWith("on")) e.addEventListener(k.slice(2), v);
    else if (v !== false && v != null) e.setAttribute(k, v === true ? "" : v);
  }
  for (const c of children) if (c != null) e.append(c);
  return e;
}

function icon(name) {
  const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  svg.setAttribute("class", "ic");
  const use = document.createElementNS("http://www.w3.org/2000/svg", "use");
  use.setAttribute("href", "#i-" + name);
  svg.appendChild(use);
  return svg;
}

function downloadBlob(blob, filename) {
  const url = URL.createObjectURL(blob);
  const a = el("a", { href: url, download: filename });
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

function isFunctionFile(code) {
  for (const raw of code.split("\n")) {
    const line = raw.trim();
    if (!line || line.startsWith("%") || line.startsWith("#")) continue;
    return /^function\b/.test(line);
  }
  return false;
}

const VALID_NAME = /^[A-Za-z][A-Za-z0-9_]*\.m$/;

function validateFileName(name, { current = null } = {}) {
  name = name.trim();
  if (!name) return { error: "Please enter a name." };
  if (!/\.[A-Za-z0-9]+$/.test(name)) name += ".m";
  if (/[\\/:*?"<>|]/.test(name) || name.startsWith(".")) return { error: "That name has characters that can't be used in file names." };
  if (name.length > 80) return { error: "Please use a shorter name." };
  if (name !== current && state.files.has(name)) return { error: `A file named ${name} already exists.` };
  const warn =
    name.endsWith(".m") && !VALID_NAME.test(name)
      ? "Tip: .m names should start with a letter and use only letters, digits and _, so the file can be called by name."
      : "";
  return { name, warn };
}

/* ================================================================== dialogs */

function promptDialog({ title, text = "", value = "", okLabel = "OK", validate }) {
  const dlg = $("dlg-prompt");
  $("prompt-title").textContent = title;
  $("prompt-text").textContent = text;
  $("prompt-text").hidden = !text;
  $("prompt-ok").textContent = okLabel;
  const input = $("prompt-input");
  const err = $("prompt-error");
  input.value = value;
  err.hidden = true;
  return new Promise((resolve) => {
    const onInput = () => {
      const r = validate ? validate(input.value) : { name: input.value };
      err.hidden = !(r.error || r.warn);
      err.textContent = r.error || r.warn || "";
      err.classList.toggle("muted", !r.error);
      $("prompt-ok").disabled = !!r.error;
    };
    const onClose = () => {
      dlg.removeEventListener("close", onClose);
      input.removeEventListener("input", onInput);
      if (dlg.returnValue !== "ok") return resolve(null);
      const r = validate ? validate(input.value) : { name: input.value };
      resolve(r.error ? null : r.name);
    };
    input.addEventListener("input", onInput);
    dlg.addEventListener("close", onClose);
    dlg.returnValue = "";
    $("prompt-ok").disabled = false;
    dlg.showModal();
    onInput();
    err.hidden = true; // don't nag before typing
    const dot = value.lastIndexOf(".");
    input.focus();
    input.setSelectionRange(0, dot > 0 ? dot : value.length);
  });
}

function confirmDialog({ title, text, okLabel = "OK" }) {
  const dlg = $("dlg-confirm");
  $("confirm-title").textContent = title;
  $("confirm-text").textContent = text;
  $("confirm-ok").textContent = okLabel;
  return new Promise((resolve) => {
    const onClose = () => {
      dlg.removeEventListener("close", onClose);
      resolve(dlg.returnValue === "ok");
    };
    dlg.addEventListener("close", onClose);
    dlg.returnValue = "";
    dlg.showModal();
  });
}

/* ================================================================== theme */

function currentTheme() {
  const t = document.documentElement.dataset.theme;
  if (t) return t;
  return matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
}

function updateThemeIcon() {
  const use = $("btn-theme").querySelector("use");
  use.setAttribute("href", currentTheme() === "dark" ? "#i-sun" : "#i-moon");
  $("btn-theme").title = currentTheme() === "dark" ? "Switch to light theme" : "Switch to dark theme";
}

function toggleTheme() {
  const next = currentTheme() === "dark" ? "light" : "dark";
  document.documentElement.dataset.theme = next;
  prefs.set("theme", next);
  updateThemeIcon();
}

/* ================================================================== files */

async function loadFiles() {
  await store.init();
  let list = await store.list();
  const seeded = prefs.get("seeded", false);
  if (!list.length && !seeded) {
    for (const ex of EXAMPLES) await store.put(ex.name, ex.content);
    prefs.set("seeded", true);
    list = await store.list();
    if (!list.length) list = EXAMPLES.map((e) => ({ ...e, modified: Date.now() }));
  }
  for (const f of list) state.files.set(f.name, { content: f.content, modified: f.modified });
  $("storage-note").textContent = store.persistent
    ? "Files are saved in this browser."
    : "This browser blocks storage: files last only until you close the tab. Download them to keep them.";
  if (!store.persistent) $("storage-note").classList.add("error-text");
}

function sortedNames() {
  return [...state.files.keys()].sort((a, b) => a.localeCompare(b, undefined, { sensitivity: "base", numeric: true }));
}

function renderFileList() {
  const ul = $("file-list");
  ul.innerHTML = "";
  for (const name of sortedNames()) {
    const f = state.files.get(name);
    const li = el(
      "li",
      {
        class: "file-item" + (name === state.active ? " active" : ""),
        role: "option",
        "aria-selected": name === state.active ? "true" : "false",
        tabindex: "0",
        title: name,
        "data-name": name,
        onclick: () => openFile(name),
        onkeydown: (ev) => {
          if (ev.key === "Enter") openFile(name);
          if (ev.key === "Delete") deleteFile(name);
          if (ev.key === "F2") renameFile(name);
        },
      },
      icon(isFunctionFile(f.content) ? "fn" : "file"),
      el("span", { class: "name" }, name),
      el(
        "span",
        { class: "actions" },
        el("button", { class: "btn icon sm ghost", title: "Rename", "aria-label": `Rename ${name}`, onclick: (e) => (e.stopPropagation(), renameFile(name)) }, icon("pencil")),
        el("button", { class: "btn icon sm ghost", title: "Download", "aria-label": `Download ${name}`, onclick: (e) => (e.stopPropagation(), exportFile(name)) }, icon("download")),
        el("button", { class: "btn icon sm ghost", title: "Delete", "aria-label": `Delete ${name}`, onclick: (e) => (e.stopPropagation(), deleteFile(name)) }, icon("trash"))
      )
    );
    ul.appendChild(li);
  }
  if (!state.files.size) ul.appendChild(el("li", { class: "empty muted" }, "No files. Create one with +."));
}

function getDoc(name) {
  let doc = state.docs.get(name);
  if (!doc) {
    const mode = name.endsWith(".m") ? "text/x-octave" : "text/plain";
    doc = CodeMirror.Doc(state.files.get(name).content, mode);
    state.docs.set(name, doc);
  }
  return doc;
}

function openFile(name, { focus = true } = {}) {
  if (!state.files.has(name)) return;
  if (!state.tabs.includes(name)) state.tabs.push(name);
  state.active = name;
  clearErrorMark();
  editor.swapDoc(getDoc(name));
  $("editor-empty")?.remove();
  editor.getWrapperElement().hidden = false;
  renderTabs();
  renderFileList();
  saveSession();
  if (focus) editor.focus();
  editor.refresh();
}

function closeTab(name) {
  const i = state.tabs.indexOf(name);
  if (i < 0) return;
  flushSave(name);
  state.tabs.splice(i, 1);
  if (state.active === name) {
    const next = state.tabs[i] || state.tabs[i - 1];
    if (next) openFile(next);
    else showNoFile();
  }
  renderTabs();
  saveSession();
}

function showNoFile() {
  state.active = null;
  editor.getWrapperElement().hidden = true;
  if (!$("editor-empty")) {
    $("editor").appendChild(
      el(
        "div",
        { id: "editor-empty", class: "editor-empty muted" },
        el("div", {}, "No file open. Pick one from the Files panel, or ", el("button", { class: "btn sm", onclick: newFile }, "create a new file"), ".")
      )
    );
  }
  renderFileList();
  renderTabs();
}

function renderTabs() {
  const box = $("tabs");
  box.innerHTML = "";
  for (const name of state.tabs) {
    const tab = el(
      "button",
      {
        class: "tab" + (name === state.active ? " active" : ""),
        role: "tab",
        "aria-selected": name === state.active ? "true" : "false",
        title: name,
        onclick: () => openFile(name),
        onauxclick: (e) => e.button === 1 && closeTab(name),
      },
      el("span", {}, name),
      el(
        "span",
        {
          class: "close",
          role: "button",
          "aria-label": `Close ${name}`,
          title: "Close",
          onclick: (e) => {
            e.stopPropagation();
            closeTab(name);
          },
        },
        icon("close")
      )
    );
    box.appendChild(tab);
  }
  const activeTab = box.querySelector(".tab.active");
  if (activeTab) activeTab.scrollIntoView({ block: "nearest", inline: "nearest" });
}

function saveSession() {
  prefs.set("session", { tabs: state.tabs, active: state.active });
}

function scheduleSave(name) {
  clearTimeout(state.saveTimers.get(name));
  $("save-state").textContent = "Editing…";
  state.saveTimers.set(
    name,
    setTimeout(() => flushSave(name), 600)
  );
}

async function flushSave(name) {
  clearTimeout(state.saveTimers.get(name));
  state.saveTimers.delete(name);
  const doc = state.docs.get(name);
  const f = state.files.get(name);
  if (!doc || !f) return;
  const content = doc.getValue();
  if (content === f.content) {
    if (!state.saveTimers.size) $("save-state").textContent = store.persistent ? "Saved" : "Not saved (no storage)";
    return;
  }
  f.content = content;
  f.modified = Date.now();
  await store.put(name, content);
  if (!state.saveTimers.size) $("save-state").textContent = store.persistent ? "Saved" : "Not saved (no storage)";
}

async function flushAll() {
  await Promise.all([...state.docs.keys()].map((n) => flushSave(n)));
}

async function createFile(name, content, { open = true } = {}) {
  state.files.set(name, { content, modified: Date.now() });
  await store.put(name, content);
  renderFileList();
  if (open) openFile(name);
}

async function newFile() {
  let base = "untitled";
  let n = 1;
  while (state.files.has(`${base}${n > 1 ? n : ""}.m`)) n++;
  const suggestion = `${base}${n > 1 ? n : ""}.m`;
  const name = await promptDialog({
    title: "New file",
    text: "Scripts and function files end in .m. A function file should have the same name as its function.",
    value: suggestion,
    okLabel: "Create",
    validate: (v) => validateFileName(v),
  });
  if (!name) return;
  const fnName = name.replace(/\.m$/, "");
  const template = /^[A-Za-z]\w*$/.test(fnName) ? `% ${fnName}.m\n\n` : "";
  await createFile(name, template);
  editor.setCursor(editor.lineCount(), 0);
}

async function renameFile(name) {
  const newName = await promptDialog({
    title: "Rename file",
    value: name,
    okLabel: "Rename",
    validate: (v) => validateFileName(v, { current: name }),
  });
  if (!newName || newName === name) return;
  await flushSave(name);
  const f = state.files.get(name);
  state.files.delete(name);
  state.files.set(newName, f);
  const doc = state.docs.get(name);
  state.docs.delete(name);
  if (doc) state.docs.set(newName, doc);
  state.tabs = state.tabs.map((t) => (t === name ? newName : t));
  if (state.active === name) state.active = newName;
  await store.put(newName, f.content);
  await store.remove(name);
  renderTabs();
  renderFileList();
  saveSession();
  toast(`Renamed to ${newName}`);
}

async function deleteFile(name) {
  const ok = await confirmDialog({
    title: `Delete ${name}?`,
    text: "The file will be removed from this browser. This can't be undone. You can download it first if you want a copy.",
    okLabel: "Delete",
  });
  if (!ok) return;
  clearTimeout(state.saveTimers.get(name));
  state.saveTimers.delete(name);
  state.files.delete(name);
  state.docs.delete(name);
  await store.remove(name);
  if (state.tabs.includes(name)) {
    const wasActive = state.active === name;
    state.tabs = state.tabs.filter((t) => t !== name);
    if (wasActive) {
      if (state.tabs.length) openFile(state.tabs[state.tabs.length - 1]);
      else showNoFile();
    }
  }
  renderTabs();
  renderFileList();
  saveSession();
}

async function exportFile(name) {
  await flushSave(name);
  const f = state.files.get(name);
  downloadBlob(new Blob([f.content], { type: "text/plain;charset=utf-8" }), name);
}

async function exportZip() {
  await flushAll();
  if (typeof JSZip === "undefined") {
    toast("The zip library hasn't loaded yet. Check your connection and try again.", "error");
    return;
  }
  const zip = new JSZip();
  for (const name of sortedNames()) zip.file(name, state.files.get(name).content);
  const blob = await zip.generateAsync({ type: "blob" });
  const d = new Date();
  const stamp = `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, "0")}-${String(d.getDate()).padStart(2, "0")}`;
  downloadBlob(blob, `mylab-files-${stamp}.zip`);
}

const IMPORT_EXT = /\.(m|txt|csv|dat|json)$/i;

async function importFiles(fileList) {
  let count = 0;
  let last = null;
  for (const file of fileList) {
    if (!IMPORT_EXT.test(file.name)) {
      toast(`Skipped ${file.name}: only .m, .txt, .csv, .dat and .json files can be imported.`, "error");
      continue;
    }
    if (file.size > 2 * 1024 * 1024) {
      toast(`Skipped ${file.name}: files over 2 MB are not supported.`, "error");
      continue;
    }
    const text = await file.text();
    let name = file.name.replace(/[\\/:*?"<>|]/g, "_");
    if (state.files.has(name)) {
      const ok = await confirmDialog({ title: `Replace ${name}?`, text: `A file named ${name} already exists. Replace it with the imported one?`, okLabel: "Replace" });
      if (!ok) continue;
      const doc = state.docs.get(name);
      if (doc) doc.setValue(text);
    }
    await createFile(name, text, { open: false });
    last = name;
    count++;
  }
  if (last) openFile(last);
  if (count) toast(`Imported ${count} file${count > 1 ? "s" : ""}.`);
}

function allFilesForEngine() {
  // Use the live editor contents, even if they haven't been saved yet.
  return [...state.files.entries()].map(([name, f]) => ({
    name,
    content: state.docs.has(name) ? state.docs.get(name).getValue() : f.content,
  }));
}

/* ================================================================== command window */

const consoleOut = () => $("console-out");
const MAX_ENTRIES = 1500;

function appendConsole(node) {
  const out = consoleOut();
  const nearBottom = out.scrollHeight - out.scrollTop - out.clientHeight < 60;
  out.appendChild(node);
  while (out.childElementCount > MAX_ENTRIES) out.firstElementChild.remove();
  if (nearBottom || node.classList.contains("echo")) out.scrollTop = out.scrollHeight;
}

function printEcho(text, label) {
  const div = el("div", { class: "entry echo" });
  if (label) div.append(el("b", {}, "▶ "), label);
  else div.append(el("b", {}, ">> "), text);
  appendConsole(div);
}

function printOut(text) {
  if (!text) return;
  appendConsole(el("div", { class: "entry out" }, text.replace(/\n{3,}$/, "\n\n")));
}

function printInfo(text, cls = "info") {
  appendConsole(el("div", { class: "entry " + cls }, text));
}

function printError(err) {
  const where = err.where && err.line ? `${err.where}, line ${err.line}` : err.line ? `line ${err.line}` : err.where || "";
  const box = el("div", { class: "entry err", role: "alert" });
  box.append(el("strong", {}, "Error" + (where ? ` (${where})` : "") + ": "), err.message);
  const actions = el("div", { class: "err-actions" });
  if (err.file && err.line && state.files.has(err.file)) {
    actions.append(el("button", { class: "linklike", type: "button", onclick: () => gotoError(err) }, `Go to line ${err.line}`));
  }
  actions.append(
    el("button", { class: "linklike", type: "button", onclick: () => (showAssistant(true), assistant.quick("error")) }, "Explain this error")
  );
  box.append(actions);
  appendConsole(box);
}

function clearConsole() {
  consoleOut().innerHTML = "";
}

function banner(info) {
  const b = el("div", { class: "entry banner" });
  b.append(
    el("strong", {}, "mylab " + (info && info.mylab ? info.mylab : "")),
    ` is ready — Python ${info && info.pyodide ? "(Pyodide " + info.pyodide + ")" : ""} is running in your browser.\n`,
    "Type a command below, or press Ctrl+Enter to run the file in the editor. Try: ",
    el("code", {}, "x = magic(4)"),
    "\n"
  );
  appendConsole(b);
}

/* ================================================================== running code */

function setRunning(running) {
  $("btn-run").hidden = running;
  $("btn-run-sel").disabled = running || !engine.ready;
  $("btn-stop").hidden = !running;
  clearInterval(state.runTimer);
  if (running) {
    state.runStarted = performance.now();
    setStatus("busy", "Running…");
    state.runTimer = setInterval(() => {
      const s = (performance.now() - state.runStarted) / 1000;
      setStatus("busy", `Running… ${s.toFixed(0)} s`);
    }, 1000);
  } else if (engine.ready) {
    setStatus("ready", "Ready");
  }
}

function setStatus(kind, text) {
  $("status").dataset.state = kind;
  $("status-text").textContent = text;
}

function clearErrorMark() {
  if (state.errorMark) {
    const { doc, handle } = state.errorMark;
    try {
      doc.removeLineClass(handle, "background", "cm-error-line");
      doc.removeLineClass(handle, "gutter", "cm-error-gutter");
    } catch {
      /* the line may be gone */
    }
    state.errorMark = null;
  }
}

function markErrorLine(file, line) {
  clearErrorMark();
  if (!state.files.has(file)) return;
  const doc = getDoc(file);
  if (line < 1 || line > doc.lineCount()) return;
  const handle = doc.addLineClass(line - 1, "background", "cm-error-line");
  doc.addLineClass(line - 1, "gutter", "cm-error-gutter");
  state.errorMark = { doc, handle, file };
}

function gotoError(err) {
  openFile(err.file);
  markErrorLine(err.file, err.line);
  editor.setCursor({ line: err.line - 1, ch: 0 });
  editor.scrollIntoView({ line: err.line - 1, ch: 0 }, 80);
  editor.focus();
}

function errorLineOf(res) {
  if (res.error_line) return res.error_line;
  const m = /\bline (\d+)/i.exec(res.error || "");
  return m ? Number(m[1]) : null;
}

/**
 * Run code in the engine.
 * ctx: { kind: "file"|"selection"|"command", file, lineOffset, label }
 */
async function runCode(code, ctx) {
  if (!engine.ready) {
    toast("The engine is still loading. Please wait a moment.");
    return null;
  }
  if (engine.busy) {
    toast("Code is already running. Press Stop to cancel it.");
    return null;
  }
  clearErrorMark();
  if (ctx.kind === "command") printEcho(code);
  else printEcho(code, ctx.label);
  setRunning(true);
  let res;
  try {
    res = await engine.run(code, allFilesForEngine());
  } catch (e) {
    setRunning(false);
    if (String(e.message) !== "Stopped.") printError({ message: e.message });
    return null;
  }
  setRunning(false);
  printOut(res.output);
  if (res.error) {
    let line = errorLineOf(res);
    const err = { message: res.error, line: null, where: "", file: null, code: ctx.kind === "command" ? code : null };
    if (line && ctx.kind !== "command") {
      line += ctx.lineOffset || 0;
      err.line = line;
      err.file = ctx.file;
      err.where = ctx.file;
      markErrorLine(ctx.file, line);
      if (state.active === ctx.file) editor.scrollIntoView({ line: line - 1, ch: 0 }, 80);
    } else if (ctx.kind === "command") {
      err.where = "Command Window";
      err.line = line && code.includes("\n") ? line : null;
    }
    printError(err);
    if (res.internal) console.warn(res.internal);
    state.lastError = err;
  } else {
    state.lastError = null;
  }
  assistant.setLastError(state.lastError);
  if (res.figures && res.figures.length) addFigures(res.figures, ctx);
  if (res.variables) renderVariables(res.variables);
  return res;
}

async function runActiveFile() {
  if (!state.active) {
    toast("Open a file to run it.");
    return;
  }
  const name = state.active;
  const code = editor.getValue();
  flushSave(name);
  if (!name.endsWith(".m")) {
    toast("Only .m files can be run.", "error");
    return;
  }
  if (isFunctionFile(code)) {
    const fn = name.replace(/\.m$/, "");
    printInfo(
      `${name} is a function file, so it needs to be called with inputs. Call it from the command window or a script, for example:  ${fn}(...)`,
      "note"
    );
    $("console-input").value = fn + "(";
    $("console-input").focus();
    return;
  }
  editor.getWrapperElement().classList.remove("cm-run-flash");
  void editor.getWrapperElement().offsetWidth;
  editor.getWrapperElement().classList.add("cm-run-flash");
  await runCode(code, { kind: "file", file: name, lineOffset: 0, label: `run ${name}` });
}

async function runSelection() {
  if (!state.active) return;
  let code = editor.getSelection();
  let from;
  if (!code.trim()) {
    const cur = editor.getCursor();
    code = editor.getLine(cur.line);
    from = cur.line;
    if (!code.trim()) {
      toast("Select some code (or put the cursor on a line) to run it.");
      return;
    }
  } else {
    from = Math.min(editor.getCursor("from").line, editor.getCursor("to").line);
  }
  const to = from + code.split("\n").length;
  const label = to - from > 1 ? `run lines ${from + 1}–${to} of ${state.active}` : `run line ${from + 1} of ${state.active}`;
  await runCode(code, { kind: "selection", file: state.active, lineOffset: from, label });
}

async function runCommand(code) {
  const trimmed = code.trim();
  if (!trimmed) return;
  pushHistory(code);
  if (/^clc\s*;?$/.test(trimmed)) {
    clearConsole();
    return;
  }
  await runCode(code, { kind: "command" });
}

function stopRun() {
  engine.restart();
  setRunning(false);
  printInfo("Stopped. The engine was restarted, so all variables were cleared. Your files are safe.", "note");
  renderVariables([]);
  showLoader(false);
}

async function resetWorkspace() {
  if (!engine.ready || engine.busy) return;
  try {
    const vars = await engine.reset();
    renderVariables(vars);
    printInfo("Workspace cleared.");
  } catch (e) {
    toast(e.message, "error");
  }
}

/* ================================================================== history & completion */

function pushHistory(code) {
  if (state.history[state.history.length - 1] !== code) state.history.push(code);
  if (state.history.length > 300) state.history = state.history.slice(-300);
  state.historyPos = -1;
  prefs.set("history", state.history);
}

function setInputValue(v) {
  const input = $("console-input");
  input.value = v;
  autosizeInput();
  input.setSelectionRange(v.length, v.length);
}

function autosizeInput() {
  const t = $("console-input");
  t.style.height = "auto";
  t.style.height = Math.min(140, t.scrollHeight) + "px";
}

function historyStep(dir) {
  const h = state.history;
  if (!h.length) return;
  const input = $("console-input");
  if (state.historyPos === -1) {
    if (dir > 0) return;
    state.historyDraft = input.value;
    state.historyPos = h.length;
  }
  // Prefix search: only show entries that start with what was typed.
  const prefix = state.historyDraft;
  let i = state.historyPos;
  do {
    i += dir;
  } while (i >= 0 && i < h.length && prefix && !h[i].startsWith(prefix));
  if (i < 0) return;
  if (i >= h.length) {
    state.historyPos = -1;
    setInputValue(state.historyDraft);
    return;
  }
  state.historyPos = i;
  setInputValue(h[i]);
}

function identifierBefore(text) {
  const m = /([A-Za-z_][A-Za-z0-9_]*)$/.exec(text);
  return m ? m[1] : "";
}

async function completeConsole() {
  const input = $("console-input");
  const pos = input.selectionStart;
  const before = input.value.slice(0, pos);
  const prefix = identifierBefore(before);
  const box = $("completions");
  if (!prefix || !engine.ready || engine.busy) return;
  let names = [];
  try {
    names = await engine.complete(prefix);
  } catch {
    return;
  }
  const apply = (name) => {
    input.value = before.slice(0, before.length - prefix.length) + name + input.value.slice(pos);
    const p = pos - prefix.length + name.length;
    input.setSelectionRange(p, p);
    box.hidden = true;
    input.focus();
  };
  if (names.length === 1) return apply(names[0]);
  if (!names.length) return;
  // Extend to the longest common prefix.
  let common = names[0];
  for (const n of names) while (!n.startsWith(common)) common = common.slice(0, -1);
  if (common.length > prefix.length) return apply(common);
  box.innerHTML = "";
  for (const n of names.slice(0, 60)) box.appendChild(el("button", { type: "button", onclick: () => apply(n) }, n));
  if (names.length > 60) box.appendChild(el("span", { class: "muted small" }, `+${names.length - 60} more`));
  box.hidden = false;
}

function editorHint(cm) {
  return new Promise((resolve) => {
    const cur = cm.getCursor();
    const line = cm.getLine(cur.line);
    let start = cur.ch;
    while (start > 0 && /\w/.test(line.charAt(start - 1))) start--;
    const prefix = line.slice(start, cur.ch);
    if (!prefix || /^\d/.test(prefix)) return resolve(null);
    // Words from the current file plus names the engine knows.
    const words = new Set();
    const re = /[A-Za-z_]\w*/g;
    const text = cm.getValue();
    let m;
    while ((m = re.exec(text))) if (m[0] !== prefix && m[0].startsWith(prefix)) words.add(m[0]);
    for (const n of state.files.keys()) {
      const base = n.replace(/\.m$/, "");
      if (base.startsWith(prefix) && base !== prefix) words.add(base);
    }
    const finish = (extra) => {
      for (const e of extra) words.add(e);
      const list = [...words].sort((a, b) => a.length - b.length || a.localeCompare(b)).slice(0, 80);
      resolve(list.length ? { list, from: CodeMirror.Pos(cur.line, start), to: CodeMirror.Pos(cur.line, cur.ch) } : null);
    };
    if (engine.ready && !engine.busy) engine.complete(prefix).then(finish, () => finish([]));
    else finish([]);
  });
}
editorHint.async = false;

/* ================================================================== workspace */

function renderVariables(vars) {
  const before = new Map(state.variables.map((v) => [v.name, v.preview + v.size]));
  state.variables = vars || [];
  const tbody = $("var-table").querySelector("tbody");
  tbody.innerHTML = "";
  for (const v of state.variables) {
    const changed = before.get(v.name) !== v.preview + v.size;
    const tr = el(
      "tr",
      {
        tabindex: "0",
        class: changed ? "changed" : "",
        title: `Click to show ${v.name}`,
        "data-name": v.name,
        onclick: () => showVariable(v),
        onkeydown: (e) => e.key === "Enter" && showVariable(v),
      },
      el("td", {}, v.name),
      el("td", {}, v.size),
      el("td", {}, v.class),
      el("td", { title: v.preview }, v.preview)
    );
    tbody.appendChild(tr);
  }
  $("var-empty").hidden = state.variables.length > 0;
  $("var-table").hidden = state.variables.length === 0;
  $("var-count").textContent = state.variables.length ? `${state.variables.length}` : "";
}

async function showVariable(v) {
  if (!engine.ready || engine.busy) {
    toast("Wait for the running code to finish.");
    return;
  }
  try {
    const text = await engine.showVariable(v.name);
    $("var-title").textContent = v.name;
    $("var-meta").textContent = `${v.size} ${v.class}`;
    $("var-body").textContent = text.replace(/\n+$/, "");
    $("dlg-var").showModal();
  } catch (e) {
    toast(e.message, "error");
  }
}

/* ================================================================== plots */

function addFigures(figs, ctx) {
  const box = $("plots");
  $("plots-empty").hidden = true;
  const source = ctx.kind === "command" ? "Command Window" : ctx.file;
  const time = new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
  figs.forEach((b64, i) => {
    state.figureCount++;
    const n = state.figureCount;
    const url = "data:image/png;base64," + b64;
    const filename = `${(source || "figure").replace(/\.m$/, "").replace(/\W+/g, "_")}_${n}.png`;
    const img = el("img", { src: url, alt: `Figure ${n} from ${source}`, loading: "lazy", onclick: () => enlargeFigure(url, filename) });
    const card = el(
      "figure",
      { class: "plot-card new", style: "margin:0" },
      img,
      el(
        "figcaption",
        { class: "plot-meta" },
        el("span", {}, `Figure ${n}${figs.length > 1 ? ` (${i + 1}/${figs.length})` : ""} · ${source} · ${time}`),
        el("button", { class: "btn icon sm ghost", title: "Enlarge", "aria-label": "Enlarge", onclick: () => enlargeFigure(url, filename) }, icon("expand")),
        el(
          "a",
          { class: "btn icon sm ghost", title: "Download PNG", "aria-label": "Download PNG", href: url, download: filename },
          icon("download")
        ),
        el("button", { class: "btn icon sm ghost", title: "Remove", "aria-label": "Remove plot", onclick: () => (card.remove(), updatePlotsEmpty()) }, icon("trash"))
      )
    );
    box.insertBefore(card, box.querySelector(".plot-card"));
  });
  // Keep memory bounded on low-end devices.
  const cards = box.querySelectorAll(".plot-card");
  for (let i = 40; i < cards.length; i++) cards[i].remove();
  box.scrollTop = 0;
  const note = el("div", { class: "entry fig-note", onclick: () => $("plots").scrollIntoView({ block: "nearest" }) },
    `${figs.length} figure${figs.length > 1 ? "s" : ""} drawn → see the Plots panel`);
  appendConsole(note);
}

function updatePlotsEmpty() {
  $("plots-empty").hidden = !!$("plots").querySelector(".plot-card");
}

function enlargeFigure(url, filename) {
  $("figure-big").src = url;
  $("figure-dl").href = url;
  $("figure-dl").download = filename;
  $("dlg-figure").showModal();
}

/* ================================================================== layout & resizing */

const SIZES = prefs.get("sizes", {});

function applySizes() {
  const root = document.documentElement.style;
  if (SIZES.files) root.setProperty("--files-w", SIZES.files + "px");
  if (SIZES.side) root.setProperty("--side-w", SIZES.side + "px");
  if (SIZES.ai) root.setProperty("--ai-w", SIZES.ai + "px");
  if (SIZES.console) root.setProperty("--console-h", SIZES.console + "px");
  if (SIZES.workspace) root.setProperty("--workspace-h", SIZES.workspace + "px");
}

function setupGutters() {
  const limits = { files: [140, 420], side: [220, 700], ai: [280, 720], console: [90, 900], workspace: [80, 900] };
  document.querySelectorAll(".gutter").forEach((g) => {
    const key = g.dataset.resize;
    const target = {
      files: () => $("files-panel"),
      side: () => $("side"),
      ai: () => $("ai-panel"),
      console: () => document.querySelector(".console-panel"),
      workspace: () => document.querySelector(".workspace-panel"),
    }[key];
    const vertical = g.classList.contains("gutter-v");
    // Panels to the right/bottom of the gutter grow when the gutter moves toward them... except files.
    const sign = key === "files" || key === "workspace" ? 1 : -1;
    const set = (px) => {
      const [lo, hi] = limits[key];
      px = Math.round(Math.max(lo, Math.min(hi, px)));
      SIZES[key] = px;
      const prop = { files: "--files-w", side: "--side-w", ai: "--ai-w", console: "--console-h", workspace: "--workspace-h" }[key];
      document.documentElement.style.setProperty(prop, px + "px");
      editor && editor.refresh();
    };
    g.addEventListener("pointerdown", (ev) => {
      ev.preventDefault();
      g.setPointerCapture(ev.pointerId);
      const start = vertical ? ev.clientX : ev.clientY;
      const rect = target().getBoundingClientRect();
      const startSize = vertical ? rect.width : rect.height;
      g.classList.add("dragging");
      document.body.classList.add("resizing");
      const move = (e) => set(startSize + sign * ((vertical ? e.clientX : e.clientY) - start));
      const up = () => {
        g.classList.remove("dragging");
        document.body.classList.remove("resizing");
        g.removeEventListener("pointermove", move);
        g.removeEventListener("pointerup", up);
        g.removeEventListener("pointercancel", up);
        prefs.set("sizes", SIZES);
      };
      g.addEventListener("pointermove", move);
      g.addEventListener("pointerup", up);
      g.addEventListener("pointercancel", up);
    });
    g.addEventListener("keydown", (ev) => {
      const keys = vertical ? ["ArrowLeft", "ArrowRight"] : ["ArrowUp", "ArrowDown"];
      const i = keys.indexOf(ev.key);
      if (i < 0) return;
      ev.preventDefault();
      const rect = target().getBoundingClientRect();
      const cur = vertical ? rect.width : rect.height;
      set(cur + sign * (i === 0 ? -20 : 20));
      prefs.set("sizes", SIZES);
    });
  });
}

function showAssistant(show) {
  const panel = $("ai-panel");
  const visible = show === undefined ? panel.hidden : show;
  panel.hidden = !visible;
  document.querySelector(".ai-gutter").hidden = !visible;
  $("layout").classList.toggle("with-ai", visible);
  $("btn-ai").setAttribute("aria-pressed", String(visible));
  prefs.set("aiOpen", visible);
  editor && setTimeout(() => editor.refresh(), 0);
  if (visible) {
    if (matchMedia("(max-width: 820px)").matches) panel.scrollIntoView({ behavior: "smooth", block: "start" });
    setTimeout(() => $("ai-input").focus(), 50);
  }
}

/* ================================================================== loader */

const STEP_ORDER = ["python", "packages", "mylab", "start"];
const STEP_PROGRESS = { python: 8, packages: 30, mylab: 80, start: 92 };

function showLoader(show) {
  $("loader").hidden = !show;
  if (show) {
    document.querySelectorAll(".steps li").forEach((li) => li.classList.remove("active", "done", "failed"));
    $("loader-bar").style.width = "3%";
    $("loader-error").hidden = true;
    $("loader-retry").hidden = true;
    $("loader-hide").hidden = false;
  }
}

function onProgress({ step, label }) {
  const idx = STEP_ORDER.indexOf(step);
  document.querySelectorAll(".steps li").forEach((li) => {
    const i = STEP_ORDER.indexOf(li.dataset.step);
    li.classList.toggle("done", i < idx);
    li.classList.toggle("active", i === idx);
  });
  $("loader-bar").style.width = (STEP_PROGRESS[step] || 5) + "%";
  setStatus("loading", `${label} (${idx + 1}/4)`);
}

function onReady(info) {
  document.querySelectorAll(".steps li").forEach((li) => {
    li.classList.remove("active");
    li.classList.add("done");
  });
  $("loader-bar").style.width = "100%";
  setTimeout(() => ($("loader").hidden = true), 250);
  setStatus("ready", "Ready");
  document.body.dataset.engine = "ready";
  $("btn-run").disabled = false;
  $("btn-run-sel").disabled = false;
  const input = $("console-input");
  input.disabled = false;
  input.placeholder = "Type a command and press Enter";
  $("about-versions").textContent = `mylab ${info.mylab} · Pyodide ${info.pyodide} · engine running in a Web Worker`;
  if (!state.bannerShown) {
    banner(info);
    state.bannerShown = true;
  } else {
    printInfo("Engine restarted and ready.");
  }
  if (document.activeElement === document.body) input.focus();
}

function onFatal({ error }) {
  document.body.dataset.engine = "error";
  setStatus("error", "Engine failed to load");
  $("loader").hidden = false;
  document.querySelectorAll(".steps li.active").forEach((li) => {
    li.classList.remove("active");
    li.classList.add("failed");
  });
  const msg = $("loader-error");
  msg.hidden = false;
  msg.textContent =
    "The engine couldn't start: " + error + " — Check your internet connection. On very old browsers, WebAssembly may not be supported.";
  $("loader-retry").hidden = false;
  $("loader-hide").hidden = false;
  printError({ message: "The engine failed to load: " + error });
}

/* ================================================================== setup */

function setupEditor() {
  const mod = isMac ? "Cmd" : "Ctrl";
  editor = CodeMirror($("editor"), {
    value: "",
    mode: "text/x-octave",
    lineNumbers: true,
    indentUnit: 4,
    tabSize: 4,
    indentWithTabs: false,
    matchBrackets: true,
    autoCloseBrackets: "()[]{}''\"\"",
    styleActiveLine: true,
    lineWrapping: false,
    viewportMargin: 30,
    extraKeys: {
      [`${mod}-Enter`]: () => runActiveFile(),
      [`Shift-${mod}-Enter`]: () => runSelection(),
      [`${mod}-S`]: () => {
        if (state.active) flushSave(state.active).then(() => toast("Saved."));
      },
      [`${mod}-/`]: (cm) => cm.toggleComment({ indent: true }),
      "Ctrl-Space": (cm) => cm.showHint({ hint: editorHint, completeSingle: true }),
      Tab: (cm) => {
        if (cm.somethingSelected()) cm.indentSelection("add");
        else cm.replaceSelection(" ".repeat(cm.getOption("indentUnit")), "end");
      },
      "Shift-Tab": (cm) => cm.indentSelection("subtract"),
      [`${mod}-L`]: () => $("console-input").focus(),
    },
  });
  editor.on("change", (cm, change) => {
    if (!state.active) return;
    scheduleSave(state.active);
    if (state.errorMark && state.errorMark.file === state.active && change.origin !== "setValue") clearErrorMark();
  });
}

function setupConsole() {
  const input = $("console-input");
  input.addEventListener("keydown", (ev) => {
    const box = $("completions");
    if (ev.key === "Enter" && !ev.shiftKey && !ev.isComposing) {
      ev.preventDefault();
      if (engine.busy) return;
      const code = input.value;
      setInputValue("");
      box.hidden = true;
      runCommand(code);
      return;
    }
    if (ev.key === "Tab" && !ev.shiftKey) {
      ev.preventDefault();
      completeConsole();
      return;
    }
    if (ev.key === "Escape") {
      box.hidden = true;
      return;
    }
    const atFirstLine = !input.value.slice(0, input.selectionStart).includes("\n");
    const atLastLine = !input.value.slice(input.selectionEnd).includes("\n");
    if (ev.key === "ArrowUp" && atFirstLine) {
      ev.preventDefault();
      historyStep(-1);
    } else if (ev.key === "ArrowDown" && atLastLine) {
      ev.preventDefault();
      historyStep(1);
    } else if (ev.key === "l" && (ev.ctrlKey || ev.metaKey)) {
      ev.preventDefault();
      clearConsole();
    }
  });
  input.addEventListener("input", () => {
    autosizeInput();
    state.historyPos = -1;
    $("completions").hidden = true;
  });
  // Clicking on empty console space focuses the prompt (unless text is being selected).
  consoleOut().addEventListener("mouseup", () => {
    if (!getSelection().toString() && !input.disabled) input.focus();
  });
}

function setupDragDrop() {
  let depth = 0;
  const overlay = $("drop-overlay");
  const hasFiles = (e) => e.dataTransfer && [...e.dataTransfer.types].includes("Files");
  window.addEventListener("dragenter", (e) => {
    if (!hasFiles(e)) return;
    e.preventDefault();
    depth++;
    overlay.hidden = false;
  });
  window.addEventListener("dragover", (e) => {
    if (hasFiles(e)) e.preventDefault();
  });
  window.addEventListener("dragleave", (e) => {
    if (!hasFiles(e)) return;
    depth = Math.max(0, depth - 1);
    if (!depth) overlay.hidden = true;
  });
  window.addEventListener("drop", (e) => {
    if (!hasFiles(e)) return;
    e.preventDefault();
    depth = 0;
    overlay.hidden = true;
    importFiles([...e.dataTransfer.files]);
  });
}

function setupButtons() {
  $("btn-run").addEventListener("click", runActiveFile);
  $("btn-run-sel").addEventListener("click", runSelection);
  $("btn-stop").addEventListener("click", stopRun);
  $("btn-new").addEventListener("click", newFile);
  $("btn-import").addEventListener("click", () => $("file-input").click());
  $("file-input").addEventListener("change", (e) => {
    importFiles([...e.target.files]);
    e.target.value = "";
  });
  $("btn-zip").addEventListener("click", exportZip);
  $("btn-clear-console").addEventListener("click", clearConsole);
  $("btn-reset").addEventListener("click", resetWorkspace);
  $("btn-clear-plots").addEventListener("click", () => {
    $("plots").querySelectorAll(".plot-card").forEach((c) => c.remove());
    updatePlotsEmpty();
  });
  $("btn-theme").addEventListener("click", toggleTheme);
  $("btn-keys").addEventListener("click", () => $("dlg-keys").showModal());
  $("btn-about").addEventListener("click", () => $("dlg-about").showModal());
  $("btn-ai").addEventListener("click", () => showAssistant());
  $("btn-ai-close").addEventListener("click", () => showAssistant(false));
  $("loader-hide").addEventListener("click", () => ($("loader").hidden = true));
  $("status").addEventListener("click", () => {
    if (!engine.ready) $("loader").hidden = false;
  });
  $("loader-retry").addEventListener("click", () => {
    showLoader(true);
    setStatus("loading", "Loading engine…");
    engine.restart();
  });
  $("var-copy").addEventListener("click", async () => {
    try {
      await navigator.clipboard.writeText($("var-body").textContent);
      toast("Copied.");
    } catch {
      toast("Couldn't access the clipboard.", "error");
    }
  });
  // Close dialogs when clicking the backdrop.
  document.querySelectorAll("dialog").forEach((d) =>
    d.addEventListener("click", (e) => {
      if (e.target === d) d.close("cancel");
    })
  );
  document.addEventListener("keydown", (e) => {
    const mod = isMac ? e.metaKey : e.ctrlKey;
    if (e.altKey && (e.key === "n" || e.key === "N" || e.code === "KeyN")) {
      e.preventDefault();
      newFile();
    } else if (mod && e.key === "Enter" && !editor.hasFocus() && document.activeElement !== $("console-input") && document.activeElement !== $("ai-input")) {
      e.preventDefault();
      e.shiftKey ? runSelection() : runActiveFile();
    } else if (mod && (e.key === "e" || e.key === "E") && !e.shiftKey) {
      e.preventDefault();
      editor.focus();
    } else if (mod && (e.key === "l" || e.key === "L") && document.activeElement !== $("console-input")) {
      e.preventDefault();
      $("console-input").focus();
    } else if (mod && (e.key === "s" || e.key === "S")) {
      e.preventDefault();
      flushAll().then(() => toast("Saved."));
    }
  });
  window.addEventListener("beforeunload", () => {
    for (const name of state.saveTimers.keys()) flushSave(name);
  });
  document.addEventListener("visibilitychange", () => {
    if (document.visibilityState === "hidden") flushAll();
  });
  matchMedia("(prefers-color-scheme: dark)").addEventListener?.("change", updateThemeIcon);
}

function insertIntoEditor(code) {
  if (!state.active) {
    toast("Open a file first.", "error");
    return;
  }
  const text = code.endsWith("\n") ? code : code + "\n";
  editor.replaceSelection(text, "end");
  editor.focus();
  toast(`Inserted into ${state.active}.`);
}

async function replaceFileWith(code) {
  if (!state.active) {
    toast("Open a file first.", "error");
    return;
  }
  const ok = await confirmDialog({
    title: `Replace ${state.active}?`,
    text: "The whole file will be replaced with the assistant's code. You can undo with Ctrl+Z.",
    okLabel: "Replace",
  });
  if (!ok) return;
  const doc = editor.getDoc();
  doc.replaceRange(code.endsWith("\n") ? code : code + "\n", CodeMirror.Pos(doc.firstLine(), 0), CodeMirror.Pos(doc.lastLine()));
  editor.focus();
}

function assistantContext() {
  const ctx = { fileName: null, code: "", selection: null, lastError: state.lastError, variables: state.variables };
  if (state.active) {
    ctx.fileName = state.active;
    ctx.code = editor.getValue();
    const sel = editor.getSelection();
    if (sel.trim() && sel.length < ctx.code.length) {
      const from = editor.getCursor("from").line + 1;
      ctx.selection = { text: sel, from, to: from + sel.split("\n").length - 1 };
    }
  }
  return ctx;
}

async function main() {
  updateThemeIcon();
  applySizes();
  setupEditor();
  setupConsole();
  setupButtons();
  setupGutters();
  setupDragDrop();

  // Start the engine as early as possible; it loads while the files are read.
  engine = new Engine();
  engine.addEventListener("progress", (e) => onProgress(e.detail));
  engine.addEventListener("ready", (e) => onReady(e.detail));
  engine.addEventListener("fatal", (e) => onFatal(e.detail));
  engine.addEventListener("restarting", () => {
    document.body.dataset.engine = "loading";
    $("btn-run").disabled = true;
    $("btn-run-sel").disabled = true;
    setStatus("loading", "Restarting engine…");
  });

  assistant = new Assistant({
    getContext: assistantContext,
    insertCode: insertIntoEditor,
    replaceFile: replaceFileWith,
    runCode: (code) => runCode(code, { kind: "command" }),
    toast,
  });
  if (prefs.get("aiOpen", false)) showAssistant(true);

  await loadFiles();
  renderFileList();
  const session = prefs.get("session", null);
  const tabs = (session && session.tabs ? session.tabs : ["welcome.m"]).filter((t) => state.files.has(t));
  state.tabs = tabs;
  const first = session && state.files.has(session.active) ? session.active : tabs[0] || sortedNames()[0];
  if (first) openFile(first, { focus: false });
  else showNoFile();
  $("save-state").textContent = store.persistent ? "Saved" : "Not saved (no storage)";
}

main().catch((e) => {
  console.error(e);
  toast("mylab failed to start: " + e.message, "error");
});

// Exposed for automated tests and debugging.
window.mylab = { state, get engine() { return engine; }, runCode, openFile, createFile };
