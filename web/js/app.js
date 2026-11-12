// mylab web app: wires the editor, files, command window, workspace, plots and assistant together.
// Feature modules (projects, sharing, reports, step-through, assignments) plug into the `app` object.

import { Engine } from "./engine.js";
import { ProjectStore, DEFAULT_PROJECT, newId, prefs } from "./storage.js";
import { EXAMPLES } from "./examples.js";
import { Assistant } from "./ai.js";
import { $, el, icon, toast, isMac, downloadBlob, promptDialog, confirmDialog } from "./ui.js";
import { parseHash, toStored, isImageName, mimeOf, formatSize } from "./codec.js";
import { splitSections, sectionAt, sectionCode, tailCode } from "./sections.js";
import { initProjects } from "./projects.js";
import { initShare } from "./share.js";
import { initPublish } from "./publish.js";
import { initStepper } from "./stepper.js";
import { initAssignments } from "./assignment.js";

/* ================================================================== state */

const state = {
  project: null, // { id, name, kind, meta, temp }
  projects: [],
  files: new Map(), // path -> { content: string | Uint8Array, modified }
  docs: new Map(), // path -> CodeMirror.Doc (text files that have been opened)
  tabs: [],
  active: null,
  collapsed: new Set(), // collapsed folders
  saveTimers: new Map(),
  sent: new Map(), // path -> content last copied into the engine's working folder
  fsReset: true, // wipe the engine's working folder on the next request (new project)
  lastError: null, // { message, line, where, code }
  variables: [],
  history: prefs.get("history", []),
  historyPos: -1,
  historyDraft: "",
  figureCount: 0,
  errorMark: null,
  runStarted: 0,
  runTimer: null,
  sectionView: null,
};

const store = new ProjectStore();
let editor;
let engine;
let assistant;

/** Shared with the feature modules. Filled in below and by the modules' init functions. */
const app = { state, store, prefs };

/* ================================================================== helpers */

function isFunctionFile(code) {
  for (const raw of code.split("\n")) {
    const line = raw.trim();
    if (!line || line.startsWith("%") || line.startsWith("#")) continue;
    return /^function\b/.test(line);
  }
  return false;
}

const VALID_NAME = /^[A-Za-z][A-Za-z0-9_]*\.m$/;
const baseName = (p) => p.slice(p.lastIndexOf("/") + 1);
const dirName = (p) => (p.includes("/") ? p.slice(0, p.lastIndexOf("/")) : "");
const isText = (path) => typeof (state.files.get(path) || {}).content === "string";

function normalizePath(input) {
  return String(input).trim().replace(/\\/g, "/").replace(/\/+/g, "/").replace(/^\/|\/$/g, "");
}

function validatePath(input, { current = null, addExt = true } = {}) {
  let p = normalizePath(input);
  if (!p) return { error: "Please enter a name." };
  const parts = p.split("/");
  if (addExt && !/\.[A-Za-z0-9]+$/.test(parts[parts.length - 1])) {
    parts[parts.length - 1] += ".m";
    p = parts.join("/");
  }
  if (parts.some((s) => /[:*?"<>|]/.test(s) || s.startsWith(".") || !s.trim())) return { error: "That name has characters that can't be used in file names." };
  if (p.length > 160 || parts.some((s) => s.length > 80)) return { error: "Please use a shorter name." };
  if (p !== current && state.files.has(p)) return { error: `A file named ${p} already exists.` };
  for (const other of state.files.keys()) {
    if (other === current) continue;
    if (other.startsWith(p + "/")) return { error: `${p} is already a folder.` };
    if (p.startsWith(other + "/")) return { error: `${other} is a file, so it can't be used as a folder.` };
  }
  const base = parts[parts.length - 1];
  const warn =
    base.endsWith(".m") && !VALID_NAME.test(base)
      ? "Tip: .m names should start with a letter and use only letters, digits and _, so the file can be called by name."
      : parts.length > 1 && current === null
        ? `The file will be created in the folder ${parts.slice(0, -1).join("/")}.`
        : "";
  return { name: p, warn };
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

/* ================================================================== projects */

async function loadProjects() {
  await store.init();
  state.projects = await store.listProjects();
  const seeded = prefs.get("seeded", false);
  if (!state.projects.some((p) => p.id === DEFAULT_PROJECT) && (!state.projects.length || !seeded)) {
    const p = await store.putProject({ id: DEFAULT_PROJECT, name: "My files", kind: "files" });
    if (!seeded) {
      await store.putFiles(p.id, EXAMPLES.map((e) => ({ path: e.name, content: e.content })));
      prefs.set("seeded", true);
    }
    state.projects = await store.listProjects();
  }
  sortProjects();
  $("storage-note").textContent = store.persistent
    ? "Files are saved in this browser."
    : "This browser blocks storage: files last only until you close the tab. Download them to keep them.";
  $("storage-note").classList.toggle("error-text", !store.persistent);
  const wanted = prefs.get("project", DEFAULT_PROJECT);
  const first = state.projects.find((p) => p.id === wanted) || state.projects.find((p) => p.id === DEFAULT_PROJECT) || state.projects[0];
  await switchProject(first.id, { initial: true });
}

function sortProjects() {
  state.projects.sort((a, b) => (a.id === DEFAULT_PROJECT ? -1 : b.id === DEFAULT_PROJECT ? 1 : a.name.localeCompare(b.name, undefined, { sensitivity: "base", numeric: true })));
}

function sessionFor(id) {
  const all = prefs.get("sessions", {});
  if (all[id]) return all[id];
  if (id === DEFAULT_PROJECT) return prefs.get("session", null) || { tabs: ["welcome.m"] };
  return null;
}

function saveSession() {
  if (!state.project || state.project.temp) return;
  const all = prefs.get("sessions", {});
  all[state.project.id] = { tabs: state.tabs, active: state.active, collapsed: [...state.collapsed] };
  // Forget sessions of deleted projects.
  for (const id of Object.keys(all)) if (!state.projects.some((p) => p.id === id)) delete all[id];
  prefs.set("sessions", all);
}

/** Open another project. Returns false if it could not be opened. */
async function switchProject(id, { initial = false, open = null } = {}) {
  if (!initial && engine && engine.busy) {
    toast("Wait for the running code to finish (or press Stop) before switching projects.");
    return false;
  }
  const project = state.projects.find((p) => p.id === id);
  if (!project) return false;
  if (!initial) {
    await flushAll();
    saveSession();
    app.exitStep && app.exitStep();
  }
  clearErrorMark();
  for (const t of state.saveTimers.values()) clearTimeout(t);
  state.saveTimers.clear();
  state.project = project;
  state.files.clear();
  state.docs.clear();
  state.sent.clear();
  state.fsReset = true;
  for (const f of await store.listFiles(id)) state.files.set(f.path, { content: f.content, modified: f.modified });
  const session = sessionFor(id);
  state.collapsed = new Set((session && session.collapsed) || []);
  state.tabs = ((session && session.tabs) || []).filter((t) => isText(t));
  if (!project.temp) prefs.set("project", id);
  renderProjectHead();
  renderFileList();
  let first = open && state.files.has(open) ? open : session && isText(session.active) ? session.active : state.tabs[0];
  if (!first) first = sortedPaths().find((p) => p.endsWith(".m")) || sortedPaths().find(isText);
  if (first) openFile(first, { focus: false });
  else showNoFile();
  $("save-state").textContent = saveLabel();
  app.onProjectChanged && app.onProjectChanged();
  return true;
}

/** Create a project (optionally with files) and open it. Returns the project. */
async function createProject({ name, kind = "files", meta = {}, temp = false, files = [], open = null, switchTo = true }) {
  const project = await store.putProject({ id: newId(), name: uniqueProjectName(name), kind, meta, temp });
  if (files.length) await store.putFiles(project.id, files);
  state.projects.push(project);
  sortProjects();
  if (switchTo) await switchProject(project.id, { open });
  return project;
}

function uniqueProjectName(name) {
  name = (name || "Project").trim().slice(0, 60) || "Project";
  const taken = new Set(state.projects.map((p) => p.name.toLowerCase()));
  if (!taken.has(name.toLowerCase())) return name;
  let n = 2;
  while (taken.has(`${name} (${n})`.toLowerCase())) n++;
  return `${name} (${n})`;
}

async function saveProjectRecord() {
  await store.putProject(state.project);
}

async function deleteProject(id) {
  const i = state.projects.findIndex((p) => p.id === id);
  if (i < 0) return;
  const wasCurrent = state.project && state.project.id === id;
  state.projects.splice(i, 1);
  await store.deleteProject(id);
  if (!state.projects.length) {
    state.projects.push(await store.putProject({ id: DEFAULT_PROJECT, name: "My files", kind: "files" }));
  }
  if (wasCurrent) {
    state.docs.clear();
    state.saveTimers.clear();
    const next = state.projects.find((p) => p.id === prefs.get("project", DEFAULT_PROJECT) && !p.temp) || state.projects.find((p) => !p.temp) || state.projects[0];
    await switchProject(next.id, { initial: true });
  } else {
    renderProjectHead();
  }
}

const KIND_LABEL = { assignment: "Assignment", github: "From GitHub", shared: "Shared" };

function renderProjectHead() {
  const p = state.project;
  $("project-name").textContent = p.name;
  $("project-btn").title = `Project: ${p.name} — click to switch, create or import projects`;
  const banner = $("temp-banner");
  banner.hidden = !p.temp;
  if (p.temp) {
    $("temp-text").textContent =
      p.kind === "shared"
        ? "Shared code, opened from a link. This is a temporary copy: it is not saved and your own files are untouched."
        : "Temporary project: it is not saved in this browser.";
  }
  document.body.dataset.project = p.kind || "files";
}

/* ================================================================== files */

function sortedPaths() {
  return [...state.files.keys()].sort((a, b) => a.localeCompare(b, undefined, { sensitivity: "base", numeric: true }));
}

function saveLabel() {
  if (state.project && state.project.temp) return "Temporary (not saved)";
  return store.persistent ? "Saved" : "Not saved (no storage)";
}

function fileIcon(path, f) {
  if (typeof f.content !== "string") return isImageName(path) ? "image" : "data";
  if (path.endsWith(".m")) return isFunctionFile(f.content) ? "fn" : "file";
  return /\.(csv|tsv|dat|json)$/i.test(path) ? "table" : "text";
}

function buildTree() {
  const root = { path: "", folders: new Map(), files: [] };
  for (const path of sortedPaths()) {
    const parts = path.split("/");
    let node = root;
    for (let i = 0; i < parts.length - 1; i++) {
      if (!node.folders.has(parts[i])) node.folders.set(parts[i], { path: parts.slice(0, i + 1).join("/"), folders: new Map(), files: [] });
      node = node.folders.get(parts[i]);
    }
    node.files.push(path);
  }
  return root;
}

function renderFileList() {
  const ul = $("file-list");
  const scroll = ul.scrollTop;
  ul.innerHTML = "";
  const indent = (depth) => `padding-left:${8 + depth * 14}px`;
  const act = (title, ic, fn, name) =>
    el("button", { class: "btn icon sm ghost", title, "aria-label": `${title} ${name}`, onclick: (e) => (e.stopPropagation(), fn()) }, icon(ic));

  const renderNode = (node, depth) => {
    const folders = [...node.folders.entries()].sort((a, b) => a[0].localeCompare(b[0], undefined, { sensitivity: "base", numeric: true }));
    for (const [name, child] of folders) {
      const open = !state.collapsed.has(child.path);
      const toggle = () => {
        if (state.collapsed.has(child.path)) state.collapsed.delete(child.path);
        else state.collapsed.add(child.path);
        renderFileList();
        saveSession();
      };
      ul.appendChild(
        el(
          "li",
          {
            class: "folder-item" + (open ? " open" : ""),
            role: "treeitem",
            "aria-expanded": String(open),
            tabindex: "0",
            title: child.path + "/",
            "data-folder": child.path,
            style: indent(depth),
            onclick: toggle,
            onkeydown: (ev) => {
              if (ev.key === "Enter" || ev.key === " ") (ev.preventDefault(), toggle());
              if (ev.key === "Delete") deleteFolder(child.path);
              if (ev.key === "F2") renameFolder(child.path);
            },
          },
          icon("caret", "caret"),
          icon("folder"),
          el("span", { class: "name" }, name),
          el(
            "span",
            { class: "actions" },
            act("New file in", "plus", () => newFile(child.path), child.path),
            act("Rename folder", "pencil", () => renameFolder(child.path), child.path),
            act("Delete folder", "trash", () => deleteFolder(child.path), child.path)
          )
        )
      );
      if (open) renderNode(child, depth + 1);
    }
    for (const path of node.files) {
      const f = state.files.get(path);
      const name = baseName(path);
      ul.appendChild(
        el(
          "li",
          {
            class: "file-item" + (path === state.active ? " active" : ""),
            role: "treeitem",
            "aria-selected": path === state.active ? "true" : "false",
            tabindex: "0",
            title: path + (typeof f.content === "string" ? "" : ` (${formatSize(f.content.length)})`),
            "data-name": path,
            style: indent(depth),
            onclick: () => openFile(path),
            onkeydown: (ev) => {
              if (ev.key === "Enter") openFile(path);
              if (ev.key === "Delete") deleteFile(path);
              if (ev.key === "F2") renameFile(path);
            },
          },
          icon(fileIcon(path, f)),
          el("span", { class: "name" }, name),
          el(
            "span",
            { class: "actions" },
            path.endsWith(".m") ? null : act("Preview", "eye", () => viewFile(path), path),
            act("Rename", "pencil", () => renameFile(path), path),
            act("Download", "download", () => exportFile(path), path),
            act("Delete", "trash", () => deleteFile(path), path)
          )
        )
      );
    }
  };
  renderNode(buildTree(), 0);
  if (!state.files.size) ul.appendChild(el("li", { class: "empty muted" }, "No files. Create one with +."));
  ul.scrollTop = scroll;
}

function getDoc(path) {
  let doc = state.docs.get(path);
  if (!doc) {
    const mode = path.endsWith(".m") ? "text/x-octave" : "text/plain";
    doc = CodeMirror.Doc(state.files.get(path).content, mode);
    state.docs.set(path, doc);
  }
  return doc;
}

function liveContent(path) {
  const doc = state.docs.get(path);
  return doc ? doc.getValue() : state.files.get(path).content;
}

function expandTo(path) {
  let changed = false;
  const parts = path.split("/");
  for (let i = 1; i < parts.length; i++) changed = state.collapsed.delete(parts.slice(0, i).join("/")) || changed;
  return changed;
}

function openFile(path, { focus = true } = {}) {
  if (!state.files.has(path)) return;
  if (!isText(path)) return viewFile(path);
  if (!state.tabs.includes(path)) state.tabs.push(path);
  state.active = path;
  clearErrorMark();
  expandTo(path);
  editor.swapDoc(getDoc(path));
  $("editor-empty")?.remove();
  editor.getWrapperElement().hidden = false;
  renderTabs();
  renderFileList();
  saveSession();
  if (focus) editor.focus();
  editor.refresh();
  updateSections(true);
  app.onFileOpened && app.onFileOpened(path);
}

function closeTab(path) {
  const i = state.tabs.indexOf(path);
  if (i < 0) return;
  flushSave(path);
  state.tabs.splice(i, 1);
  if (state.active === path) {
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
        el("div", {}, "No file open. Pick one from the Files panel, or ", el("button", { class: "btn sm", onclick: () => newFile() }, "create a new file"), ".")
      )
    );
  }
  renderFileList();
  renderTabs();
  updateSections(true);
}

function renderTabs() {
  const box = $("tabs");
  box.innerHTML = "";
  const counts = new Map();
  for (const t of state.tabs) counts.set(baseName(t), (counts.get(baseName(t)) || 0) + 1);
  for (const path of state.tabs) {
    const name = baseName(path);
    const tab = el(
      "button",
      {
        class: "tab" + (path === state.active ? " active" : ""),
        role: "tab",
        "aria-selected": path === state.active ? "true" : "false",
        title: path,
        "data-name": path,
        onclick: () => openFile(path),
        onauxclick: (e) => e.button === 1 && closeTab(path),
      },
      el("span", {}, counts.get(name) > 1 && dirName(path) ? el("span", { class: "tab-dir" }, dirName(path) + "/") : null, name),
      el(
        "span",
        {
          class: "close",
          role: "button",
          "aria-label": `Close ${name}`,
          title: "Close",
          onclick: (e) => {
            e.stopPropagation();
            closeTab(path);
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

function scheduleSave(path) {
  clearTimeout(state.saveTimers.get(path));
  $("save-state").textContent = "Editing…";
  state.saveTimers.set(
    path,
    setTimeout(() => flushSave(path), 600)
  );
}

async function flushSave(path) {
  clearTimeout(state.saveTimers.get(path));
  state.saveTimers.delete(path);
  const doc = state.docs.get(path);
  const f = state.files.get(path);
  if (!doc || !f) return;
  const content = doc.getValue();
  if (content !== f.content) {
    f.content = content;
    f.modified = Date.now();
    await store.putFile(state.project.id, path, content);
  }
  if (!state.saveTimers.size) $("save-state").textContent = saveLabel();
}

async function flushAll() {
  await Promise.all([...state.docs.keys()].map((n) => flushSave(n)));
}

/** Add or replace several files in the current project. files: [{ path, content }] */
async function writeFiles(files, { render = true } = {}) {
  if (!files.length) return;
  const now = Date.now();
  for (const f of files) {
    state.files.set(f.path, { content: f.content, modified: now });
    const doc = state.docs.get(f.path);
    if (doc) {
      if (typeof f.content === "string") {
        if (doc.getValue() !== f.content) doc.setValue(f.content);
      } else {
        state.docs.delete(f.path);
        if (state.tabs.includes(f.path)) closeTab(f.path);
      }
    }
  }
  await store.putFiles(state.project.id, files);
  if (render) renderFileList();
}

async function createFile(path, content, { open = true } = {}) {
  await writeFiles([{ path, content }]);
  if (open) openFile(path);
}

async function newFile(folder = "") {
  if (typeof folder !== "string") folder = "";
  if (!folder && state.active) folder = dirName(state.active);
  const prefix = folder ? folder + "/" : "";
  let n = 1;
  while (state.files.has(`${prefix}untitled${n > 1 ? n : ""}.m`)) n++;
  const path = await promptDialog({
    title: "New file",
    text: "Scripts and function files end in .m. A function file should have the same name as its function. Use a / to put the file in a folder, for example homework1/solve.m.",
    value: `${prefix}untitled${n > 1 ? n : ""}.m`,
    okLabel: "Create",
    validate: (v) => validatePath(v),
  });
  if (!path) return;
  const fnName = baseName(path).replace(/\.m$/, "");
  const template = path.endsWith(".m") && /^[A-Za-z]\w*$/.test(fnName) ? `% ${fnName}.m\n\n` : "";
  await createFile(path, template);
  editor.setCursor(editor.lineCount(), 0);
}

/** Move files to new paths (rename a file or a whole folder). moves: [[from, to]] */
async function movePaths(moves) {
  await flushAll();
  const puts = [];
  for (const [from, to] of moves) {
    const f = state.files.get(from);
    state.files.delete(from);
    state.files.set(to, f);
    const doc = state.docs.get(from);
    state.docs.delete(from);
    if (doc) state.docs.set(to, doc);
    state.tabs = state.tabs.map((t) => (t === from ? to : t));
    if (state.active === from) state.active = to;
    puts.push({ path: to, content: f.content });
  }
  await store.putFiles(state.project.id, puts);
  await store.removeFiles(state.project.id, moves.map((m) => m[0]));
  renderTabs();
  renderFileList();
  saveSession();
}

async function renameFile(path) {
  const newPath = await promptDialog({
    title: "Rename or move file",
    text: "Change the name, or add a folder in front (folder/name.m) to move the file.",
    value: path,
    okLabel: "Rename",
    validate: (v) => validatePath(v, { current: path }),
  });
  if (!newPath || newPath === path) return;
  await movePaths([[path, newPath]]);
  toast(`Renamed to ${newPath}`);
}

async function renameFolder(folder) {
  const inside = sortedPaths().filter((p) => p.startsWith(folder + "/"));
  const target = await promptDialog({
    title: "Rename folder",
    value: folder,
    okLabel: "Rename",
    selectAll: true,
    validate: (v) => {
      const p = normalizePath(v);
      if (!p) return { error: "Please enter a name." };
      if (p.split("/").some((s) => /[:*?"<>|]/.test(s) || s.startsWith(".") || !s.trim())) return { error: "That name has characters that can't be used in folder names." };
      if (p !== folder && (p + "/").startsWith(folder + "/")) return { error: "A folder can't be moved into itself." };
      if (p !== folder && [...state.files.keys()].some((o) => o === p || inside.some((i) => o === p + i.slice(folder.length)))) return { error: "Files with those names already exist there." };
      return { name: p };
    },
  });
  if (!target || target === folder) return;
  if (state.collapsed.delete(folder)) state.collapsed.add(target);
  await movePaths(inside.map((p) => [p, target + p.slice(folder.length)]));
  toast(`Folder renamed to ${target}`);
}

async function removePaths(paths) {
  for (const path of paths) {
    clearTimeout(state.saveTimers.get(path));
    state.saveTimers.delete(path);
    state.files.delete(path);
    state.docs.delete(path);
  }
  await store.removeFiles(state.project.id, paths);
  const wasActive = paths.includes(state.active);
  state.tabs = state.tabs.filter((t) => !paths.includes(t));
  if (wasActive) {
    if (state.tabs.length) openFile(state.tabs[state.tabs.length - 1]);
    else showNoFile();
  }
  renderTabs();
  renderFileList();
  saveSession();
}

async function deleteFile(path) {
  const ok = await confirmDialog({
    title: `Delete ${baseName(path)}?`,
    text: "The file will be removed from this browser. This can't be undone. You can download it first if you want a copy.",
    okLabel: "Delete",
  });
  if (ok) await removePaths([path]);
}

async function deleteFolder(folder) {
  const inside = sortedPaths().filter((p) => p.startsWith(folder + "/"));
  const ok = await confirmDialog({
    title: `Delete the folder ${folder}?`,
    text: `The folder and the ${inside.length} file${inside.length === 1 ? "" : "s"} in it will be removed from this browser. This can't be undone.`,
    okLabel: "Delete folder",
  });
  if (ok) await removePaths(inside);
}

function fileBlob(path) {
  const content = liveContent(path);
  return typeof content === "string" ? new Blob([content], { type: "text/plain;charset=utf-8" }) : new Blob([content], { type: mimeOf(path) });
}

async function exportFile(path) {
  await flushSave(path);
  downloadBlob(fileBlob(path), baseName(path));
}

/* ---------- viewer for images, tables and other data files ---------- */

let viewerUrl = null;

function parseCsvPreview(text, maxRows = 200, maxCols = 40) {
  const sep = (text.split("\n", 1)[0].match(/\t/g) || []).length > (text.split("\n", 1)[0].match(/,/g) || []).length ? "\t" : ",";
  const rows = [];
  for (const line of text.split(/\r?\n/)) {
    if (!line.trim()) continue;
    if (rows.length >= maxRows) break;
    const cells = [];
    let cur = "";
    let quoted = false;
    for (let i = 0; i < line.length; i++) {
      const c = line[i];
      if (quoted) {
        if (c === '"' && line[i + 1] === '"') (cur += '"'), i++;
        else if (c === '"') quoted = false;
        else cur += c;
      } else if (c === '"') quoted = true;
      else if (c === sep) (cells.push(cur), (cur = ""));
      else cur += c;
    }
    cells.push(cur);
    rows.push(cells.slice(0, maxCols));
  }
  return rows;
}

function viewFile(path) {
  const f = state.files.get(path);
  if (!f) return;
  const content = liveContent(path);
  const body = $("viewer-body");
  body.innerHTML = "";
  if (viewerUrl) URL.revokeObjectURL(viewerUrl);
  viewerUrl = null;
  const size = typeof content === "string" ? new Blob([content]).size : content.length;
  $("viewer-title").textContent = path;
  $("viewer-meta").textContent = formatSize(size);
  if (isImageName(path)) {
    viewerUrl = URL.createObjectURL(fileBlob(path));
    body.append(el("img", { src: viewerUrl, alt: path, class: "viewer-img" }));
  } else if (typeof content === "string" && /\.(csv|tsv|dat)$/i.test(path)) {
    const rows = parseCsvPreview(content);
    const total = content.split(/\r?\n/).filter((l) => l.trim()).length;
    body.append(
      el("div", { class: "viewer-table" }, el("table", { class: "data-table" }, el("tbody", {}, rows.map((r) => el("tr", {}, r.map((c) => el("td", {}, c))))))),
      total > rows.length ? el("p", { class: "muted small" }, `Showing the first ${rows.length} of ${total} rows.`) : null
    );
  } else if (typeof content === "string") {
    body.append(el("pre", { class: "var-body" }, content.length > 200000 ? content.slice(0, 200000) + "\n…" : content));
  } else {
    body.append(
      el(
        "p",
        { class: "muted" },
        /\.mat$/i.test(path)
          ? "This is a MAT data file. Read it in your code with load('" + baseName(path) + "')."
          : "This is a binary file, so it can't be shown as text. Your code can read it, and you can download it."
      )
    );
  }
  $("viewer-edit").hidden = typeof content !== "string";
  $("viewer-edit").onclick = () => {
    $("dlg-viewer").close();
    openFile(path);
  };
  $("viewer-download").onclick = () => exportFile(path);
  if (!$("dlg-viewer").open) $("dlg-viewer").showModal();
}

/* ---------- engine file sync ---------- */

/** What the worker needs to mirror this project into its working folder (see worker.js). */
function fsSpec() {
  const files = [];
  for (const path of state.files.keys()) {
    const content = liveContent(path);
    if (!state.fsReset && state.sent.get(path) === content) files.push({ path });
    else {
      files.push({ path, content });
      state.sent.set(path, content);
    }
  }
  const spec = { reset: state.fsReset, files };
  state.fsReset = false;
  return spec;
}

/** Files the code wrote: add them to the project so they show up in the tree. */
async function applyWritten(files) {
  if (!files) return;
  for (const p of files.missing || []) state.sent.delete(p);
  const written = (files.written || []).map((w) => ({ path: w.path, content: toStored(w.path, w.content), size: w.content.length, existed: state.files.has(w.path) }));
  if (written.length) {
    await writeFiles(written.map((w) => ({ path: w.path, content: w.content })));
    for (const w of written) {
      expandTo(w.path);
      state.sent.set(w.path, liveContent(w.path));
    }
    renderFileList();
    const note = el("div", { class: "entry file-note" }, written.length === 1 ? "Your code saved a file: " : `Your code saved ${written.length} files: `);
    written.slice(0, 8).forEach((w, i) => {
      if (i) note.append(", ");
      note.append(el("button", { class: "linklike", type: "button", title: `Open ${w.path}`, onclick: () => openFile(w.path) }, w.path), ` (${formatSize(w.size)})`);
    });
    if (written.length > 8) note.append(` and ${written.length - 8} more`);
    note.append(" → see the Files panel");
    appendConsole(note);
  }
  if ((files.skipped || []).length) {
    printInfo(
      "Not copied into your project (over 5 MB each or 20 MB together): " +
        files.skipped.map((s) => `${s.path} (${formatSize(s.size)})`).join(", ") +
        ". These files exist only until the engine restarts.",
      "note"
    );
  }
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

const RUN_BUTTONS = ["btn-run-section", "btn-run-sel", "btn-step", "btn-publish"];

function setRunning(running, label = "Running") {
  $("btn-run").hidden = running;
  for (const id of RUN_BUTTONS) $(id).disabled = running || !engine.ready;
  $("btn-stop").hidden = !running;
  clearInterval(state.runTimer);
  if (running) {
    state.runStarted = performance.now();
    setStatus("busy", `${label}…`);
    state.runTimer = setInterval(() => {
      const s = (performance.now() - state.runStarted) / 1000;
      setStatus("busy", `${label}… ${s.toFixed(0)} s`);
    }, 1000);
  } else if (engine.ready) {
    setStatus("ready", "Ready");
  }
}

function setStatus(kind, text) {
  $("status").dataset.state = kind;
  $("status-text").textContent = text;
}

/** True when a new run may start; otherwise tells the user why not. */
function canRun() {
  if (!engine.ready) {
    toast("The engine is still loading. Please wait a moment.");
    return false;
  }
  if (engine.busy) {
    toast("Code is already running. Press Stop to cancel it.");
    return false;
  }
  return true;
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
  if (!isText(file)) return;
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
 * Where an error happened, as { file, line } in project terms.
 * ctx: { file, lineOffset, codeLines, tailStart } describes the code that was sent.
 */
function locateError(res, ctx) {
  let line = errorLineOf(res);
  if (!line) return { file: null, line: null };
  if (res.error_file && state.files.has(res.error_file)) return { file: res.error_file, line };
  if (!ctx.file) return { file: null, line };
  if (ctx.codeLines && line > ctx.codeLines && ctx.tailStart != null) return { file: ctx.file, line: ctx.tailStart + (line - ctx.codeLines) };
  return { file: ctx.file, line: line + (ctx.lineOffset || 0) };
}

/**
 * Run code in the engine.
 * ctx: { kind: "file"|"section"|"selection"|"command", file, lineOffset, label, tail, tailStart }
 */
async function runCode(code, ctx) {
  if (!canRun()) return null;
  app.exitStep && app.exitStep();
  clearErrorMark();
  if (ctx.kind === "command") printEcho(code);
  else printEcho(code, ctx.label);
  setRunning(true);
  let res;
  const sent = ctx.tail ? code + "\n" + ctx.tail : code;
  const runCtx = { ...ctx, codeLines: ctx.tail ? code.split("\n").length : 0 };
  try {
    res = await engine.run(sent, { fs: fsSpec(), cwd: ctx.kind === "command" ? undefined : dirName(ctx.file || "") });
  } catch (e) {
    setRunning(false);
    if (String(e.message) !== "Stopped.") printError({ message: e.message });
    return null;
  }
  setRunning(false);
  printOut(res.output);
  if (res.error) {
    const err = { message: res.error, line: null, where: "", file: null, code: ctx.kind === "command" ? code : null };
    const at = locateError(res, runCtx);
    if (at.file) {
      err.line = at.line;
      err.file = at.file;
      err.where = at.file;
      markErrorLine(at.file, at.line);
      if (state.active === at.file) editor.scrollIntoView({ line: at.line - 1, ch: 0 }, 80);
    } else if (ctx.kind === "command") {
      err.where = "Command Window";
      err.line = at.line && code.includes("\n") ? at.line : null;
    }
    printError(err);
    if (res.internal) console.warn(res.internal);
    state.lastError = err;
  } else {
    state.lastError = null;
  }
  assistant.setLastError(state.lastError);
  if (res.figures && res.figures.length) addFigures(res.figures, ctx);
  if (res.animations && res.animations.length) addFigures(res.animations, ctx, "gif");
  if (res.variables) renderVariables(res.variables);
  await applyWritten(res.files);
  return res;
}

/** The active file if it is a script that can be run; otherwise explains why not. */
function runnableFile() {
  if (!state.active) {
    toast("Open a file to run it.");
    return null;
  }
  const name = state.active;
  const code = editor.getValue();
  flushSave(name);
  if (!name.endsWith(".m")) {
    toast("Only .m files can be run.", "error");
    return null;
  }
  if (isFunctionFile(code)) {
    const fn = baseName(name).replace(/\.m$/, "");
    printInfo(
      `${baseName(name)} is a function file, so it needs to be called with inputs. Call it from the command window or a script, for example:  ${fn}(...)`,
      "note"
    );
    $("console-input").value = fn + "(";
    $("console-input").focus();
    return null;
  }
  return { name, code };
}

function flashEditor() {
  editor.getWrapperElement().classList.remove("cm-run-flash");
  void editor.getWrapperElement().offsetWidth;
  editor.getWrapperElement().classList.add("cm-run-flash");
}

async function runActiveFile() {
  const f = runnableFile();
  if (!f) return null;
  flashEditor();
  return runCode(f.code, { kind: "file", file: f.name, lineOffset: 0, label: `run ${f.name}` });
}

async function runSection({ advance = false } = {}) {
  const f = runnableFile();
  if (!f) return null;
  const split = splitSections(f.code);
  if (!split.hasSections) return runActiveFile();
  const cur = editor.getCursor().line;
  const i = sectionAt(split, cur);
  if (i < 0) {
    toast("The cursor is in the functions at the end of the file. Put it in a section to run that section.");
    return null;
  }
  const s = split.sections[i];
  const title = s.title ? `“${s.title}”` : `${i + 1}`;
  const label = `run section ${title} (lines ${s.start + 1}–${s.end}) of ${f.name}`;
  if (advance) {
    const next = split.sections[i + 1];
    if (next) {
      const line = Math.min(next.start + (next.header ? 1 : 0), editor.lineCount() - 1);
      editor.setCursor({ line, ch: 0 });
      editor.scrollIntoView({ line, ch: 0 }, 100);
    }
  }
  return runCode(sectionCode(split, i), { kind: "section", file: f.name, lineOffset: s.start, label, tail: tailCode(split), tailStart: split.tailStart });
}

/** Ctrl+Enter: the section under the cursor when the file has sections, otherwise the file. */
function runSmart() {
  return state.sectionView && state.sectionView.split.hasSections ? runSection() : runActiveFile();
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

/* ================================================================== sections in the editor */

function clearSectionMarks() {
  const v = state.sectionView;
  if (!v) return;
  for (const [h, where, cls] of v.marks) {
    try {
      v.doc.removeLineClass(h, where, cls);
    } catch {
      /* line is gone */
    }
  }
  v.marks = [];
}

/** Mark "%%" lines and tint the section that holds the cursor. */
function updateSections(force = false) {
  if (!editor) return;
  const doc = editor.getDoc();
  const isCode = !!state.active && state.active.endsWith(".m") && !editor.getWrapperElement().hidden;
  let v = state.sectionView;
  if (!isCode) {
    clearSectionMarks();
    state.sectionView = null;
    $("btn-run-section").hidden = true;
    return;
  }
  const gen = doc.changeGeneration();
  const fresh = force || !v || v.doc !== doc || v.gen !== gen;
  const split = fresh ? splitSections(doc.getValue()) : v.split;
  const current = split.hasSections ? sectionAt(split, editor.getCursor().line) : -1;
  if (!fresh && current === v.current) return;
  editor.operation(() => {
    clearSectionMarks();
    v = state.sectionView = { doc, gen, split, current, marks: [] };
    if (!split.hasSections) return;
    split.sections.forEach((s, i) => {
      if (s.header) v.marks.push([doc.addLineClass(s.start, "wrap", "cm-section-head"), "wrap", "cm-section-head"]);
      if (i === current) {
        for (let l = s.start; l < s.end && l < doc.lineCount(); l++) v.marks.push([doc.addLineClass(l, "wrap", "cm-section-cur"), "wrap", "cm-section-cur"]);
      }
    });
  });
  $("btn-run-section").hidden = !split.hasSections;
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
      if (!n.endsWith(".m")) continue;
      const base = baseName(n).replace(/\.m$/, "");
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
  if (app.stepActive && app.stepActive()) return; // the step-through view owns the table for now
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

/** Add figures (base64 PNGs) or animations (base64 GIFs, type "gif") to the Plots panel. */
function addFigures(figs, ctx, type = "png") {
  const box = $("plots");
  $("plots-empty").hidden = true;
  const source = ctx.kind === "command" ? "Command Window" : baseName(ctx.file || "figure");
  const time = new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
  const word = type === "gif" ? "Animation" : "Figure";
  figs.forEach((b64, i) => {
    state.figureCount++;
    const n = state.figureCount;
    const url = `data:image/${type};base64,` + b64;
    const filename = `${(source || "figure").replace(/\.m$/, "").replace(/\W+/g, "_")}_${n}.${type}`;
    const img = el("img", { src: url, alt: `${word} ${n} from ${source}`, loading: "lazy", onclick: () => enlargeFigure(url, filename) });
    const card = el(
      "figure",
      { class: "plot-card new" + (type === "gif" ? " animation" : ""), style: "margin:0", "data-type": type },
      img,
      el(
        "figcaption",
        { class: "plot-meta" },
        el("span", {}, `${word} ${n}${figs.length > 1 ? ` (${i + 1}/${figs.length})` : ""} · ${source} · ${time}`),
        el("button", { class: "btn icon sm ghost", title: "Enlarge", "aria-label": "Enlarge", onclick: () => enlargeFigure(url, filename) }, icon("expand")),
        el(
          "a",
          { class: "btn icon sm ghost", title: `Download ${type.toUpperCase()}`, "aria-label": `Download ${type.toUpperCase()}`, href: url, download: filename },
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
  const what = type === "gif" ? `animation${figs.length > 1 ? "s" : ""}` : `figure${figs.length > 1 ? "s" : ""}`;
  const note = el("div", { class: "entry fig-note", onclick: () => $("plots").scrollIntoView({ block: "nearest" }) },
    `${figs.length} ${what} drawn → see the Plots panel`);
  appendConsole(note);
}

function updatePlotsEmpty() {
  $("plots-empty").hidden = !!$("plots").querySelector(".plot-card");
}

function enlargeFigure(url, filename) {
  $("figure-big").src = url;
  $("figure-dl").href = url;
  $("figure-dl").download = filename;
  $("figure-dl").lastChild.textContent = /\.gif$/.test(filename) ? "Download GIF" : "Download PNG";
  $("dlg-figure").showModal();
}

/* ================================================================== layout & resizing */

const SIZES = prefs.get("sizes", {});

function applySizes() {
  const root = document.documentElement.style;
  if (SIZES.files) root.setProperty("--files-w", SIZES.files + "px");
  if (SIZES.side) root.setProperty("--side-w", SIZES.side + "px");
  if (SIZES.ai) root.setProperty("--ai-w", SIZES.ai + "px");
  if (SIZES.asg) root.setProperty("--asg-w", SIZES.asg + "px");
  if (SIZES.console) root.setProperty("--console-h", SIZES.console + "px");
  if (SIZES.workspace) root.setProperty("--workspace-h", SIZES.workspace + "px");
}

function setupGutters() {
  const limits = { files: [140, 420], side: [220, 700], ai: [280, 720], asg: [220, 640], console: [90, 900], workspace: [80, 900] };
  document.querySelectorAll(".gutter").forEach((g) => {
    const key = g.dataset.resize;
    const target = {
      files: () => $("files-panel"),
      side: () => $("side"),
      ai: () => $("ai-panel"),
      asg: () => $("asg-panel"),
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
      const prop = { files: "--files-w", side: "--side-w", ai: "--ai-w", asg: "--asg-w", console: "--console-h", workspace: "--workspace-h" }[key];
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
  for (const id of RUN_BUTTONS) $(id).disabled = false;
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

/* ================================================================== welcome */

/** Make sure an example file exists in the current project (never overwrites) and open it. */
async function openExample(name) {
  const ex = EXAMPLES.find((e) => e.name === name);
  if (!ex) return;
  if (state.project.kind !== "files" || state.project.temp) await switchProject((state.projects.find((p) => p.id === DEFAULT_PROJECT) || state.projects.find((p) => !p.temp)).id);
  if (!state.files.has(name)) await createFile(name, ex.content, { open: false });
  openFile(name);
}

function showWelcome() {
  $("dlg-welcome").showModal();
}

function setupWelcome() {
  const dlg = $("dlg-welcome");
  dlg.addEventListener("close", () => prefs.set("welcomed", 2));
  const go = (fn) => () => {
    dlg.close();
    fn();
  };
  $("welcome-examples").addEventListener("click", go(() => openExample("welcome.m")));
  $("welcome-github").addEventListener("click", go(() => app.openGithubDialog()));
  $("welcome-assignment").addEventListener("click", go(() => app.openExampleAssignment()));
  $("welcome-step").addEventListener(
    "click",
    go(async () => {
      await openExample("step_demo.m");
      if (engine.ready && !engine.busy) app.startStep();
      else toast("When the engine has loaded, press “Step through” to walk through this file line by line.");
    })
  );
}

/* ================================================================== setup */

function setupEditor() {
  const mod = isMac ? "Cmd" : "Ctrl";
  editor = CodeMirror($("editor"), {
    value: "",
    mode: "text/x-octave",
    lineNumbers: true,
    gutters: ["bp-gutter", "CodeMirror-linenumbers"],
    indentUnit: 4,
    tabSize: 4,
    indentWithTabs: false,
    matchBrackets: true,
    autoCloseBrackets: "()[]{}''\"\"",
    styleActiveLine: true,
    lineWrapping: false,
    viewportMargin: 30,
    extraKeys: {
      [`${mod}-Enter`]: () => runSmart(),
      [`Shift-${mod}-Enter`]: () => runSelection(),
      "Alt-Enter": () => runSection({ advance: true }),
      F5: () => runActiveFile(),
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
  let sectionTimer = null;
  editor.on("cursorActivity", () => updateSections());
  editor.on("changes", () => {
    clearTimeout(sectionTimer);
    sectionTimer = setTimeout(() => updateSections(), 120);
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
  consoleOut().addEventListener("mouseup", (e) => {
    if (e.target.closest("button, a")) return;
    if (!getSelection().toString() && !input.disabled) input.focus();
  });
}

function setupButtons() {
  $("btn-run").addEventListener("click", runActiveFile);
  $("btn-run-section").addEventListener("click", () => runSection());
  $("btn-run-sel").addEventListener("click", runSelection);
  $("btn-stop").addEventListener("click", stopRun);
  $("btn-new").addEventListener("click", () => newFile());
  $("btn-import").addEventListener("click", () => $("file-input").click());
  $("btn-clear-console").addEventListener("click", clearConsole);
  $("btn-reset").addEventListener("click", resetWorkspace);
  $("btn-clear-plots").addEventListener("click", () => {
    $("plots").querySelectorAll(".plot-card").forEach((c) => c.remove());
    updatePlotsEmpty();
  });
  $("btn-theme").addEventListener("click", toggleTheme);
  $("btn-keys").addEventListener("click", () => $("dlg-keys").showModal());
  $("btn-about").addEventListener("click", () => $("dlg-about").showModal());
  $("foot-about").addEventListener("click", () => $("dlg-about").showModal());
  $("foot-welcome").addEventListener("click", showWelcome);
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
    const inField = document.activeElement === $("console-input") || document.activeElement === $("ai-input");
    if (document.querySelector("dialog[open]")) return;
    if (e.altKey && (e.key === "n" || e.key === "N" || e.code === "KeyN")) {
      e.preventDefault();
      newFile();
    } else if (mod && e.key === "Enter" && !editor.hasFocus() && !inField) {
      e.preventDefault();
      e.shiftKey ? runSelection() : runSmart();
    } else if (e.key === "F5" && !e.ctrlKey && !e.metaKey && !e.shiftKey && !editor.hasFocus()) {
      e.preventDefault();
      runActiveFile();
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

/** Links such as #code=…, #project=…, #gh=owner/repo, #assignment=url (see docs/WEB.md). */
async function handleHash() {
  const params = parseHash(location.hash);
  try {
    if (params.assignment || params.assignmentdata) return await app.openAssignmentFromLink(params);
    if (params.gh) return await app.openGithubLink(params.gh);
    if (params.code || params.project) return await app.openSharedLink(params);
  } catch (e) {
    console.warn(e);
    toast(e.message || String(e), "error");
  }
  return false;
}

function clearHash() {
  if (location.hash) history.replaceState(null, "", location.pathname + location.search);
}

Object.defineProperty(app, "editor", { get: () => editor });
Object.defineProperty(app, "engine", { get: () => engine });
Object.assign(app, {
  baseName,
  dirName,
  isText,
  isFunctionFile,
  sortedPaths,
  liveContent,
  fileBlob,
  validatePath,
  normalizePath,
  getDoc,
  openFile,
  viewFile,
  createFile,
  writeFiles,
  removePaths,
  flushAll,
  renderFileList,
  renderProjectHead,
  switchProject,
  createProject,
  deleteProject,
  saveProjectRecord,
  uniqueProjectName,
  sortProjects,
  fsSpec,
  canRun,
  runCode,
  runActiveFile,
  runnableFile,
  stopRun,
  setRunning,
  setStatus,
  printInfo,
  printOut,
  printError,
  printEcho,
  appendConsole,
  addFigures,
  renderVariables,
  locateError,
  markErrorLine,
  clearErrorMark,
  clearHash,
  showWelcome,
  openExample,
});

async function main() {
  updateThemeIcon();
  applySizes();
  setupEditor();
  setupConsole();
  setupButtons();
  setupGutters();
  setupWelcome();

  // Start the engine as early as possible; it loads while the files are read.
  engine = new Engine();
  engine.addEventListener("progress", (e) => onProgress(e.detail));
  engine.addEventListener("ready", (e) => onReady(e.detail));
  engine.addEventListener("fatal", (e) => onFatal(e.detail));
  engine.addEventListener("restarting", () => {
    document.body.dataset.engine = "loading";
    $("btn-run").disabled = true;
    for (const id of RUN_BUTTONS) $(id).disabled = true;
    setStatus("loading", "Restarting engine…");
    // The new engine starts with an empty working folder.
    state.sent.clear();
    state.fsReset = true;
  });

  assistant = new Assistant({
    getContext: assistantContext,
    insertCode: insertIntoEditor,
    replaceFile: replaceFileWith,
    runCode: (code) => runCode(code, { kind: "command" }),
    toast,
  });
  if (prefs.get("aiOpen", false)) showAssistant(true);

  initProjects(app);
  initShare(app);
  initPublish(app);
  initStepper(app);
  initAssignments(app);

  await loadProjects();
  document.body.dataset.files = "ready";
  const fromLink = await handleHash();
  window.addEventListener("hashchange", () => handleHash());
  if (!fromLink && prefs.get("welcomed", 0) < 2) showWelcome();
}

main().catch((e) => {
  console.error(e);
  toast("mylab failed to start: " + e.message, "error");
});

// Exposed for automated tests and debugging.
window.mylab = { app, state, get engine() { return engine; }, get editor() { return editor; }, runCode, openFile, createFile };
