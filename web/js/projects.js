// Projects: the switcher menu, .zip import/export, "Open folder", drag-and-drop and "Open from GitHub".

import { $, el, toast, downloadBlob, promptDialog, confirmDialog, summaryDialog, openMenu, dateStamp } from "./ui.js";
import { toStored, formatSize } from "./codec.js";
import { parseGithub, githubSpecText, fetchGithub } from "./github.js";
import { DEFAULT_PROJECT } from "./storage.js";

export const IMPORT_EXT = /\.(m|txt|csv|tsv|dat|json|md|xml|png|jpe?g|gif|bmp|mat)$/i;
const MAX_FILE = 5 * 1024 * 1024;
const MAX_TOTAL = 30 * 1024 * 1024;
const MAX_COUNT = 500;
const IGNORED_DIRS = new Set(["__MACOSX", "node_modules", "__pycache__"]);

/** "a\\b/./c.m" -> "a/b/c.m"; null for hidden files, junk folders and unsafe paths. */
function cleanImportPath(path) {
  const parts = String(path).replace(/\\/g, "/").split("/").filter((s) => s && s !== ".");
  if (!parts.length || parts.some((s) => s === ".." || s.startsWith(".") || IGNORED_DIRS.has(s))) return null;
  return parts.map((s) => s.replace(/[:*?"<>|]/g, "_")).join("/");
}

/** Remove a folder that wraps everything (as in most .zip files and picked folders). */
function stripCommonFolder(entries) {
  if (!entries.length) return "";
  const first = entries[0].path.split("/")[0];
  if (entries.every((e) => e.path.includes("/") && e.path.split("/")[0] === first)) {
    for (const e of entries) e.path = e.path.slice(first.length + 1);
    return first;
  }
  return "";
}

/**
 * Apply the import rules. entries: [{ path, size, read: async () => Uint8Array }]
 * Returns { files: [{ path, content }], skipped: [{ path, reason }] }.
 */
async function gather(entries) {
  const files = [];
  const skipped = [];
  let total = 0;
  for (const e of entries) {
    if (!IMPORT_EXT.test(e.path)) {
      skipped.push({ path: e.path, reason: "file type not used by mylab" });
    } else if (e.size > MAX_FILE) {
      skipped.push({ path: e.path, reason: `larger than 5 MB (${formatSize(e.size)})` });
    } else if (files.length >= MAX_COUNT) {
      skipped.push({ path: e.path, reason: `more than ${MAX_COUNT} files` });
    } else if (total + e.size > MAX_TOTAL) {
      skipped.push({ path: e.path, reason: "over the 30 MB total" });
    } else {
      try {
        const bytes = await e.read();
        if (bytes.length > MAX_FILE) {
          skipped.push({ path: e.path, reason: `larger than 5 MB (${formatSize(bytes.length)})` });
          continue;
        }
        total += bytes.length;
        files.push({ path: e.path, content: toStored(e.path, bytes) });
      } catch {
        skipped.push({ path: e.path, reason: "could not be read" });
      }
    }
  }
  return { files, skipped };
}

const fileEntry = (path, file) => ({ path, size: file.size, read: async () => new Uint8Array(await file.arrayBuffer()) });

/** The file to show first: main.m, else the shallowest script, else a README. */
function pickMain(files) {
  const scripts = files.filter((f) => f.path.endsWith(".m")).map((f) => f.path);
  const depth = (p) => p.split("/").length;
  scripts.sort((a, b) => depth(a) - depth(b) || a.localeCompare(b));
  return scripts.find((p) => /(^|\/)main\.m$/i.test(p)) || scripts[0] || (files.find((f) => /readme\.md$/i.test(f.path)) || files.find((f) => typeof f.content === "string") || {}).path || null;
}

export function initProjects(app) {
  const { state, store } = app;

  function reportImport(what, { files, skipped }) {
    const n = files.length;
    if (skipped.length) {
      summaryDialog({
        title: `${what}: ${n} file${n === 1 ? "" : "s"} imported, ${skipped.length} skipped`,
        text: "mylab imports .m files and common data files (.csv, .txt, .json, .md, .mat, images) up to 5 MB each and 30 MB in total.",
        groups: [{ title: "Skipped", items: skipped }, { title: "Imported", items: files.map((f) => ({ path: f.path })) }],
      });
    } else {
      toast(`${what}: ${n} file${n === 1 ? "" : "s"} imported.`);
    }
  }

  async function importAsProject(name, entries, what, extra = {}) {
    const cleaned = [];
    for (const e of entries) {
      const p = cleanImportPath(e.path);
      if (p) cleaned.push({ ...e, path: p });
    }
    const top = stripCommonFolder(cleaned);
    const result = await gather(cleaned);
    if (!result.files.length) {
      summaryDialog({
        title: `${what}: nothing to import`,
        text: "No .m files or data files were found. mylab imports .m, .csv, .txt, .json, .md, .mat and image files up to 5 MB each.",
        groups: [{ title: "Skipped", items: result.skipped }],
      });
      return null;
    }
    const project = await app.createProject({ name: name || top || "Imported project", files: result.files, open: pickMain(result.files), ...extra });
    reportImport(what, result);
    return project;
  }

  /* ---------- loose files into the current project ---------- */

  async function importFiles(fileList) {
    const entries = [...fileList].map((f) => fileEntry(f.name.replace(/[\\/:*?"<>|]/g, "_"), f));
    const result = await gather(entries);
    const accepted = [];
    for (const f of result.files) {
      if (state.files.has(f.path)) {
        const ok = await confirmDialog({ title: `Replace ${f.path}?`, text: `A file named ${f.path} already exists. Replace it with the imported one?`, okLabel: "Replace" });
        if (!ok) continue;
      }
      accepted.push(f);
    }
    await app.writeFiles(accepted);
    const last = accepted.filter((f) => typeof f.content === "string").pop();
    if (last) app.openFile(last.path);
    if (result.skipped.length) {
      for (const s of result.skipped.slice(0, 3)) toast(`Skipped ${s.path}: ${s.reason}.`, "error");
    }
    if (accepted.length) toast(`Imported ${accepted.length} file${accepted.length > 1 ? "s" : ""}.`);
  }

  /* ---------- .zip ---------- */

  function zipReady() {
    if (typeof JSZip === "undefined") {
      toast("The zip library hasn't loaded yet. Check your connection and try again.", "error");
      return false;
    }
    return true;
  }

  async function exportZip() {
    await app.flushAll();
    if (!zipReady()) return;
    const zip = new JSZip();
    for (const path of app.sortedPaths()) zip.file(path, app.liveContent(path));
    const blob = await zip.generateAsync({ type: "blob", compression: "DEFLATE" });
    const slug = state.project.id === DEFAULT_PROJECT ? "files" : state.project.name.replace(/[^A-Za-z0-9_-]+/g, "-").replace(/^-|-$/g, "") || "project";
    downloadBlob(blob, `mylab-${slug}-${dateStamp()}.zip`);
  }

  async function importZip(file) {
    if (!zipReady()) return null;
    let zip;
    try {
      zip = await JSZip.loadAsync(file);
    } catch {
      toast(`${file.name} is not a .zip file that can be read.`, "error");
      return null;
    }
    const entries = [];
    zip.forEach((path, entry) => {
      if (!entry.dir) entries.push({ path, size: (entry._data && entry._data.uncompressedSize) || 0, read: () => entry.async("uint8array") });
    });
    return importAsProject(file.name.replace(/\.zip$/i, ""), entries, "Import .zip");
  }

  /* ---------- folders ---------- */

  function importFolderInput(fileList) {
    const files = [...fileList];
    if (!files.length) return null;
    const entries = files.map((f) => fileEntry(f.webkitRelativePath || f.name, f));
    return importAsProject("", entries, "Open folder");
  }

  function readEntries(dirEntry) {
    const reader = dirEntry.createReader();
    return new Promise((resolve, reject) => {
      const all = [];
      const next = () =>
        reader.readEntries((batch) => {
          if (!batch.length) return resolve(all);
          all.push(...batch);
          next();
        }, reject);
      next();
    });
  }

  async function walkEntry(entry, prefix, out) {
    if (out.length > 5000) return;
    if (entry.isFile) {
      const file = await new Promise((resolve, reject) => entry.file(resolve, reject));
      out.push(fileEntry(prefix + entry.name, file));
    } else if (entry.isDirectory) {
      if (entry.name.startsWith(".") || IGNORED_DIRS.has(entry.name)) return;
      for (const child of await readEntries(entry)) await walkEntry(child, prefix + entry.name + "/", out);
    }
  }

  async function handleDrop(dataTransfer) {
    const roots = [...(dataTransfer.items || [])].map((it) => (it.webkitGetAsEntry ? it.webkitGetAsEntry() : null)).filter(Boolean);
    const files = [...dataTransfer.files];
    if (roots.some((r) => r.isDirectory)) {
      const out = [];
      try {
        for (const r of roots) await walkEntry(r, "", out);
      } catch {
        toast("The dropped folder could not be read.", "error");
        return;
      }
      const dirs = roots.filter((r) => r.isDirectory);
      await importAsProject(dirs.length === 1 && roots.length === 1 ? dirs[0].name : "", out, "Open folder");
      return;
    }
    if (files.length === 1 && /\.zip$/i.test(files[0].name)) return void (await importZip(files[0]));
    if (files.length === 1 && /\.json$/i.test(files[0].name) && files[0].size < MAX_FILE) {
      const text = await files[0].text();
      if (/"format"\s*:\s*"mylab-assignment-/.test(text)) return void (await app.openAssignmentText(text, { source: files[0].name }));
    }
    await importFiles(files);
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
      handleDrop(e.dataTransfer);
    });
  }

  /* ---------- GitHub ---------- */

  let githubAbort = null;

  function githubStatus(text, kind = "") {
    const s = $("github-status");
    s.textContent = text;
    s.className = "small " + (kind === "error" ? "error-text" : "muted");
    s.hidden = !text;
  }

  async function downloadGithub(spec) {
    const source = "gh:" + githubSpecText(spec);
    githubAbort = new AbortController();
    $("github-open").disabled = true;
    $("github-progress").hidden = false;
    $("github-bar").style.width = "4%";
    githubStatus(`Asking GitHub for the file list of ${spec.owner}/${spec.repo}…`);
    try {
      const result = await fetchGithub(spec, {
        signal: githubAbort.signal,
        onProgress: ({ phase, done, total }) => {
          if (phase === "download") {
            $("github-bar").style.width = 8 + (92 * done) / Math.max(1, total) + "%";
            githubStatus(`Downloading files… ${done} of ${total}`);
          }
        },
      });
      const name = spec.path ? `${spec.repo}/${spec.path.split("/").pop()}` : spec.repo;
      await app.createProject({
        name,
        kind: "github",
        meta: { source, url: `https://github.com/${spec.owner}/${spec.repo}`, ref: spec.ref, path: spec.path, fetched: Date.now() },
        files: result.files,
        open: pickMain(result.files),
      });
      if ($("dlg-github").open) $("dlg-github").close();
      const notes = [];
      if (result.truncated) notes.push("This repository is very large, so GitHub sent only part of its file list.");
      if (result.other) notes.push(`${result.other} other file${result.other === 1 ? " was" : "s were"} left out (only .m, .csv, .txt, .json, .md, .png, .jpg and .mat are imported).`);
      if (result.skipped.length || result.truncated) {
        summaryDialog({
          title: `Opened ${spec.owner}/${spec.repo}: ${result.files.length} files, ${result.skipped.length} skipped`,
          text: notes.join(" "),
          groups: [{ title: "Skipped", items: result.skipped }],
        });
      } else {
        toast(`Opened ${spec.owner}/${spec.repo}: ${result.files.length} file${result.files.length === 1 ? "" : "s"}.`);
      }
      return true;
    } catch (e) {
      if (e.name === "AbortError") return false;
      if (!$("dlg-github").open) openGithubDialog(githubSpecText(spec));
      githubStatus(e.message || String(e), "error");
      return false;
    } finally {
      githubAbort = null;
      $("github-open").disabled = false;
      $("github-progress").hidden = true;
    }
  }

  function openGithubDialog(value = "") {
    if (typeof value !== "string") value = "";
    $("github-url").value = value;
    githubStatus("");
    $("github-progress").hidden = true;
    $("github-open").disabled = false;
    if (!$("dlg-github").open) $("dlg-github").showModal();
    $("github-url").focus();
  }

  async function submitGithub() {
    const spec = parseGithub($("github-url").value);
    if (!spec) {
      githubStatus("That doesn't look like a GitHub repository. Paste an address such as https://github.com/owner/repository.", "error");
      return;
    }
    const branch = $("github-ref").value.trim();
    if (branch) spec.ref = branch;
    const existing = state.projects.find((p) => p.meta && p.meta.source === "gh:" + githubSpecText(spec));
    if (existing) {
      const fresh = await confirmDialog({
        title: "You already opened this repository",
        text: `The project “${existing.name}” was made from it, and may hold your changes. Download a fresh copy as a new project, or cancel and pick “${existing.name}” from the project list.`,
        okLabel: "Download a fresh copy",
        danger: false,
      });
      if (!fresh) return;
    }
    await downloadGithub(spec);
  }

  /** #gh=owner/repo[/sub/folder][@ref] */
  async function openGithubLink(text) {
    const spec = parseGithub(text);
    if (!spec) throw new Error("The link's GitHub address is not valid. It should look like #gh=owner/repository.");
    const existing = state.projects.find((p) => p.meta && p.meta.source === "gh:" + githubSpecText(spec));
    if (existing) {
      await app.switchProject(existing.id);
      toast(`Opened your copy of ${spec.owner}/${spec.repo}. Your changes are kept.`);
      return true;
    }
    openGithubDialog(githubSpecText(spec));
    await downloadGithub(spec);
    return true;
  }

  /* ---------- project menu ---------- */

  async function newProject() {
    const name = await promptDialog({
      title: "New project",
      text: "A project is a set of files that belong together, for example one homework or one lab.",
      value: "My project",
      okLabel: "Create",
      selectAll: true,
      validate: (v) => (v.trim() ? { name: v.trim().slice(0, 60) } : { error: "Please enter a name." }),
    });
    if (!name) return;
    await app.createProject({ name, files: [{ path: "main.m", content: `%% ${name}\n\n` }], open: "main.m" });
  }

  async function renameProject() {
    const name = await promptDialog({
      title: "Rename project",
      value: state.project.name,
      okLabel: "Rename",
      selectAll: true,
      validate: (v) => (v.trim() ? { name: v.trim().slice(0, 60) } : { error: "Please enter a name." }),
    });
    if (!name || name === state.project.name) return;
    state.project.name = name;
    await app.saveProjectRecord();
    app.sortProjects();
    app.renderProjectHead();
  }

  async function removeProject() {
    const p = state.project;
    const ok = await confirmDialog({
      title: `Delete the project “${p.name}”?`,
      text: p.temp
        ? "This temporary project will be closed."
        : `All ${state.files.size} file${state.files.size === 1 ? "" : "s"} in it will be removed from this browser. This can't be undone. Export it as a .zip first if you want a copy.`,
      okLabel: p.temp ? "Close project" : "Delete project",
    });
    if (!ok) return;
    if (p.temp) app.clearHash();
    await app.deleteProject(p.id);
  }

  function projectMenu() {
    const kindBadge = (p) => (p.temp ? "temporary" : { assignment: "assignment", github: "GitHub" }[p.kind] || "");
    openMenu(
      $("project-btn"),
      [
        { heading: "Projects" },
        ...state.projects.map((p) => ({
          label: p.name,
          checked: p.id === state.project.id,
          icon: "folder",
          badge: kindBadge(p),
          value: p.id,
          onclick: () => {
            if (p.id === state.project.id) return;
            app.clearHash(); // a link that opened another project should not come back on reload
            app.switchProject(p.id);
          },
        })),
        "sep",
        { label: "New project…", icon: "plus", id: "menu-new-project", onclick: newProject },
        { label: "Open folder…", icon: "folder", id: "menu-open-folder", onclick: () => $("folder-input").click() },
        { label: "Open from GitHub…", icon: "repo", id: "menu-open-github", onclick: () => openGithubDialog() },
        { label: "Open assignment…", icon: "task", id: "menu-open-assignment", onclick: () => app.openAssignmentDialog() },
        { label: "Import .zip…", icon: "upload", id: "menu-import-zip", onclick: () => $("zip-input").click() },
        "sep",
        { label: "Export this project as .zip", icon: "zip", id: "menu-export-zip", onclick: exportZip },
        { label: "Rename this project…", icon: "pencil", id: "menu-rename-project", onclick: renameProject },
        { label: state.project.temp ? "Close this project" : "Delete this project…", icon: "trash", danger: true, id: "menu-delete-project", onclick: removeProject },
      ],
      { id: "project-menu" }
    );
  }

  $("project-btn").addEventListener("click", projectMenu);
  $("btn-zip").addEventListener("click", exportZip);
  $("file-input").addEventListener("change", (e) => {
    importFiles([...e.target.files]);
    e.target.value = "";
  });
  $("folder-input").addEventListener("change", (e) => {
    importFolderInput(e.target.files);
    e.target.value = "";
  });
  $("zip-input").addEventListener("change", (e) => {
    if (e.target.files[0]) importZip(e.target.files[0]);
    e.target.value = "";
  });
  $("github-form").addEventListener("submit", (e) => {
    e.preventDefault();
    submitGithub();
  });
  $("github-cancel").addEventListener("click", () => {
    if (githubAbort) githubAbort.abort();
    $("dlg-github").close();
  });
  $("dlg-github").addEventListener("close", () => githubAbort && githubAbort.abort());
  setupDragDrop();

  Object.assign(app, { importFiles, importZip, importFolderInput, exportZip, openGithubDialog, openGithubLink, pickMain });
}
