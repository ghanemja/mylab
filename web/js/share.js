// Share links: the code travels inside the link itself (in the part after "#", which browsers
// never send to a server).  #code=<payload>  holds one file,  #project=<payload>  several.
// Add  &run=1  to run the main file once the engine has loaded.

import { $, toast, copyText } from "./ui.js";
import { encodePayload, decodePayload, bytesToBase64, base64ToBytes } from "./codec.js";

export const LONG_LINK = 8000;

export function initShare(app) {
  const { state } = app;
  let current = { scope: "file", link: "" };

  async function buildLink(scope, run) {
    await app.flushAll();
    let fragment;
    if (scope === "file") {
      fragment = "code=" + (await encodePayload({ n: app.baseName(state.active), c: app.liveContent(state.active) }));
    } else {
      const files = {};
      for (const path of app.sortedPaths()) {
        const content = app.liveContent(path);
        files[path] = typeof content === "string" ? content : { b: bytesToBase64(content) };
      }
      fragment = "project=" + (await encodePayload({ n: state.project.name.replace(/^Shared: /, ""), m: state.active, f: files }));
    }
    return location.origin + location.pathname + "#" + fragment + (run ? "&run=1" : "");
  }

  async function refresh() {
    const scope = document.querySelector("input[name='share-scope']:checked").value;
    const run = $("share-run").checked;
    $("share-copied").hidden = true;
    let link;
    try {
      link = await buildLink(scope, run);
    } catch (e) {
      $("share-link").value = "";
      $("share-length").textContent = "Couldn't build the link: " + e.message;
      return;
    }
    current = { scope, link };
    $("share-link").value = link;
    $("share-length").textContent = `${link.length.toLocaleString()} characters`;
    $("share-warning").hidden = link.length <= LONG_LINK;
    $("share-copy").disabled = false;
  }

  function openShareDialog() {
    const hasFile = !!state.active && app.isText(state.active);
    const fileOpt = $("share-scope-file");
    fileOpt.disabled = !hasFile;
    $("share-file-name").textContent = hasFile ? app.baseName(state.active) : "no file open";
    const n = state.files.size;
    $("share-project-name").textContent = `${state.project.name} (${n} file${n === 1 ? "" : "s"})`;
    (hasFile ? fileOpt : $("share-scope-project")).checked = true;
    $("share-run").checked = false;
    $("share-link").value = "";
    $("share-length").textContent = "";
    $("share-warning").hidden = true;
    $("dlg-share").showModal();
    refresh();
  }

  async function copyLink() {
    const ok = await copyText(current.link);
    const note = $("share-copied");
    note.hidden = false;
    note.textContent = ok ? "Copied! Paste it into a chat, an e-mail or your course page." : "Couldn't copy automatically. Select the link above and copy it by hand.";
    note.classList.toggle("error-text", !ok);
    if (!ok) $("share-link").select();
  }

  /** Open #code=… / #project=… in a temporary project. The visitor's own files are not touched. */
  async function openSharedLink(params) {
    const payload = params.code || params.project;
    const already = state.projects.find((p) => p.temp && p.meta && p.meta.payload === payload);
    let opened = already;
    if (already) {
      if (state.project.id !== already.id) await app.switchProject(already.id);
    } else {
      const data = await decodePayload(payload);
      let files;
      let main = null;
      let name;
      if (params.code) {
        const fileName = app.normalizePath(String(data.n || "shared.m")).split("/").pop() || "shared.m";
        files = [{ path: fileName, content: String(data.c ?? "") }];
        main = fileName;
        name = fileName;
      } else {
        files = [];
        for (const [path, value] of Object.entries(data.f || {})) {
          const clean = app.normalizePath(path);
          if (!clean || clean.split("/").some((s) => s === ".." || s.startsWith("."))) continue;
          files.push({ path: clean, content: typeof value === "string" ? value : base64ToBytes(String((value && value.b) || "")) });
        }
        if (!files.length) throw new Error("This link has no files in it.");
        main = typeof data.m === "string" && files.some((f) => f.path === data.m) ? data.m : app.pickMain(files);
        name = String(data.n || "project");
      }
      opened = await app.createProject({ name: "Shared: " + name, kind: "shared", temp: true, meta: { payload, name }, files, open: main });
      toast("Opened shared code in a temporary project. Use “Keep a copy” to save it.");
    }
    if (params.run === "1" || params.run === "true") {
      app.engine.whenReady().then(
        () => {
          if (state.project.id === opened.id && !app.engine.busy) app.runActiveFile();
        },
        () => {}
      );
    }
    return true;
  }

  async function keepCopy() {
    const temp = state.project;
    if (!temp.temp) return;
    await app.flushAll();
    const files = app.sortedPaths().map((path) => ({ path, content: app.liveContent(path) }));
    const name = app.uniqueProjectName((temp.meta && temp.meta.name ? String(temp.meta.name) : temp.name).replace(/\.m$/, ""));
    const open = state.active;
    app.clearHash();
    await app.createProject({ name, files, open });
    await app.deleteProject(temp.id);
    app.renderProjectHead();
    toast(`Saved as the project “${name}”. It now lives in this browser.`);
  }

  async function closeTemp() {
    const temp = state.project;
    if (!temp.temp) return;
    app.clearHash();
    await app.deleteProject(temp.id);
  }

  $("btn-share").addEventListener("click", openShareDialog);
  document.querySelectorAll("input[name='share-scope']").forEach((r) => r.addEventListener("change", refresh));
  $("share-run").addEventListener("change", refresh);
  $("share-copy").addEventListener("click", copyLink);
  $("share-link").addEventListener("focus", (e) => e.target.select());
  $("temp-keep").addEventListener("click", keepCopy);
  $("temp-close").addEventListener("click", closeTemp);

  Object.assign(app, { openShareDialog, openSharedLink, buildShareLink: buildLink, keepCopy });
}
