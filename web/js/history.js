// Local version history: the last saves of every file, kept in this browser (IndexedDB database
// "mylab-history"), with a view that compares a saved version with the file as it is now.
// Nothing leaves the browser. Temporary projects and browsers without storage keep versions in memory.

import { $, el, toast } from "./ui.js";
import { renderDiff, statsText } from "./diff.js";

const DB_NAME = "mylab-history";
const STORE = "versions";
export const MAX_VERSIONS = 30;
const AUTO_GAP = 3 * 60 * 1000; // autosaves add a version at most this often
const MAX_SIZE = 400000; // bigger files are not versioned

const fileKey = (project, path) => project + "\n" + path;

function openDb() {
  return new Promise((resolve, reject) => {
    let req;
    try {
      req = indexedDB.open(DB_NAME, 1);
    } catch (e) {
      reject(e);
      return;
    }
    req.onupgradeneeded = () => {
      const s = req.result.createObjectStore(STORE, { keyPath: "id", autoIncrement: true });
      s.createIndex("file", "file", { unique: false });
    };
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error);
    req.onblocked = () => reject(new Error("IndexedDB is blocked"));
  });
}

function run(db, mode, fn) {
  return new Promise((resolve, reject) => {
    const t = db.transaction(STORE, mode);
    let result;
    const r = fn(t.objectStore(STORE));
    if (r) r.onsuccess = () => (result = r.result);
    t.oncomplete = () => resolve(result);
    t.onerror = () => reject(t.error);
    t.onabort = () => reject(t.error);
  });
}

export class HistoryStore {
  constructor() {
    this.db = null;
    this.mem = new Map(); // file key -> [{ id, time, content, reason }]
    this.nextId = 1;
    this.ready = this._open();
  }

  async _open() {
    try {
      if (typeof indexedDB === "undefined") throw new Error("no IndexedDB");
      this.db = await openDb();
    } catch (e) {
      console.warn("mylab: version history is kept in memory only.", e);
      this.db = null;
    }
  }

  /** Versions of a file, newest first. */
  async list(project, path, { temp = false } = {}) {
    await this.ready;
    const key = fileKey(project, path);
    if (this.db && !temp) {
      try {
        const all = await run(this.db, "readonly", (s) => s.index("file").getAll(IDBKeyRange.only(key)));
        return (all || []).sort((a, b) => b.time - a.time || b.id - a.id);
      } catch (e) {
        console.warn("mylab: could not read the version history.", e);
        return [];
      }
    }
    return [...(this.mem.get(key) || [])].sort((a, b) => b.time - a.time || b.id - a.id);
  }

  /** Add a version unless the newest one already has this content. Returns true when one was added. */
  async add(project, path, content, reason, { temp = false } = {}) {
    if (typeof content !== "string" || content.length > MAX_SIZE) return false;
    const versions = await this.list(project, path, { temp });
    if (versions.length && versions[0].content === content) return false;
    const rec = { file: fileKey(project, path), project, path, time: Date.now(), content, reason };
    const extra = versions.slice(MAX_VERSIONS - 1);
    if (this.db && !temp) {
      try {
        await run(this.db, "readwrite", (s) => {
          s.add(rec);
          for (const old of extra) s.delete(old.id);
        });
        return true;
      } catch (e) {
        console.warn("mylab: could not save a version.", e);
        return false;
      }
    }
    rec.id = this.nextId++;
    const list = this.mem.get(rec.file) || [];
    list.push(rec);
    this.mem.set(rec.file, list.filter((v) => !extra.includes(v)));
    return true;
  }

  /** Keep the history when a file is renamed or moved. */
  async move(project, from, to) {
    await this.ready;
    const a = fileKey(project, from);
    const b = fileKey(project, to);
    if (this.mem.has(a)) {
      this.mem.set(b, this.mem.get(a).map((v) => ({ ...v, file: b, path: to })));
      this.mem.delete(a);
    }
    if (!this.db) return;
    try {
      await run(this.db, "readwrite", (s) => {
        const cur = s.index("file").openCursor(IDBKeyRange.only(a));
        cur.onsuccess = () => {
          const c = cur.result;
          if (!c) return;
          c.update({ ...c.value, file: b, path: to });
          c.continue();
        };
      });
    } catch (e) {
      console.warn("mylab: could not move the version history.", e);
    }
  }

  /** Forget the versions of some files (or of a whole project when paths is null). */
  async forget(project, paths = null) {
    await this.ready;
    const range = paths ? null : IDBKeyRange.bound(project + "\n", project + "\n￿");
    for (const key of [...this.mem.keys()]) {
      if (paths ? paths.some((p) => fileKey(project, p) === key) : key.startsWith(project + "\n")) this.mem.delete(key);
    }
    if (!this.db) return;
    try {
      await run(this.db, "readwrite", (s) => {
        const ranges = paths ? paths.map((p) => IDBKeyRange.only(fileKey(project, p))) : [range];
        for (const r of ranges) {
          const cur = s.index("file").openKeyCursor(r);
          cur.onsuccess = () => {
            const c = cur.result;
            if (!c) return;
            s.delete(c.primaryKey);
            c.continue();
          };
        }
      });
    } catch (e) {
      console.warn("mylab: could not clear the version history.", e);
    }
  }
}

function timeLabel(t) {
  const d = new Date(t);
  const today = new Date();
  const sameDay = d.toDateString() === today.toDateString();
  const time = d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
  if (sameDay) return `Today, ${time}`;
  const y = new Date(today.getTime() - 86400000);
  if (d.toDateString() === y.toDateString()) return `Yesterday, ${time}`;
  return `${d.toLocaleDateString([], { month: "short", day: "numeric" })}, ${time}`;
}

export function initHistory(app) {
  const { state } = app;
  const store = new HistoryStore();
  const lastAuto = new Map(); // file key -> time of the last automatic version
  const opts = () => ({ temp: !!(state.project && state.project.temp) });

  /** Called after a file was written to storage. */
  async function onSaved({ path, content, previous, manual }) {
    if (!state.project || typeof content !== "string") return;
    const project = state.project.id;
    const key = fileKey(project, path);
    const versions = await store.list(project, path, opts());
    // The first time a file changes, keep what it looked like before.
    if (!versions.length && typeof previous === "string" && previous !== content && previous.trim()) {
      await store.add(project, path, previous, "Before your edits", opts());
    }
    const now = Date.now();
    if (!manual && now - (lastAuto.get(key) || 0) < AUTO_GAP && versions.length) return;
    if (await store.add(project, path, content, manual ? "Saved" : "Autosaved", opts())) lastAuto.set(key, now);
  }

  /** Keep the current text of a file before something replaces it. */
  async function snapshot(path, reason) {
    if (!state.project || !app.isText(path)) return;
    const project = state.project.id;
    const content = app.liveContent(path);
    const o = opts();
    queue = queue.then(() => store.add(project, path, content, reason, o)).catch((e) => console.warn(e));
    await queue;
  }

  let shown = null; // { path, versions, index }

  function renderVersion(i) {
    shown.index = i;
    const v = shown.versions[i];
    const current = app.liveContent(shown.path);
    document.querySelectorAll("#history-list .history-item").forEach((b, k) => {
      b.classList.toggle("active", k === i);
      b.setAttribute("aria-pressed", k === i ? "true" : "false");
    });
    const stats = renderDiff($("history-diff"), v.content, current);
    $("history-summary").textContent =
      stats.added || stats.removed ? `Since this version: ${statsText(stats)}.` : "This version is the same as the file now.";
    $("history-restore").disabled = !(stats.added || stats.removed);
  }

  async function openHistory(path = state.active) {
    if (!path || !app.isText(path)) {
      toast("Open a text file to see its history.");
      return;
    }
    await app.flushAll();
    const versions = await store.list(state.project.id, path, opts());
    shown = { path, versions, index: 0 };
    $("history-title").textContent = app.baseName(path);
    const list = $("history-list");
    list.innerHTML = "";
    $("history-empty").hidden = versions.length > 0;
    $("history-main").hidden = versions.length === 0;
    versions.forEach((v, i) => {
      const lines = v.content.split("\n").length;
      list.append(
        el(
          "button",
          { type: "button", class: "history-item", "aria-pressed": "false", "data-id": String(v.id), onclick: () => renderVersion(i) },
          el("strong", {}, timeLabel(v.time)),
          el("span", { class: "muted small" }, `${v.reason} · ${lines} line${lines === 1 ? "" : "s"}`)
        )
      );
    });
    if (versions.length) renderVersion(0);
    if (!$("dlg-history").open) $("dlg-history").showModal();
  }

  $("history-restore").addEventListener("click", async () => {
    if (!shown) return;
    const v = shown.versions[shown.index];
    await snapshot(shown.path, "Before restoring");
    app.replaceText(shown.path, v.content);
    $("dlg-history").close();
    toast(`Restored the version from ${timeLabel(v.time)}. Undo with Ctrl+Z.`);
  });

  // One save at a time, so two quick saves can't both think they are the first.
  let queue = Promise.resolve();
  app.on("saved", (info) => {
    queue = queue.then(() => onSaved(info)).catch((e) => console.warn(e));
  });
  app.on("moved", (from, to) => state.project && store.move(state.project.id, from, to));
  app.on("removed", (paths) => state.project && store.forget(state.project.id, paths));
  app.on("projectDeleted", (id) => store.forget(id));

  Object.assign(app, { openHistory, snapshot, historyStore: store });
}
