// Project and file storage: IndexedDB when available, otherwise in memory (lost on reload).
//
// A project is { id, name, kind, meta, created }. kind is "files" (ordinary), "assignment",
// "github" or "shared". Projects created with temp:true are never written to IndexedDB;
// they hold code opened from a share link until the visitor chooses to keep a copy.
// A file is { path, content, modified }: path may contain "/" (folders) and content is a
// string (text) or a Uint8Array (binary data such as images).

const DB_NAME = "mylab";
const DB_VERSION = 2;
const LEGACY = "files"; // version 1: flat list of { name, content, modified }
const PROJECTS = "projects";
const FILES = "pfiles";

export const DEFAULT_PROJECT = "main";

const fileKey = (project, path) => project + "\n" + path;

function openDb() {
  return new Promise((resolve, reject) => {
    let req;
    try {
      req = indexedDB.open(DB_NAME, DB_VERSION);
    } catch (e) {
      reject(e);
      return;
    }
    req.onupgradeneeded = (ev) => {
      const db = req.result;
      if (!db.objectStoreNames.contains(PROJECTS)) db.createObjectStore(PROJECTS, { keyPath: "id" });
      if (!db.objectStoreNames.contains(FILES)) {
        const s = db.createObjectStore(FILES, { keyPath: "key" });
        s.createIndex("project", "project", { unique: false });
      }
      // Move files saved by the first version of the app into the default project.
      if (ev.oldVersion === 1 && db.objectStoreNames.contains(LEGACY)) {
        const t = req.transaction;
        const all = t.objectStore(LEGACY).getAll();
        all.onsuccess = () => {
          const files = t.objectStore(FILES);
          for (const f of all.result || []) {
            files.put({ key: fileKey(DEFAULT_PROJECT, f.name), project: DEFAULT_PROJECT, path: f.name, content: f.content, modified: f.modified || Date.now() });
          }
          if ((all.result || []).length) {
            t.objectStore(PROJECTS).put({ id: DEFAULT_PROJECT, name: "My files", kind: "files", meta: {}, created: Date.now() });
          }
        };
      }
    };
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error);
    req.onblocked = () => reject(new Error("IndexedDB is blocked"));
  });
}

function tx(db, stores, mode, fn) {
  return new Promise((resolve, reject) => {
    const t = db.transaction(stores, mode);
    let result;
    const r = fn(...[].concat(stores).map((s) => t.objectStore(s)));
    if (r) r.onsuccess = () => (result = r.result);
    t.oncomplete = () => resolve(result);
    t.onerror = () => reject(t.error);
    t.onabort = () => reject(t.error);
  });
}

export function newId() {
  return "p" + Date.now().toString(36) + Math.random().toString(36).slice(2, 7);
}

export class ProjectStore {
  constructor() {
    this.db = null;
    this.persistent = false;
    this.mem = { projects: new Map(), files: new Map() }; // temp projects, and everything when there is no IndexedDB
  }

  async init() {
    try {
      if (typeof indexedDB === "undefined") throw new Error("no IndexedDB");
      this.db = await openDb();
      this.persistent = true;
    } catch (e) {
      console.warn("mylab: IndexedDB unavailable, files will only be kept in memory.", e);
      this.db = null;
      this.persistent = false;
    }
    return this;
  }

  _inMemory(id) {
    return !this.db || this.mem.projects.has(id);
  }

  _fallback(e) {
    console.warn("mylab: IndexedDB failed, switching to in-memory storage.", e);
    this.db = null;
    this.persistent = false;
  }

  async _try(fn, memFn) {
    if (this.db) {
      try {
        return await fn();
      } catch (e) {
        this._fallback(e);
      }
    }
    return memFn();
  }

  /* ---- projects ---- */

  async listProjects() {
    const temp = [...this.mem.projects.values()].map((p) => ({ ...p }));
    if (!this.db) return temp;
    const saved = await this._try(() => tx(this.db, PROJECTS, "readonly", (s) => s.getAll()), () => []);
    return [...(saved || []), ...temp];
  }

  /** Create or update a project record. */
  async putProject(project) {
    const rec = { kind: "files", meta: {}, created: Date.now(), ...project };
    if (rec.temp || this._inMemory(rec.id)) {
      this.mem.projects.set(rec.id, rec);
      if (!this.mem.files.has(rec.id)) this.mem.files.set(rec.id, new Map());
      return rec;
    }
    await this._try(
      () => tx(this.db, PROJECTS, "readwrite", (s) => s.put(rec)),
      () => {
        this.mem.projects.set(rec.id, rec);
        this.mem.files.set(rec.id, this.mem.files.get(rec.id) || new Map());
      }
    );
    return rec;
  }

  async deleteProject(id) {
    if (this._inMemory(id)) {
      this.mem.projects.delete(id);
      this.mem.files.delete(id);
      return;
    }
    await this._try(
      () =>
        tx(this.db, [PROJECTS, FILES], "readwrite", (projects, files) => {
          projects.delete(id);
          const cursor = files.index("project").openKeyCursor(IDBKeyRange.only(id));
          cursor.onsuccess = () => {
            const c = cursor.result;
            if (c) {
              files.delete(c.primaryKey);
              c.continue();
            }
          };
        }),
      () => {}
    );
  }

  /* ---- files ---- */

  async listFiles(project) {
    const fromMem = () => [...(this.mem.files.get(project) || new Map()).values()].map((f) => ({ ...f }));
    if (this._inMemory(project)) return fromMem();
    const recs = await this._try(() => tx(this.db, FILES, "readonly", (s) => s.index("project").getAll(IDBKeyRange.only(project))), fromMem);
    return (recs || []).map((r) => ({ path: r.path, content: r.content, modified: r.modified }));
  }

  /** Save several files in one transaction. files: [{ path, content }] */
  async putFiles(project, files) {
    const now = Date.now();
    const recs = files.map((f) => ({ key: fileKey(project, f.path), project, path: f.path, content: f.content, modified: f.modified || now }));
    const toMem = () => {
      if (!this.mem.files.has(project)) this.mem.files.set(project, new Map());
      const m = this.mem.files.get(project);
      for (const r of recs) m.set(r.path, { path: r.path, content: r.content, modified: r.modified });
    };
    if (this._inMemory(project)) return toMem();
    await this._try(
      () =>
        tx(this.db, FILES, "readwrite", (s) => {
          for (const r of recs) s.put(r);
        }),
      toMem
    );
  }

  putFile(project, path, content) {
    return this.putFiles(project, [{ path, content }]);
  }

  async removeFiles(project, paths) {
    const fromMem = () => {
      const m = this.mem.files.get(project);
      if (m) for (const p of paths) m.delete(p);
    };
    if (this._inMemory(project)) return fromMem();
    await this._try(
      () =>
        tx(this.db, FILES, "readwrite", (s) => {
          for (const p of paths) s.delete(fileKey(project, p));
        }),
      fromMem
    );
  }

  removeFile(project, path) {
    return this.removeFiles(project, [path]);
  }
}

/** localStorage helpers that never throw. */
export const prefs = {
  get(key, fallback = null) {
    try {
      const v = localStorage.getItem("mylab." + key);
      return v === null ? fallback : JSON.parse(v);
    } catch {
      return fallback;
    }
  },
  set(key, value) {
    try {
      localStorage.setItem("mylab." + key, JSON.stringify(value));
      return true;
    } catch {
      return false;
    }
  },
  remove(key) {
    try {
      localStorage.removeItem("mylab." + key);
    } catch {
      /* ignore */
    }
  },
};
