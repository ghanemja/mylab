// File storage: IndexedDB when available, otherwise an in-memory map (files are lost on reload).

const DB_NAME = "mylab";
const STORE = "files";

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
      const db = req.result;
      if (!db.objectStoreNames.contains(STORE)) db.createObjectStore(STORE, { keyPath: "name" });
    };
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error);
    req.onblocked = () => reject(new Error("IndexedDB is blocked"));
  });
}

function tx(db, mode, fn) {
  return new Promise((resolve, reject) => {
    const t = db.transaction(STORE, mode);
    const store = t.objectStore(STORE);
    let result;
    const r = fn(store);
    if (r) r.onsuccess = () => (result = r.result);
    t.oncomplete = () => resolve(result);
    t.onerror = () => reject(t.error);
    t.onabort = () => reject(t.error);
  });
}

export class FileStore {
  constructor() {
    this.db = null;
    this.memory = new Map();
    this.persistent = false;
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

  async list() {
    if (!this.db) return [...this.memory.values()].map((f) => ({ ...f }));
    try {
      return (await tx(this.db, "readonly", (s) => s.getAll())) || [];
    } catch (e) {
      this._fallback(e);
      return this.list();
    }
  }

  async put(name, content) {
    const rec = { name, content, modified: Date.now() };
    if (!this.db) {
      this.memory.set(name, rec);
      return rec;
    }
    try {
      await tx(this.db, "readwrite", (s) => s.put(rec));
    } catch (e) {
      this._fallback(e);
      this.memory.set(name, rec);
    }
    return rec;
  }

  async remove(name) {
    if (!this.db) {
      this.memory.delete(name);
      return;
    }
    try {
      await tx(this.db, "readwrite", (s) => s.delete(name));
    } catch (e) {
      this._fallback(e);
      this.memory.delete(name);
    }
  }

  async rename(oldName, newName) {
    const all = await this.list();
    const rec = all.find((f) => f.name === oldName);
    if (!rec) return;
    await this.put(newName, rec.content);
    await this.remove(oldName);
  }

  _fallback(e) {
    console.warn("mylab: IndexedDB failed, switching to in-memory storage.", e);
    this.db = null;
    this.persistent = false;
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
