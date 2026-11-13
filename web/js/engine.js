// Client for the engine Web Worker. Requests carry ids so replies can be matched.

export class Engine extends EventTarget {
  constructor(url = new URL("../worker.js", import.meta.url)) {
    super();
    this.url = url;
    this.worker = null;
    this.pending = new Map();
    this.quiet = new Map(); // background requests (problems, help, variable pages): they do not count as "busy"
    this.nextId = 1;
    this.ready = false;
    this.info = null;
    this.start();
  }

  start() {
    this.ready = false;
    this.worker = new Worker(this.url);
    this.worker.onmessage = (ev) => this._onMessage(ev.data);
    this.worker.onerror = (ev) => {
      ev.preventDefault?.();
      this._emit("fatal", { error: ev.message || "The engine worker crashed." });
    };
  }

  _emit(type, detail) {
    this.dispatchEvent(new CustomEvent(type, { detail }));
  }

  _onMessage(msg) {
    if (!msg) return;
    if (msg.type === "progress") return this._emit("progress", msg);
    if (msg.type === "ready") {
      this.ready = true;
      this.info = msg.info;
      return this._emit("ready", msg.info);
    }
    if (msg.type === "fatal") return this._emit("fatal", msg);
    if (msg.type === "event") return this._emit(msg.name, msg.data); // mylab Blocks: data sent while a request runs
    const q = this.quiet.get(msg.id);
    if (q) {
      this.quiet.delete(msg.id);
      return msg.ok ? q.resolve(msg.result) : q.reject(new Error(msg.error));
    }
    const p = this.pending.get(msg.id);
    if (!p) return;
    this.pending.delete(msg.id);
    if (this.pending.size === 0) this._emit("idle", {});
    msg.ok ? p.resolve(msg.result) : p.reject(new Error(msg.error));
  }

  request(type, args = {}) {
    if (!this.ready) return Promise.reject(new Error("The engine is still loading."));
    const id = this.nextId++;
    return new Promise((resolve, reject) => {
      this.pending.set(id, { resolve, reject, type });
      if (this.pending.size === 1) this._emit("busy", { type });
      this.worker.postMessage({ id, type, ...args });
    });
  }

  /** Like request(), but the engine does not look busy: Run stays available while it waits. */
  background(type, args = {}) {
    if (!this.ready) return Promise.reject(new Error("The engine is still loading."));
    const id = this.nextId++;
    return new Promise((resolve, reject) => {
      this.quiet.set(id, { resolve, reject, type });
      this.worker.postMessage({ id, type, ...args });
    });
  }

  get busy() {
    return this.pending.size > 0;
  }

  /** opts: { fs, cwd } - see worker.js */
  run(code, opts = {}) { return this.request("run", { code, ...opts }); }
  trace(code, opts = {}) { return this.request("trace", { code, ...opts }); }
  publish(sections, opts = {}) { return this.request("publish", { sections, ...opts }); }
  test(files, code) { return this.request("test", { files, code }); }
  variables() { return this.request("variables"); }
  showVariable(name) { return this.request("showVariable", { name }); }
  complete(prefix) { return this.request("complete", { prefix }); }
  help(name) { return this.background("help", { name }); }
  lint(code, known) { return this.background("lint", { code, known }); }
  helpIndex() { return this.background("helpIndex"); }
  inspect(name, opts = {}) { return this.background("inspect", { name, ...opts }); }
  csv(name, opts = {}) { return this.background("csv", { name, ...opts }); }
  reset() { return this.request("reset"); }

  /** Resolves when the engine is usable (again); rejects if loading fails. */
  whenReady() {
    if (this.ready) return Promise.resolve(this.info);
    return new Promise((resolve, reject) => {
      const done = (fn) => (e) => {
        this.removeEventListener("ready", onReady);
        this.removeEventListener("fatal", onFatal);
        fn(e);
      };
      const onReady = done((e) => resolve(e.detail));
      const onFatal = done((e) => reject(new Error(e.detail.error)));
      this.addEventListener("ready", onReady);
      this.addEventListener("fatal", onFatal);
    });
  }

  /** Kill the worker (stops running code) and start a fresh one. Workspace is lost. */
  restart() {
    this.worker.terminate();
    for (const p of this.pending.values()) p.reject(new Error("Stopped."));
    for (const q of this.quiet.values()) q.reject(new Error("Stopped."));
    this.pending.clear();
    this.quiet.clear();
    this._emit("idle", {});
    this._emit("restarting", {});
    this.start();
  }
}
