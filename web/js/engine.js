// Client for the engine Web Worker. Requests carry ids so replies can be matched.

export class Engine extends EventTarget {
  constructor(url = new URL("../worker.js", import.meta.url)) {
    super();
    this.url = url;
    this.worker = null;
    this.pending = new Map();
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

  get busy() {
    return this.pending.size > 0;
  }

  run(code, files) { return this.request("run", { code, files }); }
  variables() { return this.request("variables"); }
  showVariable(name) { return this.request("showVariable", { name }); }
  complete(prefix) { return this.request("complete", { prefix }); }
  help(name) { return this.request("help", { name }); }
  reset() { return this.request("reset"); }

  /** Kill the worker (stops running code) and start a fresh one. Workspace is lost. */
  restart() {
    this.worker.terminate();
    for (const p of this.pending.values()) p.reject(new Error("Stopped."));
    this.pending.clear();
    this._emit("idle", {});
    this._emit("restarting", {});
    this.start();
  }
}
