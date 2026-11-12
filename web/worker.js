/* mylab engine worker: runs the Python interpreter in Pyodide, off the UI thread.
 *
 * Protocol (main -> worker):  { id, type, ...args }
 *   "run"      { code, fs, cwd }            run in the persistent session
 *   "trace"    { code, fs, cwd, maxSteps }  step-through trace in a fresh session
 *   "publish"  { sections, tail, fs, cwd }  run sections one by one in a fresh session
 *   "test"     { files, code }              one autograder test in a clean folder and session
 *   "variables", "showVariable" {name}, "complete" {prefix}, "help" {name}, "reset"
 *
 *   fs  = { reset, files: [{ path, content? }] } mirrors the project into the working folder.
 *         content (string or Uint8Array) is left out for files that did not change since the
 *         last request; files that are no longer listed are deleted. reset wipes the folder.
 *   cwd = project folder to run in ("" for the project root); omit to stay where we are.
 *
 * Protocol (worker -> main):
 *   { type: "progress", step, label }        while loading
 *   { type: "ready", info }                  once the engine is usable
 *   { type: "fatal", error }                 loading failed
 *   { id, ok: true, result } / { id, ok: false, error }
 *
 * Results of "run" also carry  files: { written: [{ path, content: Uint8Array }],
 * skipped: [{ path, size }], missing: [path] }  - what the code created, changed or deleted.
 */
"use strict";

const PYODIDE_VERSION = "0.27.7";
const PYODIDE_URL = `https://cdn.jsdelivr.net/pyodide/v${PYODIDE_VERSION}/full/`;
const WORKDIR = "/home/pyodide/work";
const GRADEDIR = "/home/pyodide/grade";
const MAX_WRITTEN_FILE = 5 * 1024 * 1024; // files written by code that are copied back into the project
const MAX_WRITTEN_TOTAL = 20 * 1024 * 1024;

let pyodide = null;
let api = null; // Python dict of helper functions
let stdoutBuffer = [];
const synced = new Set(); // project paths currently mirrored in WORKDIR

function progress(step, label) {
  self.postMessage({ type: "progress", step, label });
}

const PY_GLUE = `
import json, os, shutil, sys, traceback, base64

WORKDIR = ${JSON.stringify(WORKDIR)}
GRADEDIR = ${JSON.stringify(GRADEDIR)}
os.makedirs(WORKDIR, exist_ok=True)
os.chdir(WORKDIR)
os.environ["MPLBACKEND"] = "Agg"

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot  # warm up font cache now, not on the first plot

from mylab.api import Session
from mylab.errors import MatlabError
import mylab

def _no_input(prompt=""):
    raise MatlabError("input() is not available in the browser yet")

_session = None
_auto_path = []

def _make_session():
    s = Session()
    s.interp.input_func = _no_input
    return s

def _new_session():
    global _session
    _session = _make_session()
    return _session

_new_session()

def _vars():
    return [v.to_dict() for v in _session.variables()]

def _search_path(root):
    """The project root and every subfolder that holds .m files."""
    dirs = [root]
    for d, subdirs, names in os.walk(root):
        subdirs[:] = sorted(s for s in subdirs if not s.startswith("."))
        if d != root and any(n.endswith(".m") for n in names):
            dirs.append(d)
    return dirs

def _enter(session, cwd, root=WORKDIR):
    """chdir into the project folder cwd (None: stay) and put the project on the search path."""
    global _auto_path
    if cwd is not None:
        target = os.path.normpath(os.path.join(root, cwd))
        if not (target == root or target.startswith(root + os.sep)) or not os.path.isdir(target):
            target = root
        os.chdir(target)
    else:
        try:
            os.getcwd()
        except OSError:
            os.chdir(root)
    here = os.getcwd()
    auto = [here] + [d for d in _search_path(root) if d != here]
    if session is _session:
        user = [p for p in session.interp.path if p not in _auto_path]
        _auto_path = auto
        session.interp.path = user + auto
    else:
        session.interp.path = auto

def _rel(path, root=WORKDIR):
    if not path:
        return None
    path = os.path.abspath(path)
    if path.startswith(root + os.sep):
        return path[len(root) + 1:].replace(os.sep, "/")
    return os.path.basename(path)

def _run_session(session, code, root=WORKDIR, runner=None):
    """session.run(code) as a dict, plus the file an error happened in (error_file)."""
    interp = session.interp
    original = interp.execute
    info = {}
    def execute(src, file=None):
        try:
            return original(src, file)
        except MatlabError as e:
            info["file"] = getattr(e, "file", None)
            raise
    interp.execute = execute
    try:
        res = runner(session, code) if runner else session.run(code)
        out = res.to_dict()
    except Exception as e:  # engine bug: report it instead of killing the worker
        out = {"output": "", "error": f"Internal error ({type(e).__name__}): {e}",
               "error_line": None, "figures": [], "internal": traceback.format_exc()}
        try:
            out["figures"] = [base64.b64encode(f).decode() for f in session.render_figures()]
        except Exception:
            pass
    finally:
        try:
            del interp.execute
        except AttributeError:
            pass
    out.setdefault("error_file", _rel(info.get("file"), root) if out.get("error") else None)
    return out

def _scan():
    """Every file below WORKDIR with its (mtime, size), to find what a run wrote."""
    found = {}
    for d, subdirs, names in os.walk(WORKDIR):
        subdirs[:] = [s for s in subdirs if not s.startswith(".") and s != "__pycache__"]
        for n in names:
            p = os.path.join(d, n)
            try:
                st = os.stat(p)
            except OSError:
                continue
            found[p[len(WORKDIR) + 1:].replace(os.sep, "/")] = [st.st_mtime_ns, st.st_size]
            if len(found) > 5000:
                return found
    return found

def _wipe(folder):
    os.chdir("/")
    shutil.rmtree(folder, ignore_errors=True)
    os.makedirs(folder, exist_ok=True)
    os.chdir(folder)

def _run(code, cwd):
    _enter(_session, cwd)
    before = _scan()
    out = _run_session(_session, code)
    try:
        out["variables"] = _vars()
    except Exception:
        out["variables"] = []
    after = _scan()
    out["_changed"] = [[p, v[1]] for p, v in after.items() if before.get(p) != v]
    out["_present"] = list(after)
    return json.dumps(out)

def _trace(code, cwd, max_steps):
    from mylab.debugger import trace
    session = _make_session()
    home = os.getcwd()
    try:
        _enter(session, cwd)
        out = _run_session(session, code, runner=lambda s, c: trace(s, c, max_steps=int(max_steps)))
    finally:
        try:
            os.chdir(home)
        except OSError:
            os.chdir(WORKDIR)
    return json.dumps(out)

def _publish(sections_json, tail, cwd):
    sections = json.loads(sections_json)
    session = _make_session()
    home = os.getcwd()
    results = []
    try:
        _enter(session, cwd)
        failed = False
        for code in sections:
            if failed:
                results.append({"skipped": True})
                continue
            if not code.strip():
                results.append({"output": "", "error": None, "error_line": None, "figures": []})
                continue
            out = _run_session(session, code + ("\\n" + tail if tail else ""))
            results.append(out)
            failed = bool(out.get("error"))
    finally:
        try:
            os.chdir(home)
        except OSError:
            os.chdir(WORKDIR)
    return json.dumps(results)

def _test(code):
    """Run one autograder test in GRADEDIR (already filled by the worker) with a clean session."""
    session = _make_session()
    home = os.getcwd()
    try:
        os.chdir(GRADEDIR)
        session.interp.path = [GRADEDIR]
        out = _run_session(session, code, root=GRADEDIR)
        out.pop("figures", None)
        out.pop("animations", None)
    finally:
        try:
            os.chdir(home)
        except OSError:
            os.chdir(WORKDIR)
    return json.dumps(out)

def _show(name):
    return _session.show_variable(name)

def _complete(prefix):
    return json.dumps(_session.complete(prefix)[:200])

def _help(name):
    return _session.help(name)

def _reset():
    _session.reset()
    return json.dumps(_vars())

def _cwd():
    return _rel(os.getcwd()) if os.getcwd() != WORKDIR else ""

_api = {
    "run": _run, "trace": _trace, "publish": _publish, "test": _test, "wipe": _wipe,
    "vars": lambda: json.dumps(_vars()), "show": _show,
    "complete": _complete, "help": _help, "reset": _reset, "cwd": _cwd,
    "clear_cache": lambda: _session.clear_file_cache(),
    "version": getattr(mylab, "__version__", "?"),
}
_api
`;

async function init() {
  try {
    progress("python", "Downloading Python (Pyodide)…");
    importScripts(PYODIDE_URL + "pyodide.js");
    pyodide = await self.loadPyodide({
      indexURL: PYODIDE_URL,
      stdout: (s) => stdoutBuffer.push(s),
      stderr: (s) => stdoutBuffer.push(s),
    });

    progress("packages", "Loading NumPy, SciPy and Matplotlib…");
    await pyodide.loadPackage(["numpy", "scipy", "matplotlib", "pillow", "micropip"], {
      messageCallback: () => {},
      errorCallback: (m) => stdoutBuffer.push(String(m)),
    });

    progress("mylab", "Installing the mylab engine…");
    const manifestUrl = new URL("wheels/manifest.json", self.location.href);
    const resp = await fetch(manifestUrl, { cache: "no-cache" });
    if (!resp.ok) {
      throw new Error(
        `Could not load ${manifestUrl.pathname} (HTTP ${resp.status}). ` +
          "Run `python scripts/build_web.py` to build the mylab wheel."
      );
    }
    const manifest = await resp.json();
    const wheelUrl = new URL("wheels/" + manifest.wheel, self.location.href).href;
    const micropip = pyodide.pyimport("micropip");
    await micropip.install(wheelUrl, { deps: false });

    progress("start", "Starting the interpreter…");
    api = await pyodide.runPythonAsync(PY_GLUE);
    stdoutBuffer = [];
    self.postMessage({
      type: "ready",
      info: { pyodide: pyodide.version, mylab: api.get("version") },
    });
  } catch (err) {
    self.postMessage({ type: "fatal", error: String((err && err.message) || err) });
  }
}

function call(name, ...args) {
  const fn = api.get(name);
  try {
    const r = fn(...args);
    return r;
  } finally {
    fn.destroy && fn.destroy();
  }
}

function pyErrorMessage(err) {
  const msg = String((err && err.message) || err);
  // PythonError messages contain a traceback; the last line is the useful part.
  const lines = msg.trim().split("\n");
  const last = lines[lines.length - 1] || msg;
  const m = last.match(/^(?:mylab\.errors\.)?MatlabError: (.*)$/);
  return m ? m[1] : last;
}

/* ---------- files ---------- */

/** A safe relative path ("a/b.m"), or null for anything that tries to leave the folder. */
function cleanPath(p) {
  const parts = String(p).replace(/\\/g, "/").split("/").filter((s) => s && s !== ".");
  if (!parts.length || parts.some((s) => s === "..")) return null;
  return parts.join("/");
}

function writeInto(root, path, content) {
  const full = root + "/" + path;
  const dir = full.slice(0, full.lastIndexOf("/"));
  pyodide.FS.mkdirTree(dir);
  pyodide.FS.writeFile(full, content);
}

function syncFiles(fs) {
  if (!fs) return;
  let changed = false;
  if (fs.reset) {
    call("wipe", WORKDIR);
    synced.clear();
    changed = true;
  }
  const wanted = new Set();
  for (const f of fs.files || []) {
    const p = cleanPath(f.path);
    if (p) wanted.add(p);
  }
  for (const p of [...synced]) {
    if (wanted.has(p)) continue;
    try {
      pyodide.FS.unlink(WORKDIR + "/" + p);
    } catch {
      /* already gone */
    }
    synced.delete(p);
    changed = true;
  }
  for (const f of fs.files || []) {
    const p = cleanPath(f.path);
    if (!p || f.content === undefined || f.content === null) continue;
    writeInto(WORKDIR, p, f.content);
    synced.add(p);
    changed = true;
  }
  // Re-parse edited files even if the clock did not move between writes.
  if (changed) call("clear_cache");
}

/** Collect files the run created or changed, and note project files it deleted. */
function collectWritten(out) {
  const changed = out._changed || [];
  const present = new Set(out._present || []);
  delete out._changed;
  delete out._present;
  const files = { written: [], skipped: [], missing: [] };
  let total = 0;
  for (const [path, size] of changed) {
    if (size > MAX_WRITTEN_FILE || total + size > MAX_WRITTEN_TOTAL) {
      files.skipped.push({ path, size });
      continue;
    }
    try {
      files.written.push({ path, content: pyodide.FS.readFile(WORKDIR + "/" + path) });
      synced.add(path);
      total += size;
    } catch {
      /* vanished or unreadable */
    }
  }
  for (const p of synced) if (!present.has(p)) files.missing.push(p);
  for (const p of files.missing) synced.delete(p);
  out.files = files;
  return files.written.map((f) => f.content.buffer);
}

function withStdout(out) {
  if (stdoutBuffer.length) {
    // Text Python printed directly (e.g. library warnings).
    out.output = (out.output || "") + stdoutBuffer.join("\n") + "\n";
    stdoutBuffer = [];
  }
  return out;
}

const handlers = {
  run({ code, fs, cwd }) {
    syncFiles(fs);
    stdoutBuffer = [];
    const out = withStdout(JSON.parse(call("run", String(code), cwd === undefined ? null : cwd)));
    const transfer = collectWritten(out);
    return { result: out, transfer };
  },
  trace({ code, fs, cwd, maxSteps }) {
    syncFiles(fs);
    stdoutBuffer = [];
    const out = JSON.parse(call("trace", String(code), cwd === undefined ? null : cwd, maxSteps || 5000));
    stdoutBuffer = [];
    return { result: out };
  },
  publish({ sections, tail, fs, cwd }) {
    syncFiles(fs);
    stdoutBuffer = [];
    const out = JSON.parse(call("publish", JSON.stringify(sections || []), tail || "", cwd === undefined ? null : cwd));
    stdoutBuffer = [];
    return { result: out };
  },
  test({ files, code }) {
    call("wipe", GRADEDIR);
    for (const f of files || []) {
      const p = cleanPath(f.path);
      if (p) writeInto(GRADEDIR, p, f.content);
    }
    stdoutBuffer = [];
    const out = JSON.parse(call("test", String(code)));
    stdoutBuffer = [];
    return { result: out };
  },
  variables() {
    return { result: JSON.parse(call("vars")) };
  },
  showVariable({ name }) {
    return { result: call("show", String(name)) };
  },
  complete({ prefix }) {
    return { result: JSON.parse(call("complete", String(prefix))) };
  },
  help({ name }) {
    return { result: call("help", String(name)) };
  },
  reset() {
    return { result: JSON.parse(call("reset")) };
  },
};

self.onmessage = (ev) => {
  const msg = ev.data || {};
  const { id, type } = msg;
  if (!api) {
    self.postMessage({ id, ok: false, error: "The engine is still loading." });
    return;
  }
  const h = handlers[type];
  if (!h) {
    self.postMessage({ id, ok: false, error: `Unknown request: ${type}` });
    return;
  }
  try {
    const { result, transfer } = h(msg);
    self.postMessage({ id, ok: true, result }, transfer || []);
  } catch (err) {
    self.postMessage({ id, ok: false, error: pyErrorMessage(err) });
  }
};

init();
