/* mylab engine worker: runs the Python interpreter in Pyodide, off the UI thread.
 *
 * Protocol (main -> worker):  { id, type, ...args }
 *   type: "run" {code, files}, "variables", "showVariable" {name},
 *         "complete" {prefix}, "help" {name}, "reset"
 * Protocol (worker -> main):
 *   { type: "progress", step, label }        while loading
 *   { type: "ready", info }                  once the engine is usable
 *   { type: "fatal", error }                 loading failed
 *   { id, ok: true, result } / { id, ok: false, error }
 */
"use strict";

const PYODIDE_VERSION = "0.27.7";
const PYODIDE_URL = `https://cdn.jsdelivr.net/pyodide/v${PYODIDE_VERSION}/full/`;
const WORKDIR = "/home/pyodide/work";

let pyodide = null;
let api = null; // Python dict of helper functions
let stdoutBuffer = [];

function progress(step, label) {
  self.postMessage({ type: "progress", step, label });
}

const PY_GLUE = `
import json, os, sys, traceback

WORKDIR = ${JSON.stringify(WORKDIR)}
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

def _new_session():
    global _session
    _session = Session(cwd=WORKDIR)
    _session.interp.input_func = _no_input
    return _session

_new_session()

def _vars():
    return [v.to_dict() for v in _session.variables()]

def _sync(files):
    """Mirror the user's files into WORKDIR. files: list of {name, content}."""
    wanted = {}
    for f in files or []:
        name = os.path.basename(str(f["name"]))
        if name and not name.startswith("."):
            wanted[name] = str(f["content"])
    changed = False
    for name in os.listdir(WORKDIR):
        p = os.path.join(WORKDIR, name)
        if os.path.isfile(p) and name.endswith(".m") and name not in wanted:
            os.remove(p)
            changed = True
    for name, content in wanted.items():
        p = os.path.join(WORKDIR, name)
        try:
            with open(p, encoding="utf-8") as fh:
                if fh.read() == content:
                    continue
        except OSError:
            pass
        with open(p, "w", encoding="utf-8") as fh:
            fh.write(content)
        changed = True
    if changed:
        # Re-parse edited files even if the clock did not move between writes.
        cache = getattr(_session.interp, "_file_cache", None)
        if isinstance(cache, dict):
            cache.clear()

def _run(code, files_json):
    _sync(json.loads(files_json))
    os.chdir(WORKDIR)
    try:
        res = _session.run(code)
        out = res.to_dict()
    except Exception as e:  # engine bug: report it instead of killing the worker
        out = {"output": "", "error": f"Internal error ({type(e).__name__}): {e}",
               "error_line": None, "figures": [], "internal": traceback.format_exc()}
        try:
            out["figures"] = [__import__("base64").b64encode(f).decode() for f in _session.render_figures()]
        except Exception:
            pass
    try:
        out["variables"] = _vars()
    except Exception:
        out["variables"] = []
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

_api = {
    "run": _run, "vars": lambda: json.dumps(_vars()), "show": _show,
    "complete": _complete, "help": _help, "reset": _reset,
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
    await pyodide.loadPackage(["numpy", "scipy", "matplotlib", "micropip"], {
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

const handlers = {
  run({ code, files }) {
    stdoutBuffer = [];
    const out = JSON.parse(call("run", String(code), JSON.stringify(files || [])));
    if (stdoutBuffer.length) {
      // Text Python printed directly (e.g. library warnings).
      out.output = (out.output || "") + stdoutBuffer.join("\n") + "\n";
      stdoutBuffer = [];
    }
    return out;
  },
  variables() {
    return JSON.parse(call("vars"));
  },
  showVariable({ name }) {
    return call("show", String(name));
  },
  complete({ prefix }) {
    return JSON.parse(call("complete", String(prefix)));
  },
  help({ name }) {
    return call("help", String(name));
  },
  reset() {
    return JSON.parse(call("reset"));
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
    self.postMessage({ id, ok: true, result: h(msg) });
  } catch (err) {
    self.postMessage({ id, ok: false, error: pyErrorMessage(err) });
  }
};

init();
