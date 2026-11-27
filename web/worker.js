/* mylab engine worker: runs the Python interpreter in Pyodide, off the UI thread.
 *
 * Protocol (main -> worker):  { id, type, ...args }
 *   "run"      { code, fs, cwd }            run in the persistent session
 *   "trace"    { code, fs, cwd, maxSteps }  step-through trace in a fresh session
 *   "publish"  { sections, tail, fs, cwd }  run sections one by one in a fresh session
 *   "test"     { files, code }              one autograder test in a clean folder and session
 *   "variables", "showVariable" {name}, "complete" {prefix}, "help" {name}, "reset"
 *   "lint"     { code, known }              problems in one file (mylab.lint.check_source)
 *   "helpIndex"                             { name: [usage lines, one-line description] }
 *   "members"  { name }                     { class, properties, methods } of a workspace variable, or null
 *   "uiEvent"  { event }                    one user action in an app window (mylab.ui); runs callbacks
 *   "uiClose"                               close every app window
 *   "inspect"  { name, path, row, rows, col, cols }  one page of a variable as a grid
 *   "csv"      { name, path, maxCells }     a variable as CSV text
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
// SHA-384 of Pyodide's core files (written by scripts/sri.py). Every one is checked before it runs:
// the two scripts by verifiedScript(), the rest by the browser through the integrity option that
// the fetch wrapper below adds. Packages are checked by Pyodide against the SHA-256 values in
// the (verified) lock file.
const PYODIDE_SRI = {
  // BEGIN SRI (scripts/sri.py)
  "pyodide.js": "sha384-90so5tCKvl0xs9agU29IMKlAVzhfzFX7QO//YxQkRhJG58bBZrFN+2ZTRB026X5X",
  "pyodide.asm.js": "sha384-J3SxzZJh4tkb2bgoUc72zArF45FJ+ngsbiqNsmxqkkecNVpZ/bUHAM0JMp+YPVpx",
  "pyodide.asm.wasm": "sha384-wDap6JvXxf3C8g5beY3kd1xMIjAtw/oTKjTqGltdxc+B9VXnYyYBoM7hveNKilxM",
  "python_stdlib.zip": "sha384-VLeJs79hxVEBNt1hRU5Wo7VmBcyb4sdW564EJAvEWDin3Aa3QXLHm3E5mOg2GiFH",
  "pyodide-lock.json": "sha384-1cFE0bj3+mm1a83kcqiWxoaBhrf6jPPJJORlx1TComnZ0WaHwE4znlohwPXL5tUL"
  // END SRI
};
const WORKDIR = "/home/pyodide/work";
const GRADEDIR = "/home/pyodide/grade";
const MAX_WRITTEN_FILE = 5 * 1024 * 1024; // files written by code that are copied back into the project
const MAX_WRITTEN_TOTAL = 20 * 1024 * 1024;
// Figures are drawn at 100 pixels per inch (a 560×420 figure is 560×420 pixels), times this scale
// for sharp pictures on high-resolution screens. The page passes it as ?scale=2 and shows each
// picture at its pixel size divided by the scale.
const FIG_SCALE = Math.min(2, Math.max(1, Math.round(Number(new URLSearchParams(self.location.search).get("scale")) || 1)));

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

_messages = []

def _dialog_handler(kind, payload):
    """msgbox/errordlg/warndlg/helpdlg are passed to the page; other dialogs use their defaults."""
    if kind == "msgbox" and len(_messages) < 20:
        _messages.append({"message": str(payload.get("message", "")), "title": str(payload.get("title", "")),
                          "icon": str(payload.get("icon", "none"))})
        return True
    return None

def _make_session():
    s = Session(dpi=${100 * FIG_SCALE})
    s.interp.input_func = _no_input
    # Nobody can answer a blocking dialog inside the worker: dialogs return their default
    # answer and print a one-line note (see mylab/builtins/dialogs.py).
    s.interp.state["dialogs_noninteractive"] = True
    return s

def _new_session():
    global _session
    _session = _make_session()
    _session.interp.state["dialog_handler"] = _dialog_handler
    return _session

_new_session()

def _vars():
    out = []
    for v in _session.variables():
        d = v.to_dict()
        if not d["preview"]:
            d["preview"] = _object_preview(_session.interp.base.vars.get(v.name), v.name)
        out.append(d)
    return out

_HEADER = None

def _object_preview(v, name="ans", limit=60):
    """One line for the Workspace panel for objects (fits, interpolants, dates, maps, ...).

    Session.variables() leaves objects without a preview; this one is taken from the printed
    form: the model formula of a fit, the properties of an object, the text of a date."""
    global _HEADER
    import numpy as np
    from mylab.values import Cell, Struct, StringArray, shape_of, class_name
    from mylab.display import format_display
    try:
        if v is None or isinstance(v, (np.ndarray, Cell, Struct, StringArray)) or class_name(v) in ("table", "timetable"):
            return ""
        n = 1
        for k in shape_of(v):
            n *= int(k)
        if n > 12:
            return ""
        if _HEADER is None:
            _HEADER = _re.compile(r"^(\\d+(×\\d+)+ \\S+( array)?|\\S+ with properties:|datetime|duration|calendarDuration|categorical|timeseries|Common Properties:)$")
        lines = [s.strip() for s in format_display(name, v).splitlines() if s.strip()]
        lines = [s for s in lines[1:] if not _HEADER.match(s)]
        if not lines:
            return ""
        if lines[0].startswith("handle to deleted "):
            return lines[0][len("handle to deleted "):] + " (figure closed after drawing)"
        formula = next((s for s in lines if _re.match(r"^\\w+\\([\\w,]*\\) = ", s)), None)
        s = formula or "  ".join(_re.sub(r"\\s+", " ", s) for s in lines)
    except Exception:
        return ""
    return s if len(s) <= limit else s[: limit - 1] + "\\u2026"

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
               "error_line": None, "figures": [], "animations": [], "internal": traceback.format_exc()}
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
    del _messages[:]
    out = _run_session(_session, code)
    out["dialogs"] = list(_messages)
    del _messages[:]
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

# ---- editor helpers: problems, help index, variable viewer ----
import re as _re

def _lint(text, known_json):
    """Problems for one file. Names of project functions and workspace variables are not flagged."""
    from mylab.lint import check_source
    known = set(json.loads(known_json))
    known |= set(_session.interp.base.vars) | set(_session.interp.base.functions)
    out = []
    for d in check_source(text):
        m = _re.match(r"'(\\w+)' is not a function", d.get("message", ""))
        if m:
            if m.group(1) in known:
                continue
            d["name"] = m.group(1)
        out.append(d)
    return json.dumps(out)

def _help_index():
    """name -> [usage lines, one-line description] for every function with help text."""
    from mylab.help import _entries
    out = {}
    for name, entry in _entries().items():
        usage, desc = entry
        usages = list(usage) if isinstance(usage, (list, tuple)) else [usage]
        out[name] = [usages[:5], str(desc).split(". ")[0].rstrip(".") + "."]
    for name in _session.interp.builtins:
        out.setdefault(name, [[], ""])
    return json.dumps(out)

def _cell_text(v, limit=80):
    """Short text for one value inside a grid cell."""
    import numpy as np
    from mylab.display import summary, format_scalar, size_str
    from mylab.values import MString, is_char, to_str, class_name, shape_of
    try:
        if v is None:
            s = "<missing>"
        elif isinstance(v, MatlabError):
            s = "<cannot be read: " + str(v) + ">"
        elif isinstance(v, str):
            s = str(v)
        elif isinstance(v, np.ndarray) and is_char(v) and v.shape[0] <= 1:
            s = to_str(v)
        elif isinstance(v, np.ndarray) and v.size == 1:
            s = format_scalar(v.flat[0]).strip()
        elif isinstance(v, (bool, int, float, complex, np.generic)):
            s = format_scalar(v).strip()
        else:
            s = summary(v, True)
            if s.startswith("<") and " object at " in s:
                s = size_str(shape_of(v)) + " " + class_name(v)
    except Exception:
        s = "?"
    return s if len(s) <= limit else s[: limit - 1] + "\\u2026"

def _num_text(x):
    """A number with full precision, for CSV."""
    import numpy as np
    if isinstance(x, (bool, np.bool_)):
        return "1" if x else "0"
    if isinstance(x, (complex, np.complexfloating)):
        return f"{x.real:.15g}{x.imag:+.15g}i"
    if isinstance(x, (int, np.integer)):
        return str(int(x))
    try:
        return f"{float(x):.15g}"
    except (TypeError, ValueError):
        return str(x)

def _walk_path(v, path):
    """Follow [["f", field, index] | ["c", row, col] | ["t", column] | ["p", property]] steps
    into structs, cells, tables and objects."""
    from mylab.values import Cell, Struct
    for step in path:
        if step[0] == "f" and isinstance(v, Struct):
            v = v.data.flatten(order="F")[int(step[2])][step[1]]
        elif step[0] == "c" and isinstance(v, Cell):
            v = v.data[int(step[1]), int(step[2])]
        elif step[0] == "t" and hasattr(v, "columns"):
            v = v.columns[int(step[1])]
        elif step[0] == "p" and step[1] in _object_props(v):
            v = _prop_value(v, step[1])
        else:
            raise MatlabError("That value is no longer there.")
    return v

def _object_props(v):
    """Public property names of an object (fit, interpolant, triangulation, class instance), else []."""
    import numpy as np
    from mylab.values import Cell, Struct, StringArray
    if v is None or isinstance(v, (np.ndarray, Cell, Struct, StringArray, str, bool, int, float, complex)):
        return []
    try:
        from mylab.builtins.objfuncs import _member_lists
        return [str(p) for p in _member_lists(_session.interp, v)[1]]
    except Exception:
        return []

def _prop_value(v, name):
    try:
        return v.get_field(_session.interp, name)
    except Exception as e:
        return MatlabError(str(e))

def _members(name):
    """Properties and methods of a workspace variable, for completions after 'name.'."""
    from mylab.values import Struct, class_name
    try:
        v = _session.interp.get_var(name)
    except Exception:
        return json.dumps(None)
    if isinstance(v, Struct):
        return json.dumps({"class": "struct", "properties": list(v.fields), "methods": []})
    try:
        from mylab.builtins.objfuncs import _member_lists
        cls, props, methods = _member_lists(_session.interp, v)
    except Exception:
        return json.dumps(None)
    return json.dumps({"class": class_name(v), "properties": [str(p) for p in props], "methods": [str(m) for m in methods]})

def _grid_of(v):
    """(kind, rows, cols, colnames, rownames, get(r, c), note, plottable), or None for values shown as text."""
    import numpy as np
    from mylab.values import Cell, Struct, StringArray, is_char, class_name
    if isinstance(v, MatlabError):
        return None
    if class_name(v) in ("table", "timetable") and hasattr(v, "columns"):
        names, cols = list(v.names), list(v.columns)
        n = v.nrows
        rownames = v.row_names
        if class_name(v) == "timetable" and hasattr(v.row_times, "element_texts"):
            rownames = _LazyTexts(v.row_times)  # the row times label the rows
        def elem(col, r):
            if isinstance(col, np.ndarray):
                if is_char(col):
                    return "".join(str(x) for x in np.atleast_2d(col)[r])
                row = col[r]
                return row if np.size(row) == 1 else np.atleast_2d(row)
            if isinstance(col, (Cell, StringArray)):
                return col.data[r, 0] if col.data.shape[1] == 1 else Cell(col.data[r:r + 1, :])
            try:
                from mylab.display import display_lines
                from mylab.builtins.table import _rows
                return " ".join(s.strip() for s in display_lines(_rows(col, np.array([r]))) if s.strip())
            except Exception:
                return col
        return ("table", n, len(cols), names, rownames, lambda r, c: elem(cols[c], r), "", False)
    if isinstance(v, Struct):
        items = v.data.flatten(order="F")
        fields = list(v.fields)
        if len(items) == 1:
            return ("struct", len(fields), 1, ["Value"], fields, lambda r, c: items[0][fields[r]], "", False)
        return ("structarray", len(items), len(fields), fields, None, lambda r, c: items[r][fields[c]],
                "Each row is one element of the struct array.", False)
    if isinstance(v, Cell):
        d = v.data
        return ("cell", d.shape[0], d.shape[1], None, None, lambda r, c: d[r, c], "", False)
    if isinstance(v, StringArray):
        d = v.data
        return ("string", d.shape[0], d.shape[1], None, None, lambda r, c: d[r, c], "", False)
    if isinstance(v, np.ndarray) and v.dtype != object and v.size > 1 and not (is_char(v) and v.shape[0] <= 1):
        note = ""
        a = v
        if a.ndim > 2:
            note = "This array has more than two dimensions. Showing the first page, (:, :, 1)."
            a = a.reshape(a.shape[0], a.shape[1], -1)[:, :, 0]
        elif a.ndim < 2:
            a = np.atleast_2d(a)
        if is_char(v):
            return ("char", a.shape[0], 1, ["Text"], None, lambda r, c: "".join(str(x) for x in a[r]), note, False)
        return ("matrix", a.shape[0], a.shape[1], None, None, lambda r, c: a[r, c], note, v.ndim <= 2 and a.dtype.kind in "fiub")
    if hasattr(v, "element_texts") and isinstance(getattr(v, "data", None), np.ndarray) and v.data.ndim == 2 and v.data.size > 1:
        rows = _LazyTexts(v)  # datetime, duration and categorical arrays: one text per element
        return ("values", v.data.shape[0], v.data.shape[1], None, None, lambda r, c: rows.row(r)[c], "", False)
    props = _object_props(v)
    if props:
        note = ""
        if any(isinstance(_prop_value(v, p), MatlabError) for p in props[:1]):
            note = "Some properties could not be read."
        return ("object", len(props), 1, ["Value"], props, lambda r, c: _prop_value(v, props[r]), note, False)
    return None

class _LazyTexts:
    """The element texts of a datetime, duration or categorical array, one row at a time."""
    def __init__(self, arr):
        self.arr, self.cache = arr, {}
    def row(self, r):
        if r not in self.cache:
            if len(self.cache) > 400:
                self.cache.clear()
            self.cache[r] = [str(t) for t in self.arr[r:r + 1].element_texts().ravel()]
        return self.cache[r]
    def __getitem__(self, key):
        if isinstance(key, slice):
            return [self.row(r)[0] for r in range(*key.indices(self.arr.data.shape[0]))]
        return self.row(key)[0]

def _can_open(x):
    """Can the viewer step into this value? Grids, and objects (shown as printed)."""
    import numpy as np
    from mylab.values import MString
    if x is None or isinstance(x, (MatlabError, str, bool, int, float, complex, np.generic, MString)):
        return False
    if isinstance(x, np.ndarray):
        return _grid_of(x) is not None
    return True

def _inspect(name, path_json, r0, nr, c0, nc):
    """One page of a variable as a grid of short texts (or its printed form when it is not a grid)."""
    from mylab.values import class_name, shape_of
    from mylab.display import size_str, format_display
    path = json.loads(path_json or "[]")
    v = _walk_path(_session.interp.get_var(name), path)
    out = {"class": class_name(v), "size": size_str(shape_of(v))}
    g = _grid_of(v)
    if g is None:
        out["kind"] = "text"
        out["text"] = str(v) if isinstance(v, MatlabError) else format_display("ans" if path else name, v)
        if out["class"].startswith("matlab.graphics."):
            out["note"] = ("A graphics object. Figures are drawn into the Plots panel and closed after each run, "
                           "so a handle kept in a variable reads as deleted afterwards.")
        return json.dumps(out)
    kind, rows, cols, colnames, rownames, get, note, plottable = g
    r0, c0 = max(0, int(r0)), max(0, int(c0))
    r1, c1 = min(rows, r0 + int(nr)), min(cols, c0 + int(nc))
    cells, opens = [], []
    for r in range(r0, r1):
        row = []
        for c in range(c0, c1):
            x = get(r, c)
            row.append(_cell_text(x))
            if kind in ("struct", "structarray", "cell", "object") and _can_open(x):
                opens.append([r, c])
        cells.append(row)
    out.update(kind=kind, rows=rows, cols=cols, r0=r0, c0=c0, cells=cells, opens=opens, note=note, plottable=plottable,
               colnames=colnames[c0:c1] if colnames else None, rownames=rownames[r0:r1] if rownames else None)
    return json.dumps(out)

def _csv(name, path_json, max_cells):
    """The whole value as CSV text (numbers with full precision)."""
    import numpy as np
    path = json.loads(path_json or "[]")
    v = _walk_path(_session.interp.get_var(name), path)
    g = _grid_of(v)
    if g is None:
        return json.dumps({"csv": _cell_text(v, 10 ** 6) + "\\n", "truncated": False, "rows": 1})
    kind, rows, cols, colnames, rownames, get, note, plottable = g
    def q(s):
        s = str(s)
        return '"' + s.replace('"', '""') + '"' if any(ch in s for ch in ',"\\n\\r') else s
    def text(x):
        if isinstance(x, np.ndarray) and x.size == 1 and x.dtype.kind in "fiubc":
            return _num_text(x.flat[0])
        if isinstance(x, (bool, int, float, complex, np.generic)):
            return _num_text(x)
        return q(_cell_text(x, 10 ** 5))
    limit = max(1, int(max_cells) // max(1, cols))
    lines = []
    if colnames and kind != "struct":
        lines.append(",".join(([""] if rownames else []) + [q(n) for n in colnames]))
    for r in range(min(rows, limit)):
        lines.append(",".join(([q(rownames[r])] if rownames else []) + [text(get(r, c)) for c in range(cols)]))
    return json.dumps({"csv": "\\n".join(lines) + "\\n", "truncated": rows > limit, "rows": min(rows, limit)})

# -- app windows (uifigure, uicontrol): see mylab/ui.py --
def _ui_event(event_json):
    before = _scan()
    del _messages[:]
    try:
        out = _session.ui_event(json.loads(event_json))
    except Exception as e:  # engine bug: report it instead of killing the worker
        out = {"output": "", "error": f"Internal error ({type(e).__name__}): {e}", "update": None,
               "figures": [], "animations": [], "internal": traceback.format_exc()}
    out["dialogs"] = list(_messages)
    del _messages[:]
    try:
        out["variables"] = _vars()
    except Exception:
        out["variables"] = []
    after = _scan()
    out["_changed"] = [[p, v[1]] for p, v in after.items() if before.get(p) != v]
    out["_present"] = list(after)
    return json.dumps(out)

def _ui_close():
    _session.ui_close()
    return "null"

# -- mylab Blocks: the diagram editor's requests (see mylab/blocks/web.py) --
def _blocks(op, args_json, cwd, emit):
    from mylab.blocks import web
    _enter(_session, cwd)
    out = json.loads(web.handle(_session, op, args_json, emit))
    if op == "run":
        out["variables"] = _vars()
    return json.dumps(out)

_api = {
    "ui_event": _ui_event, "ui_close": _ui_close,
    "blocks": _blocks,
    "run": _run, "trace": _trace, "publish": _publish, "test": _test, "wipe": _wipe,
    "vars": lambda: json.dumps(_vars()), "show": _show,
    "complete": _complete, "help": _help, "reset": _reset, "cwd": _cwd,
    "clear_cache": lambda: _session.clear_file_cache(),
    "lint": _lint, "help_index": _help_index, "inspect": _inspect, "csv": _csv, "members": _members,
    "version": getattr(mylab, "__version__", "?"),
}
_api
`;

/** Download one of Pyodide's scripts, let the browser check its hash, and run it. */
async function verifiedScript(name) {
  const resp = await fetch(PYODIDE_URL + name, { integrity: PYODIDE_SRI[name] });
  if (!resp.ok) throw new Error(`Could not download ${name} (HTTP ${resp.status}).`);
  const url = URL.createObjectURL(new Blob([await resp.text()], { type: "text/javascript" }));
  try {
    importScripts(url);
  } finally {
    URL.revokeObjectURL(url);
  }
}

// Pyodide downloads its other core files with fetch: add their hashes, so that the browser refuses a
// file that differs from the one the app was tested with.
const nativeFetch = self.fetch.bind(self);
self.fetch = (input, init) => {
  const url = typeof input === "string" ? input : input instanceof URL ? input.href : input.url;
  const name = url.startsWith(PYODIDE_URL) ? url.slice(PYODIDE_URL.length).split(/[?#]/)[0] : null;
  if (name && PYODIDE_SRI[name] && !(init && init.integrity)) init = { ...(init || {}), integrity: PYODIDE_SRI[name] };
  return nativeFetch(input, init);
};

async function init() {
  try {
    progress("python", "Downloading Python (Pyodide)…");
    await verifiedScript("pyodide.js");
    await verifiedScript("pyodide.asm.js"); // defines _createPyodideModule, so loadPyodide does not fetch it again
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
    // deps=False must be a keyword argument (a JS object would be taken as the next positional argument):
    // the requirements are loaded above, and h5py only when a program needs it.
    await micropip.install.callKwargs(wheelUrl, { deps: false });

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
  noteProjectM(fs);
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
  // mylab Blocks: op is "run", "check", "validate" or "import_slx"; scope data is streamed as "event" messages.
  blocks({ op, args, fs, cwd }) {
    syncFiles(fs);
    stdoutBuffer = [];
    const emit = (text) => self.postMessage({ type: "event", name: "blocks", data: JSON.parse(text) });
    const out = withStdout(JSON.parse(call("blocks", String(op), JSON.stringify(args || {}), cwd === undefined ? null : cwd, emit)));
    return { result: out };
  },
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
  lint({ code, known }) {
    return { result: JSON.parse(call("lint", String(code), JSON.stringify(known || []))) };
  },
  helpIndex() {
    return { result: JSON.parse(call("help_index")) };
  },
  members({ name }) {
    return { result: JSON.parse(call("members", String(name))) };
  },
  inspect({ name, path, row, rows, col, cols }) {
    return { result: JSON.parse(call("inspect", String(name), JSON.stringify(path || []), row || 0, rows || 50, col || 0, cols || 20)) };
  },
  // app windows
  uiEvent({ event }) {
    stdoutBuffer = [];
    const out = withStdout(JSON.parse(call("ui_event", JSON.stringify(event || {}))));
    const transfer = collectWritten(out);
    return { result: out, transfer };
  },
  uiClose() {
    return { result: JSON.parse(call("ui_close")) };
  },
  csv({ name, path, maxCells }) {
    return { result: JSON.parse(call("csv", String(name), JSON.stringify(path || []), maxCells || 200000)) };
  },
};

/* ---------- optional packages ---------- */

// SymPy (symbolic math) is a large download, so it is fetched the first time code uses it.
const SYMBOLIC_RE =
  /\b(?:syms|sym|str2sym|symfun|vpa|vpasolve|dsolve|poly2sym|piecewise|assume|assumeAlso|laplace|ilaplace|fourier|ifourier|ztrans|iztrans|symsum|symprod|taylor|solve|digits|heaviside|dirac)\b/;
let sympyState = "none"; // none | loaded | failed

// h5py (HDF5 files and version 7.3 MAT-files) is also fetched on first use: by an h5* function,
// matfile or '-v7.3' in the code, or by load/whos/... while the project holds an HDF5 file.
const HDF5_RE =
  /\b(?:h5create|h5write|h5read|h5writeatt|h5readatt|h5info|h5disp|matfile|nccreate|ncwrite|ncread|ncinfo|ncdisp|ncreadatt|ncwriteatt)\b|\bH5(?:F|D|S|G|A|T|ML)\.\w|v7\.3/;
const HDF5_READERS_RE = /\b(?:load|who|whos|importdata|open|uiimport)\b/;
const HDF5_MAGIC = [0x89, 0x48, 0x44, 0x46, 0x0d, 0x0a, 0x1a, 0x0a];
const hdf5Files = new Set(); // project paths whose content is an HDF5 file
let h5pyState = "none"; // none | loaded | failed

// OpenCV (Pyodide's opencv-python package) backs most computer vision functions; it is a large
// download too, so it is fetched the first time code calls one of the functions that need it.
const VISION_RE =
  /\b(?:detect(?:Harris|MinEigen|FAST|ORB|SIFT|BRISK|KAZE|MSER)Features|extractFeatures|estimateEssentialMatrix|insert(?:Shape|Marker|Text|ObjectAnnotation)|detectCheckerboardPoints|estimateCameraParameters|undistortImage|rectifyStereoImages|disparity(?:SGM|BM)|reconstructScene|estimateExtrinsics|extrinsics|estworldpose|CascadeObjectDetector|PeopleDetector|peopleDetector|yolo(?:v4|x|v8)ObjectDetector|PointTracker|opticalFlowFarneback)\b/;
let opencvState = "none"; // none | loaded | failed

function isHdf5(bytes) {
  if (!(bytes instanceof Uint8Array)) return false;
  // the signature is at the start, or after the 512-byte header of a version 7.3 MAT-file
  return [0, 512].some((at) => bytes.length >= at + 8 && HDF5_MAGIC.every((b, k) => bytes[at + k] === b));
}

function requestFiles(msg) {
  return [...((msg.fs && msg.fs.files) || []), ...(msg.files || [])];
}

// The .m files of the project, as last sent (a request only carries the files that changed).
const projectM = new Map(); // path -> text

/**
 * The code a request runs, and the project's .m files it can reach: those it calls by name,
 * and those that they call, and so on. Files nobody calls (an example in another folder) do
 * not make the app download SymPy, h5py or OpenCV. An autograder test reaches all of its files.
 */
function programTexts(msg) {
  const texts = [msg.code, msg.tail];
  for (const s of msg.sections || []) texts.push(typeof s === "string" ? s : s && s.code);
  for (const f of msg.files || []) {
    if (typeof f.content === "string" && /\.m$/i.test(f.path || "")) texts.push(f.content);
  }
  noteProjectM(msg.fs);
  const out = texts.filter((t) => typeof t === "string");
  const left = [...projectM].map(([p, t]) => [p.slice(p.lastIndexOf("/") + 1, -2), t]).filter(([name]) => /^[A-Za-z]\w*$/.test(name));
  for (let grew = true; grew && left.length; ) {
    grew = false;
    const all = out.join("\n");
    for (let i = left.length - 1; i >= 0; i--) {
      if (!new RegExp("\\b" + left[i][0] + "\\b").test(all)) continue;
      out.push(left[i][1]);
      left.splice(i, 1);
      grew = true;
    }
  }
  return out;
}

/** Keep projectM in step with the project mirrored by a request's fs (see syncFiles). */
function noteProjectM(fs) {
  if (!fs) return;
  if (fs.reset) projectM.clear();
  const listed = new Set((fs.files || []).map((f) => f.path));
  for (const p of [...projectM.keys()]) if (!listed.has(p)) projectM.delete(p);
  for (const f of fs.files || []) {
    if (typeof f.content === "string" && /\.m$/i.test(f.path || "")) projectM.set(f.path, f.content);
  }
}

function usesHdf5(msg, texts) {
  for (const f of requestFiles(msg)) {
    if (f.content === undefined || f.content === null) continue;
    if (isHdf5(f.content)) hdf5Files.add(f.path);
    else hdf5Files.delete(f.path);
  }
  return texts.some((t) => HDF5_RE.test(t) || (hdf5Files.size > 0 && HDF5_READERS_RE.test(t)));
}

async function loadPackageOnce(name, notice) {
  self.postMessage({ type: "notice", text: notice });
  try {
    await pyodide.loadPackage(name, { messageCallback: () => {}, errorCallback: () => {} });
    return "loaded";
  } catch {
    return "failed"; // the program still runs; the functions that need the package say it is missing
  }
}

async function loadOptional(msg) {
  const texts = programTexts(msg);
  if (sympyState === "none" && texts.some((t) => SYMBOLIC_RE.test(t))) {
    sympyState = await loadPackageOnce("sympy", "Loading symbolic math…");
  }
  if (h5pyState === "none" && usesHdf5(msg, texts)) {
    h5pyState = await loadPackageOnce("h5py", "Loading HDF5 support…");
  }
  if (opencvState === "none" && texts.some((t) => VISION_RE.test(t))) {
    opencvState = await loadPackageOnce("opencv-python", "Loading computer vision (OpenCV)…");
  }
}

let queue = Promise.resolve(); // requests run one at a time, in the order they arrive

self.onmessage = (ev) => {
  queue = queue.then(() => handle(ev.data || {}));
};

async function handle(msg) {
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
  if (type === "run" || type === "trace" || type === "publish" || type === "test") await loadOptional(msg);
  try {
    const { result, transfer } = h(msg);
    self.postMessage({ id, ok: true, result }, transfer || []);
  } catch (err) {
    self.postMessage({ id, ok: false, error: pyErrorMessage(err) });
  }
}

init();
