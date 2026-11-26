"""Data files: MAT-files (save, load, matfile), HDF5, NetCDF, XML, struct files and memory maps.

The MAT container code lives in ``_matio`` (Level 5 files), ``_mat73`` (HDF5-based version 7.3
files, through h5py) and ``_matcodec`` (how values without a standard MAT type are stored).
"""

from __future__ import annotations

import fnmatch
import os
import re
from typing import Any

import numpy as np

from ..errors import MatlabError
from ..objects import MObject
from ..values import (
    Cell, FunctionHandle, MString, StringArray, Struct, char, class_name, shape_of,
)
from ._registry import builtin
from ._util import bool_scalar, is_text, text

# -- locating files --------------------------------------------------------------------------------


def find_file(interp, name: str, extensions: tuple = ()) -> str | None:
    """The file as named, with one of the extensions added, or the same on the search path."""
    name = os.path.expanduser(name)
    tries = [name] + ([name + e for e in extensions] if not os.path.splitext(name)[1] else [])
    for t in tries:
        if os.path.isfile(t):
            return t
    if not os.path.isabs(name):
        for d in getattr(interp, "path", []):
            for t in tries:
                if os.path.isfile(os.path.join(d, t)):
                    return os.path.join(d, t)
    return None


def _valid_name(stem: str) -> str:
    s = re.sub(r"\W", "_", stem)
    if not s or not (s[0].isalpha()):
        s = "X" + s
    return s


# -- which variables -------------------------------------------------------------------------------

def workspace_names(frame) -> list[str]:
    names = set(frame.vars) | {n for n, store in frame.globals.items() if n in store}
    shared = getattr(frame, "shared", None)
    if shared:
        names |= {n for n in shared if frame.parent.has(n)}
    return sorted(names)


def _regex(pattern: str):
    try:
        return re.compile(pattern)
    except re.error as e:
        raise MatlabError(f"Invalid regular expression '{pattern}': {e}.") from None


def select_names(available: list[str], patterns: list[str], regexps: list[str]) -> tuple[list[str], list[str]]:
    """(chosen names in the order of ``available``, patterns that matched nothing)."""
    if not patterns and not regexps:
        return list(available), []
    chosen: set[str] = set()
    missing = []
    for p in patterns:
        hits = [n for n in available if fnmatch.fnmatchcase(n, p)] if any(c in p for c in "*?[") else \
            [n for n in available if n == p]
        if not hits:
            missing.append(p)
        chosen.update(hits)
    compiled = [_regex(r) for r in regexps]
    for n in available:
        if any(c.search(n) for c in compiled):
            chosen.add(n)
    return [n for n in available if n in chosen], missing


_SAVE_FLAGS = {"-append", "-ascii", "-double", "-tabs", "-mat", "-nocompression"}
_VERSIONS = {"-v4": "4", "-v6": "6", "-v7": "7", "-v7.3": "7.3"}


def _words(args: list, fn: str) -> list[str]:
    out = []
    for a in args:
        if not is_text(a):
            raise MatlabError(f"{fn}: the file name, variable names and options must be text. To save a value "
                              "that is not in a variable, assign it to one first.")
        out.append(text(a))
    return out


def _parse_options(words: list[str], fn: str, flags: set[str]) -> dict:
    out: dict = {"file": None, "names": [], "regexp": [], "struct": None, "fields": [], "flags": set(),
                 "version": None}
    mode = "names"
    for k, w in enumerate(words):
        low = w.lower()
        if w.startswith("-") and len(w) > 1:
            if low in flags:
                out["flags"].add(low[1:])
                mode = "names"
            elif low in _VERSIONS and fn == "save":
                out["version"] = _VERSIONS[low]
                mode = "names"
            elif low == "-struct" and fn == "save":
                mode = "struct"
            elif low == "-regexp":
                mode = "regexp"
            else:
                raise MatlabError(f"{fn}: unknown option '{w}'.")
            continue
        if k == 0:
            out["file"] = w
        elif mode == "struct":
            if out["struct"] is None:
                out["struct"] = w
            else:
                out["fields"].append(w)
        elif mode == "regexp":
            out["regexp"].append(w)
        else:
            out["names"].append(w)
    return out


# -- save ----------------------------------------------------------------------------------------

def _gather(interp, opts: dict) -> tuple[dict, set[str]]:
    """The variables a save call names: {name: value} and the names that are global."""
    frame = interp.frame
    if opts["struct"] is not None:
        sname = opts["struct"]
        if not frame.has(sname):
            raise MatlabError(f"save: variable '{sname}' not found.", "MATLAB:save:couldNotWriteFile")
        s = frame.get(sname)
        if not isinstance(s, Struct) or s.data.size != 1:
            raise MatlabError(f"save: the argument after -struct must name a scalar structure variable; "
                              f"'{sname}' is not one.")
        chosen, missing = select_names(list(s.fields), opts["fields"], opts["regexp"])
        if missing:
            raise MatlabError(f"save: the structure '{sname}' has no field '{missing[0]}'.")
        return {f: s.data.flat[0][f] for f in chosen}, set()
    chosen, missing = select_names(workspace_names(frame), opts["names"], opts["regexp"])
    if missing:
        raise MatlabError(f"save: variable '{missing[0]}' not found.", "MATLAB:save:couldNotWriteFile")
    is_global = {n for n in chosen if frame.globals.get(n) is interp.global_vars}
    return {n: frame.get(n) for n in chosen}, is_global


def _ascii_rows(name: str, v: Any, interp) -> np.ndarray:
    from .sparse import SparseMatrix
    if isinstance(v, SparseMatrix):
        v = v.to_dense()
    if isinstance(v, MString):
        v = char(str(v))
    if not isinstance(v, np.ndarray) or v.dtype.kind == "O":
        raise MatlabError(f"save: variable '{name}' of class '{class_name(v)}' cannot be written to an ASCII "
                          "file; only numeric, logical and char arrays can.", "MATLAB:save:unsupportedType")
    if v.ndim > 2:
        raise MatlabError(f"save: variable '{name}' has more than two dimensions and cannot be written to an "
                          "ASCII file.")
    if v.dtype.kind == "U":
        from ..values import char_codes
        v = char_codes(v)
    if v.dtype.kind == "c":
        interp.warn(f"The imaginary part of variable '{name}' is not written to an ASCII file.")
        v = v.real
    return v.astype(float)


def _write_ascii(interp, path: str, variables: dict, flags: set[str]) -> None:
    digits = 16 if "double" in flags else 7
    width = digits + 9
    lines = []
    for name, v in variables.items():
        for row in _ascii_rows(name, v, interp):
            cells = []
            for x in row:
                if np.isnan(x):
                    t = "NaN"
                elif np.isinf(x):
                    t = "Inf" if x > 0 else "-Inf"
                else:
                    t = f"{x:.{digits}e}"
                cells.append(t)
            lines.append("\t".join(cells) if "tabs" in flags else "".join(c.rjust(width) for c in cells))
    try:
        with open(path, "a" if "append" in flags else "w", encoding="ascii", newline="\n") as fh:
            fh.write("".join(ln + "\n" for ln in lines))
    except OSError as e:
        raise MatlabError(f"save: unable to write the file '{path}': {e.strerror or e}.") from None


def _write_v4(path: str, variables: dict) -> None:
    from scipy.io import savemat

    from .sparse import SparseMatrix
    data = {}
    for name, v in variables.items():
        if isinstance(v, SparseMatrix):
            data[name] = v.m.astype(float)
        elif isinstance(v, np.ndarray) and v.ndim == 2 and v.dtype.kind != "O":
            data[name] = v if v.dtype.kind != "U" else np.array(["".join(r) for r in v])
        else:
            raise MatlabError(f"save: variable '{name}' of class '{class_name(v)}' cannot be stored in a version 4 "
                              "MAT-file, which holds only two-dimensional numeric, char and sparse arrays.")
    try:
        savemat(path, data, format="4")
    except (OSError, ValueError, TypeError) as e:
        raise MatlabError(f"save: unable to write the file '{path}': {e}.") from None


def write_mat(interp, path: str, variables: dict, is_global: set[str] = frozenset(), version: str | None = None,
              append: bool = False, compress: bool = True) -> None:
    """Write variables (mylab values) to a MAT-file, replacing it or adding to it."""
    from . import _mat73, _matio
    from ._matcodec import Encoder
    enc = Encoder(interp)
    encoded = {name: enc.encode(v, name) for name, v in variables.items()}
    existing = None
    if append and os.path.isfile(path) and os.path.getsize(path) > 0:
        with open(path, "rb") as fh:
            existing = fh.read()
        if _mat73.is_hdf5(existing[:520]):
            if version not in (None, "7.3"):
                raise MatlabError("save: cannot append in another format to a version 7.3 MAT-file.")
            version = "7.3"
            existing = None
        elif not _matio.MatFile5.is_level5(existing[:128]):
            raise MatlabError(f"save: cannot append to '{path}': it is not a MAT-file that mylab can extend "
                              "(version 5, 6, 7 or 7.3).")
        elif version == "7.3":
            raise MatlabError("save: cannot append version 7.3 data to an older MAT-file.")
    if version == "7.3":
        _mat73.write(path, encoded, append)
        return
    if version == "4":
        _write_v4(path, encoded)
        return
    parts = []
    subsys_at = 0
    pos = 128
    if existing is not None:
        old = _matio.MatFile5(existing)
        if old.endian != "<":
            raise MatlabError(f"save: cannot append to '{path}': it was written with the opposite byte order.")
        if old.damage:
            raise MatlabError(f"save: cannot append to '{path}': the file is damaged ({old.damage}).")
        for e in old.elements():
            if e["name"] in encoded and not e["subsystem"]:
                continue  # replaced below
            raw = old.raw_element(e)
            if e["subsystem"]:
                subsys_at = pos
            parts.append(raw)
            pos += len(raw)
        header = bytearray(existing[:128])
    else:
        header = bytearray(_matio.file_header())
    for name, v in encoded.items():
        parts.append(_matio.top_level(_matio.encode_array(v, name, name in is_global), compress))
    if existing is not None:
        import struct
        header[116:124] = struct.pack("<Q", subsys_at) if subsys_at else b" " * 8
    try:
        tmp = path + ".tmp~" if existing is not None else path
        with open(tmp, "wb") as fh:
            fh.write(bytes(header))
            for p in parts:
                fh.write(p)
        if tmp != path:
            os.replace(tmp, path)
    except OSError as e:
        raise MatlabError(f"save: unable to write the file '{path}': {e.strerror or e}.") from None


@builtin("save")
def _save(interp, args, nargout):
    opts = _parse_options(_words(args, "save"), "save", _SAVE_FLAGS)
    flags = opts["flags"]
    fname = os.path.expanduser(opts["file"] or "matlab.mat")
    variables, is_global = _gather(interp, opts)
    if "ascii" in flags:
        if opts["version"] or "mat" in flags:
            raise MatlabError("save: -ascii cannot be combined with -mat or a MAT-file version.")
        _write_ascii(interp, fname, variables, flags)
        return None
    if "double" in flags or "tabs" in flags:
        raise MatlabError("save: -double and -tabs are only used together with -ascii.")
    if not os.path.splitext(fname)[1]:
        fname += ".mat"
    compress = "nocompression" not in flags and opts["version"] not in ("6",)
    write_mat(interp, fname, variables, is_global, opts["version"], "append" in flags, compress)
    return None


# -- load ----------------------------------------------------------------------------------------

def _file_kind(path: str) -> str:
    from . import _mat73, _matio
    with open(path, "rb") as fh:
        head = fh.read(520)
    if _mat73.is_hdf5(head):
        return "7.3"
    if _matio.MatFile5.is_level5(head) and head[124:126] in (b"\x00\x01", b"\x01\x00"):
        return "5"
    if len(head) >= 20 and 0 in head[:4] and not head[:20].strip(b"\0\t\r\n ").isdigit():
        return "4"
    return "ascii"


def _read_v4(path: str) -> dict:
    import warnings

    import scipy.sparse as sp
    from scipy.io import loadmat

    from .sparse import SparseMatrix
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            raw = loadmat(path, mat_dtype=True, chars_as_strings=False, squeeze_me=False)
    except Exception as e:  # scipy raises several unrelated types for unreadable files
        raise MatlabError(f"Unable to read the file '{path}': it is not a MAT-file mylab understands "
                          f"({type(e).__name__}: {e}).") from None
    out = {}
    for k, v in raw.items():
        if k.startswith("__"):
            continue
        if sp.issparse(v):
            out[k] = SparseMatrix(v)
        elif isinstance(v, np.ndarray) and v.dtype.kind == "U":
            out[k] = np.atleast_2d(v).astype("<U1") if v.size else np.zeros((0, 0), dtype="<U1")
        elif isinstance(v, np.ndarray):
            a = np.atleast_2d(v)
            out[k] = np.ascontiguousarray(a.astype(a.dtype.newbyteorder("=")))
    return out


class MatSource:
    """One MAT-file opened for reading: its variable list and values (any supported version)."""

    def __init__(self, interp, path: str, kind: str | None = None):
        self.interp = interp
        self.path = path
        self.kind = kind or _file_kind(path)
        if self.kind == "ascii":
            raise MatlabError(f"Unable to read the MAT-file '{path}': it is not a binary MAT-file. "
                              "Use load(filename, '-ascii') for text files.", "MATLAB:load:notBinaryFile")
        self._mat5 = None
        self._v4: dict | None = None
        if self.kind == "5":
            from . import _matio
            with open(path, "rb") as fh:
                self._mat5 = _matio.MatFile5(fh.read())
            found = self._mat5.variables()  # reading the list is what notices a damaged file
            if self._mat5.damage and not found:
                raise MatlabError(f"Unable to read the MAT-file '{path}': the file is damaged "
                                  f"({self._mat5.damage}).", "MATLAB:load:unableToReadMatFile")
            if self._mat5.damage:
                interp.warn(f"The MAT-file '{path}' is damaged ({self._mat5.damage}); only the variables before "
                            "the damage were read.", "MATLAB:load:unableToReadMatFile")
        elif self.kind == "4":
            self._v4 = _read_v4(path)

    def names(self) -> list[str]:
        if self._mat5 is not None:
            return [e["name"] for e in self._mat5.variables()]
        if self._v4 is not None:
            return list(self._v4)
        from . import _mat73
        return _mat73.names(self.path)

    def read(self, names: list[str]) -> tuple[dict, set[str]]:
        """({name: value}, names stored as global) for the given variables."""
        from ._matcodec import Decoder
        wanted = set(names)
        out: dict = {}
        is_global: set[str] = set()
        if self._mat5 is not None:
            from . import _matio
            mf = self._mat5
            dec = None
            for e in mf.variables():
                if e["name"] not in wanted:
                    continue
                raw = mf.read(e)
                if dec is None:
                    dec = Decoder(self.interp, mf.subsystem() if mf.subsys_offset else None)
                dec.var = e["name"]
                out[e["name"]] = dec.decode(raw)
                if e["flags"] & _matio.FLAG_GLOBAL:
                    is_global.add(e["name"])
            return out, is_global
        if self._v4 is not None:
            return {k: v for k, v in self._v4.items() if k in wanted}, is_global
        from . import _mat73
        raw = _mat73.read(self.path, lambda n: n in wanted,
                          lambda msg: self.interp.warn(f"load: {msg}", "MATLAB:load:classNotFound"))
        dec = Decoder(self.interp)
        for k, v in raw.items():
            dec.var = k
            out[k] = dec.decode(v)
        return out, is_global


def _read_ascii(path: str) -> np.ndarray:
    try:
        with open(path, encoding="utf-8-sig", errors="replace") as fh:
            lines = fh.read().splitlines()
    except OSError as e:
        raise MatlabError(f"Unable to read the file '{path}': {e.strerror or e}.") from None
    rows: list[list[float]] = []
    width = None
    for k, raw in enumerate(lines, 1):
        ln = raw.split("%", 1)[0].strip()
        if not ln:
            continue
        toks = [t for t in re.split(r"[\s,;]+", ln) if t]
        vals = []
        for t in toks:
            try:
                vals.append(float(t.replace("D", "e").replace("d", "e")))
            except ValueError:
                raise MatlabError(f"Unable to read the file '{path}': it is not a MAT-file, and line {k} "
                                  f"('{raw.strip()[:40]}') is not numeric text.",
                                  "MATLAB:load:unknownText") from None
        if width is None:
            width = len(vals)
        elif len(vals) != width:
            raise MatlabError(f"Unable to read the file '{path}': line {k} has {len(vals)} columns but the "
                              f"lines before it have {width}. An ASCII file must have the same number of "
                              "columns in every row.", "MATLAB:load:numColumnsNotSame")
        rows.append(vals)
    if not rows:
        return np.zeros((0, 0))
    return np.array(rows, dtype=float)


@builtin("load")
def _load(interp, args, nargout):
    if not args:
        words = ["matlab.mat"]
    else:
        words = _words(args, "load")
    opts = _parse_options(words, "load", {"-mat", "-ascii"})
    name = opts["file"]
    if name is None:
        raise MatlabError("load: a file name is needed, for example load('data.mat').")
    path = find_file(interp, name, (".mat",))
    if path is None:
        raise MatlabError(f"Unable to find file or directory '{name}'.", "MATLAB:load:couldNotReadFile")
    if "ascii" in opts["flags"]:
        kind = "ascii"
    else:
        kind = _file_kind(path)
        if kind == "ascii" and ("mat" in opts["flags"] or path.lower().endswith(".mat")):
            raise MatlabError(f"Unable to read the MAT-file '{path}': it is not a binary MAT-file.",
                              "MATLAB:load:notBinaryFile")
    if kind == "ascii":
        data = _read_ascii(path)
        if nargout:
            return data
        interp.frame.set(_valid_name(os.path.splitext(os.path.basename(path))[0]), data)
        return None
    src = MatSource(interp, path, kind)
    chosen, missing = select_names(src.names(), opts["names"], opts["regexp"])
    for m in missing:
        interp.warn(f"Variable '{m}' not found.", "MATLAB:load:variableNotFound")
    values, is_global = src.read(chosen)
    ordered = {n: values[n] for n in chosen if n in values}
    if nargout:
        return Struct.scalar(ordered) if ordered else Struct()
    frame = interp.frame
    for k, v in ordered.items():
        if k in is_global:
            interp.global_vars[k] = v
            frame.vars.pop(k, None)
            frame.globals[k] = interp.global_vars
        else:
            frame.set(k, v)
    return None


# -- sizes -----------------------------------------------------------------------------------------

def value_bytes(v: Any) -> int:
    """Bytes a value occupies, counted the way the workspace browser of MATLAB counts them."""
    from .sparse import SparseMatrix
    if isinstance(v, np.ndarray):
        if v.dtype.kind == "U":
            return 2 * v.size
        return int(v.nbytes)
    if isinstance(v, SparseMatrix):
        m = v.m
        per = 1 if m.dtype == bool else (16 if m.dtype.kind == "c" else 8)
        return max(int(m.nnz), 1) * (per + 8) + (m.shape[1] + 1) * 8
    if isinstance(v, Cell):
        return sum(104 + value_bytes(x) for x in v.data.flat)
    if isinstance(v, Struct):
        return 64 * len(v.fields) + sum(104 + value_bytes(d[f]) for d in v.data.flat for f in v.fields)
    if isinstance(v, MString):
        return 150 + 2 * max(len(v) - 10, 0)
    if isinstance(v, StringArray):
        return 54 + sum(96 + 2 * len(x or "") for x in v.data.flat)
    if isinstance(v, FunctionHandle):
        return 32
    parts: list = []
    if hasattr(v, "columns"):
        parts = list(v.columns)
    elif isinstance(getattr(v, "props", None), dict):
        parts = list(v.props.values())
    elif isinstance(getattr(v, "store", None), dict):
        parts = list(v.store.values())
    total = sum(value_bytes(part) for part in parts)
    data = getattr(v, "data", None)
    if isinstance(data, np.ndarray) and data.dtype.kind != "O":
        total += int(data.nbytes)
    return total


def whos_struct(rows: list[dict]) -> Struct:
    fields = ["name", "size", "bytes", "class", "global", "sparse", "complex", "nesting", "persistent"]
    data = np.empty((len(rows), 1), dtype=object)
    for k, r in enumerate(rows):
        nesting = Struct.scalar({"function": char(r.get("function", "")), "level": np.array([[1.0]])})
        data[k, 0] = {"name": char(r["name"]), "size": np.array([[float(d) for d in r["size"]]]),
                      "bytes": np.array([[float(r["bytes"])]]), "class": char(r["class"]),
                      "global": np.array([[bool(r.get("global"))]]), "sparse": np.array([[bool(r.get("sparse"))]]),
                      "complex": np.array([[bool(r.get("complex"))]]), "nesting": nesting,
                      "persistent": np.array([[bool(r.get("persistent"))]])}
    return Struct(data, fields)


def whos_row(name: str, v: Any, is_global: bool = False, persistent: bool = False) -> dict:
    from .sparse import SparseMatrix
    return {"name": name, "size": tuple(shape_of(v)), "bytes": value_bytes(v), "class": class_name(v),
            "global": is_global, "sparse": isinstance(v, SparseMatrix), "persistent": persistent,
            "complex": bool(getattr(getattr(v, "dtype", None), "kind", "") == "c")}


def whos_text(rows: list[dict]) -> str:
    if not rows:
        return ""
    sizes = ["x".join(str(d) for d in r["size"]) for r in rows]
    attrs = [", ".join(a for a in ("global", "persistent", "sparse", "complex") if r.get(a)) for r in rows]
    w = max(4, max(len(r["name"]) for r in rows))
    ws = max(4, max(len(s) for s in sizes))
    wb = max(5, max(len(str(r["bytes"])) for r in rows))
    wc = max(5, max(len(r["class"]) for r in rows))
    lines = [f"  {'Name':<{w}}      {'Size':<{ws}}    {'Bytes':>{wb}}  {'Class':<{wc}}    Attributes", ""]
    for r, s, a in zip(rows, sizes, attrs):
        lines.append(f"  {r['name']:<{w}}      {s:<{ws}}    {r['bytes']:>{wb}}  {r['class']:<{wc}}    {a}".rstrip())
    return "\n".join(lines) + "\n\n"


def file_rows(interp, args: list, fn: str) -> list[dict]:
    """whos('-file', name, vars...) / who('-file', ...): one row per stored variable."""
    words = _words(args, fn)
    k = [w.lower() for w in words].index("-file")
    if k + 1 >= len(words):
        raise MatlabError(f"{fn}: a file name must follow -file.")
    rest = words[:k] + words[k + 2:]
    opts = _parse_options(["_"] + rest, "load", set())
    path = find_file(interp, words[k + 1], (".mat",))
    if path is None:
        raise MatlabError(f"Unable to find file or directory '{words[k + 1]}'.", "MATLAB:whos:fileNotFound")
    src = MatSource(interp, path)
    chosen, _ = select_names(src.names(), opts["names"], opts["regexp"])
    values, is_global = src.read(chosen)
    return [whos_row(n, values[n], n in is_global) for n in sorted(chosen) if n in values]


def listing_rows(interp, args: list, fn: str) -> list[dict]:
    """Rows for who/whos: the current workspace, the global workspace or a MAT-file."""
    words = _words(args, fn)
    if any(w.lower() == "-file" for w in words):
        return file_rows(interp, args, fn)
    frame = interp.frame
    if words and words[0] == "global":
        opts = _parse_options(["_"] + words[1:], "load", set())
        chosen, _ = select_names(sorted(interp.global_vars), opts["names"], opts["regexp"])
        return [whos_row(n, interp.global_vars[n], True) for n in chosen]
    opts = _parse_options(["_"] + words, "load", set())
    chosen, _ = select_names(workspace_names(frame), opts["names"], opts["regexp"])
    rows = []
    for n in chosen:
        store = frame.globals.get(n)
        row = whos_row(n, frame.get(n), store is interp.global_vars, store is not None and
                       store is not interp.global_vars)
        row["function"] = "" if frame is interp.base else frame.name
        rows.append(row)
    return rows


# -- matfile ---------------------------------------------------------------------------------------

def _new_matfile_version() -> str | None:
    """matfile creates a version 7.3 file, as MATLAB does; the default version if h5py is missing."""
    try:
        import h5py  # noqa: F401
    except ImportError:
        return None
    return "7.3"


class MatFileObject(MObject):
    """m = matfile(name): variables of a MAT-file read and written as properties of m."""

    is_handle = True
    _METHODS = {"who", "whos", "size", "isprop", "properties", "fieldnames"}

    def __init__(self, interp, path: str, writable: bool):
        self.path = path
        self.writable = writable
        self._stamp: tuple | None = None
        self._source: MatSource | None = None
        self._cache: dict = {}

    def class_name(self) -> str:
        return "matlab.io.MatFile"

    def isa_class(self, name: str) -> bool:
        return name in ("matlab.io.MatFile", "handle")

    def has_method(self, name: str) -> bool:
        return name in self._METHODS

    def _open(self, interp) -> MatSource | None:
        if not os.path.isfile(self.path):
            return None
        st = os.stat(self.path)
        stamp = (st.st_mtime_ns, st.st_size)
        if stamp != self._stamp:
            self._source = MatSource(interp, self.path)
            self._stamp = stamp
            self._cache = {}
        return self._source

    def names(self, interp) -> list[str]:
        src = self._open(interp)
        return src.names() if src else []

    def value(self, interp, name: str) -> Any:
        src = self._open(interp)
        if src is None or name not in src.names():
            raise MatlabError(f"The variable '{name}' does not exist in '{self.path}'.",
                              "MATLAB:MatFile:VariableNotInFile")
        if name not in self._cache:
            self._cache[name] = src.read([name])[0][name]
        return self._cache[name]

    def get_field(self, interp, name: str) -> Any:
        if name == "Properties":
            return Struct.scalar({"Source": char(self.path), "Writable": np.array([[self.writable]])})
        return self.value(interp, name)

    def _is_v73(self, interp) -> bool:
        src = self._open(interp)
        return src is not None and src.kind == "7.3"

    def _describe(self, interp, name: str) -> dict | None:
        """size, class and bytes of a variable of a version 7.3 file without reading its data."""
        if name in self._cache or not self._is_v73(interp) or name not in self.names(interp):
            return None
        from . import _mat73
        return _mat73.describe(self.path, name)

    def index_field(self, interp, name: str, arg_nodes: list, frame) -> Any:
        """m.name(i, j, ...): read only the selected part of a numeric, char or logical variable.

        Returns None, before evaluating any index, when the general path (reading the whole variable)
        applies: version 7 files, variables already read, and variables of other classes.
        """
        info = self._describe(interp, name)
        if info is None or info["sparse"] or info["class"] in ("cell", "struct") or 0 in info["size"]:
            return None
        from ..interpreter import index_value
        from . import _mat73
        shape_only = np.broadcast_to(np.zeros((), dtype=float), info["size"])  # for 'end'; holds no data
        args = interp.eval_index_args(arg_nodes, shape_only, frame)
        part = _mat73.read_part(self.path, name, args)
        if part is not None:
            return part
        return index_value(self.value(interp, name), args)

    def assign_field_index(self, interp, name: str, arg_nodes: list, value: Any, frame) -> bool:
        """m.name(i, j, ...) = value: write only that part of the file when the variable keeps its size
        and class. False, before evaluating any index, when the general path applies."""
        if not self.writable or not isinstance(value, np.ndarray) or value.dtype.kind not in "biufcU" \
                or (value.shape == (0, 0) and value.dtype == float):
            return False
        info = self._describe(interp, name)
        if info is None or info["sparse"] or info["class"] in ("cell", "struct") or 0 in info["size"]:
            return False
        from . import _mat73
        shape_only = np.broadcast_to(np.zeros((), dtype=float), info["size"])
        args = interp.eval_index_args(arg_nodes, shape_only, frame)
        try:
            done = _mat73.write_part(self.path, name, args, value)
        except OSError as e:
            raise MatlabError(f"Unable to write the file '{self.path}': {e}") from None
        self._stamp = None  # the next read sees the change
        if not done:  # the variable grows or changes class: rewrite it as a whole
            from .. import indexing as ix
            from ..interpreter import coerce_for_assignment
            cur, val = coerce_for_assignment(self.value(interp, name), value)
            fill = (lambda: "\0") if cur.dtype.kind == "U" else (lambda: False) if cur.dtype == bool else (lambda: 0)
            self.set_field(interp, name, ix.set_(cur, args, val, fill))
        return True

    def set_field(self, interp, name: str, value: Any) -> "MatFileObject":
        if name == "Properties":
            if not isinstance(value, Struct) or value.data.size != 1 or "Writable" not in value.fields:
                raise MatlabError("Only m.Properties.Writable can be changed.")
            self.writable = bool(np.asarray(value.data.flat[0]["Writable"]).any())
            return self
        if not self.writable:
            raise MatlabError(f"The file '{self.path}' was opened read-only. Use matfile(name, 'Writable', true) "
                              "or set m.Properties.Writable = true to change it.",
                              "MATLAB:MatFile:NotWritable")
        if not re.fullmatch(r"[A-Za-z]\w*", name):
            raise MatlabError(f"'{name}' is not a valid variable name.")
        version = None
        if not os.path.isfile(self.path) or os.path.getsize(self.path) == 0:
            version = _new_matfile_version()
        write_mat(interp, self.path, {name: value}, version=version, append=True)
        self._stamp = None
        return self

    def call_method(self, interp, name: str, args: list, nargout: int) -> list:
        rest = [a for a in args if a is not self]
        if name in ("who", "fieldnames", "properties"):
            names = sorted(self.names(interp))
            if name != "who":
                names = ["Properties"] + names
            if nargout or name != "who":
                return [Cell.from_list([char(n) for n in names], column=True)]
            interp.out("\nYour variables are:\n\n" + "  ".join(names) + "\n\n" if names else "")
            return []
        if name == "whos":
            wanted = [text(a) for a in rest]
            names, _ = select_names(sorted(self.names(interp)), wanted, [])
            rows = [self._whos_row(interp, n) for n in names]
            if nargout:
                return [whos_struct(rows)]
            interp.out(whos_text(rows))
            return []
        if name == "isprop":
            return [bool_scalar(bool(rest) and is_text(rest[0]) and
                                (text(rest[0]) == "Properties" or text(rest[0]) in self.names(interp)))]
        if name == "size":
            if not rest:
                dims: tuple = (1, 1)
            else:
                dims = self._size(interp, text(rest[0], "variable name"))
                if len(rest) > 1:
                    d = int(np.asarray(rest[1]).flat[0])
                    return [np.array([[float(dims[d - 1] if d <= len(dims) else 1)]])]
            if nargout <= 1:
                return [np.array([[float(d) for d in dims]])]
            vals = [float(d) for d in dims[:nargout - 1]] + [float(np.prod(dims[nargout - 1:]))]
            vals += [1.0] * (nargout - len(vals))
            return [np.array([[x]]) for x in vals]
        return super().call_method(interp, name, args, nargout)

    def _whos_row(self, interp, name: str) -> dict:
        info = self._describe(interp, name)
        if info is None:
            return whos_row(name, self.value(interp, name))
        return {"name": name, "size": tuple(info["size"]), "bytes": info["bytes"], "class": info["class"],
                "global": False, "sparse": info["sparse"], "persistent": False, "complex": info["complex"]}

    def _size(self, interp, name: str) -> tuple:
        info = self._describe(interp, name)
        return tuple(info["size"]) if info is not None else tuple(shape_of(self.value(interp, name)))

    def display_lines(self) -> list[str]:
        from ..interpreter import current_interpreter
        interp = current_interpreter()
        lines = ["  matlab.io.MatFile", "", "  Properties:", f"      Properties.Source: '{self.path}'",
                 f"    Properties.Writable: {'true' if self.writable else 'false'}"]
        if interp is not None:
            try:
                for n in sorted(self.names(interp)):
                    row = self._whos_row(interp, n)
                    size = "x".join(str(d) for d in row["size"])
                    lines.append(f"    {n:>19}: [{size} {row['class']}]")
            except MatlabError:
                pass
        return lines


@builtin("matfile")
def _matfile(interp, args, nargout):
    if not args or not is_text(args[0]):
        raise MatlabError("matfile needs a file name, for example m = matfile('data.mat').")
    name = os.path.expanduser(text(args[0]))
    rest = list(args[1:])
    writable = None
    if len(rest) % 2:
        raise MatlabError("matfile: options must be name-value pairs, for example 'Writable', true.")
    for k in range(0, len(rest), 2):
        key = text(rest[k], "option name").lower()
        if key != "writable":
            raise MatlabError(f"matfile: unknown option '{text(rest[k])}'. The only option is 'Writable'.")
        writable = bool(np.asarray(rest[k + 1]).any())
    if not os.path.splitext(name)[1]:
        name += ".mat"
    path = find_file(interp, name)
    if path is None:
        path = name
        if writable is None:
            writable = True  # a file that does not exist yet is created by the first assignment
    return MatFileObject(interp, os.path.abspath(path), bool(writable))


# -- import options and datastores -------------------------------------------------------------------

@builtin("detectImportOptions")
def _detect_import_options(interp, args, nargout):
    if not args or not is_text(args[0]):
        raise MatlabError("detectImportOptions needs a file name, for example detectImportOptions('data.csv').")
    from . import _textimport
    name = text(args[0])
    if os.path.splitext(name)[1].lower() in (".xls", ".xlsx", ".xlsm", ".xlsb", ".ods"):
        raise MatlabError("detectImportOptions: import options for spreadsheet files are not supported; call "
                          "readtable, readmatrix or readcell on the workbook with their Sheet and Range options.")
    nv = _textimport.name_values(list(args[1:]), _textimport.KNOWN, "detectImportOptions")
    return _textimport.detect(interp, name, nv)


@builtin("delimitedTextImportOptions")
def _delimited_text_import_options(interp, args, nargout):
    from . import _textimport
    nv = _textimport.name_values(list(args), _textimport._known(
        "NumVariables", "Delimiter", "VariableNames", "VariableTypes", "DataLines", "VariableNamesLine",
        "RowNamesColumn", "Encoding", "MissingRule", "ImportErrorRule", "EmptyLineRule",
        "ConsecutiveDelimitersRule", "LeadingDelimitersRule", "TrailingDelimitersRule", "ExtraColumnsRule",
        "VariableNamingRule", "PreserveVariableNames", "CommentStyle", "LineEnding", "Whitespace",
        "SelectedVariableNames"), "delimitedTextImportOptions")
    o = _textimport.ImportOptions()
    count = 0
    if "NumVariables" in nv:
        count = int(np.asarray(nv.pop("NumVariables")).flat[0])
    elif "VariableNames" in nv:
        count = len(_textimport.text_items(nv["VariableNames"], "VariableNames"))
    elif "VariableTypes" in nv:
        count = len(_textimport.text_items(nv["VariableTypes"], "VariableTypes"))
    o.names = [f"Var{k + 1}" for k in range(count)]
    o.types = ["char"] * count
    o.var_opts = [{"Name": n, "Type": "char"} for n in o.names]
    for key in ("Delimiter", "VariableNames", "VariableTypes"):
        if key in nv:
            o = o.set_field(interp, key, nv.pop(key))
    for key, v in nv.items():
        o = o.set_field(interp, key, v)
    return o


def _datastore_args(interp, args: list, fn: str):
    from . import _textimport
    if not args:
        raise MatlabError(f"{fn} needs a file, folder or pattern, for example {fn}('data/*.csv').")
    extra = _textimport._known("FileExtensions", "IncludeSubfolders", "ReadSize", "SelectedVariableNames",
                               "SelectedFormats", "Type", "AlternateFileSystemRoots",
                               unsupported=("TextscanFormats",))
    known = dict(_textimport.KNOWN)
    known.update(extra)
    nv = _textimport.name_values(list(args[1:]), known, fn)
    kind = text(nv.pop("Type")).lower() if "Type" in nv else "tabulartext"
    if kind != "tabulartext":
        raise MatlabError(f"{fn}: only datastores of type 'tabulartext' are supported, not '{kind}'.")
    nv.pop("AlternateFileSystemRoots", None)
    exts = None
    if "FileExtensions" in nv:
        exts = tuple(e.lower() for e in _textimport.text_items(nv.pop("FileExtensions"), "FileExtensions"))
    sub = "IncludeSubfolders" in nv and bool(np.asarray(nv.pop("IncludeSubfolders")).any())
    files = _textimport.datastore_files(interp, args[0], exts, sub)
    if exts is None and len(files) > 1:
        texty = [f for f in files if os.path.splitext(f)[1].lower() in (".csv", ".txt", ".dat", ".tsv", ".asc")]
        files = texty or files
    read_size = nv.pop("ReadSize", None)
    selected = nv.pop("SelectedVariableNames", None)
    formats = nv.pop("SelectedFormats", None)
    ds = _textimport.TabularTextDatastore(interp, files, nv)
    if selected is not None:
        ds.set_field(interp, "SelectedVariableNames", selected)
    if formats is not None:
        ds.set_field(interp, "SelectedFormats", formats)
    if read_size is not None:
        ds.set_field(interp, "ReadSize", read_size)
    return ds


@builtin("tabularTextDatastore")
def _tabular_text_datastore(interp, args, nargout):
    return _datastore_args(interp, list(args), "tabularTextDatastore")


@builtin("datastore")
def _datastore(interp, args, nargout):
    return _datastore_args(interp, list(args), "datastore")


# -- memory-mapped files ----------------------------------------------------------------------------

_MAP_TYPES = {"int8": "i1", "uint8": "u1", "int16": "i2", "uint16": "u2", "int32": "i4", "uint32": "u4",
              "int64": "i8", "uint64": "u8", "single": "f4", "double": "f8"}


class MemoryMap(MObject):
    """memmapfile: the bytes of a file seen as an array (or as records with named fields)."""

    is_handle = True

    def __init__(self, path: str):
        self.path = path
        self.writable = False
        self.offset = 0
        self.repeat: float = np.inf
        self.format: Any = "uint8"  # a type name, or a list of (type, dims, field name)

    def class_name(self) -> str:
        return "memmapfile"

    def isa_class(self, name: str) -> bool:
        return name == "memmapfile"

    def set_format(self, v: Any) -> None:
        if is_text(v):
            name = text(v)
            if name not in _MAP_TYPES:
                raise MatlabError(f"memmapfile: unknown Format '{name}'. Types are: {', '.join(_MAP_TYPES)}.")
            self.format = name
            return
        if not isinstance(v, Cell) or v.data.ndim != 2 or v.data.shape[1] not in (2, 3) or v.data.shape[0] < 1:
            raise MatlabError("memmapfile: Format must be a type name or a cell array with one row "
                              "{type, size, fieldname} per field.")
        rows = []
        for r in range(v.data.shape[0]):
            name = text(v.data[r, 0], "the type")
            if name not in _MAP_TYPES:
                raise MatlabError(f"memmapfile: unknown type '{name}' in Format.")
            dims = tuple(int(x) for x in np.asarray(v.data[r, 1]).flatten())
            if len(dims) < 2 or any(d < 1 for d in dims):
                raise MatlabError("memmapfile: each size in Format must have at least two positive dimensions.")
            field = text(v.data[r, 2], "the field name") if v.data.shape[1] == 3 else "x"
            if not re.fullmatch(r"[A-Za-z]\w*", field):
                raise MatlabError(f"memmapfile: '{field}' is not a valid field name.")
            rows.append((name, dims, field))
        if len({r[2] for r in rows}) != len(rows):
            raise MatlabError("memmapfile: the field names in Format must be different.")
        if v.data.shape[1] == 2 and len(rows) > 1:
            raise MatlabError("memmapfile: a Format with several rows needs a field name in each row.")
        self.format = rows

    def _dtype(self) -> np.dtype:
        if isinstance(self.format, str):
            return np.dtype(_MAP_TYPES[self.format])
        # each field is stored column by column: reversed dimensions in C order
        return np.dtype([(f, _MAP_TYPES[t], dims[::-1]) for t, dims, f in self.format])

    def _map(self, mode: str):
        if not os.path.isfile(self.path):
            raise MatlabError(f"memmapfile: the file '{self.path}' does not exist.",
                              "MATLAB:memmapfile:inaccessibleFile")
        dt = self._dtype()
        size = os.path.getsize(self.path) - self.offset
        fit = size // dt.itemsize if size > 0 else 0
        count = fit if np.isinf(self.repeat) else int(self.repeat)
        if count < 1 or count > fit:
            raise MatlabError(f"memmapfile: the file is too small to map {'one' if count < 2 else count} "
                              f"element{'s' if count > 1 else ''} of {dt.itemsize} bytes at offset {self.offset} "
                              f"({max(size, 0)} bytes are available).", "MATLAB:memmapfile:mapfile:fileTooSmall")
        try:
            return np.memmap(self.path, dtype=dt, mode=mode, offset=self.offset, shape=(count,))
        except (OSError, ValueError) as e:
            raise MatlabError(f"memmapfile: cannot map '{self.path}': {e}") from None

    def data(self) -> Any:
        mm = self._map("r")
        try:
            if isinstance(self.format, str):
                return np.array(mm).reshape(-1, 1)
            out = np.empty((mm.shape[0], 1), dtype=object)
            for k in range(mm.shape[0]):
                rec = {}
                for _t, _dims, f in self.format:
                    rec[f] = np.ascontiguousarray(np.array(mm[f][k]).transpose())
                out[k, 0] = rec
            return Struct(out, [f for _t, _d, f in self.format])
        finally:
            del mm

    def store(self, value: Any) -> None:
        if not self.writable:
            raise MatlabError("memmapfile: the map is read-only. Create it with 'Writable', true (or set "
                              "m.Writable = true) to change the file.",
                              "MATLAB:memmapfile:illegalSubscriptedAssignment")
        mm = self._map("r+")
        try:
            if isinstance(self.format, str):
                if not isinstance(value, np.ndarray) or value.size != mm.shape[0]:
                    raise MatlabError(f"memmapfile: Data must keep its {mm.shape[0]} elements.")
                if class_name(value) != self.format:
                    raise MatlabError(f"memmapfile: Data must be of class {self.format}, not {class_name(value)}.")
                mm[:] = value.reshape(-1, order="F")
            else:
                if not isinstance(value, Struct) or value.data.size != mm.shape[0]:
                    raise MatlabError(f"memmapfile: Data must stay a {mm.shape[0]}-by-1 structure array.")
                for k, rec in enumerate(value.data.flatten(order="F")):
                    for t, dims, f in self.format:
                        v = rec.get(f)
                        if not isinstance(v, np.ndarray) or tuple(v.shape) != dims or class_name(v) != t:
                            raise MatlabError(f"memmapfile: field '{f}' must be a {'x'.join(map(str, dims))} "
                                              f"{t} array.")
                        mm[f][k] = v.transpose()
            mm.flush()
        finally:
            del mm

    def get_field(self, interp, name: str) -> Any:
        if name == "Data":
            return self.data()
        if name == "Filename":
            return char(self.path)
        if name == "Writable":
            return bool_scalar(self.writable)
        if name == "Offset":
            return np.array([[float(self.offset)]])
        if name == "Repeat":
            return np.array([[float(self.repeat)]])
        if name == "Format":
            if isinstance(self.format, str):
                return char(self.format)
            data = np.empty((len(self.format), 3), dtype=object)
            for r, (t, dims, f) in enumerate(self.format):
                data[r, 0], data[r, 1], data[r, 2] = char(t), np.array([[float(d) for d in dims]]), char(f)
            return Cell(data)
        raise MatlabError(f"Unrecognized property '{name}' for class 'memmapfile'.")

    def set_field(self, interp, name: str, value: Any) -> "MemoryMap":
        if name == "Data":
            self.store(value)
        elif name == "Writable":
            self.writable = bool(np.asarray(value).any())
        elif name == "Offset":
            off = float(np.asarray(value, dtype=float).flat[0])
            if off < 0 or off != int(off):
                raise MatlabError("memmapfile: Offset must be a non-negative whole number of bytes.")
            self.offset = int(off)
        elif name == "Repeat":
            r = float(np.asarray(value, dtype=float).flat[0])
            if not (np.isinf(r) or (r >= 1 and r == int(r))):
                raise MatlabError("memmapfile: Repeat must be a positive integer or Inf.")
            self.repeat = r
        elif name == "Format":
            self.set_format(value)
        elif name == "Filename":
            self.path = os.path.abspath(os.path.expanduser(text(value, "Filename")))
        else:
            raise MatlabError(f"Unrecognized property '{name}' for class 'memmapfile'.")
        return self

    def display_lines(self) -> list[str]:
        fmt = f"'{self.format}'" if isinstance(self.format, str) else \
            "{" + "; ".join(f"'{t}' [{' '.join(map(str, d))}] '{f}'" for t, d, f in self.format) + "}"
        try:
            n = self._map("r").shape[0]
            shown = f"{n}x1 {self.format} array" if isinstance(self.format, str) else f"{n}x1 struct array"
        except MatlabError:
            shown = "(the file cannot be mapped)"
        return [f"    Filename: '{self.path}'", f"    Writable: {'true' if self.writable else 'false'}",
                f"      Offset: {self.offset}", f"      Format: {fmt}",
                f"      Repeat: {'Inf' if np.isinf(self.repeat) else int(self.repeat)}", f"        Data: {shown}"]


@builtin("memmapfile")
def _memmapfile(interp, args, nargout):
    if not args or not is_text(args[0]):
        raise MatlabError("memmapfile needs a file name, for example m = memmapfile('data.bin', 'Format', 'double').")
    path = find_file(interp, text(args[0]))
    if path is None:
        raise MatlabError(f"memmapfile: the file '{text(args[0])}' does not exist.",
                          "MATLAB:memmapfile:inaccessibleFile")
    m = MemoryMap(os.path.abspath(path))
    rest = list(args[1:])
    if len(rest) % 2:
        raise MatlabError("memmapfile: options must be name-value pairs.")
    for k in range(0, len(rest), 2):
        key = text(rest[k], "option name").lower()
        name = {"format": "Format", "offset": "Offset", "repeat": "Repeat", "writable": "Writable"}.get(key)
        if name is None:
            raise MatlabError(f"memmapfile: unknown option '{text(rest[k])}'. Options are Format, Offset, "
                              "Repeat and Writable.")
        m.set_field(interp, name, rest[k + 1])
    m._map("r")  # report a file that is too small now, as MATLAB does
    return m


# -- XML documents (xmlread, xmlwrite) ---------------------------------------------------------------

_NODE_CONSTANTS = {"ELEMENT_NODE": 1, "ATTRIBUTE_NODE": 2, "TEXT_NODE": 3, "CDATA_SECTION_NODE": 4,
                   "ENTITY_REFERENCE_NODE": 5, "ENTITY_NODE": 6, "PROCESSING_INSTRUCTION_NODE": 7,
                   "COMMENT_NODE": 8, "DOCUMENT_NODE": 9, "DOCUMENT_TYPE_NODE": 10,
                   "DOCUMENT_FRAGMENT_NODE": 11, "NOTATION_NODE": 12}


def _xml_text(node) -> str:
    """The text of a node and everything below it (what getTextContent returns)."""
    if node.nodeType in (3, 4):
        return node.data
    if node.nodeType in (7, 8):
        return node.data if node.nodeType == 8 else ""
    if node.nodeType == 2:
        return node.value
    return "".join(_xml_text(c) for c in node.childNodes if c.nodeType not in (7, 8))


def _xml_wrap(v: Any) -> Any:
    """A MATLAB value for what a DOM call returned."""
    if v is None:
        return np.zeros((0, 0))
    if isinstance(v, bool):
        return bool_scalar(v)
    if isinstance(v, (int, float)):
        return np.array([[float(v)]])
    if isinstance(v, str):
        return char(v)
    if isinstance(v, XmlList):
        return v
    return XmlNode(v)


class XmlList(MObject):
    """A list of nodes (getElementsByTagName, getChildNodes) or of attributes (getAttributes)."""

    is_handle = True

    def __init__(self, items, named: bool = False):
        self.items = items  # a live DOM list, or a function that returns the current nodes
        self.named = named

    def class_name(self) -> str:
        return "org.w3c.dom.NamedNodeMap" if self.named else "org.w3c.dom.NodeList"

    def nodes(self) -> list:
        items = self.items() if callable(self.items) else self.items
        if self.named:
            return [items.item(k) for k in range(items.length)]
        return list(items)

    def has_method(self, name: str) -> bool:
        return name in ("getLength", "item", "getNamedItem")

    def get_field(self, interp, name: str) -> Any:
        if name == "getLength":
            return np.array([[float(len(self.nodes()))]])
        raise MatlabError(f"Unrecognized method '{name}' for a {self.class_name()}.")

    def call_method(self, interp, name: str, args: list, nargout: int) -> list:
        rest = [a for a in args if a is not self]
        nodes = self.nodes()
        if name == "getLength":
            return [np.array([[float(len(nodes))]])]
        if name == "item":
            if not rest:
                raise MatlabError("item needs an index; the first item is item(0).")
            k = float(np.asarray(rest[0], dtype=float).flat[0])
            if k != int(k) or k < 0 or k >= len(nodes):
                return [np.zeros((0, 0))]  # out of range gives no node, as the DOM does
            return [XmlNode(nodes[int(k)])]
        if name == "getNamedItem" and self.named:
            want = text(rest[0], "the attribute name") if rest else ""
            hit = next((n for n in nodes if n.name == want), None)
            return [_xml_wrap(hit)]
        return super().call_method(interp, name, args, nargout)

    def matlab_shape(self) -> tuple:
        return (len(self.nodes()), 1)

    def display_lines(self) -> list[str]:
        kind = self.class_name().split(".")[-1]
        return [f"  {kind} with {len(self.nodes())} item(s); use item(k), counting from 0."]


class XmlNode(MObject):
    """A node of an XML document, with the method names of the W3C DOM (as MATLAB's xmlread returns)."""

    is_handle = True
    _METHODS = {
        "getNodeName", "getNodeValue", "setNodeValue", "getNodeType", "getParentNode", "getChildNodes",
        "getFirstChild", "getLastChild", "getNextSibling", "getPreviousSibling", "getAttributes",
        "getOwnerDocument", "hasChildNodes", "hasAttributes", "appendChild", "removeChild", "insertBefore",
        "replaceChild", "cloneNode", "getTextContent", "setTextContent", "normalize", "getLocalName",
        "getNamespaceURI", "getPrefix", "isEqualNode", "isSameNode",
        # elements
        "getTagName", "getAttribute", "setAttribute", "removeAttribute", "hasAttribute", "getAttributeNode",
        "getElementsByTagName",
        # documents
        "getDocumentElement", "createElement", "createTextNode", "createComment", "createCDATASection",
        "createProcessingInstruction", "createAttribute", "createDocumentFragment", "getElementById",
        "importNode", "getXmlVersion", "getXmlEncoding", "getDoctype", "toString",
        # text and attributes
        "getData", "setData", "getLength", "getName", "getValue", "setValue", "getTarget", "getOwnerElement",
    }

    def __init__(self, node):
        self.node = node

    def class_name(self) -> str:
        return {1: "org.w3c.dom.Element", 2: "org.w3c.dom.Attr", 3: "org.w3c.dom.Text",
                4: "org.w3c.dom.CDATASection", 7: "org.w3c.dom.ProcessingInstruction", 8: "org.w3c.dom.Comment",
                9: "org.w3c.dom.Document", 10: "org.w3c.dom.DocumentType",
                11: "org.w3c.dom.DocumentFragment"}.get(self.node.nodeType, "org.w3c.dom.Node")

    def isa_class(self, name: str) -> bool:
        return name in (self.class_name(), "org.w3c.dom.Node")

    def has_method(self, name: str) -> bool:
        return name in self._METHODS

    def object_equals(self, other: Any) -> bool:
        return isinstance(other, XmlNode) and other.node is self.node

    def get_field(self, interp, name: str) -> Any:
        if name in _NODE_CONSTANTS:
            return np.array([[float(_NODE_CONSTANTS[name])]])
        if name in self._METHODS:
            res = self.call_method(interp, name, [self], 1)
            if not res:
                raise MatlabError(f"'{name}' does not return a value.")
            return res[0]
        raise MatlabError(f"Unrecognized method or property '{name}' for a {self.class_name()} node.")

    @staticmethod
    def _other(v: Any, what: str):
        if not isinstance(v, XmlNode):
            raise MatlabError(f"{what} must be an XML node.")
        return v.node

    def call_method(self, interp, name: str, args: list, nargout: int) -> list:
        rest = list(args)
        for k, a in enumerate(rest):
            if a is self:
                del rest[k]
                break
        try:
            out = self._call(name, rest)
        except MatlabError:
            raise
        except Exception as e:  # DOM errors (wrong document, hierarchy problems) become MATLAB errors
            raise MatlabError(f"XML error in {name}: {e}") from None
        if out is _NO_VALUE:
            return []
        return [_xml_wrap(out)]

    def _call(self, name: str, a: list) -> Any:
        n = self.node
        kind = n.nodeType
        s = lambda k, what: text(a[k], what)  # noqa: E731
        if len(a) < _XML_ARGS.get(name, 0):
            raise MatlabError(f"{name} needs {_XML_ARGS[name]} argument(s).")
        if name == "getNodeName":
            return n.nodeName
        if name in ("getNodeValue", "getData", "getValue"):
            if name != "getNodeValue" and kind not in (2, 3, 4, 7, 8):
                raise MatlabError(f"{name} applies to text, comment and attribute nodes.")
            return n.nodeValue
        if name in ("setNodeValue", "setData", "setValue"):
            value = s(0, "the value")
            if kind in (3, 4, 8):
                n.data = value
            elif kind == 2:
                n.value = value
            else:
                raise MatlabError(f"{name} applies to text, comment and attribute nodes.")
            return _NO_VALUE
        if name == "getNodeType":
            return kind
        if name == "getParentNode":
            return n.parentNode
        if name == "getChildNodes":
            return XmlList(n.childNodes)
        if name == "getFirstChild":
            return n.firstChild
        if name == "getLastChild":
            return n.lastChild
        if name == "getNextSibling":
            return n.nextSibling
        if name == "getPreviousSibling":
            return n.previousSibling
        if name == "getAttributes":
            return XmlList((lambda: n.attributes), named=True) if kind == 1 else None
        if name == "getOwnerDocument":
            return n.ownerDocument
        if name == "hasChildNodes":
            return bool(n.childNodes)
        if name == "hasAttributes":
            return kind == 1 and n.attributes.length > 0
        if name == "appendChild":
            return n.appendChild(self._other(a[0], "The child"))
        if name == "removeChild":
            return n.removeChild(self._other(a[0], "The child"))
        if name == "insertBefore":
            ref = None if isinstance(a[1], np.ndarray) and a[1].size == 0 else self._other(a[1], "The reference")
            return n.insertBefore(self._other(a[0], "The new node"), ref)
        if name == "replaceChild":
            return n.replaceChild(self._other(a[0], "The new node"), self._other(a[1], "The old node"))
        if name == "cloneNode":
            return n.cloneNode(bool(np.asarray(a[0]).any()) if a else False)
        if name == "getTextContent":
            return None if kind in (9, 10) else _xml_text(n)
        if name == "setTextContent":
            value = s(0, "the text")
            if kind in (3, 4, 8):
                n.data = value
            elif kind == 2:
                n.value = value
            elif kind in (1, 11):
                for c in list(n.childNodes):
                    n.removeChild(c)
                if value:
                    n.appendChild(n.ownerDocument.createTextNode(value))
            else:
                raise MatlabError("setTextContent applies to element, text, comment and attribute nodes.")
            return _NO_VALUE
        if name == "normalize":
            n.normalize()
            return _NO_VALUE
        if name == "getLocalName":
            return n.localName
        if name == "getNamespaceURI":
            return n.namespaceURI
        if name == "getPrefix":
            return n.prefix
        if name == "isSameNode":
            return isinstance(a[0], XmlNode) and a[0].node is n
        if name == "isEqualNode":
            return isinstance(a[0], XmlNode) and a[0].node.toxml() == n.toxml()
        if name == "toString":
            return n.toxml() if kind != 2 else f'{n.name}="{n.value}"'
        if name == "getLength":
            if kind in (3, 4, 8):
                return len(n.data)
            raise MatlabError("getLength applies to text nodes and node lists.")
        if name == "getName":
            if kind in (2, 10):
                return n.name
            raise MatlabError("getName applies to attribute nodes.")
        if name == "getTarget":
            return n.target if kind == 7 else None
        if name == "getOwnerElement":
            return n.ownerElement if kind == 2 else None
        if kind == 1 or name in ("getElementsByTagName",):
            if name == "getTagName" and kind == 1:
                return n.tagName
            if name == "getAttribute" and kind == 1:
                return n.getAttribute(s(0, "the attribute name"))
            if name == "setAttribute" and kind == 1:
                value = a[1]
                if isinstance(value, np.ndarray) and value.dtype.kind in "biuf" and value.size == 1:
                    from ..display import format_scalar_plain
                    value_text = format_scalar_plain(value.flat[0])
                else:
                    value_text = text(value, "the attribute value")
                n.setAttribute(s(0, "the attribute name"), value_text)
                return _NO_VALUE
            if name == "removeAttribute" and kind == 1:
                if n.hasAttribute(s(0, "the attribute name")):
                    n.removeAttribute(s(0, "the attribute name"))
                return _NO_VALUE
            if name == "hasAttribute" and kind == 1:
                return n.hasAttribute(s(0, "the attribute name"))
            if name == "getAttributeNode" and kind == 1:
                return n.getAttributeNode(s(0, "the attribute name"))
            if name == "getElementsByTagName" and kind in (1, 9):
                tag = s(0, "the tag name")
                return XmlList(lambda: n.getElementsByTagName(tag))
        if kind == 9:
            if name == "getDocumentElement":
                return n.documentElement
            if name == "createElement":
                return n.createElement(s(0, "the tag name"))
            if name == "createTextNode":
                return n.createTextNode(s(0, "the text"))
            if name == "createComment":
                return n.createComment(s(0, "the comment"))
            if name == "createCDATASection":
                return n.createCDATASection(s(0, "the text"))
            if name == "createProcessingInstruction":
                return n.createProcessingInstruction(s(0, "the target"), s(1, "the data"))
            if name == "createAttribute":
                return n.createAttribute(s(0, "the attribute name"))
            if name == "createDocumentFragment":
                return n.createDocumentFragment()
            if name == "importNode":
                deep = bool(np.asarray(a[1]).any()) if len(a) > 1 else True
                return n.importNode(self._other(a[0], "The node"), deep)
            if name == "getElementById":
                want = s(0, "the id")
                for e in n.getElementsByTagName("*"):
                    if e.getAttribute("id") == want:
                        return e
                return None
            if name == "getXmlVersion":
                return n.version or "1.0"
            if name == "getXmlEncoding":
                return n.encoding
            if name == "getDoctype":
                return n.doctype
        raise MatlabError(f"The method '{name}' does not apply to a {self.class_name().split('.')[-1]} node.")

    def display_lines(self) -> list[str]:
        n = self.node
        if n.nodeType == 9:
            root = n.documentElement.tagName if n.documentElement is not None else "(empty)"
            return [f"  XML document, root element <{root}>", "  [#document: null]"]
        shown = n.nodeValue if n.nodeType != 1 else None
        return [f"  [{n.nodeName}: {'null' if shown is None else shown}]"]


_NO_VALUE = object()
_XML_ARGS = {"setNodeValue": 1, "setData": 1, "setValue": 1, "appendChild": 1, "removeChild": 1, "insertBefore": 2,
             "replaceChild": 2, "setTextContent": 1, "isSameNode": 1, "isEqualNode": 1, "getAttribute": 1,
             "setAttribute": 2, "removeAttribute": 1, "hasAttribute": 1, "getAttributeNode": 1,
             "getElementsByTagName": 1, "createElement": 1, "createTextNode": 1, "createComment": 1,
             "createCDATASection": 1, "createProcessingInstruction": 2, "createAttribute": 1, "importNode": 1,
             "getElementById": 1}


def _parse_xml(source: str, from_text: bool = False):
    from xml.dom import minidom
    from xml.parsers.expat import ExpatError
    try:
        return minidom.parseString(source) if from_text else minidom.parse(source)
    except ExpatError as e:
        where = "the text" if from_text else f"'{source}'"
        raise MatlabError(f"Unable to read {where} as XML: {e}.", "MATLAB:xml:parseError") from None
    except OSError as e:
        raise MatlabError(f"Unable to read the file '{source}': {e.strerror or e}.") from None


@builtin("xmlread")
def _xmlread(interp, args, nargout):
    if not args or not is_text(args[0]):
        raise MatlabError("xmlread needs a file name, for example doc = xmlread('data.xml').")
    for a in args[1:]:
        if not (is_text(a) and text(a).lower() == "allowdoctype") and not isinstance(a, np.ndarray):
            raise MatlabError("xmlread: the only option is 'AllowDoctype', true or false.")
    name = text(args[0])
    path = find_file(interp, name)
    if path is None:
        raise MatlabError(f"Unable to find the file '{name}'.", "MATLAB:xmlread:FileNotFound")
    return XmlNode(_parse_xml(path))


def _strip_layout(node) -> None:
    """Remove white space between elements, so the pretty printer can lay the document out afresh."""
    for c in list(node.childNodes):
        if c.nodeType == 3 and not c.data.strip() and any(x.nodeType == 1 for x in node.childNodes):
            node.removeChild(c)
        elif c.nodeType == 1:
            _strip_layout(c)


def xml_text(node) -> str:
    from xml.dom import minidom
    if node.nodeType == 9:
        doc = node.cloneNode(True)
    else:
        doc = minidom.Document()
        doc.appendChild(doc.importNode(node, True))
    _strip_layout(doc)
    out = doc.toprettyxml(indent="   ", encoding="utf-8").decode("utf-8")
    return "\n".join(ln for ln in out.split("\n") if ln.strip()) + "\n"


@builtin("xmlwrite")
def _xmlwrite(interp, args, nargout):
    if len(args) == 1 and isinstance(args[0], XmlNode):
        return char(xml_text(args[0].node))
    if len(args) != 2 or not is_text(args[0]) or not isinstance(args[1], XmlNode):
        raise MatlabError("xmlwrite(filename, node) writes a file; text = xmlwrite(node) returns the XML text.")
    try:
        with open(os.path.expanduser(text(args[0])), "w", encoding="utf-8", newline="\n") as fh:
            fh.write(xml_text(args[1].node))
    except OSError as e:
        raise MatlabError(f"xmlwrite: unable to write '{text(args[0])}': {e.strerror or e}.") from None
    return None


# -- structs in XML and JSON files (readstruct, writestruct) -------------------------------------------

def _file_type(name: str, opts: dict, fn: str) -> str:
    kind = text(opts["FileType"]).lower() if "FileType" in opts else None
    if kind is None:
        ext = os.path.splitext(name)[1].lower()
        kind = {".xml": "xml", ".json": "json"}.get(ext)
        if kind is None:
            raise MatlabError(f"{fn}: cannot tell the file type from the extension '{ext}'. Use a .xml or .json "
                              "file name or the option 'FileType', 'xml' or 'json'.")
    if kind not in ("xml", "json"):
        raise MatlabError(f"{fn}: FileType must be 'xml' or 'json'.")
    return kind


def _struct_options(args: list, names: tuple, fn: str) -> dict:
    if len(args) % 2:
        raise MatlabError(f"{fn}: options must be name-value pairs.")
    out = {}
    for k in range(0, len(args), 2):
        key = text(args[k], "option name")
        match = [n for n in names if n.lower() == key.lower()]
        if not match:
            raise MatlabError(f"{fn}: unknown or unsupported option '{key}'. Options are: {', '.join(names)}.")
        out[match[0]] = args[k + 1]
    return out


def _scalar_text(v: Any, what: str) -> str | None:
    """Text of a scalar value for an XML node, None when the value is not a scalar."""
    from ..display import format_scalar_plain
    if isinstance(v, MString):
        return str(v)
    if isinstance(v, np.ndarray):
        if v.dtype.kind == "U":
            if v.ndim == 2 and v.shape[0] <= 1:
                from ..values import to_str
                return to_str(v)
            return None
        if v.size == 1:
            x = v.flat[0]
            if v.dtype == bool:
                return "true" if x else "false"
            if v.dtype.kind == "f" and not np.isfinite(x):
                return "NaN" if np.isnan(x) else ("Inf" if x > 0 else "-Inf")
            if v.dtype.kind == "c":
                raise MatlabError(f"writestruct: complex numbers cannot be written ({what}).")
            return format_scalar_plain(x) if v.dtype.kind != "f" else repr(float(x)).removesuffix(".0")
        return None
    if hasattr(v, "element_texts") and getattr(v, "size", 0) == 1:
        return str(v.element_texts().flat[0])
    return None


def _elements(v: Any) -> list:
    """The elements of an array value, each as a scalar of the same kind."""
    if isinstance(v, StringArray):
        return [MString(x) if x is not None else MString("") for x in v.data.flatten(order="F")]
    if isinstance(v, Cell):
        return list(v.data.flatten(order="F"))
    if isinstance(v, Struct):
        out = []
        for d in v.data.flatten(order="F"):
            data = np.empty((1, 1), dtype=object)
            data[0, 0] = d
            out.append(Struct(data, list(v.fields)))
        return out
    if isinstance(v, np.ndarray):
        return [v.reshape(-1, order="F")[k:k + 1].reshape(1, 1) for k in range(v.size)]
    if hasattr(v, "like") and hasattr(v, "data"):
        flat = np.asarray(v.data).reshape(-1, order="F")
        return [v.like(flat[k:k + 1].reshape(1, 1)) for k in range(flat.size)]
    return [v]


def _struct_to_xml(doc, parent, s: Struct, suffix: str, where: str) -> None:
    d = s.data.flat[0]
    for f in s.fields:
        v = d[f]
        if suffix and f.endswith(suffix) and len(f) > len(suffix):
            t = _scalar_text(v, f"{where}.{f}")
            if t is None:
                raise MatlabError(f"writestruct: the attribute field '{where}.{f}' must hold one number, logical "
                                  "or piece of text.")
            parent.setAttribute(f[:-len(suffix)], t)
            continue
        only_attributes = all(x == "Text" or (suffix and x.endswith(suffix)) for x in s.fields)
        if f == "Text" and only_attributes and _scalar_text(v, f) is not None:
            parent.appendChild(doc.createTextNode(_scalar_text(v, f)))  # the node's own text
            continue
        items = [v] if _scalar_text(v, f) is not None or (isinstance(v, Struct) and v.data.size == 1) \
            else _elements(v)
        for item in items:
            node = doc.createElement(f)
            parent.appendChild(node)
            if isinstance(item, Struct):
                _struct_to_xml(doc, node, item, suffix, f"{where}.{f}")
                continue
            t = _scalar_text(item, f"{where}.{f}")
            if t is None:
                raise MatlabError(f"writestruct: the field '{where}.{f}' holds a value of class "
                                  f"'{class_name(item)}' that cannot be written to XML.")
            if t != "":
                node.appendChild(doc.createTextNode(t))


@builtin("writestruct")
def _writestruct(interp, args, nargout):
    if len(args) < 2 or not isinstance(args[0], Struct) or not is_text(args[1]):
        raise MatlabError("writestruct(S, filename) needs a structure and a file name.")
    s, name = args[0], os.path.expanduser(text(args[1]))
    opts = _struct_options(list(args[2:]), ("FileType", "StructNodeName", "AttributeSuffix", "PrettyPrint"),
                           "writestruct")
    kind = _file_type(name, opts, "writestruct")
    pretty = bool(np.asarray(opts["PrettyPrint"]).any()) if "PrettyPrint" in opts else True
    if kind == "json":
        from ._registry import BUILTINS
        extra = [char("PrettyPrint"), bool_scalar(pretty)]
        body = text(BUILTINS["jsonencode"](interp, [s, *extra], 1))
    else:
        if s.data.size != 1:
            raise MatlabError("writestruct: the structure written to an XML file must be a scalar structure.")
        from xml.dom import minidom
        suffix = text(opts["AttributeSuffix"]) if "AttributeSuffix" in opts else "Attribute"
        root_name = text(opts["StructNodeName"]) if "StructNodeName" in opts else "struct"
        doc = minidom.Document()
        try:
            root = doc.createElement(root_name)
            doc.appendChild(root)
            _struct_to_xml(doc, root, s, suffix, root_name)
        except MatlabError:
            raise
        except Exception as e:  # invalid tag or attribute names
            raise MatlabError(f"writestruct: cannot build the XML document: {e}") from None
        body = xml_text(doc) if pretty else doc.toxml(encoding="utf-8").decode("utf-8")
    try:
        with open(name, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(body if body.endswith("\n") else body + "\n")
    except OSError as e:
        raise MatlabError(f"writestruct: unable to write '{name}': {e.strerror or e}.") from None
    return None


def _leaf_value(t: str) -> Any:
    """A number or logical when the text is one, otherwise a string."""
    from ._textimport import parse_number
    s = t.strip()
    if s == "":
        return MString("")
    low = s.lower()
    if low in ("true", "false"):
        return bool_scalar(low == "true")
    v = parse_number(s)
    if v is not None:
        return np.array([[v]])
    return MString(t.strip())


def _combine(values: list) -> Any:
    """Repeated nodes become one array when they are alike, otherwise a cell array."""
    if len(values) == 1:
        return values[0]
    if all(isinstance(v, np.ndarray) and v.dtype != bool and v.size == 1 for v in values):
        return np.array([[v.flat[0] for v in values]])
    if all(isinstance(v, np.ndarray) and v.dtype == bool for v in values):
        return np.array([[bool(v.flat[0]) for v in values]])
    if all(isinstance(v, MString) for v in values):
        data = np.empty((1, len(values)), dtype=object)
        for k, v in enumerate(values):
            data[0, k] = str(v)
        return StringArray(data)
    if all(isinstance(v, Struct) for v in values):
        fields: list[str] = []
        for v in values:
            for f in v.fields:
                if f not in fields:
                    fields.append(f)
        data = np.empty((1, len(values)), dtype=object)
        missing = np.empty((1, 1), dtype=object)
        missing[0, 0] = None
        for k, v in enumerate(values):
            d = dict(v.data.flat[0])
            for f in fields:
                d.setdefault(f, StringArray(missing.copy()))
            data[0, k] = d
        return Struct(data, fields)
    return Cell.from_list(values)


def _xml_to_struct(node, suffix: str, attributes: bool) -> Any:
    fields: dict[str, list] = {}
    order: list[str] = []

    def add(name: str, value: Any) -> None:
        if name not in fields:
            fields[name] = []
            order.append(name)
        fields[name].append(value)

    if attributes and node.attributes is not None:
        for k in range(node.attributes.length):
            a = node.attributes.item(k)
            if a.name == "xmlns" or a.name.startswith("xmlns:"):
                continue
            add(re.sub(r"\W", "_", a.name) + suffix, _leaf_value(a.value))
    children = [c for c in node.childNodes if c.nodeType == 1]
    own_text = "".join(c.data for c in node.childNodes if c.nodeType in (3, 4)).strip()
    if not children and not fields:
        return _leaf_value(own_text)
    for c in children:
        add(re.sub(r"\W", "_", c.tagName), _xml_to_struct(c, suffix, attributes))
    if own_text:
        add("Text", _leaf_value(own_text))
    return Struct.scalar({name: _combine(fields[name]) for name in order})


def _json_strings(v: Any) -> Any:
    """readstruct returns text as strings: convert the char arrays that jsondecode produced."""
    if isinstance(v, np.ndarray) and v.dtype.kind == "U":
        from ..values import to_str
        return MString(to_str(v)) if v.ndim == 2 and v.shape[0] <= 1 else v
    if isinstance(v, Struct):
        data = np.empty(v.data.shape, dtype=object)
        for idx in np.ndindex(v.data.shape):
            data[idx] = {k: _json_strings(x) for k, x in v.data[idx].items()}
        return Struct(data, list(v.fields))
    if isinstance(v, Cell):
        if v.data.size and all(isinstance(x, np.ndarray) and x.dtype.kind == "U" for x in v.data.flat):
            from ..values import to_str
            data = np.empty(v.data.shape, dtype=object)
            for idx in np.ndindex(v.data.shape):
                data[idx] = to_str(v.data[idx])
            return StringArray(data)
        data = np.empty(v.data.shape, dtype=object)
        for idx in np.ndindex(v.data.shape):
            data[idx] = _json_strings(v.data[idx])
        return Cell(data)
    return v


@builtin("readstruct")
def _readstruct(interp, args, nargout):
    if not args or not is_text(args[0]):
        raise MatlabError("readstruct needs a file name, for example S = readstruct('data.xml').")
    name = text(args[0])
    opts = _struct_options(list(args[1:]), ("FileType", "StructNodeName", "AttributeSuffix", "ImportAttributes"),
                           "readstruct")
    path = find_file(interp, name)
    if path is None:
        raise MatlabError(f"Unable to find or open '{name}'. Check the path and filename or file permissions.")
    kind = _file_type(name, opts, "readstruct")
    if kind == "json":
        from ._registry import BUILTINS
        try:
            with open(path, encoding="utf-8-sig") as fh:
                body = fh.read()
        except OSError as e:
            raise MatlabError(f"Unable to read the file '{name}': {e.strerror or e}.") from None
        value = _json_strings(BUILTINS["jsondecode"](interp, [char(body)], 1))
        if not isinstance(value, Struct):
            raise MatlabError("readstruct: the JSON file does not hold an object, so it cannot become a structure. "
                              "Use jsondecode(fileread(name)) for arrays and plain values.")
        return value
    doc = _parse_xml(path)
    node = doc.documentElement
    if "StructNodeName" in opts:
        want = text(opts["StructNodeName"])
        hits = [node] if node.tagName == want else list(doc.getElementsByTagName(want))
        if not hits:
            raise MatlabError(f"readstruct: the file has no element named '{want}'.")
        node = hits[0]
    suffix = text(opts["AttributeSuffix"]) if "AttributeSuffix" in opts else "Attribute"
    attributes = bool(np.asarray(opts["ImportAttributes"]).any()) if "ImportAttributes" in opts else True
    value = _xml_to_struct(node, suffix, attributes)
    if not isinstance(value, Struct):
        return Struct.scalar({"Text": value})
    return value


# -- NetCDF (classic format, through scipy.io) -----------------------------------------------------------

_NC_TYPES = {"double": ("d", np.float64, 9.9692099683868690e+36), "single": ("f", np.float32, 9.9692099683868690e+36),
             "int32": ("i", np.int32, -2147483647), "int16": ("h", np.int16, -32767),
             "int8": ("b", np.int8, -127), "char": ("c", np.dtype("S1"), b"\0")}
_NC_CLASS = {"d": "double", "f": "single", "i": "int32", "h": "int16", "b": "int8", "c": "char", "B": "int8"}


class _NcFile:
    """A classic NetCDF file in memory: dimensions, variables with their data, attributes."""

    def __init__(self):
        self.dims: dict = {}      # name -> length, None for the unlimited dimension
        self.vars: dict = {}      # name -> {"type": code, "dims": names in file order, "data": array, "attrs": {}}
        self.attrs: dict = {}
        self.version = 1

    def records(self) -> int:
        counts = [len(v["data"]) for v in self.vars.values() if v["dims"] and self.dims.get(v["dims"][0]) is None]
        return max(counts) if counts else 0

    def dim_length(self, name: str) -> int:
        n = self.dims[name]
        return self.records() if n is None else n


def _nc_load(path: str) -> _NcFile:
    from scipy.io import netcdf_file
    try:
        with open(path, "rb") as fh:
            magic = fh.read(8)
    except OSError as e:
        raise MatlabError(f"Unable to open the file '{path}': {e.strerror or e}.", "MATLAB:imagesci:netcdf:fileOpen") \
            from None
    if magic[:3] != b"CDF":
        if magic == b"\x89HDF\r\n\x1a\n":
            raise MatlabError(f"'{path}' is a NetCDF-4 file (stored as HDF5), which is not supported. mylab reads "
                              "and writes the classic NetCDF formats; the h5read functions can read the data of "
                              "a NetCDF-4 file.", "mylab:netcdf:netcdf4")
        raise MatlabError(f"'{path}' is not a NetCDF file.", "MATLAB:imagesci:netcdf:fileOpen")
    nc = _NcFile()
    try:
        f = netcdf_file(path, "r", mmap=False, maskandscale=False)
    except Exception as e:  # scipy raises several types for damaged files
        raise MatlabError(f"Unable to read the NetCDF file '{path}': {e}") from None
    try:
        nc.version = f.version_byte
        nc.dims = dict(f.dimensions)
        nc.attrs = {k: v for k, v in f._attributes.items()}
        for name, v in f.variables.items():
            data = np.array(v.data)
            if data.dtype.kind != "S":
                data = data.astype(data.dtype.newbyteorder("="))
            nc.vars[name] = {"type": v.typecode(), "dims": tuple(v.dimensions), "data": data,
                             "attrs": dict(v._attributes)}
    finally:
        f.close()
    return nc


def _nc_fill(code: str, attrs: dict) -> Any:
    if "_FillValue" in attrs:
        return np.asarray(attrs["_FillValue"]).reshape(-1)[0]
    return next(t[2] for t in _NC_TYPES.values() if t[0] == code)


_NC_CODES = {"b": (1, ">i1"), "B": (1, ">i1"), "c": (2, "S1"), "h": (3, ">i2"), "i": (4, ">i4"), "f": (5, ">f4"),
             "d": (6, ">f8")}


def _nc_attr_bytes(attrs: dict) -> bytes:
    import struct
    if not attrs:
        return b"\0" * 8
    out = [struct.pack(">ii", 0x0C, len(attrs))]
    for name, v in attrs.items():
        raw_name = name.encode("utf-8")
        out.append(struct.pack(">i", len(raw_name)) + raw_name + b"\0" * (-len(raw_name) % 4))
        if isinstance(v, str):
            v = v.encode("utf-8")
        if isinstance(v, bytes):
            code, data, count = 2, v, len(v)
        else:
            a = np.atleast_1d(np.asarray(v))
            if a.dtype.kind == "S":
                data = b"".join(a.reshape(-1).tolist())
                code, count = 2, len(data)
            else:
                kind = {"float64": "d", "float32": "f", "int32": "i", "int16": "h", "int8": "b"}.get(a.dtype.name)
                if kind is None:
                    kind = "d" if a.dtype.kind == "f" else "i"
                code, dt = _NC_CODES[kind]
                data = a.reshape(-1).astype(dt).tobytes()
                count = a.size
        out.append(struct.pack(">ii", code, count) + data + b"\0" * (-len(data) % 4))
    return b"".join(out)


def _nc_save(path: str, nc: _NcFile) -> None:
    """Write the classic NetCDF file: a header, the fixed-size variables, then the records."""
    import struct
    recs = nc.records()
    dim_ids = {name: k for k, name in enumerate(nc.dims)}
    offset_fmt = ">i" if nc.version == 1 else ">q"
    info = []
    for name, v in nc.vars.items():
        code, dt = _NC_CODES[v["type"]]
        is_rec = bool(v["dims"]) and nc.dims[v["dims"][0]] is None
        data = np.asarray(v["data"])
        if is_rec and len(data) < recs:
            pad = np.empty((recs - len(data),) + data.shape[1:], dtype=data.dtype)
            pad[...] = _nc_fill(v["type"], v["attrs"])
            data = np.concatenate([data, pad]) if len(data) else pad
        data = data.astype(dt) if code != 2 else data.astype("S1")
        per = int(np.prod(data.shape[1:], dtype=np.int64)) * data.dtype.itemsize if is_rec else data.nbytes
        info.append({"name": name, "v": v, "code": code, "rec": is_rec, "data": data, "size": per})
    rec_vars = [x for x in info if x["rec"]]
    for x in info:
        # sizes are padded to four bytes, except for a single record variable
        x["vsize"] = x["size"] if (x["rec"] and len(rec_vars) == 1) else x["size"] + (-x["size"] % 4)

    def header(begins: dict) -> bytes:
        out = [b"CDF", bytes([nc.version]), struct.pack(">i", recs)]
        if nc.dims:
            out.append(struct.pack(">ii", 0x0A, len(nc.dims)))
            for name, n in nc.dims.items():
                raw = name.encode("utf-8")
                out.append(struct.pack(">i", len(raw)) + raw + b"\0" * (-len(raw) % 4) + struct.pack(">i", n or 0))
        else:
            out.append(b"\0" * 8)
        out.append(_nc_attr_bytes(nc.attrs))
        if info:
            out.append(struct.pack(">ii", 0x0B, len(info)))
            for x in info:
                raw = x["name"].encode("utf-8")
                dims = x["v"]["dims"]
                out.append(struct.pack(">i", len(raw)) + raw + b"\0" * (-len(raw) % 4))
                out.append(struct.pack(">i", len(dims)) + b"".join(struct.pack(">i", dim_ids[d]) for d in dims))
                out.append(_nc_attr_bytes(x["v"]["attrs"]))
                out.append(struct.pack(">ii", x["code"], min(x["vsize"] + (-x["vsize"] % 4), 2 ** 31 - 1)))
                out.append(struct.pack(offset_fmt, begins.get(x["name"], 0)))
        else:
            out.append(b"\0" * 8)
        return b"".join(out)

    pos = len(header({}))
    begins = {}
    for x in info:
        if not x["rec"]:
            begins[x["name"]] = pos
            pos += x["vsize"]
    for x in rec_vars:
        begins[x["name"]] = pos
        pos += x["vsize"]
    if nc.version == 1 and pos >= 2 ** 31:
        raise MatlabError("The data is too large for a classic NetCDF file; create the file with "
                          "'Format', '64bit'.")
    tmp = path + ".tmp~"
    try:
        with open(tmp, "wb") as fh:
            fh.write(header(begins))
            for x in info:
                if not x["rec"]:
                    raw = x["data"].tobytes()
                    fh.write(raw + b"\0" * (x["vsize"] - len(raw)))
            for r in range(recs):
                for x in rec_vars:
                    raw = x["data"][r].tobytes()
                    fh.write(raw + b"\0" * (x["vsize"] - len(raw)))
        os.replace(tmp, path)
    except OSError as e:
        raise MatlabError(f"Unable to write the NetCDF file '{path}': {e.strerror or e}.") from None
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def _nc_path(interp, v: Any, must_exist: bool = True) -> str:
    name = os.path.expanduser(text(v, "the file name"))
    path = find_file(interp, name)
    if path is None:
        if must_exist:
            raise MatlabError(f"Unable to open the file '{name}': it does not exist.",
                              "MATLAB:imagesci:netcdf:fileOpen")
        return name
    return path


def _nc_var(nc: _NcFile, name: str) -> dict:
    key = name.lstrip("/")
    if key not in nc.vars:
        raise MatlabError(f"The NetCDF file has no variable named '{name}'.",
                          "MATLAB:imagesci:netcdf:unknownLocation")
    return nc.vars[key]


def _nc_attr_in(v: Any) -> Any:
    """A MATLAB attribute value as scipy wants it."""
    if isinstance(v, MString) or (isinstance(v, np.ndarray) and v.dtype.kind == "U"):
        return text(v, "the attribute value")
    if isinstance(v, np.ndarray) and v.dtype.kind in "biuf":
        if class_name(v) not in ("double", "single", "int32", "int16", "int8", "logical"):
            raise MatlabError(f"Attributes of class '{class_name(v)}' cannot be stored in a classic NetCDF file; "
                              "use double, single, int32, int16, int8 or text.")
        a = v.reshape(-1, order="F")
        return a.astype(np.int8) if a.dtype == bool else a
    raise MatlabError("A NetCDF attribute must be numeric or text.")


def _nc_attr_out(v: Any) -> Any:
    if isinstance(v, bytes):
        return char(v.decode("utf-8", errors="replace"))
    if isinstance(v, str):
        return char(v)
    a = np.atleast_1d(np.asarray(v))
    if a.dtype.kind == "S":
        return char(b"".join(a.tolist()).decode("utf-8", errors="replace"))
    return np.ascontiguousarray(a.astype(a.dtype.newbyteorder("="))).reshape(1, -1)


# NetCDF-4 files (HDF5 storage) are handled by _netcdf4, through h5py; the format is detected on reading.
_NC4_FORMATS = ("netcdf4", "netcdf4_classic")


def _nc_is4(path: str) -> bool:
    from . import _netcdf4
    return _netcdf4.is_netcdf4(path)


def _nc4_dims(spec: Any) -> list:
    """{'x', 10, 'y', Inf, 'z'} -> [(name, length or None, unlimited)] in MATLAB order."""
    if not isinstance(spec, Cell):
        raise MatlabError("nccreate: Dimensions must be a cell array such as {'x', 10, 'y', Inf}.")
    items = spec.values_f()
    out = []
    k = 0
    while k < len(items):
        dname = text(items[k], "the dimension name")
        if k + 1 < len(items) and not is_text(items[k + 1]):
            x = float(np.asarray(items[k + 1], dtype=float).flat[0])
            if not np.isinf(x) and (x < 1 or x != int(x)):
                raise MatlabError(f"nccreate: the length of dimension '{dname}' must be a positive integer or Inf.")
            out.append((dname, None if np.isinf(x) else int(x), bool(np.isinf(x))))
            k += 2
        else:
            out.append((dname, None, False))
            k += 1
    return out


def _nc4_location(h5, f, loc: str, fn: str):
    """The group or variable an attribute location names: '/', '/grp', 'var' or '/grp/var'."""
    from . import _netcdf4
    obj = f.get("/" + loc.strip("/")) if loc.strip("/") else f
    if isinstance(obj, h5.Group):
        return obj
    return _netcdf4.find_var(h5, f, loc, fn)


def _nccreate4(path: str, loc: str, opts: dict, fmt: str | None, exists: bool) -> None:
    from . import _netcdf4
    dims = _nc4_dims(opts["Dimensions"]) if "Dimensions" in opts else []
    kind = text(opts["Datatype"]) if "Datatype" in opts else "double"
    if exists:
        h5, f = _netcdf4.open_file(path, "r+")
        have = _netcdf4.file_format(f)
        if fmt is not None and fmt != have:
            f.close()
            raise MatlabError(f"nccreate: '{path}' is a {have} file; its format cannot be changed to '{fmt}'.")
    else:
        h5, f = _netcdf4.create_file(path, fmt)
    with f:
        _netcdf4.create_var(h5, f, loc, dims, kind, opts)


@builtin("nccreate")
def _nccreate(interp, args, nargout):
    if len(args) < 2:
        raise MatlabError("nccreate(filename, varname, ...) needs a file name and a variable name.")
    path = _nc_path(interp, args[0], must_exist=False)
    name = text(args[1], "the variable name").lstrip("/")
    opts = _struct_options(list(args[2:]), ("Dimensions", "Datatype", "Format", "FillValue", "ChunkSize",
                                             "DeflateLevel", "Shuffle"), "nccreate")
    fmt = text(opts["Format"]).lower() if "Format" in opts else None
    if fmt is not None and fmt not in ("classic", "64bit") + _NC4_FORMATS:
        raise MatlabError(f"nccreate: the format '{fmt}' is not supported; use 'classic', '64bit', 'netcdf4' or "
                          "'netcdf4_classic'.")
    exists = os.path.isfile(path)
    if (exists and _nc_is4(path)) or (not exists and fmt in _NC4_FORMATS):
        _nccreate4(path, text(args[1], "the variable name"), opts, fmt, exists)
        return None
    if exists and fmt in _NC4_FORMATS:
        raise MatlabError(f"nccreate: '{path}' is a classic NetCDF file; adding a variable with 'Format', '{fmt}' is "
                          "not supported, because the format of an existing file cannot be changed.")
    for key in ("ChunkSize", "DeflateLevel", "Shuffle"):
        if key in opts:
            raise MatlabError(f"nccreate: '{key}' applies to NetCDF-4 files only; create the file with 'Format', "
                              "'netcdf4' or 'netcdf4_classic'.")
    fmt = fmt or "classic"
    nc = _nc_load(path) if exists else _NcFile()
    if not exists:
        nc.version = 2 if fmt == "64bit" else 1
    if name in nc.vars:
        raise MatlabError(f"The variable '{name}' already exists in '{path}'.", "MATLAB:imagesci:netcdf:variableExists")
    kind = text(opts["Datatype"]) if "Datatype" in opts else "double"
    if kind not in _NC_TYPES:
        raise MatlabError(f"nccreate: the datatype '{kind}' cannot be stored in a classic NetCDF file. Use double, "
                          "single, int32, int16, int8 or char.")
    code, dtype, default_fill = _NC_TYPES[kind]
    dims: list[str] = []
    if "Dimensions" in opts:
        spec = opts["Dimensions"]
        if not isinstance(spec, Cell):
            raise MatlabError("nccreate: Dimensions must be a cell array such as {'x', 10, 'y', Inf}.")
        items = spec.values_f()
        k = 0
        while k < len(items):
            dname = text(items[k], "the dimension name")
            length = None
            if k + 1 < len(items) and not is_text(items[k + 1]):
                x = float(np.asarray(items[k + 1], dtype=float).flat[0])
                length = None if np.isinf(x) else int(x)
                if length is not None and (length < 1 or length != x):
                    raise MatlabError(f"nccreate: the length of dimension '{dname}' must be a positive integer "
                                      "or Inf.")
                k += 2
                if dname in nc.dims and nc.dims[dname] != length:
                    raise MatlabError(f"nccreate: the dimension '{dname}' already exists with another length.")
                nc.dims.setdefault(dname, length)
            else:
                k += 1
                if dname not in nc.dims:
                    raise MatlabError(f"nccreate: the dimension '{dname}' does not exist yet; give its length.")
            dims.append(dname)
    if len(set(dims)) != len(dims):
        raise MatlabError("nccreate: a variable cannot use the same dimension twice.")
    if sum(nc.dims[d] is None for d in nc.dims) > 1:
        raise MatlabError("nccreate: a classic NetCDF file can have only one unlimited (Inf) dimension.")
    file_dims = tuple(reversed(dims))  # MATLAB lists dimensions fastest first; the file stores them slowest first
    if any(nc.dims[d] is None for d in file_dims[1:]):
        raise MatlabError("nccreate: the unlimited (Inf) dimension must be the last dimension of the variable.")
    attrs: dict = {}
    fill: Any = default_fill
    if "FillValue" in opts:
        fv = opts["FillValue"]
        if is_text(fv) and text(fv).lower() == "disable":
            fill = 0 if kind != "char" else b"\0"
        elif kind == "char":
            fill = text(fv, "the fill value").encode("latin-1")[:1] or b"\0"
            attrs["_FillValue"] = fill
        else:
            fill = np.asarray(fv, dtype=float).flat[0]
            attrs["_FillValue"] = np.array([fill], dtype=dtype)
    shape = tuple(0 if nc.dims[d] is None else nc.dims[d] for d in file_dims)
    data = np.empty(shape, dtype=dtype)
    data[...] = fill
    nc.vars[name] = {"type": code, "dims": file_dims, "data": data, "attrs": attrs}
    _nc_save(path, nc)
    return None


def _nc_region(args: list, shape: tuple, fn: str) -> tuple[list, list, list]:
    """(start, count, stride) per MATLAB dimension, zero-based start; count None = to the end."""
    nd = max(len(shape), 1)
    vals = []
    for a in args:
        v = [float(x) for x in np.asarray(a, dtype=float).flatten()]
        if len(v) != nd and not (len(shape) == 0 and len(v) == 1):
            raise MatlabError(f"{fn}: start, count and stride need one value per dimension ({nd}).")
        vals.append(v)
    start = [int(x) - 1 for x in vals[0]] if vals else [0] * nd
    count: list = [None if np.isinf(x) else int(x) for x in vals[1]] if len(vals) > 1 else [None] * nd
    stride = [int(x) for x in vals[2]] if len(vals) > 2 else [1] * nd
    if any(s < 0 for s in start) or any(s < 1 for s in stride) or any(c is not None and c < 0 for c in count):
        raise MatlabError(f"{fn}: start and stride must be positive and count must not be negative.")
    return start, count, stride


@builtin("ncwrite")
def _ncwrite(interp, args, nargout):
    if len(args) < 3 or len(args) > 5:
        raise MatlabError("ncwrite(filename, varname, data) or ncwrite(filename, varname, data, start, stride).")
    path = _nc_path(interp, args[0])
    if _nc_is4(path):
        from . import _netcdf4
        h5, f = _netcdf4.open_file(path, "r+")
        with f:
            name = text(args[1], "the variable name")
            ds = _netcdf4.find_var(h5, f, name, "ncwrite")
            nd = len(_netcdf4.var_dims(h5, f, ds))
            mshape = (1,) * max(nd, 1)
            start, _, stride = _nc_region(([args[3], np.full((1, max(nd, 1)), np.inf), args[4]] if len(args) > 4
                                           else [args[3]] if len(args) > 3 else []), mshape, "ncwrite")
            _netcdf4.write_var(h5, f, ds, args[2], start[::-1][:nd], stride[::-1][:nd], name)
        return None
    nc = _nc_load(path)
    name = text(args[1], "the variable name")
    var = _nc_var(nc, name)
    code = var["type"]
    data = args[2]
    if isinstance(data, MString):
        data = char(str(data))
    if not isinstance(data, np.ndarray):
        raise MatlabError("ncwrite: the data must be a numeric or char array.")
    mdims = tuple(reversed(var["dims"]))  # MATLAB order
    shape = tuple(nc.dim_length(d) for d in mdims)
    start, _, stride = _nc_region(([args[3], np.full((1, max(len(shape), 1)), np.inf), args[4]] if len(args) > 4
                                   else [args[3]] if len(args) > 3 else []), shape, "ncwrite")
    nd = len(mdims)
    if nd == 0:
        if data.size != 1:
            raise MatlabError(f"ncwrite: the variable '{name}' holds one value.")
        block = data.reshape(())
    elif nd == 1:
        if data.ndim > 2 or (data.ndim == 2 and min(data.shape) > 1):
            raise MatlabError(f"ncwrite: the variable '{name}' has one dimension, so the data must be a vector.")
        block = data.reshape(-1, order="F")
    else:
        if data.ndim > nd:
            raise MatlabError(f"ncwrite: the data has more dimensions than the variable '{name}' ({nd}).")
        block = data.reshape(data.shape + (1,) * (nd - data.ndim)).transpose()  # file order
    if code == "c":
        if data.dtype.kind != "U":
            raise MatlabError(f"ncwrite: the variable '{name}' holds characters, so the data must be text.")
        from ..values import char_codes
        stored = char_codes(np.asarray(block)).astype(np.uint8).view("S1").reshape(block.shape)
    else:
        if data.dtype.kind == "U":
            raise MatlabError(f"ncwrite: the variable '{name}' is numeric, so the data must not be text.")
        dtype = next(t[1] for t in _NC_TYPES.values() if t[0] == code)
        vals = np.asarray(block)
        if vals.dtype.kind == "c":
            raise MatlabError("ncwrite: complex data cannot be stored in a NetCDF file.")
        vals = vals.astype(float)
        attrs = var["attrs"]
        if "add_offset" in attrs or "scale_factor" in attrs:
            vals = (vals - float(np.asarray(attrs.get("add_offset", 0)).flat[0])) / \
                float(np.asarray(attrs.get("scale_factor", 1)).flat[0])
        missing = np.isnan(vals)
        if np.dtype(dtype).kind == "i":
            from ..values import saturate_int
            stored = saturate_int(vals, dtype)
        else:
            stored = vals.astype(dtype)
        if missing.any():
            stored = stored.copy()
            stored[missing] = _nc_fill(code, attrs)
    if nd == 0:
        var["data"] = np.array(stored, dtype=var["data"].dtype).reshape(())
        _nc_save(path, nc)
        return None
    # the block's place in the file's (reversed) order
    fstart = list(reversed(start))
    fstride = list(reversed(stride))
    target = var["data"]
    ends = [fstart[k] + (stored.shape[k] - 1) * fstride[k] + 1 if stored.shape[k] else fstart[k]
            for k in range(nd)]
    limits = [nc.dims[d] for d in var["dims"]]
    for k in range(nd):
        if limits[k] is not None and ends[k] > limits[k]:
            raise MatlabError(f"ncwrite: the data does not fit in the variable '{name}' (dimension "
                              f"'{var['dims'][k]}' has length {limits[k]}).", "MATLAB:imagesci:netcdf:libraryFailure")
    if limits[0] is None and ends[0] > target.shape[0]:  # grow along the unlimited dimension
        pad = np.empty((ends[0] - target.shape[0],) + target.shape[1:], dtype=target.dtype)
        pad[...] = _nc_fill(code, var["attrs"])
        target = np.concatenate([target, pad]) if target.shape[0] else pad
    index = tuple(slice(fstart[k], ends[k], fstride[k]) for k in range(nd))
    target[index] = stored
    var["data"] = target
    _nc_save(path, nc)
    return None


@builtin("ncread")
def _ncread(interp, args, nargout):
    if len(args) < 2 or len(args) > 5:
        raise MatlabError("ncread(filename, varname) or ncread(filename, varname, start, count, stride).")
    path = _nc_path(interp, args[0])
    if _nc_is4(path):
        from . import _netcdf4
        h5, f = _netcdf4.open_file(path)
        with f:
            name = text(args[1], "the variable name")
            ds = _netcdf4.find_var(h5, f, name, "ncread")
            nd = len(_netcdf4.var_dims(h5, f, ds))
            start, count, stride = _nc_region(list(args[2:]), (1,) * nd, "ncread")
            raw, cls = _netcdf4.read_var(h5, f, ds, start[::-1][:nd], count[::-1][:nd], stride[::-1][:nd], name)
            return _netcdf4.unpack(h5, ds, _netcdf4.to_matlab(h5, raw, cls, nd), cls)
    nc = _nc_load(path)
    name = text(args[1], "the variable name")
    var = _nc_var(nc, name)
    data = var["data"]
    nd = len(var["dims"])
    if nd == 0:
        out = data.reshape(1, 1)
    else:
        shape = tuple(reversed(data.shape))
        start, count, stride = _nc_region(list(args[2:]), shape, "ncread")
        index = []
        for k in range(nd):
            n = shape[k]
            if start[k] >= n and n:
                raise MatlabError(f"ncread: start ({start[k] + 1}) is beyond the length ({n}) of dimension "
                                  f"{k + 1} of '{name}'.", "MATLAB:imagesci:netcdf:libraryFailure")
            stop = n if count[k] is None else start[k] + (count[k] - 1) * stride[k] + 1
            if stop > n:
                raise MatlabError(f"ncread: start, count and stride reach beyond the length ({n}) of dimension "
                                  f"{k + 1} of '{name}'.", "MATLAB:imagesci:netcdf:libraryFailure")
            index.append(slice(start[k], stop, stride[k]))
        picked = data[tuple(reversed(index))]
        out = picked.transpose() if nd > 1 else picked.reshape(-1, 1)
    if var["type"] == "c":
        codes = np.frombuffer(np.ascontiguousarray(out).tobytes(), dtype=np.uint8).reshape(out.shape)
        return np.ascontiguousarray(codes.astype("<u4")).view("<U1").reshape(out.shape)
    attrs = var["attrs"]
    out = np.ascontiguousarray(out)
    scaled = "add_offset" in attrs or "scale_factor" in attrs
    fill = _nc_fill(var["type"], attrs)
    if out.dtype.kind == "f" or scaled:  # integers keep their class and their fill values
        is_fill = out == fill
        if scaled:
            out = out.astype(float)
        if is_fill.any():
            out = out.copy()
            out[is_fill] = np.nan
        if scaled:
            out = out * float(np.asarray(attrs.get("scale_factor", 1)).flat[0]) + \
                float(np.asarray(attrs.get("add_offset", 0)).flat[0])
    return out


def _nc_attr_struct(attrs: dict) -> Struct:
    data = np.empty((1, len(attrs)), dtype=object)
    for k, (name, v) in enumerate(attrs.items()):
        data[0, k] = {"Name": char(name), "Value": _nc_attr_out(v)}
    return Struct(data, ["Name", "Value"])


def _nc_dim_struct(nc: _NcFile, names: list) -> Struct:
    data = np.empty((1, len(names)), dtype=object)
    for k, d in enumerate(names):
        data[0, k] = {"Name": char(d), "Length": np.array([[float(nc.dim_length(d))]]),
                      "Unlimited": bool_scalar(nc.dims[d] is None)}
    return Struct(data, ["Name", "Length", "Unlimited"])


def _nc_var_info(nc: _NcFile, name: str, path: str) -> dict:
    v = nc.vars[name]
    mdims = list(reversed(v["dims"]))
    size = [float(nc.dim_length(d)) for d in mdims]
    attrs = {k: a for k, a in v["attrs"].items()}
    fill = _nc_attr_out(v["attrs"]["_FillValue"]) if "_FillValue" in v["attrs"] else \
        (char("\0") if v["type"] == "c" else np.array([[_nc_fill(v["type"], {})]]))
    return {"Filename": char(os.path.abspath(path)), "Name": char(name), "Dimensions": _nc_dim_struct(nc, mdims),
            "Size": np.array([size]) if size else np.array([[1.0]]), "Datatype": char(_NC_CLASS[v["type"]]),
            "Attributes": _nc_attr_struct(attrs), "ChunkSize": np.zeros((0, 0)), "FillValue": fill,
            "DeflateLevel": np.zeros((0, 0)), "Shuffle": bool_scalar(False),
            "Format": char("64bit" if nc.version == 2 else "classic")}


@builtin("ncinfo")
def _ncinfo(interp, args, nargout):
    if not args or len(args) > 2:
        raise MatlabError("ncinfo(filename) or ncinfo(filename, varname).")
    path = _nc_path(interp, args[0])
    if _nc_is4(path):
        from . import _netcdf4
        h5, f = _netcdf4.open_file(path)
        with f:
            loc = text(args[1], "the location") if len(args) == 2 else "/"
            obj = f.get("/" + loc.strip("/")) if loc.strip("/") else f
            if isinstance(obj, h5.Group):
                return Struct.scalar(_netcdf4.group_info(h5, f, obj, path))
            return Struct.scalar(_netcdf4.var_info(h5, f, _netcdf4.find_var(h5, f, loc, "ncinfo"), path))
    nc = _nc_load(path)
    if len(args) == 2 and text(args[1]).strip("/") != "":
        name = text(args[1]).lstrip("/")
        _nc_var(nc, name)
        return Struct.scalar(_nc_var_info(nc, name, path))
    fields = ["Name", "Dimensions", "Size", "Datatype", "Attributes", "ChunkSize", "FillValue", "DeflateLevel",
              "Shuffle"]
    vdata = np.empty((1, len(nc.vars)), dtype=object)
    for k, name in enumerate(nc.vars):
        info = _nc_var_info(nc, name, path)
        vdata[0, k] = {f: info[f] for f in fields}
    return Struct.scalar({"Filename": char(os.path.abspath(path)), "Name": char("/"),
                          "Dimensions": _nc_dim_struct(nc, list(nc.dims)), "Variables": Struct(vdata, fields),
                          "Attributes": _nc_attr_struct(nc.attrs), "Groups": np.zeros((0, 0)),
                          "Format": char("64bit" if nc.version == 2 else "classic")})


def _nc_show(v: Any) -> str:
    a = _nc_attr_out(v)
    if isinstance(a, np.ndarray) and a.dtype.kind == "U":
        return "'" + "".join(a.reshape(-1).tolist()) + "'"
    flat = np.asarray(a).reshape(-1)
    return " ".join(f"{x:g}" for x in flat) if flat.size != 1 else f"{flat[0]:g}"


@builtin("ncdisp")
def _ncdisp(interp, args, nargout):
    if not args or len(args) > 3:
        raise MatlabError("ncdisp(filename) or ncdisp(filename, varname).")
    path = _nc_path(interp, args[0])
    if _nc_is4(path):
        from . import _netcdf4
        h5, f = _netcdf4.open_file(path)
        with f:
            loc = text(args[1], "the location") if len(args) > 1 else "/"
            obj = f.get("/" + loc.strip("/")) if loc.strip("/") else f
            only = None
            if not isinstance(obj, h5.Group):
                ds = _netcdf4.find_var(h5, f, loc, "ncdisp")
                obj, only = ds.parent, ds.name.rsplit("/", 1)[-1]
            lines = ["Source:", f"           {os.path.abspath(path)}", "Format:",
                     f"           {_netcdf4.file_format(f)}"]
            lines += _netcdf4.disp_lines(h5, f, obj, only, "")
        interp.out("\n".join(lines) + "\n")
        return None
    nc = _nc_load(path)
    only = text(args[1]).lstrip("/") if len(args) > 1 and text(args[1]).strip("/") else None
    if only is not None:
        _nc_var(nc, only)
    lines = ["Source:", f"           {os.path.abspath(path)}", "Format:",
             f"           {'64bit' if nc.version == 2 else 'classic'}"]
    if nc.attrs and only is None:
        lines.append("Global Attributes:")
        w = max(len(k) for k in nc.attrs)
        lines += [f"           {k:<{w}} = {_nc_show(v)}" for k, v in nc.attrs.items()]
    if nc.dims and only is None:
        lines.append("Dimensions:")
        w = max(len(k) for k in nc.dims)
        for k, n in nc.dims.items():
            extra = "  (UNLIMITED)" if n is None else ""
            lines.append(f"           {k:<{w}} = {nc.dim_length(k)}{extra}")
    lines.append("Variables:")
    for name, v in nc.vars.items():
        if only is not None and name != only:
            continue
        mdims = list(reversed(v["dims"]))
        size = "x".join(str(nc.dim_length(d)) for d in mdims) or "1x1"
        lines += [f"    {name}", f"           Size:       {size}",
                  f"           Dimensions: {','.join(mdims)}",
                  f"           Datatype:   {_NC_CLASS[v['type']]}"]
        if v["attrs"]:
            lines.append("           Attributes:")
            w = max(len(k) for k in v["attrs"])
            lines += [f"                       {k:<{w}} = {_nc_show(a)}" for k, a in v["attrs"].items()]
    interp.out("\n".join(lines) + "\n")
    return None


@builtin("ncreadatt")
def _ncreadatt(interp, args, nargout):
    if len(args) != 3:
        raise MatlabError("ncreadatt(filename, location, attname): the location is '/' or a variable name.")
    path = _nc_path(interp, args[0])
    loc, att = text(args[1], "the location"), text(args[2], "the attribute name")
    if _nc_is4(path):
        from . import _netcdf4
        h5, f = _netcdf4.open_file(path)
        with f:
            obj = _nc4_location(h5, f, loc, "ncreadatt")
            if att not in obj.attrs:
                raise MatlabError(f"The attribute '{att}' was not found at '{loc}'.",
                                  "MATLAB:imagesci:netcdf:libraryFailure")
            return _netcdf4.attr_out(h5, obj, att)
    nc = _nc_load(path)
    attrs = nc.attrs if loc.strip("/") == "" else _nc_var(nc, loc)["attrs"]
    if att not in attrs:
        raise MatlabError(f"The attribute '{att}' was not found at '{loc}'.", "MATLAB:imagesci:netcdf:libraryFailure")
    return _nc_attr_out(attrs[att])


@builtin("ncwriteatt")
def _ncwriteatt(interp, args, nargout):
    if len(args) != 4:
        raise MatlabError("ncwriteatt(filename, location, attname, value): the location is '/' or a variable name.")
    path = _nc_path(interp, args[0])
    if _nc_is4(path):
        from . import _netcdf4
        h5, f = _netcdf4.open_file(path, "r+")
        with f:
            loc, att = text(args[1], "the location"), text(args[2], "the attribute name")
            obj = _nc4_location(h5, f, loc, "ncwriteatt")
            if att in _netcdf4._HIDDEN:
                raise MatlabError(f"ncwriteatt: '{att}' is reserved for the NetCDF-4 storage layout.")
            dtype = obj.dtype if att == "_FillValue" and isinstance(obj, h5.Dataset) else None
            _netcdf4.attr_write(h5, obj, att, args[3], _netcdf4.file_format(f) == "netcdf4_classic", dtype)
        return None
    nc = _nc_load(path)
    loc, att = text(args[1], "the location"), text(args[2], "the attribute name")
    attrs = nc.attrs if loc.strip("/") == "" else _nc_var(nc, loc)["attrs"]
    attrs[att] = _nc_attr_in(args[3])
    _nc_save(path, nc)
    return None


# -- HDF5 (through the h5py package) -------------------------------------------------------------------

_H5_TYPES = {"double": np.float64, "single": np.float32, "int8": np.int8, "uint8": np.uint8, "int16": np.int16,
             "uint16": np.uint16, "int32": np.int32, "uint32": np.uint32, "int64": np.int64, "uint64": np.uint64}


def _h5():
    from ._mat73 import h5py_module
    return h5py_module("HDF5 files")


def _h5_open(interp, name_value: Any, mode: str, fn: str):
    h5 = _h5()
    name = os.path.expanduser(text(name_value, "the file name"))
    path = find_file(interp, name) or name
    if mode == "r" and not os.path.isfile(path):
        raise MatlabError(f"{fn}: the file '{name}' does not exist.", "MATLAB:imagesci:hdf5io:resourceNotFound")
    try:
        return h5, h5.File(path, mode)
    except OSError as e:
        raise MatlabError(f"{fn}: unable to open '{name}' as an HDF5 file ({e}).",
                          "MATLAB:imagesci:hdf5lib:fileOpenErr") from None


def _h5_name(v: Any, fn: str) -> str:
    name = text(v, "the dataset name")
    if not name.startswith("/"):
        raise MatlabError(f"{fn}: the name '{name}' must be a full path starting with '/'.")
    return name


def _h5_object(f, name: str, fn: str):
    if name not in f:
        raise MatlabError(f"{fn}: '{name}' was not found in the file.", "MATLAB:imagesci:h5info:unableToFind")
    return f[name]


@builtin("h5create")
def _h5create(interp, args, nargout):
    if len(args) < 3:
        raise MatlabError("h5create(filename, datasetname, size, ...) needs a file, a dataset name and a size.")
    name = _h5_name(args[1], "h5create")
    size = [float(x) for x in np.asarray(args[2], dtype=float).flatten()]
    if not size or any(x < 0 or (not np.isinf(x) and x != int(x)) for x in size):
        raise MatlabError("h5create: the size must be a vector of non-negative integers (Inf for unlimited).")
    opts = _struct_options(list(args[3:]), ("Datatype", "ChunkSize", "Deflate", "FillValue", "Fletcher32",
                                             "Shuffle", "TextEncoding"), "h5create")
    h5, f = _h5_open(interp, args[0], "a", "h5create")
    with f:
        if name in f:
            raise MatlabError(f"h5create: the dataset '{name}' already exists.",
                              "MATLAB:imagesci:h5create:datasetAlreadyExists")
        kind = text(opts["Datatype"]) if "Datatype" in opts else "double"
        if kind == "string":
            enc = text(opts["TextEncoding"]).lower() if "TextEncoding" in opts else "utf-8"
            if enc not in ("utf-8", "system"):
                raise MatlabError("h5create: TextEncoding must be 'UTF-8' or 'system'.")
            dtype = h5.string_dtype(encoding="utf-8")
        elif kind in _H5_TYPES:
            dtype = np.dtype(_H5_TYPES[kind])
        else:
            raise MatlabError(f"h5create: unknown Datatype '{kind}'. Types are: {', '.join(_H5_TYPES)}, string.")
        unlimited = [np.isinf(x) for x in size]
        chunks = None
        if "ChunkSize" in opts:
            chunks = tuple(int(x) for x in np.asarray(opts["ChunkSize"], dtype=float).flatten())
            if len(chunks) != len(size) or any(c < 1 for c in chunks):
                raise MatlabError("h5create: ChunkSize needs one positive value per dimension.")
        if any(unlimited) and chunks is None:
            raise MatlabError("h5create: a dataset with an unlimited (Inf) dimension needs a ChunkSize.")
        kw: dict = {}
        if "Deflate" in opts:
            level = int(np.asarray(opts["Deflate"]).flat[0])
            if not 0 <= level <= 9:
                raise MatlabError("h5create: Deflate must be between 0 and 9.")
            if level:
                kw.update(compression="gzip", compression_opts=level)
        for key, arg in (("Fletcher32", "fletcher32"), ("Shuffle", "shuffle")):
            if key in opts and bool(np.asarray(opts[key]).any()):
                kw[arg] = True
        if kw and chunks is None:
            raise MatlabError("h5create: Deflate, Shuffle and Fletcher32 need a ChunkSize.")
        if "FillValue" in opts and kind != "string":
            kw["fillvalue"] = np.asarray(opts["FillValue"]).astype(dtype).flat[0]
        # MATLAB's first dimension varies fastest: the file stores the dimensions reversed
        shape = tuple(0 if u else int(x) for x, u in zip(size, unlimited))[::-1]
        maxshape = tuple(None if u else int(x) for x, u in zip(size, unlimited))[::-1] if any(unlimited) else None
        try:
            f.create_dataset(name, shape=shape, maxshape=maxshape, dtype=dtype,
                             chunks=None if chunks is None else chunks[::-1], **kw)
        except (ValueError, TypeError, OSError) as e:
            raise MatlabError(f"h5create: cannot create '{name}': {e}") from None
    return None


def _h5_region(args: list, shape: tuple, fn: str) -> tuple[list, list, list]:
    nd = len(shape)
    vals = []
    for a in args:
        v = [float(x) for x in np.asarray(a, dtype=float).flatten()]
        if len(v) != nd:
            raise MatlabError(f"{fn}: start, count and stride need one value per dimension ({nd}).")
        vals.append(v)
    start = [int(x) - 1 for x in vals[0]] if vals else [0] * nd
    count: list = [None if np.isinf(x) else int(x) for x in vals[1]] if len(vals) > 1 else [None] * nd
    stride = [int(x) for x in vals[2]] if len(vals) > 2 else [1] * nd
    if any(s < 0 for s in start) or any(s < 1 for s in stride):
        raise MatlabError(f"{fn}: start and stride must be positive.")
    return start, count, stride


def _h5_text_items(v: Any) -> list | None:
    if isinstance(v, MString):
        return [str(v)]
    if isinstance(v, StringArray):
        return ["" if x is None else x for x in v.data.flatten(order="C")]
    if isinstance(v, Cell) and all(is_text(x) for x in v.data.flat):
        return [text(x) for x in v.data.flatten(order="C")]
    if isinstance(v, np.ndarray) and v.dtype.kind == "U":
        return [text(v)] if v.shape[0] <= 1 else ["".join(r) for r in v]
    return None


@builtin("h5write")
def _h5write(interp, args, nargout):
    if len(args) not in (3, 5, 6):
        raise MatlabError("h5write(filename, datasetname, data) or h5write(..., start, count) or "
                          "h5write(..., start, count, stride).")
    name = _h5_name(args[1], "h5write")
    data = args[2]
    h5, f = _h5_open(interp, args[0], "r+", "h5write")
    with f:
        ds = _h5_object(f, name, "h5write")
        if not isinstance(ds, h5.Dataset):
            raise MatlabError(f"h5write: '{name}' is a group, not a dataset.")
        is_string = h5.check_string_dtype(ds.dtype) is not None
        if is_string:
            items = _h5_text_items(data)
            if items is None:
                raise MatlabError(f"h5write: the dataset '{name}' holds strings, so the data must be text.")
            dshape = (1, 1) if isinstance(data, MString) or len(items) == 1 else \
                (data.data.shape if hasattr(data, "data") else (len(items), 1))
            values = np.array(items, dtype=object).reshape(dshape)  # items are in row-major order
        else:
            if not isinstance(data, np.ndarray) or data.dtype.kind not in "biuf":
                raise MatlabError(f"h5write: the dataset '{name}' is numeric, so the data must be a real numeric "
                                  "array.")
            values = data
        nd = ds.ndim
        if nd == 0:
            if values.size != 1:
                raise MatlabError(f"h5write: the dataset '{name}' holds one value.")
            ds[()] = values.reshape(-1)[0]
            return None
        mshape = ds.shape[::-1]  # the size as MATLAB sees it
        start, count, stride = _h5_region(list(args[3:]), mshape, "h5write")
        if len(args) > 3:
            want = [c if c is not None else max(mshape[k] - start[k], 0) for k, c in enumerate(count)]
        elif not any(m is None for m in ds.maxshape):
            want = list(mshape)
        elif values.ndim == nd:
            want = list(values.shape)
        elif nd == 1:
            want = [values.size]
        elif values.ndim < nd:
            want = list(values.shape) + [1] * (nd - values.ndim)
        else:
            raise MatlabError(f"h5write: the data has more dimensions than the dataset '{name}' ({nd}).")
        if values.size != int(np.prod(want)):
            given = "x".join(map(str, values.shape))
            raise MatlabError(f"h5write: the data ({given}) does not match the size to be written "
                              f"({'x'.join(map(str, want))}) of the dataset '{name}'.",
                              "MATLAB:imagesci:h5write:fullDatasetDataMismatch")
        block = np.asarray(values).reshape(want, order="F").transpose()  # the file stores dimensions reversed
        fstart, fstride = start[::-1], stride[::-1]
        ends = [fstart[k] + (block.shape[k] - 1) * fstride[k] + 1 if block.shape[k] else fstart[k]
                for k in range(nd)]
        grow = [max(ds.shape[k], ends[k]) for k in range(nd)]
        if tuple(grow) != tuple(ds.shape):
            for k in range(nd):
                if ds.maxshape[k] is not None and grow[k] > ds.maxshape[k]:
                    raise MatlabError(f"h5write: the data reaches beyond the size of the dataset '{name}'.")
            ds.resize(tuple(grow))
        try:
            ds[tuple(slice(fstart[k], ends[k], fstride[k]) for k in range(nd))] = block
        except (TypeError, ValueError, OSError) as e:
            raise MatlabError(f"h5write: cannot write to '{name}': {e}") from None
    return None


def _h5_value(h5, f, raw: Any, dtype) -> Any:
    """MATLAB value for data read from a dataset or attribute."""
    a = np.asarray(raw)
    if h5.check_string_dtype(dtype) is not None or a.dtype.kind in "SO":
        flat = a.reshape(-1)
        items = [x.decode("utf-8", errors="replace") if isinstance(x, bytes) else str(x) for x in flat.tolist()]
        if a.ndim == 0:
            return MString(items[0])
        out = np.empty(len(items), dtype=object)
        for k, s in enumerate(items):
            out[k] = s
        shaped = out.reshape(a.shape).transpose() if a.ndim > 1 else out.reshape(-1, 1)
        return StringArray(np.ascontiguousarray(shaped)) if shaped.size != 1 else MString(items[0])
    if a.dtype.names:
        fields = {}
        for n in a.dtype.names:
            fields[n] = _h5_value(h5, f, a[n], a.dtype[n])
        return Struct.scalar(fields)
    if a.dtype.kind == "b":
        a = a.astype(bool)
    elif a.dtype.kind not in "iuf":
        raise MatlabError(f"HDF5 data of type {a.dtype} cannot be converted to a MATLAB value.")
    a = a.astype(a.dtype.newbyteorder("="))
    if a.ndim == 0:
        return a.reshape(1, 1)
    if a.ndim == 1:
        return np.ascontiguousarray(a.reshape(-1, 1))
    return np.ascontiguousarray(a.transpose())


@builtin("h5read")
def _h5read(interp, args, nargout):
    if len(args) < 2 or len(args) > 5:
        raise MatlabError("h5read(filename, datasetname) or h5read(filename, datasetname, start, count, stride).")
    name = _h5_name(args[1], "h5read")
    h5, f = _h5_open(interp, args[0], "r", "h5read")
    with f:
        ds = _h5_object(f, name, "h5read")
        if not isinstance(ds, h5.Dataset):
            raise MatlabError(f"h5read: '{name}' is a group, not a dataset.")
        if ds.ndim == 0 or len(args) == 2:
            raw = ds[()]
        else:
            mshape = ds.shape[::-1]
            start, count, stride = _h5_region(list(args[2:]), mshape, "h5read")
            index = []
            for k in range(ds.ndim):
                n = mshape[k]
                stop = n if count[k] is None else start[k] + (count[k] - 1) * stride[k] + 1
                if start[k] >= n or stop > n:
                    raise MatlabError(f"h5read: start, count and stride reach beyond the size of '{name}' "
                                      f"({'x'.join(map(str, mshape))}).")
                index.append(slice(start[k], stop, stride[k]))
            raw = ds[tuple(index[::-1])]
        return _h5_value(h5, f, raw, ds.dtype)


def _h5_location(f, v: Any, fn: str):
    loc = text(v, "the location")
    if not loc.startswith("/"):
        raise MatlabError(f"{fn}: the location '{loc}' must be a full path starting with '/'.")
    return _h5_object(f, loc, fn)


@builtin("h5writeatt")
def _h5writeatt(interp, args, nargout):
    if len(args) < 4:
        raise MatlabError("h5writeatt(filename, location, attname, value).")
    opts = _struct_options(list(args[4:]), ("TextEncoding",), "h5writeatt")
    if "TextEncoding" in opts and text(opts["TextEncoding"]).lower() not in ("utf-8", "system"):
        raise MatlabError("h5writeatt: TextEncoding must be 'UTF-8' or 'system'.")
    att = text(args[2], "the attribute name")
    v = args[3]
    h5, f = _h5_open(interp, args[0], "r+", "h5writeatt")
    with f:
        obj = _h5_location(f, args[1], "h5writeatt")
        items = _h5_text_items(v)
        if items is not None:
            value: Any = items[0] if len(items) == 1 else np.array(items, dtype=h5.string_dtype())
        elif isinstance(v, np.ndarray) and v.dtype.kind in "biuf":
            value = v.reshape(-1)[0] if v.size == 1 else (v.reshape(-1) if 1 in v.shape else v.transpose())
        else:
            raise MatlabError("h5writeatt: the attribute value must be numeric or text.")
        if att in obj.attrs:
            del obj.attrs[att]
        obj.attrs[att] = value
    return None


@builtin("h5readatt")
def _h5readatt(interp, args, nargout):
    if len(args) != 3:
        raise MatlabError("h5readatt(filename, location, attname).")
    att = text(args[2], "the attribute name")
    h5, f = _h5_open(interp, args[0], "r", "h5readatt")
    with f:
        obj = _h5_location(f, args[1], "h5readatt")
        if att not in obj.attrs:
            raise MatlabError(f"h5readatt: the attribute '{att}' was not found at '{text(args[1])}'.",
                              "MATLAB:imagesci:h5readatt:cannotOpenAttribute")
        raw = obj.attrs[att]
        dtype = obj.attrs.get_id(att).dtype
        value = _h5_value(h5, f, raw, dtype)
        if isinstance(value, MString):
            return char(str(value))  # a text attribute comes back as a character vector
        return value


def _h5_class(h5, dtype) -> tuple[str, str]:
    """(HDF5 class name, type name) for the Datatype part of h5info."""
    if h5.check_string_dtype(dtype) is not None or dtype.kind in "SO":
        return "H5T_STRING", "H5T_C_S1"
    if dtype.names:
        return "H5T_COMPOUND", "H5T_COMPOUND"
    order = "BE" if dtype.byteorder == ">" else "LE"
    if dtype.kind == "f":
        return "H5T_FLOAT", f"H5T_IEEE_F{dtype.itemsize * 8}{order}"
    if dtype.kind in "iu":
        return "H5T_INTEGER", f"H5T_STD_{'I' if dtype.kind == 'i' else 'U'}{dtype.itemsize * 8}{order}"
    if dtype.kind == "b":
        return "H5T_ENUM", "H5T_STD_I8LE"
    return "H5T_OPAQUE", str(dtype)


def _h5_attributes(h5, f, obj) -> Struct:
    names = list(obj.attrs.keys())
    data = np.empty((len(names), 1), dtype=object)
    for k, n in enumerate(names):
        dtype = obj.attrs.get_id(n).dtype
        cls, tname = _h5_class(h5, dtype)
        try:
            value = _h5_value(h5, f, obj.attrs[n], dtype)
            if isinstance(value, MString):
                value = char(str(value))
        except MatlabError:
            value = np.zeros((0, 0))
        shape = np.shape(obj.attrs[n])
        data[k, 0] = {"Name": char(n), "Datatype": _h5_datatype(cls, tname, dtype),
                      "Dataspace": _h5_dataspace(shape, shape), "Value": value}
    return Struct(data, ["Name", "Datatype", "Dataspace", "Value"])


def _h5_datatype(cls: str, tname: str, dtype) -> Struct:
    return Struct.scalar({"Name": char(""), "Class": char(cls), "Type": char(tname),
                          "Size": np.array([[float(dtype.itemsize)]]), "Attributes": np.zeros((0, 0))})


def _h5_dataspace(shape: tuple, maxshape: tuple) -> Struct:
    size = np.array([[float(x) for x in shape[::-1]]]) if shape else np.array([[1.0]])
    mx = np.array([[np.inf if x is None else float(x) for x in maxshape[::-1]]]) if maxshape else np.array([[1.0]])
    return Struct.scalar({"Size": size, "MaxSize": mx, "Type": char("simple" if shape else "scalar")})


def _h5_dataset_info(h5, f, name: str, ds) -> dict:
    cls, tname = _h5_class(h5, ds.dtype)
    filters = []
    if ds.shuffle:
        filters.append({"Name": char("shuffle"), "Data": np.zeros((0, 0))})
    if ds.compression == "gzip":
        filters.append({"Name": char("deflate"), "Data": np.array([[float(ds.compression_opts or 0)]])})
    if ds.fletcher32:
        filters.append({"Name": char("fletcher32"), "Data": np.zeros((0, 0))})
    fdata = np.empty((len(filters), 1), dtype=object)
    for k, x in enumerate(filters):
        fdata[k, 0] = x
    fill: Any = np.zeros((0, 0))
    if ds.dtype.kind in "iufb" and ds.fillvalue is not None:
        fill = np.array([[ds.fillvalue]], dtype=ds.dtype.newbyteorder("="))
    return {"Name": char(name), "Datatype": _h5_datatype(cls, tname, ds.dtype),
            "Dataspace": _h5_dataspace(ds.shape, ds.maxshape),
            "ChunkSize": np.array([[float(x) for x in ds.chunks[::-1]]]) if ds.chunks else np.zeros((0, 0)),
            "FillValue": fill, "Filters": Struct(fdata, ["Name", "Data"]) if filters else np.zeros((0, 0)),
            "Attributes": _h5_attributes(h5, f, ds)}


_H5_DATASET_FIELDS = ["Name", "Datatype", "Dataspace", "ChunkSize", "FillValue", "Filters", "Attributes"]


def _h5_group_info(h5, f, g, path: str) -> dict:
    groups, datasets = [], []
    for key in g.keys():
        child = g.get(key, getclass=True)
        if child is h5.Group:
            groups.append(key)
        elif child is h5.Dataset:
            datasets.append(key)
    gdata = np.empty((len(groups), 1), dtype=object)
    for k, key in enumerate(groups):
        sub = g[key]
        gdata[k, 0] = _h5_group_info(h5, f, sub, sub.name)
        gdata[k, 0].pop("Filename", None)
    ddata = np.empty((len(datasets), 1), dtype=object)
    for k, key in enumerate(datasets):
        ddata[k, 0] = _h5_dataset_info(h5, f, key, g[key])
    gfields = ["Name", "Groups", "Datasets", "Datatypes", "Links", "Attributes"]
    return {"Filename": char(os.path.abspath(path)), "Name": char(g.name),
            "Groups": Struct(gdata, gfields) if groups else np.zeros((0, 0)),
            "Datasets": Struct(ddata, _H5_DATASET_FIELDS) if datasets else np.zeros((0, 0)),
            "Datatypes": np.zeros((0, 0)), "Links": np.zeros((0, 0)), "Attributes": _h5_attributes(h5, f, g)}


@builtin("h5info")
def _h5info(interp, args, nargout):
    if not args or len(args) > 2:
        raise MatlabError("h5info(filename) or h5info(filename, location).")
    h5, f = _h5_open(interp, args[0], "r", "h5info")
    with f:
        loc = text(args[1], "the location") if len(args) > 1 else "/"
        obj = _h5_location(f, char(loc), "h5info")
        if isinstance(obj, h5.Dataset):
            info = _h5_dataset_info(h5, f, obj.name.rsplit("/", 1)[-1], obj)
            info = {"Filename": char(os.path.abspath(f.filename)), **info}
            return Struct.scalar(info)
        info = _h5_group_info(h5, f, obj, f.filename)
        info["Filename"] = char(os.path.abspath(f.filename))
        return Struct.scalar(info)


def _h5_show_value(v: Any) -> str:
    if isinstance(v, np.ndarray) and v.dtype.kind == "U":
        return "'" + "".join(v.reshape(-1).tolist()) + "'"
    if isinstance(v, (MString, StringArray)):
        items = [str(v)] if isinstance(v, MString) else [x or "" for x in v.data.flatten(order="F")]
        return ", ".join(f"'{x}'" for x in items)
    if isinstance(v, np.ndarray):
        return " ".join(f"{x:g}" for x in v.reshape(-1)[:20])
    return ""


def _h5_disp(h5, f, obj, indent: str, lines: list) -> None:
    def attrs(o, pad: str) -> None:
        if len(o.attrs):
            lines.append(f"{pad}Attributes:")
            for n in o.attrs.keys():
                try:
                    value = _h5_value(h5, f, o.attrs[n], o.attrs.get_id(n).dtype)
                    if isinstance(value, MString):
                        value = char(str(value))
                    shown = _h5_show_value(value)
                except MatlabError:
                    shown = "(not shown)"
                lines.append(f"{pad}    '{n}':  {shown}")

    if isinstance(obj, h5.Dataset):
        cls, tname = _h5_class(h5, obj.dtype)
        shape = obj.shape[::-1]
        mx = ["Inf" if m is None else str(m) for m in obj.maxshape[::-1]]
        lines.append(f"{indent}Dataset '{obj.name.rsplit('/', 1)[-1]}' ")
        lines.append(f"{indent}    Size:  {'x'.join(map(str, shape)) or 'scalar'}")
        lines.append(f"{indent}    MaxSize:  {'x'.join(mx) or 'scalar'}")
        lines.append(f"{indent}    Datatype:   {tname} ({class_name_of(obj.dtype)})")
        lines.append(f"{indent}    ChunkSize:  {'x'.join(map(str, obj.chunks[::-1])) if obj.chunks else '[]'}")
        filters = [n for n, on in (("shuffle", obj.shuffle), (f"deflate({obj.compression_opts})",
                                                               obj.compression == "gzip"),
                                   ("fletcher32", obj.fletcher32)) if on]
        lines.append(f"{indent}    Filters:  {', '.join(filters) if filters else 'none'}")
        if obj.dtype.kind in "iufb" and obj.fillvalue is not None:
            lines.append(f"{indent}    FillValue:  {float(obj.fillvalue):f}")
        attrs(obj, indent + "    ")
        return
    lines.append(f"{indent}Group '{obj.name}' ")
    attrs(obj, indent + "    ")
    for key in obj.keys():
        child = obj.get(key, getclass=True)
        if child in (h5.Group, h5.Dataset):
            _h5_disp(h5, f, obj[key], indent + "    ", lines)


def class_name_of(dtype) -> str:
    if dtype.kind in "SO":
        return "string"
    return {v: k for k, v in _H5_TYPES.items()}.get(np.dtype(dtype.newbyteorder("=")).type, str(dtype))


@builtin("h5disp")
def _h5disp(interp, args, nargout):
    if not args or len(args) > 3:
        raise MatlabError("h5disp(filename) or h5disp(filename, location).")
    h5, f = _h5_open(interp, args[0], "r", "h5disp")
    with f:
        loc = char(text(args[1], "the location")) if len(args) > 1 else char("/")
        obj = _h5_location(f, loc, "h5disp")
        lines = [f"HDF5 {os.path.basename(f.filename)} "]
        _h5_disp(h5, f, obj, "", lines)
        interp.out("\n".join(lines) + "\n")
    return None


# -- video files ---------------------------------------------------------------------------------------

class VideoReaderObject(MObject):
    """VideoReader: frames of a video file. AVI files with Motion JPEG or uncompressed frames are decoded
    here; other files are decoded by ffmpeg when it is installed."""

    is_handle = True
    _METHODS = {"readFrame", "hasFrame", "read", "get", "getFileFormats"}

    def __init__(self, path: str):
        from . import _avi
        self.path = path
        self.avi = None
        self.decoded = None  # frames decoded by ffmpeg (a memory-mapped array)
        self.next = 0        # index of the frame that readFrame returns next
        self.tag = char("")
        self.user_data: Any = np.zeros((0, 0))
        with open(path, "rb") as fh:
            head = fh.read(12)
        first: np.ndarray | None = None
        if head[:4] == b"RIFF" and head[8:12] == b"AVI ":
            self.avi = _avi.AviFile(path)
            try:
                first = self.avi.frame(0) if self.avi.frames else None
            except _avi.UnsupportedCodec:
                self.avi = None
        if self.avi is None:
            self._decode_all()
            first = np.asarray(self.decoded[0]) if self.count else None
        else:
            self.rate = self.avi.rate or 30.0
            self.count = len(self.avi.frames)
        self.height = int(first.shape[0]) if first is not None else int(abs(getattr(self.avi, "height", 0) or 0))
        self.width = int(first.shape[1]) if first is not None else int(getattr(self.avi, "width", 0) or 0)
        self.gray = first is not None and first.ndim == 2

    def _decode_all(self) -> None:
        from . import _avi
        from .plotting import _find_ffmpeg
        exe = _find_ffmpeg()
        if exe is None:
            raise MatlabError(f"VideoReader: '{os.path.basename(self.path)}' is not an AVI file with Motion JPEG or "
                              "uncompressed frames, and reading other video formats needs the ffmpeg program, "
                              "which was not found. Install ffmpeg, or convert the video to a Motion JPEG AVI.",
                              "MATLAB:audiovideo:VideoReader:InitializationFailed")
        self.decoded, self.rate = _avi.decode_with_ffmpeg(self.path, exe)
        self.count = int(self.decoded.shape[0])
        self._temp = self.decoded.filename

    def __del__(self):
        temp = getattr(self, "_temp", None)
        if temp:
            try:
                self.decoded = None
                os.remove(temp)
            except OSError:
                pass

    def class_name(self) -> str:
        return "VideoReader"

    def isa_class(self, name: str) -> bool:
        return name in ("VideoReader", "handle")

    def has_method(self, name: str) -> bool:
        return name in self._METHODS

    def frame(self, k: int) -> np.ndarray:
        if self.avi is not None:
            return np.ascontiguousarray(self.avi.frame(k))
        return np.array(self.decoded[k])

    def frames(self, a: int, b: int) -> np.ndarray:
        """Frames a..b (from zero, inclusive) stacked along the fourth dimension."""
        items = [self.frame(k) for k in range(a, b + 1)]
        if len(items) == 1:
            return items[0]
        if self.gray:
            return np.stack(items, axis=-1)[:, :, None, :]
        return np.stack(items, axis=-1)

    _PROPS = ("Name", "Path", "Duration", "CurrentTime", "Tag", "UserData", "NumFrames", "NumberOfFrames",
              "FrameRate", "Height", "Width", "BitsPerPixel", "VideoFormat")

    def get_field(self, interp, name: str) -> Any:
        one = lambda x: np.array([[float(x)]])  # noqa: E731
        values = {
            "Name": lambda: char(os.path.basename(self.path)), "Path": lambda: char(os.path.dirname(self.path)),
            "Duration": lambda: one(self.count / self.rate if self.rate else 0.0),
            "CurrentTime": lambda: one(self.next / self.rate if self.rate else 0.0),
            "Tag": lambda: self.tag, "UserData": lambda: self.user_data,
            "NumFrames": lambda: one(self.count), "NumberOfFrames": lambda: one(self.count),
            "FrameRate": lambda: one(self.rate), "Height": lambda: one(self.height), "Width": lambda: one(self.width),
            "BitsPerPixel": lambda: one(8 if self.gray else 24),
            "VideoFormat": lambda: char("Grayscale" if self.gray else "RGB24"),
        }
        for k, fn in values.items():
            if k.lower() == name.lower():
                return fn()
        if name in self._METHODS:
            return super().get_field(interp, name)
        raise MatlabError(f"Unrecognized property '{name}' for class 'VideoReader'.")

    def set_field(self, interp, name: str, value: Any) -> "VideoReaderObject":
        key = name.lower()
        if key == "currenttime":
            t = float(np.asarray(value, dtype=float).flat[0])
            duration = self.count / self.rate if self.rate else 0.0
            if not 0 <= t <= duration:
                raise MatlabError(f"CurrentTime must be between 0 and the duration of the video ({duration:g} s).")
            self.next = min(int(round(t * self.rate)), self.count)
        elif key == "tag":
            self.tag = char(text(value, "Tag"))
        elif key == "userdata":
            self.user_data = value
        else:
            raise MatlabError(f"Unable to set the '{name}' property of class 'VideoReader': it is read-only or "
                              "does not exist.")
        return self

    def call_method(self, interp, name: str, args: list, nargout: int) -> list:
        rest = [a for a in args if a is not self]
        if name in ("read", "readFrame") and rest and is_text(rest[-1]):
            # frames are uint8 already, which is also what 'native' asks for
            if text(rest.pop()).lower() not in ("native", "default"):
                raise MatlabError(f"{name}: the only option is 'native'.")
        if name == "hasFrame":
            return [bool_scalar(self.next < self.count)]
        if name == "readFrame":
            if rest:
                raise MatlabError("readFrame takes the VideoReader and, optionally, 'native'.")
            if self.next >= self.count:
                raise MatlabError("No more frames available to read from file.",
                                  "MATLAB:audiovideo:VideoReader:EndOfFile")
            f = self.frame(self.next)
            self.next += 1
            return [f]
        if name == "read":
            if not rest:
                a, b = 1.0, float(self.count)
            else:
                idx = [float(x) for x in np.asarray(rest[0], dtype=float).flatten()]
                if len(idx) == 1:
                    a = b = idx[0]
                elif len(idx) == 2:
                    a, b = idx
                else:
                    raise MatlabError("read(v, index): the index is one frame number or [first last].")
                if np.isinf(b):
                    b = float(self.count)
            if a != int(a) or b != int(b) or a < 1 or b < a or b > self.count:
                raise MatlabError(f"read: the frame index must lie between 1 and {self.count} (the number of "
                                  "frames).", "MATLAB:audiovideo:VideoReader:invalidFrame")
            self.next = int(b)
            return [self.frames(int(a) - 1, int(b) - 1)]
        if name == "get":
            if rest:
                return [self.get_field(interp, text(rest[0], "property name"))]
            return [Struct.scalar({k: self.get_field(interp, k) for k in self._PROPS})]
        if name == "getFileFormats":
            return [Cell.from_list([char("avi")])]
        return super().call_method(interp, name, args, nargout)

    def display_lines(self) -> list[str]:
        duration = self.count / self.rate if self.rate else 0.0
        return ["  VideoReader with properties:", "", "   General Properties:",
                f"            Name: '{os.path.basename(self.path)}'",
                f"            Path: '{os.path.dirname(self.path)}'",
                f"        Duration: {duration:g}", f"     CurrentTime: {self.next / self.rate if self.rate else 0:g}",
                f"       NumFrames: {self.count}", "", "   Video Properties:",
                f"           Width: {self.width}", f"          Height: {self.height}",
                f"       FrameRate: {self.rate:g}", f"    BitsPerPixel: {8 if self.gray else 24}",
                f"     VideoFormat: '{'Grayscale' if self.gray else 'RGB24'}'"]


@builtin("VideoReader")
def _videoreader(interp, args, nargout):
    if not args or not is_text(args[0]):
        raise MatlabError("VideoReader needs a file name, for example v = VideoReader('movie.avi').")
    name = text(args[0])
    path = find_file(interp, name)
    if path is None:
        raise MatlabError(f"VideoReader: the file '{name}' was not found.",
                          "MATLAB:audiovideo:VideoReader:FileNotFound")
    v = VideoReaderObject(os.path.abspath(path))
    rest = list(args[1:])
    if len(rest) % 2:
        raise MatlabError("VideoReader: options must be name-value pairs, for example 'CurrentTime', 1.5.")
    for k in range(0, len(rest), 2):
        key = text(rest[k], "option name")
        if key.lower() not in ("currenttime", "tag", "userdata"):
            raise MatlabError(f"VideoReader: unknown option '{key}'. Options are CurrentTime, Tag and UserData.")
        v.set_field(interp, key, rest[k + 1])
    return v


@builtin("readFrame")
def _read_frame(interp, args, nargout):
    raise MatlabError("readFrame expects a VideoReader object as its first input.")


@builtin("hasFrame")
def _has_frame(interp, args, nargout):
    raise MatlabError("hasFrame expects a VideoReader object as its first input.")


@builtin("mmfileinfo")
def _mmfileinfo(interp, args, nargout):
    if not args or not is_text(args[0]):
        raise MatlabError("mmfileinfo needs a file name.")
    path = find_file(interp, text(args[0]))
    if path is None:
        raise MatlabError(f"mmfileinfo: the file '{text(args[0])}' was not found.")
    v = VideoReaderObject(os.path.abspath(path))
    video = Struct.scalar({"Format": char("Grayscale" if v.gray else "RGB24"), "Height": float(v.height),
                           "Width": float(v.width)})
    audio = Struct.scalar({"Format": char(""), "NumChannels": np.zeros((0, 0))})
    return Struct.scalar({"Filename": char(os.path.basename(path)), "Path": char(os.path.dirname(v.path)),
                          "Duration": float(v.count / v.rate if v.rate else 0.0), "Audio": audio, "Video": video})
