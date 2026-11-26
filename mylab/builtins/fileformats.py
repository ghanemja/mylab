"""Data files: MAT-files (save, load, matfile), HDF5, NetCDF, XML, struct files and memory maps.

The MAT container code lives in ``_matio`` (Level 5 files), ``_mat73`` (HDF5-based version 7.3
files, optional h5py) and ``_matcodec`` (how values without a standard MAT type are stored).
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
            if self._mat5.damage and not self._mat5.variables():
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
        write_mat(interp, self.path, {name: value}, append=True)
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
            rows = [whos_row(n, self.value(interp, n)) for n in names]
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
                dims = tuple(shape_of(self.value(interp, text(rest[0], "variable name"))))
                if len(rest) > 1:
                    d = int(np.asarray(rest[1]).flat[0])
                    return [np.array([[float(dims[d - 1] if d <= len(dims) else 1)]])]
            if nargout <= 1:
                return [np.array([[float(d) for d in dims]])]
            vals = [float(d) for d in dims[:nargout - 1]] + [float(np.prod(dims[nargout - 1:]))]
            vals += [1.0] * (nargout - len(vals))
            return [np.array([[x]]) for x in vals]
        return super().call_method(interp, name, args, nargout)

    def display_lines(self) -> list[str]:
        from ..interpreter import current_interpreter
        interp = current_interpreter()
        lines = ["  matlab.io.MatFile", "", "  Properties:", f"      Properties.Source: '{self.path}'",
                 f"    Properties.Writable: {'true' if self.writable else 'false'}"]
        if interp is not None:
            try:
                for n in sorted(self.names(interp)):
                    v = self.value(interp, n)
                    size = "x".join(str(d) for d in shape_of(v))
                    lines.append(f"    {n:>19}: [{size} {class_name(v)}]")
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
