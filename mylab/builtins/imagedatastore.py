"""imageDatastore, and the transform and combine wrappers that work on any datastore.

An image datastore is a list of image files (with optional labels) that ``read`` walks through one
file, or ``ReadSize`` files, at a time. It is a handle: copies of the variable share the read
position. Labels come from the names of the folders that hold the files ('LabelSource',
'foldernames') or are given directly, and are kept as a column vector, usually categorical.
``transform(ds, fcn)`` applies a function to every read of a datastore and ``combine(ds1, ds2)``
reads several datastores side by side; both accept the tabular text datastore as well.
"""

from __future__ import annotations

import copy as _copy
import os
from typing import Any

import numpy as np

from ..errors import MatlabError
from ..objects import MObject
from ..values import Cell, FunctionHandle, MString, StringArray, Struct, char, to_str
from ._registry import BUILTINS, builtin
from ._util import is_text, match_option, parse_options, text
from .categorical import CategoricalArray, _value_keys, make_categorical

_NO_DATA = "MATLAB:datastoreio:splittabledatastore:noMoreData"


def _image_extensions() -> tuple[str, ...]:
    from .images import _FORMATS
    return tuple("." + k for k in _FORMATS)


def _scalar(x: float) -> np.ndarray:
    return np.array([[float(x)]])


def _flag(v: Any, what: str) -> bool:
    if is_text(v):
        s = text(v).lower()
        if s in ("on", "true"):
            return True
        if s in ("off", "false"):
            return False
        raise MatlabError(f"{what} must be true or false.")
    a = np.asarray(v)
    if a.size != 1 or a.dtype.kind not in "biuf":
        raise MatlabError(f"{what} must be true or false.")
    return bool(a.flat[0])


def _positive_int(v: Any, what: str) -> int:
    a = np.asarray(v) if not is_text(v) else None
    if a is None or a.size != 1 or a.dtype.kind not in "biuf" or not np.isfinite(a.flat[0]) \
            or a.flat[0] != int(a.flat[0]) or a.flat[0] < 1:
        raise MatlabError(f"{what} must be a positive integer.")
    return int(a.flat[0])


def _items(v: Any, what: str) -> list[str]:
    if isinstance(v, Cell):
        if v.data.size == 0:
            return []
        return [to_str(x, what) for x in v.values_f()]
    if isinstance(v, StringArray):
        return ["" if x is None else x for x in v.data.flatten(order="F")]
    if is_text(v):
        return [text(v)]
    raise MatlabError(f"{what} must be text, a cell array of text or a string array.")


# -- labels: a column vector (categorical, cell of text, string, numeric or logical) or None ----------

def _as_labels(v: Any, n: int) -> Any:
    if isinstance(v, np.ndarray) and v.size == 0 and v.dtype.kind != "U":
        if n:
            raise MatlabError(f"Labels must have one element per file ({n}).")
        return None
    if isinstance(v, Cell) and v.data.size == 0 and n == 0:
        return None
    if isinstance(v, CategoricalArray):
        out = v.like(v.data.reshape(-1, 1, order="F"))
    elif isinstance(v, Cell):
        if not all(is_text(x) for x in v.data.flat):
            raise MatlabError("Labels given as a cell array must hold text.")
        out = Cell.from_list([char(to_str(x)) for x in v.values_f()], column=True)
    elif isinstance(v, StringArray):
        out = StringArray(v.data.reshape(-1, 1, order="F"))
    elif isinstance(v, MString):
        out = StringArray(np.array([str(v)], dtype=object).reshape(1, 1))
    elif is_text(v):
        out = Cell.from_list([char(text(v))], column=True)
    elif isinstance(v, np.ndarray) and v.dtype.kind in "biuf":
        out = v.reshape(-1, 1, order="F").copy()
    else:
        raise MatlabError("Labels must be a categorical, numeric, logical or text vector.")
    if _count(out) != n:
        raise MatlabError(f"Labels must have one element per file: {n} files, {_count(out)} labels.")
    return out


def _count(labels: Any) -> int:
    if labels is None:
        return 0
    return int(labels.data.size if isinstance(labels, (CategoricalArray, Cell, StringArray)) else labels.size)


def _take(labels: Any, idx) -> Any:
    if labels is None:
        return None
    idx = np.asarray(idx, dtype=int).reshape(-1)
    if isinstance(labels, CategoricalArray):
        return labels.like(labels.data.reshape(-1)[idx].reshape(-1, 1))
    if isinstance(labels, Cell):
        return Cell(labels.data.reshape(-1)[idx].reshape(-1, 1))
    if isinstance(labels, StringArray):
        return StringArray(labels.data.reshape(-1)[idx].reshape(-1, 1))
    return labels.reshape(-1)[idx].reshape(-1, 1)


def _labels_value(labels: Any) -> Any:
    return Cell.empty(0, 0) if labels is None else labels


def _as_categorical(labels: Any) -> CategoricalArray:
    return labels if isinstance(labels, CategoricalArray) else make_categorical([labels])


# -- shared plumbing for every datastore ----------------------------------------------------------------

def _read(interp, ds: MObject, nargout: int = 2) -> list:
    out = ds.call_method(interp, "read", [ds], nargout)
    return list(out) + [Struct.scalar({})] * (nargout - len(out))


def _hasdata(interp, ds: MObject) -> bool:
    return bool(np.asarray(ds.call_method(interp, "hasdata", [ds], 1)[0]).all())


def _clone(ds: MObject) -> MObject:
    return ds.clone() if hasattr(ds, "clone") else _copy.copy(ds)


def _fresh(interp, ds: MObject) -> MObject:
    """An independent copy of a datastore, back at its start (what transform and combine wrap)."""
    other = _clone(ds)
    other.call_method(interp, "reset", [other], 0)
    return other


def _stack_reads(parts: list) -> Any:
    """readall of a wrapped datastore: tables and cell arrays are stacked, anything else goes in a cell."""
    from .table import TableObject
    if not parts:
        return Cell.empty(0, 1)
    if all(isinstance(p, TableObject) for p in parts):
        return parts[0] if len(parts) == 1 else TableObject.concat(parts, 0)
    if all(isinstance(p, Cell) and p.data.ndim == 2 for p in parts) and \
            len({p.data.shape[1] for p in parts}) == 1:
        return Cell(np.concatenate([p.data for p in parts], axis=0))
    return Cell.from_list(parts, column=True)


def _require_ds(v: Any, fname: str) -> MObject:
    if not (isinstance(v, MObject) and v.isa_class("matlab.io.Datastore")):
        raise MatlabError(f"{fname}: expected a datastore.")
    return v


class _DatastoreBase(MObject):
    is_handle = True
    _PROPS: tuple[str, ...] = ()
    _METHODS: tuple[str, ...] = ()

    def isa_class(self, name: str) -> bool:
        return name in (self.class_name(), "matlab.io.Datastore", "matlab.io.datastore.Partitionable",
                        "matlab.io.datastore.Shuffleable", "handle")

    def property_names(self) -> list[str]:
        return list(self._PROPS)

    def method_names(self) -> list[str]:
        return sorted(self._METHODS, key=str.lower)

    def has_method(self, name: str) -> bool:
        return name in self._METHODS

    def clone(self) -> "_DatastoreBase":
        return _copy.copy(self)

    def _common(self, interp, name: str, rest: list, nargout: int) -> list | None:
        """Methods every datastore here shares; None when ``name`` is not one of them."""
        if name == "transform":
            return [_make_transform(interp, self, rest)]
        if name == "combine":
            return [CombinedDatastore([_fresh(interp, d) for d in [self] + [_require_ds(r, "combine") for r in rest]])]
        if name in ("isPartitionable", "isShuffleable"):
            return [np.array([[True]])]
        if name == "copy":
            return [self.clone()]
        if name == "preview":
            other = self.clone()
            other.call_method(interp, "reset", [other], 0)
            if not _hasdata(interp, other):
                return [np.zeros((0, 0))]
            return [_read(interp, other, 1)[0]]
        return None


# -- imageDatastore ----------------------------------------------------------------------------------------

class ImageDatastore(_DatastoreBase):
    """A list of image files read in order, with optional labels."""

    _PROPS = ("Files", "Folders", "Labels", "ReadSize", "ReadFcn", "AlternateFileSystemRoots",
              "SupportedOutputFormats", "DefaultOutputFormat")
    _METHODS = ("read", "readimage", "readall", "hasdata", "reset", "preview", "progress", "countEachLabel",
                "splitEachLabel", "shuffle", "subset", "partition", "numpartitions", "transform", "combine",
                "copy", "isPartitionable", "isShuffleable")

    def __init__(self, files: list[str], labels: Any = None, read_fcn: FunctionHandle | None = None,
                 read_size: int = 1, folders: list[str] | None = None):
        self.files = [os.path.abspath(f) for f in files]
        self.labels = labels
        self.read_fcn = read_fcn
        self.read_size = read_size
        self.folders = folders if folders is not None else _parent_folders(self.files)
        self.pos = 0

    def class_name(self) -> str:
        return "matlab.io.datastore.ImageDatastore"

    def _derive(self, idx) -> "ImageDatastore":
        idx = [int(i) for i in idx]
        return ImageDatastore([self.files[i] for i in idx], _take(self.labels, idx) if idx else
                              (None if self.labels is None else _take(self.labels, [])),
                              self.read_fcn, self.read_size, list(self.folders))

    # reading ---------------------------------------------------------------------------------------------
    def _image(self, interp, k: int) -> Any:
        path = char(self.files[k])
        if self.read_fcn is None:
            return BUILTINS["imread"](interp, [path], 1)
        out = interp.call_handle(self.read_fcn, [path], 1)
        if not out:
            raise MatlabError("The ReadFcn of the datastore must return a value.")
        return out[0]

    def _info(self, idx: list[int], block: bool = False) -> Struct:
        if not block:
            k = idx[0]
            return Struct.scalar({"Filename": char(self.files[k]),
                                  "FileSize": _scalar(os.path.getsize(self.files[k])),
                                  "Label": _labels_value(_take(self.labels, [k]))})
        return Struct.scalar({
            "Filename": Cell.from_list([char(self.files[k]) for k in idx], column=True),
            "FileSize": np.array([float(os.path.getsize(self.files[k])) for k in idx]).reshape(-1, 1),
            "Label": _labels_value(_take(self.labels, idx))})

    def _check_index(self, v: Any) -> int:
        k = _positive_int(v, "readimage: the image index")
        if k > len(self.files):
            raise MatlabError(f"readimage: the index {k} exceeds the number of files ({len(self.files)}).",
                              "MATLAB:datastoreio:imagedatastore:invalidIndex")
        return k - 1

    def read(self, interp) -> list:
        if self.pos >= len(self.files):
            raise MatlabError("No more data to read. Use reset to read from the start, or check hasdata first.",
                              _NO_DATA)
        idx = list(range(self.pos, min(self.pos + self.read_size, len(self.files))))
        self.pos = idx[-1] + 1
        if self.read_size == 1:
            return [self._image(interp, idx[0]), self._info(idx)]
        return [Cell.from_list([self._image(interp, k) for k in idx], column=True), self._info(idx, True)]

    # labels ------------------------------------------------------------------------------------------------
    def count_each_label(self):
        from .table import TableObject
        if self.labels is None:
            raise MatlabError("countEachLabel: the datastore has no labels. Create it with 'LabelSource', "
                              "'foldernames' or set its Labels property.")
        cat = _as_categorical(self.labels)
        codes = cat.codes.reshape(-1)
        counts = np.array([float(np.sum(codes == i)) for i in range(len(cat.categories))]).reshape(-1, 1)
        label = CategoricalArray(np.arange(1.0, len(cat.categories) + 1).reshape(-1, 1), cat.categories,
                                 cat.ordinal, cat.protected)
        return TableObject(["Label", "Count"], [label, counts])

    def split_each_label(self, interp, args: list, nargout: int) -> list:
        if self.labels is None:
            raise MatlabError("splitEachLabel: the datastore has no labels. Create it with 'LabelSource', "
                              "'foldernames' or set its Labels property.")
        sizes: list[float] = []
        k = 0
        while k < len(args) and not is_text(args[k]):
            a = np.asarray(args[k])
            if a.size != 1 or a.dtype.kind not in "biuf" or not np.isfinite(a.flat[0]) or a.flat[0] <= 0:
                raise MatlabError("splitEachLabel: each split must be a proportion between 0 and 1 or a "
                                  "positive number of files.")
            sizes.append(float(a.flat[0]))
            k += 1
        if not sizes:
            raise MatlabError("splitEachLabel needs the proportion or number of files for the first part, "
                              "for example splitEachLabel(ds, 0.7).")
        randomized = False
        rest = list(args[k:])
        if rest and is_text(rest[0]) and match_option(text(rest[0]), ["randomized"]) is not None \
                and len(text(rest[0])) >= 6:
            randomized = True
            rest = rest[1:]
        opts = parse_options(rest, ("Include", "Exclude"), "splitEachLabel")
        proportions = all(s < 1 for s in sizes)
        if not proportions and any(s < 1 or s != int(s) for s in sizes):
            raise MatlabError("splitEachLabel: give either proportions (all below 1) or numbers of files "
                              "(whole numbers), not a mix.")
        if proportions and sum(sizes) > 1 + 1e-12:
            raise MatlabError("splitEachLabel: the proportions must add up to 1 or less.")
        keys, _, _ = _value_keys(self.labels, "splitEachLabel")
        order: list = []
        groups: dict = {}
        for i, key in enumerate(keys):
            if key is None:
                continue
            if key not in groups:
                groups[key] = []
                order.append(key)
            groups[key].append(i)
        if isinstance(self.labels, CategoricalArray):
            order = [c for c in self.labels.categories if c in groups]
        else:
            order = sorted(order, key=lambda x: (str(type(x)), x))
        if "Include" in opts:
            want = _label_values(opts["Include"])
            order = [key for key in order if key in want]
        if "Exclude" in opts:
            drop = _label_values(opts["Exclude"])
            order = [key for key in order if key not in drop]
        parts: list[list[int]] = [[] for _ in range(len(sizes) + 1)]
        from .core import _rng
        for key in order:
            members = groups[key]
            n = len(members)
            if randomized:
                members = [members[i] for i in _rng(interp).permutation(n)]
            if proportions:
                bounds = [int(np.floor(c * n + 0.5)) for c in np.cumsum(sizes)]
            else:
                if sum(sizes) > n:
                    raise MatlabError(f"splitEachLabel: the label '{key}' has {n} files, fewer than the "
                                      f"{int(sum(sizes))} requested.")
                bounds = [int(c) for c in np.cumsum(sizes)]
            start = 0
            for j, b in enumerate(bounds + [n]):
                b = min(max(b, start), n)
                parts[j].extend(members[start:b])
                start = b
        if not randomized:
            parts = [sorted(p) for p in parts]
        return [self._derive(p) for p in parts[:max(nargout, 1)]]

    # methods -------------------------------------------------------------------------------------------------
    def call_method(self, interp, name: str, args: list, nargout: int) -> list:
        if name not in self._METHODS:
            return super().call_method(interp, name, args, nargout)
        if not args or args[0] is not self:
            if name == "combine":
                return [CombinedDatastore([_fresh(interp, _require_ds(a, "combine")) for a in args])]
            raise MatlabError(f"{name}: the first argument must be the datastore.")
        rest = list(args[1:])
        if name == "read":
            if rest:
                raise MatlabError("read takes only the datastore.", "MATLAB:TooManyInputs")
            return self.read(interp)[:max(nargout, 1)]
        if name == "readimage":
            if len(rest) != 1:
                raise MatlabError("readimage needs the datastore and the index of the file, readimage(ds, k).")
            k = self._check_index(rest[0])
            return [self._image(interp, k), self._info([k])][:max(nargout, 1)]
        if name == "readall":
            parse_options(rest, ("UseParallel",), "readall")
            return [Cell.from_list([self._image(interp, k) for k in range(len(self.files))], column=True)]
        if name == "hasdata":
            return [np.array([[self.pos < len(self.files)]])]
        if name == "reset":
            self.pos = 0
            return []
        if name == "progress":
            return [_scalar(self.pos / len(self.files) if self.files else 1.0)]
        if name == "countEachLabel":
            return [self.count_each_label()]
        if name == "splitEachLabel":
            return self.split_each_label(interp, rest, nargout)
        if name == "shuffle":
            from .core import _rng
            return [self._derive(_rng(interp).permutation(len(self.files)))]
        if name == "subset":
            if len(rest) != 1:
                raise MatlabError("subset needs the datastore and the indices of the files, subset(ds, idx).")
            return [self._derive(_indices(rest[0], len(self.files), "subset"))]
        if name == "numpartitions":
            return [_scalar(len(self.files))]
        if name == "preview":
            return [self._image(interp, 0) if self.files else np.zeros((0, 0))]
        if name == "partition":
            return [self._derive(_partition_indices(rest, self.files))]
        out = self._common(interp, name, rest, nargout)
        if out is None:
            raise MatlabError(f"The datastore function '{name}' is not supported.")
        return out

    # properties ---------------------------------------------------------------------------------------------
    def get_field(self, interp, name: str) -> Any:
        if name == "Files":
            return Cell.from_list([char(f) for f in self.files], column=True) if self.files else Cell.empty(0, 1)
        if name == "Folders":
            return Cell.from_list([char(f) for f in self.folders], column=True) if self.folders \
                else Cell.empty(0, 1)
        if name == "Labels":
            return _labels_value(self.labels)
        if name == "ReadSize":
            return _scalar(self.read_size)
        if name == "ReadFcn":
            return self.read_fcn if self.read_fcn is not None else FunctionHandle("imread")
        if name == "AlternateFileSystemRoots":
            return Cell.empty(0, 0)
        if name == "SupportedOutputFormats":
            fmts = ["png", "jpg", "jpeg", "tif", "tiff"]
            return StringArray(np.array(fmts, dtype=object).reshape(1, -1))
        if name == "DefaultOutputFormat":
            return MString("png")
        if self.has_method(name):
            return self.call_method(interp, name, [self], 1)[0]
        raise MatlabError(f"Unrecognized method, property, or field '{name}' for class '{self.class_name()}'.",
                          "MATLAB:noSuchMethodOrField")

    def set_field(self, interp, name: str, value: Any) -> "ImageDatastore":
        if name == "Labels":
            self.labels = _as_labels(value, len(self.files))
        elif name == "ReadSize":
            self.read_size = _positive_int(value, "ReadSize")
        elif name == "ReadFcn":
            if not isinstance(value, FunctionHandle):
                raise MatlabError("ReadFcn must be a function handle that takes a file name and returns the "
                                  "data, for example @(f) imresize(imread(f), [64 64]).")
            self.read_fcn = value
        elif name == "Files":
            files = [os.path.abspath(f) for f in _items(value, "Files")]
            old = {f: i for i, f in enumerate(self.files)}
            if self.labels is not None and all(f in old for f in files):
                self.labels = _take(self.labels, [old[f] for f in files])  # files kept keep their labels
            else:
                self.labels = None
            self.files = files
            self.pos = 0
        elif name in self._PROPS:
            raise MatlabError(f"Unable to set the '{name}' property of class 'ImageDatastore' because it is "
                              "read-only.", "MATLAB:class:SetProhibited")
        else:
            raise MatlabError(f"Unrecognized property '{name}' for class '{self.class_name()}'.",
                              "MATLAB:noPublicFieldForClass")
        return self

    def display_lines(self) -> list[str]:
        def listing(items: list[str]) -> list[str]:
            if not items:
                return ["{}"]
            shown = [f"'{f}'" for f in items[:3]]
            more = [f" ... and {len(items) - 3} more"] if len(items) > 3 else []
            return ["{"] + [s + (";" if i < len(shown) - 1 else "") for i, s in enumerate(shown)] + more + ["}"]

        pad = " " * 30
        lines = ["  ImageDatastore with properties:", ""]
        for title, items in (("Files", self.files), ("Folders", self.folders)):
            block = listing(items)
            lines.append(f"{title:>28}: {block[0]}")
            lines += [pad + b for b in block[1:]]
        if self.labels is None:
            lines.append(f"{'Labels':>28}: {{}}")
        else:
            names = [str(k) for k in _value_keys(self.labels, "Labels")[0]]
            more = f" ... and {len(names) - 3} more" if len(names) > 3 else ""
            lines.append(f"{'Labels':>28}: [{'; '.join(names[:3])}{more} {type_name(self.labels)}]")
        lines += [f"{'AlternateFileSystemRoots':>28}: {{}}",
                  f"{'ReadSize':>28}: {self.read_size}",
                  f"{'SupportedOutputFormats':>28}: [\"png\"    \"jpg\"    \"jpeg\"    \"tif\"    \"tiff\"]",
                  f"{'DefaultOutputFormat':>28}: \"png\"",
                  f"{'ReadFcn':>28}: {self.read_fcn!r}" if self.read_fcn is not None else
                  f"{'ReadFcn':>28}: @imread"]
        return lines


def type_name(v: Any) -> str:
    from ..values import class_name
    return class_name(v)


def _label_values(v: Any) -> set:
    keys, _, _ = _value_keys(v, "splitEachLabel")
    return {k for k in keys if k is not None}


def _indices(v: Any, n: int, fname: str) -> list[int]:
    a = np.asarray(v)
    if a.dtype == bool:
        if a.size != n:
            raise MatlabError(f"{fname}: a logical index must have one element per file ({n}).")
        return [int(i) for i in np.flatnonzero(a.reshape(-1, order="F"))]
    if a.dtype.kind not in "iuf":
        raise MatlabError(f"{fname}: the indices must be numbers or a logical vector.")
    flat = a.reshape(-1, order="F")
    if flat.size and (np.any(flat != np.round(flat)) or flat.min() < 1 or flat.max() > n):
        raise MatlabError(f"{fname}: the indices must be whole numbers from 1 to {n}.",
                          "MATLAB:datastoreio:imagedatastore:invalidIndex")
    return [int(i) - 1 for i in flat]


def _partition_indices(rest: list, files: list[str]) -> list[int]:
    if len(rest) != 2:
        raise MatlabError("partition needs the datastore, the number of parts and the part to return, "
                          "partition(ds, n, k), or partition(ds, 'Files', k).")
    if is_text(rest[0]):
        if text(rest[0]).lower() != "files":
            raise MatlabError(f"partition: the strategy '{text(rest[0])}' is not supported; use a number of "
                              "parts or 'Files'.")
        if is_text(rest[1]):
            want = os.path.abspath(text(rest[1]))
            if want not in files:
                raise MatlabError(f"partition: the datastore has no file '{text(rest[1])}'.")
            return [files.index(want)]
        k = _positive_int(rest[1], "partition: the file index")
        if k > len(files):
            raise MatlabError(f"partition: the file index must be at most {len(files)}.")
        return [k - 1]
    n = _positive_int(rest[0], "partition: the number of parts")
    k = _positive_int(rest[1], "partition: the part index")
    if k > n:
        raise MatlabError("partition: the part index must be between 1 and the number of parts.")
    edges = np.floor(np.arange(n + 1) * len(files) / n + 1e-9).astype(int)
    return list(range(edges[k - 1], edges[k]))


def _parent_folders(files: list[str]) -> list[str]:
    seen: list[str] = []
    for f in files:
        d = os.path.dirname(f)
        if d not in seen:
            seen.append(d)
    return seen


_IDS_OPTIONS = ("IncludeSubfolders", "FileExtensions", "LabelSource", "Labels", "ReadFcn", "ReadSize",
                "AlternateFileSystemRoots")


def make_image_datastore(interp, args: list, fname: str = "imageDatastore") -> ImageDatastore:
    from ._textimport import datastore_files
    if not args:
        raise MatlabError(f"{fname} needs a folder, a file, a pattern with * or a cell array of these, for "
                          f"example {fname}('photos', 'IncludeSubfolders', true).")
    opts = parse_options(list(args[1:]), _IDS_OPTIONS, fname, first_arg=2)
    location = args[0]
    sub = _flag(opts["IncludeSubfolders"], "IncludeSubfolders") if "IncludeSubfolders" in opts else False
    if "FileExtensions" in opts:
        exts = tuple(("." + e.lstrip(".")).lower() for e in _items(opts["FileExtensions"], "FileExtensions"))
    else:
        exts = _image_extensions()
    items = _items(location, f"{fname}: the location")
    files: list[str] = []
    folders: list[str] = []
    for loc in items:
        found = datastore_files(interp, char(loc), None, sub)
        files += [f for f in found if os.path.splitext(f)[1].lower() in exts]
        path = os.path.abspath(os.path.expanduser(loc))
        if os.path.isdir(path):
            folders.append(path)
        else:
            folders += [d for d in _parent_folders([os.path.abspath(f) for f in found])]
    if items and not files:
        raise MatlabError(f"{fname}: no files with the extensions {', '.join(exts)} were found at the location.",
                          "MATLAB:datastoreio:pathlookup:fileNotFound")
    folders = list(dict.fromkeys(folders))
    labels = None
    source = "none"
    if "LabelSource" in opts:
        source = text(opts["LabelSource"], "LabelSource").lower()
        if source not in ("none", "foldernames"):
            raise MatlabError(f"{fname}: 'LabelSource' must be 'none' or 'foldernames', not '{source}'.")
    if source == "foldernames":
        if "Labels" in opts:
            raise MatlabError(f"{fname}: give 'LabelSource' or 'Labels', not both.")
        names = [os.path.basename(os.path.dirname(os.path.abspath(f))) for f in files]
        labels = make_categorical([Cell.from_list([char(n) for n in names], column=True)])
    elif "Labels" in opts:
        labels = _as_labels(opts["Labels"], len(files))
    read_fcn = None
    if "ReadFcn" in opts:
        if not isinstance(opts["ReadFcn"], FunctionHandle):
            raise MatlabError(f"{fname}: 'ReadFcn' must be a function handle.")
        read_fcn = opts["ReadFcn"]
    size = _positive_int(opts["ReadSize"], "ReadSize") if "ReadSize" in opts else 1
    return ImageDatastore(files, labels, read_fcn, size, folders)


@builtin("imageDatastore")
def _image_datastore(interp, args, nargout):
    return make_image_datastore(interp, list(args))


# -- transform and combine ----------------------------------------------------------------------------------

class TransformedDatastore(_DatastoreBase):
    """Every read of the underlying datastore passed through a function."""

    _PROPS = ("UnderlyingDatastores", "Transforms", "IncludeInfo", "SupportedOutputFormats")
    _METHODS = ("read", "readall", "hasdata", "reset", "preview", "transform", "combine", "copy", "shuffle",
                "subset", "partition", "numpartitions", "isPartitionable", "isShuffleable")

    def __init__(self, under: MObject, fcn: FunctionHandle, include_info: bool):
        self.under = under
        self.fcn = fcn
        self.include_info = include_info

    def class_name(self) -> str:
        return "matlab.io.datastore.TransformedDatastore"

    def clone(self) -> "TransformedDatastore":
        return TransformedDatastore(_clone(self.under), self.fcn, self.include_info)

    def read(self, interp) -> list:
        data, info = _read(interp, self.under, 2)
        if self.include_info:
            out = interp.call_handle(self.fcn, [data, info], 2)
            if len(out) < 2:
                raise MatlabError("transform: with 'IncludeInfo' true the function must return the data and "
                                  "the info.")
            return [out[0], out[1]]
        out = interp.call_handle(self.fcn, [data], 1)
        if not out:
            raise MatlabError("transform: the function must return a value.")
        return [out[0], info]

    def _wrap(self, under: MObject) -> "TransformedDatastore":
        return TransformedDatastore(under, self.fcn, self.include_info)

    def call_method(self, interp, name: str, args: list, nargout: int) -> list:
        if name not in self._METHODS:
            return super().call_method(interp, name, args, nargout)
        if not args or args[0] is not self:
            if name == "combine":
                return [CombinedDatastore([_fresh(interp, _require_ds(a, "combine")) for a in args])]
            raise MatlabError(f"{name}: the first argument must be the datastore.")
        rest = list(args[1:])
        if name == "read":
            return self.read(interp)[:max(nargout, 1)]
        if name == "hasdata":
            return [np.array([[_hasdata(interp, self.under)]])]
        if name == "reset":
            self.under.call_method(interp, "reset", [self.under], 0)
            return []
        if name == "readall":
            parse_options(rest, ("UseParallel",), "readall")
            other = self.clone()
            other.under.call_method(interp, "reset", [other.under], 0)
            parts = []
            while _hasdata(interp, other.under):
                parts.append(other.read(interp)[0])
            return [_stack_reads(parts)]
        if name in ("shuffle", "subset", "partition", "numpartitions"):
            if not self.under.has_method(name):
                raise MatlabError(f"{name}: the underlying datastore does not support it.")
            out = self.under.call_method(interp, name, [self.under] + rest, 1)
            return out if name == "numpartitions" else [self._wrap(out[0])]
        out = self._common(interp, name, rest, nargout)
        if out is None:
            raise MatlabError(f"The datastore function '{name}' is not supported.")
        return out

    def get_field(self, interp, name: str) -> Any:
        if name == "UnderlyingDatastores":
            return Cell.from_list([self.under])
        if name == "Transforms":
            return Cell.from_list([self.fcn])
        if name == "IncludeInfo":
            return np.array([[self.include_info]])
        if name == "SupportedOutputFormats":
            return StringArray(np.array(["txt", "csv", "xlsx", "xls", "parquet", "parq", "png", "jpg", "jpeg",
                                         "tif", "tiff", "wav", "flac", "ogg", "mp4", "m4a"],
                                        dtype=object).reshape(1, -1))
        if self.has_method(name):
            return self.call_method(interp, name, [self], 1)[0]
        raise MatlabError(f"Unrecognized method, property, or field '{name}' for class '{self.class_name()}'.",
                          "MATLAB:noSuchMethodOrField")

    def display_lines(self) -> list[str]:
        return ["  TransformedDatastore with properties:", "",
                f"    UnderlyingDatastores: {{{type(self.under).__name__}}}",
                f"              Transforms: {{{self.fcn!r}}}",
                f"             IncludeInfo: {int(self.include_info)}"]


def _make_transform(interp, ds: MObject, rest: list) -> TransformedDatastore:
    if not rest or not isinstance(rest[0], FunctionHandle):
        raise MatlabError("transform needs a datastore and a function handle, transform(ds, @fcn).")
    opts = parse_options(list(rest[1:]), ("IncludeInfo",), "transform", first_arg=3)
    include = _flag(opts["IncludeInfo"], "IncludeInfo") if "IncludeInfo" in opts else False
    return TransformedDatastore(_fresh(interp, ds), rest[0], include)


class CombinedDatastore(_DatastoreBase):
    """Several datastores read side by side: each read returns one cell column per datastore."""

    _PROPS = ("UnderlyingDatastores", "SupportedOutputFormats")
    _METHODS = ("read", "readall", "hasdata", "reset", "preview", "transform", "combine", "copy", "shuffle",
                "subset", "partition", "numpartitions", "isPartitionable", "isShuffleable")

    def __init__(self, unders: list[MObject]):
        if not unders:
            raise MatlabError("combine needs at least one datastore.")
        self.unders = [_clone(u) for u in unders]

    def class_name(self) -> str:
        return "matlab.io.datastore.CombinedDatastore"

    def clone(self) -> "CombinedDatastore":
        return CombinedDatastore(self.unders)

    def read(self, interp) -> list:
        blocks, infos = [], []
        for u in self.unders:
            data, info = _read(interp, u, 2)
            blocks.append(data.data if isinstance(data, Cell) else Cell.from_list([data]).data)
            infos.append(info)
        if len({b.shape[0] for b in blocks}) != 1:
            raise MatlabError("combine: the datastores returned different numbers of rows in one read.")
        return [Cell(np.concatenate(blocks, axis=1)), Cell.from_list(infos)]

    def call_method(self, interp, name: str, args: list, nargout: int) -> list:
        if name not in self._METHODS:
            return super().call_method(interp, name, args, nargout)
        if not args or args[0] is not self:
            if name == "combine":
                return [CombinedDatastore([_fresh(interp, _require_ds(a, "combine")) for a in args])]
            raise MatlabError(f"{name}: the first argument must be the datastore.")
        rest = list(args[1:])
        if name == "read":
            return self.read(interp)[:max(nargout, 1)]
        if name == "hasdata":
            return [np.array([[all(_hasdata(interp, u) for u in self.unders)]])]
        if name == "reset":
            for u in self.unders:
                u.call_method(interp, "reset", [u], 0)
            return []
        if name == "readall":
            parse_options(rest, ("UseParallel",), "readall")
            other = self.clone()
            other.call_method(interp, "reset", [other], 0)
            parts = []
            while all(_hasdata(interp, u) for u in other.unders):
                parts.append(other.read(interp)[0])
            return [_stack_reads(parts)]
        if name == "numpartitions":
            counts = [float(np.asarray(u.call_method(interp, name, [u], 1)[0]).flat[0]) for u in self.unders]
            return [_scalar(min(counts))]
        if name in ("subset", "partition"):
            return [CombinedDatastore([u.call_method(interp, name, [u] + rest, 1)[0] for u in self.unders])]
        if name == "shuffle":
            counts = [len(u.files) if isinstance(u, ImageDatastore) else -1 for u in self.unders]
            if min(counts) < 0 or len(set(counts)) != 1:
                raise MatlabError("shuffle: a combined datastore can be shuffled only when all its datastores "
                                  "are image datastores with the same number of files.")
            from .core import _rng
            order = _rng(interp).permutation(counts[0])
            return [CombinedDatastore([u._derive(order) for u in self.unders])]
        out = self._common(interp, name, rest, nargout)
        if out is None:
            raise MatlabError(f"The datastore function '{name}' is not supported.")
        return out

    def get_field(self, interp, name: str) -> Any:
        if name == "UnderlyingDatastores":
            return Cell.from_list(list(self.unders))
        if name == "SupportedOutputFormats":
            return StringArray(np.array(["txt", "csv", "xlsx", "png", "jpg", "tif"], dtype=object).reshape(1, -1))
        if self.has_method(name):
            return self.call_method(interp, name, [self], 1)[0]
        raise MatlabError(f"Unrecognized method, property, or field '{name}' for class '{self.class_name()}'.",
                          "MATLAB:noSuchMethodOrField")

    def display_lines(self) -> list[str]:
        kinds = ", ".join(type(u).__name__ for u in self.unders)
        return ["  CombinedDatastore with properties:", "", f"    UnderlyingDatastores: {{{kinds}}}"]
