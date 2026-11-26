"""Fast paths for reading and writing plain numeric delimited text.

The general reader and writer in ``io.py`` handle every field separately (quotes, text columns,
NaN spellings, header rows ...).  A file that holds nothing but decimal numbers needs none of that,
so it is converted in bulk here.  Both functions refuse (return ``None`` / ``False``) whenever the
input is not plainly numeric, and the general code then runs as before; when they do accept the
input they produce exactly what the general code would.
"""

from __future__ import annotations

import re

import numpy as np

from ..errors import MatlabError

_SIMPLE_FORMAT = re.compile(r"%[-+ 0#]*\d*(\.\d+)?[eEfFgG]")
_PLAIN_FORMAT = re.compile(r"%(\.\d+)?[eEfFgG]")
_NUMBER_CHARS = "0123456789eE+-. "
_CHUNK = 60000  # numbers formatted per write


def write_numeric(fname: str, m, delim: str, fmt: str | None, append: bool, newline: str = "\n") -> bool:
    """Write a real numeric matrix as delimited text; False if the general writer must do it."""
    if type(m) is not np.ndarray or m.ndim != 2 or m.dtype.kind not in "fiu" or m.dtype.itemsize > 8:
        return False
    fmt = "%.15g" if fmt is None else fmt
    if not _SIMPLE_FORMAT.fullmatch(fmt):
        return False
    if m.dtype.kind != "f":
        m = m.astype(float)
    finite = bool(np.isfinite(m).all())
    if not finite and not _PLAIN_FORMAT.fullmatch(fmt):
        return False  # a field width pads numbers but not the NaN/Inf words: leave it to the general code
    nrows, ncols = m.shape
    row_format = delim.replace("%", "%%").join([fmt] * ncols) + newline.replace("%", "%%")
    step = max(1, _CHUNK // max(ncols, 1))
    try:
        with open(fname, "a" if append else "w", encoding="utf-8", newline="") as fh:
            for start in range(0, nrows, step):
                block = m[start:start + step]
                chunk = (row_format * block.shape[0]) % tuple(block.ravel().tolist())
                if not finite:
                    chunk = chunk.replace("nan", "NaN").replace("inf", "Inf")
                fh.write(chunk)
    except OSError as e:
        raise MatlabError(f"Unable to write to '{fname}': {e.strerror}.")
    return True


def parse_numeric(lines: list[str], delim: str | None):
    """The matrix held by lines of plain decimal numbers; None if anything else is in there."""
    if not lines or delim is None or len(delim) != 1 or delim in _NUMBER_CHARS:
        return None
    first = lines[0]
    ncols = first.count(delim) + 1
    joined = delim.join(lines)
    if not _allowed(delim).fullmatch(joined):
        return None
    want = ncols - 1
    for line in lines:
        if line.count(delim) != want:
            return None  # ragged rows: the general reader pads them
    try:
        values = list(map(float, joined.split(delim)))
    except ValueError:  # an empty or malformed field
        return None
    return np.array(values).reshape(len(lines), ncols)


_ALLOWED: dict[str, re.Pattern] = {}


def _allowed(delim: str) -> re.Pattern:
    pattern = _ALLOWED.get(delim)
    if pattern is None:
        pattern = _ALLOWED[delim] = re.compile("[0-9eE+\\-. " + re.escape(delim) + "]*")
    return pattern
