"""Reading MATLAB source files whatever encoding they were saved in."""

from __future__ import annotations

import os


def decode_source(data: bytes) -> str:
    """The text of a MATLAB source file given as bytes.

    UTF-8 (with or without a byte-order mark) is tried first, as MATLAB does since R2020a. Files
    saved by older versions in a Windows code page, or in another one (Shift-JIS, GBK), fall back
    to Windows-1252 and then Latin-1, which decodes any byte: comments and character literals with
    such characters read differently, but the code runs. Line ends are normalised to \\n, as
    reading in text mode does.
    """
    for encoding in ("utf-8-sig", "cp1252"):
        try:
            text = data.decode(encoding)
            break
        except UnicodeDecodeError:
            continue
    else:
        text = data.decode("latin-1")
    return text.replace("\r\n", "\n").replace("\r", "\n")


def read_source(path: str | os.PathLike) -> str:
    """The text of the MATLAB source file at ``path`` (see decode_source)."""
    with open(path, "rb") as fh:
        return decode_source(fh.read())
