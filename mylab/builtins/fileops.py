"""Search path, operating system, archives, the web, workspaces and small compatibility functions."""

from __future__ import annotations

import glob
import gzip as _gzip
import os
import shutil
import sys
import tarfile
import zipfile

import numpy as np

from ..errors import MatlabError
from ..values import (
    Cell, MString, StringArray, Struct, as_2d, char, is_char, to_int, to_str,
)
from ._registry import BUILTINS, builtin
from ._util import bool_scalar, is_text, nargs, num, text

IN_BROWSER = sys.platform == "emscripten"


def _texts(v, what: str = "argument") -> list[str]:
    """A list of Python strings from a char row, char matrix, string array or cellstr."""
    if isinstance(v, Cell):
        return [to_str(x, what) for x in v.values_f()]
    if isinstance(v, StringArray):
        return ["" if x is None else x for x in v.values_f()]
    if is_char(v) and v.ndim == 2 and v.shape[0] > 1:
        return ["".join(r).rstrip() for r in v]
    return [text(v, what)]


def _cellstr(items: list[str], column: bool = True) -> Cell:
    if not items:
        return Cell(np.empty((0, 1) if column else (1, 0), dtype=object))
    return Cell.from_list([char(s) for s in items], column=column)


def _status(interp, nargout: int, ok: bool, msg: str, ident: str):
    """Common [status, message, messageId] convention of the file-management functions."""
    if nargout == 0:
        if not ok:
            raise MatlabError(msg, ident)
        return None
    outs = (bool_scalar(ok), char(msg), char("" if ok else ident))
    return outs[:nargout] if nargout > 1 else outs[0]


# -- search path ------------------------------------------------------------------------------

_SKIP_FOLDERS = ("private", "resources")


def _genpath(folder: str) -> list[str]:
    out = [folder]
    try:
        names = sorted(os.listdir(folder))
    except OSError:
        return []
    for n in names:
        full = os.path.join(folder, n)
        if n.startswith((".", "@", "+")) or n in _SKIP_FOLDERS or not os.path.isdir(full):
            continue
        out.extend(_genpath(full))
    return out


@builtin("genpath")
def _genpath_b(interp, args, nargout):
    nargs(args, 0, None, "genpath")
    folder = os.path.expanduser(text(args[0])) if args else os.getcwd()
    if not os.path.isdir(folder):
        return char("")
    return char(os.pathsep.join(_genpath(folder)))


@builtin("filesep")
def _filesep(interp, args, nargout):
    return char(os.sep)


@builtin("pathsep")
def _pathsep(interp, args, nargout):
    return char(os.pathsep)


@builtin("mfilename")
def _mfilename(interp, args, nargout):
    nargs(args, 0, 1, "mfilename")
    path = interp.frame.file
    if not path:
        return char("")
    opt = text(args[0]).lower() if args else ""
    if opt == "fullpathext":
        return char(os.path.abspath(path))
    if opt == "fullpath":
        return char(os.path.splitext(os.path.abspath(path))[0])
    if opt == "class":
        return char("")
    if opt:
        raise MatlabError("mfilename: the option must be 'fullpath', 'fullpathext' or 'class'.")
    return char(os.path.splitext(os.path.basename(path))[0])


def _which(interp, name: str) -> list[str]:
    frame = interp.frame
    found: list[str] = []
    if frame.has(name):
        found.append("variable")
    if name in frame.functions:
        found.append(os.path.abspath(frame.file) if frame.file else f"{name} is a function defined in this session")
    base, ext = os.path.splitext(name)
    if ext == "" or ext == ".m":
        hit = interp.find_file_function(base)
        if hit is not None:
            found.append(os.path.abspath(hit[1]))
    if ext and ext != ".m" or os.sep in name:
        for d in [os.getcwd(), *interp.path]:
            p = os.path.join(d, name)
            if os.path.isfile(p):
                found.append(os.path.abspath(p))
    if name in getattr(interp, "classes", {}):
        cls_file = getattr(interp.classes[name], "file", None)
        if cls_file:
            found.append(os.path.abspath(cls_file))
    if name in BUILTINS:
        found.append(f"built-in (mylab/{name})")
    seen: list[str] = []
    for f in found:
        if f not in seen:
            seen.append(f)
    return seen


@builtin("which")
def _which_b(interp, args, nargout):
    nargs(args, 1, None, "which")
    names = [text(a) for a in args]
    want_all = "-all" in names
    names = [n for n in names if n != "-all"]
    if not names:
        raise MatlabError("which: a name is required.")
    name = names[0]
    found = _which(interp, name)
    if nargout > 0:
        if want_all:
            return _cellstr(found)
        return char(found[0] if found else "")
    if not found:
        interp.out(f"'{name}' not found.\n")
    for f in (found if want_all else found[:1]):
        interp.out(f"{name} is a variable.\n" if f == "variable" else f + "\n")
    return None


@builtin("what")
def _what(interp, args, nargout):
    nargs(args, 0, 1, "what")
    folder = os.path.expanduser(text(args[0])) if args else os.getcwd()
    if not os.path.isdir(folder):
        folder = next((p for p in interp.path if os.path.basename(p) == folder and os.path.isdir(p)), folder)
    if not os.path.isdir(folder):
        raise MatlabError(f"what: '{folder}' is not a folder.")
    names = sorted(os.listdir(folder))
    groups = {"m": [n for n in names if n.endswith(".m")], "mat": [n for n in names if n.endswith(".mat")],
              "mex": [], "mlapp": [], "mlx": [n for n in names if n.endswith(".mlx")], "mdl": [], "slx": [],
              "p": [], "classes": [n[1:] for n in names if n.startswith("@")],
              "packages": [n[1:] for n in names if n.startswith("+")]}
    if nargout == 0:
        for key, title in (("m", "MATLAB Code files"), ("mat", "MAT-files")):
            if groups[key]:
                interp.out(f"\n{title} in folder {os.path.abspath(folder)}\n\n" + "\n".join(groups[key]) + "\n")
        interp.out("\n")
        return None
    fields: dict = {"path": char(os.path.abspath(folder))}
    for key, items in groups.items():
        fields[key] = _cellstr(items)
    return Struct.scalar(fields)


# -- operating system --------------------------------------------------------------------------

def _run_command(interp, name: str, args, nargout):
    nargs(args, 1, None, name)
    cmd = text(args[0], "command")
    echo = any(is_text(a) and text(a).lower() == "-echo" for a in args[1:])
    if IN_BROWSER:
        raise MatlabError(f"{name}: operating system commands are not available when mylab runs in the browser.")
    import subprocess
    try:
        proc = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                              stdin=subprocess.DEVNULL)
    except OSError as e:
        raise MatlabError(f"{name}: could not run the command: {e.strerror or e}.")
    out = proc.stdout.decode("utf-8", errors="replace").replace("\r\n", "\n")
    if nargout < 2 or echo:
        interp.out(out)
    status = float(proc.returncode)
    return (status, char(out)) if nargout > 1 else (status if nargout else None)


@builtin("system")
def _system(interp, args, nargout):
    return _run_command(interp, "system", args, nargout)


@builtin("unix")
def _unix(interp, args, nargout):
    if os.name == "nt":
        return (0.0, char("")) if nargout > 1 else (0.0 if nargout else None)
    return _run_command(interp, "unix", args, nargout)


@builtin("dos")
def _dos(interp, args, nargout):
    return _run_command(interp, "dos", args, nargout)


@builtin("setenv")
def _setenv(interp, args, nargout):
    nargs(args, 1, 2, "setenv")
    os.environ[text(args[0], "variable name")] = text(args[1], "value") if len(args) > 1 else ""
    return None


def _sources(pattern: str) -> list[str]:
    pattern = os.path.expanduser(pattern)
    if any(ch in pattern for ch in "*?"):
        return sorted(glob.glob(pattern))
    return [pattern] if os.path.exists(pattern) else []


def _transfer(interp, args, nargout, name: str, move: bool):
    nargs(args, 1, 3, name)
    src = text(args[0], "source")
    dst = os.path.expanduser(text(args[1], "destination")) if len(args) > 1 else os.getcwd()
    ident = f"MATLAB:{name.upper()}:FileNotFound"
    items = _sources(src)
    if not items:
        return _status(interp, nargout, False, f"{name}: no file or folder matches '{src}'.", ident)
    many = len(items) > 1
    try:
        if many:
            os.makedirs(dst, exist_ok=True)
        for item in items:
            target = dst
            if os.path.isdir(dst) and (many or not os.path.isdir(item) or move):
                target = os.path.join(dst, os.path.basename(os.path.normpath(item)))
            if os.path.abspath(target) == os.path.abspath(item):
                continue
            if move:
                if os.path.isfile(target):
                    os.remove(target)
                shutil.move(item, target)
            elif os.path.isdir(item):
                shutil.copytree(item, target, dirs_exist_ok=True)
            else:
                parent = os.path.dirname(target)
                if parent and not os.path.isdir(parent):
                    os.makedirs(parent, exist_ok=True)
                shutil.copy2(item, target)
    except (OSError, shutil.Error) as e:
        reason = getattr(e, "strerror", None) or str(e)
        return _status(interp, nargout, False, f"{name}: cannot {'move' if move else 'copy'} '{src}': {reason}.",
                       f"MATLAB:{name.upper()}:OSError")
    return _status(interp, nargout, True, "", "")


@builtin("copyfile")
def _copyfile(interp, args, nargout):
    return _transfer(interp, args, nargout, "copyfile", False)


@builtin("movefile")
def _movefile(interp, args, nargout):
    return _transfer(interp, args, nargout, "movefile", True)


@builtin("ls")
def _ls(interp, args, nargout):
    target = os.path.expanduser(text(args[0])) if args and not text(args[0]).startswith("-") else "."
    if any(ch in target for ch in "*?"):
        names = sorted(os.path.basename(p) if os.path.dirname(p) in ("", ".") else p for p in glob.glob(target))
    elif os.path.isdir(target):
        names = sorted((n for n in os.listdir(target) if not n.startswith(".")), key=str.lower)
    elif os.path.exists(target):
        names = [target]
    else:
        raise MatlabError(f"ls: cannot access '{target}': no such file or folder.")
    if nargout > 0:
        if not names:
            return np.zeros((0, 0), dtype="<U1")
        width = max(len(n) for n in names)
        return np.array([list(n.ljust(width)) for n in names], dtype="<U1")
    if names:
        width = max(len(n) for n in names) + 2
        per_line = max(1, 80 // width)
        lines = ["".join(n.ljust(width) for n in names[i:i + per_line]).rstrip()
                 for i in range(0, len(names), per_line)]
        interp.out("\n".join(lines) + "\n\n")
    return None


@builtin("memory")
def _memory(interp, args, nargout):
    total = avail = 0.0
    try:
        page = os.sysconf("SC_PAGE_SIZE")
        total = float(page * os.sysconf("SC_PHYS_PAGES"))
        avail = float(page * os.sysconf("SC_AVPHYS_PAGES"))
    except (ValueError, OSError, AttributeError):
        pass
    used = 0.0
    try:
        import resource
        used = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * (1 if sys.platform == "darwin" else 1024)
    except (ImportError, OSError, ValueError):
        pass
    user = Struct.scalar({"MaxPossibleArrayBytes": avail, "MemAvailableAllArrays": avail, "MemUsedMATLAB": used})
    if nargout == 0:
        interp.out(f"Memory available for arrays: {avail / 2**20:.0f} MB\n"
                   f"Memory used by mylab:        {used / 2**20:.0f} MB\n"
                   f"Physical memory (RAM):       {total / 2**20:.0f} MB\n")
        return None
    system = Struct.scalar({"PhysicalMemory": Struct.scalar({"Available": avail, "Total": total})})
    return (user, system) if nargout > 1 else user


# -- archives -----------------------------------------------------------------------------------

def _archive_members(files_arg, root: str, name: str) -> list[tuple[str, str]]:
    """(path on disk, name inside the archive) for every file to store."""
    out: list[tuple[str, str]] = []
    for pattern in _texts(files_arg, "file name"):
        full = os.path.join(root, os.path.expanduser(pattern))
        hits = _sources(full)
        if not hits:
            raise MatlabError(f"{name}: no file or folder matches '{pattern}'.")
        for h in hits:
            base = os.path.dirname(os.path.normpath(h)) if os.path.isabs(pattern) else root
            if os.path.isdir(h):
                for d, _dirs, fnames in sorted(os.walk(h)):
                    for fn in sorted(fnames):
                        p = os.path.join(d, fn)
                        out.append((p, os.path.relpath(p, base)))
            else:
                out.append((h, os.path.relpath(h, base)))
    return out


def _with_ext(path: str, ext: str) -> str:
    return path if os.path.splitext(path)[1] else path + ext


def _safe_target(outdir: str, member: str, name: str) -> str:
    target = os.path.abspath(os.path.join(outdir, member))
    if os.path.commonpath([os.path.abspath(outdir), target]) != os.path.abspath(outdir):
        raise MatlabError(f"{name}: the archive entry '{member}' would be written outside the output folder.")
    return target


def _maybe_download(interp, src: str, name: str) -> tuple[str, str | None]:
    """Local path for src; a URL is downloaded to a temporary file (returned second, to delete)."""
    if src.lower().startswith(("http://", "https://")):
        import tempfile
        fd, tmp = tempfile.mkstemp(suffix=os.path.splitext(src)[1] or ".bin")
        with os.fdopen(fd, "wb") as fh:
            fh.write(_fetch(src, name))
        return tmp, tmp
    return os.path.expanduser(src), None


@builtin("zip")
def _zip(interp, args, nargout):
    nargs(args, 2, 3, "zip")
    target = _with_ext(os.path.expanduser(text(args[0], "zip file name")), ".zip")
    root = os.path.expanduser(text(args[2], "root folder")) if len(args) > 2 else os.getcwd()
    members = [m for m in _archive_members(args[1], root, "zip") if os.path.abspath(m[0]) != os.path.abspath(target)]
    try:
        with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as zf:
            for path, arc in members:
                zf.write(path, arc.replace(os.sep, "/"))
    except OSError as e:
        raise MatlabError(f"zip: cannot write '{target}': {e.strerror or e}.")
    return _cellstr([m[1] for m in members], column=False) if nargout else None


@builtin("unzip")
def _unzip(interp, args, nargout):
    nargs(args, 1, 2, "unzip")
    src, tmp = _maybe_download(interp, text(args[0], "zip file name"), "unzip")
    if not os.path.isfile(src) and os.path.isfile(src + ".zip"):
        src += ".zip"
    outdir = os.path.expanduser(text(args[1], "output folder")) if len(args) > 1 else os.getcwd()
    names: list[str] = []
    try:
        with zipfile.ZipFile(src) as zf:
            for info in zf.infolist():
                target = _safe_target(outdir, info.filename, "unzip")
                if info.is_dir():
                    os.makedirs(target, exist_ok=True)
                    continue
                os.makedirs(os.path.dirname(target), exist_ok=True)
                with zf.open(info) as fin, open(target, "wb") as fout:
                    shutil.copyfileobj(fin, fout)
                names.append(os.path.join(outdir, info.filename) if len(args) > 1 else info.filename)
    except FileNotFoundError:
        raise MatlabError(f"unzip: file '{src}' not found.")
    except (zipfile.BadZipFile, OSError) as e:
        raise MatlabError(f"unzip: cannot read '{src}': {e}.")
    finally:
        if tmp:
            os.remove(tmp)
    return _cellstr(names, column=False) if nargout else None


@builtin("gzip")
def _gzip_b(interp, args, nargout):
    nargs(args, 1, 2, "gzip")
    outdir = os.path.expanduser(text(args[1], "output folder")) if len(args) > 1 else None
    if outdir:
        os.makedirs(outdir, exist_ok=True)
    made = []
    for path, _arc in _archive_members(args[0], os.getcwd(), "gzip"):
        target = os.path.join(outdir or os.path.dirname(path), os.path.basename(path) + ".gz")
        with open(path, "rb") as fin, _gzip.open(target, "wb") as fout:
            shutil.copyfileobj(fin, fout)
        made.append(target)
    return _cellstr(made, column=False) if nargout else None


@builtin("gunzip")
def _gunzip(interp, args, nargout):
    nargs(args, 1, 2, "gunzip")
    outdir = os.path.expanduser(text(args[1], "output folder")) if len(args) > 1 else None
    if outdir:
        os.makedirs(outdir, exist_ok=True)
    made = []
    for pattern in _texts(args[0], "file name"):
        src, tmp = _maybe_download(interp, pattern, "gunzip")
        hits = _sources(src)
        if not hits:
            raise MatlabError(f"gunzip: file '{pattern}' not found.")
        for path in hits:
            base = os.path.basename(pattern if tmp else path)
            base = base[:-3] if base.lower().endswith(".gz") else (
                base[:-4] + ".tar" if base.lower().endswith(".tgz") else base + ".out")
            target = os.path.join(outdir or (os.getcwd() if tmp else os.path.dirname(path)), base)
            try:
                with _gzip.open(path, "rb") as fin, open(target, "wb") as fout:
                    shutil.copyfileobj(fin, fout)
            except (OSError, EOFError) as e:
                raise MatlabError(f"gunzip: cannot read '{pattern}': {e}.")
            made.append(target)
        if tmp:
            os.remove(tmp)
    return _cellstr(made, column=False) if nargout else None


@builtin("tar")
def _tar(interp, args, nargout):
    nargs(args, 2, 3, "tar")
    target = os.path.expanduser(text(args[0], "tar file name"))
    gz = target.lower().endswith((".tgz", ".gz"))
    if not gz:
        target = _with_ext(target, ".tar")
    root = os.path.expanduser(text(args[2], "root folder")) if len(args) > 2 else os.getcwd()
    members = [m for m in _archive_members(args[1], root, "tar") if os.path.abspath(m[0]) != os.path.abspath(target)]
    try:
        with tarfile.open(target, "w:gz" if gz else "w") as tf:
            for path, arc in members:
                tf.add(path, arc.replace(os.sep, "/"))
    except OSError as e:
        raise MatlabError(f"tar: cannot write '{target}': {e.strerror or e}.")
    return _cellstr([m[1] for m in members], column=False) if nargout else None


@builtin("untar")
def _untar(interp, args, nargout):
    nargs(args, 1, 2, "untar")
    src, tmp = _maybe_download(interp, text(args[0], "tar file name"), "untar")
    if not os.path.isfile(src) and os.path.isfile(src + ".tar"):
        src += ".tar"
    outdir = os.path.expanduser(text(args[1], "output folder")) if len(args) > 1 else os.getcwd()
    names: list[str] = []
    try:
        with tarfile.open(src) as tf:
            for member in tf.getmembers():
                target = _safe_target(outdir, member.name, "untar")
                if member.isdir():
                    os.makedirs(target, exist_ok=True)
                elif member.isfile():
                    os.makedirs(os.path.dirname(target), exist_ok=True)
                    with tf.extractfile(member) as fin, open(target, "wb") as fout:
                        shutil.copyfileobj(fin, fout)
                    names.append(os.path.join(outdir, member.name) if len(args) > 1 else member.name)
    except FileNotFoundError:
        raise MatlabError(f"untar: file '{src}' not found.")
    except (tarfile.TarError, OSError) as e:
        raise MatlabError(f"untar: cannot read '{src}': {e}.")
    finally:
        if tmp:
            os.remove(tmp)
    return _cellstr(names, column=False) if nargout else None


# -- the web ------------------------------------------------------------------------------------

def _query_string(pairs: list) -> str:
    from urllib.parse import urlencode
    items = []
    for k in range(0, len(pairs) - 1, 2):
        v = pairs[k + 1]
        if is_text(v):
            val = text(v)
        else:
            x = num(v).flat[0] if isinstance(v, np.ndarray) and v.size else 0
            val = str(int(x)) if float(x) == int(x) else repr(float(x))
        items.append((text(pairs[k], "parameter name"), val))
    return urlencode(items)


def _fetch(url: str, name: str, data: bytes | None = None, timeout: float = 30.0,
           want_type: bool = False):
    if IN_BROWSER:
        raise MatlabError(f"{name}: web access is not available when mylab runs in the browser.",
                          "MATLAB:webservices:Unavailable")
    import urllib.error
    import urllib.request
    if not url.lower().startswith(("http://", "https://", "file://", "ftp://")):
        raise MatlabError(f"{name}: '{url}' is not a valid URL; it must start with http:// or https://.")
    req = urllib.request.Request(url, data=data, headers={"User-Agent": "mylab"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read()
            ctype = resp.headers.get("Content-Type", "") if hasattr(resp, "headers") else ""
    except urllib.error.HTTPError as e:
        raise MatlabError(f"{name}: the server returned status {e.code} ({e.reason}) for '{url}'.",
                          "MATLAB:webservices:HTTP" + str(e.code))
    except (urllib.error.URLError, OSError, ValueError) as e:
        reason = getattr(e, "reason", e)
        raise MatlabError(f"{name}: could not reach '{url}' ({reason}). Check the address and that this "
                          "computer has network access.", "MATLAB:webservices:ConnectionFailed")
    return (body, ctype) if want_type else body


def _decode_body(body: bytes, ctype: str = "") -> str:
    enc = "utf-8"
    if "charset=" in ctype.lower():
        enc = ctype.lower().split("charset=", 1)[1].split(";")[0].strip() or "utf-8"
    try:
        return body.decode(enc)
    except (UnicodeDecodeError, LookupError):
        return body.decode("latin-1")


def _split_web_options(args: list) -> tuple[list, float]:
    """Drop a trailing weboptions struct / 'Timeout' pair; returns (other args, timeout)."""
    timeout = 30.0
    rest = list(args)
    if rest and isinstance(rest[-1], Struct):
        opts = rest.pop().data.flat[0]
        if "Timeout" in opts:
            timeout = float(num(opts["Timeout"]).flat[0])
    out = []
    i = 0
    while i < len(rest):
        if is_text(rest[i]) and text(rest[i]).lower() == "timeout" and i + 1 < len(rest):
            timeout = float(num(rest[i + 1]).flat[0])
            i += 2
            continue
        out.append(rest[i])
        i += 1
    return out, timeout


def _url_legacy(args: list, name: str) -> tuple[str, bytes | None, float]:
    """urlread/urlwrite extras: 'get'/'post' with a cell of parameters, and 'Timeout'."""
    rest, timeout = _split_web_options(list(args))
    url = text(rest[0], "URL")
    data = None
    if len(rest) >= 3 and is_text(rest[1]) and isinstance(rest[2], Cell):
        query = _query_string(rest[2].values_f())
        if text(rest[1]).lower() == "post":
            data = query.encode()
        elif query:
            url += ("&" if "?" in url else "?") + query
    return url, data, timeout


@builtin("urlread")
def _urlread(interp, args, nargout):
    nargs(args, 1, None, "urlread")
    url, data, timeout = _url_legacy(args, "urlread")
    try:
        body, ctype = _fetch(url, "urlread", data, timeout, want_type=True)
    except MatlabError:
        if nargout > 1:
            return char(""), bool_scalar(False)
        raise
    s = char(_decode_body(body, ctype))
    return (s, bool_scalar(True)) if nargout > 1 else s


@builtin("urlwrite")
def _urlwrite(interp, args, nargout):
    nargs(args, 2, None, "urlwrite")
    target = os.path.expanduser(text(args[1], "file name"))
    url, data, timeout = _url_legacy([args[0], *args[2:]], "urlwrite")
    try:
        body = _fetch(url, "urlwrite", data, timeout)
        with open(target, "wb") as fh:
            fh.write(body)
    except (MatlabError, OSError) as e:
        if nargout > 1:
            return char(""), bool_scalar(False)
        raise e if isinstance(e, MatlabError) else MatlabError(f"urlwrite: cannot write '{target}'.")
    full = char(os.path.abspath(target))
    return (full, bool_scalar(True)) if nargout > 1 else (full if nargout else None)


@builtin("webread")
def _webread(interp, args, nargout):
    nargs(args, 1, None, "webread")
    rest, timeout = _split_web_options(list(args))
    url = text(rest[0], "URL")
    query = _query_string(rest[1:])
    if query:
        url += ("&" if "?" in url else "?") + query
    body, ctype = _fetch(url, "webread", None, timeout, want_type=True)
    kind = ctype.lower()
    if kind.startswith("image/") and "svg" not in kind and "imread" in BUILTINS:
        import tempfile
        fd, tmp = tempfile.mkstemp(suffix="." + kind.split("/")[1].split(";")[0])
        try:
            with os.fdopen(fd, "wb") as fh:
                fh.write(body)
            return BUILTINS["imread"](interp, [char(tmp)], 1)
        finally:
            os.remove(tmp)
    s = _decode_body(body, ctype)
    if "json" in kind and "jsondecode" in BUILTINS:
        return BUILTINS["jsondecode"](interp, [char(s)], 1)
    return char(s)


@builtin("websave")
def _websave(interp, args, nargout):
    nargs(args, 2, None, "websave")
    target = os.path.expanduser(text(args[0], "file name"))
    rest, timeout = _split_web_options(list(args[1:]))
    url = text(rest[0], "URL")
    query = _query_string(rest[1:])
    if query:
        url += ("&" if "?" in url else "?") + query
    body = _fetch(url, "websave", None, timeout)
    try:
        with open(target, "wb") as fh:
            fh.write(body)
    except OSError as e:
        raise MatlabError(f"websave: cannot write '{target}': {e.strerror or e}.")
    return char(os.path.abspath(target)) if nargout else None


@builtin("weboptions")
def _weboptions(interp, args, nargout):
    fields: dict = {"Timeout": 5.0, "ContentType": char("auto"), "RequestMethod": char("auto")}
    for k in range(0, len(args) - 1, 2):
        fields[text(args[k], "option name")] = args[k + 1]
    return Struct.scalar(fields)


# -- versions and capabilities ----------------------------------------------------------------

def _version_tuple(s: str) -> tuple[int, ...]:
    import re
    parts = re.findall(r"\d+", s)
    if not parts:
        raise MatlabError(f"verLessThan: '{s}' is not a version number such as '9.4'.")
    return tuple(int(p) for p in parts[:4])


@builtin("verLessThan")
def _verlessthan(interp, args, nargout):
    nargs(args, 2, 2, "verLessThan")
    from .system import PRODUCTS
    product = text(args[0], "toolbox name").lower()
    wanted = _version_tuple(text(args[1], "version"))
    current = next((ver for _name, folder, ver in PRODUCTS if folder == product), None)
    if current is None:
        return bool_scalar(True)  # a toolbox mylab does not have counts as older than any version
    mine = _version_tuple(current)
    n = max(len(mine), len(wanted))
    pad = lambda t: t + (0,) * (n - len(t))
    return bool_scalar(pad(mine) < pad(wanted))


@builtin("isMATLABReleaseOlderThan")
def _is_release_older(interp, args, nargout):
    nargs(args, 1, None, "isMATLABReleaseOlderThan")
    from .system import MATLAB_RELEASE
    key = lambda r: (int(r[1:5]), r[5:6].lower())
    want = text(args[0], "release")
    if len(want) < 6 or not want[1:5].isdigit():
        raise MatlabError("isMATLABReleaseOlderThan: the release must look like 'R2021b'.")
    return bool_scalar(key(MATLAB_RELEASE) < key(want))


@builtin("usejava")
def _usejava(interp, args, nargout):
    return bool_scalar(False)


@builtin("license")
def _license(interp, args, nargout):
    """Scripts ask license('test', feature) before using a toolbox; say yes for those mylab covers."""
    covered = {"matlab", "image_toolbox", "signal_toolbox", "statistics_toolbox", "optimization_toolbox"}
    if not args:
        return char("mylab (MIT)")
    cmd = text(args[0]).lower()
    if cmd in ("test", "checkout") and len(args) > 1:
        return bool_scalar(text(args[1]).lower() in covered)
    if cmd == "inuse":
        return Struct.scalar({"feature": char("matlab"), "user": char("")}) if nargout else None
    return char("mylab (MIT)")


@builtin("mex")
def _mex(interp, args, nargout):
    raise MatlabError("mex: compiled MEX files are not supported by mylab. Use the plain MATLAB-language "
                      "version of the function if the project ships one.", "mylab:mex:unsupported")


@builtin("gpuArray")
def _gpuarray(interp, args, nargout):
    nargs(args, 1, 1, "gpuArray")
    return args[0]


@builtin("gather")
def _gather(interp, args, nargout):
    nargs(args, 1, None, "gather")
    return tuple(args[:max(nargout, 1)]) if nargout > 1 else args[0]


@builtin("canUseGPU", "canUseParallelPool")
def _no_gpu(interp, args, nargout):
    return bool_scalar(False)


@builtin("gpuDevice")
def _gpudevice(interp, args, nargout):
    if nargout == 0 and args:
        return None  # gpuDevice(1) or gpuDevice([]) to select or reset a device
    return Struct.scalar({"Name": char("none (mylab computes on the CPU)"), "Index": 1.0,
                          "DeviceSupported": bool_scalar(False), "DeviceAvailable": bool_scalar(False),
                          "AvailableMemory": 0.0, "TotalMemory": 0.0})


@builtin("gpuDeviceCount")
def _gpu_count(interp, args, nargout):
    return 0.0


def _pool() -> Struct:
    return Struct.scalar({"NumWorkers": 1.0, "Connected": bool_scalar(True), "Cluster": char("local"),
                          "IdleTimeout": float("inf"), "SpmdEnabled": bool_scalar(False)})


@builtin("parpool")
def _parpool(interp, args, nargout):
    return _pool() if nargout else None


@builtin("gcp")
def _gcp(interp, args, nargout):
    if args and is_text(args[0]) and text(args[0]).lower() == "nocreate":
        return np.zeros((0, 0))
    return _pool()


@builtin("matlabpool")
def _matlabpool(interp, args, nargout):
    if args and is_text(args[0]) and text(args[0]).lower() == "size":
        return 0.0
    return None


# -- preferences (kept in memory for the session) ---------------------------------------------

def _prefs(interp) -> dict:
    return interp.state.setdefault("prefs", {})


def _pref_names(v) -> list[str]:
    return _texts(v, "preference name")


@builtin("setpref", "addpref")
def _setpref(interp, args, nargout):
    nargs(args, 3, 3, "setpref")
    group = _prefs(interp).setdefault(text(args[0], "group"), {})
    names = _pref_names(args[1])
    if isinstance(args[1], Cell):
        if not isinstance(args[2], Cell) or len(args[2].values_f()) != len(names):
            raise MatlabError("setpref: with a cell array of names, the values must be a cell array of the "
                              "same size.")
        values = args[2].values_f()
    else:
        values = [args[2]]
    for n, v in zip(names, values):
        group[n] = v
    return None


@builtin("getpref")
def _getpref(interp, args, nargout):
    nargs(args, 0, 3, "getpref")
    prefs = _prefs(interp)
    if not args:
        return Struct.scalar({g: Struct.scalar(dict(p)) for g, p in prefs.items()})
    gname = text(args[0], "group")
    group = prefs.get(gname, {})
    if len(args) == 1:
        return Struct.scalar(dict(group))
    names = _pref_names(args[1])
    defaults = None
    if len(args) > 2:
        defaults = args[2].values_f() if isinstance(args[1], Cell) and isinstance(args[2], Cell) else [args[2]]
    out = []
    for i, n in enumerate(names):
        if n not in group:
            if defaults is None or i >= len(defaults):
                raise MatlabError(f"getpref: group '{gname}' has no preference named '{n}'.",
                                  "MATLAB:getpref:InvalidPreference")
            group = prefs.setdefault(gname, {})
            group[n] = defaults[i]
        out.append(group[n])
    return Cell.from_list(out) if isinstance(args[1], Cell) else out[0]


@builtin("ispref")
def _ispref(interp, args, nargout):
    nargs(args, 1, 2, "ispref")
    group = _prefs(interp).get(text(args[0], "group"))
    if len(args) == 1:
        return bool_scalar(group is not None)
    flags = [group is not None and n in group for n in _pref_names(args[1])]
    return np.array([flags], dtype=bool) if isinstance(args[1], Cell) else bool_scalar(flags[0])


@builtin("rmpref")
def _rmpref(interp, args, nargout):
    nargs(args, 1, 2, "rmpref")
    prefs = _prefs(interp)
    gname = text(args[0], "group")
    if gname not in prefs:
        raise MatlabError(f"rmpref: there is no preference group named '{gname}'.")
    if len(args) == 1:
        del prefs[gname]
        return None
    for n in _pref_names(args[1]):
        if n not in prefs[gname]:
            raise MatlabError(f"rmpref: group '{gname}' has no preference named '{n}'.")
        del prefs[gname][n]
    return None


# -- workspaces ---------------------------------------------------------------------------------

def _workspace(interp, v, name: str):
    ws = text(v, "workspace").lower()
    if ws == "base":
        return interp.base
    if ws == "caller":
        return interp.frames[-2] if len(interp.frames) > 1 else interp.base
    raise MatlabError(f"{name}: the workspace must be 'base' or 'caller'.")


@builtin("assignin")
def _assignin(interp, args, nargout):
    nargs(args, 3, 3, "assignin")
    import re
    frame = _workspace(interp, args[0], "assignin")
    name = text(args[1], "variable name")
    if not re.fullmatch(r"[A-Za-z]\w*", name):
        raise MatlabError(f"assignin: '{name}' is not a valid variable name.")
    frame.set(name, args[2])
    return None


@builtin("evalin")
def _evalin(interp, args, nargout):
    nargs(args, 2, 3, "evalin")
    from ..parser import parse
    frame = _workspace(interp, args[0], "evalin")

    def run(code: str):
        program = parse(code)
        if nargout > 0 and len(program.body) == 1 and type(program.body[0]).__name__ == "ExprStmt":
            vals = interp.eval_multi(program.body[0].expr, frame, nargout)
            return tuple(vals) if len(vals) > 1 else (vals[0] if vals else None)
        interp.run_program(program, frame)
        return None

    # the target workspace becomes the current one while the code runs
    interp.frames.append(frame)
    try:
        try:
            return run(text(args[1], "expression"))
        except MatlabError:
            if len(args) < 3:
                raise
            return run(text(args[2], "catch expression"))
    finally:
        interp.frames.pop()


@builtin("inputname")
def _inputname(interp, args, nargout):
    return char("")


class _Diary:
    """Copies everything printed to a file while the diary is on."""

    def __init__(self, interp):
        self.interp = interp
        self.inner = interp.out
        self.file = "diary"
        self.on = False

    def __call__(self, s):
        self.inner(s)
        if self.on:
            try:
                with open(self.file, "a", encoding="utf-8") as fh:
                    fh.write(s)
            except OSError:
                self.on = False


@builtin("diary")
def _diary(interp, args, nargout):
    nargs(args, 0, 1, "diary")
    d = interp.state.get("diary")
    if d is None:
        d = interp.state["diary"] = _Diary(interp)
    if interp.out is not d:
        # (re)attach around whatever output function is current
        d.inner = interp.out
        interp.out = d
    arg = text(args[0], "diary argument") if args else None
    if arg is None:
        d.on = not d.on
    elif arg.lower() == "on":
        d.on = True
    elif arg.lower() == "off":
        d.on = False
    else:
        d.file = os.path.expanduser(arg)
        d.on = True
    if d.on:
        try:
            open(d.file, "a", encoding="utf-8").close()
        except OSError as e:
            d.on = False
            raise MatlabError(f"diary: cannot write to '{d.file}': {e.strerror or e}.")
    return None


# -- small compatibility functions -----------------------------------------------------------

@builtin("isstr")
def _isstr(interp, args, nargout):
    nargs(args, 1, 1, "isstr")
    return bool_scalar(is_char(args[0]))


@builtin("strmatch")
def _strmatch(interp, args, nargout):
    nargs(args, 2, 3, "strmatch")
    key = text(args[0], "search text")
    exact = len(args) > 2 and text(args[2]).lower() == "exact"
    v = args[1]
    if isinstance(v, (Cell, StringArray)):
        rows = _texts(v)
        if exact:
            hits = [i for i, r in enumerate(rows) if r == key]
        else:
            hits = [i for i, r in enumerate(rows) if r.startswith(key)]
    elif is_char(v) or isinstance(v, MString):
        arr = char(str(v)) if isinstance(v, MString) else v
        rows = ["".join(r) for r in arr] if arr.size else []
        if exact:
            # in a char matrix the rows are padded, so trailing blanks do not count
            hits = [i for i, r in enumerate(rows) if r.rstrip(" \0") == key.rstrip(" \0")]
        else:
            hits = [i for i, r in enumerate(rows) if r.startswith(key)]
    else:
        raise MatlabError("strmatch: the second input must be a char array or a cell array of text.")
    if not hits:
        return np.zeros((0, 1))
    return np.array(hits, dtype=float).reshape(-1, 1) + 1


@builtin("findstr")
def _findstr(interp, args, nargout):
    nargs(args, 2, 2, "findstr")
    a, b = text(args[0]), text(args[1])
    long, short = (a, b) if len(a) >= len(b) else (b, a)
    if not short:
        return np.zeros((0, 0))
    hits = [i + 1 for i in range(len(long) - len(short) + 1) if long.startswith(short, i)]
    return np.array([hits], dtype=float) if hits else np.zeros((0, 0))


@builtin("nargchk")
def _nargchk(interp, args, nargout):
    nargs(args, 3, 4, "nargchk")
    lo, hi, n = (float(num(a).flat[0]) for a in args[:3])
    as_struct = len(args) > 3 and text(args[3]).lower() == "struct"
    msg = ident = ""
    if n < lo:
        msg, ident = "Not enough input arguments.", "MATLAB:nargchk:notEnoughInputs"
    elif n > hi:
        msg, ident = "Too many input arguments.", "MATLAB:nargchk:tooManyInputs"
    if as_struct:
        if not msg:
            return Struct(np.empty((0, 1), dtype=object), ["message", "identifier"])
        return Struct.scalar({"message": char(msg), "identifier": char(ident)})
    return char(msg) if msg else np.zeros((0, 0))


@builtin("shiftdim")
def _shiftdim(interp, args, nargout):
    nargs(args, 1, 2, "shiftdim")
    a = args[0]
    data = a.data if isinstance(a, (Cell, StringArray)) else a
    if not isinstance(data, np.ndarray):
        raise MatlabError("shiftdim: the input must be an array.")
    rewrap = (lambda x: type(a)(x)) if isinstance(a, (Cell, StringArray)) else (lambda x: x)

    def fix(x):
        return as_2d(x) if x.ndim >= 2 else x.reshape(-1, 1) if x.ndim == 1 else x.reshape(1, 1)

    if len(args) == 1:
        lead = 0
        while lead < data.ndim - 1 and data.shape[lead] == 1:
            lead += 1
        if data.size == 1:
            lead = 0
        shape = data.shape[lead:]
        out = data.reshape(shape + (1,) * max(0, 2 - len(shape)), order="F") if lead else data
        out = rewrap(fix(out))
        return (out, float(lead)) if nargout > 1 else out
    n = to_int(args[1], "shift")
    if n == 0:
        return a
    if n > 0:
        k = n % data.ndim
        out = np.transpose(data, list(range(k, data.ndim)) + list(range(k)))
        if out.ndim == 1:
            out = out.reshape(-1, 1)
        return rewrap(fix(out))
    return rewrap(data.reshape((1,) * (-n) + data.shape, order="F"))


@builtin("flipdim")
def _flipdim(interp, args, nargout):
    nargs(args, 2, 2, "flipdim")
    return BUILTINS["flip"](interp, list(args), 1)


@builtin("swapbytes")
def _swapbytes(interp, args, nargout):
    nargs(args, 1, 1, "swapbytes")
    a = args[0]
    if not isinstance(a, np.ndarray) or a.dtype.kind not in "iuf":
        raise MatlabError("swapbytes: the input must be a numeric array.")
    return a.byteswap()


@builtin("blkdiag")
def _blkdiag(interp, args, nargout):
    blocks = []
    for a in args:
        m = num(a) if not isinstance(a, np.ndarray) or a.dtype.kind == "U" else a
        if m.ndim > 2:
            raise MatlabError("blkdiag: the inputs must be matrices.")
        blocks.append(m)
    if not blocks:
        return np.zeros((0, 0))
    rows = sum(b.shape[0] for b in blocks)
    cols = sum(b.shape[1] for b in blocks)
    dtype = np.result_type(*[b.dtype if b.dtype != bool else np.float64 for b in blocks])
    if all(b.dtype == bool for b in blocks):
        dtype = np.dtype(float)
    out = np.zeros((rows, cols), dtype=dtype)
    r = c = 0
    for b in blocks:
        out[r:r + b.shape[0], c:c + b.shape[1]] = b
        r += b.shape[0]
        c += b.shape[1]
    return out


def _field_path(interp, args: list, name: str) -> tuple[str, dict]:
    """Source text for a chain of fields and {subscripts}, plus the variables it refers to."""
    code = ""
    env: dict = {}
    for k, a in enumerate(args):
        if isinstance(a, Cell):
            subs = []
            for j, s in enumerate(a.values_f()):
                if is_text(s) and text(s) == ":":
                    subs.append(":")
                else:
                    env[f"i{k}_{j}__"] = s
                    subs.append(f"i{k}_{j}__")
            code += "(" + ", ".join(subs) + ")"
        elif is_text(a):
            env[f"f{k}__"] = char(text(a))
            code += f".(f{k}__)"
        else:
            raise MatlabError(f"{name}: field names must be text, and subscripts must be in a cell array.")
    return code, env


def _eval_with(interp, code: str, env: dict, result: str | None):
    from ..interpreter import Frame
    from ..parser import parse
    frame = Frame(interp, functions=interp.frame.functions)
    frame.vars.update(env)
    interp.frames.append(frame)
    try:
        program = parse(code)
        if result is None:
            return interp.eval(program.body[0].expr, frame)
        interp.run_program(program, frame)
        return frame.vars[result]
    finally:
        interp.frames.pop()


@builtin("getfield")
def _getfield(interp, args, nargout):
    nargs(args, 1, None, "getfield")
    s = args[0]
    if len(args) == 2 and isinstance(s, Struct) and is_text(args[1]):
        return s.get_field(0, text(args[1]))
    if len(args) == 1:
        return s
    if not isinstance(s, Struct) and not hasattr(s, "isa_class"):
        raise MatlabError("getfield: the first input must be a struct or an object.")
    code, env = _field_path(interp, list(args[1:]), "getfield")
    env["s__"] = s
    return _eval_with(interp, "s__" + code, env, None)


@builtin("setfield")
def _setfield(interp, args, nargout):
    nargs(args, 3, None, "setfield")
    s = args[0]
    if isinstance(s, np.ndarray) and s.size == 0:
        s = Struct()
    if len(args) == 3 and isinstance(s, Struct) and is_text(args[1]) and s.data.size == 1:
        out = s.copy()
        out.set_field(0, text(args[1]), args[2])
        return out
    if not isinstance(s, Struct) and not hasattr(s, "isa_class"):
        raise MatlabError("setfield: the first input must be a struct or an object.")
    code, env = _field_path(interp, list(args[1:-1]), "setfield")
    env["s__"] = s.copy() if isinstance(s, Struct) else s
    env["v__"] = args[-1]
    return _eval_with(interp, f"s__{code} = v__;", env, "s__")


@builtin("substruct")
def _substruct(interp, args, nargout):
    nargs(args, 2, None, "substruct")
    if len(args) % 2:
        raise MatlabError("substruct: arguments must come in type, subscript pairs.")
    n = len(args) // 2
    data = np.empty((1, n), dtype=object)
    for k in range(n):
        kind = text(args[2 * k], "subscript type")
        subs = args[2 * k + 1]
        if kind not in ("()", "{}", "."):
            raise MatlabError("substruct: the type must be '()', '{}' or '.'.")
        if kind == "." and not is_text(subs):
            raise MatlabError("substruct: a '.' subscript must be a field name.")
        if kind != "." and not isinstance(subs, Cell):
            raise MatlabError("substruct: '()' and '{}' subscripts must be given in a cell array.")
        data[0, k] = {"type": char(kind), "subs": subs}
    return Struct(data, ["type", "subs"])


def _apply_subs(interp, a, s, name: str, value=None, assign: bool = False):
    if not isinstance(s, Struct) or "type" not in s.fields or "subs" not in s.fields:
        raise MatlabError(f"{name}: the subscript must be a struct with fields 'type' and 'subs' (see substruct).")
    code = ""
    env: dict = {"a__": a}
    for k, el in enumerate(s.data.flatten(order="F")):
        kind = text(el["type"])
        subs = el["subs"]
        if kind == ".":
            env[f"f{k}__"] = char(text(subs))
            code += f".(f{k}__)"
            continue
        items = subs.values_f() if isinstance(subs, Cell) else [subs]
        parts = []
        for j, x in enumerate(items):
            if is_text(x) and text(x) == ":":
                parts.append(":")
            else:
                env[f"i{k}_{j}__"] = x
                parts.append(f"i{k}_{j}__")
        code += ("(" if kind == "()" else "{") + ", ".join(parts) + (")" if kind == "()" else "}")
    if assign:
        env["v__"] = value
        return _eval_with(interp, f"a__{code} = v__;", env, "a__")
    return _eval_with(interp, "a__" + code, env, None)


@builtin("subsref")
def _subsref(interp, args, nargout):
    nargs(args, 2, 2, "subsref")
    return _apply_subs(interp, args[0], args[1], "subsref")


@builtin("subsasgn")
def _subsasgn(interp, args, nargout):
    nargs(args, 3, 3, "subsasgn")
    a = args[0]
    if isinstance(a, Struct):
        a = a.copy()
    elif isinstance(a, np.ndarray):
        a = a.copy()
    elif isinstance(a, Cell):
        a = Cell(a.data.copy())
    return _apply_subs(interp, a, args[1], "subsasgn", args[2], assign=True)
