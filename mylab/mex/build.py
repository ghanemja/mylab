"""The ``mex`` command: compile C/C++ sources into a MEX file with the system compiler."""

from __future__ import annotations

import os
import shlex
import shutil
import subprocess
import sys
import tempfile

from ..errors import MatlabError
from . import IN_BROWSER, INCLUDE_DIR, MEX_SUFFIX, MEXEXT

C_EXTS = (".c",)
CXX_EXTS = (".cpp", ".cc", ".cxx", ".c++", ".C")
LINK_EXTS = (".o", ".obj", ".a", ".lib", ".so", ".dylib", ".dll")
ENTRY_FILE = INCLUDE_DIR / "mylab_mex_entry.c"

USAGE = """\
Usage: mex [options] source1.c [source2.c ...] [object or library files]

  -output NAME   name of the MEX file (default: the first source's name)
  -outdir DIR    folder to write it to (default: the current folder)
  -I<dir>        add an include folder          -D<name>[=value]  define a macro
  -L<dir>        add a library folder           -U<name>          undefine a macro
  -l<name>       link a library
  -O             optimise (the default)         -g                debug build (enables mxAssert)
  -R2018a        interleaved complex API        -R2017b           separate complex API (default)
  -v             show the compiler commands     -silent           print nothing on success
  -n             show the commands without running them
  CFLAGS="..."   CXXFLAGS="..."   LDFLAGS="..."  replace the default flags ($CFLAGS keeps them)

Compiler: $CC / $CXX if set, otherwise cc, gcc or clang (c++, g++ or clang++ for C++).
MEX files are written with the extension ."""


class _Options:
    def __init__(self):
        self.sources: list[str] = []
        self.link_inputs: list[str] = []
        self.includes: list[str] = []
        self.libdirs: list[str] = []
        self.libs: list[str] = []
        self.defines: list[str] = []
        self.undefs: list[str] = []
        self.output: str | None = None
        self.outdir: str | None = None
        self.debug = False
        self.verbose = False
        self.silent = False
        self.dry_run = False
        self.interleaved = False
        self.overrides: dict[str, str] = {}


def _find(names: list[str]) -> str | None:
    for n in names:
        if not n:
            continue
        parts = shlex.split(n) if " " in n.strip() else [n]
        if parts and shutil.which(parts[0]):
            return n
    return None


def find_c_compiler() -> str | None:
    names = ["gcc", "clang", "cl"] if sys.platform == "win32" else ["cc", "gcc", "clang"]
    return _find([os.environ.get("CC", ""), *names])


def find_cxx_compiler() -> str | None:
    names = ["g++", "clang++", "cl"] if sys.platform == "win32" else ["c++", "g++", "clang++"]
    return _find([os.environ.get("CXX", ""), *names])


def _is_msvc(compiler: str) -> bool:
    return os.path.basename(shlex.split(compiler)[0]).lower() in ("cl", "cl.exe")


def _parse(argv: list[str]) -> _Options:
    o = _Options()
    i = 0

    def value(flag: str) -> str:
        nonlocal i
        if i + 1 >= len(argv):
            raise MatlabError(f"mex: {flag} needs a value.", "mylab:mex:badOption")
        i += 1
        return argv[i]

    while i < len(argv):
        a = argv[i]
        low = a.lower()
        if a == "-output":
            o.output = value(a)
        elif a == "-outdir":
            o.outdir = value(a)
        elif a == "-O":
            o.debug = False
        elif a == "-g":
            o.debug = True
        elif a == "-v":
            o.verbose = True
        elif a == "-silent":
            o.silent = True
        elif a == "-n":
            o.dry_run = o.verbose = True
        elif low in ("-r2018a", "-interleaved"):
            o.interleaved = True
        elif low in ("-r2017b", "-largearraydims", "-compatiblearraydims"):
            o.interleaved = False if low == "-r2017b" else o.interleaved
        elif a[:2] in ("-I", "-L", "-l", "-D", "-U"):
            rest = a[2:] or value(a)
            {"-I": o.includes, "-L": o.libdirs, "-l": o.libs, "-U": o.undefs}.get(a[:2], o.defines).append(
                rest.replace("#", "=", 1) if a[:2] == "-D" else rest)
        elif "=" in a and a.split("=", 1)[0] in ("CFLAGS", "CXXFLAGS", "LDFLAGS", "COPTIMFLAGS", "CXXOPTIMFLAGS",
                                                  "LDOPTIMFLAGS", "CDEBUGFLAGS", "CXXDEBUGFLAGS"):
            k, v = a.split("=", 1)
            o.overrides[k] = v.strip().strip('"').strip("'")
        elif a.startswith("-"):
            raise MatlabError(f"mex: unknown or unsupported option '{a}'. Type 'mex -help' for the options mylab "
                              "supports.", "mylab:mex:badOption")
        else:
            ext = os.path.splitext(a)[1]
            if ext in C_EXTS or ext in CXX_EXTS or ext.lower() in (".cpp", ".cc", ".cxx"):
                o.sources.append(a)
            elif ext.lower() in LINK_EXTS:
                o.link_inputs.append(a)
            else:
                raise MatlabError(f"mex: '{a}' is not a C or C++ source file (.c, .cpp, .cc, .cxx) or an object or "
                                  "library file.", "mylab:mex:badFile")
        i += 1
    return o


def _expand(override: str | None, default: list[str], var: str) -> list[str]:
    if override is None:
        return default
    return shlex.split(override.replace("$" + var, " ".join(shlex.quote(x) for x in default)))


def mex_command(interp, argv: list[str], nargout: int):
    if IN_BROWSER:
        raise MatlabError("mex: MEX files cannot be built or run when mylab runs in the browser. Install mylab on "
                          "your computer to use them.", "mylab:mex:browser")
    if not argv or argv[0] in ("-help", "-h", "--help"):
        interp.out(USAGE + MEXEXT + " on this computer.\n")
        return 0.0 if nargout else None
    if argv[0] == "-setup":
        cc, cxx = find_c_compiler(), find_cxx_compiler()
        interp.out(f"C compiler: {cc or 'none found'}\nC++ compiler: {cxx or 'none found'}\n"
                   f"MEX files are written with the extension {MEX_SUFFIX}.\n")
        return 0.0 if nargout else None
    o = _parse(argv)
    if not o.sources:
        raise MatlabError("mex: no C or C++ source file was given.", "mylab:mex:noSource")
    cwd = os.getcwd()
    sources = []
    for s in o.sources:
        p = os.path.abspath(os.path.expanduser(s))
        if not os.path.isfile(p):
            found = next((os.path.join(d, s) for d in interp.path if os.path.isfile(os.path.join(d, s))), None)
            if found is None:
                raise MatlabError(f"mex: cannot find the source file '{s}'.", "mylab:mex:fileNotFound")
            p = os.path.abspath(found)
        sources.append(p)
    link_inputs = []
    for s in o.link_inputs:
        p = os.path.abspath(os.path.expanduser(s))
        if not os.path.isfile(p):
            raise MatlabError(f"mex: cannot find the file '{s}'.", "mylab:mex:fileNotFound")
        link_inputs.append(p)
    cxx_mode = any(os.path.splitext(s)[1] != ".c" for s in sources)
    cc = find_c_compiler()
    cxx = find_cxx_compiler() if cxx_mode else None
    if cc is None and not cxx_mode:
        raise MatlabError("mex: no C compiler was found. Install gcc or clang (on Windows: MinGW-w64 gcc, or run "
                          "mylab from a Visual Studio developer prompt), or set the CC environment variable.",
                          "mylab:mex:noCompiler")
    if cxx_mode and cxx is None:
        raise MatlabError("mex: no C++ compiler was found for the C++ sources. Install g++ or clang++, or set the "
                          "CXX environment variable.", "mylab:mex:noCompiler")
    if cc is None:
        cc = cxx
    name = o.output or os.path.splitext(os.path.basename(sources[0]))[0]
    if name.endswith(MEX_SUFFIX):
        name = name[: -len(MEX_SUFFIX)]
    outdir = os.path.abspath(os.path.expanduser(o.outdir)) if o.outdir else cwd
    if os.path.dirname(name):
        outdir = os.path.join(outdir, os.path.dirname(name))
        name = os.path.basename(name)
    target = os.path.join(outdir, name + MEX_SUFFIX)
    if not o.dry_run:
        os.makedirs(outdir, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="mylab-mexbuild-") as tmp:
        built = os.path.join(tmp, name + MEX_SUFFIX)
        commands = _commands(o, cc, cxx, cxx_mode, sources, link_inputs, tmp, built)
        tool = os.path.basename(shlex.split(cxx if cxx_mode else cc)[0])
        if not o.silent:
            interp.out(f"Building {name}{MEX_SUFFIX} with {tool}.\n")
        for cmd in commands:
            if o.verbose:
                interp.out(" ".join(shlex.quote(c) for c in cmd) + "\n")
            if o.dry_run:
                continue
            try:
                proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                      cwd=cwd)
            except OSError as e:
                raise MatlabError(f"mex: could not run the compiler '{cmd[0]}': {e.strerror or e}.",
                                  "mylab:mex:noCompiler")
            text = proc.stdout.decode("utf-8", "replace")
            if o.verbose and text:
                interp.out(text if text.endswith("\n") else text + "\n")
            if proc.returncode != 0:
                raise MatlabError(f"mex: building {name}{MEX_SUFFIX} failed.\n{text.rstrip()}",
                                  "mylab:mex:buildFailed")
            if text and not o.verbose and not o.silent:
                interp.out(text if text.endswith("\n") else text + "\n")  # compiler warnings
        if o.dry_run:
            return 0.0 if nargout else None
        from . import runtime_for
        rt = runtime_for(interp, create=False)
        if rt is not None:
            rt.unload_path(target)   # the new build replaces the loaded one (as a fresh load)
        shutil.copyfile(built, target + ".tmp")
        os.replace(target + ".tmp", target)
    interp.lookup_changed()
    if not o.silent:
        interp.out(f"MEX file {name}{MEX_SUFFIX} built.\n")
    return 0.0 if nargout else None


def _commands(o: _Options, cc: str, cxx: str | None, cxx_mode: bool, sources: list[str], link_inputs: list[str],
              tmp: str, out: str) -> list[list[str]]:
    msvc = _is_msvc(cxx if cxx_mode else cc)
    defines = ["MATLAB_MEX_FILE", "MYLAB_MEX_BUILD", *o.defines]
    if o.interleaved:
        defines.append("MX_HAS_INTERLEAVED_COMPLEX=1")
    if o.debug:
        defines.append("MYLAB_MEX_DEBUG")
    else:
        defines.append("NDEBUG")
    includes = [os.path.abspath(os.path.expanduser(d)) for d in o.includes] + [str(INCLUDE_DIR)]
    objs: list[str] = []
    cmds: list[list[str]] = []
    if msvc:
        common = ["/nologo", "/c"] + (["/Zi", "/Od"] if o.debug else ["/O2"])
        common += [f"/I{d}" for d in includes] + [f"/D{d}" for d in defines] + [f"/U{u}" for u in o.undefs]
        for k, src in enumerate([*sources, str(ENTRY_FILE)]):
            is_cxx = os.path.splitext(src)[1] != ".c" or (src == str(ENTRY_FILE) and cxx_mode)
            comp = cxx if is_cxx else cc
            obj = os.path.join(tmp, f"{k}_{os.path.splitext(os.path.basename(src))[0]}.obj")
            flags = _expand(o.overrides.get("CXXFLAGS" if is_cxx else "CFLAGS"), ["/EHsc"] if cxx_mode else [],
                            "CXXFLAGS" if is_cxx else "CFLAGS")
            cmds.append([*shlex.split(comp), *common, *flags, *(["/TP"] if is_cxx else []), src, f"/Fo{obj}"])
            objs.append(obj)
        link = shlex.split(cxx if cxx_mode else cc)
        cmds.append([*link, "/nologo", "/LD", *objs, *link_inputs, f"/Fe{out}", "/link",
                     *[f"/LIBPATH:{d}" for d in o.libdirs], *[f"{lib}.lib" for lib in o.libs],
                     *_expand(o.overrides.get("LDFLAGS"), [], "LDFLAGS")])
        return cmds
    pic = [] if sys.platform == "win32" else ["-fPIC"]
    optim = ["-g", "-O0"] if o.debug else ["-O2"]
    common = [*optim, *pic, "-fvisibility=hidden", *[f"-I{d}" for d in includes], *[f"-D{d}" for d in defines],
              *[f"-U{u}" for u in o.undefs]]
    for k, src in enumerate([*sources, str(ENTRY_FILE)]):
        is_entry = src == str(ENTRY_FILE)
        is_cxx = (os.path.splitext(src)[1] != ".c") or (is_entry and cxx_mode)
        comp = cxx if is_cxx else cc
        var = "CXXFLAGS" if is_cxx else "CFLAGS"
        default = ["-fexceptions"] if (cxx_mode and not is_cxx) else []
        flags = _expand(o.overrides.get(var), default, var)
        obj = os.path.join(tmp, f"{k}_{os.path.splitext(os.path.basename(src))[0]}.o")
        lang = ["-x", "c++"] if (is_entry and cxx_mode) else []
        cmds.append([*shlex.split(comp), *common, *flags, "-c", *lang, src, "-o", obj])
        objs.append(obj)
    link = shlex.split(cxx if cxx_mode else cc)
    ldflags = _expand(o.overrides.get("LDFLAGS"), [], "LDFLAGS")
    libs = [f"-l{lib}" for lib in o.libs]
    if sys.platform.startswith("linux"):
        libs.append("-lm")
    # MinGW links its C++ and gcc runtimes as DLLs that are usually not on the loader's search path, so a
    # MEX file built with g++ would fail to load. Link those runtimes into the MEX file instead.
    static_rt = []
    tool = os.path.basename(link[0]).lower()
    if sys.platform == "win32" and ("g++" in tool or "gcc" in tool):
        static_rt = ["-static-libgcc", "-static-libstdc++", "-static"] if cxx_mode else ["-static-libgcc"]
    cmds.append([*link, "-shared", *optim, *objs, *link_inputs, "-o", out, *[f"-L{d}" for d in o.libdirs], *libs,
                 *static_rt, *ldflags])
    return cmds
