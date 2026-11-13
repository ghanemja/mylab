"""mylab Blocks from code: build, change and simulate block-diagram models."""

from __future__ import annotations

import os
from typing import Any

import numpy as np

from ..errors import MatlabError
from ..values import Cell, FunctionHandle, MString, Struct, char, is_char, to_str
from ._registry import builtin
from ._util import is_text, nargs, text

EXTENSIONS = (".mlb.json", ".slx", ".json")


# -- state ---------------------------------------------------------------------------------------

def _state(interp) -> dict:
    st = interp.state.get("blockdiagram")
    if st is None:
        st = interp.state["blockdiagram"] = {"models": {}, "current": None, "block": None, "handles": {}, "ids": {},
                                             "next": 1000.0, "options": {}, "opened": []}
    return st


def _handle(interp, key: tuple) -> float:
    st = _state(interp)
    if key not in st["ids"]:
        st["next"] += 1.0
        st["ids"][key] = st["next"]
        st["handles"][st["next"]] = key
    return st["ids"][key]


def _blocks():
    from .. import blocks
    return blocks


def _wrap(fn):
    """Turn model and simulation errors into MATLAB errors."""
    def run(interp, args, nargout):
        b = _blocks()
        try:
            return fn(interp, args, nargout)
        except b.SimError as exc:
            err = MatlabError(exc.message, "mylab:blocks:simulation")
            err.block_path = exc.path  # type: ignore[attr-defined]
            err.diagnostics = exc.diagnostics  # type: ignore[attr-defined]
            raise err from None
        except b.ModelError as exc:
            raise MatlabError(str(exc), "mylab:blocks:model") from None
    run.__name__ = fn.__name__
    run.__doc__ = fn.__doc__
    return run


def _find_file(interp, name: str) -> str | None:
    folders = [os.getcwd(), *interp.path]
    for d in folders:
        direct = os.path.join(d, name)
        if os.path.isfile(direct) and name.lower().endswith(EXTENSIONS + (".mdl",)):
            return direct
        for ext in EXTENSIONS:
            p = os.path.join(d, name + ext)
            if os.path.isfile(p):
                return p
    return None


def _model_name(name: str) -> str:
    base = os.path.basename(name)
    for ext in EXTENSIONS + (".mdl",):
        if base.lower().endswith(ext):
            return base[:-len(ext)]
    return base


def _model(interp, name: str, load: bool = True):
    st = _state(interp)
    key = _model_name(name)
    m = st["models"].get(key)
    if m is None and load:
        path = _find_file(interp, name)
        if path is None:
            raise MatlabError(f"There is no model named '{key}'. Create it with new_system('{key}'), or put "
                              f"{key}.mlb.json in the current folder.", "mylab:blocks:noModel")
        m = _blocks().load_model(path)
        m.name = key
        st["models"][key] = m
        for note in m.notes:
            interp.warn(note, "mylab:blocks:import")
    if m is None:
        raise MatlabError(f"The model '{key}' is not open.", "mylab:blocks:noModel")
    return m


def _target(interp, obj: Any):
    """(kind, model, parts...) for a path or handle. kind is 'model', 'block', 'system', 'port' or 'line'."""
    st = _state(interp)
    if not is_text(obj):
        a = np.asarray(obj)
        if a.size == 1 and float(a.flat[0]) in st["handles"]:
            key = st["handles"][float(a.flat[0])]
            m = _model(interp, key[1])
            return (key[0], m) + tuple(key[2:])
        raise MatlabError("This is not a handle of a model, block or port.", "mylab:blocks:badHandle")
    path = text(obj)
    parts = _blocks().model.split_path(path)
    if not parts:
        raise MatlabError("A model or block name is needed.", "mylab:blocks:badPath")
    m = _model(interp, parts[0])
    if len(parts) == 1:
        return ("model", m)
    rel = "/".join(p.replace("/", "//") for p in parts[1:])
    return ("block", m, rel)


def _system(interp, obj: Any):
    """(model, system, relative path) for a model or subsystem path."""
    t = _target(interp, obj)
    m = t[1]
    if t[0] == "model":
        return m, m.root, ""
    if t[0] == "block":
        _, b = m.find(t[2])
        if b.system is None:
            raise MatlabError(f"'{m.name}/{t[2]}' is not a subsystem.", "mylab:blocks:notSubsystem")
        return m, b.system, t[2]
    raise MatlabError("A model or subsystem is needed here.", "mylab:blocks:badPath")


def _as_text(v: Any) -> str:
    """A parameter value as the text stored in the model."""
    if is_text(v):
        return text(v)
    if isinstance(v, FunctionHandle):
        raise MatlabError("Give a function handle as text, e.g. '@(u) u.^2'.")
    a = np.asarray(v)
    if a.dtype == bool:
        a = a.astype(float)
    if a.size == 1:
        return _num(a.flat[0])
    if a.ndim <= 2:
        a2 = a.reshape(1, -1) if a.ndim < 2 else a
        return "[" + ";".join(" ".join(_num(x) for x in row) for row in a2) + "]"
    raise MatlabError("Parameter values must be text, numbers or matrices.")


def _num(x: Any) -> str:
    if np.iscomplexobj(x) and complex(x).imag != 0:
        c = complex(x)
        return f"{_num(c.real)}{'+' if c.imag >= 0 else '-'}{_num(abs(c.imag))}i"
    f = float(np.real(x))
    if f == int(f) and abs(f) < 1e15:
        return str(int(f))
    return repr(f)


def _pairs(args: list, what: str) -> list[tuple[str, Any]]:
    if len(args) % 2:
        raise MatlabError(f"{what}: parameters come in name, value pairs.")
    return [(text(args[i], "parameter name"), args[i + 1]) for i in range(0, len(args), 2)]


# -- models --------------------------------------------------------------------------------------

@builtin("new_system")
@_wrap
def _new_system(interp, args, nargout):
    st = _state(interp)
    name = text(args[0]) if args else "untitled"
    if not name.replace("_", "a").isalnum() or name[0].isdigit():
        raise MatlabError(f"'{name}' is not a valid model name. Use letters, digits and underscores, starting with a letter.")
    if name in st["models"]:
        raise MatlabError(f"A model named '{name}' is already open. Close it first with close_system('{name}', 0).")
    m = _blocks().Model(name)
    m.dirty = True
    st["models"][name] = m
    st["current"] = name
    return _handle(interp, ("model", name)) if nargout else None


@builtin("load_system")
@_wrap
def _load_system(interp, args, nargout):
    nargs(args, 1, 1, "load_system")
    m = _model(interp, text(args[0]))
    _state(interp)["current"] = m.name
    return _handle(interp, ("model", m.name)) if nargout else None


@builtin("open_system")
@_wrap
def _open_system(interp, args, nargout):
    nargs(args, 1, None, "open_system")
    st = _state(interp)
    t = _target(interp, args[0])
    st["current"] = t[1].name
    st["opened"].append(t[1].name + ("/" + t[2] if t[0] == "block" else ""))
    if not interp.state.get("blocks_ui") and not st.get("noted"):
        st["noted"] = True
        interp.out("open_system: the diagram editor is part of the mylab web app. The model is loaded and can be "
                   "changed and simulated from code.\n")
    return None


@builtin("save_system")
@_wrap
def _save_system(interp, args, nargout):
    st = _state(interp)
    m = _model(interp, text(args[0])) if args else _model(interp, st["current"] or "")
    if len(args) > 1 and is_text(args[1]):
        target = text(args[1])
        new_name = _model_name(target)
        if not target.lower().endswith(".json"):
            target += ".mlb.json"
        if target.lower().endswith((".slx", ".mdl")):
            raise MatlabError("mylab saves models as .mlb.json files; it cannot write .slx or .mdl.")
        if new_name != m.name:
            st["models"].pop(m.name, None)
            m.name = new_name
            st["models"][new_name] = m
            st["current"] = new_name
    else:
        target = m.file if m.file and m.file.lower().endswith(".json") else m.name + ".mlb.json"
    _blocks().save_model(m, target)
    return char(os.path.abspath(target)) if nargout else None


@builtin("close_system", "bdclose")
@_wrap
def _close_system(interp, args, nargout):
    st = _state(interp)
    names = [text(args[0])] if args and is_text(args[0]) else ([st["current"]] if st["current"] else [])
    if names == ["all"]:
        names = list(st["models"])
    for name in names:
        key = _model_name(name.split("/")[0])
        m = st["models"].get(key)
        if m is None:
            continue
        if len(args) > 1 and not is_text(args[1]) and float(np.asarray(args[1]).flat[0]) == 1:
            _blocks().save_model(m, m.file if m.file and m.file.endswith(".json") else key + ".mlb.json")
        del st["models"][key]
        if st["current"] == key:
            st["current"] = next(iter(st["models"]), None)
    return None


@builtin("bdIsLoaded")
def _bd_is_loaded(interp, args, nargout):
    nargs(args, 1, 1, "bdIsLoaded")
    return np.array([[_model_name(text(args[0])) in _state(interp)["models"]]])


@builtin("bdroot")
@_wrap
def _bdroot(interp, args, nargout):
    st = _state(interp)
    if args:
        return char(_target(interp, args[0])[1].name)
    return char(st["current"] or "")


@builtin("gcs")
def _gcs(interp, args, nargout):
    return char(_state(interp)["current"] or "")


@builtin("gcb")
def _gcb(interp, args, nargout):
    return char(_state(interp)["block"] or "")


# -- blocks and lines ----------------------------------------------------------------------------

def _set_block_param(m, system, b, name: str, value: Any) -> None:
    low = name.lower()
    if low == "position":
        p = np.ravel(np.asarray(value if not is_text(value) else [float(x) for x in text(value).strip("[]").replace(",", " ").split()],
                                float))
        if p.size != 4:
            raise MatlabError("Position needs four numbers: [left top right bottom].")
        b.pos = [float(p[0]), float(p[1]), float(p[2] - p[0]), float(p[3] - p[1])]
    elif low == "name":
        new = text(value)
        if system.block(new) is not None and new != b.name:
            raise MatlabError(f"There is already a block named '{new}' here.")
        for ln in system.lines:
            if ln.src[0] == b.name:
                ln.src = (new, ln.src[1])
            if ln.dst[0] == b.name:
                ln.dst = (new, ln.dst[1])
        b.name = new
    elif low == "orientation":
        b.rot = {"right": 0, "down": 1, "left": 2, "up": 3}.get(text(value).lower(), 0)
    elif low == "blockmirror":
        b.flip = text(value).lower() == "on"
    else:
        cls = _blocks().find_type(b.type)
        known = {p.name.lower(): p.name for p in (cls.params if cls else [])}
        if low in known:
            b.params[known[low]] = _as_text(value)
        elif low in ("maskvalues", "mask"):
            raise MatlabError("Set mask parameters by name, e.g. set_param(block, 'm', '2').")
        elif b.system is not None and isinstance(b.extra.get("mask"), dict) and name in b.extra["mask"]:
            b.extra["mask"][name] = _as_text(value)
        elif low in ("showname", "namelocation", "backgroundcolor", "foregroundcolor", "fontsize", "zorder", "priority",
                     "tag", "description", "sampletime", "outdatatypestr", "linkstatus", "commented"):
            b.extra.setdefault("other", {})[name] = _as_text(value)
        else:
            have = ", ".join(p.name for p in (cls.params if cls else [])) or "none"
            raise MatlabError(f"The block '{b.name}' ({b.type}) has no parameter '{name}'. Its parameters are: {have}.",
                              "mylab:blocks:noParam")
    m.dirty = True


@builtin("add_block")
@_wrap
def _add_block(interp, args, nargout):
    nargs(args, 2, None, "add_block")
    B = _blocks()
    src, dest = text(args[0]), text(args[1])
    opts = _pairs(list(args[2:]), "add_block")
    parts = B.model.split_path(dest)
    if len(parts) < 2:
        raise MatlabError("The destination must be 'model/BlockName' (or 'model/Subsystem/BlockName').")
    m = _model(interp, parts[0])
    system = m.system_at(parts[1:-1])
    name = parts[-1]
    unique = any(k.lower() == "makenameunique" and is_text(v) and text(v).lower() == "on" for k, v in opts)
    opts = [(k, v) for k, v in opts if k.lower() not in ("makenameunique", "copyoption")]
    if system.block(name) is not None:
        if not unique:
            raise MatlabError(f"There is already a block named '{name}' in '{'/'.join(parts[:-1])}'.", "mylab:blocks:exists")
        name = system.unique_name(name)
    cls = B.find_type(src)
    src_parts = B.model.split_path(src)
    st = _state(interp)
    if cls is None and len(src_parts) >= 2 and _model_name(src_parts[0]) in st["models"]:
        _, other = _model(interp, src_parts[0]).find("/".join(src_parts[1:]))  # copy an existing block
        b = other.copy()
        b.name = name
    elif cls is None:
        raise MatlabError(f"There is no block '{src}' in the mylab Blocks library. Block names are listed in docs/BLOCKS.md "
                          "(for example 'blocks/Gain' or 'simulink/Continuous/Integrator').", "mylab:blocks:noBlock")
    else:
        b = B.Block(name, cls.type, {})
        if cls.type == "Subsystem":
            b.system = B.System()
        if cls.type in ("Inport", "Outport"):
            same = [x for x in system.blocks if getattr(B.find_type(x.type), "type", "") == cls.type]
            b.params["Port"] = str(len(same) + 1)
        k = len(system.blocks)
        b.pos = [60 + 110 * (k % 6), 60 + 90 * (k // 6), 0, 0]
    system.blocks.append(b)
    for k, v in opts:
        _set_block_param(m, system, b, k, v)
    m.dirty = True
    path = "/".join(parts[:-1] + [name])
    st["block"] = path
    st["current"] = m.name
    return _handle(interp, ("block", m.name, "/".join(parts[1:-1] + [name]))) if nargout else None


def _endpoint(interp, m, system, sys_rel: str, spec: Any, out: bool) -> tuple[str, Any]:
    B = _blocks()
    if is_text(spec):
        s = text(spec)
        name, _, port = s.rpartition("/")
        if not name:
            raise MatlabError(f"'{s}' must be 'BlockName/port', for example 'Gain/1'.")
        p: Any = int(port) if port.isdigit() else port.lower()
        if not isinstance(p, int) and p not in ("enable", "trigger"):
            raise MatlabError(f"'{s}': the port must be a number, 'Enable' or 'Trigger'.")
    else:
        t = _target(interp, spec)
        if t[0] != "port" or t[1] is not m:
            raise MatlabError("A port handle from get_param(block, 'PortHandles') is needed here.")
        _, rel, kind, p = t[1:]
        if (kind == "out") != out:
            raise MatlabError("add_line(system, outputPort, inputPort): the ports are the wrong way round.")
        parent, _, name = rel.rpartition("/")
        if parent != sys_rel:
            raise MatlabError("The port belongs to a block in another system.")
    b = system.block(name)
    if b is None:
        raise MatlabError(f"There is no block '{name}' in this system.", "mylab:blocks:noBlock")
    if isinstance(p, int):
        n = B.engine.port_count(b, "out" if out else "in")
        if not 1 <= p <= n:
            raise MatlabError(f"'{name}' has {n} {'output' if out else 'input'} port(s); there is no port {p}.",
                              "mylab:blocks:noPort")
    return name, p


@builtin("add_line")
@_wrap
def _add_line(interp, args, nargout):
    nargs(args, 3, None, "add_line")
    B = _blocks()
    m, system, rel = _system(interp, args[0])
    if not is_text(args[1]) and np.asarray(args[1]).ndim == 2 and np.asarray(args[1]).shape[1] == 2 \
            and np.asarray(args[1]).shape[0] > 1 and len(args) == 2:
        raise MatlabError("add_line with a list of points is not supported; name the ports instead.")
    _pairs(list(args[3:]), "add_line")  # 'autorouting', 'on' and friends are accepted
    src = _endpoint(interp, m, system, rel, args[1], True)
    dsts = args[2].values_f() if isinstance(args[2], Cell) else [args[2]]
    h = None
    for d in dsts:
        dst = _endpoint(interp, m, system, rel, d, False)
        if any(ln.dst == dst for ln in system.lines):
            raise MatlabError(f"Input {dst[1]} of '{dst[0]}' already has a line.", "mylab:blocks:connected")
        system.lines.append(B.Line(src, dst))
        h = _handle(interp, ("line", m.name, rel, src[0], src[1], dst[0], dst[1]))
    m.dirty = True
    return h if nargout else None


@builtin("delete_line")
@_wrap
def _delete_line(interp, args, nargout):
    nargs(args, 1, 3, "delete_line")
    if len(args) == 1:
        t = _target(interp, args[0])
        if t[0] != "line":
            raise MatlabError("delete_line needs a line handle, or (system, 'Block/port', 'Block/port').")
        m, rel, src, dst = t[1], t[2], (t[3], t[4]), (t[5], t[6])
        system = m.system_at(_blocks().model.split_path(rel))
    else:
        nargs(args, 3, 3, "delete_line")
        m, system, rel = _system(interp, args[0])
        src = _endpoint(interp, m, system, rel, args[1], True)
        dst = _endpoint(interp, m, system, rel, args[2], False)
    before = len(system.lines)
    system.lines = [ln for ln in system.lines if not (ln.src == src and ln.dst == dst)]
    if len(system.lines) == before:
        raise MatlabError(f"There is no line from '{src[0]}/{src[1]}' to '{dst[0]}/{dst[1]}'.", "mylab:blocks:noLine")
    m.dirty = True
    return None


@builtin("delete_block")
@_wrap
def _delete_block(interp, args, nargout):
    nargs(args, 1, 1, "delete_block")
    t = _target(interp, args[0])
    if t[0] != "block":
        raise MatlabError("delete_block needs a block path such as 'model/Gain'.")
    m = t[1]
    system, b = m.find(t[2])
    system.blocks.remove(b)
    system.lines = [ln for ln in system.lines if ln.src[0] != b.name and ln.dst[0] != b.name]
    m.dirty = True
    return None


@builtin("set_param")
@_wrap
def _set_param(interp, args, nargout):
    nargs(args, 3, None, "set_param")
    t = _target(interp, args[0])
    m = t[1]
    for name, value in _pairs(list(args[1:]), "set_param"):
        if t[0] == "model":
            if name.lower() == "simulationcommand":
                if text(value).lower() == "start":
                    _run_sim(interp, m, {}, 0)
                continue
            known = {k.lower(): k for k in m.config}
            alias = {"solvername": "Solver", "solvertype": None, "zerocrosscontrol": None}
            key = known.get(name.lower(), alias.get(name.lower(), name))
            if key is not None:
                m.config[key] = _as_text(value)
            m.dirty = True
        elif t[0] == "block":
            system, b = m.find(t[2])
            _set_block_param(m, system, b, name, value)
        elif t[0] == "line":
            system = m.system_at(_blocks().model.split_path(t[2]))
            for ln in system.lines:
                if ln.src == (t[3], t[4]) and (ln.dst == (t[5], t[6]) or name.lower() == "name"):
                    if name.lower() == "name":
                        ln.name = text(value)
                    elif name.lower() in ("datalogging", "log"):
                        ln.log = _as_text(value).lower() in ("on", "1")
            m.dirty = True
        else:
            raise MatlabError("set_param works on models, blocks and lines.")
    return None


def _cellstr(items: list[str], column: bool = True) -> Cell:
    return Cell.from_list([char(s) for s in items], column=column)


@builtin("get_param")
@_wrap
def _get_param(interp, args, nargout):
    nargs(args, 2, 2, "get_param")
    B = _blocks()
    t = _target(interp, args[0])
    m = t[1]
    name = text(args[1])
    low = name.lower()
    if t[0] == "model":
        if low == "name":
            return char(m.name)
        if low == "blocks":
            return _cellstr([b.name for b in m.root.blocks])
        if low == "handle":
            return _handle(interp, ("model", m.name))
        if low == "dirty":
            return char("on" if m.dirty else "off")
        if low == "filename":
            return char(os.path.abspath(m.file) if m.file else "")
        if low == "type":
            return char("block_diagram")
        if low == "lines":
            return _lines_struct(m.root)
        for k, v in m.config.items():
            if k.lower() == low or (low == "solvername" and k == "Solver"):
                return char(v)
        raise MatlabError(f"The model has no setting '{name}'. Settings: {', '.join(m.config)}.", "mylab:blocks:noParam")
    if t[0] == "port":
        _, _, rel, kind, p = t
        return {"parent": char(f"{m.name}/{rel}"), "portnumber": float(p) if isinstance(p, int) else char(str(p)),
                "porttype": char("outport" if kind == "out" else "inport")}.get(low) if low in (
                    "parent", "portnumber", "porttype") else _no_param(name)
    if t[0] != "block":
        raise MatlabError("get_param works on models, blocks and ports.")
    system, b = m.find(t[2])
    cls = B.find_type(b.type)
    parent = m.name + ("/" + t[2].rpartition("/")[0] if "/" in t[2] else "")
    if low == "name":
        return char(b.name)
    if low == "blocktype":
        return char({"Subsystem": "SubSystem"}.get(b.type, b.type))
    if low == "parent":
        return char(parent)
    if low == "handle":
        return _handle(interp, ("block", m.name, t[2]))
    if low == "type":
        return char("block")
    if low == "position":
        x, y, w, h = (b.pos + [0, 0, 0, 0])[:4]
        return np.array([[x, y, x + w, y + h]], dtype=float)
    if low == "orientation":
        return char(["right", "down", "left", "up"][b.rot % 4])
    if low == "blocks" and b.system is not None:
        return _cellstr([x.name for x in b.system.blocks])
    if low == "lines" and b.system is not None:
        return _lines_struct(b.system)
    if low == "ports":
        return np.array([[B.engine.port_count(b, "in"), B.engine.port_count(b, "out")]], dtype=float)
    if low == "porthandles":
        ins = [_handle(interp, ("port", m.name, t[2], "in", k + 1)) for k in range(B.engine.port_count(b, "in"))]
        outs = [_handle(interp, ("port", m.name, t[2], "out", k + 1)) for k in range(B.engine.port_count(b, "out"))]
        extra = {}
        for which, kind in (("Enable", "EnablePort"), ("Trigger", "TriggerPort")):
            has = b.system is not None and any(getattr(B.find_type(x.type), "type", "") == kind for x in b.system.blocks)
            extra[which] = np.array([[_handle(interp, ("port", m.name, t[2], "in", which.lower()))]]) if has else np.zeros((1, 0))
        return Struct.scalar({"Inport": np.array([ins], dtype=float).reshape(1, -1),
                              "Outport": np.array([outs], dtype=float).reshape(1, -1), **extra})
    if low == "dialogparameters":
        return Struct.scalar({p.name: char(p.label) for p in (cls.params if cls else [])})
    for p in (cls.params if cls else []):
        if p.name.lower() == low:
            return char(b.params.get(p.name, p.default))
    if b.system is not None and isinstance(b.extra.get("mask"), dict) and name in b.extra["mask"]:
        return char(str(b.extra["mask"][name]))
    other = {k.lower(): v for k, v in b.extra.get("other", {}).items()}
    if low in other:
        return char(other[low])
    have = ", ".join(p.name for p in (cls.params if cls else [])) or "none"
    raise MatlabError(f"The block '{b.name}' ({b.type}) has no parameter '{name}'. Its parameters are: {have}.",
                      "mylab:blocks:noParam")


def _no_param(name: str):
    raise MatlabError(f"There is no parameter '{name}' here.", "mylab:blocks:noParam")


def _lines_struct(system) -> Struct:
    data = np.empty((len(system.lines), 1), dtype=object)
    for i, ln in enumerate(system.lines):
        data[i, 0] = {"SrcBlock": char(ln.src[0]), "SrcPort": char(str(ln.src[1])), "DstBlock": char(ln.dst[0]),
                      "DstPort": char(str(ln.dst[1])), "Name": char(ln.name)}
    return Struct(data, ["SrcBlock", "SrcPort", "DstBlock", "DstPort", "Name"])


@builtin("find_system")
@_wrap
def _find_system(interp, args, nargout):
    B = _blocks()
    st = _state(interp)
    args = list(args)
    if not args or (is_text(args[0]) and text(args[0]).lower() in ("searchdepth", "type", "blocktype", "name", "regexp",
                                                                    "lookundermasks", "followlinks")):
        roots = list(st["models"])
    else:
        roots = [text(args.pop(0))] if is_text(args[0]) else [_target(interp, args.pop(0))[1].name]
    depth = None
    filters: list[tuple[str, str]] = []
    want_type = None
    for k, v in _pairs(args, "find_system"):
        low = k.lower()
        if low == "searchdepth":
            depth = int(np.asarray(v).flat[0]) if not is_text(v) else int(text(v))
        elif low == "type":
            want_type = text(v).lower()
        elif low in ("lookundermasks", "followlinks", "regexp", "casesensitive", "findall", "firstresultonly",
                     "matchfilter", "variants", "includecommented"):
            continue
        else:
            filters.append((low, _as_text(v)))
    found: list[str] = []
    for root in roots:
        parts = B.model.split_path(root)
        m = _model(interp, parts[0])
        system = m.system_at(parts[1:])
        prefix = "/".join([m.name] + parts[1:])
        if want_type in (None, "block_diagram") and len(parts) == 1 and not filters:
            found.append(m.name)
        if want_type == "block_diagram":
            continue
        for rel, b, _ in system.walk():
            level = rel.count("/") + 1
            if depth is not None and level > depth:
                continue
            cls = B.find_type(b.type)
            ok = True
            for key, val in filters:
                if key == "blocktype":
                    other = B.find_type(val)
                    ok = ok and (b.type == val or (other is not None and cls is other))
                elif key == "name":
                    ok = ok and b.name == val
                elif key == "parent":
                    ok = ok and (prefix + ("/" + rel.rpartition("/")[0] if "/" in rel else "")) == val
                else:
                    params = {p.name.lower(): b.params.get(p.name, p.default) for p in (cls.params if cls else [])}
                    ok = ok and params.get(key) == val
            if ok:
                found.append(f"{prefix}/{rel}")
    return _cellstr(found)


# -- simulation ----------------------------------------------------------------------------------

def _timeseries(t: np.ndarray, data: np.ndarray) -> Struct:
    return Struct.scalar({"Time": t.reshape(-1, 1), "Data": data})


def _field(name: str, used: set) -> str:
    import re
    f = re.sub(r"\W", "_", name)
    if not f or not f[0].isalpha():
        f = "x" + f
    base, k = f, 1
    while f in used:
        k += 1
        f = f"{base}_{k}"
    used.add(f)
    return f


def result_struct(res) -> tuple[Struct, dict]:
    """The object sim returns, and the variables that To Workspace blocks define."""
    out: dict[str, Any] = {"tout": res.t.reshape(-1, 1), "yout": res.y}
    if res.x.size or res.state_names:
        out["xout"] = res.x if res.x.size else np.zeros((res.t.size, 0))
    variables: dict[str, Any] = {}
    for name, w in res.workspace.items():
        fmt = w["format"]
        if fmt == "array":
            v: Any = w["data"]
        elif fmt.startswith("structure"):
            sigs = Struct.scalar({"values": w["data"], "dimensions": float(w["data"].shape[1]) if w["data"].ndim > 1 else 1.0,
                                  "label": char("")})
            v = Struct.scalar({"time": w["t"].reshape(-1, 1) if fmt == "structurewithtime" else np.zeros((0, 0)),
                               "signals": sigs, "blockName": char(w["path"])})
        else:
            v = _timeseries(w["t"], w["data"])
        variables[name] = v
        out[name] = v
    if res.logs:
        out["logsout"] = Struct.scalar({n: _timeseries(d["t"], d["data"]) for n, d in res.logs.items()})
    if res.scopes:
        used: set = set()
        scopes = {}
        for path, s in res.scopes.items():
            cols = [d["data"] for d in s["signals"]] or [np.zeros((s["t"].size, 0))]
            scopes[_field(path.split("/")[-1], used)] = _timeseries(s["t"], np.hstack(cols))
        out["scopes"] = Struct.scalar(scopes)
    if res.displays:
        used = set()
        out["displays"] = Struct.scalar({_field(p.split("/")[-1], used): np.atleast_2d(v) for p, v in res.displays.items()})
    info = {"Solver": char(res.solver), "Steps": float(res.steps), "StoppedBy": char(res.stopped_by or "")}
    out["SimulationMetadata"] = Struct.scalar(info)
    return Struct.scalar(out), variables


def _sim_config(interp, m, args: list) -> tuple[dict, dict]:
    config: dict[str, str] = dict(_state(interp)["options"])
    extra: dict[str, Any] = {}
    args = list(args)
    if args and not is_text(args[0]) and not isinstance(args[0], Struct):
        span = np.ravel(np.asarray(args.pop(0), float))
        if span.size == 1:
            config["StopTime"] = repr(float(span[0]))
        elif span.size >= 2:
            config["StartTime"], config["StopTime"] = repr(float(span[0])), repr(float(span[-1]))
    pairs: list[tuple[str, Any]] = []
    while args and isinstance(args[0], Struct):
        s = args.pop(0)
        pairs += [(f, s.get_field(0, f)) for f in s.fields]
    pairs += _pairs(args, "sim")
    known = {k.lower(): k for k in m.config}
    for k, v in pairs:
        low = k.lower()
        if low == "externalinput" and not is_text(v):
            extra["external_input"] = np.asarray(v, float)
        elif low == "externalinput":
            extra["external_input"] = np.asarray(interp.eval_expression(text(v)), float)
        elif low in ("savestate", "saveoutput", "savetime", "saveformat", "srcworkspace", "returnworkspaceoutputs",
                     "simulationmode", "loadexternalinput", "outputsavename", "statesavename", "timesavename",
                     "signallogging", "limitdatapoints", "debug", "trace", "dstworkspace", "fastrestart"):
            if low == "savestate":
                extra["keep_states"] = _as_text(v).lower() == "on"
        elif low in ("solvername", "solver"):
            config["Solver"] = _as_text(v)
        elif isinstance(v, np.ndarray) and v.size == 0 and not is_char(v):
            continue
        else:
            config[known.get(low, k)] = _as_text(v)
    return config, extra


def _run_sim(interp, m, config: dict, nargout: int, **extra):
    B = _blocks()
    hook = interp.state.get("blocks_on_step")
    res = B.simulate(m, interp, interp.frame, config, on_step=hook, **extra)
    for w in dict.fromkeys(res.warnings):
        interp.warn(w, "mylab:blocks:unconnected")
    return res


@builtin("sim")
@_wrap
def _sim(interp, args, nargout):
    nargs(args, 1, None, "sim")
    t = _target(interp, args[0])
    if t[0] != "model":
        raise MatlabError("sim needs the name of a model, e.g. sim('mymodel').")
    m = t[1]
    config, extra = _sim_config(interp, m, list(args[1:]))
    if nargout > 1:
        extra["keep_states"] = True
    res = _run_sim(interp, m, config, nargout, **extra)
    interp.state["blocks_last_result"] = res
    if nargout > 1:
        return (res.t.reshape(-1, 1), res.x if res.x.size else np.zeros((res.t.size, 0)), res.y)[:nargout]
    out, variables = result_struct(res)
    if nargout == 1:
        return out
    frame = interp.frame
    frame.set("out", out)
    frame.set("tout", res.t.reshape(-1, 1))
    frame.set("yout", res.y)
    for name, v in variables.items():
        frame.set(name, v)
    return None


@builtin("simset")
def _simset(interp, args, nargout):
    args = list(args)
    base = Struct.scalar({})
    if args and isinstance(args[0], Struct):
        base = args.pop(0).copy()
    for k, v in _pairs(args, "simset"):
        base.set_field(0, k, v)
    return base


@builtin("simget")
@_wrap
def _simget(interp, args, nargout):
    nargs(args, 1, 2, "simget")
    m = _target(interp, args[0])[1]
    if len(args) == 2:
        want = text(args[1]).lower()
        for k, v in m.config.items():
            if k.lower() == want:
                return char(v)
        raise MatlabError(f"The model has no setting '{text(args[1])}'.")
    return Struct.scalar({k: char(v) for k, v in m.config.items()})


@builtin("linmod", "linearize", "linmod2")
@_wrap
def _linmod(interp, args, nargout):
    nargs(args, 1, 3, "linmod")
    m = _target(interp, args[0])[1]
    x0 = np.asarray(args[1], float) if len(args) > 1 and np.size(args[1]) else None
    u0 = np.asarray(args[2], float) if len(args) > 2 and np.size(args[2]) else None
    r = _blocks().linearize(m, interp, interp.frame, x0, u0)
    if nargout <= 1:
        return Struct.scalar({"a": r["A"], "b": r["B"], "c": r["C"], "d": r["D"],
                              "StateName": _cellstr(r["states"]), "OperPoint": Struct.scalar(
                                  {"x": r["x"].reshape(-1, 1), "u": r["u"].reshape(-1, 1)})})
    return (r["A"], r["B"], r["C"], r["D"])[:nargout]


@builtin("trim")
@_wrap
def _trim(interp, args, nargout):
    nargs(args, 1, 3, "trim")
    m = _target(interp, args[0])[1]
    x0 = np.asarray(args[1], float) if len(args) > 1 and np.size(args[1]) else None
    u0 = np.asarray(args[2], float) if len(args) > 2 and np.size(args[2]) else None
    r = _blocks().trim(m, interp, interp.frame, x0, u0)
    return tuple(r[k].reshape(-1, 1) for k in ("x", "u", "y", "dx"))[:max(nargout, 1)]


def _unused() -> None:  # keep the imports that only type checkers see
    _ = (MString, to_str)
