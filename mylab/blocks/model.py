"""The block-diagram data structure and its JSON file format.

A :class:`Model` has a name, a configuration (stop time, solver, ...) and a
root :class:`System`. A system is a list of blocks and a list of lines. A
block of type ``Subsystem`` owns a nested system. A line joins one output
port to one input port; a signal that goes to several places is stored as
several lines with the same source.

Ports are numbered from 1. The control inputs of conditionally run
subsystems are named ``"enable"`` and ``"trigger"``.
"""

from __future__ import annotations

import json
import os
from typing import Any, Iterator

FORMAT = "mylab-blocks-1"
EXTENSION = ".mlb.json"

DEFAULT_CONFIG = {
    "StartTime": "0",
    "StopTime": "10",
    "Solver": "ode45",
    "MaxStep": "auto",
    "RelTol": "1e-3",
    "AbsTol": "1e-6",
    "FixedStep": "auto",
    "ZeroCrossings": "on",
}


class ModelError(Exception):
    """A problem with the structure of a model (bad path, missing block, ...)."""


class Block:
    __slots__ = ("name", "type", "params", "pos", "rot", "flip", "system", "extra")

    def __init__(self, name: str, type: str, params: dict | None = None, pos: list | None = None,
                 rot: int = 0, flip: bool = False, system: "System | None" = None):
        self.name = name
        self.type = type
        self.params: dict[str, str] = dict(params or {})
        self.pos = list(pos) if pos else [0, 0, 0, 0]  # x, y, width, height
        self.rot = rot  # quarter turns clockwise
        self.flip = flip
        self.system = system
        self.extra: dict[str, Any] = {}

    def to_json(self) -> dict:
        d: dict[str, Any] = {"name": self.name, "type": self.type}
        if any(self.pos):
            d["position"] = [_num(v) for v in self.pos]
        if self.rot:
            d["rotation"] = self.rot
        if self.flip:
            d["flip"] = True
        if self.params:
            d["params"] = dict(self.params)
        d.update(self.extra)
        if self.system is not None:
            d.update(self.system.to_json())
        return d

    @classmethod
    def from_json(cls, d: dict) -> "Block":
        if not isinstance(d, dict) or "name" not in d or "type" not in d:
            raise ModelError("Every block needs a name and a type.")
        b = cls(str(d["name"]), str(d["type"]), {k: _text(v) for k, v in (d.get("params") or {}).items()},
                d.get("position"), int(d.get("rotation", 0)) % 4, bool(d.get("flip", False)))
        if "blocks" in d or "lines" in d or b.type == "Subsystem":
            b.system = System.from_json(d)
        known = {"name", "type", "position", "rotation", "flip", "params", "blocks", "lines"}
        b.extra = {k: v for k, v in d.items() if k not in known}
        return b

    def copy(self) -> "Block":
        return Block.from_json(json.loads(json.dumps(self.to_json())))


class Line:
    __slots__ = ("src", "dst", "name", "points", "log")

    def __init__(self, src: tuple, dst: tuple, name: str = "", points: list | None = None, log: bool = False):
        self.src = (src[0], _port(src[1]))
        self.dst = (dst[0], _port(dst[1]))
        self.name = name
        self.points = points or []
        self.log = log

    def to_json(self) -> dict:
        d: dict[str, Any] = {"from": list(self.src), "to": list(self.dst)}
        if self.name:
            d["name"] = self.name
        if self.log:
            d["log"] = True
        if self.points:
            d["points"] = [[_num(p[0]), _num(p[1])] for p in self.points]
        return d

    @classmethod
    def from_json(cls, d: dict) -> "Line":
        try:
            return cls(tuple(d["from"]), tuple(d["to"]), str(d.get("name", "")), d.get("points"),
                       bool(d.get("log", False)))
        except (KeyError, TypeError, IndexError):
            raise ModelError('Every line needs "from": [block, port] and "to": [block, port].') from None


class System:
    __slots__ = ("blocks", "lines")

    def __init__(self, blocks: list[Block] | None = None, lines: list[Line] | None = None):
        self.blocks: list[Block] = blocks or []
        self.lines: list[Line] = lines or []

    def block(self, name: str) -> Block | None:
        for b in self.blocks:
            if b.name == name:
                return b
        return None

    def to_json(self) -> dict:
        return {"blocks": [b.to_json() for b in self.blocks], "lines": [ln.to_json() for ln in self.lines]}

    @classmethod
    def from_json(cls, d: dict) -> "System":
        s = cls([Block.from_json(b) for b in d.get("blocks", [])], [Line.from_json(ln) for ln in d.get("lines", [])])
        seen = set()
        for b in s.blocks:
            if b.name in seen:
                raise ModelError(f"Two blocks are named '{b.name}' in the same system.")
            seen.add(b.name)
        return s

    def unique_name(self, base: str) -> str:
        if self.block(base) is None:
            return base
        stem = base.rstrip("0123456789") or base
        n = 1
        while self.block(f"{stem}{n}") is not None:
            n += 1
        return f"{stem}{n}"

    def walk(self, prefix: str = "") -> Iterator[tuple[str, Block, "System"]]:
        """Yield (path, block, owning system) for every block, depth first."""
        for b in self.blocks:
            path = f"{prefix}/{b.name}" if prefix else b.name
            yield path, b, self
            if b.system is not None:
                yield from b.system.walk(path)


class Model:
    def __init__(self, name: str = "untitled"):
        self.name = name
        self.config: dict[str, str] = dict(DEFAULT_CONFIG)
        self.root = System()
        self.file: str | None = None
        self.dirty = False
        self.notes: list[str] = []  # messages from an importer

    def to_json(self) -> dict:
        d: dict[str, Any] = {"format": FORMAT, "name": self.name, "config": dict(self.config)}
        d.update(self.root.to_json())
        return d

    @classmethod
    def from_json(cls, d: dict, name: str | None = None) -> "Model":
        if not isinstance(d, dict) or "blocks" not in d:
            raise ModelError("This is not a mylab Blocks model (no \"blocks\" list).")
        fmt = d.get("format", FORMAT)
        if fmt != FORMAT:
            raise ModelError(f"Unknown model format '{fmt}'. This version of mylab reads '{FORMAT}'.")
        m = cls(str(name or d.get("name") or "untitled"))
        m.config.update({k: _text(v) for k, v in (d.get("config") or {}).items()})
        m.root = System.from_json(d)
        return m

    def dumps(self) -> str:
        return _json(self.to_json()) + "\n"

    # -- paths ------------------------------------------------------------------------
    def system_at(self, parts: list[str]) -> System:
        """The system reached by following subsystem names from the root."""
        s = self.root
        walked = [self.name]
        for p in parts:
            b = s.block(p)
            if b is None:
                raise ModelError(f"There is no block '{p}' in '{'/'.join(walked)}'.")
            if b.system is None:
                raise ModelError(f"'{'/'.join(walked + [p])}' is not a subsystem.")
            s = b.system
            walked.append(p)
        return s

    def find(self, path: str) -> tuple[System, Block]:
        """(owning system, block) for a path relative to the model, e.g. 'Plant/Gain'."""
        parts = split_path(path)
        if not parts:
            raise ModelError("A block path is needed, e.g. 'model/Gain'.")
        s = self.system_at(parts[:-1])
        b = s.block(parts[-1])
        if b is None:
            where = "/".join([self.name] + parts[:-1])
            raise ModelError(f"There is no block '{parts[-1]}' in '{where}'.")
        return s, b


def split_path(path: str) -> list[str]:
    """Split 'a/b/c' at single slashes; a doubled slash stands for a slash inside a name."""
    parts = [p.replace("\0", "/") for p in path.replace("//", "\0").split("/")]
    return [p for p in parts if p != ""]


def _port(p: Any) -> Any:
    if isinstance(p, str) and not p.isdigit():
        return p
    return int(p)


def _num(v: Any) -> Any:
    f = float(v)
    return int(f) if f == int(f) else round(f, 2)


def _text(v: Any) -> str:
    if isinstance(v, str):
        return v
    if isinstance(v, bool):
        return "on" if v else "off"
    if isinstance(v, (int, float)):
        return repr(v) if isinstance(v, int) or v != int(v) else str(int(v))
    return json.dumps(v)


def build(name: str, blocks: list, lines: list, **config: str) -> Model:
    """Make a model from short descriptions (handy in tests and scripts).

    ``blocks`` holds ``(name, type)`` or ``(name, type, params)`` tuples; a
    fourth item is a nested ``(blocks, lines)`` pair for a subsystem.
    ``lines`` holds strings like ``"Step/1 -> Sum/1"``; add ``" : label"``
    to name the signal.
    """
    def system(bl: list, ll: list) -> System:
        sy = System()
        for k, item in enumerate(bl):
            b = Block(item[0], item[1], {key: _text(v) for key, v in (item[2] if len(item) > 2 and item[2] else {}).items()},
                      [60 + 120 * (k % 6), 60 + 100 * (k // 6), 0, 0])
            if len(item) > 3:
                b.system = system(*item[3])
            elif item[1] == "Subsystem":
                b.system = System()
            sy.blocks.append(b)
        for text in ll:
            text, _, label = text.partition(" : ")
            a, _, b2 = text.partition("->")
            src, dst = a.strip().rsplit("/", 1), b2.strip().rsplit("/", 1)
            sy.lines.append(Line((src[0], src[1]), (dst[0], dst[1]), label.strip()))
        return sy

    m = Model(name)
    m.root = system(blocks, lines)
    m.config.update({k: _text(v) for k, v in config.items()})
    return m


def _json(v: Any, indent: int = 0) -> str:
    """JSON with short lists on one line, so model files stay readable and diff well."""
    pad = " " * indent
    if isinstance(v, dict):
        if not v:
            return "{}"
        items = [f'{pad}  {json.dumps(k)}: {_json(x, indent + 2)}' for k, x in v.items()]
        return "{\n" + ",\n".join(items) + "\n" + pad + "}"
    if isinstance(v, list):
        if all(not isinstance(x, (dict, list)) or (isinstance(x, list) and all(not isinstance(y, (dict, list)) for y in x))
               for x in v):
            return json.dumps(v)
        return "[\n" + ",\n".join(f"{pad}  {_json(x, indent + 2)}" for x in v) + "\n" + pad + "]"
    return json.dumps(v)


def load_model(path: str) -> Model:
    """Read a model file. ``.slx`` files go through the importer."""
    if path.lower().endswith(".slx"):
        from .slx import import_slx
        return import_slx(path)
    if path.lower().endswith(".mdl"):
        raise ModelError("mylab cannot read .mdl files. Save the model as .slx, or rebuild it in mylab Blocks.")
    try:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
    except json.JSONDecodeError as exc:
        raise ModelError(f"{os.path.basename(path)} is not valid JSON: {exc}") from None
    base = os.path.basename(path)
    name = base[:-len(EXTENSION)] if base.endswith(EXTENSION) else os.path.splitext(base)[0]
    m = Model.from_json(data, name)
    m.file = path
    return m


def save_model(model: Model, path: str) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(model.dumps())
    model.file = path
    model.dirty = False
