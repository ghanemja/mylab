"""Reader for ``.mdl`` model files, the older text format.

An ``.mdl`` file is plain text: nested sections ``Name { ... }`` holding
``Key value`` lines, where a value is a quoted string (long strings go on in
quoted pieces on the next lines), a bracketed array or a bare word. This
reader was written from the structure of public, permissively licensed
files, without vendor code or documentation.

The text is turned into the same XML element tree an ``.slx`` file holds,
so the ``.slx`` importer (:mod:`.slx`) converts both: blocks, lines and
branches, subsystems, masks, library links, MATLAB Function code and the
solver settings. Like the ``.slx`` reader, it treats the file as data: sizes
are limited and nothing in it is run.
"""

from __future__ import annotations

import re
from xml.etree import ElementTree as ET

from .model import ModelError

MAX_FILE = 128 * 1024 * 1024
MAX_DEPTH = 200


class Section:
    __slots__ = ("name", "items")

    def __init__(self, name: str):
        self.name = name
        self.items: list = []  # (key, str value) or Section

    def get(self, key: str, default: str = "") -> str:
        for it in self.items:
            if isinstance(it, tuple) and it[0] == key:
                return it[1]
        return default

    def sections(self, name: str | None = None) -> list["Section"]:
        return [it for it in self.items if isinstance(it, Section) and (name is None or it.name == name)]

    def values(self) -> list[tuple[str, str]]:
        return [it for it in self.items if isinstance(it, tuple)]


_ESC = {"n": "\n", "t": "\t", "r": "\r", '"': '"', "\\": "\\", "'": "'"}


def _unquote(s: str) -> str:
    """The text of one quoted piece (without its quotes), with escapes resolved."""
    out = []
    i = 0
    while i < len(s):
        c = s[i]
        if c == "\\" and i + 1 < len(s):
            out.append(_ESC.get(s[i + 1], s[i + 1]))
            i += 2
            continue
        out.append(c)
        i += 1
    return "".join(out)


_QUOTED = re.compile(r'"((?:[^"\\]|\\.)*)"')


def parse(text: str) -> Section:
    """The tree of sections of an .mdl file."""
    root = Section("")
    stack = [root]
    lines = text.splitlines()
    i = 0
    n = len(lines)
    while i < n:
        raw = lines[i].strip()
        i += 1
        if not raw or raw.startswith("#"):
            continue
        if raw == "}":
            if len(stack) > 1:
                stack.pop()
            continue
        m = re.match(r"^(\S+)\s*\{$", raw)
        if m:
            sec = Section(m.group(1))
            stack[-1].items.append(sec)
            stack.append(sec)
            if len(stack) > MAX_DEPTH:
                raise ModelError("The .mdl file is nested too deeply to import.")
            continue
        m = re.match(r"^(\S+)\s+(.*)$", raw)
        if not m:
            stack[-1].items.append((raw, ""))
            continue
        key, rest = m.group(1), m.group(2).strip()
        if rest.startswith('"'):
            parts = [_unquote(p) for p in _QUOTED.findall(rest)]
            while i < n and lines[i].lstrip().startswith('"'):  # a long string goes on in the next lines
                parts += [_unquote(p) for p in _QUOTED.findall(lines[i].strip())]
                i += 1
            value = "".join(parts)
        elif rest.startswith("[") and rest.count("[") > rest.count("]"):
            buf = [rest]
            while i < n and sum(b.count("[") - b.count("]") for b in buf) > 0:
                buf.append(lines[i].strip())
                i += 1
            value = " ".join(buf)
        else:
            value = rest
        stack[-1].items.append((key, value))
    return root


# -------------------------------------------------------------------------------------
# The same XML an .slx file holds
# -------------------------------------------------------------------------------------


def _p(parent: ET.Element, name: str, value: str) -> None:
    e = ET.SubElement(parent, "P", Name=name)
    e.text = value


def _port(text: str, out: bool) -> str | None:
    t = text.strip()
    if t.isdigit():
        return f"{'out' if out else 'in'}:{t}"
    low = t.lower()
    if low in ("enable", "trigger", "ifaction", "reset", "state"):
        return low
    m = re.fullmatch(r"(lconn|rconn)(\d+)", low)
    if m:
        return f"{m.group(1)}:{m.group(2)}"
    return None


class _Converter:
    def __init__(self):
        self.next_sid = 0

    def sid(self) -> str:
        self.next_sid += 1
        return f"mdl{self.next_sid}"

    def mask(self, obj: Section, parent: ET.Element) -> None:
        mask = ET.SubElement(parent, "Mask")
        for key, tag in (("Type", "Type"), ("Initialization", "Initialization"), ("Display", "Display")):
            v = obj.get(key)
            if v:
                ET.SubElement(mask, tag).text = v
        params = []
        for sub in obj.sections():
            if sub.name == "Object" and "MaskParameter" in sub.get("$ClassName"):
                params.append(sub)
            elif sub.name == "Array" and "MaskParameter" in sub.get("Type"):
                params += [o for o in sub.sections() if o.name == "Object"]
        for mp in params:
            el = ET.SubElement(
                mask, "MaskParameter", Name=mp.get("Name"), Type=mp.get("Type", "edit") or "edit",
                Evaluate=mp.get("Evaluate", "on") or "on",
            )
            ET.SubElement(el, "Value").text = mp.get("Value")
            opts = [o for o in mp.sections() if o.name == "Array"]
            if opts or mp.get("TypeOptions"):
                te = ET.SubElement(el, "TypeOptions")
                raw = mp.get("TypeOptions")
                items = [x for x in raw.strip("{}").split("|")] if raw else []
                for o in opts:
                    items += [v for k, v in o.values() if k == "Cell"]
                for item in items:
                    ET.SubElement(te, "Option").text = item

    def block(self, sec: Section, parent: ET.Element) -> str:
        sid = sec.get("SID") or self.sid()
        el = ET.SubElement(parent, "Block", BlockType=sec.get("BlockType"), Name=sec.get("Name"), SID=sid)
        for key, value in sec.values():
            if key not in ("BlockType", "Name", "SID"):
                _p(el, key, value)
        for sub in sec.sections():
            if sub.name == "System":
                el.append(self.system(sub))
            elif sub.name == "Object" and sub.get("$ClassName") == "Simulink.Mask":
                self.mask(sub, el)
        return sid

    def line(self, sec: Section, parent: ET.Element, sids: dict, src: str | None, tag: str = "Line") -> None:
        el = ET.SubElement(parent, tag)
        name = sec.get("Name")
        if name:
            _p(el, "Name", name)
        if sec.get("SrcBlock"):
            port = _port(sec.get("SrcPort", "1"), True)
            if sec.get("SrcBlock") in sids and port:
                src = f"{sids[sec.get('SrcBlock')]}#{port}"
            else:
                src = None
            _p(el, "Src", src or "?#?")
        pts = sec.get("Points")
        if pts:
            _p(el, "Points", pts)
        if sec.get("DstBlock"):
            port = _port(sec.get("DstPort", "1"), False)
            dst = f"{sids[sec.get('DstBlock')]}#{port}" if sec.get("DstBlock") in sids and port else "?#?"
            _p(el, "Dst", dst)
        for br in sec.sections("Branch"):
            self.line(br, el, sids, src, "Branch")

    def system(self, sec: Section) -> ET.Element:
        el = ET.Element("System")
        sids = {}
        for b in sec.sections("Block"):
            sids[b.get("Name")] = self.block(b, el)
        for ln in sec.sections("Line"):
            self.fix_line(ln)
            self.line(ln, el, sids, None)
        return el

    @staticmethod
    def fix_line(ln: Section) -> None:
        """A line whose source is given in a branch (physical connections): move it to the top."""
        if ln.get("SrcBlock"):
            return
        for br in ln.sections("Branch"):
            if br.get("SrcBlock") and not br.get("DstBlock") and not br.sections("Branch"):
                ln.items.insert(0, ("SrcBlock", br.get("SrcBlock")))
                ln.items.insert(1, ("SrcPort", br.get("SrcPort", "1")))
                ln.items.remove(br)
                return


def _solver_settings(top: Section) -> list[tuple[str, str]]:
    found = []

    def walk(s: Section) -> None:
        if s.name in ("Simulink.SolverCC",):
            found.extend(s.values())
        for sub in s.sections():
            walk(sub)

    walk(top)
    return found


def to_parts(text: str) -> dict[str, ET.Element]:
    """The XML parts an .slx file with the same content would have."""
    tree = parse(text)
    top = next((s for s in tree.sections() if s.name in ("Model", "Library")), None)
    if top is None:
        raise ModelError("The .mdl file contains neither a model nor a library.")
    conv = _Converter()
    info = ET.Element("ModelInformation")
    model = ET.SubElement(info, top.name)
    for key, value in _solver_settings(top):
        if key in ("StartTime", "StopTime", "Solver", "SolverName", "MaxStep", "RelTol", "AbsTol", "FixedStep",
                   "ZeroCrossings", "AlgebraicLoop"):
            _p(model, key, value)
    for key, value in top.values():
        _p(model, key, value)
    for d in top.sections("BlockParameterDefaults"):
        de = ET.SubElement(model, "BlockParameterDefaults")
        for b in d.sections("Block"):
            be = ET.SubElement(de, "Block", BlockType=b.get("BlockType"))
            for key, value in b.values():
                if key != "BlockType":
                    _p(be, key, value)
    systems = top.sections("System")
    if not systems:
        raise ModelError("The .mdl file has no block diagram in it.")
    model.append(conv.system(systems[0]))
    parts = {"simulink/blockdiagram.xml": info}
    fwd = top.get("ForwardingTableString")
    if fwd:
        table = ET.Element("ForwardingTable")
        for m in re.finditer(r"__slOldName__\|\|(.*?)\|\|__slNewName__\|\|(.*?)(?=\|\|__slOldName__|$)", fwd):
            entry = ET.SubElement(table, "Entry")
            ET.SubElement(entry, "OldBlock").text = m.group(1).strip("|")
            ET.SubElement(entry, "NewBlock").text = m.group(2).strip("|")
        parts["simulink/forwardingTable.xml"] = table
    sf = next((s for s in tree.sections() if s.name == "Stateflow"), None)
    if sf is not None:
        parts["simulink/stateflow/stateflow.xml"] = stateflow_xml(sf)
    return parts


def stateflow_xml(sf: Section) -> ET.Element:
    """The Stateflow section (a flat list of objects linked by id) as nested XML: charts hold their states,
    transitions, junctions and data, the way an .slx file stores them."""
    root = ET.Element("Stateflow")
    charts: dict[str, ET.Element] = {}
    for c in sf.sections("chart"):
        el = ET.SubElement(root, "chart", id=c.get("id"))
        for key, value in c.values():
            _p(el, key, value)
        for sub in c.sections():
            _section_xml(sub, el)
        charts[c.get("id")] = el
    for kind in ("state", "transition", "junction", "data", "event"):
        for obj in sf.sections(kind):
            owner = obj.get("chart")
            if kind in ("data", "event"):
                link = obj.get("linkNode").strip("[]").split()
                owner = link[0] if link else owner
            parent = charts.get(owner)
            if parent is None:
                continue
            _section_xml(obj, parent)
    return root


def _section_xml(sec: Section, parent: ET.Element) -> None:
    el = ET.SubElement(parent, sec.name)
    if sec.get("id"):
        el.set("id", sec.get("id"))
    for key, value in sec.values():
        _p(el, key, value)
    for sub in sec.sections():
        _section_xml(sub, el)


class MdlArchive:
    """Looks like the archive of an .slx file to the importer."""

    def __init__(self, path: str):
        import os

        try:
            size = os.path.getsize(path)
            if size > MAX_FILE:
                raise ModelError("The .mdl file is too large to import.")
            with open(path, "rb") as fh:
                data = fh.read()
        except OSError as exc:
            raise ModelError(f"{path} cannot be read: {exc}") from None
        text = None
        for enc in ("utf-8", "latin-1"):
            try:
                text = data.decode(enc)
                break
            except UnicodeDecodeError:
                continue
        if not text or not re.match(r"\s*(#.*\n\s*)*(Model|Library)\s*\{", text):
            raise ModelError(f"{os.path.basename(path)} is not a model file in the text (.mdl) format.")
        self.mylab = text.startswith("# Written by mylab")
        self.parts = to_parts(text)
        self.sizes = {k: 0 for k in self.parts}

    def xml(self, name: str) -> ET.Element | None:
        return self.parts.get(name)

    def names(self) -> list[str]:
        return list(self.parts)
