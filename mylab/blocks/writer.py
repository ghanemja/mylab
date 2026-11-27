"""Writing models as ``.slx`` and ``.mdl`` files, for mylab to read back.

The files hold the layout the readers in :mod:`.slx` and :mod:`.mdl` expect
(the layout of the public files those readers were written from): blocks
under their type names with their parameters, lines with branches written
out as one line per input, subsystems nested in place, masks, MATLAB
Function code in a Stateflow part or section, and the solver settings.

These files are meant for mylab. They were not made or tested with any
other program, and other programs may not open them.

What is kept: blocks, parameters, positions and rotation, lines with names
and corner points, subsystems with their variables and initialization code,
variant choices, MATLAB Function code, model callbacks, the settings of the
configuration, and the port counts of placeholder and Model blocks. What is
not: flipped blocks without a rotation (they come back turned half way
round), the logging flag of lines, and display settings set with
``set_param`` (colours, name placement).
"""

from __future__ import annotations

import io
import zipfile
from typing import Any
from xml.etree import ElementTree as ET

from .library import find_type
from .model import Block, Model, System

# Our type -> the block type name the readers know, where they differ.
OUT_TYPE = {
    "SineWave": "Sin",
    "Saturation": "Saturate",
    "Sign": "Signum",
    "LogicalOperator": "Logic",
    "MathFunction": "Math",
    "TrigonometricFunction": "Trigonometry",
    "Polynomial": "Polyval",
    "StopSimulation": "Stop",
    "Subsystem": "SubSystem",
    "DiscreteTimeIntegrator": "DiscreteIntegrator",
    "MultiportSwitch": "MultiPortSwitch",
    "InterpretedFunction": "MATLABFcn",
    "SFunction": "S-Function",
    "MSFunction": "M-S-Function",
    "Prelookup": "PreLookup",
    "DirectLookupTable": "LookupNDDirect",
}
# Parameters stored under another name: {our type: {our name: their name}} (the reverse of slx.PARAM_MAP).
OUT_PARAM = {
    "Memory": {"InitialCondition": "X0"},
    "UnitDelay": {"InitialCondition": "X0"},
    "StateSpace": {"InitialCondition": "X0"},
    "DiscreteStateSpace": {"InitialCondition": "X0"},
    "Polynomial": {"Coefs": "coefs"},
    "ManualSwitch": {"sw": "CurrentSetting"},
}
CONFIG = ("StartTime", "StopTime", "Solver", "MaxStep", "RelTol", "AbsTol", "FixedStep", "ZeroCrossings",
          "AlgebraicLoop")
CALLBACKS = ("PreLoadFcn", "PostLoadFcn", "InitFcn", "StartFcn")


class _Node:
    """A neutral tree (a section with values and children) that becomes XML or .mdl text."""

    def __init__(self, tag: str, attrs: dict | None = None):
        self.tag = tag
        self.attrs = dict(attrs or {})
        self.values: list[tuple[str, str]] = []
        self.children: list[_Node] = []
        self.text: str | None = None

    def add(self, tag: str, attrs: dict | None = None) -> "_Node":
        n = _Node(tag, attrs)
        self.children.append(n)
        return n


def _num(v: float) -> str:
    v = float(v)
    return str(int(v)) if v == int(v) else repr(round(v, 4))


class _Builder:
    def __init__(self, model: Model):
        self.model = model
        self.next_sid = 0
        self.charts: list[tuple[str, str]] = []  # (block path inside the model, MATLAB Function code)

    def sid(self) -> str:
        self.next_sid += 1
        return str(self.next_sid)

    def block(self, b: Block, prefix: str) -> tuple[_Node, str]:
        cls = find_type(b.type)
        ours = cls.type if cls is not None else b.type
        sid = self.sid()
        kind = OUT_TYPE.get(ours, ours)
        if ours == "MATLABFunction":
            kind = "SubSystem"
        node = _Node("Block", {"BlockType": kind, "Name": b.name, "SID": sid})
        x, y, w, h = (b.pos + [0, 0, 0, 0])[:4]
        node.values.append(("Position", f"[{_num(x)}, {_num(y)}, {_num(x + w)}, {_num(y + h)}]"))
        if b.rot:
            node.values.append(("BlockRotation", {1: "270", 2: "180", 3: "90"}[b.rot % 4]))
        if b.flip:
            node.values.append(("BlockMirror", "on"))
        from .engine import port_count

        ports = b.extra.get("ports")
        if isinstance(ports, list) and len(ports) == 2:
            node.values.append(("Ports", f"[{int(ports[0])}, {int(ports[1])}]"))
        elif cls is not None:
            node.values.append(("Ports", f"[{port_count(b, 'in')}, {port_count(b, 'out')}]"))
        rename = OUT_PARAM.get(ours, {})
        path = f"{prefix}{b.name}"
        if ours == "MATLABFunction":
            node.values.append(("SFBlockType", "MATLAB Function"))
            code = b.params.get("Script") or (cls.params[0].default if cls else "")
            self.charts.append((path, code))
        else:
            # every parameter, defaults too, so that a reader that assumes other defaults gets the same block
            values = {p.name: p.default for p in (cls.params if cls is not None else [])}
            values.update(b.params)
            for k, v in values.items():
                node.values.append((rename.get(k, k), v))
        if b.system is not None:
            variants = b.extra.get("variants")
            if isinstance(variants, dict) and variants:
                node.values.append(("Variant", "on"))
                label = b.extra.get("variantLabel")
                if isinstance(label, str) and label:
                    node.values.append(("VariantControlMode", "label"))
                    node.values.append(("LabelModeActiveChoice", label))
            mask = b.extra.get("mask")
            init = b.extra.get("maskInit")
            if (isinstance(mask, dict) and mask) or (isinstance(init, str) and init.strip()):
                m = node.add("Mask")
                if isinstance(init, str) and init.strip():
                    m.add("Initialization").text = init
                for name, expr in (mask or {}).items():
                    mp = m.add("MaskParameter", {"Name": name, "Type": "edit", "Evaluate": "on"})
                    mp.add("Value").text = str(expr)
            inner = self.system(b.system, path + "/", variants if isinstance(variants, dict) else None)
            node.children.append(inner)
        return node, sid

    def system(self, sy: System, prefix: str, variants: dict | None = None) -> _Node:
        node = _Node("System")
        sids: dict[str, str] = {}
        boxes: dict[str, tuple] = {}
        for b in sy.blocks:
            bn, sid = self.block(b, prefix)
            if variants and b.name in variants:
                bn.values.append(("VariantControl", str(variants[b.name])))
            node.children.append(bn)
            sids[b.name] = sid
            boxes[b.name] = tuple(b.pos)
        from .slx import _Importer

        for ln in sy.lines:
            if ln.src[0] not in sids or ln.dst[0] not in sids:
                continue
            ln_node = node.add("Line")
            if ln.name:
                ln_node.values.append(("Name", ln.name))
            src_port = f"out:{ln.src[1]}" if isinstance(ln.src[1], int) else str(ln.src[1])
            dst_port = f"in:{ln.dst[1]}" if isinstance(ln.dst[1], int) else str(ln.dst[1])
            ln_node.values.append(("Src", f"{sids[ln.src[0]]}#{src_port}"))
            if ln.points:
                at = _Importer._port_xy(sy, ln.src[0], ln.src[1], True, boxes)
                rel = []
                for px, py in ln.points:
                    rel.append(f"{_num(px - at[0])}, {_num(py - at[1])}")
                    at = (px, py)
                ln_node.values.append(("Points", "[" + "; ".join(rel) + "]"))
            ln_node.values.append(("Dst", f"{sids[ln.dst[0]]}#{dst_port}"))
        return node

    def build(self) -> tuple[_Node, dict]:
        m = self.model
        top = _Node("Library" if m.library else "Model")
        top.values.append(("Name", m.name))
        if m.description:
            top.values.append(("Description", m.description))
        for k in CALLBACKS:
            if m.callbacks.get(k, "").strip():
                top.values.append((k, m.callbacks[k]))
        config = {k: v for k, v in m.config.items() if k in CONFIG}
        top.children.append(self.system(m.root, ""))
        return top, config


# -------------------------------------------------------------------------------------
# .slx
# -------------------------------------------------------------------------------------


def _xml(node: _Node) -> ET.Element:
    if node.tag in ("Model", "Library"):
        el = ET.Element(node.tag)
    else:
        el = ET.Element(node.tag, node.attrs)
    if node.text is not None:
        el.text = node.text
    for k, v in node.values:
        p = ET.SubElement(el, "P", Name=k)
        p.text = v
    for c in node.children:
        el.append(_xml(c))
    return el


def _bytes(el: ET.Element) -> bytes:
    ET.indent(el)
    return b'<?xml version="1.0" encoding="utf-8"?>\n' + ET.tostring(el, encoding="utf-8")


CONTENT_TYPES = """<?xml version="1.0" encoding="UTF-8" standalone="yes" ?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default ContentType="application/vnd.openxmlformats-package.relationships+xml" Extension="rels"/>
  <Default ContentType="application/xml" Extension="xml"/>
</Types>
"""
RELS = """<?xml version="1.0" encoding="UTF-8" standalone="yes" ?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="blockDiagram" Target="simulink/blockdiagram.xml" Type="http://schemas.mathworks.com/simulink/2010/relationships/blockDiagram"/>
</Relationships>
"""


def slx_bytes(model: Model) -> bytes:
    """The model as the bytes of an .slx file (a zip archive of XML parts)."""
    b = _Builder(model)
    top, config = b.build()
    info = ET.Element("ModelInformation", Version="1.0")
    info.append(_xml(top))
    cs = ET.Element("ConfigSet")
    for k, v in config.items():
        p = ET.SubElement(cs, "P", Name=k)
        p.text = v
    meta = ET.Element("MylabModel")
    ET.SubElement(meta, "P", Name="Writer").text = "mylab Blocks"
    meta_p = ET.SubElement(meta, "P", Name="Note")
    meta_p.text = "Written by mylab for mylab; not made with or tested in any other program."
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("[Content_Types].xml", CONTENT_TYPES)
        z.writestr("_rels/.rels", RELS)
        z.writestr("simulink/blockdiagram.xml", _bytes(info))
        z.writestr("simulink/configSet0.xml", _bytes(cs))
        if b.charts:
            z.writestr("simulink/stateflow/stateflow.xml", _bytes(_charts_xml(b.charts)))
        z.writestr("metadata/mylab.xml", _bytes(meta))
    return buf.getvalue()


def _charts_xml(charts: list) -> ET.Element:
    root = ET.Element("Stateflow")
    machine = ET.SubElement(root, "machine", id="1")
    kids = ET.SubElement(machine, "Children")
    k = 1
    for path, code in charts:
        k += 1
        chart = ET.SubElement(kids, "chart", id=str(k))
        ET.SubElement(chart, "P", Name="name").text = path
        ET.SubElement(chart, "P", Name="type").text = "EML_CHART"
        inner = ET.SubElement(chart, "Children")
        k += 1
        state = ET.SubElement(inner, "state", id=str(k))
        eml = ET.SubElement(state, "eml")
        ET.SubElement(eml, "P", Name="isEML").text = "1"
        ET.SubElement(eml, "P", Name="script").text = code
    return root


# -------------------------------------------------------------------------------------
# .mdl
# -------------------------------------------------------------------------------------


def _quote(s: str) -> str:
    s = str(s).replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n").replace("\t", "\\t").replace("\r", "")
    return f'"{s}"'


def _mdl(node: _Node, out: list, depth: int, sids: dict | None = None) -> None:
    pad = "  " * depth
    out.append(f"{pad}{node.tag} {{")
    inner = pad + "  "
    for k, v in node.attrs.items():
        if k == "BlockType":
            out.append(f"{inner}BlockType {v}")
        else:
            out.append(f"{inner}{k} {_quote(v)}")
    for k, v in node.values:
        out.append(f"{inner}{k} {_quote(v)}")
    if node.text is not None:
        out.append(f"{inner}Text {_quote(node.text)}")
    for c in node.children:
        _mdl(c, out, depth + 1)
    out.append(f"{pad}}}")


def _mdl_system(sys_node: _Node, out: list, depth: int) -> None:
    """A System section: blocks by name, lines with SrcBlock/SrcPort/DstBlock/DstPort."""
    pad = "  " * depth
    inner = pad + "  "
    out.append(f"{pad}System {{")
    names = {}
    for c in sys_node.children:
        if c.tag == "Block":
            names[c.attrs["SID"]] = c.attrs["Name"]
    for c in sys_node.children:
        if c.tag == "Block":
            _mdl_block(c, out, depth + 1)
    for c in sys_node.children:
        if c.tag != "Line":
            continue
        vals = dict(c.values)
        out.append(f"{inner}Line {{")
        if "Name" in vals:
            out.append(f"{inner}  Name {_quote(vals['Name'])}")
        for side, key in (("Src", vals.get("Src", "")), ("Dst", vals.get("Dst", ""))):
            sid, _, port = key.partition("#")
            kind, _, num = port.partition(":")
            out.append(f"{inner}  {side}Block {_quote(names.get(sid, sid))}")
            out.append(f"{inner}  {side}Port {num if num else kind}")
            if side == "Src" and "Points" in vals:
                out.append(f"{inner}  Points {vals['Points']}")
        out.append(f"{inner}}}")
    out.append(f"{pad}}}")


def _mdl_block(node: _Node, out: list, depth: int) -> None:
    pad = "  " * depth
    inner = pad + "  "
    out.append(f"{pad}Block {{")
    out.append(f"{inner}BlockType {node.attrs['BlockType']}")
    out.append(f"{inner}Name {_quote(node.attrs['Name'])}")
    out.append(f"{inner}SID {_quote(node.attrs['SID'])}")
    for k, v in node.values:
        if k in ("Position", "Ports") and v.startswith("["):
            out.append(f"{inner}{k} {v}")
        else:
            out.append(f"{inner}{k} {_quote(v)}")
    for c in node.children:
        if c.tag == "Mask":
            out.append(f"{inner}Object {{")
            out.append(f'{inner}  $PropName "MaskObject"')
            out.append(f'{inner}  $ClassName "Simulink.Mask"')
            for mc in c.children:
                if mc.tag == "Initialization":
                    out.append(f"{inner}  Initialization {_quote(mc.text or '')}")
                elif mc.tag == "MaskParameter":
                    out.append(f"{inner}  Object {{")
                    out.append(f'{inner}    $PropName "Parameters"')
                    out.append(f'{inner}    $ClassName "Simulink.MaskParameter"')
                    out.append(f"{inner}    Type {_quote(mc.attrs.get('Type', 'edit'))}")
                    out.append(f"{inner}    Name {_quote(mc.attrs['Name'])}")
                    value = next((x.text for x in mc.children if x.tag == "Value"), "")
                    out.append(f"{inner}    Value {_quote(value or '')}")
                    out.append(f"{inner}  }}")
            out.append(f"{inner}}}")
        elif c.tag == "System":
            _mdl_system(c, out, depth + 1)
    out.append(f"{pad}}}")


def mdl_text(model: Model) -> str:
    """The model in the text (.mdl) format."""
    b = _Builder(model)
    top, config = b.build()
    out = ["# Written by mylab Blocks for mylab; not made with or tested in any other program.", f"{top.tag} {{"]
    for k, v in top.values:
        out.append(f"  {k} {_quote(v)}")
    out.append("  Array {")
    out.append('    Type "Handle"')
    out.append("    Simulink.ConfigSet {")
    out.append("      Array {")
    out.append("        Simulink.SolverCC {")
    for k, v in config.items():
        out.append(f"          {k} {_quote(v)}")
    out.append("        }")
    out.append("      }")
    out.append("    }")
    out.append("  }")
    for c in top.children:
        if c.tag == "System":
            _mdl_system(c, out, 1)
    out.append("}")
    if b.charts:
        out.append("Stateflow {")
        k = 1
        for path, code in b.charts:
            k += 1
            out += ["  chart {", f"    id {k}", f"    name {_quote(path)}", "    type EML_CHART", "  }"]
            out += ["  state {", f"    id {k + 1}", f"    chart {k}", "    eml {", "      isEML 1",
                    f"      script {_quote(code)}", "    }", "  }"]
            k += 1
        out.append("}")
    return "\n".join(out) + "\n"


def save_as(model: Model, path: str) -> None:
    """Write a model as .slx or .mdl (chosen by the extension)."""
    low = path.lower()
    if low.endswith(".slx"):
        data: Any = slx_bytes(model)
        with open(path, "wb") as fh:
            fh.write(data)
    elif low.endswith(".mdl"):
        with open(path, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(mdl_text(model))
    else:
        raise ValueError("save_as writes .slx or .mdl files")
