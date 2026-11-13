"""Reader for simple ``.slx`` model files.

An ``.slx`` file is a zip archive of XML parts. This reader was written from
the structure of files found in public repositories; it uses no vendor code
or documentation. It understands two layouts: everything in
``simulink/blockdiagram.xml``, or one ``simulink/systems/system_*.xml`` part
per subsystem. It converts the blocks that mylab Blocks has an equivalent
for, keeps every other block under its original type name (so the model
shows it and the simulator names it), and lists what it could not convert in
``model.notes`` and ``model.import_report``.

The file is treated as untrusted data: sizes are limited, XML with a
document type or entity declarations is refused, and nothing in the file is
executed.
"""

from __future__ import annotations

import re
import zipfile
from collections import Counter
from typing import Any
from xml.etree import ElementTree as ET

from .library import find_type
from .slx_library import KINEMATIC, coordinate_conversion, kinematic_model
from .model import Block, Line, Model, ModelError, System

MAX_PART = 64 * 1024 * 1024
MAX_TOTAL = 400 * 1024 * 1024
MAX_MEMBERS = 5000

# Their block type -> ours, where the names differ (the rest is found through the library's aliases).
TYPE_MAP = {
    "Sin": "SineWave",
    "Saturate": "Saturation",
    "Signum": "Sign",
    "Logic": "LogicalOperator",
    "Math": "MathFunction",
    "Trigonometry": "TrigonometricFunction",
    "Polyval": "Polynomial",
    "Stop": "StopSimulation",
    "SubSystem": "Subsystem",
    "DiscreteIntegrator": "DiscreteTimeIntegrator",
    "MultiPortSwitch": "MultiportSwitch",
    "MATLABFcn": "InterpretedFunction",
    "Record": "Scope",
    "Lookup": "LookupTable1D",
    "Lookup2D": "LookupTable2D",
    "DiscretePulseGenerator": "PulseGenerator",
    "ToWorkspace": "ToWorkspace",
    "DigitalClock": "Clock",
    "SignalConversion": "DataTypeConversion",
    "SignalSpecification": "DataTypeConversion",
    "InitialCondition": "DataTypeConversion",
    "Concatenate": "Concatenate",
    "DiscreteFir": "DiscreteFir",
    "S-Function": "SFunction",
    "M-S-Function": "MSFunction",
    "PreLookup": "Prelookup",
    "LookupNDDirect": "DirectLookupTable",
    "Sum": "Sum",
}
# Parameter names that differ: {our type: {their name: our name}}.
PARAM_MAP = {
    "Memory": {"X0": "InitialCondition"},
    "UnitDelay": {"X0": "InitialCondition"},
    "StateSpace": {"X0": "InitialCondition"},
    "DiscreteStateSpace": {"X0": "InitialCondition"},
    "LookupTable1D": {"InputValues": "BreakpointsForDimension1"},
    "LookupTable2D": {"RowIndex": "BreakpointsForDimension1", "ColumnIndex": "BreakpointsForDimension2"},
    "Polynomial": {"coefs": "Coefs"},
    "ManualSwitch": {"CurrentSetting": "sw"},
    "Ramp": {"slope": "Slope", "start": "Start", "X0": "InitialOutput"},
    "TransportDelay": {"InitialOutput": "InitialOutput"},
    "TransferFcn": {"N": "Numerator", "D": "Denominator", "X0": "InitialStates"},
}
# Display-only sinks: nothing to compute. They become a Viewer block and a note.
VIEWERS = {"videoviewer", "tovideodisplay", "uavanimation", "vrsink", "matrixviewer"}
# Dashboard controls and gauges have no signal ports; the value they set is already in the block they point to.
DASHBOARD = re.compile(r"(Slider|Knob|Gauge|Switch|Lamp|Button|Checkbox|EditField|ComboBox|RadioButton|Display|DashboardScope)Block$")
# Library links ("Reference" blocks) we can stand in for: SourceType or last part of SourceBlock -> (our type, fixed params).  # noqa: E501
REFERENCE_MAP = {
    "comparetoconstant": ("CompareToConstant", {}),
    "comparetozero": ("CompareToZero", {}),
    "bandlimitedwhitenoise": ("BandLimitedWhiteNoise", {}),
    "ramp": ("Ramp", {}),
    "repeatingsequencestair": ("RepeatingSequenceStair", {}),
    "repeatingtable": ("RepeatingSequence", {}),
    "repeatingsequence": ("RepeatingSequence", {}),
    "radianstodegrees": ("Gain", {"Gain": "180/pi"}),
    "degreestoradians": ("Gain", {"Gain": "pi/180"}),
    "pid1dof": ("PIDController", {}),
    "pidcontroller": ("PIDController", {}),
    "saturationdynamic": ("SaturationDynamic", {}),
    "chirp": ("Chirp", {}),
    "chirpsignal": ("Chirp", {}),
    "xyscope": ("XYGraph", {}),
    "xygraph": ("XYGraph", {}),
    "discretederivative": ("DiscreteDerivative", {}),
    "ratelimiterdynamic": ("RateLimiterDynamic", {}),
    "minmaxrunningresettable": ("MinMaxRunningResettable", {}),
    "transferfunctionwithinitialstates": ("TransferFcn", {}),
    "transferfcnwithinitialstates": ("TransferFcn", {}),
    "transpose": ("MathFunction", {"Operator": "transpose"}),
    "coulombviscousfriction": ("CoulombViscousFriction", {}),
    "coulombicandviscousfriction": ("CoulombViscousFriction", {}),
}
# Defaults of their block types that differ from ours, for files that do not list their defaults.
IMPORT_DEFAULTS = {
    "Demux": {"Outputs": "4"},
    "Mux": {"Inputs": "4"},
    "DiscretePulseGenerator": {"PulseType": "Sample based", "PulseWidth": "1", "SampleTime": "1"},
}
# Blocks whose port count is one parameter: the count saved in the file wins over a default.
COUNTED = {
    "Mux": ("Inputs", 0),
    "Demux": ("Outputs", 1),
    "BusCreator": ("Inputs", 0),
    "Merge": ("Inputs", 0),
    "Concatenate": ("NumInputs", 0),
    "Scope": ("NumInputPorts", 0),
}
SILENT = {"cmblock", "modelinfo", "docblock"}  # notes on the canvas: no ports, nothing to simulate
SOLVERS = {
    "variablestepauto": "ode45",
    "fixedstepauto": "ode3",
    "variablestepdiscrete": "VariableStepDiscrete",
    "fixedstepdiscrete": "FixedStepDiscrete",
}


def _key(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower())


class _Archive:
    def __init__(self, path: str):
        try:
            self.zip = zipfile.ZipFile(path)
        except (zipfile.BadZipFile, OSError) as exc:
            raise ModelError(
                f"{path} is not a readable .slx file ({exc}). Old-style .mdl text files are not supported."
            ) from None
        infos = self.zip.infolist()
        if len(infos) > MAX_MEMBERS or sum(i.file_size for i in infos) > MAX_TOTAL:
            raise ModelError("The .slx file is too large to import.")
        self.sizes = {i.filename: i.file_size for i in infos}

    def xml(self, name: str) -> ET.Element | None:
        if name not in self.sizes:
            return None
        if self.sizes[name] > MAX_PART:
            raise ModelError(f"The part {name} of the .slx file is too large to import.")
        data = self.zip.read(name)
        head = data[:4096]
        if b"<!DOCTYPE" in head or b"<!ENTITY" in data:
            raise ModelError(f"The part {name} of the .slx file declares XML entities; it is not imported.")
        try:
            return ET.fromstring(data)
        except ET.ParseError as exc:
            raise ModelError(f"The part {name} of the .slx file is not valid XML: {exc}") from None

    def names(self) -> list[str]:
        return list(self.sizes)


def _params(el: ET.Element) -> dict[str, str]:
    out = {}
    for p in el.findall("P"):
        out[p.get("Name", "")] = p.text or ""
    inst = el.find("InstanceData")
    if inst is not None:
        for p in inst.findall("P"):
            out.setdefault(p.get("Name", ""), p.text or "")
    return out


def _numbers(text: str) -> list[float]:
    out = []
    for tok in re.split(r"[\s,;\[\]]+", text or ""):
        if tok:
            try:
                out.append(float(tok))
            except ValueError:
                pass
    return out


class _Stats:
    def __init__(self):
        self.unsupported: Counter = Counter()
        self.notes: list[str] = []
        self.total = 0
        self.converted = 0
        self.skipped_lines = 0
        self.linked = 0


class _Importer:
    def __init__(self, path: str, root: "_Importer | None" = None):
        import os

        self.path = os.path.abspath(path)
        self.arc = _Archive(path)
        self.root = root or self
        self.stats = self.root.stats if root else _Stats()
        self.defaults: dict[str, dict[str, str]] = {}
        self.scripts: dict[str, str] = {}
        self.top: ET.Element | None = None
        self.top_system: ET.Element | None = None
        self.libs: dict[str, "_Importer | None"] = {}
        self.index: dict[str, str] | None = None
        self._forward: dict[str, str] | None = None
        self.depth = 0

    # counters live in one place, shared with the libraries a model links to
    unsupported = property(lambda self: self.stats.unsupported)
    notes = property(lambda self: self.stats.notes)

    def _count(self, name: str, by: int = 1) -> None:
        setattr(self.stats, name, getattr(self.stats, name) + by)

    def open(self) -> None:
        root = self.arc.xml("simulink/blockdiagram.xml")
        if root is None:
            raise ModelError("This .slx file has no simulink/blockdiagram.xml part; it cannot be imported.")
        top = root.find("Model")
        if top is None:
            top = root.find("Library")
        if top is None:
            raise ModelError("The .slx file contains neither a model nor a library.")
        for d in top.iter("BlockParameterDefaults"):
            for b in d.findall("Block"):
                self.defaults.setdefault(b.get("BlockType", ""), {}).update(_params(b))
        self._read_scripts()
        sys_el = top.find("System")
        if sys_el is None:
            raise ModelError("The .slx file has no block diagram in it.")
        self.top, self.top_system = top, self._deref(sys_el)

    # -- library links ----------------------------------------------------------------
    def library(self, name: str) -> "_Importer | None":
        """The importer of a library file named ``name``.slx found near the model, or None."""
        import os

        root = self.root
        if name in root.libs:
            return root.libs[name]
        found = None
        if re.fullmatch(r"[\w.-]+", name):
            if root.index is None:
                root.index = {}
                here = os.path.dirname(root.path)
                seen = 0
                for _level in range(5):  # the model's folder, then up to four folders above it
                    for folder, dirs, files in os.walk(here):
                        dirs[:] = [d for d in dirs if not d.startswith(".")]
                        if folder[len(here) :].count(os.sep) >= 5:
                            dirs[:] = []
                        for f in files:
                            seen += 1
                            if f.lower().endswith(".slx"):
                                root.index.setdefault(f[:-4], os.path.join(folder, f))
                        if seen > 20000:
                            break
                    parent = os.path.dirname(here)
                    if parent == here or seen > 20000:
                        break
                    here = parent
            path = root.index.get(name)
            if path and os.path.abspath(path) == root.path:
                found = root
            elif path:
                try:
                    found = _Importer(path, root)
                    found.open()
                except ModelError:
                    found = None
        root.libs[name] = found
        return found

    def find_block(self, parts: list[str]) -> tuple[ET.Element, str] | None:
        """The XML element of the block at a path inside this file, and the path prefix it sits under."""
        sys_el = self.top_system
        el = None
        prefix = ""
        for i, part in enumerate(parts):
            if sys_el is None:
                return None
            want = re.sub(r"\s+", " ", part).strip()
            el = next(
                (b for b in sys_el.findall("Block") if re.sub(r"\s+", " ", b.get("Name", "")).strip() == want), None
            )
            if el is None:
                return None
            if i < len(parts) - 1:
                sub = el.find("System")
                sys_el = self._deref(sub) if sub is not None else None
                prefix += want + "/"
        return (el, prefix) if el is not None else None

    def forwarding(self) -> dict[str, str]:
        """Old block path -> new block path, from the library's forwarding table."""
        if self._forward is None:
            self._forward = {}
            el = self.arc.xml("simulink/forwardingTable.xml")
            for entry in el.iter("Entry") if el is not None else []:
                old, new = entry.findtext("OldBlock"), entry.findtext("NewBlock")
                if old and new:
                    self._forward[re.sub(r"\s+", " ", old).strip()] = new
        return self._forward

    def resolve_link(self, source: str, depth: int = 0) -> tuple["_Importer", ET.Element, str] | None:
        parts = [q.replace("\0", "/") for q in source.replace("//", "\0").split("/")]
        if len(parts) < 2 or depth > 6:
            return None
        lib = self if parts[0] == "$bdroot" else self.library(parts[0])
        if lib is None:
            return None
        hit = lib.find_block(parts[1:])
        if hit:
            return lib, hit[0], hit[1]
        # the block moved: the library says where to
        moved = lib.forwarding().get(re.sub(r"\s+", " ", source).strip())
        if moved and moved != source:
            return self.resolve_link(moved, depth + 1)
        # a library that is only a menu: its top-level blocks open other library files
        top = lib.find_block(parts[1:2])
        if top is not None and len(parts) > 2:
            m = re.fullmatch(r"\s*(?:open(?:_system)?\(\s*['\"]([\w.-]+)['\"]\s*\)|([\w.-]+))\s*;?\s*", _params(top[0]).get("OpenFcn", ""))
            other = (m.group(1) or m.group(2)) if m else None
            if other and other != parts[0]:
                return self.resolve_link("/".join([other] + [q.replace("/", "//") for q in parts[2:]]), depth + 1)
        return None

    # -- parts ------------------------------------------------------------------------
    def run(self, name: str) -> Model:
        self.open()
        st = self.stats
        model = Model(name)
        model.root = self.system(self.top_system, "", "")
        for p in self.top.findall("P"):
            if p.get("Name") in ("PreLoadFcn", "PostLoadFcn", "InitFcn", "StartFcn") and (p.text or "").strip():
                model.callbacks[p.get("Name")] = p.text.strip()
        self._config(model, self.top)
        notes = list(dict.fromkeys(st.notes))
        if st.unsupported:
            n = sum(st.unsupported.values())
            listing = ", ".join(f"{k} ({v})" for k, v in st.unsupported.most_common())
            notes.insert(
                0,
                f"{n} of {st.total} blocks have no equivalent in mylab Blocks and were kept as "
                f"placeholders: {listing}. The model cannot be simulated until they are replaced.",
            )
        if st.skipped_lines:
            notes.append(
                f"{st.skipped_lines} connection(s) of kinds mylab Blocks does not have (physical "
                "connections, action ports, state ports) were left out."
            )
        model.notes = notes
        model.import_report = {  # type: ignore[attr-defined]
            "blocks": st.total,
            "converted": st.converted,
            "unsupported": dict(st.unsupported),
            "notes": notes,
            "skipped_lines": st.skipped_lines,
            "linked": st.linked,
            "libraries": sorted(k for k, v in self.libs.items() if v is not None),
        }
        return model

    def _deref(self, sys_el: ET.Element) -> ET.Element:
        ref = sys_el.get("Ref")
        if not ref:
            return sys_el
        if not re.fullmatch(r"[\w.-]+", ref):
            raise ModelError("The .slx file refers to a part with an unexpected name.")
        el = self.arc.xml(f"simulink/systems/{ref}.xml")
        if el is None:
            raise ModelError(f"The .slx file is incomplete: the part simulink/systems/{ref}.xml is missing.")
        return el

    def _read_scripts(self) -> None:
        """Code of MATLAB Function blocks: chart name (block path) -> script."""
        for part in self.arc.names():
            if part.startswith("simulink/stateflow") and part.endswith(".xml"):
                el = self.arc.xml(part)
                if el is None:
                    continue
                for chart in el.iter("chart"):
                    cname = next((p.text or "" for p in chart.findall("P") if p.get("Name") == "name"), "")
                    for p in chart.iter("P"):
                        if p.get("Name") == "script" and p.text:
                            self.scripts[cname] = p.text
                            break

    def _config(self, model: Model, top: ET.Element) -> None:
        found: dict[str, str] = {}
        sources = [top] + [el for part in ("simulink/configSet0.xml",) if (el := self.arc.xml(part)) is not None]
        want = {"StartTime", "StopTime", "SolverName", "Solver", "MaxStep", "RelTol", "AbsTol", "FixedStep"}
        for src in sources:
            for p in src.iter("P"):
                n = p.get("Name")
                if n in want and n not in found and p.text:
                    found[n] = p.text.strip()
        solver = found.pop("SolverName", None) or found.pop("Solver", None)
        if solver:
            model.config["Solver"] = SOLVERS.get(solver.lower(), solver)
        for k, v in found.items():
            if k in model.config:
                model.config[k] = v

    # -- systems ----------------------------------------------------------------------
    def system(self, el: ET.Element, prefix: str, chart_prefix: str) -> System:
        sy = System()
        sid_name: dict[str, str] = {}
        boxes: dict[str, tuple] = {}
        for b_el in el.findall("Block"):
            self._count("total")
            blk = self.block(b_el, prefix, chart_prefix)
            if blk is None:
                self._count("total", -1)
                continue
            blk.name = sy.unique_name(blk.name)
            sy.blocks.append(blk)
            sid_name[b_el.get("SID", "")] = blk.name
            boxes[blk.name] = tuple(blk.pos)
        for ln in el.findall("Line"):
            self.line(ln, sy, sid_name, boxes)
        return sy

    def block(self, el: ET.Element, prefix: str, chart_prefix: str, instance: dict | None = None) -> Block | None:
        btype = el.get("BlockType", "")
        name = re.sub(r"\s+", " ", el.get("Name", btype)).strip() or btype
        own = _params(el)
        if instance:  # a library block used through a link: the values set on the link win
            own.update(instance)
        if own.get("Commented", "off") == "on":
            self.notes.append(f"'{prefix}{name}' is commented out in the file and was left out.")
            return None
        p = dict(IMPORT_DEFAULTS.get(btype, {}))
        p.update(self.defaults.get(btype, {}))
        p.update(own)
        path = f"{prefix}{name}"
        ports = [int(v) for v in _numbers(p.get("Ports", ""))]
        counts = el.find("PortCounts")
        if counts is not None:
            ports = [int(counts.get("in", "0") or 0), int(counts.get("out", "0") or 0)]
        pos = _numbers(p.get("Position", ""))
        box = [pos[0], pos[1], pos[2] - pos[0], pos[3] - pos[1]] if len(pos) == 4 else [0, 0, 0, 0]
        rot = {"270": 1, "180": 2, "90": 3}.get(p.get("BlockRotation", "0").strip(), 0)
        flip = p.get("BlockMirror", "off") == "on"
        if rot == 0 and flip:
            rot, flip = 2, False
        blk = Block(name, btype, {}, box, rot, flip)

        def unsupported(label: str) -> Block:
            self.unsupported[label] += 1
            blk.type = label
            blk.extra["ports"] = [ports[0] if ports else 0, ports[1] if len(ports) > 1 else 0]
            return blk

        target = None
        fixed: dict[str, str] = {}
        if btype == "Reference":
            src_type = p.get("SourceType", "")
            src_block = p.get("SourceBlock", "").replace("\n", " ")
            for cand in (src_type, src_block.split("/")[-1]):
                k = _key(cand)
                if k in SILENT:
                    return None
                if k in REFERENCE_MAP:
                    target, fixed = REFERENCE_MAP[k]
                    break
                if k in VIEWERS:
                    target, fixed = "Viewer", {"NumInputPorts": str(ports[0] if ports else 1), "Kind": cand}
                    self.notes.append(f"'{path}' ({cand}) only draws; it was replaced by a Viewer block that ignores its inputs.")
                    break
                if k.endswith("coordinatetransformationconversion") and (ports or [1, 1])[:2] == [1, 1]:
                    expr = coordinate_conversion(p)
                    if expr:
                        target, fixed = "InterpretedFunction", {"MATLABFcn": expr}
                        break
                if k in KINEMATIC:
                    made = kinematic_model(KINEMATIC[k], name, p)
                    if made is not None:
                        made.pos, made.rot, made.flip = box, rot, flip
                        self._count("converted")
                        self._count("total", len(made.system.blocks))
                        self._count("converted", len(made.system.blocks))
                        self.notes.append(f"'{path}' ({cand}) was replaced by a subsystem with the same equations, built from mylab blocks.")
                        return made
            if target is None:
                link = self.resolve_link(p.get("SourceBlock", "")) if self.root.depth < 12 else None
                if link is None:
                    return unsupported(f"Library link: {src_block or src_type}")
                lib, lib_el, lib_prefix = link
                keep = {
                    k: v
                    for k, v in own.items()
                    if k
                    not in (
                        "SourceBlock",
                        "SourceType",
                        "LibraryVersion",
                        "SourceProductName",
                        "SourceProductBaseCode",
                        "ZOrder",
                        "ContentPreviewEnabled",
                        "Ports",
                        "Position",
                    )
                }
                self.root.depth += 1
                try:
                    inner = lib.block(lib_el, prefix, lib_prefix, keep)
                finally:
                    self.root.depth -= 1
                if inner is None:
                    return None
                inner.name, inner.pos, inner.rot, inner.flip = name, box, rot, flip
                self._count("linked")
                return inner
        elif btype == "SubSystem":
            sf = p.get("SFBlockType", "NONE")
            if sf == "MATLAB Function":
                script = self.scripts.get(chart_prefix + name) or self.scripts.get(path) or self.scripts.get(name)
                if script is None:
                    return unsupported("MATLAB Function (code not found in the file)")
                blk.type = "MATLABFunction"
                blk.params["Script"] = script
                self._count("converted")
                return blk
            if sf not in ("NONE", ""):
                return unsupported(f"Stateflow block ({sf})")
            if p.get("Variant", "off") == "on":
                return unsupported("Variant Subsystem")
            sub = el.find("System")
            blk.type = "Subsystem"
            blk.system = (
                self.system(self._deref(sub), path + "/", chart_prefix + name + "/") if sub is not None else System()
            )
            mask = el.find("Mask")
            if mask is not None and _key(mask.findtext("Type") or "") in SILENT:
                return None
            if mask is not None:
                values = {}
                for mp in mask.findall("MaskParameter"):
                    v = mp.find("Value")
                    mname, kind = mp.get("Name"), mp.get("Type", "edit")
                    if not mname or v is None:
                        continue
                    val = own.get(mname, v.text or "")
                    literal = mp.get("Evaluate", "on") == "off"
                    if kind == "checkbox":
                        values[mname] = "1" if val.strip() == "on" else "0"
                    elif kind == "popup" and not literal:
                        options = [o.text or "" for o in mp.iter("Option")]
                        values[mname] = str(options.index(val) + 1) if val in options else "1"
                    elif kind in ("edit", "popup", "radiobutton"):
                        values[mname] = "'" + val.replace("'", "''") + "'" if literal else (val.strip() or "[]")
                if values:
                    blk.extra["mask"] = values
                init = (mask.findtext("Initialization") or "").strip()
                if init:
                    blk.extra["maskInit"] = init
            self._count("converted")
            return blk
        elif DASHBOARD.search(btype) and not ports:
            self.notes.append(f"'{path}' is a dashboard control ({btype}); it has no signals and was left out.")
            return None
        elif _key(btype) in VIEWERS:
            target, fixed = "Viewer", {"NumInputPorts": str(ports[0] if ports else 1), "Kind": btype}
            self.notes.append(f"'{path}' ({btype}) only draws; it was replaced by a Viewer block that ignores its inputs.")
        elif btype == "Lookup_n-D":
            dims = (p.get("NumberOfTableDimensions") or "2").strip()
            if p.get("TableSource", "Dialog") == "Input port":
                return unsupported("Lookup Table with the table from an input port")
            if not dims.isdigit():
                return unsupported(f"{dims}-D Lookup Table")
            if dims not in ("1", "2"):
                target = "LookupTableND"
                grids = [p.get(f"BreakpointsForDimension{k + 1}", "[1,2,3]").strip() for k in range(int(dims))]
                fixed = {"Breakpoints": ", ".join(grids)}
            else:
                target = "LookupTable1D" if dims == "1" else "LookupTable2D"
            if dims == "1":
                p.setdefault("Table", "tanh([-5:5])")
                p.setdefault("BreakpointsForDimension1", "[-5:5]")
            if p.get("ExtrapMethod", "Linear") not in ("Linear", "Clip"):
                p["ExtrapMethod"] = "Linear"
        if target is None:
            target = TYPE_MAP.get(btype, btype)
        cls = find_type(target)
        if cls is None:
            return unsupported(btype)
        blk.type = cls.type
        known = {q.name: q for q in cls.params}
        rename = PARAM_MAP.get(cls.type, {})
        for k, v in p.items():
            k2 = rename.get(k, k)
            if k2 in known:
                blk.params[k2] = v.strip() if known[k2].kind != "code" else v
        blk.params.update(fixed)
        problem = self._adapt(cls.type, btype, blk, p, ports, own)
        if problem:
            blk.params = {}
            return unsupported(problem)
        self._count("converted")
        return blk

    def _adapt(self, kind: str, btype: str, blk: Block, p: dict, ports: list[int], own: dict) -> str | None:
        """Type-specific fixes. Returns a label when the block uses a feature we do not have."""
        q = blk.params
        n_in = ports[0] if ports else None
        if kind in COUNTED and ports:
            pname, side = COUNTED[kind]
            if pname not in own and len(ports) > side and ports[side] > 0:
                q[pname] = str(ports[side])
        if kind in ("SFunction", "MSFunction"):
            q["NumInputs"] = str(ports[0] if ports else 1)
            q["NumOutputs"] = str(ports[1] if len(ports) > 1 else (0 if ports else 1))
        elif kind == "DirectLookupTable":
            q.setdefault("NumberOfTableDimensions", "2")
            if n_in is not None and "InputsSelectThisObjectFromTable" not in q:
                q["InputsSelectThisObjectFromTable"] = "Element"
        elif kind == "If":
            q["NumInputs"] = str(n_in or 1)
            n_out = ports[1] if len(ports) > 1 else 1
            n_else_if = len([e for e in q.get("ElseIfExpressions", "").split(",") if e.strip()])
            q["ShowElse"] = "on" if n_out > 1 + n_else_if else "off"
        if kind == "Scope":
            if btype == "Record":
                q["NumInputPorts"] = str(n_in or 1)
            q.pop("Title", None)
        elif kind in ("Inport", "Outport"):
            q.setdefault("Port", "1")
            if kind == "Outport" and q.get("InitialOutput", "[]").strip() in ("", "[]"):
                q.pop("InitialOutput", None)
        elif kind == "SineWave":
            if p.get("SineType", "Time based") != "Time based":
                return "Sample-based Sine Wave"
        elif kind == "PulseGenerator":
            if p.get("TimeSource", "Use simulation time") != "Use simulation time":
                return "Pulse Generator with an external time input"
            if p.get("PulseType", "Time based") == "Sample based":
                ts = p.get("SampleTime", "1")
                per, wid, dly = q.get("Period", "2"), q.get("PulseWidth", "1"), q.get("PhaseDelay", "0")
                q["Period"] = f"({per})*({ts})"
                q["PulseWidth"] = f"100*({wid})/({per})"
                q["PhaseDelay"] = f"({dly})*({ts})"
        elif kind == "Integrator":
            if p.get("ShowSaturationPort", "off") == "on":
                return "Integrator with a saturation port"
        elif kind == "DiscreteTimeIntegrator":
            if "Accumulation" in p.get("IntegratorMethod", ""):
                return "Discrete-Time Integrator (accumulation method)"
        elif kind in ("Selector", "Assignment"):
            dims = (p.get("NumberOfDimensions") or "1").strip()
            if dims not in ("1", "2"):
                return f"{kind} with {dims} dimensions"
            opts = p.get("IndexOptions", "Index vector (dialog)")
            if "Starting" in opts or (dims == "2" and "port" in opts) or "Assign all" in opts and "port" in opts:
                return f"{kind} ({opts})"
            q["IndexOptions"] = opts.replace("Assign all", "Select all")
        elif kind == "Switch":
            if "Threshold" not in q.get("Criteria", "u2 >= Threshold") and "~=" not in q.get("Criteria", ""):
                return f"Switch ({q.get('Criteria')})"
        elif kind == "Demux":
            if p.get("BusSelectionMode", "off") == "on":
                return "Demux in bus selection mode"
        elif kind == "ToWorkspace":
            fmt = q.get("SaveFormat", "Timeseries")
            if _key(fmt) not in ("timeseries", "array", "structurewithtime", "structure"):
                q["SaveFormat"] = "Timeseries"
        elif kind == "MathFunction":
            from .blocks_basic import _MATH

            if q.get("Operator", "exp") not in _MATH:
                return f"Math Function ({q.get('Operator')})"
        elif kind == "TrigonometricFunction":
            from .blocks_basic import _TRIG

            if q.get("Operator", "sin") not in _TRIG:
                return f"Trigonometric Function ({q.get('Operator')})"
        elif kind == "LogicalOperator":
            if n_in and q.get("Operator", "AND") != "NOT":
                q["Inputs"] = str(n_in)
        elif kind == "MinMax":
            if n_in:
                q["Inputs"] = str(n_in)
        elif kind == "Gain":
            pass
        elif kind == "PIDController":
            if p.get("TimeDomain", "Continuous-time") != "Continuous-time":
                return "Discrete PID Controller"
            aw = _key(p.get("AntiWindupMode", "none"))
            q["AntiWindupMode"] = {"backcalculation": "back-calculation", "clamping": "clamping"}.get(aw, "none")
            if (
                p.get("ExternalReset", "none") != "none"
                or _key(p.get("InitialConditionSource", "internal")) != "internal"
            ):
                return "PID Controller with a reset or external initial value"
        elif kind == "RateTransition":
            if "OutPortSampleTime" not in q:
                q["OutPortSampleTime"] = "-1"
        elif kind == "Concatenate":
            if q.get("Mode", "Vector") == "Multidimensional array" and q.get(
                "ConcatenateDimension", "1"
            ).strip() not in ("1", "2"):
                return "Concatenate along a third dimension"
        elif kind == "FromWorkspace":
            if _key(q.get("OutputAfterFinalValue", "Extrapolation")) not in (
                "extrapolation",
                "settingtozero",
                "holdingfinalvalue",
            ):
                return f"From Workspace ({q.get('OutputAfterFinalValue')})"
        elif kind == "Delay":
            if (
                p.get("DelayLengthSource", "Dialog") != "Dialog"
                or p.get("InitialConditionSource", "Dialog") != "Dialog"
            ):
                return "Delay with inputs for its length or initial value"
        elif kind == "ManualSwitch":
            if "sw" in q:
                q["sw"] = "1" if q["sw"].strip() in ("1", "on") else "0"
        return None

    # -- lines ------------------------------------------------------------------------
    @staticmethod
    def _end(text: str) -> tuple[str, Any] | None:
        sid, _, port = (text or "").partition("#")
        kind, _, num = port.partition(":")
        if kind in ("in", "out") and num.isdigit():
            return sid, int(num)
        if kind in ("enable", "trigger"):
            return sid, kind
        if kind == "ifaction":  # the control input of an if-action subsystem works like an enable input
            return sid, "enable"
        return None

    def line(self, el: ET.Element, sy: System, sid_name: dict, boxes: dict) -> None:
        p = _params(el)
        src = self._end(p.get("Src", ""))
        if src is None or src[0] not in sid_name:
            if "Src" in p or el.findall("Branch"):
                self._count("skipped_lines")
            return
        sname = sid_name[src[0]]
        start = self._port_xy(sy, sname, src[1], True, boxes)
        label = re.sub(r"\s+", " ", p.get("Name", "")).strip()
        if label.startswith("<"):
            label = ""

        def walk(node: ET.Element, at: tuple, trail: list) -> None:
            q = _params(node)
            pts = _numbers(q.get("Points", ""))
            trail = list(trail)
            for i in range(0, len(pts) - 1, 2):
                at = (at[0] + pts[i], at[1] + pts[i + 1])
                trail.append([at[0], at[1]])
            if "Dst" in q:
                dst = self._end(q["Dst"])
                if dst is None or dst[0] not in sid_name:
                    self._count("skipped_lines")
                else:
                    target = (sid_name[dst[0]], dst[1])
                    if any(ln.dst == target for ln in sy.lines):
                        self._count("skipped_lines")
                    else:
                        sy.lines.append(Line((sname, src[1]), target, label, trail))
            for br in node.findall("Branch"):
                walk(br, at, trail)

        walk(el, start, [])

    @staticmethod
    def _port_xy(sy: System, name: str, port: Any, out: bool, boxes: dict) -> tuple:
        from .engine import port_count

        x, y, w, h = boxes.get(name, (0, 0, 30, 30))
        b = sy.block(name)
        n = max(
            port_count(b, "out" if out else "in")
            if b is not None and find_type(b.type)
            else (b.extra.get("ports", [1, 1])[1 if out else 0] if b is not None else 1),
            1,
        )
        k = port if isinstance(port, int) else 1
        rot = b.rot if b is not None else 0
        frac = (min(k, n) - 0.5) / n
        if rot == 0:
            return (x + w, y + h * frac)
        if rot == 2:
            return (x, y + h * frac)
        if rot == 1:
            return (x + w * frac, y + h)
        return (x + w * frac, y)


def import_slx(path: str, name: str | None = None) -> Model:
    """Read an ``.slx`` file into a :class:`Model`. See ``model.notes`` for what was not converted."""
    import os

    base = name or os.path.splitext(os.path.basename(path))[0]
    base = re.sub(r"\W", "_", base) or "model"
    model = _Importer(path).run(base)
    model.file = path
    return model
