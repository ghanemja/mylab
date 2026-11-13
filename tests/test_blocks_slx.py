"""mylab Blocks: the .slx reader.

The first part uses small files built here, so it always runs. The second
part reads the 82 real model files listed in datasets/matlab-corpus/models.jsonl
when they have been downloaded (python datasets/matlab-corpus/tools/fetch_models.py);
it is skipped otherwise.
"""

import json
import os
import zipfile
from pathlib import Path

import numpy as np
import pytest

from mylab.api import Session
from mylab.blocks import ModelError, compile_model, load_model, simulate
from mylab.blocks.import_report import check_file, defaults_table, main, summarize, survey
from mylab.blocks.slx import import_slx

ROOT = Path(__file__).resolve().parent.parent


def P(name, value):
    value = str(value).replace("&", "&amp;").replace("<", "&lt;")
    return f'<P Name="{name}">{value}</P>'


def block(kind, name, sid, pos, *params, extra=""):
    return f'<Block BlockType="{kind}" Name="{name}" SID="{sid}">{P("Position", pos)}{"".join(params)}{extra}</Block>'


def line(src, dst=None, points=None, name=None, branches=()):
    parts = [P("Src", src)]
    if name:
        parts.append(P("Name", name))
    if points:
        parts.append(P("Points", points))
    if dst:
        parts.append(P("Dst", dst))
    for b in branches:
        parts.append("<Branch>" + (P("Points", b[1]) if len(b) > 1 else "") + P("Dst", b[0]) + "</Branch>")
    return "<Line>" + "".join(parts) + "</Line>"


def write_slx(path, system, *, config="", parts=None, defaults="", split=None):
    """A minimal .slx: everything in blockdiagram.xml, or (split) one part per system."""
    with zipfile.ZipFile(path, "w") as z:
        if split is None:
            z.writestr(
                "simulink/blockdiagram.xml",
                f'<?xml version="1.0" encoding="utf-8"?><ModelInformation Version="1.0"><Model>'
                f"{defaults}{config}<System>{system}</System></Model></ModelInformation>",
            )
        else:
            z.writestr(
                "simulink/blockdiagram.xml",
                '<?xml version="1.0"?><ModelInformation><Model><System Ref="system_root"/></Model></ModelInformation>',
            )
            z.writestr("simulink/systems/system_root.xml", f'<?xml version="1.0"?><System>{system}</System>')
            for name, body in split.items():
                z.writestr(f"simulink/systems/{name}.xml", f'<?xml version="1.0"?><System>{body}</System>')
        for name, text in (parts or {}).items():
            z.writestr(name, text)
    return str(path)


FIRST_ORDER = (
    block("Step", "Step", 1, "[40, 40, 70, 70]", P("Time", "0"))
    + block("Sum", "Sum", 2, "[110, 45, 130, 65]", P("Ports", "[2, 1]"), P("Inputs", "|+-"), P("IconShape", "round"))
    + block("Gain", "K", 3, "[160, 40, 190, 70]", P("Gain", "2"))
    + block("Integrator", "Int", 4, "[230, 40, 260, 70]")
    + block("Scope", "Scope", 5, "[330, 39, 360, 71]")
    + block("Outport", "Out1", 6, "[330, 100, 360, 114]")
    + line("1#out:1", "2#in:1")
    + line("2#out:1", "3#in:1", name="error")
    + line("3#out:1", "4#in:1")
    + line("4#out:1", points="[30, 0]", branches=[("5#in:1",), ("6#in:1", "[0, 52]"), ("2#in:2", "[0, 90; -170, 0]")])
)


# -- small files made here ------------------------------------------------------------------------


def test_import_blocks_lines_branches_and_simulate(tmp_path):
    m = import_slx(write_slx(tmp_path / "fo.slx", FIRST_ORDER))
    assert m.name == "fo" and [b.type for b in m.root.blocks] == [
        "Step",
        "Sum",
        "Gain",
        "Integrator",
        "Scope",
        "Outport",
    ]
    assert m.root.block("K").params == {"Gain": "2"} and m.root.block("Sum").params["Inputs"] == "|+-"
    assert m.root.block("K").pos == [160, 40, 30, 30]
    ends = sorted((ln.src, ln.dst) for ln in m.root.lines)
    assert ends == [
        (("Int", 1), ("Out1", 1)),
        (("Int", 1), ("Scope", 1)),
        (("Int", 1), ("Sum", 2)),
        (("K", 1), ("Int", 1)),
        (("Step", 1), ("Sum", 1)),
        (("Sum", 1), ("K", 1)),
    ]
    assert [ln.name for ln in m.root.lines if ln.src[0] == "Sum"] == ["error"]
    fb = next(ln for ln in m.root.lines if ln.dst == ("Sum", 2))
    assert fb.points == [[290, 55], [290, 145], [120, 145]]  # relative moves in the file become absolute corners
    assert m.import_report["unsupported"] == {} and m.notes == []
    r = simulate(m, config={"StopTime": "3", "RelTol": "1e-8", "AbsTol": "1e-10"})
    assert np.max(np.abs(r.y[:, 0] - (1 - np.exp(-2 * r.t)))) < 1e-7


def test_import_split_layout_subsystems_and_settings(tmp_path):
    inner = (
        block("Inport", "u", "7::1", "[20, 30, 50, 44]")
        + block("Gain", "Twice", "7::2", "[90, 20, 120, 50]", P("Gain", "k"))
        + block("Outport", "y", "7::3", "[170, 30, 200, 44]")
        + line("7::1#out:1", "7::2#in:1")
        + line("7::2#out:1", "7::3#in:1")
    )
    root = (
        block("Constant", "C", 1, "[30, 30, 60, 60]", P("Value", "3"))
        + block(
            "SubSystem",
            "Sub",
            7,
            "[100, 25, 180, 65]",
            extra='<Mask><MaskParameter Name="k" Type="edit"><Prompt>gain</Prompt>'
            '<Value>2</Value></MaskParameter><MaskParameter Name="mode" Type="popup" Evaluate="off"><Value>fast mode</Value>'  # noqa: E501
            '</MaskParameter></Mask><System Ref="system_7"/>',
        )
        + block("Outport", "Out", 9, "[240, 38, 270, 52]")
        + line("1#out:1", "7#in:1")
        + line("7#out:1", "9#in:1")
    )
    config = (
        '<?xml version="1.0"?><ConfigSet><Object ClassName="Simulink.SolverCC">'
        + P("StartTime", "0.0")
        + P("StopTime", "2.5")
        + P("SolverName", "FixedStepDiscrete")
        + P("FixedStep", "0.5")
        + "</Object></ConfigSet>"
    )
    m = import_slx(
        write_slx(tmp_path / "split.slx", root, split={"system_7": inner}, parts={"simulink/configSet0.xml": config})
    )
    sub = m.root.block("Sub")
    assert [b.name for b in sub.system.blocks] == ["u", "Twice", "y"] and sub.extra["mask"] == {
        "k": "2",
        "mode": "'fast mode'",
    }
    assert (
        m.config["StopTime"] == "2.5" and m.config["Solver"] == "FixedStepDiscrete" and m.config["FixedStep"] == "0.5"
    )
    r = simulate(m)
    assert np.allclose(r.t, np.arange(6) * 0.5) and np.all(r.y == 6)


def test_unsupported_blocks_are_named_not_dropped(tmp_path):
    body = (
        block("Constant", "C", 1, "[0, 0, 30, 30]")
        + block("ForIterator", "Engine", 2, "[60, 0, 120, 40]", P("Ports", "[1, 2]"))
        + block(
            "Reference",
            "Arm",
            3,
            "[60, 80, 120, 120]",
            P("Ports", "[1, 1]"),
            P("SourceBlock", "robotlib/Arm"),
            P("SourceType", "Arm"),
        )
        + block("Lookup_n-D", "Map", 4, "[150, 0, 200, 40]", P("TableSource", "Input port"))
        + block("Terminator", "T", 5, "[240, 0, 260, 20]")
        + line("1#out:1", "2#in:1")
        + line("2#out:1", "5#in:1")
        + line("2#out:2", "3#in:1")
        + "<Line>"
        + P("Src", "3#rconn:1")
        + P("Dst", "4#lconn:1")
        + "</Line>"
    )
    m = import_slx(write_slx(tmp_path / "odd.slx", body))
    assert m.import_report["unsupported"] == {
        "ForIterator": 1,
        "Library link: robotlib/Arm": 1,
        "Lookup Table with the table from an input port": 1,
    }
    assert (
        m.import_report["blocks"] == 5 and m.import_report["converted"] == 2 and m.import_report["skipped_lines"] == 1
    )
    assert "3 of 5 blocks have no equivalent" in m.notes[0] and "ForIterator (1)" in m.notes[0]
    assert "physical" in m.notes[1]
    eng = m.root.block("Engine")
    assert eng.type == "ForIterator" and eng.extra["ports"] == [1, 2] and len(m.root.lines) == 3
    errors = [d["message"] for d in compile_model(m, strict=False).errors()]
    assert any("no block type 'ForIterator'" in e for e in errors) and len(errors) == 3
    row = check_file(str(tmp_path / "odd.slx"))
    assert row["import"] == "partial" and row["simulate"] == "no: unsupported blocks"


def test_library_references_matlab_function_and_defaults(tmp_path):
    chart = (
        '<?xml version="1.0"?><Stateflow><machine><Children><chart id="2">'
        + P("name", "Sub/Square")
        + "<Children><state><eml>"
        + P("script", "function y = fcn(u)\ny = u^2;\n")
        + "</eml></state></Children></chart></Children></machine></Stateflow>"
    )
    inner = (
        block("Inport", "u", "9::1", "[0, 0, 30, 14]")
        + block(
            "SubSystem",
            "Square",
            "9::2",
            "[60, 0, 120, 40]",
            P("SFBlockType", "MATLAB Function"),
            extra="<System></System>",
        )
        + block("Outport", "y", "9::3", "[160, 0, 190, 14]")
        + line("9::1#out:1", "9::2#in:1")
        + line("9::2#out:1", "9::3#in:1")
    )
    body = (
        block("Constant", "C", 1, "[0, 0, 30, 30]", P("Value", "3"))
        + block(
            "Reference",
            "ToDeg",
            2,
            "[50, 0, 80, 30]",
            P("SourceBlock", "simulink_extras/Transformations/Radians\nto Degrees"),
            P("SourceType", "RadiansToDegrees"),
        )
        + block(
            "Reference",
            "Cmp",
            3,
            "[100, 0, 130, 30]",
            P("SourceBlock", "simulink/Logic and Bit\nOperations/Compare\nTo Constant"),
            P("SourceType", "Compare To Constant"),
            extra="<InstanceData>" + P("relop", ">") + P("const", "100") + "</InstanceData>",
        )
        + block("SubSystem", "Sub", 9, "[50, 60, 110, 100]", extra=f"<System>{inner}</System>")
        + block("Demux", "Split", 4, "[150, 60, 155, 140]", P("Ports", "[1, 4]"))
        + block(
            "Reference",
            "Info",
            5,
            "[0, 200, 100, 240]",
            P("SourceBlock", "simulink/Model-Wide\nUtilities/Model Info"),
            P("SourceType", "CMBlock"),
        )
        + block("Gain", "Off", 6, "[0, 300, 30, 330]", P("Commented", "on"))
        + block("Outport", "O1", 7, "[200, 0, 230, 14]")
        + block("Outport", "O2", 8, "[200, 60, 230, 74]", P("Port", "2"))
        + line("1#out:1", branches=[("2#in:1",), ("9#in:1",)])
        + line("2#out:1", "3#in:1")
        + line("3#out:1", "7#in:1")
        + line("9#out:1", "8#in:1")
    )
    defaults = (
        '<BlockParameterDefaults><Block BlockType="Demux">' + P("Outputs", "4") + "</Block></BlockParameterDefaults>"
    )
    m = import_slx(write_slx(tmp_path / "refs.slx", body, parts={"simulink/stateflow.xml": chart}, defaults=defaults))
    assert [b.name for b in m.root.blocks] == [
        "C",
        "ToDeg",
        "Cmp",
        "Sub",
        "Split",
        "O1",
        "O2",
    ]  # the note block and the commented block are gone
    assert m.root.block("ToDeg").type == "Gain" and m.root.block("ToDeg").params["Gain"] == "180/pi"
    assert m.root.block("Cmp").params == {"relop": ">", "const": "100"}
    sq = m.root.block("Sub").system.block("Square")
    assert sq.type == "MATLABFunction" and "y = u^2" in sq.params["Script"]
    assert m.root.block("Split").params["Outputs"] == "4"
    assert any("commented out" in n for n in m.notes)
    r = simulate(m, config={"StopTime": "1"})
    assert np.allclose(r.y[-1], [1.0, 9.0])  # 3 rad = 171.9 degrees > 100; 3 squared


def test_links_to_a_library_file_next_to_the_model_are_followed(tmp_path):
    lib_inner = (
        block("Inport", "in", "1::1", "[0, 0, 30, 14]")
        + block("Gain", "G", "1::2", "[60, 0, 90, 30]", P("Gain", "factor"))
        + block("Outport", "out", "1::3", "[130, 0, 160, 14]")
        + line("1::1#out:1", "1::2#in:1")
        + line("1::2#out:1", "1::3#in:1")
    )
    lib = block(
        "SubSystem",
        "Scale it",
        1,
        "[0, 0, 80, 40]",
        extra='<Mask><MaskParameter Name="factor" Type="edit"><Value>1</Value>'
        f"</MaskParameter></Mask><System>{lib_inner}</System>",
    )
    with zipfile.ZipFile(tmp_path / "mylib.slx", "w") as z:
        z.writestr(
            "simulink/blockdiagram.xml",
            f'<?xml version="1.0"?><ModelInformation><Library><System>{lib}</System></Library></ModelInformation>',
        )
    body = (
        block("Constant", "C", 1, "[0, 0, 30, 30]", P("Value", "4"))
        + block(
            "Reference",
            "Scaled",
            2,
            "[60, 0, 140, 40]",
            P("SourceBlock", "mylib/Scale it"),
            P("SourceType", "SubSystem"),
            extra="<InstanceData>" + P("factor", "2.5") + "</InstanceData>",
        )
        + block(
            "Reference",
            "Gone",
            3,
            "[60, 80, 140, 120]",
            P("Ports", "[1, 1]"),
            P("SourceBlock", "otherlib/Thing"),
            P("SourceType", "x"),
        )
        + block("Outport", "O", 4, "[200, 10, 230, 24]")
        + line("1#out:1", "2#in:1")
        + line("2#out:1", "4#in:1")
    )
    m = import_slx(write_slx(tmp_path / "uses.slx", body))
    sub = m.root.block("Scaled")
    assert sub.type == "Subsystem" and sub.pos == [60, 0, 80, 40] and sub.extra["mask"] == {"factor": "2.5"}
    assert m.import_report["libraries"] == ["mylib"] and m.import_report["linked"] == 1
    assert m.import_report["unsupported"] == {"Library link: otherlib/Thing": 1}
    m.root.blocks.remove(m.root.block("Gone"))
    assert simulate(m, config={"StopTime": "1"}).y[-1, 0] == 10


def test_load_system_and_sim_read_slx_files(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    write_slx(tmp_path / "plant.slx", FIRST_ORDER)
    s = Session()
    r = s.run(
        "o = sim('plant', 'StopTime', '3'); y = o.yout(end); n = numel(find_system('plant')); save_system('plant');"
    )
    assert r.error is None, r.error
    assert abs(float(s.interp.get_var("y")[0, 0]) - (1 - np.exp(-6))) < 1e-3 and s.interp.get_var("n")[0, 0] == 7
    again = load_model(str(tmp_path / "plant.mlb.json"))  # saved in mylab's own format
    assert [b.name for b in again.root.blocks] == ["Step", "Sum", "K", "Int", "Scope", "Out1"]
    write_slx(tmp_path / "odd.slx", block("ForIterator", "X", 1, "[0, 0, 30, 30]"))
    r = s.run("load_system('odd');")
    assert "Warning: 1 of 1 blocks have no equivalent in mylab Blocks" in r.output and "ForIterator (1)" in r.output


def test_bad_archives_are_refused(tmp_path):
    p = tmp_path / "notzip.slx"
    p.write_text("Model { }")
    with pytest.raises(ModelError, match="not a readable .slx file"):
        import_slx(str(p))
    with zipfile.ZipFile(tmp_path / "empty.slx", "w") as z:
        z.writestr("readme.txt", "hello")
    with pytest.raises(ModelError, match="no simulink/blockdiagram.xml"):
        import_slx(str(tmp_path / "empty.slx"))
    with zipfile.ZipFile(tmp_path / "entity.slx", "w") as z:
        z.writestr(
            "simulink/blockdiagram.xml",
            '<?xml version="1.0"?><!DOCTYPE x [<!ENTITY a "aaaa">]><ModelInformation><Model><System>'
            "</System></Model></ModelInformation>",
        )
    with pytest.raises(ModelError, match="declares XML entities"):
        import_slx(str(tmp_path / "entity.slx"))
    with zipfile.ZipFile(tmp_path / "broken.slx", "w") as z:
        z.writestr("simulink/blockdiagram.xml", "<ModelInformation><Model>")
    with pytest.raises(ModelError, match="not valid XML"):
        import_slx(str(tmp_path / "broken.slx"))
    with zipfile.ZipFile(tmp_path / "ref.slx", "w") as z:
        z.writestr(
            "simulink/blockdiagram.xml",
            '<ModelInformation><Model><System Ref="../../etc/passwd"/></Model></ModelInformation>',
        )
    with pytest.raises(ModelError, match="unexpected name"):
        import_slx(str(tmp_path / "ref.slx"))
    with pytest.raises(ModelError, match=".mdl"):
        load_model(str(tmp_path / "old.mdl"))


def test_import_report_tool(tmp_path, capsys):
    write_slx(tmp_path / "good.slx", FIRST_ORDER)
    write_slx(
        tmp_path / "needs.slx",
        block("Constant", "C", 1, "[0, 0, 30, 30]", P("Value", "plant.gain"))
        + block("Terminator", "T", 2, "[60, 0, 80, 20]")
        + line("1#out:1", "2#in:1"),
    )
    write_slx(tmp_path / "odd.slx", block("ForIterator", "X", 1, "[0, 0, 30, 30]"))
    (tmp_path / "junk.slx").write_text("not a zip")
    rows = {r["file"]: r for r in survey(str(tmp_path))}
    assert rows["good.slx"]["import"] == "full" and rows["good.slx"]["simulate"] == "yes"
    assert (
        rows["needs.slx"]["import"] == "full"
        and rows["needs.slx"]["simulate"] == "no: needs workspace variables or toolbox functions"
    )
    assert rows["odd.slx"]["import"] == "partial" and rows["odd.slx"]["unsupported"] == {"ForIterator": 1}
    assert rows["junk.slx"]["import"] == "failed" and "not a readable" in rows["junk.slx"]["detail"]
    s = summarize(list(rows.values()))
    assert s["import"] == {"full": 2, "partial": 1, "failed": 1} and s["simulate"]["yes"] == 1
    assert main([str(tmp_path), "--json", str(tmp_path / "out.json")]) == 0
    text = capsys.readouterr().out
    assert "4 files. Import: failed 1, full 2, partial 1" in text and "unsupported: ForIterator (1)" in text
    assert json.loads((tmp_path / "out.json").read_text())["summary"]["files"] == 4


# -- structure and set-up of imported models -------------------------------------------------------


def _io_system(prefix, inner):
    return (
        block("Inport", "u", f"{prefix}::1", "[0, 0, 30, 14]")
        + inner
        + block("Outport", "y", f"{prefix}::3", "[160, 0, 190, 14]")
        + line(f"{prefix}::1#out:1", f"{prefix}::2#in:1")
        + line(f"{prefix}::2#out:1", f"{prefix}::3#in:1")
    )


def test_variant_subsystems_run_the_active_choice(tmp_path):
    def choice(name, sid, gain, control):
        inner = _io_system(sid, block("Gain", "K", f"{sid}::2", "[60, 0, 90, 30]", P("Gain", gain)))
        return block("SubSystem", name, sid, "[60, 0, 120, 40]", P("VariantControl", control), extra=f"<System>{inner}</System>")

    variant = (
        block("Inport", "u", "20::1", "[0, 0, 30, 14]")
        + choice("Slow", "21", "2", "MODE == 1")
        + choice("Fast", "22", "30", "fast_variant")
        + block("Outport", "y", "20::3", "[160, 0, 190, 14]")
    )  # no lines inside: the choices are wired to the ports by name
    body = (
        block("Constant", "C", 1, "[0, 0, 30, 30]", P("Value", "5"))
        + block("SubSystem", "Plant", 20, "[60, 0, 140, 40]", P("Variant", "on"), extra=f"<System>{variant}</System>")
        + block("Outport", "O", 3, "[200, 10, 230, 24]")
        + line("1#out:1", "20#in:1")
        + line("20#out:1", "3#in:1")
    )
    m = import_slx(write_slx(tmp_path / "var.slx", body))
    assert m.import_report["unsupported"] == {}
    assert m.root.block("Plant").extra["variants"] == {"Slow": "MODE == 1", "Fast": "fast_variant"}
    s = Session()
    s.run("MODE = 1; fast_variant = Simulink.Variant('MODE == 2');")
    assert simulate(m, s.interp, config={"StopTime": "1"}).y[-1, 0] == 10
    s.run("MODE = 2;")
    assert simulate(m, s.interp, config={"StopTime": "1"}).y[-1, 0] == 150
    s.run("MODE = 3;")
    with pytest.raises(Exception, match="no choice of this variant subsystem is active"):
        simulate(m, s.interp, config={"StopTime": "1"})
    again = type(m).from_json(json.loads(m.dumps()))  # the choice survives saving in mylab's format
    s.run("MODE = 1;")
    assert simulate(again, s.interp, config={"StopTime": "1"}).y[-1, 0] == 10


def test_model_callbacks_and_mask_initialization_run_before_the_simulation(tmp_path):
    inner = _io_system("7", block("Gain", "K", "7::2", "[60, 0, 90, 30]", P("Gain", "twice")))
    mask = (
        '<Mask><MaskParameter Name="k" Type="edit"><Value>base_gain</Value></MaskParameter>'
        "<Initialization>twice = 2 * k;\nerror('later lines do not matter');</Initialization></Mask>"
    )
    body = (
        block("Constant", "C", 1, "[0, 0, 30, 30]", P("Value", "offset"))
        + block("SubSystem", "Scale", 7, "[60, 0, 140, 40]", extra=f"{mask}<System>{inner}</System>")
        + block("Outport", "O", 3, "[200, 10, 230, 24]")
        + line("1#out:1", "7#in:1")
        + line("7#out:1", "3#in:1")
    )
    (tmp_path / "plant_data.m").write_text("base_gain = 4;\n")
    cfg = P("PreLoadFcn", "plant_data") + P("InitFcn", "offset = 3;") + P("StopFcn", "error('never run')")
    m = import_slx(write_slx(tmp_path / "cb.slx", body, config=cfg))
    assert m.callbacks == {"PreLoadFcn": "plant_data", "InitFcn": "offset = 3;"}
    assert m.root.block("Scale").extra["maskInit"].startswith("twice = 2 * k;")
    assert simulate(m, config={"StopTime": "1"}).y[-1, 0] == 24  # the script next to the model is found
    again = type(m).from_json(json.loads(m.dumps()))
    assert again.callbacks == m.callbacks
    m.callbacks["InitFcn"] = "offset = undefined_thing + 1;"
    with pytest.raises(Exception) as info:
        simulate(m, config={"StopTime": "1"})
    assert "offset" in str(info.value) and "InitFcn callback stopped" in str(info.value)


def test_library_blocks_that_moved_or_are_replaced_by_equivalents(tmp_path):
    lib_inner = _io_system("1", block("Gain", "G", "1::2", "[60, 0, 90, 30]", P("Gain", "7")))
    lib = block("SubSystem", "New name", 1, "[0, 0, 80, 40]", extra=f"<System>{lib_inner}</System>")
    with zipfile.ZipFile(tmp_path / "movedlib.slx", "w") as z:
        z.writestr(
            "simulink/blockdiagram.xml",
            f'<?xml version="1.0"?><ModelInformation><Library><System>{lib}</System></Library></ModelInformation>',
        )
        z.writestr(
            "simulink/forwardingTable.xml",
            '<?xml version="1.0"?><ForwardingTable><Entry><OldBlock>movedlib/Old name</OldBlock>'
            "<NewBlock>movedlib/New name</NewBlock></Entry></ForwardingTable>",
        )
    assert import_slx(str(tmp_path / "movedlib.slx")).library is True
    body = (
        block("Constant", "C", 1, "[0, 0, 30, 30]", P("Value", "[1 0.5]"))
        + block("Reference", "Old", 2, "[60, 0, 140, 40]", P("SourceBlock", "movedlib/Old name"), P("SourceType", "SubSystem"))
        + block("Outport", "O", 3, "[200, 10, 230, 24]")
        + block(
            "Reference",
            "Car",
            4,
            "[60, 100, 180, 160]",
            P("Ports", "[2, 2]"),
            P("SourceBlock", "robotmobilelib/Bicycle Kinematic Model"),
            P("SourceType", "Bicycle Kinematic Model"),
            extra="<InstanceData>" + P("WheelBase", "2") + P("VehicleInputs", "Vehicle Speed & Steering Angle") + "</InstanceData>",
        )
        + block("Constant", "v", 5, "[0, 100, 30, 120]", P("Value", "1"))
        + block("Constant", "steer", 6, "[0, 140, 30, 160]", P("Value", "0"))
        + block("Outport", "pose", 7, "[220, 110, 250, 124]", P("Port", "2"))
        + block("Terminator", "T", 8, "[220, 140, 240, 160]")
        + block(
            "Reference",
            "Movie",
            9,
            "[220, 200, 260, 240]",
            P("Ports", "[1]"),
            P("SourceBlock", "visionsinks/Video Viewer"),
            P("SourceType", "Video Viewer"),
        )
        + block("SliderBlock", "Knob", 10, "[0, 300, 60, 330]")
        + line("1#out:1", "2#in:1")
        + line("2#out:1", "3#in:1")
        + line("5#out:1", "4#in:1")
        + line("6#out:1", "4#in:2")
        + line("4#out:1", branches=[("7#in:1",), ("9#in:1",)])
        + line("4#out:2", "8#in:1")
    )
    path = write_slx(tmp_path / "uses2.slx", body)
    m = import_slx(path)
    assert m.library is False and m.import_report["unsupported"] == {}
    assert m.root.block("Old").system.block("G").params["Gain"] == "7"  # found through the forwarding table
    car = m.root.block("Car")
    assert car.type == "Subsystem" and car.extra["mask"]["L"] == "2"
    assert m.root.block("Movie").type == "Viewer" and m.root.block("Knob") is None
    notes = " ".join(m.notes)
    assert "replaced by a subsystem with the same equations" in notes and "only draws" in notes and "dashboard" in notes
    r = simulate(m, config={"StopTime": "2", "RelTol": "1e-8"})
    assert np.allclose(r.y[-1], [7, 3.5, 2, 0, 0], atol=1e-6)  # driving straight at 1 m/s for 2 s
    row = check_file(path)
    assert row["simulate"] == "yes (with stand-ins)" and row["standins"] == 1 and row["library"] is False
    lib_row = check_file(str(tmp_path / "movedlib.slx"))
    assert lib_row["import"] == "full" and lib_row["simulate"] == "library: not a model to run"


def test_if_blocks_s_functions_and_large_tables_are_imported(tmp_path):
    def action(sid, gain):
        inner = (
            block("Inport", "u", f"{sid}::1", "[0, 0, 30, 14]")
            + block("ActionPort", "Action Port", f"{sid}::4", "[60, 40, 90, 60]")
            + block("Gain", "K", f"{sid}::2", "[60, 0, 90, 30]", P("Gain", gain))
            + block("Outport", "y", f"{sid}::3", "[160, 0, 190, 14]")
            + line(f"{sid}::1#out:1", f"{sid}::2#in:1")
            + line(f"{sid}::2#out:1", f"{sid}::3#in:1")
        )
        return block("SubSystem", f"A{sid}", sid, "[100, 0, 160, 40]", extra=f"<System>{inner}</System>")

    body = (
        block("Constant", "C", 1, "[0, 0, 30, 30]", P("Value", "-2"))
        + block("If", "If", 2, "[40, 0, 80, 40]", P("Ports", "[1, 2]"), P("IfExpression", "u1 < 0"))
        + action(11, "10")
        + action(12, "100")
        + block("Merge", "Merge", 3, "[200, 0, 220, 40]", P("Ports", "[2, 1]"))
        + block("Outport", "O", 4, "[260, 10, 290, 24]")
        + block("S-Function", "Plant", 5, "[0, 100, 60, 140]", P("Ports", "[1, 1]"), P("FunctionName", "missing_mex"))
        + block("Terminator", "T", 6, "[100, 100, 120, 120]")
        + block(
            "Lookup_n-D",
            "Map",
            7,
            "[0, 200, 60, 240]",
            P("Ports", "[3, 1]"),
            P("NumberOfTableDimensions", "3"),
            P("Table", "reshape(0:7, 2, 2, 2)"),
            P("BreakpointsForDimension1", "[0 1]"),
            P("BreakpointsForDimension2", "[0 1]"),
            P("BreakpointsForDimension3", "[0 1]"),
        )
        + block("Outport", "O2", 8, "[100, 210, 130, 224]", P("Port", "2"))
        + line("1#out:1", branches=[("2#in:1",), ("11#in:1",), ("12#in:1",), ("5#in:1",), ("7#in:1",), ("7#in:2",), ("7#in:3",)])
        + line("2#out:1", "11#ifaction")
        + line("2#out:2", "12#ifaction")
        + line("11#out:1", "3#in:1")
        + line("12#out:1", "3#in:2")
        + line("3#out:1", "4#in:1")
        + line("5#out:1", "6#in:1")
        + line("7#out:1", "8#in:1")
    )
    m = import_slx(write_slx(tmp_path / "ifs.slx", body))
    assert m.import_report["unsupported"] == {}
    assert m.root.block("Plant").type == "SFunction" and m.root.block("Map").type == "LookupTableND"
    assert m.root.block("Map").params["Breakpoints"].count("[0 1]") == 3
    # the S-function has no .m file here: it is compiled code, which is reported by name and not replaced
    with pytest.raises(Exception, match="missing_mex"):
        simulate(m, config={"StopTime": "1"})
    row = check_file(str(tmp_path / "ifs.slx"))
    assert row["simulate"].startswith("no:") and "missing_mex" in row["detail"]
    m.root.blocks.remove(m.root.block("Plant"))
    m.root.blocks.remove(m.root.block("T"))
    m.root.lines = [ln for ln in m.root.lines if "Plant" not in (ln.src[0], ln.dst[0])]
    r = simulate(m, config={"StopTime": "1"})
    assert r.y[-1, 0] == -20 and r.y[-1, 1] == pytest.approx(-14)  # the if branch; the table continued outside


def test_report_finds_set_up_scripts_in_the_sources_of_the_repository(tmp_path):
    repo = tmp_path / "files" / "owner__repo" / "models"
    repo.mkdir(parents=True)
    write_slx(
        repo / "needs.slx",
        block("Constant", "C", 1, "[0, 0, 30, 30]", P("Value", "plant.gain * helper_fn(2)"))
        + block("Outport", "O", 2, "[60, 0, 80, 20]")
        + line("1#out:1", "2#in:1"),
    )
    src = tmp_path / "sources" / "owner__repo"
    (src / "scripts").mkdir(parents=True)
    (src / "lib").mkdir()
    (src / "scripts" / "init_plant.m").write_text("plant.gain = 3;\n")
    (src / "scripts" / "other.m").write_text("unrelated = 1;\n")
    (src / "lib" / "helper_fn.m").write_text("function y = helper_fn(x)\ny = 10 * x;\n")
    rows = survey(str(tmp_path / "files"), sources=str(tmp_path / "sources"), keep_results=True)
    assert len(rows) == 1 and rows[0]["simulate"] == "yes", rows[0]
    assert rows[0]["setup"] == [os.path.join("scripts", "init_plant.m")]
    assert rows[0]["result"].y[-1, 0] == 60
    assert summarize(rows)["with_setup"] == 1
    without = survey(str(tmp_path / "files"))
    assert without[0]["simulate"] == "no: needs workspace variables or toolbox functions"
    assert without[0]["missing"] == ["plant"]


def test_defaults_table_compares_listed_defaults_with_the_importer(tmp_path, capsys):
    defaults = (
        '<BlockParameterDefaults><Block BlockType="Gain">'
        + P("Gain", "1")
        + '</Block><Block BlockType="Constant">'
        + P("Value", "7")
        + "</Block></BlockParameterDefaults>"
    )
    write_slx(tmp_path / "d.slx", FIRST_ORDER, defaults=defaults)
    rows = {(r["block"], r["parameter"]): r for r in defaults_table(str(tmp_path))}
    assert rows[("Gain", "Gain")]["agree"] and rows[("Gain", "Gain")]["common"] == "2"
    assert not rows[("Constant", "Value")]["agree"] and rows[("Constant", "Value")]["ours"] == "1"
    assert main([str(tmp_path), "--defaults"]) == 0
    assert "!! Constant.Value: files list '7'" in capsys.readouterr().out


# -- the real files (only when they have been downloaded) ------------------------------------------


def _cache():
    d = Path(os.environ.get("MYLAB_SLX_CACHE") or Path.home() / ".cache" / "mylab-corpus-models")
    rows = [json.loads(x) for x in (ROOT / "datasets" / "matlab-corpus" / "models.jsonl").read_text().splitlines()]
    have = all((d / r["repo"].replace("/", "__") / r["path"]).is_file() for r in rows)
    return d if have else None


needs_corpus = pytest.mark.skipif(
    _cache() is None, reason="model files not downloaded: run datasets/matlab-corpus/tools/fetch_models.py"
)


def test_models_list_matches_the_manifest():
    rows = [json.loads(x) for x in (ROOT / "datasets" / "matlab-corpus" / "models.jsonl").read_text().splitlines()]
    pinned = {}
    for x in (ROOT / "datasets" / "matlab-corpus" / "manifest.jsonl").read_text().splitlines():
        r = json.loads(x)
        pinned[r["repo"]] = r
    assert len(rows) == 82 and len({(r["repo"], r["path"]) for r in rows}) == 82
    for r in rows:
        assert r["commit"] == pinned[r["repo"]]["commit"] and r["license"] == pinned[r["repo"]]["license"]
        assert r["path"].endswith(".slx") and len(r["sha256"]) == 64 and r["bytes"] > 0
    assert not list((ROOT / "datasets").rglob("*.slx")) and not list((ROOT / "tests").rglob("*.slx"))  # never committed


@needs_corpus
def test_real_files_import_and_simulate_at_the_documented_rates():
    rows = survey(str(_cache()), time_limit=60)
    s = summarize(rows)
    assert s["files"] == 82
    # every file is read: none fails, and nothing is dropped silently
    assert s["import"].get("failed", 0) == 0
    assert s["import"]["full"] >= 33 and s["import"]["full"] + s["import"]["partial"] == 82
    for r in rows:
        if r["import"] == "partial":
            assert r["unsupported"] and all(k and v > 0 for k, v in r["unsupported"].items()), r["file"]
            assert r["simulate"] == "no: unsupported blocks"
    assert s["simulate"].get("yes", 0) >= 14
    assert not any(v.startswith("no: internal error") for v in s["simulate"])
    ran = {r["file"].replace(os.sep, "/") for r in rows if r["simulate"] == "yes"}
    for name in (
        "ivaquero__book-control/matlab/control_pi.slx",
        "ivaquero__book-control/matlab/vibration.slx",
        "petercorke__RVC3-MATLAB/toolbox/sl_sea.slx",
        "cybergalactic__MSS/SIMULINK/mssSimulinkDemos/demoKalmanWavefilterAutop.slx",
    ):
        assert name in ran, name


@needs_corpus
def test_real_files_keep_positions_routes_and_subsystems():
    d = _cache()
    m = import_slx(str(d / "ivaquero__book-control" / "matlab" / "control_pi.slx"))
    assert m.import_report["unsupported"] == {} and len(m.root.blocks) == 16
    for b in m.root.blocks:
        assert b.pos[2] > 0 and b.pos[3] > 0
    assert sum(1 for ln in m.root.lines if ln.points) >= 5  # line corners are kept
    for ln in m.root.lines:  # corners of a route are joined by horizontal or vertical runs
        for a, b in zip(ln.points, ln.points[1:]):
            assert abs(a[0] - b[0]) < 1e-6 or abs(a[1] - b[1]) < 1e-6
    r = simulate(m)
    assert (
        r.t[-1] == 10
        and r.scopes
        and all(np.all(np.isfinite(sig["data"])) for sc in r.scopes.values() for sig in sc["signals"])
    )
    big = import_slx(str(d / "cybergalactic__MSS" / "SIMULINK" / "mssSimulinkDemos" / "demoKalmanWavefilterAutop.slx"))
    subs = [p for p, b, _ in big.root.walk() if b.system is not None]
    assert len(subs) >= 5 and big.import_report["libraries"] == ["mssSimulink"] and big.import_report["linked"] >= 3
    again = type(big).from_json(json.loads(big.dumps()))
    assert again.to_json() == big.to_json()


@needs_corpus
def test_parameter_defaults_listed_in_the_real_files_agree_with_the_importer():
    rows = defaults_table(str(_cache()))
    assert len(rows) >= 15
    assert [f"{r['block']}.{r['parameter']}" for r in rows if not r["agree"]] == []
