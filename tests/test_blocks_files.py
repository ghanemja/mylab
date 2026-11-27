"""mylab Blocks: reading .mdl files, and writing .slx and .mdl files that mylab reads back."""

import glob
from pathlib import Path

import numpy as np
import pytest

from mylab.api import Session
from mylab.blocks import ModelError, build, find_type, load_model, simulate
from mylab.blocks.writer import mdl_text, save_as, slx_bytes

ROOT = Path(__file__).resolve().parent.parent

# A small model in the text format, written here in the layout of public .mdl files.
SMALL_MDL = r'''
Model {
  Name              "small"
  Version          8.6
  PreLoadFcn          "k = 2;"
  Array {
    Type            "Handle"
    Dimension            1
    Simulink.ConfigSet {
      $ObjectID              1
      Array {
    Type            "Handle"
    Simulink.SolverCC {
      StartTime          "0.0"
      StopTime          "3"
      RelTol          "1e-8"
      AbsTol          "1e-10"
      Solver          "ode45"
    }
      }
    }
  }
  BlockParameterDefaults {
    Block {
      BlockType              Gain
      Gain              "1"
    }
  }
  System {
    Name            "small"
    Block {
      BlockType              Step
      Name              "Step"
      SID              "1"
      Position              [40, 40, 70, 70]
      Time              "0"
    }
    Block {
      BlockType              Sum
      Name              "Sum"
      SID              "2"
      Ports              [2, 1]
      Position              [110, 45, 130, 65]
      Inputs              "|+-"
    }
    Block {
      BlockType              SubSystem
      Name              "Plant\nmodel"
      SID              "3"
      Ports              [1, 1]
      Position              [160, 40, 220, 70]
      Object {
    $PropName        "MaskObject"
    $ObjectID        2
    $ClassName        "Simulink.Mask"
    Initialization        "kk = gain"
    "*1;"
    Object {
      $PropName          "Parameters"
      $ObjectID          3
      $ClassName          "Simulink.MaskParameter"
      Type              "edit"
      Name              "gain"
      Value              "k"
    }
      }
      System {
    Name            "Plant\nmodel"
    Block {
      BlockType          Inport
      Name              "u"
      SID              "3:1"
      Position          [20, 20, 40, 34]
    }
    Block {
      BlockType          Gain
      Name              "K"
      SID              "3:2"
      Position          [80, 15, 110, 45]
      Gain              "kk"
    }
    Block {
      BlockType          Integrator
      Name              "Int"
      SID              "3:3"
      Position          [140, 15, 170, 45]
    }
    Block {
      BlockType          Outport
      Name              "y"
      SID              "3:4"
      Position          [200, 23, 220, 37]
    }
    Line {
      SrcBlock          "u"
      SrcPort          1
      DstBlock          "K"
      DstPort          1
    }
    Line {
      SrcBlock          "K"
      SrcPort          1
      DstBlock          "Int"
      DstPort          1
    }
    Line {
      SrcBlock          "Int"
      SrcPort          1
      DstBlock          "y"
      DstPort          1
    }
      }
    }
    Block {
      BlockType              SubSystem
      Name              "Twice"
      SID              "4"
      Ports              [1, 1]
      Position              [260, 40, 300, 70]
      SFBlockType          "MATLAB Function"
      System {
    Name            "Twice"
      }
    }
    Block {
      BlockType              Outport
      Name              "Out1"
      SID              "5"
      Position              [330, 48, 360, 62]
    }
    Block {
      BlockType              Outport
      Name              "Out2"
      SID              "6"
      Position              [330, 98, 360, 112]
      Port              "2"
    }
    Line {
      SrcBlock              "Step"
      SrcPort              1
      DstBlock              "Sum"
      DstPort              1
    }
    Line {
      Name              "error"
      SrcBlock              "Sum"
      SrcPort              1
      DstBlock              "Plant\nmodel"
      DstPort              1
    }
    Line {
      SrcBlock              "Plant\nmodel"
      SrcPort              1
      Points              [15, 0]
      Branch {
    DstBlock        "Twice"
    DstPort            1
      }
      Branch {
    Points            [0, 50; -150, 0]
    DstBlock        "Sum"
    DstPort            2
      }
    }
    Line {
      SrcBlock              "Twice"
      SrcPort              1
      DstBlock              "Out1"
      DstPort              1
    }
    Line {
      SrcBlock              "Plant\nmodel"
      SrcPort              1
      DstBlock              "Out2"
      DstPort              1
    }
  }
}
Stateflow {
  machine {
    id                1
    name            "small"
  }
  chart {
    id                2
    name            "Twice"
    type            EML_CHART
  }
  state {
    id                3
    chart            2
    eml {
      isEML              1
      script              "function y = fcn(u)\n% doubles \"u\"\ny = 2*u;"
    }
  }
}
'''


def test_mdl_file_is_read_like_an_slx_file(tmp_path):
    p = tmp_path / "small.mdl"
    p.write_text(SMALL_MDL, encoding="latin-1")
    m = load_model(str(p))
    assert m.notes == [] and m.import_report["blocks"] == 10 and m.import_report["converted"] == 10
    assert m.config["StopTime"] == "3" and m.config["RelTol"] == "1e-8" and m.callbacks["PreLoadFcn"] == "k = 2;"
    plant = m.root.block("Plant model")
    assert plant.extra["mask"] == {"gain": "k"} and plant.extra["maskInit"] == "kk = gain*1;"
    twice = m.root.block("Twice")
    assert twice.type == "MATLABFunction" and 'doubles "u"' in twice.params["Script"]
    assert {(ln.src, ln.dst, ln.name) for ln in m.root.lines} >= {
        (("Sum", 1), ("Plant model", 1), "error"),
        (("Plant model", 1), ("Sum", 2), ""),
        (("Plant model", 1), ("Twice", 1), ""),
    }
    s = Session()
    s.run("k = 2;")
    r = simulate(m, s.interp)
    # closed loop dx/dt = 2 (1 - x): x = 1 - exp(-2 t); the function block doubles it
    assert np.max(np.abs(r.y[:, 1] - (1 - np.exp(-2 * r.t)))) < 1e-6
    assert np.allclose(r.y[:, 0], 2 * r.y[:, 1])


def test_mdl_errors(tmp_path):
    bad = tmp_path / "bad.mdl"
    bad.write_text("this is not a model\n")
    with pytest.raises(ModelError, match="not a model file in the text"):
        load_model(str(bad))
    nosys = tmp_path / "nosys.mdl"
    nosys.write_text('Model {\n  Name "x"\n}\n')
    with pytest.raises(ModelError, match="no block diagram"):
        load_model(str(nosys))


def normal(model):
    """Model JSON without parameters that only repeat a default."""
    d = model.to_json()

    def clean(sys):
        for b in sys.get("blocks", []):
            cls = find_type(b["type"])
            if cls is not None and "params" in b:
                defaults = {p.name: p.default for p in cls.params}
                b["params"] = {k: v for k, v in b["params"].items() if defaults.get(k) != v}
                if not b["params"]:
                    del b["params"]
            clean(b)

    clean(d)
    d.pop("name", None)
    return d


@pytest.mark.parametrize("ext", [".slx", ".mdl"])
@pytest.mark.parametrize("path", sorted(glob.glob(str(ROOT / "examples" / "blocks" / "*.mlb.json"))))
def test_examples_survive_writing_and_reading(tmp_path, path, ext):
    m = load_model(path)
    target = str(tmp_path / (m.name + ext))
    save_as(m, target)
    again = load_model(target)
    assert again.notes == []
    assert normal(again) == normal(m)
    a, b = simulate(m), simulate(again)
    assert a.y.shape == b.y.shape and np.allclose(a.y, b.y)


def rich_model():
    inner = (
        [("u", "Inport"), ("E", "EnablePort", {"StatesWhenEnabling": "reset"}), ("K", "Gain", {"Gain": "kk"}),
         ("y", "Outport", {"InitialOutput": "0"})],
        ["u/1 -> K/1", "K/1 -> y/1"],
    )
    resettable = ([("u", "Inport"), ("R", "ResetPort"), ("I", "Integrator"), ("y", "Outport")],
                  ["u/1 -> I/1", "I/1 -> y/1"])
    m = build(
        "rich",
        [
            ("C", "Constant", {"Value": "[1 2]"}),
            ("P", "PulseGenerator", {"Period": "1"}),
            ("Sub", "Subsystem", {}, inner),
            ("Rst", "Subsystem", {}, resettable),
            ("F", "MATLABFunction", {"Script": "function [a, b] = f(u)\na = sum(u);\nb = 'x' == \"x\";\n"}),
            ("Odd", "SomeoneElsesBlock"),
            ("Out", "Outport"),
            ("Out2", "Outport", {"Port": "2"}),
            ("Out3", "Outport", {"Port": "3"}),
        ],
        ["C/1 -> Sub/1 : two values", "P/1 -> Sub/enable", "Sub/1 -> F/1", "F/1 -> Out/1", "F/2 -> Out2/1",
         "C/1 -> Rst/1", "P/1 -> Rst/reset", "Rst/1 -> Out3/1"],
        StopTime="2",
        AlgebraicLoop="error",
    )
    sub = m.root.block("Sub")
    sub.extra["mask"] = {"g": "3"}
    sub.extra["maskInit"] = "kk = g + 1;"
    sub.rot = 1
    m.root.block("Odd").extra["ports"] = [0, 0]
    m.root.lines[0].points = [[150, 60], [150, 120]]
    m.callbacks["InitFcn"] = "disp('hi \"there\"')"
    m.description = "a model"
    return m


@pytest.mark.parametrize("ext", [".slx", ".mdl"])
def test_subsystems_masks_functions_and_odd_blocks_survive(tmp_path, ext):
    m = rich_model()
    target = str(tmp_path / ("rich" + ext))
    save_as(m, target)
    again = load_model(target)
    a, b = normal(m), normal(again)
    a.pop("description"), b.pop("description", None)
    assert b == a
    m.root.blocks = [x for x in m.root.blocks if x.name != "Odd"]
    again.root.blocks = [x for x in again.root.blocks if x.name != "Odd"]
    r1, r2 = simulate(m), simulate(again)
    assert np.allclose(r1.y, r2.y) and np.allclose(r1.y[-1, :2], [4 * 3, 1])


def test_written_files_say_who_they_are_for():
    m = rich_model()
    text = mdl_text(m)
    assert text.startswith("# Written by mylab Blocks for mylab; not made with or tested in any other program.")
    import io
    import zipfile

    z = zipfile.ZipFile(io.BytesIO(slx_bytes(m)))
    assert "simulink/blockdiagram.xml" in z.namelist()
    assert b"not made with or tested in any other program" in z.read("metadata/mylab.xml")


def test_save_system_writes_slx_and_mdl_copies(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    s = Session()
    res = s.run(
        """
        new_system('lag');
        add_block('blocks/Step','lag/S','Time','0');
        add_block('blocks/TransferFcn','lag/G','Denominator','[2 1]');
        add_block('blocks/Outport','lag/O');
        add_line('lag','S/1','G/1'); add_line('lag','G/1','O/1');
        set_param('lag','StopTime','4');
        f1 = save_system('lag', 'lag_copy.slx');
        save_system('lag_copy', 'lag_text.mdl');
        save_system('lag_text');
        close_system('lag_text');
        load_system('lag_copy.slx');
        out = sim('lag_copy');
        y1 = out.yout(end);
        out = sim('lag_text.mdl');
        y2 = out.yout(end);
        """
    )
    assert res.error is None, res.error
    assert (tmp_path / "lag_copy.slx").is_file() and (tmp_path / "lag_text.mdl").is_file()
    assert (tmp_path / "lag_text.mlb.json").is_file()  # the model itself is still saved as JSON
    assert np.isclose(np.ravel(s.interp.get_var("y1"))[0], 1 - np.exp(-2), atol=1e-4)
    assert np.isclose(np.ravel(s.interp.get_var("y2"))[0], 1 - np.exp(-2), atol=1e-4)
