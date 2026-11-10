"""Every script in examples/ must run without errors."""

import os
from pathlib import Path

import pytest

from mylab import Interpreter

EXAMPLES = sorted((Path(__file__).parent.parent / "examples").glob("*.m"))


@pytest.mark.parametrize("path", EXAMPLES, ids=[p.name for p in EXAMPLES])
def test_example_runs(path, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # plots are saved into the temp dir
    out: list[str] = []
    interp = Interpreter(out=out.append)
    interp.run_file(str(path))
    interp.execute("close all")
    assert "Error" not in "".join(out)
    assert os.getcwd() == str(tmp_path)
