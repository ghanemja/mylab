"""Conformance suite: mylab against reference values recorded from GNU Octave.

Every script in tests/conformance/cases/ runs in-process on mylab and all the
variables it leaves behind are compared with tests/conformance/golden.json
(Octave results plus the MATLAB corrections of overrides.json). No Octave is
needed to run this test. See tests/conformance/README.md.

Differences that are already known are listed in
tests/conformance/known_failures.json and are reported as xfail. The test fails
when a new difference appears, and also when a listed difference disappears
(then its entry must be deleted from the list).
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

CONF = Path(__file__).parent / "conformance"


def _load_runner():
    spec = importlib.util.spec_from_file_location("conformance_run", CONF / "run.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


runner = _load_runner()
GOLDEN = runner.load_json(runner.GOLDEN)
KNOWN = runner.load_json(runner.KNOWN)
CASE_IDS = runner.discover()


def test_suite_is_complete():
    """The golden file covers every case and the bookkeeping files name only real cases."""
    assert len(CASE_IDS) >= 250
    missing = [c for c in CASE_IDS if c not in GOLDEN["cases"] and c not in GOLDEN["overrides"]]
    assert not missing, f"no expectation for {missing}: run tests/conformance/run.py --update-golden"
    for name, table in (("known_failures.json", KNOWN), ("overrides.json", runner.load_json(runner.OVERRIDES))):
        stale = sorted(set(table) - set(CASE_IDS))
        assert not stale, f"{name} names cases that do not exist: {stale}"
    assert runner.count_checks(GOLDEN, CASE_IDS) >= 1500


@pytest.mark.parametrize("case_id", CASE_IDS)
def test_case(case_id: str, tmp_path: Path):
    got = runner.run_mylab_case(case_id, tmp_path)
    diffs = runner.compare_case(case_id, got, GOLDEN)
    known = KNOWN.get(case_id, {})

    new = {name: d for name, d in diffs.items() if name not in known}
    fixed = sorted(name for name, why in known.items() if name not in diffs and not why.startswith(runner.FLAKY))
    problems = [f"{name}: [{cat}] {detail}" for name, (cat, detail) in sorted(new.items())]
    if fixed:
        problems.append(
            f"now passing: {', '.join(fixed)}. Remove these entries for {case_id!r} from "
            "tests/conformance/known_failures.json")
    if problems:
        pytest.fail(f"{case_id} ({runner.case_path(case_id)})\n  " + "\n  ".join(problems), pytrace=False)
    if known:
        pytest.xfail("known mylab differences: " + "; ".join(f"{n} ({r})" for n, r in sorted(known.items())))
