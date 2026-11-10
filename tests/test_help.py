"""Help text coverage and format."""

import pytest

from mylab.builtins import BUILTINS
from mylab.help import help_text
from mylab.help_data import HELP

# Modules whose builtins must all be documented. Builtins from modules added later are not
# required here until help text is written for them.
DOCUMENTED_MODULES = {"core", "mathfns", "linalg", "strings", "data", "system", "numerics", "plotting", "table"}

# Names that are deliberately left without help text (keep empty if possible).
ALLOWLIST: set[str] = set()


def _required_names() -> list[str]:
    names = []
    for name, fn in BUILTINS.items():
        module = getattr(fn, "__module__", "").rsplit(".", 1)[-1]
        if module in DOCUMENTED_MODULES and name not in ALLOWLIST:
            names.append(name)
    return sorted(names)


def test_every_builtin_has_help():
    missing = [n for n in _required_names() if n not in HELP]
    assert not missing, f"builtins without a HELP entry: {missing}"


@pytest.mark.parametrize("name", sorted(HELP))
def test_entry_format(name):
    entry = HELP[name]
    assert isinstance(entry, tuple) and len(entry) == 2
    usage, description = entry
    usages = usage if isinstance(usage, list) else [usage]
    assert isinstance(usage, (str, list)) and usages
    assert all(isinstance(u, str) and u.strip() for u in usages)
    assert isinstance(description, str) and description.strip()


def test_help_command_prints_usage(m):
    out = m.run("help sum")
    assert "sum(x)" in out
    assert "sum(x, 'all')" in out
    assert "Sum of elements" in out


def test_help_text_for_alias_and_unknown():
    assert help_text("nan") == help_text("NaN")
    assert "not found" in help_text("no_such_function_zzz")
