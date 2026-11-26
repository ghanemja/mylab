"""Each built-in name has one implementation: no name is registered by two modules.

A function registered under several names by one module (aliases) is fine.
"""

import mylab.builtins  # noqa: F401  (imports every built-in module)
from mylab.builtins._registry import BUILTINS, REGISTRATIONS, registered_by_several_modules

# Duplicates that were not part of the first clean-up (they belong to areas other people were
# editing at the time). Remove a name from this set when its duplicate is resolved; the second
# test fails as a reminder once a listed name has a single implementation.
NOT_YET_RESOLVED = {
    "addvars", "blkdiag", "brush",
    "datestr",
    "height",
    "mustBeInRange", "mustBeVector", "rms", "rows2vars",
    "table2struct", "varfun", "width",
}


def test_registry_records_every_registration():
    assert REGISTRATIONS
    for name, regs in REGISTRATIONS.items():
        assert regs, name
        assert name in BUILTINS
        assert all(isinstance(module, str) and callable(fn) for module, fn in regs)
    # BUILTINS[name] is still the last registration (unless a later wrapper replaced it)
    assert REGISTRATIONS["strfind"][-1][1] is BUILTINS["strfind"]


def test_each_builtin_name_has_one_module():
    dups = {name: mods for name, mods in registered_by_several_modules().items()
            if name not in NOT_YET_RESOLVED}
    lines = [f"{name}: {', '.join(mods)}" for name, mods in sorted(dups.items())]
    assert not dups, "built-in names registered by more than one module:\n" + "\n".join(lines)


def test_not_yet_resolved_list_is_current():
    fixed = sorted(NOT_YET_RESOLVED - set(registered_by_several_modules()))
    assert not fixed, "these names now have one implementation; remove them from NOT_YET_RESOLVED: " + \
        ", ".join(fixed)
