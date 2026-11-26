"""Each built-in name has one implementation: no name is registered by two modules."""

import mylab.builtins  # noqa: F401  (imports every built-in module)
from mylab.builtins._registry import BUILTINS, REGISTRATIONS, registered_by_several_modules


def test_registry_records_every_registration():
    assert REGISTRATIONS
    for name, regs in REGISTRATIONS.items():
        assert regs, name
        assert name in BUILTINS


def test_each_builtin_name_has_one_module():
    dups = registered_by_several_modules()
    lines = [f"{name}: {', '.join(mods)}" for name, mods in sorted(dups.items())]
    assert not dups, "built-in names registered by more than one module:\n" + "\n".join(lines)
