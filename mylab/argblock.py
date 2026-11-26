"""Function ``arguments`` blocks.

Handles what a block declares for each argument: a default value (which may refer to the
arguments before it), conversion to a class, a size, validator functions, name-value
arguments collected into a struct (``opts.Name`` and ``opts.?ClassName``), repeating
arguments, and ``arguments (Output)`` blocks that check what a function returns.
"""

from __future__ import annotations

from typing import Any


from . import ast as A
from .errors import MatlabError
from .values import (
    Cell, FunctionHandle, MString, Struct, is_char, to_str,
)



# -- class conversion and size -----------------------------------------------------------------
# Function arguments and classdef properties follow the same rules; the code lives in classdef.py.

def convert_class(interp, value: Any, cls: str) -> Any:
    """Convert ``value`` to the declared class, the way an arguments block does."""
    from .classdef import _convert_class
    return _convert_class(interp, value, cls, "")


def check_size(value: Any, size: list) -> Any:
    """Check a value against a declared size such as (1,1), (1,:) or (:,3), converting it where MATLAB does."""
    from .classdef import fit_size
    return fit_size(value, size)


# -- one declared argument ----------------------------------------------------------------------

def _run_validators(interp, entry: A.PropertyDef, value: Any, frame) -> None:
    for v in entry.attrs.get("_validators", []):
        if isinstance(v, A.Ident) and not frame.has(v.name):
            interp.call_named(v.name, [value], 0, frame)
        elif isinstance(v, A.FuncHandle):
            interp.call_named(v.name, [value], 0, frame)
        else:
            interp.frames.append(frame)
            try:
                result = interp.eval_multi(v, frame, 0)
            finally:
                interp.frames.pop()
            if result and isinstance(result[0], FunctionHandle):  # {@(x) assert(...)}
                interp.call_handle(result[0], [value], 0)


def convert_value(interp, entry: A.PropertyDef, value: Any, context: str, frame, bind=None) -> Any:
    """Class conversion, size check and validators of one argument; errors start with ``context``."""
    attrs = entry.attrs
    try:
        cls = attrs.get("_class")
        if cls:
            value = convert_class(interp, value, cls)
        size = attrs.get("_size")
        if size:
            value = check_size(value, size)
        if attrs.get("_validators"):
            if bind is not None:
                bind(value)
            _run_validators(interp, entry, value, frame)
    except MatlabError as e:
        if e.message.startswith("Invalid "):
            raise
        raise MatlabError(f"{context} {e.message}", e.identifier or "MATLAB:validation:IncompatibleSize") from None
    return value


# -- inputs ---------------------------------------------------------------------------------------

def _class_options(interp, cls_name: str) -> dict[str, A.PropertyDef]:
    """Public settable properties of a class, for ``opts.?ClassName``."""
    info = interp.find_class(cls_name)
    if info is None:
        raise MatlabError(f"Class '{cls_name}' named in an arguments block was not found.",
                          "MATLAB:functionValidation:UndefinedClass")
    out: dict[str, A.PropertyDef] = {}
    for pname, (pdef, _owner) in info.prop_defs.items():
        attrs = pdef.attrs
        access = str(attrs.get("SetAccess", attrs.get("Access", "public"))).lower()
        if access != "public" or attrs.get("Constant") or attrs.get("Hidden"):
            continue
        out[pname] = A.PropertyDef(pname, None, {k: v for k, v in attrs.items() if k.startswith("_")})
    return out


def _is_text(v: Any) -> bool:
    return isinstance(v, MString) or (is_char(v) and v.ndim == 2 and v.shape[0] == 1)


def apply_inputs(interp, fn: A.Function, args: list, frame, functions: dict) -> None:
    """Bind and validate the inputs of a function that has an ``arguments`` block."""
    entries = fn.arguments
    positional = [e for e in entries if "." not in e.name and not e.attrs.get("_repeating")]
    repeating = [e for e in entries if "." not in e.name and e.attrs.get("_repeating")]
    name_value = [e for e in entries if "." in e.name]

    # name-value groups: struct name -> {option name: declaration}
    groups: dict[str, dict[str, A.PropertyDef]] = {}
    for e in name_value:
        group, _, field = e.name.partition(".")
        fields = groups.setdefault(group, {})
        if field.startswith("?"):
            for pname, pdef in _class_options(interp, e.attrs["_from_class"]).items():
                fields.setdefault(pname, pdef)
        else:
            fields[field] = e
    option_names = {f.lower(): (g, f) for g, fields in groups.items() for f in fields}

    n_required = sum(1 for e in positional if e.default is None)
    # Where do the name-value pairs start? At the first text argument, after the required
    # positional ones, that names an option and leaves an even number of arguments.
    n_args = len(args)
    nv_start = n_args
    if option_names:
        first_possible = n_required if not repeating else len(positional)
        for k in range(min(first_possible, n_args), n_args):
            if (n_args - k) % 2 == 0 and _is_text(args[k]) and to_str(args[k]).lower() in option_names \
                    and all(_is_text(args[j]) for j in range(k, n_args, 2)):
                nv_start = k
                break
    lead = args[:nv_start]
    pairs = args[nv_start:]

    interp.frames.append(frame)
    try:
        for k, entry in enumerate(positional):
            context = f"Invalid argument at position {k + 1}."
            if k < len(lead):
                frame.vars[entry.name] = convert_value(
                    interp, entry, lead[k], context, frame,
                    bind=lambda v, n=entry.name: frame.vars.__setitem__(n, v))
            elif entry.default is not None:
                value = interp.eval(entry.default, frame)
                frame.vars[entry.name] = convert_value(
                    interp, entry, value, context, frame,
                    bind=lambda v, n=entry.name: frame.vars.__setitem__(n, v))
                frame.missing_inputs = frame.missing_inputs - {entry.name}
            else:
                raise MatlabError("Not enough input arguments.", "MATLAB:minrhs")
        rest = lead[len(positional):]
        if repeating:
            n = len(repeating)
            if len(rest) % n:
                raise MatlabError(f"Invalid argument list. Repeating arguments must come in groups of {n}.",
                                  "MATLAB:functionValidation:RepeatingGroupIncomplete")
            columns: list[list] = [[] for _ in repeating]
            for g in range(len(rest) // n):
                for j, entry in enumerate(repeating):
                    pos = len(positional) + g * n + j + 1
                    columns[j].append(convert_value(interp, entry, rest[g * n + j],
                                                    f"Invalid argument at position {pos}.", frame))
            for entry, vals in zip(repeating, columns):
                frame.vars[entry.name] = Cell.from_list(vals)
            frame.missing_inputs = frame.missing_inputs - {e.name for e in repeating}
        elif rest:
            has_varargin = bool(fn.params) and fn.params[-1] == "varargin"
            if not has_varargin:
                if groups and len(rest) % 2 == 0 and _is_text(rest[0]):
                    raise MatlabError(f"Invalid name-value argument '{to_str(rest[0])}'. Valid names are: "
                                      + ", ".join(f for fields in groups.values() for f in fields) + ".",
                                      "MATLAB:functionValidation:UnknownNameValue")
                if groups and len(rest) % 2:
                    raise MatlabError("Name-value arguments must come in pairs.",
                                      "MATLAB:functionValidation:NameWithoutValue")
                raise MatlabError("Too many input arguments.", "MATLAB:TooManyInputs")
        if groups:
            given: dict[tuple[str, str], Any] = {}
            for k in range(0, len(pairs), 2):
                key = to_str(pairs[k]).lower()
                if key not in option_names:
                    raise MatlabError(f"Invalid name-value argument '{to_str(pairs[k])}'. Valid names are: "
                                      + ", ".join(f for fields in groups.values() for f in fields) + ".",
                                      "MATLAB:functionValidation:UnknownNameValue")
                given[option_names[key]] = pairs[k + 1]
            for group, fields in groups.items():
                opts = Struct()
                frame.vars[group] = opts
                for fname, entry in fields.items():
                    context = f"Invalid name-value argument '{fname}'."
                    if (group, fname) in given:
                        value = given[(group, fname)]
                    elif entry.default is not None:
                        value = interp.eval(entry.default, frame)
                    else:
                        continue
                    value = convert_value(interp, entry, value, context, frame,
                                          bind=lambda v, n=fname, o=opts: o.set_field(0, n, v))
                    opts.set_field(0, fname, value)
            frame.missing_inputs = frame.missing_inputs - set(groups)
    finally:
        interp.frames.pop()


# -- outputs --------------------------------------------------------------------------------------

def apply_outputs(interp, fn: A.Function, frame) -> None:
    """Convert and validate the outputs declared in an ``arguments (Output)`` block."""
    for k, entry in enumerate(getattr(fn, "out_arguments", ())):
        name = entry.name
        if not frame.has(name):
            continue
        context = f"Invalid output argument at position {k + 1}."
        value = frame.get(name)
        if entry.attrs.get("_repeating") and isinstance(value, Cell):
            vals = [convert_value(interp, entry, v, context, frame) for v in value.values_f()]
            frame.set(name, Cell.from_list(vals))
            continue
        interp.frames.append(frame)
        try:
            frame.set(name, convert_value(interp, entry, value, context, frame,
                                          bind=lambda v, n=name: frame.set(n, v)))
        finally:
            interp.frames.pop()
