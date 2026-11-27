"""MEX files: build C/C++ functions with mex and call them like .m functions (see docs/MEX.md)."""

from __future__ import annotations

from ..errors import MatlabError
from ..values import Cell, Struct, char
from ._registry import builtin
from ._util import nargs, text


@builtin("mex")
def _mex(interp, args, nargout):
    from ..mex.build import mex_command
    return mex_command(interp, [text(a, "mex argument") for a in args], nargout)


@builtin("mexext")
def _mexext(interp, args, nargout):
    from ..mex import MEXEXT
    nargs(args, 0, 1, "mexext")
    if args:
        if text(args[0]).lower() != "all":
            raise MatlabError("mexext: the only option is 'all'.")
        table = [("glnxa64", "mexa64"), ("maci64", "mexmaci64"), ("maca64", "mexmaca64"), ("win64", "mexw64")]
        import numpy as np
        data = np.empty((len(table), 1), dtype=object)
        for i, (arch, ext) in enumerate(table):
            data[i, 0] = {"arch": char(arch), "ext": char(ext)}
        return Struct(data, ["arch", "ext"])
    return char(MEXEXT)


@builtin("mylabMexCall__")
def _mex_call(interp, args, nargout):
    """Run a MEX file: the body of the function mylab makes for each MEX file it finds."""
    from ..mex import runtime_for
    path = text(args[0])
    n = int(args[1].item()) if hasattr(args[1], "item") else int(args[1])
    frames = interp.frames
    caller = frames[-2] if len(frames) > 1 else interp.base
    outs = runtime_for(interp).call(path, list(args[2:]), n, caller)
    return Cell.from_list(outs)
