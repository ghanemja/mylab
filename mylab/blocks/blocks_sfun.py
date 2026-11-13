"""S-function blocks whose code is a MATLAB-language file: level 1 (flag protocol) and level 2 (block object).

The file is run by the mylab interpreter. Compiled S-functions (C, MEX) cannot be run and are reported as such.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .blocks_more import _split_top
from .library import BlockError, Node, P, block, sig, to_value

# The object a level-2 S-function receives, written in the MATLAB language so the interpreter can run it.
RUNTIME_CLASS = r"""
classdef mylabSFunctionBlock < handle
    properties
        NumInputPorts = 0
        NumOutputPorts = 0
        NumDialogPrms = 0
        NumContStates = 0
        NumDworks = 0
        NumRuntimePrms = 0
        InputPort = []
        OutputPort = []
        Dwork = []
        DialogPrm = []
        RuntimePrm = []
        ContStates = struct('Data', [])
        Derivatives = struct('Data', [])
        SampleTimes = [-1 0]
        CurrentTime = 0
        NextTimeHit = 0
        BlockHandle = 0
        Major = true
        Methods = struct()
        SimStateCompliance = ''
        OperatingPointCompliance = ''
        AllowSignalsWithMoreThan2D = false
        SetSimViewingDeviceValue = false
    end
    methods
        function r = RegBlockMethod(obj, name, fh)
            r = [];
            obj.Methods.(name) = fh;
        end
        function r = SetPreCompInpPortInfoToDynamic(obj), r = []; end
        function r = SetPreCompOutPortInfoToDynamic(obj), r = []; end
        function r = SetPreCompPortInfoToDefaults(obj), r = []; end
        function r = SetAccelRunOnTLC(obj, varargin), r = []; end
        function r = SetSimViewingDevice(obj, varargin), r = []; end
        function r = AutoRegRuntimePrms(obj), r = []; end
        function r = AutoUpdateRuntimePrms(obj), r = []; end
        function r = WriteRTWParam(obj, varargin), r = []; end
        function v = IsMajorTimeStep(obj), v = obj.Major; end
        function v = IsSampleHit(obj, varargin), v = true; end
        function v = IsSpecialSampleHit(obj, varargin), v = true; end
    end
end
"""


def _find(node: Node, name: str) -> None:
    interp = node.ctx.interp
    if not name.isidentifier():
        raise BlockError(f"'{name}' is not a function name")
    if interp.find_file_function(name) is None:
        raise BlockError(
            f"the S-function '{name}' was not found as a .m file on the search path. "
            "If it is compiled code (C or MEX), mylab cannot run it"
        )


def _dialog(node: Node) -> list:
    out = []
    for expr in _split_top(node.text("Parameters")) if node.text("Parameters") else []:
        try:
            out.append(node.ctx.evaluate(expr))
        except Exception as exc:
            raise BlockError(f"parameter '{expr}' cannot be evaluated: {getattr(exc, 'message', exc)}") from None
    return out


@block(
    "SFunction",
    "User-defined",
    "Runs a level-1 S-function written as a .m file: [sys, x0, str, ts] = name(t, x, u, flag, ...).",
    title="S-Function (level 1, .m file)",
    icon="sfun",
)
class SFunction(Node):
    params = [
        P("FunctionName", "system", "Function name", "text"),
        P("Parameters", "", "Extra arguments, separated by commas", "text"),
        P("NumInputs", "1", "Input ports", "text"),
        P("NumOutputs", "1", "Output ports", "text"),
    ]
    nin = ["int", "NumInputs"]
    nout = ["int", "NumOutputs"]
    time_varying = True

    def call(self, t, x, u, flag, nout=1):
        from ..values import FunctionHandle

        interp = self.ctx.interp
        args = [np.array([[float(t)]]), to_value(x), to_value(u), np.array([[float(flag)]]), *self.extra]
        return interp.call_function_value(FunctionHandle(name=self.name), args, nout)

    def init(self):
        self.name = self.text("FunctionName")
        _find(self, self.name)
        self.extra = _dialog(self)
        try:
            res = self.call(0.0, np.zeros(0), np.zeros(0), 0, 4)
        except Exception as exc:
            raise BlockError(
                f"the S-function '{self.name}' failed when asked for its sizes: {getattr(exc, 'message', exc)}"
            ) from None
        sizes = np.ravel(np.asarray(res[0], float))
        if sizes.size < 7:
            raise BlockError(f"the S-function '{self.name}' did not return its sizes (use simsizes)")
        self.nc, self.nd, self.ny, self.nu = (int(sizes[k]) for k in range(4))
        self.df = bool(sizes[5])
        x0 = np.ravel(np.asarray(res[1], float)) if len(res) > 1 else np.zeros(0)
        if x0.size != self.nc + self.nd:
            raise BlockError(
                f"the S-function '{self.name}' has {self.nc + self.nd} states but gave {x0.size} initial values"
            )
        self.x0c, self.xd = x0[: self.nc].copy(), x0[self.nc :].copy()
        self.nx = self.nc
        ts = (
            np.asarray(res[3], float).reshape(-1, 2)
            if len(res) > 3 and np.size(res[3]) >= 2
            else np.array([[-1.0, 0.0]])
        )
        self.ts = ts[0]
        if self.n_in() > 1 or self.n_out() > 1:
            raise BlockError("a level-1 S-function has one input port and one output port")

    def sample_time(self):
        period, offset = float(self.ts[0]), float(self.ts[1])
        if period == 0:
            return None
        if period < 0:
            return None if self.nc else "inherit"
        return (period, offset)

    def initial_state(self):
        return self.x0c

    def _x(self):
        return np.concatenate([self.x, self.xd]) if self.nd else self.x

    def _u(self, u):
        return np.ravel(u[0], order="F") if u else np.zeros(0)

    def out(self, t, u):
        if not self.n_out():
            self.call(t, self._x(), self._u(u), 3, 1)
            return []
        y = sig(self.call(t, self._x(), self._u(u), 3, 1)[0])
        if self.ny > 0 and y.size != self.ny:
            raise BlockError(f"the S-function '{self.name}' returned {y.size} outputs instead of {self.ny}")
        return [y]

    def der(self, t, u):
        d = np.ravel(np.asarray(self.call(t, self._x(), self._u(u), 1, 1)[0], float))
        if d.size != self.nc:
            raise BlockError(f"the S-function '{self.name}' returned {d.size} derivatives for {self.nc} states")
        return d

    def update(self, t, u):
        if self.nd:
            xd = np.ravel(np.asarray(self.call(t, self._x(), self._u(u), 2, 1)[0], float))
            if xd.size != self.nd:
                raise BlockError(
                    f"the S-function '{self.name}' returned {xd.size} discrete states instead of {self.nd}"
                )
            self.xd = xd


@block(
    "MSFunction",
    "User-defined",
    "Runs a level-2 S-function written as a .m file: a setup function that fills in a block object.",
    title="S-Function (level 2, .m file)",
    aliases=("M-S-Function", "Level2MATLABSFunction"),
    icon="sfun2",
)
class MSFunction(Node):
    params = [
        P("FunctionName", "mysfun", "Function name", "text"),
        P("Parameters", "", "Dialog parameters, separated by commas", "text"),
        P("NumInputs", "1", "Input ports", "text"),
        P("NumOutputs", "1", "Output ports", "text"),
    ]
    nin = ["int", "NumInputs"]
    nout = ["int", "NumOutputs"]
    time_varying = True

    def run(self, code: str, **names: Any) -> dict:
        """Run a few statements with the block object as ``b``; returns the variables afterwards."""
        from ..interpreter import Frame
        from ..parser import parse

        interp = self.ctx.interp
        prog = self._programs.get(code)
        if prog is None:
            prog = self._programs[code] = parse(code)
        frame = Frame(interp, interp.base.functions, name="S-function")
        frame.vars = {"b": self.obj, **names}
        interp.frames.append(frame)
        try:
            interp.run_program(prog, frame)
        finally:
            interp.frames.pop()
        return frame.vars

    def method(self, name: str) -> bool:
        v = self.run(
            "has = isfield(b.Methods, name); if has, f = b.Methods.(name); f(b); end", name=np.array([list(name)])
        )
        return bool(np.any(v["has"]))

    def init(self):
        from ..parser import parse
        from ..values import Cell, FunctionHandle

        interp = self.ctx.interp
        self.name = self.text("FunctionName")
        _find(self, self.name)
        self._programs: dict = {}
        if "mylabSFunctionBlock" not in interp.classes:
            interp.run_program(parse(RUNTIME_CLASS), interp.base)
        self.obj = interp.eval_expression("mylabSFunctionBlock()")
        prm = _dialog(self)
        try:
            interp.call_function_value(FunctionHandle(name=self.name), [self.obj], 0)
            if prm:
                self.run("for k = 1:numel(p), b.DialogPrm(k).Data = p{k}; end", p=Cell.from_list(prm))
            v = self.run(
                "nin = b.NumInputPorts; nout = b.NumOutputPorts; nc = b.NumContStates; ts = b.SampleTimes; df = true(1, nin);"  # noqa: E501
                "for k = 1:nin, if k <= numel(b.InputPort) && isfield(b.InputPort, 'DirectFeedthrough') && "
                "~isempty(b.InputPort(k).DirectFeedthrough), df(k) = logical(b.InputPort(k).DirectFeedthrough); end, end"  # noqa: E501
            )
        except BlockError:
            raise
        except Exception as exc:
            raise BlockError(
                f"the setup of the S-function '{self.name}' failed: {getattr(exc, 'message', exc)}"
            ) from None
        n_in, n_out = int(np.ravel(v["nin"])[0]), int(np.ravel(v["nout"])[0])
        if n_in != self.n_in() or n_out != self.n_out():
            raise BlockError(
                f"the S-function '{self.name}' declares {n_in} input and {n_out} output ports, "
                f"but the block is drawn with {self.n_in()} and {self.n_out()}"
            )
        self.nx = int(np.ravel(v["nc"])[0])
        self.ts = np.ravel(np.asarray(v["ts"], float))[:2]
        self.df = [bool(x) for x in np.ravel(v["df"])] or True
        self.display_only = n_out == 0 and self.nx == 0
        self.broken = False
        try:
            self.method("PostPropagationSetup")
            self.method("InitializeConditions")
            self.method("Start")
        except Exception as exc:
            raise BlockError(f"starting the S-function '{self.name}' failed: {getattr(exc, 'message', exc)}") from None
        self.x0 = np.zeros(self.nx)
        if self.nx:
            x0 = np.ravel(np.asarray(self.run("x = b.ContStates.Data;")["x"], float))
            if x0.size == self.nx:
                self.x0 = x0

    def sample_time(self):
        period = float(self.ts[0]) if self.ts.size else -1.0
        if period == 0:
            return None
        if period < 0:
            return None if self.nx else "inherit"
        return (period, float(self.ts[1]) if self.ts.size > 1 else 0.0)

    def initial_state(self):
        return self.x0

    def load(self, t, u):
        code = "b.CurrentTime = t; b.Major = major;" + "".join(
            f"b.InputPort({k + 1}).Data = u{k};" for k in range(len(u))
        )
        names = {f"u{k}": to_value(v) for k, v in enumerate(u)}
        if self.nx:
            code += "b.ContStates.Data = x;"
            names["x"] = to_value(self.x)
        self.run(code, t=np.array([[float(t)]]), major=np.array([[not self.sim.minor]]), **names)

    def out(self, t, u):
        if self.broken:
            return []
        try:
            self.load(t, u)
            self.method("Outputs")
        except Exception as exc:
            if not self.display_only:
                raise
            # nothing reads from this block, so the simulation does not depend on it
            self.broken = True
            self.sim.note(
                f"'{self.path}': the display S-function '{self.name}' stopped ({getattr(exc, 'message', exc)})."
            )
            return []
        n = self.n_out()
        if not n:
            return []
        v = self.run("y = cell(1, n); for k = 1:n, y{k} = b.OutputPort(k).Data; end", n=np.array([[float(n)]]))
        ys = [sig(c) for c in v["y"].data.flat]
        for k, y in enumerate(ys):
            if y.size == 0:
                raise BlockError(f"the S-function '{self.name}' did not set output {k + 1}")
        return ys

    def der(self, t, u):
        self.load(t, u)
        self.method("Derivatives")
        d = np.ravel(np.asarray(self.run("d = b.Derivatives.Data;")["d"], float))
        if d.size != self.nx:
            raise BlockError(f"the S-function '{self.name}' set {d.size} derivatives for {self.nx} states")
        return d

    def update(self, t, u):
        if self.broken:
            return
        self.load(t, u)
        self.method("Update")
