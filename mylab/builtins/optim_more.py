"""Optimisation: option structures and objects, and the local solvers.

The global solvers live in ``_optim_global`` and the problem-based interface in
``_optim_problem``; both are loaded from here."""

from __future__ import annotations

import numpy as np

from ..errors import MatlabError
from ..objects import MObject
from ..values import Cell, FunctionHandle, Struct, char, empty_double
from ._registry import builtin
from ._util import fnum, is_text, nargs, text

EPS = float(np.finfo(float).eps)

# -- option names -----------------------------------------------------------------------------------

# Fields of the structure made by optimset (the older option names).
OPTIMSET_FIELDS = [
    "Display", "MaxFunEvals", "MaxIter", "TolFun", "TolX", "FunValCheck", "OutputFcn", "PlotFcns",
    "ActiveConstrTol", "Algorithm", "AlwaysHonorConstraints", "DerivativeCheck", "Diagnostics",
    "DiffMaxChange", "DiffMinChange", "FinDiffRelStep", "FinDiffType", "GoalsExactAchieve", "GradConstr",
    "GradObj", "HessFcn", "Hessian", "HessMult", "HessPattern", "HessUpdate", "InitBarrierParam",
    "InitTrustRegionRadius", "Jacobian", "JacobMult", "JacobPattern", "LargeScale", "MaxNodes", "MaxPCGIter",
    "MaxProjCGIter", "MaxSQPIter", "MaxTime", "MeritFunction", "MinAbsMax", "NoStopIfFlatInfeas",
    "ObjectiveLimit", "PhaseOneTotalScaling", "Preconditioner", "PrecondBandWidth", "RelLineSrchBnd",
    "RelLineSrchBndDuration", "ScaleProblem", "SubproblemAlgorithm", "TolCon", "TolConSQP", "TolGradCon",
    "TolPCG", "TolProjCG", "TolProjCGAbs", "TypicalX", "UseParallel",
]

# older name (lower case) -> current name
LEGACY = {
    "maxiter": "MaxIterations", "maxfunevals": "MaxFunctionEvaluations", "tolx": "StepTolerance",
    "tolcon": "ConstraintTolerance", "gradobj": "SpecifyObjectiveGradient",
    "gradconstr": "SpecifyConstraintGradient", "derivativecheck": "CheckGradients",
    "findifftype": "FiniteDifferenceType", "findiffrelstep": "FiniteDifferenceStepSize",
    "hessfcn": "HessianFcn", "plotfcns": "PlotFcn", "outputfcns": "OutputFcn",
    "hessupdate": "HessianApproximation", "hessmult": "HessianMultiplyFcn", "jacobmult": "JacobianMultiplyFcn",
    "minabsmax": "AbsoluteMaxObjectiveCount", "goalsexactachieve": "EqualityGoalCount",
    "generations": "MaxGenerations", "stallgenlimit": "MaxStallGenerations",
    "popinitrange": "InitialPopulationRange", "initialpopulation": "InitialPopulationMatrix",
    "initialscores": "InitialScoresMatrix", "timelimit": "MaxTime", "vectorized": "UseVectorized",
    "tolmesh": "MeshTolerance", "stalliterlimit": "MaxStallIterations", "maxiterations": "MaxIterations",
    "meshexpansion": "MeshExpansionFactor", "meshcontraction": "MeshContractionFactor",
    "completepoll": "UseCompletePoll", "pollingorder": "PollOrderAlgorithm", "tolbind": "TolBind",
    "initialswarm": "InitialSwarmMatrix", "stalltimelimit": "MaxStallTime", "populationtype": "PopulationType",
}
_TOLFUN_IS_OPTIMALITY = {"fmincon", "fminunc", "linprog", "quadprog", "lsqlin", "fminimax", "fgoalattain",
                         "fseminf"}

_FD = {"FiniteDifferenceType": "forward", "FiniteDifferenceStepSize": "sqrt(eps)", "TypicalX":
       "ones(numberOfVariables,1)", "CheckGradients": False, "UseParallel": False}
_NL = {"Display": "final", "MaxIterations": 400, "MaxFunctionEvaluations": "100*numberOfVariables",
       "OptimalityTolerance": 1e-6, "StepTolerance": 1e-6, "FunctionTolerance": 1e-6,
       "SpecifyObjectiveGradient": False, "OutputFcn": None, "PlotFcn": None, "FunValCheck": "off", **_FD}

# Options of each solver with their default values (None = empty).
SOLVER_OPTIONS = {
    "fmincon": {**_NL, "Algorithm": "interior-point", "MaxIterations": 1000, "MaxFunctionEvaluations": 3000,
                "StepTolerance": 1e-10, "ConstraintTolerance": 1e-6, "SpecifyConstraintGradient": False,
                "ObjectiveLimit": -1e20, "HessianApproximation": "bfgs", "HessianFcn": None,
                "HessianMultiplyFcn": None, "HonorBounds": True, "ScaleProblem": False,
                "SubproblemAlgorithm": "factorization", "BarrierParamUpdate": "monotone",
                "EnableFeasibilityMode": False, "InitBarrierParam": 0.1, "InitTrustRegionRadius":
                "sqrt(numberOfVariables)", "MaxSQPIter": "10*max(numberOfVariables,numberOfInequalities)"},
    "fminunc": {**_NL, "Algorithm": "quasi-newton", "ObjectiveLimit": -1e20, "HessianApproximation": "bfgs",
                "HessianFcn": None, "HessianMultiplyFcn": None, "SubproblemAlgorithm": "cg"},
    "fsolve": {**_NL, "Algorithm": "trust-region-dogleg", "JacobianMultiplyFcn": None},
    "lsqnonlin": {**_NL, "Algorithm": "trust-region-reflective", "ConstraintTolerance": 1e-6,
                  "SpecifyConstraintGradient": False, "JacobianMultiplyFcn": None},
    "lsqcurvefit": {**_NL, "Algorithm": "trust-region-reflective", "ConstraintTolerance": 1e-6,
                    "SpecifyConstraintGradient": False, "JacobianMultiplyFcn": None},
    "fminimax": {**_NL, "ConstraintTolerance": 1e-6, "SpecifyConstraintGradient": False,
                 "AbsoluteMaxObjectiveCount": 0, "MaxSQPIter": "10*max(numberOfVariables,numberOfInequalities)"},
    "fgoalattain": {**_NL, "ConstraintTolerance": 1e-6, "SpecifyConstraintGradient": False,
                    "EqualityGoalCount": 0, "MaxSQPIter": "10*max(numberOfVariables,numberOfInequalities)"},
    "fseminf": {**_NL, "ConstraintTolerance": 1e-6, "SpecifyConstraintGradient": False},
    "linprog": {"Algorithm": "dual-simplex-highs", "Display": "final", "MaxIterations": "10*(numberOfEqualities"
                "+numberOfInequalities+numberOfVariables)", "ConstraintTolerance": 1e-7,
                "OptimalityTolerance": 1e-7, "MaxTime": np.inf, "Preprocess": "basic"},
    "intlinprog": {"Algorithm": "highs", "Display": "iter", "AbsoluteGapTolerance": 1e-6,
                   "RelativeGapTolerance": 1e-4, "IntegerTolerance": 1e-6, "ConstraintTolerance": 1e-6,
                   "MaxNodes": 1e7, "MaxTime": 7200, "MaxFeasiblePoints": np.inf, "ObjectiveCutOff": np.inf,
                   "OutputFcn": None, "PlotFcn": None},
    "quadprog": {"Algorithm": "interior-point-convex", "Display": "final", "MaxIterations": 200,
                 "OptimalityTolerance": 1e-8, "StepTolerance": 1e-12, "ConstraintTolerance": 1e-8,
                 "LinearSolver": "auto"},
    "lsqlin": {"Algorithm": "interior-point", "Display": "final", "MaxIterations": 200,
               "OptimalityTolerance": 1e-8, "StepTolerance": 1e-12, "ConstraintTolerance": 1e-8,
               "LinearSolver": "auto"},
    "ga": {"Display": "final", "PopulationSize": "50 when numberOfVariables <= 5, else 200",
           "MaxGenerations": "100*numberOfVariables", "MaxStallGenerations": 50, "FunctionTolerance": 1e-6,
           "ConstraintTolerance": 1e-3, "EliteCount": "0.05*PopulationSize", "CrossoverFraction": 0.8,
           "FitnessLimit": -np.inf, "MaxTime": np.inf, "MaxStallTime": np.inf, "InitialPopulationMatrix": None,
           "InitialPopulationRange": None, "InitialScoresMatrix": None, "PopulationType": "doubleVector",
           "UseVectorized": False, "UseParallel": False, "HybridFcn": None, "OutputFcn": None, "PlotFcn": None,
           "CreationFcn": None, "CrossoverFcn": None, "MutationFcn": None, "SelectionFcn": None,
           "FitnessScalingFcn": None, "NonlinearConstraintAlgorithm": "auglag"},
    "gamultiobj": {"Display": "final", "PopulationSize": "50 when numberOfVariables <= 5, else 200",
                   "MaxGenerations": "200*numberOfVariables", "MaxStallGenerations": 100,
                   "FunctionTolerance": 1e-4, "ConstraintTolerance": 1e-3, "CrossoverFraction": 0.8,
                   "ParetoFraction": 0.35, "MaxTime": np.inf, "InitialPopulationMatrix": None,
                   "InitialPopulationRange": None, "PopulationType": "doubleVector", "UseVectorized": False,
                   "UseParallel": False, "OutputFcn": None, "PlotFcn": None},
    "particleswarm": {"Display": "final", "SwarmSize": "min(100,10*numberOfVariables)",
                      "MaxIterations": "200*numberOfVariables", "MaxStallIterations": 20,
                      "FunctionTolerance": 1e-6, "InertiaRange": np.array([[0.1, 1.1]]),
                      "SelfAdjustmentWeight": 1.49, "SocialAdjustmentWeight": 1.49,
                      "MinNeighborsFraction": 0.25, "ObjectiveLimit": -np.inf, "MaxTime": np.inf,
                      "MaxStallTime": np.inf, "InitialSwarmMatrix": None, "InitialSwarmSpan": 2000,
                      "UseVectorized": False, "UseParallel": False, "HybridFcn": None, "OutputFcn": None,
                      "PlotFcn": None, "CreationFcn": None},
    "patternsearch": {"Display": "final", "Algorithm": "classic", "MaxIterations": "100*numberOfVariables",
                      "MaxFunctionEvaluations": "2000*numberOfVariables", "MeshTolerance": 1e-6,
                      "StepTolerance": 1e-6, "FunctionTolerance": 1e-6, "ConstraintTolerance": 1e-6,
                      "InitialMeshSize": 1.0, "MaxMeshSize": np.inf, "MeshExpansionFactor": 2.0,
                      "MeshContractionFactor": 0.5, "PollMethod": "GPSPositiveBasis2N",
                      "UseCompletePoll": False, "PollOrderAlgorithm": "Consecutive", "ScaleMesh": True,
                      "AccelerateMesh": False, "Cache": "off", "MaxTime": np.inf, "SearchFcn": None,
                      "UseVectorized": False, "UseParallel": False, "OutputFcn": None, "PlotFcn": None,
                      "TolBind": 1e-3},
    "simulannealbnd": {"Display": "final", "InitialTemperature": 100.0, "ReannealInterval": 100,
                       "MaxIterations": np.inf, "MaxFunctionEvaluations": "3000*numberOfVariables",
                       "MaxStallIterations": "500*numberOfVariables", "FunctionTolerance": 1e-6,
                       "ObjectiveLimit": -np.inf, "MaxTime": np.inf, "TemperatureFcn": None, "AnnealingFcn": None,
                       "AcceptanceFcn": None, "HybridFcn": None, "OutputFcn": None, "PlotFcn": None},
    "surrogateopt": {"Display": "final", "MaxFunctionEvaluations": "max(200,50*numberOfVariables)",
                     "MinSurrogatePoints": "max(20,2*numberOfVariables)", "MinSampleDistance": 1e-6,
                     "ObjectiveLimit": -np.inf, "MaxTime": np.inf, "InitialPoints": None,
                     "ConstraintTolerance": 1e-3, "UseVectorized": False, "UseParallel": False,
                     "OutputFcn": None, "PlotFcn": None, "BatchUpdateInterval": 1},
}
SOLVER_ALGORITHMS = {
    "fmincon": ("interior-point", "sqp", "sqp-legacy", "active-set", "trust-region-reflective"),
    "fminunc": ("quasi-newton", "trust-region"),
    "fsolve": ("trust-region-dogleg", "trust-region", "levenberg-marquardt"),
    "lsqnonlin": ("trust-region-reflective", "levenberg-marquardt", "interior-point"),
    "lsqcurvefit": ("trust-region-reflective", "levenberg-marquardt", "interior-point"),
    "linprog": ("dual-simplex-highs", "dual-simplex", "dual-simplex-legacy", "interior-point",
                "interior-point-legacy"),
    "intlinprog": ("highs", "legacy"),
    "quadprog": ("interior-point-convex", "active-set", "trust-region-reflective"),
    "lsqlin": ("interior-point", "active-set", "trust-region-reflective"),
    "patternsearch": ("classic", "nups", "nups-gps", "nups-mads"),
}
_NO_OPTIMOPTIONS = ("fminsearch", "fminbnd", "fzero", "lsqnonneg")


def _is_empty(v) -> bool:
    return v is None or (isinstance(v, np.ndarray) and v.size == 0)


def is_fn(v) -> bool:
    return isinstance(v, FunctionHandle) or (is_text(v) and len(text(v)) > 0)


def canonical(name: str, solver: str | None) -> str:
    """Current option name for an old or current name (any letter case)."""
    low = name.lower()
    if low == "tolfun":
        return "OptimalityTolerance" if solver in _TOLFUN_IS_OPTIMALITY else "FunctionTolerance"
    if low == "jacobian" and solver in ("fsolve", "lsqnonlin", "lsqcurvefit", None):
        return "SpecifyObjectiveGradient"
    if low in LEGACY:
        return LEGACY[low]
    table = SOLVER_OPTIONS.get(solver or "", {})
    for k in table:
        if k.lower() == low:
            return k
    for tab in SOLVER_OPTIONS.values():
        for k in tab:
            if k.lower() == low:
                return k
    for k in OPTIMSET_FIELDS:
        if k.lower() == low:
            return k
    return name


def _to_python(v):
    if isinstance(v, np.ndarray) and v.dtype.kind in "biuf" and v.size == 1:
        x = v.flat[0]
        return bool(x) if v.dtype == bool else float(x)
    if is_text(v):
        return text(v)
    return v


class OptimOptions(MObject):
    """Options object made by optimoptions: one solver, its options and the values set so far."""

    def __init__(self, solver: str, user: dict | None = None):
        self.solver = solver
        self.user = dict(user or {})

    def class_name(self) -> str:
        return "optim.options." + {"fmincon": "Fmincon", "fminunc": "Fminunc", "fsolve": "Fsolve",
                                   "lsqnonlin": "Lsqnonlin", "lsqcurvefit": "Lsqcurvefit",
                                   "linprog": "Linprog", "intlinprog": "Intlinprog", "quadprog": "Quadprog",
                                   "lsqlin": "Lsqlin", "fminimax": "Fminimax", "fgoalattain": "Fgoalattain",
                                   "fseminf": "Fseminf", "ga": "GaOptions", "gamultiobj": "GamultiobjOptions",
                                   "particleswarm": "Particleswarm", "patternsearch": "PatternsearchOptions",
                                   "simulannealbnd": "SimulannealbndOptions",
                                   "surrogateopt": "Surrogateopt"}.get(self.solver, self.solver)

    def isa_class(self, name: str) -> bool:
        return name in (self.class_name(), "optim.options.SolverOptions")

    def property_names(self) -> list[str]:
        return list(SOLVER_OPTIONS[self.solver])

    def name_of(self, name: str) -> str:
        key = canonical(name, self.solver)
        if key == "SolverName":
            return key
        if key not in SOLVER_OPTIONS[self.solver]:
            raise MatlabError(f"'{name}' is not an option for {self.solver.upper()}.")
        return key

    def check(self, key: str, value):
        if key == "Algorithm":
            allowed = SOLVER_ALGORITHMS.get(self.solver)
            if not is_text(value) or (allowed and text(value) not in allowed):
                raise MatlabError(f"Invalid value for the Algorithm option of {self.solver}. Choices: "
                                  + ", ".join(f"'{a}'" for a in (allowed or ())) + ".")
        if key == "Display" and (not is_text(value) or text(value) not in (
                "off", "none", "final", "iter", "notify", "final-detailed", "iter-detailed", "notify-detailed",
                "diagnose", "testing")):
            raise MatlabError("Invalid value for the Display option. Choices: 'off', 'none', 'final', "
                              "'iter', 'notify' and the '-detailed' variants.")
        if key.endswith("Tolerance") or key in ("MaxIterations", "MaxFunctionEvaluations", "MaxTime"):
            if is_text(value) or isinstance(value, (Cell, Struct, FunctionHandle)) or fnum(value).size != 1 \
                    or not np.real(fnum(value).flat[0]) >= 0:
                raise MatlabError(f"Invalid value for the {key} option: it must be a nonnegative scalar.")
        return value

    def get_field(self, interp, name):
        if name == "SolverName":
            return char(self.solver)
        key = self.name_of(name)
        if key in self.user:
            v = self.user[key]
            return char(v) if isinstance(v, str) else v
        return _default_value(SOLVER_OPTIONS[self.solver][key])

    def set_field(self, interp, name, value):
        key = self.name_of(name)
        new = OptimOptions(self.solver, self.user)
        new.user[key] = self.check(key, value)
        return new

    def display_lines(self):
        lines = [f"  {self.solver} options:", ""]
        if "Algorithm" in SOLVER_OPTIONS[self.solver]:
            alg = _to_python(self.user.get("Algorithm", SOLVER_OPTIONS[self.solver]["Algorithm"]))
            lines += [f"   Options used by current Algorithm ('{alg}'):", ""]
        table = SOLVER_OPTIONS[self.solver]
        width = max(len(k) for k in table)

        def show(v):
            v = _to_python(v)
            if v is None:
                return "[]"
            if isinstance(v, bool):
                return "1" if v else "0"
            if isinstance(v, str):
                return f"'{v}'"
            if isinstance(v, float):
                return f"{v:g}" if np.isfinite(v) else ("Inf" if v > 0 else "-Inf")
            if isinstance(v, FunctionHandle):
                return repr(v)
            if isinstance(v, np.ndarray):
                return "[" + " ".join(f"{float(x):g}" for x in v.flatten()) + "]"
            return str(v)

        lines.append("   Set properties:")
        if self.user:
            lines += [f"   {k:>{width}}: {show(self.user[k])}" for k in sorted(self.user)]
        else:
            lines.append("     No options set.")
        lines += ["", "   Default properties:"]
        lines += [f"   {k:>{width}}: {show(v)}" for k, v in sorted(table.items()) if k not in self.user]
        return lines


def _default_value(v):
    if v is None:
        return empty_double()
    if isinstance(v, bool):
        return np.array([[v]])
    if isinstance(v, str):
        return char(v)
    if isinstance(v, (int, float)):
        return np.array([[float(v)]])
    return v


class Opt:
    """Uniform reader over optimset structures and optimoptions objects for one solver."""

    def __init__(self, raw, solver: str, implemented: tuple = ()):
        self.solver = solver
        self.user: dict = {}
        self.raw = raw
        if _is_empty(raw):
            return
        if isinstance(raw, OptimOptions):
            if raw.solver != solver and not (solver in SOLVER_OPTIONS and raw.solver in SOLVER_OPTIONS):
                raise MatlabError(f"{solver}: options made for {raw.solver} cannot be used here.")
            for k, v in raw.user.items():
                if solver not in SOLVER_OPTIONS or k in SOLVER_OPTIONS[solver]:
                    self.user[k] = v
        elif isinstance(raw, Struct) and raw.numel() == 1:
            for k, v in raw.data.flat[0].items():
                if not _is_empty(v):
                    self.user[canonical(k, solver)] = v
        else:
            raise MatlabError(f"{solver}: the options argument must come from optimset or optimoptions.")
        if implemented and solver in SOLVER_OPTIONS:
            for k, v in self.user.items():
                if k in SOLVER_OPTIONS[solver] and k not in implemented and not self._is_default(k, v):
                    raise MatlabError(f"{solver}: the option {k} is not supported.")

    def _is_default(self, key, value) -> bool:
        d = SOLVER_OPTIONS[self.solver][key]
        v = _to_python(value)
        if d is None:
            return _is_empty(v)
        if isinstance(d, bool):
            return self._truth(v) == d
        if isinstance(d, str) and isinstance(v, str):
            return v.lower() == d.lower()
        if isinstance(d, (int, float)) and isinstance(v, (int, float)):
            return float(d) == float(v)
        return False

    @staticmethod
    def _truth(v) -> bool:
        if isinstance(v, str):
            return v.lower() in ("on", "true", "yes")
        if isinstance(v, np.ndarray):
            return bool(np.any(v != 0))
        return bool(v)

    def has(self, name) -> bool:
        return name in self.user

    def get(self, name, default=None):
        return self.user.get(name, default)

    def num(self, name, default=None):
        if name not in self.user:
            return default
        v = self.user[name]
        if is_text(v):
            raise MatlabError(f"{self.solver}: the option {name} must be a number.")
        a = fnum(v)
        if a.size != 1:
            raise MatlabError(f"{self.solver}: the option {name} must be a scalar.")
        return float(np.real(a.flat[0]))

    def flag(self, name, default=False) -> bool:
        if name not in self.user:
            return default
        v = self.user[name]
        if is_text(v):
            w = text(v).lower()
            if w in ("on", "true", "yes"):
                return True
            if w in ("off", "false", "no"):
                return False
            raise MatlabError(f"{self.solver}: the option {name} must be 'on' or 'off' (or true/false).")
        return bool(np.any(fnum(v) != 0))

    def word(self, name, default: str, allowed: tuple | None = None) -> str:
        if name not in self.user:
            return default
        v = self.user[name]
        if not is_text(v):
            raise MatlabError(f"{self.solver}: the option {name} must be text.")
        w = text(v)
        if allowed is not None:
            for a in allowed:
                if a.lower() == w.lower():
                    return a
            raise MatlabError(f"{self.solver}: invalid value '{w}' for the option {name}. Choices: "
                              + ", ".join(f"'{a}'" for a in allowed) + ".")
        return w

    def display(self, default="final") -> str:
        d = self.word("Display", default).lower()
        if d in ("off", "none"):
            return "off"
        if d.startswith("iter"):
            return "iter"
        if d.startswith("notify"):
            return "notify"
        if d.startswith("final") or d in ("diagnose", "testing"):
            return "final"
        raise MatlabError(f"{self.solver}: invalid value '{d}' for the option Display.")


# -- optimset, optimget, optimoptions ------------------------------------------------------------------

_BASE_DEFAULTS = {
    "fminsearch": {"Display": "notify", "MaxFunEvals": "200*numberofvariables", "MaxIter":
                   "200*numberofvariables", "TolFun": 1e-4, "TolX": 1e-4, "FunValCheck": "off"},
    "fminbnd": {"Display": "notify", "MaxFunEvals": 500, "MaxIter": 500, "TolX": 1e-4, "FunValCheck": "off"},
    "fzero": {"Display": "notify", "TolX": EPS, "FunValCheck": "off"},
    "lsqnonneg": {"Display": "notify", "TolX": "10*eps*norm(C,1)*length(C)"},
}


def _optimset_field(name: str) -> str:
    low = name.lower()
    for k in OPTIMSET_FIELDS:
        if k.lower() == low:
            return k
    hits = [k for k in OPTIMSET_FIELDS if k.lower().startswith(low)]
    if len(hits) == 1:
        return hits[0]
    # current names are accepted too and stored under their older equivalent
    back = {v.lower(): k for k, v in LEGACY.items()}
    back.update({"optimalitytolerance": "tolfun", "functiontolerance": "tolfun",
                 "specifyobjectivegradient": "gradobj"})
    if low in back:
        return _optimset_field(back[low])
    if len(hits) > 1:
        raise MatlabError(f"Ambiguous parameter name '{name}' ({', '.join(hits)}).")
    raise MatlabError(f"Unrecognized parameter name '{name}'. See optimset for the list of options.")


@builtin("optimset")
def _optimset(interp, args, nargout):
    if not args and nargout == 0:
        width = max(len(n) for n in OPTIMSET_FIELDS[:8])
        hints = {"Display": "off | iter | iter-detailed | notify | notify-detailed | final | final-detailed",
                 "MaxFunEvals": "positive scalar", "MaxIter": "positive scalar", "TolFun": "positive scalar",
                 "TolX": "positive scalar", "FunValCheck": "on | {off}", "OutputFcn": "function | {[]}",
                 "PlotFcns": "function | {[]}"}
        for n in OPTIMSET_FIELDS[:8]:
            interp.out(f"{n:>{width + 4}}: [ {hints[n]} ]\n")
        return None
    out = Struct.scalar({n: empty_double() for n in OPTIMSET_FIELDS})
    rest = list(args)
    if len(rest) == 1 and is_fn(rest[0]) and not isinstance(rest[0], Struct):
        name = rest[0].name if isinstance(rest[0], FunctionHandle) else text(rest[0])
        table = _BASE_DEFAULTS.get(name)
        if table is None and name in SOLVER_OPTIONS:
            table = {}
            for k, v in SOLVER_OPTIONS[name].items():
                for old, new in list(LEGACY.items()) + [("tolfun", canonical("TolFun", name))]:
                    if new == k and any(f.lower() == old for f in OPTIMSET_FIELDS):
                        table[_optimset_field(old)] = v
                if k in OPTIMSET_FIELDS:
                    table[k] = v
        if table is None:
            raise MatlabError(f"optimset: no default options are available for '{name}'.")
        for k, v in table.items():
            out.set_field(0, k, _default_value(v))
        return out
    while rest and (isinstance(rest[0], Struct) or (isinstance(rest[0], np.ndarray) and rest[0].size == 0
                                                    and not is_text(rest[0]))):
        s = rest.pop(0)
        if isinstance(s, Struct):
            for k, v in s.data.flat[0].items():
                if not _is_empty(v):
                    out.set_field(0, _optimset_field(k), v)
    if len(rest) % 2:
        raise MatlabError("optimset: arguments must be name-value pairs.")
    for k in range(0, len(rest), 2):
        if not is_text(rest[k]):
            raise MatlabError(f"optimset: expected an option name at argument {k + 1}.")
        out.set_field(0, _optimset_field(text(rest[k])), rest[k + 1])
    return out


@builtin("optimget")
def _optimget(interp, args, nargout):
    nargs(args, 2, 4, "optimget")
    o = args[0]
    default = args[2] if len(args) > 2 else empty_double()
    name = text(args[1])
    if isinstance(o, OptimOptions):
        key = o.name_of(name)
        return o.user[key] if key in o.user else default
    if _is_empty(o):
        return default
    if not isinstance(o, Struct):
        raise MatlabError("optimget: the first argument must be an options structure.")
    field = _optimset_field(name)
    for k, v in o.data.flat[0].items():
        if k.lower() == field.lower() and not _is_empty(v):
            return v
    return default


@builtin("optimoptions")
def _optimoptions(interp, args, nargout):
    nargs(args, 1, None, "optimoptions")
    first = args[0]
    rest = list(args[1:])
    if isinstance(first, OptimOptions):
        obj = OptimOptions(first.solver, first.user)
    else:
        if isinstance(first, FunctionHandle) and not first.is_anonymous:
            solver = first.name
        elif is_text(first):
            solver = text(first)
        else:
            raise MatlabError("optimoptions: the first argument must be a solver name, a solver function "
                              "handle or an options object.")
        if solver in _NO_OPTIMOPTIONS:
            raise MatlabError(f"optimoptions does not support {solver}; use optimset for this solver.")
        if solver not in SOLVER_OPTIONS:
            raise MatlabError(f"optimoptions: '{solver}' is not a solver with options.")
        obj = OptimOptions(solver)
        if rest and isinstance(rest[0], OptimOptions):
            old = rest.pop(0)
            for k, v in old.user.items():
                if k in SOLVER_OPTIONS[solver]:
                    obj.user[k] = v
        elif rest and isinstance(rest[0], Struct):
            old = rest.pop(0)
            for k, v in old.data.flat[0].items():
                key = canonical(k, solver)
                if not _is_empty(v) and key in SOLVER_OPTIONS[solver]:
                    obj.user[key] = v
    if len(rest) % 2:
        raise MatlabError("optimoptions: options must be name-value pairs.")
    for k in range(0, len(rest), 2):
        if not is_text(rest[k]):
            raise MatlabError(f"optimoptions: expected an option name at argument {k + 2}.")
        key = obj.name_of(text(rest[k]))
        value = rest[k + 1]
        # the on/off values of the older names become logical values
        if isinstance(SOLVER_OPTIONS[obj.solver][key], bool) and is_text(value):
            value = np.array([[Opt._truth(text(value))]])
        obj.user[key] = obj.check(key, value)
    return obj


@builtin("resetoptions")
def _resetoptions(interp, args, nargout):
    nargs(args, 2, 2, "resetoptions")
    o = args[0]
    if not isinstance(o, OptimOptions):
        raise MatlabError("resetoptions: the first argument must be an options object from optimoptions.")
    names = args[1].values_f() if isinstance(args[1], Cell) else [args[1]]
    new = OptimOptions(o.solver, o.user)
    for nm in names:
        new.user.pop(o.name_of(text(nm)), None)
    return new


# -- shared machinery for the solvers ---------------------------------------------------------------------

def _arg(args: list, i: int):
    """Numeric argument i as a float array, or None when missing or []."""
    if len(args) <= i:
        return None
    a = args[i]
    if isinstance(a, (Struct, OptimOptions, FunctionHandle, Cell)) or is_text(a):
        return None
    if isinstance(a, np.ndarray) and a.size == 0:
        return None
    return np.real(fnum(a)).astype(float)


def _options_at(args: list, i: int):
    return args[i] if len(args) > i and isinstance(args[i], (Struct, OptimOptions)) else None


def bounds(lb, ub, n: int, what: str):
    lo = np.full(n, -np.inf)
    hi = np.full(n, np.inf)
    if lb is not None:
        v = lb.flatten(order="F")
        if v.size > n:
            raise MatlabError(f"{what}: the lower bound has more entries than there are variables.")
        lo[:v.size] = v
    if ub is not None:
        v = ub.flatten(order="F")
        if v.size > n:
            raise MatlabError(f"{what}: the upper bound has more entries than there are variables.")
        hi[:v.size] = v
    if np.any(lo > hi):
        raise MatlabError(f"{what}: the lower bounds must not exceed the upper bounds.")
    return lo, hi


def linear(A, b, n: int, what: str, name: str = "A"):
    if A is None:
        return np.zeros((0, n)), np.zeros(0)
    A = np.asarray(A, dtype=float)
    if A.ndim == 1 or A.shape[1] != n:
        if A.size % n:
            raise MatlabError(f"{what}: {name} must have {n} columns.")
        A = A.reshape(-1, n)
    if b is None or b.size != A.shape[0]:
        raise MatlabError(f"{what}: {name} and its right-hand side must have the same number of rows.")
    return A, b.flatten(order="F").astype(float)


def outs(nargout: int, vals: tuple):
    if nargout <= 1:
        return vals[0]
    if nargout > len(vals):
        raise MatlabError("Too many output arguments.")
    return tuple(vals[:nargout])


def problem_args(args: list, order: tuple, what: str) -> list:
    """Expand a problem structure into positional arguments (order = field names)."""
    if len(args) == 1 and isinstance(args[0], Struct) and args[0].numel() == 1:
        d = {k.lower(): v for k, v in args[0].data.flat[0].items()}
        solver = d.get("solver")
        if solver is not None and is_text(solver) and text(solver) != what:
            raise MatlabError(f"{what}: the problem structure is for the solver '{text(solver)}'.")
        out = []
        for name in order:
            alts = name if isinstance(name, tuple) else (name,)
            v = next((d[a.lower()] for a in alts if a.lower() in d), empty_double())
            out.append(v)
        return out
    return list(args)


class Report:
    """Collects what a solver prints and returns: display level, iteration lines, exit message."""

    def __init__(self, interp, opts: Opt, default_display="final"):
        self.interp = interp
        self.level = opts.display(default_display)
        self.header_done = False

    def iter_line(self, header: str, line: str):
        if self.level == "iter":
            if not self.header_done:
                self.interp.out(header + "\n")
                self.header_done = True
            self.interp.out(line + "\n")

    def finish(self, exitflag: float, message: str):
        if self.level in ("final", "iter") or (self.level == "notify" and exitflag <= 0):
            self.interp.out("\n" + message + "\n\n")


class OutputHook:
    """Calls the user's OutputFcn(s) as stop = f(x, optimValues, state)."""

    def __init__(self, interp, opts: Opt, shape, what: str):
        self.interp, self.shape, self.what = interp, shape, what
        if opts.has("PlotFcn"):
            raise MatlabError(f"{what}: the PlotFcn option is not supported.")
        v = opts.get("OutputFcn")
        if v is None:
            self.fns = []
        elif isinstance(v, Cell):
            self.fns = v.values_f()
        else:
            self.fns = [v]
        for f in self.fns:
            if not is_fn(f):
                raise MatlabError(f"{what}: OutputFcn must be a function handle or a cell array of them.")
        self.stopped = False

    def __bool__(self):
        return bool(self.fns)

    def call(self, x, state: str, **values) -> bool:
        if not self.fns:
            return False
        fields = {}
        for k, v in values.items():
            fields[k] = char(v) if isinstance(v, str) else (
                np.array([[float(v)]]) if np.isscalar(v) else np.asarray(v, dtype=float))
        ov = Struct.scalar(fields)
        xa = np.asarray(x, dtype=float).reshape(self.shape, order="F")
        for f in self.fns:
            res = self.interp.call_function_value(f, [xa, ov, char(state)], 1)
            if res and isinstance(res[0], np.ndarray) and res[0].size and bool(np.any(res[0] != 0)):
                self.stopped = True
        return self.stopped


class Objective:
    """Wraps a MATLAB objective: reshapes x, counts calls, checks values, supplies gradients."""

    def __init__(self, interp, fun, shape, opts: Opt, what: str, extra=(), vector=False):
        if not is_fn(fun):
            raise MatlabError(f"{what}: the objective must be a function handle.")
        self.interp, self.fun, self.shape, self.what = interp, fun, shape, what
        self.extra = list(extra)
        self.vector = vector            # vector-valued (least squares, equations) or scalar
        self.count = 0
        self.check = opts.flag("FunValCheck")
        self.user_grad = opts.flag("SpecifyObjectiveGradient")
        self.nout = 2 if self.user_grad else 1
        self.central = opts.word("FiniteDifferenceType", "forward", ("forward", "central")) == "central"
        self.typical = None
        if opts.has("TypicalX") and not is_text(opts.get("TypicalX")):
            self.typical = np.abs(fnum(opts.get("TypicalX")).flatten(order="F"))
        self.rel = None
        if opts.has("FiniteDifferenceStepSize") and not is_text(opts.get("FiniteDifferenceStepSize")):
            self.rel = np.abs(fnum(opts.get("FiniteDifferenceStepSize")).flatten(order="F"))
        self.lo = self.hi = None
        self.out_shape = None
        self._key = None
        self._val = None
        self._grad = None

    def _x(self, x):
        return np.asarray(x, dtype=float).reshape(self.shape, order="F")

    def _convert(self, v):
        a = fnum(v)
        if self.check and (np.iscomplexobj(a) and np.any(np.imag(a) != 0) or not np.all(np.isfinite(a))):
            raise MatlabError(f"{self.what}: the objective function returned a complex, NaN or Inf value "
                              "(FunValCheck is 'on').")
        if self.vector:
            if self.out_shape is None:
                self.out_shape = a.shape
            return np.real(a).astype(float).flatten(order="F")
        if a.size != 1:
            raise MatlabError(f"{self.what}: the objective function must return a scalar value.")
        return float(np.real(a.flat[0]))

    def _eval(self, x, want_grad: bool):
        key = np.asarray(x, dtype=float).tobytes()
        if key == self._key and (not want_grad or self._grad is not None):
            return
        # with a user gradient the function is always asked for both outputs, so that objectives
        # written as @(x) deal(f, g) work
        nout = self.nout
        res = self.interp.call_function_value(self.fun, [self._x(x), *self.extra], nout)
        self.count += 1
        if not res:
            raise MatlabError(f"{self.what}: the objective function must return a value.")
        self._key, self._val = key, self._convert(res[0])
        self._grad = None
        if nout == 2:
            if len(res) < 2:
                raise MatlabError(f"{self.what}: with SpecifyObjectiveGradient the objective must return "
                                  "the gradient as its second output.")
            g = np.real(fnum(res[1])).astype(float)
            n = int(np.prod(self.shape))
            if self.vector:
                if g.shape != (self._val.size, n):
                    raise MatlabError(f"{self.what}: the Jacobian must be {self._val.size}-by-{n}.")
            else:
                if g.size != n:
                    raise MatlabError(f"{self.what}: the gradient must have {n} elements.")
                g = g.flatten(order="F")
            self._grad = g

    def __call__(self, x):
        self._eval(x, False)
        return self._val

    def steps(self, x):
        scale = np.maximum(np.abs(x), 1.0 if self.typical is None else self.typical)
        rel = self.rel if self.rel is not None else (EPS ** (1 / 3) if self.central else np.sqrt(EPS))
        h = rel * scale
        return np.where(x < 0, -h, h)

    def grad(self, x):
        """Gradient (scalar objective) or Jacobian (vector objective)."""
        x = np.asarray(x, dtype=float)
        if self.user_grad:
            self._eval(x, True)
            return self._grad
        f0 = self(x)
        h = self.steps(x)
        n = x.size
        J = np.zeros((np.size(f0), n))
        for j in range(n):
            hj = h[j]
            xp = x.copy()
            if self.hi is not None and x[j] + hj > self.hi[j] and x[j] - abs(hj) >= self.lo[j]:
                hj = -abs(hj)
            elif self.lo is not None and x[j] + hj < self.lo[j] and x[j] + abs(hj) <= self.hi[j]:
                hj = abs(hj)
            xp[j] = x[j] + hj
            fp = self._raw(xp)
            if self.central:
                xm = x.copy()
                xm[j] = x[j] - hj
                inside = self.lo is None or (self.lo[j] <= xm[j] <= self.hi[j])
                if inside:
                    J[:, j] = (np.atleast_1d(fp) - np.atleast_1d(self._raw(xm))) / (2 * hj)
                    continue
            J[:, j] = (np.atleast_1d(fp) - np.atleast_1d(f0)) / hj
        return J if self.vector else J[0]

    def _raw(self, x):
        res = self.interp.call_function_value(self.fun, [self._x(x), *self.extra], self.nout)
        self.count += 1
        return self._convert(res[0])

    def check_gradient(self, x):
        """CheckGradients: compare the user's derivative with finite differences at x."""
        if not self.user_grad:
            return
        g_user = np.atleast_2d(self.grad(x))
        self.user_grad = False
        try:
            self._grad = None
            g_fd = np.atleast_2d(self.grad(x))
        finally:
            self.user_grad = True
            self._key = None
        err = float(np.max(np.abs(g_user - g_fd) / np.maximum(1.0, np.abs(g_fd))))
        if err > 1e-6 * (1e3 if not self.central else 1.0):
            raise MatlabError(f"{self.what}: the supplied derivative does not match a finite-difference "
                              f"approximation (largest relative difference {err:.3g}); CheckGradients failed.")


def output_struct(**fields) -> Struct:
    d = {}
    for k, v in fields.items():
        if isinstance(v, str):
            d[k] = char(v)
        elif isinstance(v, (int, float, np.integer, np.floating)):
            d[k] = np.array([[float(v)]])
        else:
            d[k] = v
    return Struct.scalar(d)


def kkt_multipliers(g, normals: list, is_eq: list):
    """Multipliers lam with g + sum lam_i * normal_i ~ 0, lam >= 0 for inequalities."""
    if not normals:
        return np.zeros(0), float(np.max(np.abs(g), initial=0.0))
    from scipy.optimize import lsq_linear
    N = np.column_stack(normals)
    lo = np.where(np.array(is_eq), -np.inf, 0.0)
    try:
        res = lsq_linear(N, -g, bounds=(lo, np.full(N.shape[1], np.inf)), method="bvls" if N.shape[1] <= N.shape[0]
                         else "trf")
        lam = res.x
    except Exception:
        lam = np.linalg.lstsq(N, -g, rcond=None)[0]
        lam = np.where(np.array(is_eq), lam, np.maximum(lam, 0.0))
    return lam, float(np.max(np.abs(g + N @ lam), initial=0.0))


from . import _optim_local, _optim_global, _optim_problem  # noqa: E402,F401  (register the solvers)
