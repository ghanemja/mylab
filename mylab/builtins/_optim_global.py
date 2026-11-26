"""Global optimisation: genetic algorithms, particle swarm, pattern search, simulated annealing,
surrogate optimisation, and multi-start drivers."""

from __future__ import annotations

import time

import numpy as np

from ..errors import MatlabError
from ..objects import MObject
from ..values import Cell, FunctionHandle, Struct, char, empty_double
from ._registry import builtin
from ._util import fnum, is_text, nargs, text
from . import optim_more as om
from .optim_more import Opt, OutputHook, Report, outs, output_struct


def _rng(interp):
    from .core import _rng as core_rng
    return core_rng(interp)


def _fn_name(v):
    """Name of a named function handle (or the first entry of a {@fcn, args...} cell)."""
    if isinstance(v, Cell):
        items = v.values_f()
        return _fn_name(items[0]) if items else None
    if isinstance(v, FunctionHandle) and not v.is_anonymous:
        return v.name
    if is_text(v):
        return text(v)
    return None


def _fn_args(v) -> list:
    if isinstance(v, Cell):
        return [float(np.real(fnum(a).flat[0])) for a in v.values_f()[1:] if not is_text(a)]
    return []


class _Fitness:
    """Objective called with row vectors (optionally a whole population at once)."""

    def __init__(self, interp, fun, what, vectorized=False, nobj=None):
        if not om.is_fn(fun):
            raise MatlabError(f"{what}: the objective must be a function handle.")
        self.interp, self.fun, self.what, self.vectorized = interp, fun, what, vectorized
        self.count = 0
        self.nobj = nobj

    def one(self, x):
        res = self.interp.call_function_value(self.fun, [np.asarray(x, dtype=float).reshape(1, -1)], 1)
        self.count += 1
        if not res:
            raise MatlabError(f"{self.what}: the objective function must return a value.")
        v = np.real(fnum(res[0])).astype(float).flatten()
        if self.nobj is None:
            if v.size != 1:
                raise MatlabError(f"{self.what}: the objective function must return a scalar.")
            return float(v[0])
        return v

    def many(self, X):
        X = np.atleast_2d(X)
        if self.vectorized:
            res = self.interp.call_function_value(self.fun, [np.asarray(X, dtype=float)], 1)
            self.count += X.shape[0]
            v = np.real(fnum(res[0])).astype(float)
            if self.nobj is None:
                if v.size != X.shape[0]:
                    raise MatlabError(f"{self.what}: with UseVectorized the objective must return one value "
                                      "per row of the population.")
                return v.flatten(order="F")
            return v.reshape(X.shape[0], -1)
        vals = [self.one(x) for x in X]
        return np.array(vals) if self.nobj is None else np.vstack(vals)


class _Constraints:
    """Linear and nonlinear constraints with a single violation measure."""

    def __init__(self, interp, n, A, b, Aeq, beq, lo, hi, nonlcon, what, tol=1e-3):
        self.interp, self.n, self.what, self.tol = interp, n, what, tol
        self.A, self.b, self.Aeq, self.beq, self.lo, self.hi = A, b, Aeq, beq, lo, hi
        self.nonlcon = nonlcon if (nonlcon is not None and not om._is_empty(nonlcon)) else None
        if self.nonlcon is not None and not om.is_fn(self.nonlcon):
            raise MatlabError(f"{what}: the nonlinear constraint must be a function handle.")
        self.any = bool(A.shape[0] or Aeq.shape[0] or self.nonlcon is not None)

    def nonlinear(self, x):
        if self.nonlcon is None:
            return np.zeros(0), np.zeros(0)
        res = self.interp.call_function_value(self.nonlcon, [np.asarray(x, dtype=float).reshape(1, -1)], 2)
        if len(res) < 2:
            raise MatlabError(f"{self.what}: the nonlinear constraint function must return [c, ceq].")
        return (np.real(fnum(res[0])).astype(float).flatten(), np.real(fnum(res[1])).astype(float).flatten())

    def violation(self, x) -> float:
        v = 0.0
        if self.A.shape[0]:
            v += float(np.sum(np.maximum(self.A @ x - self.b, 0.0)))
        if self.Aeq.shape[0]:
            v += float(np.sum(np.abs(self.Aeq @ x - self.beq)))
        if self.nonlcon is not None:
            c, ceq = self.nonlinear(x)
            v += float(np.sum(np.maximum(c, 0.0)) + np.sum(np.abs(ceq)))
        v += float(np.sum(np.maximum(self.lo - x, 0.0)) + np.sum(np.maximum(x - self.hi, 0.0)))
        return v

    def max_violation(self, x) -> float:
        parts = [0.0]
        if self.A.shape[0]:
            parts.append(float(np.max(self.A @ x - self.b)))
        if self.Aeq.shape[0]:
            parts.append(float(np.max(np.abs(self.Aeq @ x - self.beq))))
        if self.nonlcon is not None:
            c, ceq = self.nonlinear(x)
            if c.size:
                parts.append(float(np.max(c)))
            if ceq.size:
                parts.append(float(np.max(np.abs(ceq))))
        return max(parts)

    def repair(self, x, integer=None):
        """Clip to the bounds and project onto the linear equalities."""
        x = np.clip(x, self.lo, self.hi)
        if self.Aeq.shape[0] and integer is None:
            for _ in range(5):
                r = self.Aeq @ x - self.beq
                if np.max(np.abs(r)) < 1e-12:
                    break
                x = x - np.linalg.lstsq(self.Aeq, r, rcond=None)[0]
                x = np.clip(x, self.lo, self.hi)
        return x


def _problem_type(cons: _Constraints, integer: bool) -> str:
    if integer:
        return "integerconstraints"
    if cons.nonlcon is not None:
        return "nonlinearconstr"
    if cons.A.shape[0] or cons.Aeq.shape[0]:
        return "linearconstraints"
    if np.any(np.isfinite(cons.lo)) or np.any(np.isfinite(cons.hi)):
        return "boundconstraints"
    return "unconstrained"


def _init_range(lo, hi, rng_opt, n, span=20.0):
    """Box used to create the first population."""
    a = np.where(np.isfinite(lo), lo, -np.inf)
    b = np.where(np.isfinite(hi), hi, np.inf)
    base_lo, base_hi = np.full(n, -span / 2), np.full(n, span / 2)
    if rng_opt is not None:
        r = np.real(fnum(rng_opt)).astype(float)
        if r.shape[0] != 2 or r.shape[1] not in (1, n):
            raise MatlabError("The initial range must be a 2-row matrix [lower; upper].")
        base_lo = np.broadcast_to(r[0], (n,)).copy()
        base_hi = np.broadcast_to(r[1], (n,)).copy()
    out_lo = np.where(np.isfinite(a), a, np.where(np.isfinite(b), np.minimum(base_lo, b - span), base_lo))
    out_hi = np.where(np.isfinite(b), b, np.where(np.isfinite(a), np.maximum(base_hi, a + span), base_hi))
    if rng_opt is not None:
        out_lo = np.maximum(out_lo, np.where(np.isfinite(a), np.maximum(a, base_lo), base_lo))
        out_hi = np.minimum(out_hi, np.where(np.isfinite(b), np.minimum(b, base_hi), base_hi))
        out_hi = np.maximum(out_hi, out_lo)
    return out_lo, out_hi


def _hybrid(interp, hybrid, fun, x, cons: _Constraints, what: str):
    """Run a local solver named by HybridFcn from x; returns (x, fval, flag) or None."""
    name = _fn_name(hybrid)
    if name is None:
        return None
    extra_opts = None
    if isinstance(hybrid, Cell) and len(hybrid.values_f()) > 1:
        extra_opts = hybrid.values_f()[1]
    xrow = np.asarray(x, dtype=float).reshape(1, -1)
    e = empty_double()

    def mat(M):
        return M if M.shape[0] else e

    if name == "fmincon":
        a = [fun, xrow, mat(cons.A), cons.b.reshape(-1, 1) if cons.A.shape[0] else e, mat(cons.Aeq),
             cons.beq.reshape(-1, 1) if cons.Aeq.shape[0] else e, cons.lo.reshape(1, -1), cons.hi.reshape(1, -1),
             cons.nonlcon if cons.nonlcon is not None else e,
             extra_opts if extra_opts is not None else om.OptimOptions("fmincon", {"Display": char("off")})]
    elif name == "patternsearch":
        a = [fun, xrow, mat(cons.A), cons.b.reshape(-1, 1) if cons.A.shape[0] else e, mat(cons.Aeq),
             cons.beq.reshape(-1, 1) if cons.Aeq.shape[0] else e, cons.lo.reshape(1, -1), cons.hi.reshape(1, -1),
             cons.nonlcon if cons.nonlcon is not None else e,
             extra_opts if extra_opts is not None else om.OptimOptions("patternsearch", {"Display": char("off")})]
    elif name in ("fminunc", "fminsearch"):
        if cons.any or np.any(np.isfinite(cons.lo)) or np.any(np.isfinite(cons.hi)):
            raise MatlabError(f"{what}: the hybrid function {name} cannot be used for constrained problems; "
                              "use fmincon or patternsearch.")
        a = [fun, xrow, extra_opts if extra_opts is not None else Struct.scalar({"Display": char("off")})]
    else:
        raise MatlabError(f"{what}: HybridFcn must be @fmincon, @fminunc, @fminsearch or @patternsearch.")
    res = interp.call_function_value(char(name), a, 3)
    return np.real(fnum(res[0])).astype(float).flatten(), float(fnum(res[1]).flat[0]), float(fnum(res[2]).flat[0])


def _rng_state(interp) -> Struct:
    return Struct.scalar({"Type": char("twister"), "Seed": np.array([[float(interp.state.get("rng_seed", 0) or 0)]]),
                          "State": empty_double()})


# -- genetic algorithm ---------------------------------------------------------------------------------

_GA_IMPL = ("Display", "PopulationSize", "MaxGenerations", "MaxStallGenerations", "FunctionTolerance",
            "ConstraintTolerance", "EliteCount", "CrossoverFraction", "FitnessLimit", "MaxTime", "MaxStallTime",
            "InitialPopulationMatrix", "InitialPopulationRange", "InitialScoresMatrix", "PopulationType",
            "UseVectorized", "UseParallel", "HybridFcn", "OutputFcn", "PlotFcn", "CreationFcn", "CrossoverFcn",
            "MutationFcn", "SelectionFcn", "NonlinearConstraintAlgorithm", "ParetoFraction")
_CROSSOVERS = ("crossoverscattered", "crossoverintermediate", "crossoversinglepoint", "crossovertwopoint",
               "crossoverheuristic", "crossoverarithmetic", "crossoverlaplace")
_MUTATIONS = ("mutationgaussian", "mutationuniform", "mutationadaptfeasible", "mutationpower",
              "mutationpositivebasis")
_SELECTIONS = ("selectionstochunif", "selectiontournament", "selectionroulette", "selectionuniform",
               "selectionremainder")
_CREATIONS = ("gacreationuniform", "gacreationlinearfeasible", "gacreationnonlinearfeasible",
              "gacreationuniformint", "gacreationsobol")


def _named_option(opts: Opt, key: str, allowed: tuple, default: str, what: str):
    if not opts.has(key):
        return default, []
    v = opts.get(key)
    name = _fn_name(v)
    if name is None or name not in allowed:
        raise MatlabError(f"{what}: {key} must be one of the built-in functions "
                          + ", ".join("@" + a for a in allowed) + "; custom functions are not supported.")
    return name, _fn_args(v)


def _ga_setup(interp, args, what: str):
    fun = args[0]
    nv = fnum(args[1])
    if nv.size != 1 or nv.flat[0] < 1 or nv.flat[0] != int(nv.flat[0]):
        raise MatlabError(f"{what}: the number of variables must be a positive integer.")
    n = int(nv.flat[0])
    A, b = om.linear(om._arg(args, 2), om._arg(args, 3), n, what, "A")
    Aeq, beq = om.linear(om._arg(args, 4), om._arg(args, 5), n, what, "Aeq")
    lo, hi = om.bounds(om._arg(args, 6), om._arg(args, 7), n, what)
    nonlcon = args[8] if len(args) > 8 and om.is_fn(args[8]) else None
    return fun, n, A, b, Aeq, beq, lo, hi, nonlcon


def _ga_children(rng, pop, rank_key, n_elite, n_cross, n_mut, lo_box, hi_box, sigma, intmask, cons,
                 crossover, cross_args, mutation, selection, sel_args):
    """One generation of children from a ranked population (best first)."""
    npop, n = pop.shape

    def select(k):
        if selection == "selectiontournament":
            size = int(sel_args[0]) if sel_args else 4
            cand = rng.integers(0, npop, size=(k, size))
            return cand.min(axis=1)          # the population is sorted, so the lowest index wins
        if selection == "selectionuniform":
            return rng.integers(0, npop, size=k)
        # rank-based expectation (stochastic universal sampling / roulette)
        w = 1.0 / np.sqrt(np.arange(1, npop + 1))
        w = w / w.sum()
        if selection == "selectionroulette":
            return rng.choice(npop, size=k, p=w)
        edges = np.cumsum(w)
        start = rng.random() / k
        picks = np.searchsorted(edges, start + np.arange(k) / k)
        return rng.permutation(np.minimum(picks, npop - 1))

    kids = []
    pa, pb = select(n_cross), select(n_cross)
    for i in range(n_cross):
        p1, p2 = pop[pa[i]], pop[pb[i]]
        if crossover == "crossoverscattered":
            mask = rng.random(n) < 0.5
            child = np.where(mask, p1, p2)
        elif crossover == "crossoversinglepoint":
            k = rng.integers(1, n) if n > 1 else 0
            child = np.concatenate([p1[:k], p2[k:]])
        elif crossover == "crossovertwopoint":
            k1, k2 = sorted(rng.integers(0, n + 1, size=2))
            child = p1.copy()
            child[k1:k2] = p2[k1:k2]
        elif crossover == "crossoverheuristic":
            ratio = cross_args[0] if cross_args else 1.2
            better, worse = (p1, p2) if pa[i] <= pb[i] else (p2, p1)
            child = worse + ratio * (better - worse)
        elif crossover == "crossoverarithmetic":
            a = rng.random()
            child = a * p1 + (1 - a) * p2
        else:  # intermediate (also used for laplace): a random point on and slightly beyond the segment
            ratio = cross_args[0] if cross_args else 1.0
            r = rng.uniform(-0.25, 1.25, size=n) if not cross_args else rng.random(n) * ratio
            child = p1 + r * (p2 - p1)
        kids.append(child)
    pm = select(n_mut)
    for i in range(n_mut):
        p = pop[pm[i]].copy()
        if mutation == "mutationuniform":
            rate = sigma["rate"]
            m = rng.random(n) < rate
            if not m.any():
                m[rng.integers(0, n)] = True
            p[m] = lo_box[m] + rng.random(int(m.sum())) * (hi_box[m] - lo_box[m])
        else:
            p = p + sigma["scale"] * rng.standard_normal(n)
        if intmask.any():
            jump = rng.random(n) < 0.3
            step = np.where(jump, rng.integers(-2, 3, size=n), 0)
            p[intmask] = p[intmask] + step[intmask]
        kids.append(p)
    kids = np.array(kids) if kids else np.zeros((0, n))
    for i in range(kids.shape[0]):
        kids[i] = cons.repair(kids[i], intmask if intmask.any() else None)
    if intmask.any():
        kids[:, intmask] = np.clip(np.round(kids[:, intmask]), np.ceil(cons.lo[intmask]), np.floor(cons.hi[intmask]))
    return kids


@builtin("ga")
def _ga(interp, args, nargout):
    args = om.problem_args(args, ("fitnessfcn", "nvars", "Aineq", "bineq", "Aeq", "beq", "lb", "ub", "nonlcon",
                                  "intcon", "options"), "ga")
    nargs(args, 2, 11, "ga")
    fun, n, A, b, Aeq, beq, lo, hi, nonlcon = _ga_setup(interp, args, "ga")
    intcon = None
    raw_opts = None
    for a in args[9:]:
        if isinstance(a, (Struct, om.OptimOptions)):
            raw_opts = a
        elif not om._is_empty(a) and not is_text(a):
            intcon = np.real(fnum(a)).flatten()
    if len(args) > 9 and isinstance(args[9], (Struct, om.OptimOptions)):
        raw_opts = args[9]
    opts = Opt(raw_opts, "ga", _GA_IMPL)
    intmask = np.zeros(n, dtype=bool)
    if intcon is not None and intcon.size:
        if np.any(intcon < 1) or np.any(intcon > n) or np.any(intcon != np.round(intcon)):
            raise MatlabError(f"ga: intcon must list variable numbers between 1 and {n}.")
        intmask[intcon.astype(int) - 1] = True
        if Aeq.shape[0]:
            raise MatlabError("ga: linear equality constraints cannot be combined with integer variables.")
    if opts.word("PopulationType", "doubleVector") != "doubleVector":
        raise MatlabError("ga: only PopulationType 'doubleVector' is supported.")
    rng = _rng(interp)
    tol_con = opts.num("ConstraintTolerance", 1e-3)
    cons = _Constraints(interp, n, A, b, Aeq, beq, lo, hi, nonlcon, "ga", tol_con)
    fit = _Fitness(interp, fun, "ga", opts.flag("UseVectorized"))
    npop = int(opts.num("PopulationSize", (50 if n <= 5 else 200) if not intmask.any()
                        else min(max(10 * n, 40), 100)))
    if npop < 2:
        raise MatlabError("ga: PopulationSize must be at least 2.")
    max_gen = int(opts.num("MaxGenerations", 100 * n))
    stall_gen = int(opts.num("MaxStallGenerations", 50))
    tol_fun = opts.num("FunctionTolerance", 1e-6)
    n_elite = int(opts.num("EliteCount", max(1, int(np.ceil(0.05 * npop)))))
    n_elite = min(max(n_elite, 0), npop - 1)
    frac = opts.num("CrossoverFraction", 0.8)
    if not 0 <= frac <= 1:
        raise MatlabError("ga: CrossoverFraction must be between 0 and 1.")
    limit = opts.num("FitnessLimit", -np.inf)
    max_time = opts.num("MaxTime", np.inf)
    constrained = cons.any or bool(np.any(np.isfinite(lo)) or np.any(np.isfinite(hi)))
    crossover, cross_args = _named_option(opts, "CrossoverFcn", _CROSSOVERS,
                                          "crossoverintermediate" if constrained else "crossoverscattered", "ga")
    mutation, mut_args = _named_option(opts, "MutationFcn", _MUTATIONS,
                                       "mutationadaptfeasible" if constrained else "mutationgaussian", "ga")
    selection, sel_args = _named_option(opts, "SelectionFcn", _SELECTIONS,
                                        "selectiontournament" if intmask.any() else "selectionstochunif", "ga")
    _named_option(opts, "CreationFcn", _CREATIONS, "gacreationuniform", "ga")
    rep = Report(interp, opts)
    if opts.has("OutputFcn") or opts.has("PlotFcn"):
        raise MatlabError("ga: the OutputFcn and PlotFcn options are not supported.")
    box_lo, box_hi = _init_range(lo, hi, opts.get("InitialPopulationRange"), n)
    pop = box_lo + rng.random((npop, n)) * (box_hi - box_lo)
    if opts.has("InitialPopulationMatrix"):
        init = np.real(fnum(opts.get("InitialPopulationMatrix"))).astype(float)
        if init.ndim != 2 or init.shape[1] != n:
            raise MatlabError(f"ga: InitialPopulationMatrix must have {n} columns.")
        k = min(init.shape[0], npop)
        pop[:k] = init[:k]
    for i in range(npop):
        pop[i] = cons.repair(pop[i], intmask if intmask.any() else None)
    if intmask.any():
        pop[:, intmask] = np.clip(np.round(pop[:, intmask]), np.ceil(lo[intmask]), np.floor(hi[intmask]))

    def evaluate(P):
        f = fit.many(P) if P.shape[0] else np.zeros(0)
        v = np.array([cons.violation(x) for x in P]) if (cons.any or intmask.any()) else np.zeros(P.shape[0])
        v = np.where(v <= tol_con, 0.0, v)
        return f, v

    def ranking(f, v):
        return np.lexsort((np.where(v > 0, 0.0, f), v))    # feasible first by f, then by violation

    f, v = evaluate(pop)
    order = ranking(f, v)
    pop, f, v = pop[order], f[order], v[order]
    span = np.where(np.isfinite(box_hi - box_lo), box_hi - box_lo, 20.0)
    sigma = {"scale": 0.25 * span * (mut_args[0] if mut_args else 1.0), "rate": mut_args[0] if mut_args else 0.01}
    shrink = mut_args[1] if len(mut_args) > 1 else 1.0
    n_kids = npop - n_elite
    n_cross = int(round(frac * n_kids))
    n_mut = n_kids - n_cross
    header = "Generation  Func-count        Best f(x)        Mean f(x)   Stall"
    history = [f[0] if v[0] == 0 else np.inf]
    t0 = time.time()
    flag, msg = 0, "ga stopped because the maximum number of generations was exceeded."
    gen = 0
    stall = 0
    for gen in range(1, max_gen + 1):
        kids = _ga_children(rng, pop, None, n_elite, n_cross, n_mut, box_lo, box_hi, sigma, intmask, cons,
                            crossover, cross_args, mutation, selection, sel_args)
        fk, vk = evaluate(kids)
        prev_best = (v[0], f[0])
        # step-size control for the mutation: success of the mutated children against the old best
        if n_mut and mutation != "mutationuniform":
            mut_f, mut_v = fk[n_cross:], vk[n_cross:]
            better = np.sum((mut_v < prev_best[0]) | ((mut_v == prev_best[0]) & (mut_f < prev_best[1])))
            if mutation == "mutationgaussian" and mut_args:
                sigma["scale"] = sigma["scale"] * max(1.0 - shrink * gen / max_gen, 0.0)
            else:
                sigma["scale"] = sigma["scale"] * (1.3 if better / n_mut > 0.2 else 0.85)
                sigma["scale"] = np.maximum(sigma["scale"], 1e-12 * np.maximum(span, 1.0))
        pop = np.vstack([pop[:n_elite], kids])
        f = np.concatenate([f[:n_elite], fk])
        v = np.concatenate([v[:n_elite], vk])
        order = ranking(f, v)
        pop, f, v = pop[order], f[order], v[order]
        best = f[0] if v[0] == 0 else np.inf
        history.append(best)
        improved = best < history[-2] - 1e-15 * max(1.0, abs(history[-2])) if np.isfinite(history[-2]) else \
            np.isfinite(best)
        stall = 0 if improved else stall + 1
        rep.iter_line(header, f"{gen:6d} {fit.count:12d} {f[0]:16.6g} {float(np.mean(f)):16.6g} {stall:6d}")
        if v[0] == 0 and f[0] <= limit:
            flag, msg = 5, "ga stopped because the fitness limit was reached."
            break
        if time.time() - t0 > max_time:
            flag, msg = -5, "ga stopped because the time limit was exceeded."
            break
        if gen >= stall_gen and np.isfinite(history[-1]):
            window = np.array(history[-stall_gen - 1:])
            if np.all(np.isfinite(window)):
                avg = float(np.mean(np.abs(np.diff(window)))) / max(1.0, abs(window[-1]))
                if avg <= tol_fun:
                    flag, msg = 1, ("ga stopped because the average change in the fitness value is less "
                                    "than options.FunctionTolerance.")
                    break
    x, fval = pop[0].copy(), float(f[0])
    hybrid_flag = empty_double()
    if opts.has("HybridFcn") and not intmask.any():
        res = _hybrid(interp, opts.get("HybridFcn"), fun, x, cons, "ga")
        if res is not None and (res[1] < fval or v[0] > 0):
            x, fval = res[0], res[1]
        if res is not None:
            hybrid_flag = np.array([[res[2]]])
    elif opts.has("HybridFcn") and intmask.any():
        raise MatlabError("ga: HybridFcn cannot be used with integer variables.")
    maxcon = cons.max_violation(x)
    if (cons.any and maxcon > tol_con) or v[0] > 0 and flag != 5:
        flag, msg = -2, "ga stopped: no feasible point was found."
    rep.finish(flag, msg)
    out = output_struct(problemtype=_problem_type(cons, bool(intmask.any())), rngstate=_rng_state(interp),
                        generations=gen, funccount=fit.count, message=msg, maxconstraint=maxcon,
                        hybridflag=hybrid_flag)
    return outs(nargout, (x.reshape(1, -1), fval, float(flag), out, pop, f.reshape(-1, 1)))


# -- multi-objective genetic algorithm ---------------------------------------------------------------------

def _nondominated_sort(F, V):
    """Fronts (lists of indices) with constraint domination."""
    npop = F.shape[0]
    le = np.all(F[:, None, :] <= F[None, :, :], axis=2)
    lt = np.any(F[:, None, :] < F[None, :, :], axis=2)
    same = V[:, None] == V[None, :]
    dom = (V[:, None] < V[None, :]) | (same & le & lt)       # dom[i, j]: i dominates j
    count = dom.sum(axis=0)
    fronts = []
    remaining = np.ones(npop, dtype=bool)
    while remaining.any():
        front = np.nonzero(remaining & (count == 0))[0]
        if front.size == 0:      # numerical ties: put what is left in one front
            front = np.nonzero(remaining)[0]
        fronts.append(front.tolist())
        remaining[front] = False
        count = count - dom[front].sum(axis=0)
    return fronts


def _crowding(F):
    k, m = F.shape
    d = np.zeros(k)
    for j in range(m):
        order = np.argsort(F[:, j], kind="stable")
        span = F[order[-1], j] - F[order[0], j]
        d[order[0]] = d[order[-1]] = np.inf
        if span > 0 and k > 2:
            d[order[1:-1]] += (F[order[2:], j] - F[order[:-2], j]) / span
    return d


@builtin("gamultiobj")
def _gamultiobj(interp, args, nargout):
    args = om.problem_args(args, ("fitnessfcn", "nvars", "Aineq", "bineq", "Aeq", "beq", "lb", "ub", "nonlcon",
                                  "options"), "gamultiobj")
    nargs(args, 2, 11, "gamultiobj")
    fun, n, A, b, Aeq, beq, lo, hi, nonlcon = _ga_setup(interp, args, "gamultiobj")
    raw_opts = next((a for a in args[8:] if isinstance(a, (Struct, om.OptimOptions))), None)
    opts = Opt(raw_opts, "gamultiobj", _GA_IMPL)
    if opts.has("OutputFcn") or opts.has("PlotFcn"):
        raise MatlabError("gamultiobj: the OutputFcn and PlotFcn options are not supported.")
    rng = _rng(interp)
    tol_con = opts.num("ConstraintTolerance", 1e-3)
    cons = _Constraints(interp, n, A, b, Aeq, beq, lo, hi, nonlcon, "gamultiobj", tol_con)
    fit = _Fitness(interp, fun, "gamultiobj", opts.flag("UseVectorized"), nobj=0)
    npop = int(opts.num("PopulationSize", 50 if n <= 5 else 200))
    max_gen = int(opts.num("MaxGenerations", 200 * n))
    stall_gen = int(opts.num("MaxStallGenerations", 100))
    tol_fun = opts.num("FunctionTolerance", 1e-4)
    frac = opts.num("CrossoverFraction", 0.8)
    pareto_frac = opts.num("ParetoFraction", 0.35)
    max_time = opts.num("MaxTime", np.inf)
    rep = Report(interp, opts)
    box_lo, box_hi = _init_range(lo, hi, opts.get("InitialPopulationRange"), n)
    pop = box_lo + rng.random((npop, n)) * (box_hi - box_lo)
    if opts.has("InitialPopulationMatrix"):
        init = np.real(fnum(opts.get("InitialPopulationMatrix"))).astype(float)
        if init.ndim != 2 or init.shape[1] != n:
            raise MatlabError(f"gamultiobj: InitialPopulationMatrix must have {n} columns.")
        pop[:min(init.shape[0], npop)] = init[:npop]
    for i in range(npop):
        pop[i] = cons.repair(pop[i])
    nomask = np.zeros(n, dtype=bool)

    def evaluate(P):
        F = fit.many(P)
        V = np.array([cons.violation(x) for x in P]) if cons.any else np.zeros(P.shape[0])
        return F, np.where(V <= tol_con, 0.0, V)

    F, V = evaluate(pop)
    span = np.where(np.isfinite(box_hi - box_lo), box_hi - box_lo, 20.0)
    t0 = time.time()
    flag, msg = 0, "gamultiobj stopped because the maximum number of generations was exceeded."
    spreads: list = []
    gen = 0
    header = "Generation   Func-count   Pareto size"
    for gen in range(1, max_gen + 1):
        fronts = _nondominated_sort(F, V)
        rank = np.empty(npop, dtype=int)
        crowd = np.empty(npop)
        for r, fr in enumerate(fronts):
            rank[fr] = r
            crowd[fr] = _crowding(F[fr])
        key = np.lexsort((-crowd, rank))
        ranked = pop[key]
        scale = 0.1 * span * max(1.0 - gen / max_gen, 0.02)
        n_cross = int(round(frac * npop))
        kids = _ga_children(rng, ranked, None, 0, n_cross, npop - n_cross, box_lo, box_hi,
                            {"scale": scale, "rate": 0.01}, nomask, cons, "crossoverintermediate", [],
                            "mutationgaussian", "selectiontournament", [2])
        Fk, Vk = evaluate(kids)
        allp, allF, allV = np.vstack([pop, kids]), np.vstack([F, Fk]), np.concatenate([V, Vk])
        fronts = _nondominated_sort(allF, allV)
        keep: list = []
        for fr in fronts:
            if len(keep) + len(fr) <= npop:
                keep += fr
            else:
                d = _crowding(allF[fr])
                order = np.argsort(-d, kind="stable")
                keep += [fr[i] for i in order[:npop - len(keep)]]
                break
        pop, F, V = allp[keep], allF[keep], allV[keep]
        first = _nondominated_sort(F, V)[0]
        d = _crowding(F[first])
        fin = d[np.isfinite(d)]
        spreads.append(float(np.std(fin) / (np.mean(fin) + 1e-300)) if fin.size else 0.0)
        rep.iter_line(header, f"{gen:6d} {fit.count:12d} {len(first):12d}")
        if time.time() - t0 > max_time:
            flag, msg = -5, "gamultiobj stopped because the time limit was exceeded."
            break
        if gen > stall_gen:
            w = np.array(spreads[-stall_gen - 1:])
            if float(np.mean(np.abs(np.diff(w)))) <= tol_fun * max(1.0, float(np.mean(w))):
                flag, msg = 1, ("gamultiobj stopped because the average change in the spread of Pareto "
                                "solutions is less than options.FunctionTolerance.")
                break
    fronts = _nondominated_sort(F, V)
    first = [i for i in fronts[0] if V[i] == 0] or fronts[0]
    limit = max(1, int(np.ceil(pareto_frac * npop)))
    if len(first) > limit:
        d = _crowding(F[first])
        first = [first[i] for i in np.argsort(-d, kind="stable")[:limit]]
    order = np.argsort(F[first][:, 0], kind="stable")
    X, FV = pop[first][order], F[first][order]
    if np.all(V[first] > 0):
        flag, msg = -2, "gamultiobj stopped: no feasible point was found."
    rep.finish(flag, msg)
    out = output_struct(problemtype=_problem_type(cons, False), rngstate=_rng_state(interp), generations=gen,
                        funccount=fit.count, message=msg, maxconstraint=max(cons.max_violation(x) for x in X),
                        averagedistance=float(np.mean(np.linalg.norm(X - X.mean(axis=0), axis=1))),
                        spread=spreads[-1] if spreads else 0.0)
    return outs(nargout, (X, FV, float(flag), out, pop, F))


# -- particle swarm -------------------------------------------------------------------------------------------

@builtin("particleswarm")
def _particleswarm(interp, args, nargout):
    args = om.problem_args(args, ("objective", "nvars", "lb", "ub", "options"), "particleswarm")
    nargs(args, 2, 5, "particleswarm")
    nv = fnum(args[1])
    if nv.size != 1 or nv.flat[0] < 1 or nv.flat[0] != int(nv.flat[0]):
        raise MatlabError("particleswarm: the number of variables must be a positive integer.")
    n = int(nv.flat[0])
    lo, hi = om.bounds(om._arg(args, 2), om._arg(args, 3), n, "particleswarm")
    opts = Opt(om._options_at(args, 4), "particleswarm",
               ("Display", "SwarmSize", "MaxIterations", "MaxStallIterations", "FunctionTolerance",
                "InertiaRange", "SelfAdjustmentWeight", "SocialAdjustmentWeight", "MinNeighborsFraction",
                "ObjectiveLimit", "MaxTime", "MaxStallTime", "InitialSwarmMatrix", "InitialSwarmSpan",
                "UseVectorized", "UseParallel", "HybridFcn", "OutputFcn", "PlotFcn"))
    fit = _Fitness(interp, args[0], "particleswarm", opts.flag("UseVectorized"))
    rng = _rng(interp)
    ns = int(opts.num("SwarmSize", min(100, 10 * n)))
    if ns < 2:
        raise MatlabError("particleswarm: SwarmSize must be at least 2.")
    max_it = int(opts.num("MaxIterations", 200 * n))
    stall_it = int(opts.num("MaxStallIterations", 20))
    tol_fun = opts.num("FunctionTolerance", 1e-6)
    limit = opts.num("ObjectiveLimit", -np.inf)
    max_time = opts.num("MaxTime", np.inf)
    y1, y2 = opts.num("SelfAdjustmentWeight", 1.49), opts.num("SocialAdjustmentWeight", 1.49)
    w_rng = np.real(fnum(opts.get("InertiaRange"))).flatten() if opts.has("InertiaRange") else np.array([0.1, 1.1])
    if w_rng.size != 2:
        raise MatlabError("particleswarm: InertiaRange must be a two-element vector.")
    w_lo, w_hi = float(min(w_rng)), float(max(w_rng))
    min_frac = opts.num("MinNeighborsFraction", 0.25)
    span_opt = np.real(fnum(opts.get("InitialSwarmSpan"))).flatten() if opts.has("InitialSwarmSpan") \
        else np.array([2000.0])
    span = np.broadcast_to(span_opt, (n,)).astype(float)
    rep = Report(interp, opts)
    hook = OutputHook(interp, opts, (1, n), "particleswarm")
    box_lo = np.where(np.isfinite(lo), lo, np.where(np.isfinite(hi), hi - span, -span / 2))
    box_hi = np.where(np.isfinite(hi), hi, np.where(np.isfinite(lo), lo + span, span / 2))
    X = box_lo + rng.random((ns, n)) * (box_hi - box_lo)
    if opts.has("InitialSwarmMatrix"):
        init = np.real(fnum(opts.get("InitialSwarmMatrix"))).astype(float)
        if init.ndim != 2 or init.shape[1] != n:
            raise MatlabError(f"particleswarm: InitialSwarmMatrix must have {n} columns.")
        X[:min(init.shape[0], ns)] = init[:ns]
    X = np.clip(X, lo, hi)
    vmax = box_hi - box_lo
    Vel = -vmax + 2 * vmax * rng.random((ns, n))
    fx = fit.many(X)
    P, fp = X.copy(), fx.copy()
    best = int(np.argmin(fp))
    min_nb = max(2, int(np.floor(ns * min_frac)))
    nb = min_nb
    W = w_hi
    c = 0
    history = [float(fp[best])]
    header = "Iteration     f-count            Best f(x)            Mean f(x)    Stall"
    flag, msg = 0, "Optimization ended: the number of iterations exceeded MaxIterations."
    t0 = time.time()
    it = 0
    stall = 0
    hook.call(P[best], "init", iteration=0, funccount=fit.count, bestfval=fp[best], bestx=P[best])
    for it in range(1, max_it + 1):
        for i in range(ns):
            others = rng.permutation(ns)[:nb]
            g = others[np.argmin(fp[others])]
            if fp[i] < fp[g]:
                g = i
            u1, u2 = rng.random(n), rng.random(n)
            Vel[i] = W * Vel[i] + y1 * u1 * (P[i] - X[i]) + y2 * u2 * (P[g] - X[i])
            xn = X[i] + Vel[i]
            out_lo, out_hi = xn < lo, xn > hi
            xn = np.where(out_lo, lo, np.where(out_hi, hi, xn))
            Vel[i][out_lo | out_hi] = 0.0
            X[i] = xn
        fx = fit.many(X)
        better = fx < fp
        P[better], fp[better] = X[better], fx[better]
        new_best = int(np.argmin(fp))
        improved = fp[new_best] < history[-1]
        best = new_best
        history.append(float(fp[best]))
        if improved:
            c = max(0, c - 1)
            nb = min_nb
            if c < 2:
                W = 2 * W
            elif c > 5:
                W = W / 2
            W = min(max(W, w_lo), w_hi)
            stall = 0
        else:
            c += 1
            nb = min(nb + min_nb, ns - 1)
            stall += 1
        rep.iter_line(header, f"{it:6d} {fit.count:12d} {fp[best]:20.6g} {float(np.mean(fx)):20.6g} {stall:6d}")
        if hook.call(P[best], "iter", iteration=it, funccount=fit.count, bestfval=fp[best], bestx=P[best]):
            flag, msg = -1, "Optimization stopped by the output function."
            break
        if fp[best] <= limit:
            flag, msg = -3, "Optimization ended: the best objective value is at or below ObjectiveLimit."
            break
        if time.time() - t0 > max_time:
            flag, msg = -5, "Optimization ended: the time limit was exceeded."
            break
        if it > stall_it:
            old, new = history[-stall_it - 1], history[-1]
            if abs(old - new) / max(1.0, abs(new)) < tol_fun:
                flag, msg = 1, ("Optimization ended: relative change in the objective value over the last "
                                "OPTIONS.MaxStallIterations iterations is less than OPTIONS.FunctionTolerance.")
                break
    x, fval = P[best].copy(), float(fp[best])
    if opts.has("HybridFcn"):
        e = np.zeros((0, n))
        cons = _Constraints(interp, n, e, np.zeros(0), e, np.zeros(0), lo, hi, None, "particleswarm")
        res = _hybrid(interp, opts.get("HybridFcn"), args[0], x, cons, "particleswarm")
        if res is not None and res[1] < fval:
            x, fval = res[0], res[1]
    hook.call(x, "done", iteration=it, funccount=fit.count, bestfval=fval, bestx=x)
    rep.finish(flag, msg)
    out = output_struct(rngstate=_rng_state(interp), iterations=it, funccount=fit.count, message=msg,
                        hybridflag=empty_double())
    points = Struct.scalar({"X": X, "Fval": fx.reshape(-1, 1)})
    return outs(nargout, (x.reshape(1, -1), fval, float(flag), out, points))


# -- pattern search -----------------------------------------------------------------------------------------------

def _poll_directions(n, method, rng, scale, tangent=None):
    if tangent is not None:
        basis = tangent
    elif method.startswith("MADS"):
        Q, _ = np.linalg.qr(rng.standard_normal((n, n)))
        basis = Q * scale[:, None]
    else:
        basis = np.diag(scale)
    if method.endswith("Np1"):
        return np.vstack([basis.T, -basis.sum(axis=1).reshape(1, -1)])
    return np.vstack([basis.T, -basis.T])


def _pattern_core(fmerit, x0, feasible, lo, hi, A, b, Aeq, opts: Opt, rng, counts, max_iter, max_fun,
                  rep: Report | None, hook=None, mesh0=None):
    """Generalised pattern search for a merit function with bound and linear constraints.
    Returns (x, f, flag, mesh, iterations)."""
    n = x0.size
    mesh = opts.num("InitialMeshSize", 1.0) if mesh0 is None else mesh0
    max_mesh = opts.num("MaxMeshSize", np.inf)
    expand = opts.num("MeshExpansionFactor", 2.0)
    contract = opts.num("MeshContractionFactor", 0.5)
    tol_mesh = opts.num("MeshTolerance", 1e-6)
    tol_x = opts.num("StepTolerance", 1e-6)
    tol_f = opts.num("FunctionTolerance", 1e-6)
    tol_bind = opts.num("TolBind", 1e-3)
    method = opts.word("PollMethod", "GPSPositiveBasis2N",
                       ("GPSPositiveBasis2N", "GPSPositiveBasisNp1", "GSSPositiveBasis2N", "GSSPositiveBasisNp1",
                        "MADSPositiveBasis2N", "MADSPositiveBasisNp1"))
    complete = opts.flag("UseCompletePoll")
    order_alg = opts.word("PollOrderAlgorithm", "Consecutive", ("Consecutive", "Random", "Success"))
    scale_on = opts.flag("ScaleMesh", True)
    accelerate = opts.flag("AccelerateMesh")
    scale = np.ones(n)
    if scale_on:
        both = np.isfinite(lo) & np.isfinite(hi)
        ref = np.where(both, (hi - lo) / 2.0, np.maximum(np.abs(x0), 1.0))
        scale = np.where(ref > 0, 2.0 ** np.round(np.log2(np.where(ref > 0, ref, 1.0))), 1.0)
        scale = np.where(both, np.minimum(scale, np.maximum(hi - lo, 1e-12)), scale)
    null_basis = None
    if Aeq.shape[0]:
        from scipy.linalg import null_space
        null_basis = null_space(Aeq)
        if null_basis.shape[1] == 0:
            return x0, fmerit(x0), 1, mesh, 0
    x, f = x0.copy(), fmerit(x0)
    header = "Iter     Func-count       f(x)      MeshSize     Method"
    last_success = 0
    it = 0
    flag = 0
    if rep is not None:
        rep.iter_line(header, f"{0:4d} {counts[0]:12d} {f:14.6g} {mesh:12.4g}")
    while True:
        if mesh < tol_mesh:
            flag = 1
            break
        if it >= max_iter or counts[0] >= max_fun:
            flag = 0
            break
        tangent = None
        if null_basis is not None:
            tangent = null_basis
        elif A.shape[0]:
            act = (b - A @ x) <= tol_bind * mesh * np.maximum(1.0, np.abs(b))
            if act.any():
                from scipy.linalg import null_space
                T = null_space(A[act])
                Nrm = A[act].T / np.maximum(np.linalg.norm(A[act], axis=1), 1e-300)
                tangent = np.hstack([T, -Nrm]) if T.shape[1] else -Nrm
                extra = np.eye(n) * scale
                tangent = np.hstack([tangent, extra])
        D = _poll_directions(n, method, rng, scale, tangent)
        idx = np.arange(D.shape[0])
        if order_alg == "Random":
            idx = rng.permutation(D.shape[0])
        elif order_alg == "Success" and last_success < D.shape[0]:
            idx = np.concatenate([[last_success], np.delete(idx, last_success)])
        best_x, best_f, best_k = None, f, None
        for k in idx:
            xt = x + mesh * D[k]
            if np.any(xt < lo - 1e-12) or np.any(xt > hi + 1e-12):
                continue
            xt = np.clip(xt, lo, hi)
            if not feasible(xt):
                continue
            ft = fmerit(xt)
            if ft < best_f:
                best_x, best_f, best_k = xt, ft, k
                if not complete:
                    break
            if counts[0] >= max_fun:
                break
        it += 1
        if best_x is not None:
            df, dx = f - best_f, float(np.linalg.norm(best_x - x))
            x, f = best_x, best_f
            last_success = int(best_k)
            kind = "Successful Poll"
            old_mesh = mesh
            mesh = min(mesh * expand, max_mesh)
            if old_mesh < tol_x and (dx < tol_x or df < tol_f):
                flag = 2 if dx < tol_x else 3
                if rep is not None:
                    rep.iter_line(header, f"{it:4d} {counts[0]:12d} {f:14.6g} {mesh:12.4g}     {kind}")
                break
        else:
            kind = "Refine Mesh"
            mesh = mesh * contract * (0.5 if accelerate and mesh < 1e-3 else 1.0)
        if rep is not None:
            rep.iter_line(header, f"{it:4d} {counts[0]:12d} {f:14.6g} {mesh:12.4g}     {kind}")
        if hook is not None and hook(x, f, it, mesh):
            flag = -1
            break
    return x, f, flag, mesh, it


@builtin("patternsearch")
def _patternsearch(interp, args, nargout):
    args = om.problem_args(args, ("objective", "x0", "Aineq", "bineq", "Aeq", "beq", "lb", "ub", "nonlcon",
                                  "options"), "patternsearch")
    nargs(args, 2, 10, "patternsearch")
    fun = args[0]
    if not om.is_fn(fun):
        raise MatlabError("patternsearch: the objective must be a function handle.")
    x0a = np.real(fnum(args[1])).astype(float)
    shape, n = x0a.shape, x0a.size
    if n == 0:
        raise MatlabError("patternsearch: the starting point x0 must not be empty.")
    A, b = om.linear(om._arg(args, 2), om._arg(args, 3), n, "patternsearch", "A")
    Aeq, beq = om.linear(om._arg(args, 4), om._arg(args, 5), n, "patternsearch", "Aeq")
    lo, hi = om.bounds(om._arg(args, 6), om._arg(args, 7), n, "patternsearch")
    nonlcon = args[8] if len(args) > 8 and om.is_fn(args[8]) else None
    opts = Opt(om._options_at(args, 9), "patternsearch",
               ("Display", "Algorithm", "MaxIterations", "MaxFunctionEvaluations", "MeshTolerance",
                "StepTolerance", "FunctionTolerance", "ConstraintTolerance", "InitialMeshSize", "MaxMeshSize",
                "MeshExpansionFactor", "MeshContractionFactor", "PollMethod", "UseCompletePoll",
                "PollOrderAlgorithm", "ScaleMesh", "AccelerateMesh", "Cache", "MaxTime", "UseVectorized",
                "UseParallel", "OutputFcn", "PlotFcn", "TolBind"))
    if opts.word("Algorithm", "classic") != "classic":
        raise MatlabError("patternsearch: only the 'classic' algorithm is supported.")
    if opts.flag("UseVectorized"):
        raise MatlabError("patternsearch: the UseVectorized option is not supported.")
    rng = _rng(interp)
    tol_con = opts.num("ConstraintTolerance", 1e-6)
    rep = Report(interp, opts)
    hook = OutputHook(interp, opts, shape, "patternsearch")
    use_cache = opts.word("Cache", "off", ("on", "off")) == "on"
    cache: dict = {}
    counts = [0]

    def f(x):
        key = x.tobytes() if use_cache else None
        if use_cache and key in cache:
            return cache[key]
        res = interp.call_function_value(fun, [x.reshape(shape, order="F")], 1)
        counts[0] += 1
        v = fnum(res[0])
        if v.size != 1:
            raise MatlabError("patternsearch: the objective function must return a scalar.")
        val = float(np.real(v.flat[0]))
        if use_cache:
            cache[key] = val
        return val

    cons = _Constraints(interp, n, A, b, Aeq, beq, lo, hi, nonlcon, "patternsearch", tol_con)
    x0 = np.clip(x0a.flatten(order="F"), lo, hi)
    # start from a point that satisfies the linear constraints
    lin_viol = max([0.0] + ([float(np.max(A @ x0 - b))] if A.shape[0] else [])
                   + ([float(np.max(np.abs(Aeq @ x0 - beq)))] if Aeq.shape[0] else []))
    if lin_viol > tol_con:
        from ._optim_local import solve_qp
        xq, _, qflag, _ = solve_qp(np.eye(n), -x0, A, b, Aeq, beq, lo, hi)
        if qflag != 1:
            msg = "patternsearch stopped: the linear constraints are inconsistent."
            rep.finish(-2, msg)
            return outs(nargout, (x0.reshape(shape, order="F"), f(x0), -2.0,
                                  output_struct(iterations=0, funccount=counts[0], message=msg)))
        x0 = xq

    def lin_feasible(x):
        if A.shape[0] and np.any(A @ x - b > tol_con * np.maximum(1.0, np.abs(b))):
            return False
        if Aeq.shape[0] and np.any(np.abs(Aeq @ x - beq) > max(tol_con, 1e-9) * np.maximum(1.0, np.abs(beq))):
            return False
        return True

    max_iter = int(opts.num("MaxIterations", 100 * n))
    max_fun = int(opts.num("MaxFunctionEvaluations", 2000 * n))

    def hook_call(x, fx, it, mesh):
        return hook.call(x, "iter", iteration=it, funccount=counts[0], fval=fx, meshsize=mesh)

    hook.call(x0, "init", iteration=0, funccount=0, fval=f(x0), meshsize=opts.num("InitialMeshSize", 1.0))
    if nonlcon is None:
        x, fx, flag, mesh, iters = _pattern_core(f, x0, lin_feasible, lo, hi, A, b, Aeq, opts, rng, counts,
                                                 max_iter, max_fun, rep, hook_call if hook else None)
    else:
        # augmented Lagrangian outer loop for the nonlinear constraints
        c0, e0 = cons.nonlinear(x0)
        lam, mu, rho = np.zeros(c0.size), np.zeros(e0.size), 10.0
        prev_viol = np.inf
        x, mesh, iters, flag = x0, None, 0, 0
        fx = f(x)
        for outer in range(30):
            def merit(z, lam=lam, mu=mu, rho=rho):
                c, ce = cons.nonlinear(z)
                return f(z) + float(np.sum(np.maximum(0.0, lam + rho * c) ** 2 - lam ** 2) / (2 * rho)
                                    + mu @ ce + 0.5 * rho * (ce @ ce))
            inner_opts = Opt(None, "patternsearch")
            inner_opts.user = dict(opts.user)
            x_new, _, iflag, mesh, k = _pattern_core(merit, x, lin_feasible, lo, hi, A, b, Aeq, inner_opts, rng,
                                                     counts, max(max_iter, 50 * n), max_fun, None)
            iters += k
            c, ce = cons.nonlinear(x_new)
            viol = max(float(np.max(c, initial=0.0)), float(np.max(np.abs(ce), initial=0.0)))
            step = float(np.linalg.norm(x_new - x))
            x, fx = x_new, f(x_new)
            rep.iter_line("Iter     Func-count        f(x)    Constraint", f"{outer + 1:4d} {counts[0]:12d} "
                          f"{fx:14.6g} {viol:12.4g}")
            lam = np.maximum(0.0, lam + rho * c)
            mu = mu + rho * ce
            if viol <= tol_con and (step <= 1e-6 * (1 + np.linalg.norm(x)) or outer >= 2):
                flag = 1
                if step <= 1e-6 * (1 + np.linalg.norm(x)):
                    break
            # raise the penalty only when the multiplier update alone is not reducing the violation
            if viol > tol_con and viol > 0.25 * prev_viol:
                rho = min(rho * 4.0, 1e8)
            prev_viol = viol
            if counts[0] >= max_fun:
                flag = 0 if viol > tol_con else flag
                break
            if iflag == -1:
                flag = -1
                break
        if cons.max_violation(x) > tol_con * 10:
            flag = -2
    msg = {1: "Optimization terminated: mesh size less than options.MeshTolerance.",
           2: "Optimization terminated: change in X less than options.StepTolerance.",
           3: "Optimization terminated: change in the function value less than options.FunctionTolerance.",
           0: "Maximum number of iterations or function evaluations exceeded.",
           -1: "Optimization stopped by the output function.",
           -2: "No feasible point was found."}[flag]
    hook.call(x, "done", iteration=iters, funccount=counts[0], fval=fx, meshsize=mesh or 0.0)
    rep.finish(flag, msg)
    out = output_struct(function=fun, problemtype=_problem_type(cons, False),
                        pollmethod=opts.word("PollMethod", "gpspositivebasis2n").lower(),
                        maxconstraint=cons.max_violation(x) if cons.any else 0.0, searchmethod=empty_double(),
                        iterations=iters, funccount=counts[0], meshsize=mesh or 0.0, rngstate=_rng_state(interp),
                        message=msg)
    return outs(nargout, (x.reshape(shape, order="F"), fx, float(flag), out))


# -- simulated annealing ---------------------------------------------------------------------------------------------

@builtin("simulannealbnd")
def _simulannealbnd(interp, args, nargout):
    args = om.problem_args(args, ("objective", "x0", "lb", "ub", "options"), "simulannealbnd")
    nargs(args, 2, 5, "simulannealbnd")
    fun = args[0]
    if not om.is_fn(fun):
        raise MatlabError("simulannealbnd: the objective must be a function handle.")
    x0a = np.real(fnum(args[1])).astype(float)
    shape, n = x0a.shape, x0a.size
    if n == 0:
        raise MatlabError("simulannealbnd: the starting point x0 must not be empty.")
    lo, hi = om.bounds(om._arg(args, 2), om._arg(args, 3), n, "simulannealbnd")
    opts = Opt(om._options_at(args, 4), "simulannealbnd",
               ("Display", "InitialTemperature", "ReannealInterval", "MaxIterations", "MaxFunctionEvaluations",
                "MaxStallIterations", "FunctionTolerance", "ObjectiveLimit", "MaxTime", "TemperatureFcn",
                "AnnealingFcn", "AcceptanceFcn", "HybridFcn", "OutputFcn", "PlotFcn"))
    rng = _rng(interp)
    t_init = np.broadcast_to(np.real(fnum(opts.get("InitialTemperature"))).flatten()
                             if opts.has("InitialTemperature") else np.array([100.0]), (n,)).astype(float)
    reanneal = int(opts.num("ReannealInterval", 100))
    max_it = opts.num("MaxIterations", np.inf)
    max_fun = opts.num("MaxFunctionEvaluations", 3000 * n)
    stall_it = int(opts.num("MaxStallIterations", 500 * n))
    tol_fun = opts.num("FunctionTolerance", 1e-6)
    limit = opts.num("ObjectiveLimit", -np.inf)
    max_time = opts.num("MaxTime", np.inf)
    temp_fcn, _ = _named_option(opts, "TemperatureFcn", ("temperatureexp", "temperaturefast", "temperatureboltz"),
                                "temperatureexp", "simulannealbnd")
    anneal_fcn, _ = _named_option(opts, "AnnealingFcn", ("annealingfast", "annealingboltz"), "annealingfast",
                                  "simulannealbnd")
    _named_option(opts, "AcceptanceFcn", ("acceptancesa",), "acceptancesa", "simulannealbnd")
    rep = Report(interp, opts)
    hook = OutputHook(interp, opts, shape, "simulannealbnd")
    count = [0]

    def f(x):
        res = interp.call_function_value(fun, [x.reshape(shape, order="F")], 1)
        count[0] += 1
        v = fnum(res[0])
        if v.size != 1:
            raise MatlabError("simulannealbnd: the objective function must return a scalar.")
        return float(np.real(v.flat[0]))

    x = np.clip(x0a.flatten(order="F"), lo, hi)
    fx = f(x)
    best_x, best_f = x.copy(), fx
    k = np.ones(n)            # annealing parameter per dimension
    T = t_init.copy()
    accepted = 0
    history = [best_f]
    header = "Iter   Func-count      Best f(x)   Current f(x)   Mean temperature"
    flag, msg = 0, "Maximum number of function evaluations exceeded."
    t0 = time.time()
    it = 0
    hook.call(x, "init", iteration=0, funccount=count[0], fval=fx, bestfval=best_f, bestx=best_x)
    while True:
        it += 1
        if it > max_it:
            flag, msg = 0, "Maximum number of iterations exceeded."
            break
        if count[0] >= max_fun:
            break
        d = rng.standard_normal(n)
        d = d / max(np.linalg.norm(d), 1e-300)
        step = T * d if anneal_fcn == "annealingfast" else np.sqrt(T) * d
        xn = x + step
        bad = (xn < lo) | (xn > hi)
        if bad.any():
            # move part of the way towards the bound that was crossed
            target = np.where(xn < lo, lo, hi)
            xn = np.where(bad, x + rng.random(n) * (target - x), xn)
        fn = f(xn)
        delta = fn - fx
        tmax = float(np.max(T))
        accept = delta < 0
        if not accept and tmax > 0:
            with np.errstate(over="ignore"):
                accept = rng.random() < 1.0 / (1.0 + np.exp(min(delta / tmax, 700.0)))
        if accept:
            x, fx = xn, fn
            accepted += 1
            if fx < best_f:
                best_x, best_f = x.copy(), fx
        history.append(best_f)
        k += 1
        if reanneal > 0 and accepted and accepted % reanneal == 0 and accept:
            # raise the temperature again where the objective is least sensitive
            h = 1e-6 * np.maximum(np.abs(x), 1.0)
            s = np.empty(n)
            for j in range(n):
                xp = x.copy()
                xp[j] = min(xp[j] + h[j], hi[j]) if xp[j] + h[j] <= hi[j] else xp[j] - h[j]
                s[j] = abs((f(xp) - fx) / h[j]) * (np.where(np.isfinite(hi[j] - lo[j]), hi[j] - lo[j], 1.0))
            s = np.where(s > 0, s, np.max(s) if np.max(s) > 0 else 1.0)
            with np.errstate(divide="ignore", invalid="ignore"):
                k = np.maximum(np.abs(np.log(np.maximum(t_init / np.maximum(T, 1e-300) * (np.max(s) / s), 1.0))), 1.0)
        if temp_fcn == "temperatureexp":
            T = t_init * 0.95 ** k
        elif temp_fcn == "temperaturefast":
            T = t_init / k
        else:
            T = t_init / np.log(k + 1.0)
        if it % 10 == 0:
            rep.iter_line(header, f"{it:5d} {count[0]:10d} {best_f:14.6g} {fx:14.6g} {float(np.mean(T)):14.6g}")
        if hook.call(x, "iter", iteration=it, funccount=count[0], fval=fx, bestfval=best_f, bestx=best_x):
            flag, msg = -1, "Optimization stopped by the output function."
            break
        if best_f <= limit:
            flag, msg = 5, "Optimization terminated: the best function value is at or below ObjectiveLimit."
            break
        if time.time() - t0 > max_time:
            flag, msg = -5, "Optimization terminated: the time limit was exceeded."
            break
        if it > stall_it:
            if abs(history[-stall_it - 1] - history[-1]) <= tol_fun * max(1.0, abs(history[-1])):
                flag, msg = 1, ("Optimization terminated: change in best function value less than "
                                "options.FunctionTolerance.")
                break
    x, fval = best_x, best_f
    if opts.has("HybridFcn"):
        e = np.zeros((0, n))
        cons = _Constraints(interp, n, e, np.zeros(0), e, np.zeros(0), lo, hi, None, "simulannealbnd")
        wrapped = fun
        res = _hybrid(interp, opts.get("HybridFcn"), wrapped, x.reshape(shape, order="F").reshape(-1), cons,
                      "simulannealbnd") if shape[0] == 1 else None
        if res is not None and res[1] < fval:
            x, fval = res[0], res[1]
    hook.call(x, "done", iteration=it, funccount=count[0], fval=fval, bestfval=fval, bestx=x)
    rep.finish(flag, msg)
    out = output_struct(iterations=it, funccount=count[0], message=msg, rngstate=_rng_state(interp),
                        problemtype="boundconstraints" if np.any(np.isfinite(lo) | np.isfinite(hi))
                        else "unconstrained", temperature=T.reshape(-1, 1), totaltime=time.time() - t0)
    return outs(nargout, (x.reshape(shape, order="F"), fval, float(flag), out))


# -- surrogate optimisation ------------------------------------------------------------------------------------------

@builtin("surrogateopt")
def _surrogateopt(interp, args, nargout):
    args = om.problem_args(args, ("objective", "lb", "ub", "intcon", "options"), "surrogateopt")
    nargs(args, 3, 9, "surrogateopt")
    lb, ub = om._arg(args, 1), om._arg(args, 2)
    if lb is None or ub is None:
        raise MatlabError("surrogateopt: finite lower and upper bounds are required.")
    lo, hi = lb.flatten(), ub.flatten()
    n = lo.size
    if hi.size != n or not (np.all(np.isfinite(lo)) and np.all(np.isfinite(hi))) or np.any(lo > hi):
        raise MatlabError("surrogateopt: finite lower and upper bounds of equal length are required.")
    intmask = np.zeros(n, dtype=bool)
    ic = om._arg(args, 3)
    if ic is not None:
        intmask[ic.flatten().astype(int) - 1] = True
    raw_opts = next((a for a in args[3:] if isinstance(a, (Struct, om.OptimOptions))), None)
    opts = Opt(raw_opts, "surrogateopt",
               ("Display", "MaxFunctionEvaluations", "MinSurrogatePoints", "MinSampleDistance", "ObjectiveLimit",
                "MaxTime", "InitialPoints", "UseVectorized", "UseParallel", "OutputFcn", "PlotFcn"))
    if opts.has("PlotFcn") or opts.has("OutputFcn"):
        raise MatlabError("surrogateopt: the OutputFcn and PlotFcn options are not supported.")
    fit = _Fitness(interp, args[0], "surrogateopt", opts.flag("UseVectorized"))
    rng = _rng(interp)
    max_fun = int(opts.num("MaxFunctionEvaluations", max(200, 50 * n)))
    n_init = int(opts.num("MinSurrogatePoints", max(20, 2 * n)))
    min_dist = opts.num("MinSampleDistance", 1e-6)
    limit = opts.num("ObjectiveLimit", -np.inf)
    max_time = opts.num("MaxTime", np.inf)
    rep = Report(interp, opts)
    span = np.where(hi > lo, hi - lo, 1.0)

    def snap(P):
        P = np.clip(P, lo, hi)
        if intmask.any():
            P = np.array(P, copy=True)
            P[..., intmask] = np.round(P[..., intmask])
        return P

    X = snap(lo + rng.random((n_init, n)) * (hi - lo))
    if opts.has("InitialPoints"):
        ip = opts.get("InitialPoints")
        init = np.real(fnum(ip.data.flat[0]["X"] if isinstance(ip, Struct) else ip)).astype(float).reshape(-1, n)
        X = np.vstack([snap(init), X])[:max(n_init, init.shape[0])]
    F = fit.many(X)
    t0 = time.time()
    flag, msg = 0, "surrogateopt stopped because it exceeded the function evaluation limit."
    cycle = [0.3, 0.5, 0.8, 0.95]
    k = 0
    sigma = 0.2
    while fit.count < max_fun:
        if float(np.min(F)) <= limit:
            flag, msg = 1, "surrogateopt stopped because the objective limit was reached."
            break
        if time.time() - t0 > max_time:
            flag, msg = -5, "surrogateopt stopped because the time limit was exceeded."
            break
        # cubic radial basis interpolant with a linear tail, in scaled coordinates
        Z = (X - lo) / span
        D = np.linalg.norm(Z[:, None, :] - Z[None, :, :], axis=2)
        m = Z.shape[0]
        Pm = np.hstack([np.ones((m, 1)), Z])
        K = np.block([[D ** 3, Pm], [Pm.T, np.zeros((n + 1, n + 1))]])
        coef = np.linalg.lstsq(K, np.concatenate([F, np.zeros(n + 1)]), rcond=None)[0]
        best = X[int(np.argmin(F))]
        ncand = min(1000, 500 * n)
        local = best + sigma * span * rng.standard_normal((ncand // 2, n))
        glob = lo + rng.random((ncand - ncand // 2, n)) * (hi - lo)
        C = snap(np.vstack([local, glob]))
        Zc = (C - lo) / span
        Dc = np.linalg.norm(Zc[:, None, :] - Z[None, :, :], axis=2)
        s = Dc ** 3 @ coef[:m] + np.hstack([np.ones((C.shape[0], 1)), Zc]) @ coef[m:]
        dmin = Dc.min(axis=1)
        ok = dmin > min_dist
        if not ok.any():
            sigma = min(sigma * 2, 1.0)
            X = np.vstack([X, snap(lo + rng.random((1, n)) * (hi - lo))])
            F = np.append(F, fit.many(X[-1:]))
            continue
        w = cycle[k % len(cycle)]
        k += 1
        sn = (s - s[ok].min()) / max(s[ok].max() - s[ok].min(), 1e-300)
        dn = (dmin[ok].max() - dmin) / max(dmin[ok].max() - dmin[ok].min(), 1e-300)
        merit = np.where(ok, w * sn + (1 - w) * dn, np.inf)
        xn = C[int(np.argmin(merit))]
        fn = fit.one(xn)
        if fn < float(np.min(F)):
            sigma = min(sigma * 1.5, 0.5)
        else:
            sigma = max(sigma * 0.8, 1e-4)
        X = np.vstack([X, xn])
        F = np.append(F, fn)
    i = int(np.argmin(F))
    rep.finish(flag, msg)
    out = output_struct(elapsedtime=time.time() - t0, funccount=fit.count, constrviolation=0.0, ineq=empty_double(),
                        rngstate=_rng_state(interp), message=msg)
    trials = Struct.scalar({"X": X, "Fval": F.reshape(-1, 1)})
    return outs(nargout, (X[i].reshape(1, -1), float(F[i]), float(flag), out, trials))


# -- problems and multi-start drivers ------------------------------------------------------------------------

_PROBLEM_FIELDS = {"objective": "objective", "x0": "x0", "aineq": "Aineq", "bineq": "bineq", "aeq": "Aeq",
                   "beq": "beq", "lb": "lb", "ub": "ub", "nonlcon": "nonlcon", "options": "options",
                   "xdata": "xdata", "ydata": "ydata"}


@builtin("createOptimProblem")
def _create_optim_problem(interp, args, nargout):
    nargs(args, 1, None, "createOptimProblem")
    solver = text(args[0])
    if solver not in ("fmincon", "fminunc", "lsqcurvefit", "lsqnonlin"):
        raise MatlabError("createOptimProblem: the solver must be 'fmincon', 'fminunc', 'lsqcurvefit' or "
                          "'lsqnonlin'.")
    rest = list(args[1:])
    if len(rest) % 2:
        raise MatlabError("createOptimProblem: the problem data must be given as name-value pairs.")
    fields = {"objective": empty_double(), "x0": empty_double(), "Aineq": empty_double(),
              "bineq": empty_double(), "Aeq": empty_double(), "beq": empty_double(), "lb": empty_double(),
              "ub": empty_double(), "nonlcon": empty_double(), "solver": char(solver),
              "options": om.OptimOptions(solver)}
    if solver in ("lsqcurvefit",):
        fields.update({"xdata": empty_double(), "ydata": empty_double()})
    for k in range(0, len(rest), 2):
        key = text(rest[k]).lower()
        if key not in _PROBLEM_FIELDS:
            raise MatlabError(f"createOptimProblem: unknown problem field '{text(rest[k])}'.")
        fields[_PROBLEM_FIELDS[key]] = rest[k + 1]
    return Struct.scalar(fields)


class _StartPoints(MObject):
    def __init__(self, kind, value):
        self.kind, self.value = kind, value

    def class_name(self):
        return "CustomStartPointSet" if self.kind == "custom" else "RandomStartPointSet"

    def get_field(self, interp, name):
        if self.kind == "custom" and name == "StartPoints":
            return self.value
        if self.kind == "custom" and name in ("NumStartPoints", "StartPointsDimension"):
            return np.array([[float(self.value.shape[0 if name == "NumStartPoints" else 1])]])
        if self.kind == "random" and name == "NumStartPoints":
            return np.array([[float(self.value["n"])]])
        if self.kind == "random" and name == "ArtificialBound":
            return np.array([[float(self.value["bound"])]])
        raise MatlabError(f"Unrecognized property '{name}' for class '{self.class_name()}'.")

    def has_method(self, name):
        return name == "list"

    def call_method(self, interp, name, args, nargout):
        if self.kind == "custom":
            return [self.value]
        if len(args) < 2 or not isinstance(args[1], Struct):
            raise MatlabError("list: a RandomStartPointSet needs the problem structure, list(rs, problem).")
        d = _problem_dict(args[1])
        return [_random_points(_rng(interp), d, self.value["n"], self.value["bound"])]


@builtin("CustomStartPointSet")
def _custom_start_points(interp, args, nargout):
    nargs(args, 1, 1, "CustomStartPointSet")
    M = np.real(fnum(args[0])).astype(float)
    return _StartPoints("custom", M.reshape(M.shape[0], -1))


@builtin("RandomStartPointSet")
def _random_start_points(interp, args, nargout):
    val = {"n": 10, "bound": 1000.0}
    if len(args) % 2:
        raise MatlabError("RandomStartPointSet: arguments must be name-value pairs.")
    for k in range(0, len(args), 2):
        key = text(args[k]).lower()
        if key == "numstartpoints":
            val["n"] = int(fnum(args[k + 1]).flat[0])
        elif key == "artificialbound":
            val["bound"] = float(fnum(args[k + 1]).flat[0])
        else:
            raise MatlabError(f"RandomStartPointSet: unknown property '{text(args[k])}'.")
    return _StartPoints("random", val)


def _problem_dict(problem: Struct) -> dict:
    if not isinstance(problem, Struct) or problem.numel() != 1:
        raise MatlabError("The problem must be a structure made with createOptimProblem.")
    d = {k: v for k, v in problem.data.flat[0].items()}
    if "objective" not in d or "x0" not in d or om._is_empty(d["x0"]):
        raise MatlabError("The problem structure needs the fields objective and x0.")
    return d


def _random_points(rng, d: dict, k: int, bound: float) -> np.ndarray:
    x0 = np.real(fnum(d["x0"])).astype(float).flatten(order="F")
    n = x0.size
    lo, hi = om.bounds(None if om._is_empty(d.get("lb")) else np.real(fnum(d["lb"])).astype(float),
                       None if om._is_empty(d.get("ub")) else np.real(fnum(d["ub"])).astype(float), n,
                       "start points")
    a = np.where(np.isfinite(lo), lo, np.where(np.isfinite(hi), hi - 2 * bound, -bound))
    b = np.where(np.isfinite(hi), hi, np.where(np.isfinite(lo), lo + 2 * bound, bound))
    return a + rng.random((k, n)) * (b - a)


def _local_solve(interp, d: dict, x0: np.ndarray):
    """Run the problem's local solver from x0: (x, fval, exitflag, output, funcCount)."""
    solver = text(d.get("solver", char("fmincon")))
    shape = fnum(d["x0"]).shape
    xs = x0.reshape(shape, order="F")
    options = d.get("options")
    if om._is_empty(options):
        options = om.OptimOptions(solver)
    quiet = om.OptimOptions(options.solver, {**options.user, "Display": char("off")}) \
        if isinstance(options, om.OptimOptions) else options
    if isinstance(quiet, Struct):
        quiet = quiet.copy()
        quiet.set_field(0, "Display", char("off"))
    e = empty_double()

    def g(name):
        v = d.get(name)
        return e if v is None else v

    if solver == "fmincon":
        a = [d["objective"], xs, g("Aineq"), g("bineq"), g("Aeq"), g("beq"), g("lb"), g("ub"), g("nonlcon"), quiet]
        res = interp.call_function_value(char("fmincon"), a, 4)
    elif solver == "fminunc":
        res = interp.call_function_value(char("fminunc"), [d["objective"], xs, quiet], 4)
    elif solver == "lsqnonlin":
        r = interp.call_function_value(char("lsqnonlin"), [d["objective"], xs, g("lb"), g("ub"), quiet], 5)
        res = [r[0], r[1], r[3], r[4]]
    elif solver == "lsqcurvefit":
        r = interp.call_function_value(char("lsqcurvefit"), [d["objective"], xs, g("xdata"), g("ydata"), g("lb"),
                                                             g("ub"), quiet], 5)
        res = [r[0], r[1], r[3], r[4]]
    else:
        raise MatlabError(f"The local solver '{solver}' is not supported.")
    out = res[3]
    fc = float(fnum(out.data.flat[0].get("funcCount", np.zeros((1, 1)))).flat[0]) if isinstance(out, Struct) else 0.0
    return np.real(fnum(res[0])).astype(float), float(fnum(res[1]).flat[0]), float(fnum(res[2]).flat[0]), out, fc


class GlobalSolver(MObject):
    """GlobalSearch or MultiStart: properties plus the run method."""

    DEFAULTS = {"Display": "final", "FunctionTolerance": 1e-6, "XTolerance": 1e-6, "MaxTime": np.inf,
                "StartPointsToRun": "all", "OutputFcn": None, "PlotFcn": None}
    EXTRA = {"GlobalSearch": {"NumTrialPoints": 1000, "NumStageOnePoints": 200, "BasinRadiusFactor": 0.2,
                              "DistanceThresholdFactor": 0.75, "MaxWaitCycle": 20, "PenaltyThresholdFactor": 0.2},
             "MultiStart": {"UseParallel": False}}

    def __init__(self, kind: str, pairs):
        self.kind = kind
        self.p = {**self.DEFAULTS, **self.EXTRA[kind]}
        for name, value in pairs:
            self.p[self._name(name)] = om._to_python(value)

    def _name(self, name):
        for k in self.p:
            if k.lower() == name.lower():
                return k
        raise MatlabError(f"Unrecognized property '{name}' for class '{self.kind}'.")

    def class_name(self):
        return self.kind

    def get_field(self, interp, name):
        return om._default_value(self.p[self._name(name)])

    def set_field(self, interp, name, value):
        new = GlobalSolver(self.kind, [])
        new.p = dict(self.p)
        new.p[self._name(name)] = om._to_python(value)
        return new

    def display_lines(self):
        width = max(len(k) for k in self.p)
        lines = [f"  {self.kind} with properties:", ""]
        for k, v in self.p.items():
            shown = "[]" if v is None else (f"'{v}'" if isinstance(v, str) else (f"{v:g}" if isinstance(
                v, (int, float)) and not isinstance(v, bool) else str(int(v) if isinstance(v, bool) else v)))
            lines.append(f"    {k:>{width}}: {shown}")
        return lines

    def has_method(self, name):
        return name == "run"

    def call_method(self, interp, name, args, nargout):
        if args[0] is not self or len(args) < 2:
            raise MatlabError("run: use run(gs, problem) or run(ms, problem, k).")
        if self.p["OutputFcn"] is not None or self.p["PlotFcn"] is not None:
            raise MatlabError(f"{self.kind}: the OutputFcn and PlotFcn properties are not supported.")
        d = _problem_dict(args[1])
        rng = _rng(interp)
        x0 = np.real(fnum(d["x0"])).astype(float).flatten(order="F")
        n = x0.size
        shape = fnum(d["x0"]).shape
        t0 = time.time()
        max_time = float(self.p["MaxTime"])
        starts = [x0]
        if self.kind == "MultiStart":
            if len(args) < 3:
                raise MatlabError("run: MultiStart needs the number of start points, run(ms, problem, k).")
            spec = args[2]
            if isinstance(spec, _StartPoints):
                pts = spec.value if spec.kind == "custom" else _random_points(rng, d, spec.value["n"],
                                                                              spec.value["bound"])
                if pts.shape[1] != n:
                    raise MatlabError(f"run: the start points must have {n} columns.")
                starts = list(pts)
            elif isinstance(spec, Cell):
                starts = []
                for s in spec.values_f():
                    if not isinstance(s, _StartPoints):
                        raise MatlabError("run: the start point sets must be CustomStartPointSet or "
                                          "RandomStartPointSet objects.")
                    pts = s.value if s.kind == "custom" else _random_points(rng, d, s.value["n"], s.value["bound"])
                    starts += list(pts)
            else:
                k = int(np.real(fnum(spec).flat[0]))
                if k < 1:
                    raise MatlabError("run: the number of start points must be positive.")
                starts += list(_random_points(rng, d, k - 1, 1000.0))
            mode = str(self.p["StartPointsToRun"])
            if mode not in ("all", "bounds", "bounds-ineqs"):
                raise MatlabError("MultiStart: StartPointsToRun must be 'all', 'bounds' or 'bounds-ineqs'.")
            if mode != "all":
                lo, hi = om.bounds(None if om._is_empty(d.get("lb")) else np.real(fnum(d["lb"])).astype(float),
                                   None if om._is_empty(d.get("ub")) else np.real(fnum(d["ub"])).astype(float),
                                   n, "MultiStart")
                starts = [s for s in starts if np.all(s >= lo) and np.all(s <= hi)]
        else:
            ntrial = int(self.p["NumTrialPoints"])
            nstage1 = min(int(self.p["NumStageOnePoints"]), ntrial)
            trial = _random_points(rng, d, ntrial, 1000.0)

            def score(x):
                res = interp.call_function_value(d["objective"], [x.reshape(shape, order="F")], 1)
                f = float(np.real(fnum(res[0]).flat[0]))
                pen = 0.0
                if not om._is_empty(d.get("Aineq")):
                    pen += float(np.sum(np.maximum(fnum(d["Aineq"]) @ x - fnum(d["bineq"]).flatten(), 0.0)))
                if not om._is_empty(d.get("Aeq")):
                    pen += float(np.sum(np.abs(fnum(d["Aeq"]) @ x - fnum(d["beq"]).flatten())))
                if om.is_fn(d.get("nonlcon")):
                    cr = interp.call_function_value(d["nonlcon"], [x.reshape(shape, order="F")], 2)
                    pen += float(np.sum(np.maximum(np.real(fnum(cr[0])), 0.0)) + np.sum(np.abs(np.real(fnum(cr[1])))))
                return f + 1e3 * pen

            scores = np.array([score(x) for x in trial[:nstage1]])
            starts.append(trial[int(np.argmin(scores))])
            rest_pts = trial[nstage1:]
            rest_scores = np.array([score(x) for x in rest_pts]) if rest_pts.shape[0] else np.zeros(0)
            order = np.argsort(rest_scores, kind="stable")
            starts += [rest_pts[i] for i in order]
        solutions = []
        counts = {"total": 0, "success": 0, "incomplete": 0, "nosol": 0, "fevals": 0}
        basins: list = []
        threshold = np.inf
        waited = 0
        for k, s in enumerate(starts):
            if time.time() - t0 > max_time:
                break
            if self.kind == "GlobalSearch" and k >= 2:
                # only start from points outside the known basins that look promising
                if any(np.linalg.norm(s - c) <= r for c, r in basins):
                    continue
                sc = score(s)
                if sc > threshold:
                    waited += 1
                    if waited >= int(self.p["MaxWaitCycle"]):
                        threshold += float(self.p["PenaltyThresholdFactor"]) * (1.0 + abs(threshold))
                        waited = 0
                    continue
                if counts["total"] >= 25:
                    break
            try:
                x, f, flag, out, fc = _local_solve(interp, d, np.asarray(s, dtype=float))
            except MatlabError:
                counts["total"] += 1
                counts["nosol"] += 1
                continue
            counts["total"] += 1
            counts["fevals"] += fc
            if flag > 0:
                counts["success"] += 1
            elif flag == 0:
                counts["incomplete"] += 1
            else:
                counts["nosol"] += 1
            xf = x.flatten(order="F")
            if self.kind == "GlobalSearch":
                radius = float(self.p["DistanceThresholdFactor"]) * max(float(np.linalg.norm(xf - s)), 1e-3)
                basins.append((xf, radius))
                if flag > 0:
                    threshold = min(threshold, f) if np.isfinite(threshold) else f
            if flag > 0 or not solutions:
                solutions.append({"X": x, "Fval": f, "Exitflag": flag, "Output": out,
                                  "X0": [np.asarray(s, dtype=float).reshape(shape, order="F")], "ok": flag > 0})
        if not solutions:
            raise MatlabError(f"{self.kind}: no local solver run produced a solution.")
        good = [s for s in solutions if s["ok"]] or solutions
        good.sort(key=lambda s: s["Fval"])
        xtol, ftol = float(self.p["XTolerance"]), float(self.p["FunctionTolerance"])
        distinct: list = []
        for s in good:
            for t in distinct:
                if abs(s["Fval"] - t["Fval"]) <= ftol * max(1.0, abs(t["Fval"])) and \
                        np.linalg.norm(s["X"] - t["X"]) <= xtol * max(1.0, float(np.linalg.norm(t["X"]))):
                    t["X0"] += s["X0"]
                    break
            else:
                distinct.append(s)
        best = distinct[0]
        if counts["success"] == counts["total"]:
            flag, msg = 1, f"{self.kind} stopped because all local solver runs converged with a positive exit flag."
        elif counts["success"] > 0:
            flag, msg = 2, f"{self.kind} stopped: at least one local solver run converged."
        elif counts["incomplete"] > 0:
            flag, msg = 0, f"{self.kind} stopped: no local solver run converged."
        else:
            flag, msg = -2, f"{self.kind} stopped: no feasible local minimum was found."
        if str(self.p["Display"]) not in ("off", "none"):
            interp.out(f"\n{msg}\n\n{counts['success']} out of {counts['total']} local solver runs converged "
                       "with a positive local solver exit flag.\n\n")
        out = output_struct(funcCount=counts["fevals"], localSolverTotal=counts["total"],
                            localSolverSuccess=counts["success"], localSolverIncomplete=counts["incomplete"],
                            localSolverNoSolution=counts["nosol"], message=msg)
        data = np.empty((1, len(distinct)), dtype=object)
        for i, s in enumerate(distinct):
            data[0, i] = {"X": s["X"], "Fval": np.array([[s["Fval"]]]), "Exitflag": np.array([[s["Exitflag"]]]),
                          "Output": s["Output"], "X0": Cell.from_list(s["X0"])}
        sols = Struct(data, ["X", "Fval", "Exitflag", "Output", "X0"])
        return [best["X"], np.array([[best["Fval"]]]), np.array([[float(flag)]]), out, sols][:max(nargout, 1)]


def _global_ctor(kind: str):
    def f(interp, args, nargout):
        rest = list(args)
        pairs = []
        base = None
        if rest and isinstance(rest[0], GlobalSolver):
            base = rest.pop(0)
        if len(rest) % 2:
            raise MatlabError(f"{kind}: properties must be given as name-value pairs.")
        obj = GlobalSolver(kind, pairs)
        if base is not None:
            for k, v in base.p.items():
                if k in obj.p:
                    obj.p[k] = v
        for k in range(0, len(rest), 2):
            obj.p[obj._name(text(rest[k]))] = om._to_python(rest[k + 1])
        return obj

    return f


builtin("GlobalSearch")(_global_ctor("GlobalSearch"))
builtin("MultiStart")(_global_ctor("MultiStart"))
