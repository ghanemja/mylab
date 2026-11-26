"""Help text for the ODE suite and numerical integration."""

_IVP = ("Solves y' = f(t, y) from tspan(1) to tspan(end) with initial value y0. With two entries in tspan "
        "the output holds every solver step; with more than two it holds exactly those times. "
        "[t, y, te, ye, ie] also returns the event times, states and indices when an Events function is "
        "set. sol = solver(...) returns a structure for deval and odextend. Extra arguments after options "
        "are passed on to f and to the option functions. Options (odeset): RelTol, AbsTol (scalar or "
        "vector), NormControl, MaxStep, InitialStep, Refine, NonNegative, Events, OutputFcn, OutputSel, "
        "Stats and Mass with MStateDependence. A mass matrix M (constant, M(t) or M(t, y)) solves "
        "M*y' = f(t, y).")
_STIFF = (" This is an implicit solver for stiff problems; it also uses Jacobian (matrix or function), "
          "JPattern and Vectorized.")
_OPT_GET = "Value of one option from an options structure; default (or []) when the option is not set."

HELP = {
    "odeset": (["opts = odeset(name, value, ...)", "opts = odeset(oldopts, name, value, ...)",
                "opts = odeset(oldopts, newopts)", "odeset"],
               "Creates or changes an options structure for the ODE solvers. Names are not case sensitive "
               "and may be shortened while they stay unique. The result has one field per option; unset "
               "options are []. With no arguments, lists every option. Options: RelTol (1e-3), AbsTol "
               "(1e-6), NormControl, NonNegative, OutputFcn, OutputSel, Refine, Stats, InitialStep, "
               "MaxStep, Events, Jacobian, JConstant, JPattern, Vectorized, Mass, MStateDependence "
               "('none', 'weak', 'strong'), MvPattern, MassSingular ('yes', 'no', 'maybe'), InitialSlope, "
               "MaxOrder and BDF."),
    "odeget": (["v = odeget(opts, name)", "v = odeget(opts, name, default)"], _OPT_GET),
    "bvpset": (["opts = bvpset(name, value, ...)", "opts = bvpset(oldopts, name, value, ...)"],
               "Options for bvp4c and bvp5c: RelTol (1e-3), AbsTol (1e-6), NMax (largest number of mesh "
               "points), Stats, Vectorized, FJacobian, BCJacobian and SingularTerm."),
    "bvpget": (["v = bvpget(opts, name)", "v = bvpget(opts, name, default)"], _OPT_GET),
    "ddeset": (["opts = ddeset(name, value, ...)", "opts = ddeset(oldopts, name, value, ...)"],
               "Options for dde23 and ddesd: RelTol (1e-3), AbsTol (1e-6), NormControl, InitialStep, "
               "MaxStep, InitialY, Jumps, Events, OutputFcn, OutputSel, Refine and Stats."),
    "ddeget": (["v = ddeget(opts, name)", "v = ddeget(opts, name, default)"], _OPT_GET),
    "ode45": (["[t, y] = ode45(f, tspan, y0)", "[t, y] = ode45(f, tspan, y0, options)",
               "[t, y, te, ye, ie] = ode45(f, tspan, y0, options)", "sol = ode45(...)"],
              _IVP + " ode45 is the general-purpose explicit Runge-Kutta (4,5) pair; with two entries in "
              "tspan it returns four output points per step (Refine 4)."),
    "ode23": (["[t, y] = ode23(f, tspan, y0)", "[t, y] = ode23(f, tspan, y0, options)", "sol = ode23(...)"],
              _IVP + " ode23 is a low-order explicit Runge-Kutta (2,3) pair, good for crude tolerances."),
    "ode113": (["[t, y] = ode113(f, tspan, y0)", "[t, y] = ode113(f, tspan, y0, options)", "sol = ode113(...)"],
               _IVP + " ode113 is meant for stringent tolerances and expensive functions; mylab integrates "
               "with a high-order (8th) explicit pair and a 7th-order continuous extension."),
    "ode78": (["[t, y] = ode78(f, tspan, y0)", "[t, y] = ode78(f, tspan, y0, options)", "sol = ode78(...)"],
              _IVP + " ode78 is a high-order explicit solver for smooth problems and tight tolerances "
              "(an 8th-order pair with a 7th-order continuous extension)."),
    "ode89": (["[t, y] = ode89(f, tspan, y0)", "[t, y] = ode89(f, tspan, y0, options)", "sol = ode89(...)"],
              _IVP + " ode89 is a high-order explicit solver for smooth problems and very tight tolerances "
              "(mylab uses the same 8th-order pair as ode78)."),
    "ode15s": (["[t, y] = ode15s(f, tspan, y0)", "[t, y] = ode15s(f, tspan, y0, options)", "sol = ode15s(...)"],
               _IVP + _STIFF + " ode15s is a variable-order method (numerical differentiation formulas of "
               "orders 1 to 5; BDF 'on' selects the backward differentiation formulas, MaxOrder limits the "
               "order). It solves index-1 differential-algebraic problems given by a singular mass matrix "
               "(MassSingular, InitialSlope); inconsistent initial values of the algebraic variables are "
               "corrected."),
    "ode23s": (["[t, y] = ode23s(f, tspan, y0)", "[t, y] = ode23s(f, tspan, y0, options)", "sol = ode23s(...)"],
               _IVP + _STIFF + " ode23s is a one-step Rosenbrock method of order 2. The mass matrix must "
               "be constant and nonsingular; NonNegative is not available."),
    "ode23t": (["[t, y] = ode23t(f, tspan, y0)", "[t, y] = ode23t(f, tspan, y0, options)", "sol = ode23t(...)"],
               _IVP + _STIFF + " ode23t is the trapezoidal rule, which adds no numerical damping; it also "
               "solves index-1 differential-algebraic problems with a singular mass matrix."),
    "ode23tb": (["[t, y] = ode23tb(f, tspan, y0)", "[t, y] = ode23tb(f, tspan, y0, options)",
                 "sol = ode23tb(...)"],
                _IVP + _STIFF + " ode23tb is the TR-BDF2 method: a trapezoidal stage followed by a "
                "backward differentiation stage of order 2."),
    "ode15i": (["[t, y] = ode15i(f, tspan, y0, yp0)", "[t, y] = ode15i(f, tspan, y0, yp0, options)",
                "[t, y, te, ye, ie] = ode15i(...)", "sol = ode15i(...)"],
               "Solves the fully implicit problem f(t, y, y') = 0 with a variable-order backward "
               "differentiation method. y0 and yp0 must be consistent, f(t0, y0, yp0) = 0; use decic to "
               "compute such values. The Jacobian option is a function [dfdy, dfdyp] = jac(t, y, yp) or a "
               "cell {dfdy, dfdyp}; JPattern is a cell of two sparsity patterns. An Events function is "
               "called as events(t, y, yp)."),
    "decic": (["[y0, yp0] = decic(f, t0, y0guess, fixed_y0, yp0guess, fixed_yp0)",
               "[y0, yp0, resnrm] = decic(f, t0, y0guess, fixed_y0, yp0guess, fixed_yp0, options)"],
              "Consistent initial conditions for ode15i: changes the guesses so that f(t0, y0, yp0) = 0. "
              "fixed_y0 and fixed_yp0 are vectors with 1 for the components that must keep their guessed "
              "value (or [] for none). The slopes are adjusted first; the free components of y0 only when "
              "that is not enough. resnrm is the norm of the remaining residual."),
    "deval": (["y = deval(sol, t)", "y = deval(sol, t, idx)", "[y, yp] = deval(sol, t)", "y = deval(t, sol)"],
              "Evaluates a solution structure from an ODE, DDE or BVP solver at the times t, using the "
              "solver's own continuous extension. Each column of y is the solution at one time; idx picks "
              "components. yp is the derivative of the interpolating polynomial. Times outside the "
              "solution interval raise an error."),
    "odextend": (["solext = odextend(sol, f, tfinal)", "solext = odextend(sol, [], tfinal)",
                  "solext = odextend(sol, f, tfinal, y0, options)"],
                 "Continues a solution structure to tfinal with the same solver and options. An empty f "
                 "reuses the function stored in sol; y0 restarts from a different state; options are "
                 "merged over the stored ones. The result covers the whole interval."),
    "odeplot": ("status = odeplot(t, y, flag)",
                "Output function that plots the solution components against time when the integration "
                "finishes. Use it as odeset('OutputFcn', @odeplot)."),
    "odephas2": ("status = odephas2(t, y, flag)",
                 "Output function that draws a phase-plane plot of the first two selected components."),
    "odephas3": ("status = odephas3(t, y, flag)",
                 "Output function that draws a 3-D phase plot of the first three selected components."),
    "odeprint": ("status = odeprint(t, y, flag)",
                 "Output function that prints the time and the solution at every step."),
    "bvpinit": (["solinit = bvpinit(x, yinit)", "solinit = bvpinit(x, yinit, params)",
                 "solinit = bvpinit(sol, [anew bnew])"],
                "Initial guess for bvp4c and bvp5c: the mesh x and a guess for the solution, either a "
                "vector (the same at every mesh point) or a function y = yinit(x). params is a guess for "
                "unknown parameters. The last form extends a solution to a wider interval."),
    "bvp4c": (["sol = bvp4c(odefun, bcfun, solinit)", "sol = bvp4c(odefun, bcfun, solinit, options)"],
              "Solves the two-point boundary value problem y' = odefun(x, y), bcfun(ya, yb) = 0 by "
              "4th-order collocation with residual control. With unknown parameters the functions are "
              "odefun(x, y, p) and bcfun(ya, yb, p), and bcfun returns one extra condition per parameter. "
              "The result has fields x, y, yp, parameters and stats; evaluate it with deval. Options come "
              "from bvpset."),
    "bvp5c": (["sol = bvp5c(odefun, bcfun, solinit)", "sol = bvp5c(odefun, bcfun, solinit, options)"],
              "Solves a two-point boundary value problem like bvp4c (same arguments and result). mylab "
              "uses the same collocation method for both."),
    "dde23": (["sol = dde23(ddefun, lags, history, tspan)", "sol = dde23(ddefun, lags, history, tspan, options)"],
              "Solves delay differential equations y'(t) = ddefun(t, y, Z) with constant lags: column j of "
              "Z is y(t - lags(j)). history is a vector, a function y = history(t) for t <= t0, or an "
              "earlier solution to continue. Discontinuities are tracked and stepped onto. The result has "
              "fields x, y, yp, discont and stats (and xe, ye, ie with events, called as "
              "events(t, y, Z)); evaluate it with deval."),
    "ddesd": (["sol = ddesd(ddefun, delays, history, tspan)", "sol = ddesd(ddefun, delays, history, tspan, options)"],
              "Solves delay differential equations with general delays. delays is a function d = "
              "delays(t, y) returning the delayed times (not the lags), a cell array of such functions, "
              "or a vector of constant lags. Column j of Z is y(d(j))."),
    "pdepe": (["sol = pdepe(m, pdefun, icfun, bcfun, xmesh, tspan)",
               "sol = pdepe(m, pdefun, icfun, bcfun, xmesh, tspan, options)"],
              "Solves systems of parabolic and elliptic PDEs in one space variable, "
              "c*du/dt = x^(-m)*d/dx(x^m*f) + s, with m = 0 (slab), 1 (cylinder) or 2 (sphere). "
              "[c, f, s] = pdefun(x, t, u, dudx); u0 = icfun(x); [pl, ql, pr, qr] = bcfun(xl, ul, xr, ur, t) "
              "states p + q*f = 0 at each end. sol(i, j, k) is component k at time tspan(i) and point "
              "xmesh(j). The space discretisation is second order; time integration uses ode15s with the "
              "RelTol, AbsTol, InitialStep and MaxStep options."),
    "pdeval": ("[uout, duoutdx] = pdeval(m, xmesh, ui, xout)",
               "Interpolates one component of a pdepe solution at one time, ui = sol(j, :, k), to the "
               "points xout, and returns its space derivative."),
    "ode": (["F = ode(ODEFcn = f, InitialValue = y0)", "F = ode(Name = Value, ...)"],
            "An object describing an initial value problem. Properties: ODEFcn, InitialTime, InitialValue, "
            "Parameters (passed as the third argument of ODEFcn), Solver ('auto', 'stiff', 'nonstiff' or a "
            "solver name), SelectedSolver, RelativeTolerance, AbsoluteTolerance, MassMatrix "
            "(odeMassMatrix), Jacobian (odeJacobian), EventDefinition (odeEvent), NonNegativeVariables, "
            "InitialSlope and SolverOptions (a structure of odeset options such as MaxStep). "
            "S = solve(F, t0, tf) or solve(F, t) returns Time and Solution (one column per time); "
            "fh = solutionFcn(F, t0, tf) returns a function that evaluates the solution anywhere in the "
            "interval."),
    "odeEvent": ("E = odeEvent(EventFcn = g, Direction = dir, Response = r, CallbackFcn = cb)",
                 "Event definition for an ode object: g(t, y) is watched for zero crossings. Direction is "
                 "'both', 'ascending' or 'descending'. Response is 'proceed' (record and continue), 'stop', "
                 "or 'callback', which calls [stop, y] = cb(t, y) and restarts from the returned state."),
    "odeMassMatrix": ("M = odeMassMatrix(MassMatrix = m, Singular = s, StateDependence = d)",
                      "Mass matrix for an ode object: a matrix or a function of (t) or (t, y). Singular is "
                      "'yes', 'no' or 'maybe'; StateDependence is 'none', 'weak' or 'strong'."),
    "odeJacobian": ("J = odeJacobian(Jacobian = j, SparsityPattern = p)",
                    "Jacobian for an ode object: a matrix or a function of (t, y), and optionally its "
                    "sparsity pattern."),
    "odeSolutionValue__": ("y = odeSolutionValue__(sols, t)",
                           "Internal helper behind the function returned by solutionFcn: evaluates stored "
                           "solution structures at the times t."),
    "integral": (["q = integral(f, a, b)", "q = integral(f, a, b, 'RelTol', rt, 'AbsTol', at)",
                  "q = integral(f, a, b, 'ArrayValued', true)", "q = integral(f, a, b, 'Waypoints', w)"],
                 "Integrates f from a to b by adaptive Gauss-Kronrod quadrature. f must accept a vector and "
                 "return a vector of the same size (use .*, ./ and .^). Limits may be infinite, and "
                 "integrable singularities at the end points are handled. With 'ArrayValued' f is called "
                 "with scalars and may return an array. 'Waypoints' are points the path goes through: "
                 "real ones mark discontinuities, complex ones define a piecewise-linear contour. "
                 "Defaults: RelTol 1e-6, AbsTol 1e-10."),
    "quadgk": (["q = quadgk(f, a, b)", "[q, errbnd] = quadgk(f, a, b, 'RelTol', rt, 'AbsTol', at)",
                "q = quadgk(f, a, b, 'Waypoints', w, 'MaxIntervalCount', n)"],
               "Adaptive Gauss-Kronrod (7, 15) quadrature of a vectorised function, with infinite limits, "
               "end-point singularities and complex contours. errbnd is an estimate of the absolute error."),
    "quad": (["q = quad(f, a, b)", "q = quad(f, a, b, tol)", "[q, fcnt] = quad(f, a, b, tol, trace)"],
             "Adaptive Simpson quadrature of a vectorised function over a finite interval, to an absolute "
             "tolerance (default 1e-6). fcnt is the number of function evaluations."),
    "quadl": (["q = quadl(f, a, b)", "q = quadl(f, a, b, tol)", "[q, fcnt] = quadl(f, a, b, tol, trace)"],
              "Adaptive Gauss-Lobatto quadrature of a vectorised function over a finite interval, to an "
              "absolute tolerance (default 1e-6)."),
    "quadv": (["Q = quadv(f, a, b)", "[Q, fcnt] = quadv(f, a, b, tol)"],
              "Adaptive Simpson quadrature of an array-valued function f(x) of a scalar x."),
    "integral2": (["q = integral2(f, xmin, xmax, ymin, ymax)",
                   "q = integral2(f, xmin, xmax, ymin, ymax, 'Method', m, 'RelTol', rt, 'AbsTol', at)"],
                  "Double integral of f(x, y) over xmin <= x <= xmax, ymin(x) <= y <= ymax(x). f takes "
                  "arrays and returns an array of the same size; ymin and ymax may be numbers or function "
                  "handles of x. Method is 'tiled' (rectangles, finite regions), 'iterated' (nested 1-D "
                  "integrals, needed for infinite limits) or 'auto'."),
    "integral3": (["q = integral3(f, xmin, xmax, ymin, ymax, zmin, zmax)",
                   "q = integral3(..., 'Method', m, 'RelTol', rt, 'AbsTol', at)"],
                  "Triple integral of f(x, y, z). ymin and ymax may be functions of x; zmin and zmax may "
                  "be functions of (x, y). f takes arrays of equal size."),
    "quad2d": (["q = quad2d(f, a, b, c, d)", "[q, errbnd] = quad2d(f, a, b, c, d, 'AbsTol', at, 'RelTol', rt)"],
               "Double integral of f(x, y) over a <= x <= b, c(x) <= y <= d(x) by adaptive tiling. Options: "
               "AbsTol (1e-5), RelTol, MaxFunEvals (2000) and Singular (true: use a change of variable "
               "that weakens boundary singularities)."),
    "dblquad": (["q = dblquad(f, xmin, xmax, ymin, ymax)", "q = dblquad(f, xmin, xmax, ymin, ymax, tol, method)"],
                "Double integral over a rectangle. f(x, y) is called with a vector x and a scalar y. "
                "method is @quad (default), @quadl, @quadgk or @integral."),
    "triplequad": (["q = triplequad(f, xmin, xmax, ymin, ymax, zmin, zmax)",
                    "q = triplequad(f, xmin, xmax, ymin, ymax, zmin, zmax, tol, method)"],
                   "Triple integral over a box. f(x, y, z) is called with a vector x and scalar y and z."),
}
