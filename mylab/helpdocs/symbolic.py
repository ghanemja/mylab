"""Help text for symbolic math (sym, syms and the functions that work on them)."""

_NEEDS = " Needs SymPy: pip install \"mylab-lang[symbolic]\"."

HELP = {
    # -- creating symbolic values ------------------------------------------------------------
    "syms": (["syms x y z", "syms x real", "syms x positive", "syms f(t)", "syms a [1 3]", "syms A [2 2]"],
             "Creates symbolic variables in the workspace. A word such as real, positive, integer or rational "
             "after the names sets an assumption. 'syms f(t)' creates a symbolic function f and its variable t. "
             "A size in brackets creates a vector or matrix of numbered variables (a1, a2, ... or A1_1, A1_2, ...). "
             "Without arguments it lists the symbolic variables." + _NEEDS),
    "sym": (["x = sym('x')", "x = sym('x', 'real')", "v = sym('a', [1 3])", "r = sym(1/3)", "p = sym(pi)",
             "A = sym([1 2; 3 4])", "e = sym('1/3')"],
            "Creates a symbolic variable, or converts numbers to exact symbolic values. Doubles that are close "
            "to a simple fraction, a multiple of pi or a square root become that exact value, so sym(1/3) is 1/3. "
            "sym(x, 'd') keeps a decimal number and sym(x, 'f') keeps the exact binary value. Text that is not "
            "a name is read as a number or an expression." + _NEEDS),
    "str2sym": ("e = str2sym('x^2 + 3*x - 1/2')",
                "Turns text into a symbolic expression. Numbers stay exact and names become symbolic variables. "
                "A single = sign is read as an equation." + _NEEDS),
    "symfun": ("f = symfun(x^2 + y, [x y])",
               "Creates a symbolic function with the given argument order, so that f(1, 2) substitutes values. "
               "The short form is f(x, y) = x^2 + y." + _NEEDS),
    "argnames": ("v = argnames(f)", "The arguments of a symbolic function, as a row vector of variables."),
    "formula": ("e = formula(f)", "The expression that defines a symbolic function, as a plain sym."),
    "symvar": (["v = symvar(expr)", "v = symvar(expr, n)"],
               "The symbolic variables in an expression, in alphabetical order. With n, the n variables closest "
               "to x in the alphabet (the ones functions such as diff pick by default)."),
    "assume": (["assume(x > 0)", "assume(x, 'real')", "assume(x, 'clear')"],
               "Sets an assumption on a symbolic variable, replacing earlier ones. Later simplifications and "
               "solutions use it. Sets: 'real', 'positive', 'integer', 'rational'; 'clear' removes assumptions."),
    "assumeAlso": (["assumeAlso(x < 5)", "assumeAlso(x, 'integer')"],
                   "Adds an assumption on a symbolic variable and keeps the ones already set."),
    "assumptions": (["assumptions(x)", "assumptions"],
                    "The assumptions set on a variable (or on all variables), as a vector of conditions."),
    "symtrue": ("t = symtrue", "The symbolic logical constant true."),
    "symfalse": ("f = symfalse", "The symbolic logical constant false."),
    "isSymType": ("tf = isSymType(expr, type)",
                  "True if a symbolic value is of the given kind: 'variable', 'number', 'integer', 'rational', "
                  "'constant', 'expression', 'equation', 'function', 'plus', 'times' or 'power'."),
    "has": ("tf = has(expr, sub)", "True where a symbolic expression contains the variable or subexpression sub."),
    # -- calculus ------------------------------------------------------------------------------
    "int": (["F = int(f)", "F = int(f, x)", "q = int(f, a, b)", "q = int(f, x, a, b)"],
            "Symbolic integral. Without limits it returns an antiderivative (without the constant); with limits "
            "a and b the definite integral. The variable defaults to x, or the one closest to x. An integral "
            "that cannot be found is returned unevaluated." + _NEEDS),
    "limit": (["L = limit(f, x, a)", "L = limit(f, a)", "L = limit(f, x, a, 'left')", "L = limit(f, x, a, 'right')"],
              "Limit of a symbolic expression as x approaches a (0 when a is left out). 'left' and 'right' give "
              "one-sided limits. When the two sides differ the two-sided limit is NaN." + _NEEDS),
    "taylor": (["T = taylor(f)", "T = taylor(f, x)", "T = taylor(f, x, a)", "T = taylor(f, x, 'Order', n)",
                "T = taylor(f, x, 'ExpansionPoint', a)"],
               "Taylor polynomial of f around a (default 0). 'Order' n keeps the terms below degree n; the "
               "default 6 gives terms up to the fifth power." + _NEEDS),
    "symsum": (["S = symsum(f, k, a, b)", "S = symsum(f, a, b)"],
               "Sum of the symbolic expression f for k from a to b. b may be Inf for a series." + _NEEDS),
    "symprod": (["P = symprod(f, k, a, b)", "P = symprod(f, a, b)"],
                "Product of the symbolic expression f for k from a to b." + _NEEDS),
    "jacobian": ("J = jacobian(f, v)",
                 "Jacobian matrix: the derivative of each element of the vector f with respect to each variable "
                 "in v. Row i, column j holds diff(f(i), v(j))."),
    "hessian": ("H = hessian(f, v)", "Matrix of second derivatives of the scalar expression f for the variables v."),
    "divergence": ("d = divergence(V, X)",
                   "Divergence of the symbolic vector field V with respect to the variables X."),
    "curl": ("c = curl(V, X)", "Curl of a symbolic vector field V with three components, for the variables X."),
    "laplacian": ("L = laplacian(f, X)", "Sum of the second derivatives of f with respect to each variable in X."),
    # -- algebra -------------------------------------------------------------------------------
    "simplify": ("s = simplify(expr)",
                 "Rewrites a symbolic expression in a simpler form, for example sin(x)^2 + cos(x)^2 becomes 1. "
                 "Works on each element of a matrix and on both sides of an equation." + _NEEDS),
    "simplifyFraction": ("s = simplifyFraction(expr)",
                         "Writes a symbolic expression as one fraction and cancels common factors."),
    "expand": ("e = expand(expr)",
               "Multiplies out products and powers, and expands sin and cos of sums, e.g. (x + 1)^2 becomes "
               "x^2 + 2*x + 1." + _NEEDS),
    "collect": (["c = collect(expr)", "c = collect(expr, x)"],
                "Groups the terms of an expression by powers of x (or of the default variable)."),
    "combine": (["c = combine(expr)", "c = combine(expr, 'log')", "c = combine(expr, 'sincos')"],
                "Combines terms of the same kind: powers with the same base, logarithms, or products of "
                "sines and cosines."),
    "rewrite": ("r = rewrite(expr, target)",
                "Rewrites an expression in terms of another function. Targets include 'exp', 'sin', 'cos', "
                "'tan', 'log', 'sincos', 'sqrt' and 'heaviside'."),
    "horner": ("h = horner(p, x)", "Writes a polynomial in nested (Horner) form."),
    "numden": ("[n, d] = numden(expr)",
               "Numerator and denominator of a symbolic expression after writing it as one fraction."),
    "partfrac": (["p = partfrac(expr)", "p = partfrac(expr, x)"],
                 "Partial fraction decomposition of a rational expression in x."),
    "subs": (["r = subs(expr, old, new)", "r = subs(expr, new)", "r = subs(expr)",
              "r = subs(expr, {x, y}, {1, 2})", "r = subs(expr, [x y], [1 2])"],
             "Replaces variables (or subexpressions) with new values. Several replacements are made at the same "
             "time. A vector or matrix value expands the result to that size. With one argument, variables are "
             "replaced by the workspace variables of the same name. The result is still symbolic: use double "
             "to get a number." + _NEEDS),
    "coeffs": (["c = coeffs(p)", "c = coeffs(p, x)", "[c, t] = coeffs(p, x)", "c = coeffs(p, x, 'All')"],
               "Coefficients of a polynomial. One output lists the nonzero coefficients from the lowest power "
               "up. With two outputs, c and the matching terms t run from the highest power down. 'All' also "
               "returns zero coefficients, highest power first."),
    "sym2poly": ("c = sym2poly(p)",
                 "Numeric coefficient vector of a symbolic polynomial in one variable, highest power first."),
    "poly2sym": (["p = poly2sym(c)", "p = poly2sym(c, x)"],
                 "Symbolic polynomial from a coefficient vector (highest power first), in x or the given variable."),
    "polynomialDegree": (["n = polynomialDegree(p)", "n = polynomialDegree(p, x)"],
                         "Degree of a symbolic polynomial."),
    "lhs": ("l = lhs(eqn)", "Left side of a symbolic equation or inequality."),
    "rhs": ("r = rhs(eqn)", "Right side of a symbolic equation or inequality."),
    "children": ("c = children(expr)",
                 "The parts of a symbolic expression (terms of a sum, factors of a product, arguments of a "
                 "function) as a cell array."),
    "isAlways": ("tf = isAlways(cond)",
                 "True if a symbolic condition such as sin(x)^2 + cos(x)^2 == 1 holds for all values of its "
                 "variables. False when it does not hold or cannot be decided."),
    "finverse": (["g = finverse(f)", "g = finverse(f, x)"], "Inverse function of the symbolic expression f(x)."),
    # -- solving -------------------------------------------------------------------------------
    "solve": (["s = solve(eqn, x)", "s = solve(expr)", "[x, y] = solve([eqn1, eqn2], [x, y])",
               "S = solve([eqn1, eqn2], [x, y])", "s = solve(eqn, x, 'Real', true)",
               "[s, params, conds] = solve(eqn, x, 'ReturnConditions', true)"],
              "Solves equations symbolically. An expression without == is set equal to zero. One unknown gives a "
              "column vector of solutions. A system gives one output per unknown, or with a single output a "
              "struct with one field per unknown. 'Real' keeps real solutions only. 'ReturnConditions' also "
              "returns parameters and conditions (always empty and true here: periodic solution families are "
              "not listed)." + _NEEDS),
    "vpasolve": (["s = vpasolve(eqn, x)", "s = vpasolve(eqn, x, x0)", "s = vpasolve(eqn, x, [a b])",
                  "[x, y] = vpasolve([eqn1, eqn2], [x, y])"],
                 "Solves equations numerically to the precision set by digits. A polynomial equation returns all "
                 "roots; other equations return one solution near the start value x0 or inside [a b]." + _NEEDS),
    "dsolve": (["y = dsolve(eqn)", "y = dsolve(eqn, cond1, cond2)", "[x, y] = dsolve(eqn1, eqn2, conds)"],
               "Solves ordinary differential equations. Declare the unknown with 'syms y(t)', write derivatives "
               "with diff(y, t), and give conditions such as y(0) == 1. For a condition on a derivative use "
               "Dy = diff(y, t); Dy(0) == 0. Constants of integration are named C1, C2, ..." + _NEEDS),
    # -- transforms ----------------------------------------------------------------------------
    "laplace": (["F = laplace(f)", "F = laplace(f, t, s)"],
                "Laplace transform of f(t), as a function of s." + _NEEDS),
    "ilaplace": (["f = ilaplace(F)", "f = ilaplace(F, s, t)"],
                 "Inverse Laplace transform of F(s), as a function of t (for t >= 0)." + _NEEDS),
    "fourier": (["F = fourier(f)", "F = fourier(f, x, w)"],
                "Fourier transform of f(x): the integral of f(x)*exp(-i*w*x) over all x. A transform that "
                "cannot be found is returned unevaluated." + _NEEDS),
    "ifourier": (["f = ifourier(F)", "f = ifourier(F, w, x)"],
                 "Inverse Fourier transform: 1/(2*pi) times the integral of F(w)*exp(i*w*x) over all w." + _NEEDS),
    "ztrans": (["F = ztrans(f)", "F = ztrans(f, n, z)"],
               "Z-transform of the sequence f(n): the sum of f(n)/z^n for n = 0, 1, 2, ..." + _NEEDS),
    "iztrans": (["f = iztrans(F)", "f = iztrans(F, z, n)"],
                "Inverse Z-transform of a rational function F(z), as a sequence in n." + _NEEDS),
    # -- linear algebra ------------------------------------------------------------------------
    "rref": ("R = rref(A)", "Reduced row echelon form of a symbolic matrix."),
    "charpoly": (["c = charpoly(A)", "p = charpoly(A, x)"],
                 "Characteristic polynomial of a square symbolic matrix: its coefficients (highest power "
                 "first), or the polynomial in x."),
    "colspace": ("B = colspace(A)", "Basis of the column space of a symbolic matrix, one basis vector per column."),
    "adjoint": ("B = adjoint(A)",
                "Classical adjoint (adjugate) of a square symbolic matrix: A*adjoint(A) is det(A)*I."),
    # -- numbers, text and functions -----------------------------------------------------------
    "vpa": (["v = vpa(expr)", "v = vpa(expr, d)"],
            "Evaluates the numbers in a symbolic expression to d significant digits (default: the digits "
            "setting, 32). Variables stay symbolic." + _NEEDS),
    "digits": (["digits(d)", "d = digits"],
               "Sets or returns the number of significant digits used by vpa and vpasolve (32 by default)."),
    "matlabFunction": (["f = matlabFunction(expr)", "f = matlabFunction(expr, 'Vars', [x y])"],
                       "Converts a symbolic expression into a function handle that works on numeric arrays. "
                       "The inputs are the variables in alphabetical order, or the order given by 'Vars'. "
                       "Writing a file is not supported."),
    "latex": ("s = latex(expr)", "LaTeX code for a symbolic expression, as text."),
    "pretty": ("pretty(expr)", "Prints a symbolic expression in a two-dimensional text layout."),
    "piecewise": ("p = piecewise(cond1, val1, cond2, val2, ..., otherwise)",
                  "An expression defined by cases: val1 where cond1 holds, else val2 where cond2 holds, and so "
                  "on. The optional last value is used when no condition holds (otherwise NaN)."),
    "heaviside": ("h = heaviside(x)",
                  "Step function: 0 for x < 0, 1 for x > 0 and 1/2 at x = 0. Works on numbers and on symbolic "
                  "expressions."),
    "dirac": (["d = dirac(x)", "d = dirac(n, x)"],
              "Dirac delta function (or its n-th derivative). For numbers: Inf at 0 and 0 elsewhere."),
    "kroneckerDelta": (["d = kroneckerDelta(m)", "d = kroneckerDelta(m, n)"],
                       "1 if m equals n (or 0 when n is left out), otherwise 0."),
    "lambertw": (["w = lambertw(x)", "w = lambertw(k, x)"],
                 "Lambert W function: the solution w of w*exp(w) = x, on branch k (default 0)."),
    "zeta": ("z = zeta(x)", "Riemann zeta function."),
    "sinint": ("y = sinint(x)", "Sine integral: the integral of sin(t)/t from 0 to x."),
    "cosint": ("y = cosint(x)", "Cosine integral Ci(x)."),
    "sinhint": ("y = sinhint(x)", "Hyperbolic sine integral: the integral of sinh(t)/t from 0 to x."),
    "coshint": ("y = coshint(x)", "Hyperbolic cosine integral Chi(x)."),
    "ei": ("y = ei(x)", "Exponential integral Ei(x): the principal value integral of exp(t)/t up to x."),
    # -- elementary functions that were missing -------------------------------------------------
    "cbrt": ("y = cbrt(x)", "Real cube root of each element."),
    "coth": ("y = coth(x)", "Hyperbolic cotangent, cosh(x)/sinh(x)."),
    "sech": ("y = sech(x)", "Hyperbolic secant, 1/cosh(x)."),
    "csch": ("y = csch(x)", "Hyperbolic cosecant, 1/sinh(x)."),
    "acot": ("y = acot(x)", "Inverse cotangent in radians, atan(1/x)."),
    "asec": ("y = asec(x)", "Inverse secant in radians, acos(1/x)."),
    "acsc": ("y = acsc(x)", "Inverse cosecant in radians, asin(1/x)."),
    "acoth": ("y = acoth(x)", "Inverse hyperbolic cotangent, atanh(1/x)."),
    "asech": ("y = asech(x)", "Inverse hyperbolic secant, acosh(1/x)."),
    "acsch": ("y = acsch(x)", "Inverse hyperbolic cosecant, asinh(1/x)."),
}
