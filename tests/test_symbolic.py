"""Symbolic math (sym, syms and friends). Skipped when SymPy is not installed."""

import numpy as np
import pytest

from mylab.errors import MatlabError

pytest.importorskip("sympy")


@pytest.fixture
def s(m):
    """A session with a few symbols, plus short helpers: txt(expr), dbl(expr)."""
    m.run("syms x y z t a b n k")
    m.txt = lambda expr: m.str(f"char({expr})")
    m.dbl = lambda expr: m.num(f"double({expr})")
    return m


def close(a, b, tol=1e-9):
    return np.allclose(np.asarray(a, dtype=complex), np.asarray(b, dtype=complex), atol=tol, rtol=tol)


# -- creation ---------------------------------------------------------------------------------

def test_syms_and_class(s):
    assert s.str("class(x)") == "sym"
    assert s.num("isa(x, 'sym')").item() is True
    assert s.num("isa(5, 'sym')").item() is False
    assert s.txt("x") == "x"
    assert s.num("size(x)").tolist() == [[1, 1]]


def test_syms_flags(s):
    s.run("syms p positive; syms r real; syms q")
    assert s.txt("sqrt(p^2)") == "p"
    assert s.txt("conj(r)") == "r"
    assert s.txt("conj(q)") == "conj(q)"
    assert s.txt("simplify(sqrt(q^2))") != "q"


def test_syms_function(s):
    s.run("syms f(t)")
    assert s.str("class(f)") == "symfun"
    assert s.txt("f") == "f(t)"
    assert s.txt("f(0)") == "f(0)"
    assert s.txt("diff(f, t)") == "diff(f(t), t)"
    s.run("syms g(x, y)")
    assert s.txt("g(1, 2)") == "g(1, 2)"


def test_syms_arrays(s):
    s.run("syms v [1 3]; syms M [2 2]; syms c [2 1]")
    assert s.txt("v") == "[v1, v2, v3]"
    assert s.txt("M") == "[M1_1, M1_2; M2_1, M2_2]"
    assert s.num("size(c)").tolist() == [[2, 1]]
    assert s.txt("sym('w', [1 2])") == "[w1, w2]"


def test_syms_function_call_form(s):
    s.run("syms('u', 'w')")
    assert s.txt("u + w") == "u + w"
    out = s.run("clear all; syms p q; syms")
    assert "p" in out and "q" in out


def test_sym_numbers(s):
    assert s.txt("sym(1/3)") == "1/3"
    assert s.txt("sym(pi)") == "pi"
    assert s.txt("sym(pi/2)") == "pi/2"
    assert s.txt("sym(sqrt(2))") == "2^(1/2)"
    assert s.txt("sym(0.1)") == "1/10"
    assert s.txt("sym(-0.75)") == "-3/4"
    assert s.txt("sym('1/3')") == "1/3"
    assert s.txt("sym('0.25')") == "1/4"
    assert s.txt("sym(2)^100") == "1267650600228229401496703205376"
    assert s.txt("sym('123456789012345678901234567890') + 1") == "123456789012345678901234567891"
    assert s.txt("sym(1 + 2i)") == "1 + 2i"
    assert s.txt("sym(Inf)") == "Inf"


def test_sym_matrix_and_text(s):
    assert s.txt("sym([1 2; 3 4])") == "[1, 2; 3, 4]"
    assert s.txt("sym([0.5 1/3])") == "[1/2, 1/3]"
    assert s.txt("str2sym('x^2 + 3*y - 1/2')") == "x^2 + 3*y - 1/2"
    assert s.txt("lhs(str2sym('x^2 = 4'))") == "x^2"
    assert s.txt("sym(@(u) u^2 + 1)") == "u^2 + 1"


def test_symvar(s):
    assert s.txt("symvar(a*x + b*y)") == "[a, b, x, y]"
    assert s.txt("symvar(a*t + b, 1)") == "t"
    assert s.num("isempty(symvar(sym(3)))").item() is True


def test_assumptions(s):
    s.run("assume(x > 0)")
    assert s.txt("simplify(sqrt(x^2))") == "x"
    assert "x" in s.txt("assumptions(x)") and "0" in s.txt("assumptions(x)")
    s.run("assume(x > 2)")
    assert s.txt("solve(x^2 == 9, x)") == "3"
    s.run("assumeAlso(x, 'integer')")
    assert s.num("numel(assumptions(x))").item() >= 1
    s.run("syms x")
    assert s.num("isempty(assumptions(x))").item() is True
    assert s.txt("solve(x^2 == 9, x)") == "[-3; 3]"
    s.run("assume(y, 'real')")
    assert s.txt("conj(y)") == "y"
    s.run("assume(y, 'clear')")
    assert s.txt("conj(y)") == "conj(y)"


def test_is_sym_type_and_has(s):
    assert s.num("isSymType(x, 'variable')").item() is True
    assert s.num("isSymType(sym(2), 'integer')").item() is True
    assert s.num("isSymType(x == 1, 'equation')").item() is True
    assert s.num("has(x^2 + y, y)").item() is True
    assert s.num("has(x^2, y)").item() is False


# -- the object: arithmetic, indexing, display --------------------------------------------------

def test_arithmetic_and_printing(s):
    assert s.txt("x^2 + 2*x + 1") == "x^2 + 2*x + 1"
    assert s.txt("x + x") == "2*x"
    assert s.txt("(2*x)/3") == "(2*x)/3"
    assert s.txt("x*y/2") == "(x*y)/2"
    assert s.txt("1/x") == "1/x"
    assert s.txt("x/y/2") == "x/(2*y)"
    assert s.txt("x - 1") == "x - 1"
    assert s.txt("1 - x^2") == "1 - x^2"
    assert s.txt("b^2 - 4*a") == "b^2 - 4*a"
    assert s.txt("-(x + 1)") == "- x - 1" or s.txt("-(x + 1)") == "-x - 1"
    assert s.txt("exp(x)") == "exp(x)"
    assert s.txt("exp(sym(1))") == "exp(1)"
    assert s.txt("pi*x") == "pi*x"
    assert s.txt("sqrt(x)") == "x^(1/2)"
    assert s.txt("x^(y + 1)") == "x^(y + 1)"
    assert s.txt("sin(x)^2") == "sin(x)^2"
    assert s.txt("x*1i") == "x*1i"
    assert s.txt("log(x) + abs(y)") == "log(x) + abs(y)" or s.txt("log(x) + abs(y)") == "abs(y) + log(x)"


def test_mixing_with_doubles(s):
    assert s.str("class(x + 1)") == "sym"
    assert s.str("class(2*x)") == "sym"
    assert s.txt("x + 0.5") == "x + 1/2"
    assert s.txt("2^x") == "2^x"
    assert s.txt("[1 2 3]*x") == "[x, 2*x, 3*x]"
    assert s.txt("sym(3)/4 + 0.25") == "1"
    assert s.txt("1/sym(0)") == "Inf"


def test_matrix_operations(s):
    s.run("M = [x 1; 2 y];")
    assert s.txt("M*M") == "[x^2 + 2, x + y; 2*x + 2*y, y^2 + 2]"
    assert s.txt("M.*M") == "[x^2, 1; 4, y^2]"
    assert s.txt("M.^2") == "[x^2, 1; 4, y^2]"
    assert s.txt("M.'") == "[x, 2; 1, y]"
    assert s.txt("sym([1 2; 3 4])'") == "[1, 3; 2, 4]"
    assert s.txt("sym([1 2; 3 4])^2") == "[7, 10; 15, 22]"
    assert s.txt("sym([1 2; 3 4]) \\ sym([5; 6])") == "[-4; 9/2]"
    assert s.txt("simplify(M / M)") == "[1, 0; 0, 1]"
    assert s.txt("sum(sym([1 2; 3 4]))") == "[4, 6]"
    assert s.txt("prod(sym([1 2; 3 4]), 2)") == "[2; 12]"
    with pytest.raises(MatlabError):
        s.run("[x y]*[x y]")


def test_relations(s):
    assert s.txt("x^2 == 4") == "x^2 == 4"
    assert s.txt("x ~= y") == "x ~= y"
    assert s.txt("x <= y") == "x <= y"
    assert s.txt("sym(1) == sym(2)") == "symfalse"
    assert s.num("logical(x == x)").item() is True
    assert s.num("isequal(x + x, 2*x)").item() is True
    assert s.num("isequal(x, y)").item() is False
    assert "yes" in s.run("if sym(2) > 1, disp('yes'), end")
    with pytest.raises(MatlabError):
        s.run("if x > 1, disp('no'), end")


def test_indexing_and_concatenation(s):
    s.run("A = [x x^2; 1 x+1];")
    assert s.txt("A(1, 2)") == "x^2"
    assert s.txt("A(end, end)") == "x + 1"
    assert s.txt("A(:, 1)") == "[x; 1]"
    assert s.txt("A(3)") == "x^2"
    s.run("A(2, :) = [y y]; A(1, 1) = 0;")
    assert s.txt("A") == "[0, x^2; y, y]"
    s.run("B = [A, [1; 2]]; C = [A; x y];")
    assert s.num("size(B)").tolist() == [[2, 3]]
    assert s.num("size(C)").tolist() == [[3, 2]]
    s.run("B(:, 1) = [];")
    assert s.txt("B") == "[x^2, 1; y, 2]"
    s.run("v = sym([]); v(3) = x;")
    assert s.txt("v") == "[0, 0, x]"
    assert s.num("numel(C)").item() == 6
    out = s.run("for e = [x y], disp(e), end")
    assert out.split() == ["x", "y"]


def test_symbolic_functions(s):
    s.run("f(x) = x^2 + 1; g(x, y) = x*y;")
    assert s.str("class(f)") == "symfun"
    assert s.txt("f(3)") == "10"
    assert s.txt("f(y + 1)") == "(y + 1)^2 + 1"
    assert s.txt("f([1 2 3])") == "[2, 5, 10]"
    assert s.txt("g(2, 3)") == "6"
    s.run("df = diff(f);")
    assert s.str("class(df)") == "symfun"
    assert s.txt("df(2)") == "4"
    assert s.str("class(f + 1)") == "symfun"
    assert s.txt("argnames(g)") == "[x, y]"
    assert s.str("class(formula(f))") == "sym"
    s.run("h(x) = 1;")
    assert s.txt("h(5)") == "1"
    s.run("w = symfun(x + 2*y, [y x]);")
    assert s.txt("w(1, 0)") == "2"


def test_display(s):
    assert s.run("f = x^2 + 2*x + 1") == "f =\n \nx^2 + 2*x + 1\n \n"
    assert s.run("A = [x x^2; 1 x+1]") == "A =\n \n[x,   x^2]\n[1, x + 1]\n \n"
    assert s.run("disp(x^2)") == "x^2\n"
    assert s.run("f(x) = x^3") == "f(x) =\n \nx^3\n \n"
    assert "Empty sym" in s.run("e = sym([])")
    assert "sym" in s.run("c = {x, 2}")


# -- calculus ---------------------------------------------------------------------------------

def test_diff(s):
    assert s.txt("diff(x^3)") == "3*x^2"
    assert s.txt("diff(x^3*y^2, y)") == "2*x^3*y"
    assert s.txt("diff(x^5, 3)") == "60*x^2"
    assert s.txt("diff(x^5*y, x, 2)") == "20*x^3*y"
    assert s.txt("diff(x^2*y^3, x, y)") == "6*x*y^2"
    assert s.txt("diff(sin(a*t))") == "a*cos(a*t)"
    assert s.txt("diff([x^2, sin(x)])") == "[2*x, cos(x)]"
    assert close(s.dbl("subs(diff(sin(x)*x^2), x, 2)"), 4 * np.cos(2) + 4 * np.sin(2))


def test_int(s):
    assert s.txt("int(x^2)") == "x^3/3"
    assert s.txt("int(x^2, 0, 1)") == "1/3"
    assert s.txt("int(1/x)") == "log(x)"
    assert s.txt("int(x*y, y)") == "(x*y^2)/2"
    assert s.txt("int(x, x, [0 2])") == "2"
    assert s.txt("int(exp(-x^2), x, -inf, inf)") == "pi^(1/2)"
    assert s.txt("int(1/(1 + x^2), -inf, inf)") == "pi"
    assert close(s.dbl("subs(int(sin(a*x), x), {a, x}, {2, 1})"), -np.cos(2) / 2)
    assert close(s.dbl("int(sin(x)^2, 0, pi)"), np.pi / 2)
    assert close(s.dbl("int(x*exp(x), x, 0, 1)"), 1.0)


def test_limit(s):
    assert s.txt("limit(sin(x)/x)") == "1"
    assert s.txt("limit(sin(x)/x, x, 0)") == "1"
    assert s.txt("limit(1/x, x, 0, 'right')") == "Inf"
    assert s.txt("limit(1/x, x, 0, 'left')") == "-Inf"
    assert s.txt("limit((1 + 1/n)^n, n, inf)") == "exp(1)"
    assert s.txt("limit((x^2 - 1)/(x - 1), 1)") == "2"


def test_taylor(s):
    assert s.txt("taylor(exp(x))") == "x^5/120 + x^4/24 + x^3/6 + x^2/2 + x + 1"
    assert s.txt("taylor(sin(x), x, 'Order', 4)") == "- x^3/6 + x"
    assert s.txt("taylor(cos(x), x, 0, 'Order', 5)") == "x^4/24 - x^2/2 + 1"
    assert close(s.dbl("subs(taylor(log(x), x, 'ExpansionPoint', 1, 'Order', 4), x, 1.1)"),
                 0.1 - 0.005 + 0.001 / 3)


def test_sums_and_products(s):
    assert s.txt("symsum(k, k, 1, 10)") == "55"
    assert s.txt("factor(symsum(k^2, k, 1, n))") == "[1/6, n, n + 1, 2*n + 1]"
    assert s.txt("symsum(1/k^2, k, 1, inf)") == "pi^2/6"
    assert s.txt("symsum(1/2^k, k, 0, inf)") == "2"
    assert s.txt("symprod(k, k, 1, 5)") == "120"


def test_vector_calculus(s):
    assert s.txt("jacobian([x*y; x^2 + y], [x y])") == "[y, x; 2*x, 1]"
    assert s.txt("gradient(x^2*y + z, [x y z])") == "[2*x*y; x^2; 1]"
    assert s.txt("hessian(x^2*y, [x y])") == "[2*y, 2*x; 2*x, 0]"
    assert s.txt("divergence([x^2, y^2, z^2], [x y z])") == "2*x + 2*y + 2*z"
    assert s.txt("curl([y, -x, z], [x y z])") == "[0; 0; -2]"
    assert s.txt("laplacian(x^2 + y^3, [x y])") == "6*y + 2"


# -- algebra ----------------------------------------------------------------------------------

def test_simplify_expand_factor(s):
    assert s.txt("simplify(sin(x)^2 + cos(x)^2)") == "1"
    assert s.txt("simplify((x^2 - 1)/(x - 1))") == "x + 1"
    assert s.txt("expand((x + 1)^3)") == "x^3 + 3*x^2 + 3*x + 1"
    assert s.txt("expand(sin(x + y))") == "sin(x)*cos(y) + sin(y)*cos(x)"
    assert s.txt("factor(x^2 - 1)") == "[x - 1, x + 1]"
    assert s.txt("prod(factor(x^3 - x))") == s.txt("x*(x - 1)*(x + 1)")
    assert s.txt("factor(sym(60))") == "[2, 2, 3, 5]"
    assert s.txt("simplifyFraction((x^2 - 1)/(x + 1))") == "x - 1"


def test_collect_combine_rewrite(s):
    assert s.txt("collect(x*y + x^2*y + x, x)") == "x^2*y + x*(y + 1)"
    assert s.txt("combine(exp(x)*exp(2*x))") == "exp(3*x)"
    assert s.txt("combine(log(sym(2)) + log(sym(3)), 'log')") == "log(6)"
    assert close(s.dbl("subs(rewrite(sin(x), 'exp'), x, 0.3)"), np.sin(0.3))
    assert s.txt("horner(x^2 + 2*x + 1)") == "x*(x + 2) + 1"


def test_numden_partfrac(s):
    s.run("[nn, dd] = numden(1/x + 1/y);")
    assert s.txt("nn") == "x + y" and s.txt("dd") == "x*y"
    assert s.txt("simplify(partfrac(1/(x^2 - 1)) - 1/(x^2 - 1))") == "0"
    assert "1/(2*(x - 1))" in s.txt("partfrac(1/(x^2 - 1))")


def test_subs(s):
    assert s.txt("subs(x^2 + y, x, 3)") == "y + 9"
    assert s.txt("subs(x^2 + y, {x, y}, {1, 2})") == "3"
    assert s.txt("subs(x^2 + y, [x y], [a b])") == "a^2 + b"
    assert s.txt("subs(x^2, x, [1 2 3])") == "[1, 4, 9]"
    assert s.txt("subs(x + y, x, y)") == "2*y"
    assert s.txt("subs(x*y, {x, y}, {y, x})") == "x*y"
    assert s.txt("subs(x^2 + 1, 2)") == "5"
    assert s.txt("subs(sin(x) + y, sin(x), t)") == "t + y"
    assert s.txt("subs(x^2, x == 3)") == "9"
    s.run("x = 3; syms q; r = subs(q + x);")
    assert s.txt("r") == "q + 3"
    assert s.dbl("subs(q^2 + 1, q, 2)").item() == 5.0


def test_polynomials(s):
    assert s.txt("coeffs(3*x^2 + 2*x + 1)") == "[1, 2, 3]"
    s.run("[c, tt] = coeffs(3*x^2 + 2*x + 1, x);")
    assert s.txt("c") == "[3, 2, 1]" and s.txt("tt") == "[x^2, x, 1]"
    assert s.txt("coeffs(x^3 + 1, x, 'All')") == "[1, 0, 0, 1]"
    assert s.txt("coeffs(a*x^2 + b, x)") == "[b, a]"
    assert s.num("sym2poly(3*x^2 + 2*x + 1)").tolist() == [[3, 2, 1]]
    assert s.txt("poly2sym([1 2 3])") == "x^2 + 2*x + 3"
    assert s.txt("poly2sym([1 0 -1], t)") == "t^2 - 1"
    assert s.num("polynomialDegree(x^3 + x)").item() == 3


def test_equation_parts(s):
    assert s.txt("lhs(x^2 == 4)") == "x^2"
    assert s.txt("rhs(x^2 == 4)") == "4"
    assert s.num("isAlways(sin(x)^2 + cos(x)^2 == 1)").item() is True
    assert s.num("isAlways(x == 1)").item() is False
    s.run("syms w real")
    assert s.num("isAlways(w^2 >= 0)").item() is True
    assert s.num("numel(children(x^2 + y))").item() == 2
    assert s.str("class(children(x^2 + y))") == "cell"
    assert s.txt("finverse(2*x + 3)") == "x/2 - 3/2"


# -- solving ----------------------------------------------------------------------------------

def test_solve_single(s):
    assert s.txt("solve(x^2 - 4 == 0, x)") == "[-2; 2]"
    assert s.txt("solve(x^2 - 4)") == "[-2; 2]"
    assert s.txt("solve(2*x + 6 == 0)") == "-3"
    assert s.txt("solve(x^2 + 1, x)") == "[-1i; 1i]"
    assert s.txt("solve(a*x + b, x)") == "-b/a"
    assert s.txt("solve(x^2 + 1 == 0, x, 'Real', true)") == "[]" or s.num(
        "isempty(solve(x^2 + 1 == 0, x, 'Real', true))").item() is True
    assert close(sorted(s.dbl("solve(x^3 - 6*x^2 + 11*x - 6)").flatten().real), [1, 2, 3])
    assert s.txt("solve(x^3 + x + 1 == 0, x)").startswith("[root(z^3 + z + 1, z, 1);")
    assert close(np.sort_complex(s.dbl("solve(x^3 + x + 1 == 0, x)").flatten()),
                 np.sort_complex(np.roots([1, 0, 1, 1])))
    assert "root" not in s.txt("solve(x^3 + x + 1 == 0, x, 'MaxDegree', 3)")
    s.run("r = solve(a*x^2 + b*x + 1, x);")
    assert s.num("size(r)").tolist() == [[2, 1]]
    assert s.txt("simplify(subs(a*x^2 + b*x + 1, x, r(1)))") == "0"
    assert s.txt("diff(x)") == "1"


def test_solve_systems(s):
    s.run("[sx, sy] = solve([x + y == 3, x - y == 1], [x, y]);")
    assert s.txt("sx") == "2" and s.txt("sy") == "1"
    s.run("S = solve(x + y == 3, x - y == 1);")
    assert s.str("class(S)") == "struct"
    assert s.txt("S.x") == "2" and s.txt("S.y") == "1"
    s.run("[p, q] = solve(x^2 + y^2 == 25, y == x + 1, x, y);")
    got = sorted(zip(s.dbl("p").flatten(), s.dbl("q").flatten()))
    assert close(got, [(-4, -3), (3, 4)])
    s.run("[sol, par, con] = solve(x^2 == 4, x, 'ReturnConditions', true);")
    assert s.txt("sol") == "[-2; 2]"
    assert s.num("isempty(par)").item() is True
    assert s.num("numel(con)").item() == 2
    assert s.txt("subs(x + y, S)") == "3"


def test_vpasolve(s):
    r = s.dbl("vpasolve(x^3 - 2*x - 5, x)")
    assert r.shape == (3, 1) and close(sorted(r.flatten(), key=lambda v: -v.real)[0], 2.0945514815423265)
    assert close(s.dbl("vpasolve(cos(x) == x, x)"), 0.7390851332151607)
    assert close(s.dbl("vpasolve(x^2 == 2, x, [0 5])"), np.sqrt(2))
    assert close(s.dbl("vpasolve(sin(x) == 0, x, 3)"), np.pi)
    s.run("[va, vb] = vpasolve([a + b == 3, a - b == 1], [a b]);")
    assert close(s.dbl("va"), 2) and close(s.dbl("vb"), 1)
    assert len(s.txt("vpasolve(x^2 == 2, x, [0 5])")) > 30  # 32 significant digits


def test_dsolve(s):
    s.run("syms y(t)")
    assert s.txt("dsolve(diff(y, t) == -2*y, y(0) == 1)") == "exp(-2*t)"
    assert s.txt("dsolve(diff(y, t) == y)") == "C1*exp(t)"
    s.run("Dy = diff(y, t);")
    assert s.txt("dsolve(diff(y, t, 2) + y == 0, y(0) == 1, Dy(0) == 0)") == "cos(t)"
    s.run("gen = dsolve(diff(y, t, 2) + 3*diff(y, t) + 2*y == 0);")
    assert s.txt("simplify(diff(gen, t, 2) + 3*diff(gen, t) + 2*gen)") == "0"
    assert s.num("numel(symvar(gen))").item() == 3
    s.run("syms u(t) v(t); [su, sv] = dsolve(diff(u) == v, diff(v) == -u, u(0) == 1, v(0) == 0);")
    assert s.txt("su") == "cos(t)" and s.txt("sv") == "-sin(t)"


# -- transforms -------------------------------------------------------------------------------

def test_laplace(s):
    s.run("syms s")
    assert s.txt("laplace(exp(-a*t))") == "1/(a + s)"
    assert s.txt("laplace(t^2)") == "2/s^3"
    assert s.txt("laplace(sin(2*t))") == "2/(s^2 + 4)"
    assert s.txt("laplace(exp(-x), x, y)") == "1/(y + 1)"
    assert s.txt("ilaplace(1/(s^2 + 1))") == "sin(t)"
    assert s.txt("ilaplace(1/(s + 2))") == "exp(-2*t)"
    assert s.txt("ilaplace(s/(s^2 + 4))") == "cos(2*t)"
    s.run("syms yy(t)")
    assert s.txt("laplace(diff(yy, t))") == "s*laplace(yy(t), t, s) - yy(0)"


def test_fourier_and_z(s):
    s.run("syms w")
    assert s.txt("fourier(exp(-x^2))") == "pi^(1/2)*exp(-w^2/4)"
    assert s.txt("fourier(dirac(x - 1))") == "exp(-w*1i)"
    assert s.txt("ifourier(exp(-w^2/4))") == "exp(-x^2)/pi^(1/2)"
    assert s.txt("simplify(ztrans(a^n) - z/(z - a))") == "0"
    assert s.txt("ztrans(n)") == "z/(z - 1)^2"
    assert s.txt("iztrans(z/(z - 2))") == "2^n"
    assert close(s.dbl("subs(iztrans(z/(z - 1)^2), n, 7)"), 7)


# -- linear algebra ---------------------------------------------------------------------------

def test_linear_algebra(s):
    s.run("A = [a b; 1 2]; N = sym([1 2; 3 4]);")
    assert s.txt("det(A)") == "2*a - b"
    assert s.txt("simplify(inv(A)*A)") == "[1, 0; 0, 1]"
    assert s.txt("trace(A)") == "a + 2"
    assert s.txt("transpose(A)") == "[a, 1; b, 2]"
    assert s.txt("rank(sym([1 2; 2 4]))") == "1"
    assert s.txt("null(sym([1 2; 2 4]))") == "[-2; 1]"
    assert s.txt("rref(sym([1 2 3; 4 5 6]))") == "[1, 0, -1; 0, 1, 2]"
    assert s.txt("eig(sym([2 0; 0 3]))") == "[2; 3]"
    assert s.txt("charpoly(N)") == "[1, -5, -2]"
    assert s.txt("charpoly(N, x)") == "x^2 - 5*x - 2"
    assert s.txt("linsolve(N, sym([5; 6]))") == "[-4; 9/2]"
    assert s.txt("det(N)") == "-2"
    assert s.txt("inv(N)") == "[-2, 1; 3/2, -1/2]"
    assert s.txt("expm(sym([0 1; 0 0]))") == "[1, 1; 0, 1]"
    assert s.txt("diag(N)") == "[1; 4]"
    assert s.txt("adjoint(N)") == "[4, -2; -3, 1]"
    with pytest.raises(MatlabError):
        s.run("inv(sym([1 2; 2 4]))")


def test_eig_vectors(s):
    s.run("B = sym([2 1; 1 2]); [V, D] = eig(B);")
    assert s.txt("simplify(B*V - V*D)") == "[0, 0; 0, 0]"
    assert s.txt("diag(D)") == "[1; 3]"


# -- special functions ------------------------------------------------------------------------

def test_functions_on_syms(s):
    for name in ("sin", "cos", "tan", "exp", "log", "sinh", "atan", "gamma", "erf", "floor", "sign"):
        assert s.txt(f"{name}(x)") == f"{name}(x)"
    assert s.txt("abs(x)") == "abs(x)"
    assert s.txt("sin(sym(pi)/6)") == "1/2"
    assert s.txt("cos(sym(pi))") == "-1"
    assert s.txt("exp(sym(0))") == "1"
    assert s.txt("log(exp(sym(2)))") == "2"
    assert s.txt("gamma(sym(5))") == "24"
    assert s.txt("factorial(sym(20))") == "2432902008176640000"
    assert s.txt("besselj(0, x)") == "besselj(0, x)"
    assert s.txt("atan2(sym(1), sym(1))") == "pi/4"
    assert s.txt("mod(sym(7), 3)") == "1"
    assert s.txt("max(sym([1 5 3]))") == "5"
    assert s.txt("real(sym(1 + 2i))") == "1" and s.txt("imag(sym(1 + 2i))") == "2"
    assert s.txt("nchoosek(sym(5), 2)") == "10"
    assert s.txt("sqrt(sym(8))") == "2*2^(1/2)"


def test_heaviside_dirac_piecewise(s):
    assert s.txt("heaviside(x) + dirac(x)") == "dirac(x) + heaviside(x)"
    assert s.txt("heaviside(sym(0))") == "1/2"
    assert s.num("heaviside([-1 0 2])").tolist() == [[0, 0.5, 1]]
    assert s.num("dirac([0 1])").tolist() == [[np.inf, 0]]
    assert s.txt("diff(heaviside(x))") == "dirac(x)"
    assert s.txt("int(dirac(x - 1)*x^2, -inf, inf)") == "1"
    s.run("pw = piecewise(x < 0, -1, x > 0, 1, 0);")
    assert s.txt("pw") == "piecewise(x < 0, -1, x > 0, 1, 0)"
    assert s.txt("subs(pw, x, -3)") == "-1"
    assert s.txt("subs(pw, x, 0)") == "0"


def test_new_numeric_functions(m):
    assert close(m.num("acot(2)"), np.arctan(0.5))
    assert close(m.num("coth(1)"), 1 / np.tanh(1))
    assert close(m.num("sech([0 1])"), [[1, 1 / np.cosh(1)]])
    assert close(m.num("cbrt(27)"), 3)
    assert m.str("class(acot(2))") == "double"


# -- numbers, text, handles -------------------------------------------------------------------

def test_double_and_vpa(s):
    assert s.dbl("sym(1)/3").item() == pytest.approx(1 / 3)
    assert s.dbl("sym(2)^100").item() == 2.0 ** 100
    assert s.dbl("sym([1 2; 3 4])/2").tolist() == [[0.5, 1], [1.5, 2]]
    assert s.dbl("sqrt(sym(-4))").item() == 2j
    with pytest.raises(MatlabError, match="subs"):
        s.run("double(x + 1)")
    assert s.txt("vpa(pi)") == "3.1415926535897932384626433832795"
    assert s.txt("vpa(sym(1)/3, 10)") == "0.3333333333"
    assert s.txt("vpa(sqrt(sym(2)), 50)") == "1.4142135623730950488016887242096980785696718753769"
    assert s.txt("vpa(x/4)") == "0.25*x"
    s.run("digits(8)")
    assert s.txt("vpa(pi)") == "3.1415927"
    assert s.num("digits").item() == 8
    s.run("digits(32)")
    s.run("vv = [10 20 30];")
    assert s.num("vv(sym(2))").item() == 20.0  # a constant sym works as an index


def test_text_conversions(s):
    assert s.str("char(x^2/3)") == "x^2/3"
    assert s.interp.eval_expression("string(x^2)") == "x^2"
    assert s.str("latex(x^2/2)") == r"\frac{x^{2}}{2}"
    assert "2" in s.run("pretty(x^2/2 + 1)")
    assert s.str("class(char(x))") == "char"
    assert s.str("sprintf('%d and %s', sym(3), x^2)") == "3 and x^2"
    assert s.str("num2str(sym(1)/4)") == "0.25"
    assert s.str("['f = ' char(x + 1)]") == "f = x + 1"
    s.run("x = 2;")
    assert s.num("eval(str2sym('x^2 + 1'))").item() == 5.0


def test_matlab_function(s):
    s.run("f1 = matlabFunction(x^2 + 2*x + 1);")
    assert s.str("class(f1)") == "function_handle"
    assert s.num("f1([1 2 3])").tolist() == [[4, 9, 16]]
    s.run("f2 = matlabFunction(x^2 + y, 'Vars', [y x]);")
    assert s.num("f2(1, 2)").item() == 5
    assert s.str("func2str(f2)").startswith("@(y,x)")
    s.run("f3 = matlabFunction(exp(-t)*sin(3*t)/t + sqrt(t) + pi);")
    tt = np.array([[0.5, 1.0, 2.0]])
    assert close(s.num("f3([0.5 1 2])"), np.exp(-tt) * np.sin(3 * tt) / tt + np.sqrt(tt) + np.pi)
    s.run("f4 = matlabFunction(x*y^2);")
    assert s.num("f4([1 2], [3 4])").tolist() == [[9, 32]]
    s.run("f5 = matlabFunction(piecewise(x < 0, -x, x >= 0, x^2));")
    assert s.num("f5([-2 -1 0 1 2])").tolist() == [[2, 1, 0, 1, 4]]
    s.run("g(x, y) = x - y; f6 = matlabFunction(g);")
    assert s.num("f6(5, 2)").item() == 3
    s.run("f7 = matlabFunction([x; x^2]);")
    assert s.num("f7(3)").tolist() == [[3], [9]]
    with pytest.raises(MatlabError):
        s.run("matlabFunction(x + y, 'Vars', x)")


def test_plotting_with_syms(s):
    s.run("figure; fplot(sin(x)/x, [-5 5]); hold on; fplot(sym(1));")
    s.run("figure; fsurf(x^2 - y^2); figure; fcontour(sin(x)*cos(y));")
    s.run("figure; ezplot(x^2); f(x) = x^3; figure; fplot(f, [0 1]);")
    s.run("figure; fimplicit(x^2 + y^2 == 4); close all")


def test_errors_are_clear(s):
    with pytest.raises(MatlabError, match="scalar"):
        s.run("diff(x^2, [x y; y x], 2); limit([x y], [x y], 0)")
    with pytest.raises(MatlabError):
        s.run("syms 1abc")
    with pytest.raises(MatlabError, match="syms y"):
        s.run("dsolve(x == 1)")


def test_missing_sympy_message(m, monkeypatch):
    import sys

    from mylab.builtins import symbolic
    monkeypatch.setattr(symbolic, "_sympy", None)
    monkeypatch.setitem(sys.modules, "sympy", None)
    for code in ("syms x", "x = sym('x')", "solve('x^2 - 1')", "vpa(2)", "int(3)"):
        with pytest.raises(MatlabError, match=r'pip install "mylab-lang\[symbolic\]"'):
            m.run(code)
    assert m.num("heaviside(2)").item() == 1.0  # numeric helpers do not need SymPy
    assert close(m.num("sin(1)"), np.sin(1))
