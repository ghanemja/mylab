"""The loop compiler must give exactly the same results and errors as the interpreter."""

import random

import numpy as np
import pytest

from mylab import Interpreter, MatlabError
from mylab.values import values_equal

CASES = {
    "sum_squares": "s = 0; for k = 1:500, s = s + k^2; end",
    "fill_array": "x = zeros(1, 200); for k = 1:200, x(k) = k * 2 - 1; end",
    "column_fill": "x = zeros(50, 1); for k = 1:50, x(k) = sqrt(k); end",
    "matrix_2d": "A = zeros(12, 9); for i = 1:12, for j = 1:9, A(i, j) = i * 10 + j; end, end",
    "matrix_linear": "A = zeros(4, 5); for k = 1:20, A(k) = k; end",
    "read_and_end": "v = (1:30).^2; t = 0; for k = 1:30, t = t + v(k) - v(end) / numel(v); end",
    "while_collatz": "n = 27; steps = 0; while n ~= 1, if mod(n, 2) == 0, n = n / 2; else, n = 3 * n + 1; end, "
                     "steps = steps + 1; end",
    "break_continue": "c = 0; last = 0; for k = 1:100, if mod(k, 3) == 0, continue; end, if k > 50, break; end, "
                      "c = c + 1; last = k; end",
    "elseif_chain": "a = 0; b = 0; c = 0; for k = 1:60, if k < 20, a = a + 1; elseif k < 40, b = b + k; else, "
                    "c = c - 1; end, end",
    "logical_vars": "flag = false; cnt = 0; for k = 1:40, flag = k > 20 && mod(k, 2) == 0; if flag, cnt = cnt + 1; "
                    "end, end",
    "negative_step": "s = 0; for k = 10:-0.5:1, s = s + k; end",
    "fractional_step": "s = 0; n = 0; for t = 0:0.1:1, s = s + t; n = n + 1; end",
    "math_functions": "s = 0; for k = 1:50, s = s + sin(k) * cos(k / 2) + exp(-k / 10) + abs(k - 25) + floor(k / 3) "
                      "+ round(k / 4) + max(k, 30) - min(k, 7) + sign(k - 10) + rem(k, 4) + atan2(k, 3); end",
    "newton": "x = 1; for it = 1:30, x = x - (x^3 - 2 * x - 5) / (3 * x^2 - 2); end",
    "bubble_sort": "v = [5 3 9 1 7 2 8 6 4 0]; n = numel(v); for i = 1:n, for j = 1:n - i, if v(j) > v(j + 1), "
                   "t = v(j); v(j) = v(j + 1); v(j + 1) = t; end, end, end",
    "copy_on_write": "x = zeros(1, 10); y = x; for k = 1:10, y(k) = k; end",
    "loop_var_after": "for k = 3:8, q = k; end",
    "few_iterations": "s = 0; for k = 1:2, s = s + k; end",
    "empty_range": "s = 5; for k = 1:0, s = s + k; end",
    "pi_constant": "s = 0; for k = 1:10, s = s + pi * k; end",
    "temp_inside": "s = 0; for k = 1:20, t = k * 2; u = t + 1; s = s + u; end",
    # --- cases that must bail out and still match the interpreter ---
    "division_by_zero": "s = 0; for k = -5:5, s = s + 1 / k^2 * (k ~= 0) + (k == 0); r = 1 / k; end",
    "complex_result": "s = 0; for k = -3:6, s = s + sqrt(k); end",
    "complex_power": "s = 0; for k = -2:5, s = s + k^0.5; end",
    "growth": "x = zeros(1, 5); for k = 1:12, x(k) = k; end",
    "end_plus_one": "x = 1; y = zeros(1, 3); for k = 1:8, y(k) = k; x = x + y(min(k, 3)); end",
    "bail_then_break": "x = zeros(1, 4); c = 0; for k = 1:20, x(k) = k; c = c + 1; if k == 9, break; end, end",
    "bail_in_if": "x = zeros(1, 4); for k = 1:10, if k > 2, x(k) = k; else, x(1) = -1; end, end",
    "while_bail": "x = zeros(1, 3); k = 0; while k < 9, k = k + 1; x(k) = k^2; end",
    "log_zero": "s = 0; for k = 0:5, s = s + log(k + (k == 0) * 0); end",
    "int_array": "x = zeros(1, 6, 'int32'); for k = 1:6, x(k) = k * 2.6; end",
    "logical_array": "m = false(1, 8); for k = 1:8, m(k) = mod(k, 2) == 0; end",
    "char_values": "s = 'hello'; n = 0; for k = 1:5, n = n + s(k); end",
    "calls_user_function": "function r = twice(a)\nr = 2 * a;\nend\ns = 0; for k = 1:10, s = s + twice(k); end",
    "shadowed_mod": "function r = mod(a, b)\nr = 100;\nend\ns = 0; for k = 1:10, s = s + mod(k, 3); end",
    "scalar_becomes_vector": "x = 0; for k = 1:6, x(k) = k; end",
    "display_in_loop": "s = 0; for k = 1:5, s = s + k, end",
}

ERROR_CASES = {
    "index_out_of_range": "v = [1 2 3]; s = 0; for k = 1:10, s = s + v(k); end",
    "bad_index": "v = [1 2 3]; for k = 1:10, t = v(k / 2); end",
    "nan_condition": "x = NaN; c = 0; for k = 1:10, if x, c = c + 1; end, end",
    "undefined_variable": "for k = 1:10, s = s + k; end",
    "undefined_function": "s = 0; for k = 1:10, s = s + nosuchfn(k); end",
}


def run(code: str, jit: bool):
    out: list[str] = []
    interp = Interpreter(out=out.append)
    interp.jit_enabled = jit
    error = None
    try:
        interp.execute(code)
    except MatlabError as e:
        error = e.message
    return interp, "".join(out), error


def same_workspace(a: Interpreter, b: Interpreter) -> None:
    assert sorted(a.base.vars) == sorted(b.base.vars)
    for name, value in a.base.vars.items():
        other = b.base.vars[name]
        assert type(value) is type(other), name
        if isinstance(value, np.ndarray):
            assert value.dtype == other.dtype, f"{name}: {value.dtype} vs {other.dtype}"
            assert value.shape == other.shape, name
            if value.dtype.kind in "fc":
                # math-library and NumPy scalar functions may differ in the last bit on some CPUs
                assert np.allclose(value, other, rtol=1e-12, atol=0, equal_nan=True), name
            else:
                assert np.array_equal(value, other), name
        else:
            assert values_equal(value, other), name


@pytest.mark.parametrize("name", sorted(CASES))
def test_same_result_with_and_without_compiler(name):
    on, out_on, err_on = run(CASES[name], True)
    off, out_off, err_off = run(CASES[name], False)
    assert err_on == err_off
    assert out_on == out_off
    same_workspace(on, off)


@pytest.mark.parametrize("name", sorted(ERROR_CASES))
def test_same_error_and_state(name):
    on, out_on, err_on = run(ERROR_CASES[name], True)
    off, out_off, err_off = run(ERROR_CASES[name], False)
    assert err_off is not None and err_on == err_off
    same_workspace(on, off)


def test_compiler_is_actually_used():
    from mylab.parser import parse
    interp = Interpreter(out=lambda s: None)
    program = parse("s = 0; for k = 1:100, s = s + k; end")
    interp.run_program(program, interp.base)
    loop = program.body[1]
    assert loop._jit_static and any(v is not None for v in loop._jit_static["compiled"].values())
    assert interp.get_var("s").item() == 5050


def test_random_expression_loops_match():
    rng = random.Random(1234)
    atoms = ["k", "a", "b", "v(mod(k, 5) + 1)", "2", "0.5", "3", "pi"]
    ops = ["+", "-", "*", "/", "^"]
    funcs = ["sin(%s)", "abs(%s)", "floor(%s)", "sqrt(%s)", "mod(%s, 3)", "max(%s, 1)", "round(%s)", "(%s)"]

    def expr(depth: int) -> str:
        if depth == 0 or rng.random() < 0.3:
            return rng.choice(atoms)
        if rng.random() < 0.35:
            return rng.choice(funcs) % expr(depth - 1)
        return f"({expr(depth - 1)} {rng.choice(ops)} {expr(depth - 1)})"

    for _ in range(150):
        code = (f"a = 1.5; b = -2; v = [3 1 4 1 5]; s = 0; w = zeros(1, 12); "
                f"for k = 1:12, a = {expr(3)}; if a > b, s = s + a; else, b = {expr(2)}; end, w(k) = s; end")
        on, _, err_on = run(code, True)
        off, _, err_off = run(code, False)
        assert err_on == err_off, code
        try:
            same_workspace(on, off)
        except AssertionError as e:
            raise AssertionError(f"{code}\n{e}") from None


def test_in_place_temporaries_do_not_change_variables():
    interp = Interpreter(out=lambda s: None)
    interp.execute("a = (1:200000) / 7; b = a + 1; c = a; a0 = a; b0 = b;"
                   "y = a .* b + c .* 2 - a ./ b; z = 2 + (a + b); w = (a + 0) + b; f = @(t) t; q = f(a) + 1;")
    ev = lambda code: np.asarray(interp.eval_expression(code))  # noqa: E731
    assert ev("isequal(a, a0) && isequal(b, b0) && isequal(c, a0)").item()
    a = np.arange(1, 200001) / 7
    b = a + 1
    np.testing.assert_allclose(ev("y"), (a * b + a * 2 - a / b).reshape(1, -1))
    np.testing.assert_allclose(ev("z"), (2 + (a + b)).reshape(1, -1))
    np.testing.assert_allclose(ev("q"), (a + 1).reshape(1, -1))
    assert ev("isequal(a, a0)").item()
