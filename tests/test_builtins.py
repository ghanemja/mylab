"""Built-in function library."""

import os

import numpy as np
import pytest

from mylab import MatlabError


def test_constructors(m):
    assert m.num("zeros(2, 3)").shape == (2, 3)
    assert m.num("ones(3)").shape == (3, 3)
    np.testing.assert_array_equal(m.num("eye(2)"), np.eye(2))
    assert m.num("size(rand(4, 2))").tolist() == [[4, 2]]
    np.testing.assert_allclose(m.num("linspace(0, 1, 5)"), [[0, 0.25, 0.5, 0.75, 1]])
    np.testing.assert_array_equal(m.num("repmat([1 2], 2, 2)"), [[1, 2, 1, 2], [1, 2, 1, 2]])
    np.testing.assert_array_equal(m.num("reshape(1:6, 2, 3)"), [[1, 3, 5], [2, 4, 6]])
    np.testing.assert_array_equal(m.num("reshape(1:6, [], 2)"), [[1, 4], [2, 5], [3, 6]])
    for n in (3, 4, 5, 6, 8):
        M = m.num(f"magic({n})")
        target = n * (n * n + 1) / 2
        assert set(M.flatten()) == set(range(1, n * n + 1))
        assert np.all(M.sum(0) == target) and np.all(M.sum(1) == target)
        assert np.trace(M) == target and np.trace(np.fliplr(M)) == target


def test_size_queries(m):
    m.run("A = zeros(3, 5);")
    m.run("[r, c] = size(A);")
    assert (m.val("r").item(), m.val("c").item()) == (3, 5)
    assert m.num("size(A, 2)").item() == 5
    assert m.num("numel(A)").item() == 15
    assert m.num("length(A)").item() == 5
    assert m.num("isempty([])").item()
    assert m.num("isscalar(5)").item()


def test_reductions(m):
    m.run("A = [1 2; 3 4];")
    np.testing.assert_array_equal(m.num("sum(A)"), [[4, 6]])
    np.testing.assert_array_equal(m.num("sum(A, 2)"), [[3], [7]])
    assert m.num("sum(A(:))").item() == 10
    assert m.num("sum(A, 'all')").item() == 10
    np.testing.assert_array_equal(m.num("prod(A)"), [[3, 8]])
    np.testing.assert_array_equal(m.num("cumsum(1:4)"), [[1, 3, 6, 10]])
    assert m.num("mean([1 2 3 4])").item() == 2.5
    assert m.num("median([3 1 2])").item() == 2
    assert m.num("std([2 4 4 4 5 5 7 9])").item() == pytest.approx(2.13809, rel=1e-4)
    assert m.num("var([1 2 3 4])").item() == pytest.approx(1.6667, rel=1e-4)
    assert m.num("mode([1 2 2 3])").item() == 2
    assert m.num("sum([])").item() == 0


def test_min_max(m):
    m.run("[v, i] = max([3 7 2 7]);")
    assert (m.val("v").item(), m.val("i").item()) == (7, 2)
    np.testing.assert_array_equal(m.num("max([1 5; 7 2])"), [[7, 5]])
    np.testing.assert_array_equal(m.num("max([1 5; 7 2], [], 2)"), [[5], [7]])
    np.testing.assert_array_equal(m.num("min([1 5 3], 2)"), [[1, 2, 2]])
    assert m.num("max([NaN 1 2])").item() == 2


def test_sort_find_unique(m):
    m.run("[s, i] = sort([3 1 2], 'descend');")
    np.testing.assert_array_equal(m.val("s"), [[3, 2, 1]])
    np.testing.assert_array_equal(m.val("i"), [[1, 3, 2]])
    np.testing.assert_array_equal(m.num("find([0 1 0 1])"), [[2, 4]])
    np.testing.assert_array_equal(m.num("find([0; 1; 1])"), [[2], [3]])
    np.testing.assert_array_equal(m.num("find([1 1 1], 2)"), [[1, 2]])
    np.testing.assert_array_equal(m.num("unique([3 1 2 3 1])"), [[1, 2, 3]])
    np.testing.assert_array_equal(m.num("unique([3 1 2 3 1], 'stable')"), [[3, 1, 2]])
    np.testing.assert_array_equal(m.num("ismember([1 5], [1 2 3])"), [[True, False]])
    np.testing.assert_array_equal(m.num("setdiff([1 2 3 4], [2 4])"), [[1, 3]])
    np.testing.assert_array_equal(m.num("intersect([1 2 3], [2 3 4])"), [[2, 3]])


def test_elementwise_math(m):
    assert m.num("round(2.5)").item() == 3
    assert m.num("round(-2.5)").item() == -3
    assert m.num("mod(-1, 3)").item() == 2
    assert m.num("rem(-1, 3)").item() == -1
    assert m.num("mod(5, 0)").item() == 5
    assert m.num("fix(-2.7)").item() == -2
    assert m.num("factorial(5)").item() == 120
    assert m.num("nchoosek(5, 2)").item() == 10
    np.testing.assert_array_equal(m.num("primes(20)"), [[2, 3, 5, 7, 11, 13, 17, 19]])
    assert m.num("gcd(12, 18)").item() == 6
    assert m.num("sind(30)").item() == pytest.approx(0.5)
    assert m.num("sind(180)").item() == 0
    assert m.num("exp(log(5))").item() == pytest.approx(5)
    assert m.num("hypot(3, 4)").item() == 5


def test_linear_algebra(m):
    m.run("A = [4 1; 2 3];")
    assert m.num("det(A)").item() == pytest.approx(10)
    np.testing.assert_allclose(m.num("inv(A) * A"), np.eye(2), atol=1e-12)
    np.testing.assert_allclose(np.sort(m.num("eig(A)").flatten()), [2, 5])
    m.run("[V, D] = eig(A);")
    np.testing.assert_allclose(m.num("A * V - V * D"), np.zeros((2, 2)), atol=1e-12)
    assert m.num("rank(magic(4))").item() == 3
    assert m.num("norm([3 4])").item() == 5
    assert m.num("trace(A)").item() == 7
    m.run("[U, S, W] = svd(A);")
    np.testing.assert_allclose(m.num("U * S * W'"), [[4, 1], [2, 3]], atol=1e-12)
    m.run("[Q, R] = qr(A);")
    np.testing.assert_allclose(m.num("Q * R"), [[4, 1], [2, 3]], atol=1e-12)
    np.testing.assert_allclose(m.num("kron([1 2], [1; 1])"), [[1, 2], [1, 2]])
    assert m.num("dot([1 2 3], [4 5 6])").item() == 32
    np.testing.assert_array_equal(m.num("cross([1 0 0], [0 1 0])"), [[0, 0, 1]])


def test_strings(m):
    assert m.str("upper('abc')") == "ABC"
    assert m.str("strrep('hello world', 'o', '0')") == "hell0 w0rld"
    assert m.str("strtrim('  pad  ')") == "pad"
    assert m.num("strcmp('a', 'a')").item()
    assert not m.num("strcmp('a', 'b')").item()
    np.testing.assert_array_equal(m.num("strcmp({'a', 'b'}, 'a')"), [[True, False]])
    assert m.num("strcmpi('ABC', 'abc')").item()
    np.testing.assert_array_equal(m.num("strfind('abcabc', 'bc')"), [[2, 5]])
    m.run("parts = strsplit('a,b,c', ',');")
    assert m.num("numel(parts)").item() == 3
    assert m.str("parts{2}") == "b"
    assert m.str("num2str(pi)") == "3.1416"
    assert m.str("num2str(42)") == "42"
    assert m.str("num2str([1 2 3])") == "1  2  3"
    assert m.str("mat2str([1 2; 3 4])") == "[1 2;3 4]"
    assert m.num("str2double('3.5')").item() == 3.5
    assert np.isnan(m.num("str2double('abc')").item())
    assert m.num("str2num('[1 2 3]')").shape == (1, 3)
    assert m.str("lower(\"MiXeD\")") == "mixed"
    assert m.num("contains('hello', 'ell')").item()
    assert m.num("startsWith('hello', 'he')").item()
    assert m.str("regexprep('abc123', '\\d', 'X')") == "abcXXX"
    m.run("tok = regexp('key=value', '(\\w+)=(\\w+)', 'tokens');")
    assert m.str("tok{1}{2}") == "value"
    np.testing.assert_array_equal(m.num("double('AB')"), [[65, 66]])
    assert m.str("char([72 105])") == "Hi"
    assert m.str("\"abc\" + \"def\"") == "abcdef"


def test_sprintf(m):
    assert m.str("sprintf('%d-%d', 1, 2)") == "1-2"
    assert m.str("sprintf('%5.2f', pi)") == " 3.14"
    assert m.str("sprintf('%d,', [1 2 3])") == "1,2,3,"
    assert m.str("sprintf('%s is %d', 'x', 5)") == "x is 5"
    assert m.str("sprintf('%d', 1.5)") == "1.500000e+00"
    assert m.str("sprintf('100%%')") == "100%"
    assert m.str("sprintf('%g', 0.0001)") == "0.0001"
    assert m.str("sprintf('%x', 255)") == "ff"
    assert m.str("sprintf('%d %s\\n', 1, 'a')") == "1 a\n"
    assert m.str("sprintf('%f', Inf)") == "Inf"


def test_fprintf_and_disp(m):
    assert m.run("fprintf('Hello %s!\\n', 'world');") == "Hello world!\n"
    assert m.run("fprintf('%d\\n', [1 2 3]);") == "1\n2\n3\n"
    assert m.run("disp(5)") == "     5\n"
    assert m.run("disp('text')") == "text\n"
    assert m.run("disp([1 2])") == "     1     2\n"


def test_cellfun_arrayfun(m):
    np.testing.assert_array_equal(m.num("cellfun(@numel, {'a', 'bcd', ''})"), [[1, 3, 0]])
    np.testing.assert_array_equal(m.num("cellfun('isempty', {1, [], 'x'})"), [[False, True, False]])
    m.run("u = cellfun(@(s) [s '!'], {'a', 'b'}, 'UniformOutput', false);")
    assert m.str("u{2}") == "b!"
    np.testing.assert_array_equal(m.num("arrayfun(@(x) x^2, [1 2 3])"), [[1, 4, 9]])
    m.run("[mn, mx] = cellfun(@(v) deal(min(v), max(v)), {[1 5], [2 9]});")
    np.testing.assert_array_equal(m.val("mx"), [[5, 9]])


def test_struct_functions(m):
    m.run("s = struct('a', 1, 'b', 'x');")
    m.run("f = fieldnames(s);")
    assert m.str("f{2}") == "b"
    m.run("s = rmfield(s, 'a');")
    assert not m.num("isfield(s, 'a')").item()
    m.run("c = struct2cell(struct('p', 1, 'q', 2));")
    assert m.num("c{2}").item() == 2
    np.testing.assert_array_equal(m.num("cell2mat({[1 2], [3]; [4 5], [6]})"), [[1, 2, 3], [4, 5, 6]])


def test_polynomials_and_interp(m):
    np.testing.assert_allclose(m.num("polyfit([0 1 2], [1 3 5], 1)"), [[2, 1]], atol=1e-12)
    assert m.num("polyval([1 0 -1], 3)").item() == 8
    np.testing.assert_allclose(np.sort(m.num("roots([1 -3 2])").flatten()), [1, 2])
    np.testing.assert_allclose(m.num("conv([1 1], [1 -1])"), [[1, 0, -1]])
    np.testing.assert_allclose(m.num("interp1([1 2 3], [10 20 30], 2.5)"), [[25]])
    assert np.isnan(m.num("interp1([1 2], [1 2], 5)").item())
    assert m.num("trapz([0 1 2], [0 1 4])").item() == pytest.approx(3)
    np.testing.assert_array_equal(m.num("diff([1 4 9 16])"), [[3, 5, 7]])


def test_solvers(m):
    assert m.num("fzero(@(x) x^2 - 2, [0 2])").item() == pytest.approx(np.sqrt(2))
    assert m.num("fzero(@cos, 1)").item() == pytest.approx(np.pi / 2)
    x = m.num("fminsearch(@(p) (p(1)-1)^2 + (p(2)+2)^2, [0 0])")
    np.testing.assert_allclose(x, [[1, -2]], atol=1e-3)
    assert m.num("integral(@(x) x.^2, 0, 3)").item() == pytest.approx(9)
    m.run("[t, y] = ode45(@(t, y) -y, [0 1], 1);")
    assert m.num("y(end)").item() == pytest.approx(np.exp(-1), rel=1e-3)
    assert m.num("size(t, 2)").item() == 1


def test_eval_and_feval(m):
    m.run("eval('q = 6 * 7;');")
    assert m.val("q").item() == 42
    assert m.num("feval('max', [1 9 3])").item() == 9
    m.run("fh = str2func('@(x) x + 1');")
    assert m.num("fh(1)").item() == 2
    assert m.str("func2str(@(a) a * 2)") == "@(a)a*2"
    out = m.run("eval('bad syntax here (', 'disp(''fallback'')')")
    assert out == "fallback\n"


def test_type_functions(m):
    assert m.str("class(1)") == "double"
    assert m.str("class('a')") == "char"
    assert m.str("class({})") == "cell"
    assert m.str("class(true)") == "logical"
    assert m.str("class(@sin)") == "function_handle"
    assert m.num("isnumeric(5)").item()
    assert not m.num("isnumeric('5')").item()
    assert m.num("isa(5, 'numeric')").item()
    assert m.num("isequal([1 2], [1 2])").item()
    assert not m.num("isequal([1 2], [1 2 3])").item()


def test_exist_and_workspace(m):
    m.run("v = 1;")
    assert m.num("exist('v')").item() == 1
    assert m.num("exist('sin')").item() == 5
    assert m.num("exist('no_such_thing_123')").item() == 0
    out = m.run("who")
    assert "v" in out


def test_assert_and_error(m):
    m.run("assert(true)")
    with pytest.raises(MatlabError, match="Assertion failed"):
        m.run("assert(1 == 2)")
    with pytest.raises(MatlabError, match="custom 3"):
        m.run("assert(false, 'custom %d', 3)")


def test_plotting_saves_file(m, tmp_path):
    out = tmp_path / "plot.png"
    m.run(f"""
x = linspace(0, 2*pi, 50);
figure;
plot(x, sin(x), 'r-', 'LineWidth', 2);
hold on
plot(x, cos(x), 'b--');
xlabel('x'); ylabel('y'); title('Trig'); legend('sin', 'cos'); grid on
saveas(gcf, '{out}');
close all
""")
    assert out.exists() and os.path.getsize(out) > 1000


def test_subplot_and_other_plots(m, tmp_path):
    out = tmp_path / "multi.png"
    m.run(f"""
subplot(2, 2, 1); bar([1 2 3]);
subplot(2, 2, 2); scatter(rand(1, 10), rand(1, 10), 'filled');
subplot(2, 2, 3); histogram(randn(1, 100));
subplot(2, 2, 4); [X, Y] = meshgrid(-2:0.5:2); contour(X, Y, X.^2 + Y.^2);
print('-dpng', '{out}');
close all
""")
    assert out.exists()


def test_file_io(m, tmp_path):
    p = tmp_path / "data.txt"
    m.run(f"""
fid = fopen('{p}', 'w');
fprintf(fid, 'line %d\\n', 1:3);
fclose(fid);
fid = fopen('{p}', 'r');
first = fgetl(fid);
fclose(fid);
""")
    assert m.str("first") == "line 1"
    csv = tmp_path / "m.csv"
    m.run(f"writematrix([1 2; 3 4], '{csv}'); R = readmatrix('{csv}');")
    np.testing.assert_array_equal(m.val("R"), [[1, 2], [3, 4]])


def test_rng_is_reproducible(m):
    m.run("rng(42); a = rand(1, 3); rng(42); b = rand(1, 3);")
    np.testing.assert_array_equal(m.val("a"), m.val("b"))
    m.run("r = randi(10, 1, 100);")
    r = m.val("r")
    assert r.min() >= 1 and r.max() <= 10


def test_tic_toc(m):
    m.run("t0 = tic; e = toc(t0);")
    assert m.val("e").item() >= 0
    assert m.run("tic; toc").startswith("Elapsed time is")
