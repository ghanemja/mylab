"""Command-window display: the layout MATLAB prints for each kind of value.

The expected texts were written by hand from the layout rules (blank line, ``name =``, blank
line, value, blank line with ``format loose``; column widths per numeric format).
"""

import pytest


def show(m, code):
    return m.run(code)


# -- scalars ------------------------------------------------------------------------------------

@pytest.mark.parametrize("code, body", [
    ("x = 5", "     5"),
    ("x = -5", "    -5"),
    ("x = 100", "   100"),
    ("x = 1000", "        1000"),
    ("x = 65536", "       65536"),
    ("x = 123456789", "   123456789"),
    ("x = 1234567890", "   1.2346e+09"),
    ("x = pi", "    3.1416"),
    ("x = -0.5", "   -0.5000"),
    ("x = 100.5", "  100.5000"),
    ("x = 1000.5", "   1.0005e+03"),
    ("x = 0.001", "   1.0000e-03"),
    ("x = 0.002", "    0.0020"),
    ("x = 1e-20", "   1.0000e-20"),
    ("x = NaN", "   NaN"),
    ("x = Inf", "   Inf"),
    ("x = -Inf", "  -Inf"),
    ("x = 0", "     0"),
])
def test_real_scalars_format_short(m, code, body):
    assert show(m, code) == f"\nx =\n\n{body}\n\n"


@pytest.mark.parametrize("code, body", [
    ("x = pi", "   3.141592653589793"),
    ("x = 10.5", "  10.500000000000000"),
    ("x = 100.5", "     1.005000000000000e+02"),
    ("x = 5", "     5"),
    ("x = 1/3", "   0.333333333333333"),
])
def test_real_scalars_format_long(m, code, body):
    m.run("format long")
    assert show(m, code) == f"\nx =\n\n{body}\n\n"


def test_other_numeric_formats(m):
    m.run("format shortE")
    assert show(m, "x = pi") == "\nx =\n\n   3.1416e+00\n\n"
    assert show(m, "x = [pi 100]") == "\nx =\n\n   3.1416e+00   1.0000e+02\n\n"
    assert show(m, "x = 5") == "\nx =\n\n     5\n\n"
    m.run("format longE")
    assert show(m, "x = pi") == "\nx =\n\n     3.141592653589793e+00\n\n"
    m.run("format shortG")
    assert show(m, "x = pi") == "\nx =\n\n       3.1416\n\n"
    assert show(m, "x = [pi 1e5 1e-5]") == "\nx =\n\n       3.1416        1e+05        1e-05\n\n"
    assert show(m, "x = 123456") == "\nx =\n\n      123456\n\n"
    m.run("format longG")
    assert show(m, "x = pi") == "\nx =\n\n          3.14159265358979\n\n"
    m.run("format bank")
    assert show(m, "x = pi") == "\nx =\n\n         3.14\n\n"
    m.run("format hex")
    assert show(m, "x = 1") == "\nx =\n\n   3ff0000000000000\n\n"
    m.run("format rat")
    assert "355/113" in show(m, "x = pi")
    m.run("format")
    assert show(m, "x = pi") == "\nx =\n\n    3.1416\n\n"


def test_format_command_forms(m):
    m.run("format long; format compact")
    assert m.str("string(format().NumericFormat)") == "long"
    m.run("fmt = format;")
    assert m.str("string(fmt.LineSpacing)") == "compact"
    m.run("format")  # back to short and loose
    assert show(m, "x = 1") == "\nx =\n\n     1\n\n"
    m.run("format(fmt)")
    assert show(m, "x = pi") == "x =\n   3.141592653589793\n"
    m.run("format short g")
    assert m.str("string(format().NumericFormat)") == "shortG"
    m.run("format('long', 'e'); format loose")
    assert show(m, "x = 1.5") == "\nx =\n\n     1.500000000000000e+00\n\n"
    from mylab.errors import MatlabError
    with pytest.raises(MatlabError):
        m.run("format nonsense")


def test_compact_spacing(m):
    m.run("format compact")
    assert show(m, "x = 5") == "x =\n     5\n"
    assert show(m, "A = [1 2; 3 4]") == "A =\n     1     2\n     3     4\n"
    assert show(m, "t = true") == "t =\n  logical\n   1\n"
    assert show(m, "s = struct('a', 1)") == "s = \n  struct with fields:\n    a: 1\n"
    assert show(m, "c = {1, 'a'}") == "c =\n  1×2 cell array\n    {[1]}    {'a'}\n"
    assert show(m, "x = [1000.5 1]") == "x =\n   1.0e+03 *\n    1.0005    0.0010\n"
    m.run("format loose")
    assert show(m, "x = 5") == "\nx =\n\n     5\n\n"


# -- matrices -------------------------------------------------------------------------------------

def test_integer_valued_matrices(m):
    assert show(m, "A = [1 2; 3 4]") == "\nA =\n\n     1     2\n     3     4\n\n"
    assert show(m, "A = [1 -2 3]") == "\nA =\n\n     1    -2     3\n\n"
    assert show(m, "A = [1000 2000]") == "\nA =\n\n        1000        2000\n\n"
    assert show(m, "A = [-100 -220]") == "\nA =\n\n  -100  -220\n\n"
    assert show(m, "A = [NaN 1 Inf]") == "\nA =\n\n   NaN     1   Inf\n\n"
    assert show(m, "A = [1; 20; 300]") == "\nA =\n\n     1\n    20\n   300\n\n"


def test_fraction_matrices_and_scale_factor(m):
    assert show(m, "A = [1.5 2.5; 3 4]") == "\nA =\n\n    1.5000    2.5000\n    3.0000    4.0000\n\n"
    assert show(m, "A = [0 0.5]") == "\nA =\n\n         0    0.5000\n\n"
    assert show(m, "A = [NaN 1.5]") == "\nA =\n\n       NaN    1.5000\n\n"
    assert show(m, "A = [1000.5 1]") == "\nA =\n\n   1.0e+03 *\n\n    1.0005    0.0010\n\n"
    assert show(m, "A = [1e-5 2e-5]") == "\nA =\n\n   1.0e-04 *\n\n    0.1000    0.2000\n\n"
    assert show(m, "A = [1e10 1]") == "\nA =\n\n   1.0e+10 *\n\n    1.0000    0.0000\n\n"
    assert show(m, "A = [0.5; 100.25; -3]") == "\nA =\n\n    0.5000\n  100.2500\n   -3.0000\n\n"
    m.run("format long")
    assert show(m, "A = [1.5 2.5]") == "\nA =\n\n   1.500000000000000   2.500000000000000\n\n"


def test_wide_matrices_are_split_into_column_blocks(m):
    out = show(m, "x = 0:0.1:1")
    assert out == ("\nx =\n\n  Columns 1 through 7\n\n"
                   "         0    0.1000    0.2000    0.3000    0.4000    0.5000    0.6000\n\n"
                   "  Columns 8 through 11\n\n"
                   "    0.7000    0.8000    0.9000    1.0000\n\n")
    out = show(m, "y = 1:20")
    assert "  Columns 1 through 13\n\n" in out and "  Columns 14 through 20\n\n" in out
    out = show(m, "z = (1:14) * 1000")
    assert "  Columns 1 through 6\n" in out and "  Columns 13 through 14\n" in out
    out = show(m, "w = 1:14")
    assert "  Column 14\n\n    14\n\n" in out
    m.run("format compact")
    assert show(m, "x = 1:14").startswith("x =\n  Columns 1 through 13\n     1     2")


def test_complex_numbers(m):
    assert show(m, "z = 3 + 4i") == "\nz =\n\n   3.0000 + 4.0000i\n\n"
    assert show(m, "z = 1i") == "\nz =\n\n   0.0000 + 1.0000i\n\n"
    assert show(m, "z = [1+2i 3-4i]") == "\nz =\n\n   1.0000 + 2.0000i   3.0000 - 4.0000i\n\n"
    assert show(m, "z = [1+2i; 10.5-20.25i]") == "\nz =\n\n   1.0000 + 2.0000i\n  10.5000 -20.2500i\n\n"
    assert show(m, "z = 1e3 + 2e3i") == "\nz =\n\n   1.0000e+03 + 2.0000e+03i\n\n"
    assert show(m, "z = -1.5 - 2.5i") == "\nz =\n\n  -1.5000 - 2.5000i\n\n"
    assert show(m, "z = complex(3, 0)") == "\nz =\n\n   3.0000 + 0.0000i\n\n"


def test_integer_logical_and_single_class_headers(m):
    assert show(m, "t = true") == "\nt =\n\n  logical\n\n   1\n\n"
    assert show(m, "t = [true false]") == "\nt =\n\n  1×2 logical array\n\n   1   0\n\n"
    assert show(m, "k = int8(5)") == "\nk =\n\n  int8\n\n   5\n\n"
    assert show(m, "k = int8([1 2 3])") == "\nk =\n\n  1×3 int8 row vector\n\n   1   2   3\n\n"
    assert show(m, "k = uint16([1; 20])") == "\nk =\n\n  2×1 uint16 column vector\n\n    1\n   20\n\n"
    assert show(m, "k = int16([-100 20; 3 4])") == "\nk =\n\n  2×2 int16 matrix\n\n   -100     20\n      3      4\n\n"
    assert show(m, "s = single(3.14)") == "\ns =\n\n  single\n\n    3.1400\n\n"
    assert show(m, "s = single([1 2])") == "\ns =\n\n  1×2 single row vector\n\n     1     2\n\n"
    assert show(m, "k = uint8(200)") == "\nk =\n\n  uint8\n\n   200\n\n"


def test_empty_values(m):
    assert show(m, "e = []") == "\ne =\n\n     []\n\n"
    assert show(m, "e = zeros(0, 3)") == "\ne =\n\n  0×3 empty double matrix\n\n"
    assert show(m, "e = zeros(1, 0)") == "\ne =\n\n  1×0 empty double row vector\n\n"
    assert show(m, "e = zeros(0, 1)") == "\ne =\n\n  0×1 empty double column vector\n\n"
    assert show(m, "e = ''") == "\ne =\n\n  0×0 empty char array\n\n"
    assert show(m, "e = {}") == "\ne =\n\n  0×0 empty cell array\n\n"
    assert show(m, "e = int8([])") == "\ne =\n\n  0×0 empty int8 matrix\n\n"
    assert show(m, "e = strings(0, 1)") == "\ne = \n\n  0×1 empty string array\n\n"
    assert show(m, "e = zeros(0, 3, 2)") == "\ne =\n\n  0×3×2 empty double array\n\n"
    assert show(m, "e = true(0, 0)") == "\ne =\n\n  0×0 empty logical array\n\n"


def test_nd_arrays_show_one_page_at_a_time(m):
    out = show(m, "o = ones(2, 2, 2)")
    assert out == ("\no(:,:,1) =\n\n     1     1\n     1     1\n\n"
                   "\no(:,:,2) =\n\n     1     1\n     1     1\n\n")
    out = show(m, "k = int8(ones(1, 2, 2))")
    assert out.startswith("\nk =\n\n  1×2×2 int8 array\n\nk(:,:,1) =\n\n   1   1\n\n")
    out = show(m, "ans = zeros(1, 1, 1, 2)")
    assert "ans(:,:,1,2) =" in out


# -- text ---------------------------------------------------------------------------------------------

def test_char_and_string(m):
    assert show(m, "c = 'abc'") == "\nc =\n\n    'abc'\n\n"
    assert show(m, "c = ['ab'; 'cd']") == "\nc =\n\n  2×2 char array\n\n    'ab'\n    'cd'\n\n"
    assert show(m, 's = "abc"') == '\ns = \n\n    "abc"\n\n'
    assert show(m, 's = ["a" "bb"; "ccc" "d"]') == \
        '\ns = \n\n  2×2 string array\n\n    "a"      "bb"\n    "ccc"    "d"\n\n'
    assert show(m, "s = string(missing)") == "\ns = \n\n    <missing>\n\n"
    assert show(m, 's = ["a" missing]') == '\ns = \n\n  1×2 string array\n\n    "a"    <missing>\n\n'


# -- containers ------------------------------------------------------------------------------------------

def test_cell_arrays(m):
    assert show(m, "c = {1, 'a'}") == "\nc =\n\n  1×2 cell array\n\n    {[1]}    {'a'}\n\n"
    out = show(m, "c = {[], '', {}; 1.5, [1 2 3], true}")
    # every entry of a column has the same width: bracketed values are right-aligned inside
    # their brackets, other entries are padded before the closing brace
    assert out == ("\nc =\n\n  2×3 cell array\n\n"
                   "    {0×0 double}    {0×0 char}    {0×0 cell}\n"
                   "    {[  1.5000]}    {[ 1 2 3]}    {[     1]}\n\n")
    assert show(m, "c = {1; 'text'; {1}}") == \
        "\nc =\n\n  3×1 cell array\n\n    {[     1]}\n    {'text'  }\n    {1×1 cell}\n\n"
    out = show(m, 'c = {"x", ["a" "b"], struct("q", 1), magic(3), @sin, int8(3)}')
    for piece in ('{["x"]}', "{1×2 string}", "{1×1 struct}", "{3×3 double}", "{@sin}", "{[3]}"):
        assert piece in out
    assert "{2×2 char}" in show(m, "c = {['ab'; 'cd']}")
    assert "{1×2 cell}" in show(m, "c = {{1, 2}}")


def test_structs(m):
    out = show(m, "s = struct('a', 1, 'bb', 'hello', 'c', [1 2 3], 'd', magic(3))")
    assert out == ("\ns = \n\n  struct with fields:\n\n"
                   "     a: 1\n    bb: 'hello'\n     c: [1 2 3]\n     d: [3×3 double]\n\n")
    out = show(m, "s = struct('s', struct('t', 1), 'c', {{1, 2}}, 'e', [], 'f', @(x) x.^2, "
                  "'g', \"str\", 'em', '')")
    for piece in ("    s: [1×1 struct]", "    c: {1×2 cell}", "    e: []", "    f: @(x)x.^2", '    g: "str"',
                  "   em: ''"):
        assert piece in out
    assert show(m, "s = struct('a', {1, 2, 3})") == "\ns = \n\n  1×3 struct array with fields:\n\n    a\n\n"
    assert show(m, "s = struct()") == "\ns = \n\n  struct with no fields.\n\n"
    assert "x: 3.1416" in show(m, "s.x = pi")
    assert "y: [1.5000 2.5000]" in show(m, "s.y = [1.5 2.5]")
    assert "z: [1×100 double]" in show(m, "s.z = 1:100")


def test_function_handles(m):
    assert show(m, "f = @sin") == "\nf =\n\n  function_handle with value:\n\n    @sin\n\n"
    assert show(m, "f = @(x) x + 1") == "\nf =\n\n  function_handle with value:\n\n    @(x)x+1\n\n"


def test_ans_and_display_function(m):
    assert show(m, "3 + 4") == "\nans =\n\n     7\n\n"
    assert show(m, "x = 5; display(x)") == "\nx =\n\n     5\n\n"
    assert show(m, "x = 1, y = 2") == "\nx =\n\n     1\n\n\ny =\n\n     2\n\n"
    assert show(m, "x;") == ""


# -- disp ---------------------------------------------------------------------------------------------

@pytest.mark.parametrize("code, out", [
    ("disp(5)", "     5\n"),
    ("disp(pi)", "    3.1416\n"),
    ("disp([1.5 2.5])", "    1.5000    2.5000\n"),
    ("disp([1 2; 3 4])", "     1     2\n     3     4\n"),
    ("disp('hi')", "hi\n"),
    ('disp("hi")', "hi\n"),
    ('disp(["a" "b"])', '    "a"    "b"\n'),
    ("disp(true)", "   1\n"),
    ("disp(int8([1 2]))", "   1   2\n"),
    ("disp({1, 'a'})", "    {[1]}    {'a'}\n"),
    ("disp(struct('a', 1, 'b', 'x'))", "    a: 1\n    b: 'x'\n"),
    ("disp(@sin)", "    @sin\n"),
    ("disp([1000.5 1])", "   1.0e+03 *\n\n    1.0005    0.0010\n"),
    ("disp(3 + 4i)", "   3.0000 + 4.0000i\n"),
    ("disp(['ab'; 'cd'])", "ab\ncd\n"),
    ("disp([])", ""),
])
def test_disp(m, code, out):
    assert m.run(code) == out


def test_disp_of_nd_array_and_evalc(m):
    out = m.run("disp(ones(1, 2, 2))")
    assert out == "\n(:,:,1)\n\n     1     1\n\n\n(:,:,2)\n\n     1     1\n\n"
    m.run("s = evalc('x = 5');")
    assert m.str("s") == "\nx =\n\n     5\n\n"
    m.run("s = evalc('disp(pi)');")
    assert m.str("s") == "    3.1416\n"


def test_objects_get_the_leading_blank_line(m):
    out = m.run("t = datetime(2024, 1, 5)")
    assert out.startswith("\nt = \n\n  datetime\n\n")
    out = m.run("m = containers.Map()")
    assert out.startswith("\nm = \n\n  Map with properties:\n\n")
    out = m.run("T = table([1; 2], 'VariableNames', {'a'})")
    assert out.startswith("\nT =\n\n  2×1 table\n\n")
