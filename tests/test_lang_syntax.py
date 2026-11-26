"""Language features: Octave-only syntax accepted as an extension, arguments blocks,
error identifiers, MException, literals, indexing forms."""

import os

import numpy as np
import pytest

from mylab.errors import MatlabError
from mylab.parser import parse


def ident(m, code):
    """Identifier and message of the error that ``code`` raises."""
    m.run("clear e__; try, " + code + "; catch e__, end")
    return m.str("e__.identifier"), m.str("e__.message")


# -- Octave extensions ------------------------------------------------------------------------------

def test_compound_assignment_and_increment(m):
    m.run("x = 1; x += 2; x *= 3; x -= 1; x /= 4; x ^= 3;")
    assert m.num("x").item() == 8
    m.run("x++; x++; x--;")
    assert m.num("x").item() == 9
    m.run("b = true; b &= false; c = false; c |= true; v = [1 2 3]; v(2) += 10; s.n = 1; s.n += 1;")
    assert not m.num("b").item() and m.num("c").item()
    np.testing.assert_array_equal(m.num("v"), [[1, 12, 3]])
    assert m.num("s.n").item() == 2
    # in MATLAB a++b is a + (+b) and a--b is a - (-b): that meaning is kept
    m.run("a = 1; d1 = a++2; d2 = a--2; d3 = a ++ 2;")
    assert m.num("d1").item() == 3 and m.num("d2").item() == 3 and m.num("d3").item() == 3
    assert m.num("a").item() == 1


def test_octave_operators_and_comments(m):
    m.run("""
y = 2 ** 3;        # hash comment
z = !true;
w = 2 .** [1 2 3];
ne = 3 != 4;
#{
q = 99;
}#
%{
q = 98;
%}
A = [1 2   # inside a matrix
     3 4];
""")
    assert m.num("y").item() == 8 and not m.num("z").item() and m.num("ne").item()
    np.testing.assert_array_equal(m.num("w"), [[2, 4, 8]])
    np.testing.assert_array_equal(m.num("A"), [[1, 2], [3, 4]])
    assert not m.interp.base.has("q")


def test_do_until_and_unwind_protect(m):
    m.run("x = 9; do, x--; until x < 5")
    assert m.num("x").item() == 4
    m.run("""
k = 0; total = 0;
do
  k++;
  if k == 2, continue; end
  total += k;
  if k > 100, break; end
until k >= 5
""")
    assert m.num("total").item() == 1 + 3 + 4 + 5
    m.run("unwind_protect, q = 1; unwind_protect_cleanup, q = 2; end_unwind_protect")
    assert m.num("q").item() == 2
    with pytest.raises(MatlabError, match="boom"):
        m.run("""
unwind_protect
  error('my:id', 'boom');
unwind_protect_cleanup
  cleaned = true;
end_unwind_protect
""")
    assert m.num("cleaned").item()
    # 'do' and 'until' stay ordinary names when they are not used as a loop
    m.run("do = 3; until = do + 1;")
    assert m.num("until").item() == 4


def test_octave_block_keywords_and_functions_in_scripts(m):
    m.run("""
1;
function r = addone(x)
  r = x + 1;
endfunction
s = 0;
if true, s = addone(s); endif
for i = 1:2, s += i; endfor
k = 0; while k < 2, k++; endwhile
switch s, case 4, t = 'four'; otherwise, t = 'other'; endswitch
try, error('x'); catch, u = 1; end_try_catch
function r = twice(x)
  r = 2 * x;
end
v = twice(s);
""")
    assert m.num("s").item() == 4 and m.str("t") == "four" and m.num("v").item() == 8


def test_chained_indexing_on_any_expression(m):
    m.run("a = [1 2 3](2); b = magic(4)(2, :); c = {1, 'two'}{2}; d = (1:5)(end); e = magic(3)(2, 2)(1);")
    assert m.num("a").item() == 2 and m.str("c") == "two" and m.num("d").item() == 5 and m.num("e").item() == 5
    np.testing.assert_array_equal(m.num("b"), [[5, 11, 10, 8]])
    m.run("f = @() {1, [4 5 6]}; g = f(){2}(3); x = 1:10; h = x(2:end)(1:2);")
    assert m.num("g").item() == 6
    np.testing.assert_array_equal(m.num("h"), [[2, 3]])


def test_octave_output_functions(m):
    assert m.run("printf('%d-%s\\n', 3, 'a')") == "3-a\n"
    assert m.run("puts('abc'); fputs(stdout, 'def'); fdisp(stdout, [1 2])") == "abcdef     1     2\n"
    m.run("a = ifelse(true, 1, 2); b = merge([true false true], [1 2 3], [7 8 9]); c = merge(false, 'y', 'n');")
    assert m.num("a").item() == 1 and m.str("c") == "n"
    np.testing.assert_array_equal(m.num("b"), [[1, 8, 3]])
    assert m.num("rows(ones(2, 3))").item() == 2 and m.num("columns(ones(2, 3))").item() == 3


def test_shell_escape_keeps_its_matlab_meaning():
    # '!cmd' at the start of a line is a shell escape in MATLAB, so it is not Octave's "not"
    stmt = parse("!echo hi\n").body[0]
    assert stmt.expr.base.name == "system" and stmt.expr.args[0].value == "echo hi"
    # elsewhere ! is "not"
    assert parse("x = !y;\n").body[0].value.op == "~"


def test_parfor_and_spmd_run_serially(m):
    m.run("r = 0; parfor i = 1:4, r = r + i; end\nparfor (j = 1:3, 2)\n r = r + j;\nend\nspmd\n w = 5;\nend")
    assert m.num("r").item() == 16 and m.num("w").item() == 5


# -- literals ---------------------------------------------------------------------------------------

def test_hex_binary_and_wide_integer_literals(m):
    assert m.str("class(0x1Fu16)") == "uint16" and m.num("0x1Fu16").item() == 31
    assert m.str("class(0b101s8)") == "int8" and m.num("0b101s8").item() == 5
    assert m.num("0xFFs8").item() == -1 and m.str("class(0xFF)") == "uint8" and m.str("class(0x100)") == "uint16"
    assert int(m.num("int64(9007199254740993)").item()) == 9007199254740993
    assert int(m.num("uint64(18446744073709551615)").item()) == 18446744073709551615
    assert int(m.num("-int64(9223372036854775807)").item()) == -9223372036854775807
    assert m.num("1e3i").item() == 1000j and m.num("2.5e-1j").item() == 0.25j


def test_transpose_versus_quote(m):
    m.run("x = [1 2 3]; a = [x' x']; b = {x', 'str'}; c = x''; d = 'it''s'; e = [x']'; f = {'a', 'b'}';")
    assert m.num("a").shape == (3, 2) and m.num("c").shape == (1, 3) and m.str("d") == "it's"
    assert m.num("e").shape == (1, 3)
    m.run("g = [1 -2]'; h = x(2)'; k = [x(1)' x(2)']; st.f = [1 2]; l = st.f'; n = [x' ; 4]';")
    assert m.num("g").shape == (2, 1) and m.num("k").size == 2 and m.num("l").shape == (2, 1)
    np.testing.assert_array_equal(m.num("n"), [[1, 2, 3, 4]])


# -- indexing forms -----------------------------------------------------------------------------------

def test_growth_and_deletion_forms(m):
    m.run("A = []; A(:, end+1) = [1; 2]; A(:, end+1) = [3; 4]; A(end+1, :) = [5 6];")
    np.testing.assert_array_equal(m.num("A"), [[1, 3], [2, 4], [5, 6]])
    m.run("A(:, 1) = []; B = 1:5; B([1 3]) = []; B(end) = []; C = magic(4); C([1 4], :) = []; "
          "C(:, logical([1 0 0 1])) = [];")
    np.testing.assert_array_equal(m.num("A"), [[3], [4], [6]])
    np.testing.assert_array_equal(m.num("B"), [[2, 4]])
    assert m.num("C").shape == (2, 2)
    m.run("E = 'hello'; E(end+1:end+3) = '!!!'; E([1 end]) = []; G = []; G(end+1).name = 'a'; G(end+1).name = 'b';")
    assert m.str("E") == "ello!!" and m.num("numel(G)").item() == 2
    m.run("F = zeros(2); F(:, :, 2) = 1; F(:, :, 1) = []; K = eye(3); K(:) = []; D = {1, 2, 3}; D(2) = [];")
    assert m.num("F").shape == (2, 2) and m.num("K").size == 0 and m.num("numel(D)").item() == 2


def test_end_inside_function_call_arguments_and_chained_calls(m):
    m.run("x = 1:10; a = x(min(end, 20)); b = x(max(1, end-2):end); c = x(numel(x(1:end-5))); d = x(round(end/2));")
    assert m.num("a").item() == 10 and m.num("c").item() == 5 and m.num("d").item() == 5
    np.testing.assert_array_equal(m.num("b"), [[8, 9, 10]])
    m.run("sf = @(k) struct('field', k * 2, 'c', {{1, 2, k}}); p = sf(3).field; q = sf(3).c{3};")
    assert m.num("p").item() == 6 and m.num("q").item() == 3
    m.run("s.b(2).c = {1, 2, 'three'}; r = s.b(2).c{3}; s.b(2).c{3}(1) = 'T'; t = s.b(2).c{3};")
    assert m.str("r") == "three" and m.str("t") == "Three"
    m.run("mk = @(a) @(b) @(c) a + b + c; u = mk(1)(2)(3); val = 10; fe = @() val; val = 20; w = fe();")
    assert m.num("u").item() == 6 and m.num("w").item() == 10


# -- error identifiers and messages -------------------------------------------------------------------

@pytest.mark.parametrize("code, identifier, message", [
    ("x = [1 2 3]; x(7)", "MATLAB:badsubscript",
     "Index exceeds the number of array elements. Index must not exceed 3."),
    ("x = [1 2 3]; x(0)", "MATLAB:badsubscript", "Array indices must be positive integers or logical values."),
    ("A = magic(3); A(4, 1)", "MATLAB:badsubscript",
     "Index in position 1 exceeds array bounds. Index must not exceed 3."),
    ("undefined_fn_xyz(1)", "MATLAB:UndefinedFunction", "Unrecognized function or variable 'undefined_fn_xyz'."),
    ("s.a = 1; s.b", "MATLAB:nonExistentField", 'Unrecognized field name "b".'),
    ("ones(2, 3) + ones(3, 2)", "MATLAB:sizeDimensionsMustMatch", "Arrays have incompatible sizes for this operation."),
    ("[1 2; 3]", "MATLAB:catenate:dimensionMismatch", "Dimensions of arrays being concatenated are not consistent."),
    ("x = 1:3; x(1:2) = [1 2 3]", "MATLAB:subsassignnumelmismatch",
     "Unable to perform assignment because the left and right sides have a different number of elements."),
    ("A = eye(2); A(:, 1) = [1 2 3]", "MATLAB:subsassigndimmismatch",
     "Unable to perform assignment because the size of the left side is 2-by-1 and the size of the right "
     "side is 1-by-3."),
    ("x = 5; x{1}", "MATLAB:cellRefFromNonCell", "Brace indexing is not supported for variables of this type."),
    ("x = 5; x.f", "MATLAB:structRefFromNonStruct",
     "Dot indexing is not supported for variables of this type."),
    ("f = @(x) x; f(1, 2)", "MATLAB:TooManyInputs", "Too many input arguments."),
    ("sin(1, 2)", "MATLAB:TooManyInputs", "Too many input arguments."),
    ("sin()", "MATLAB:minrhs", "Not enough input arguments."),
    ("[a, b] = sin(1)", "MATLAB:TooManyOutputs", "Too many output arguments."),
    ("c = {1}; c + 1", "MATLAB:UndefinedFunction", "Operator '+' is not supported for operands of type 'cell'."),
    ("int8(1) + int16(1)", "MATLAB:mixedClasses",
     "Integers can only be combined with integers of the same class, or scalar doubles."),
    ("A = eye(3); A(1, 2) = []", "MATLAB:subsdeldimmismatch", "A null assignment can have only one non-colon index."),
    ("x = NaN; if x, end", "MATLAB:nologicalnan", "NaN's cannot be converted to logicals."),
    ("double({1})", "MATLAB:invalidConversion", "Conversion to double from cell is not possible."),
    ("ones(2) \\ ones(3, 1)", "MATLAB:dimagree", "Matrix dimensions must agree."),
    ("assert(false)", "MATLAB:assertion:failed", "Assertion failed."),
    ("m = containers.Map(); m('k')", "MATLAB:Containers:Map:NoKey",
     "The specified key is not present in this container."),
    ("error('Simple message')", "", "Simple message"),
    ("error('a:b:c', 'With id %d', 3)", "a:b:c", "With id 3"),
    ("error('Too many input arguments.')", "", "Too many input arguments."),
])
def test_error_identifiers_and_messages(m, code, identifier, message):
    assert ident(m, code) == (identifier, message)


def test_more_error_identifiers(m):
    assert ident(m, "[1 2 3] * [4 5 6]")[0] == "MATLAB:innerdim"
    assert ident(m, "if [1 2] && true, end")[0] == "MATLAB:nonLogicalConditional"
    assert ident(m, "reshape(1:6, 4, 2)")[0] == "MATLAB:getReshapeDims:notSameNumel"
    assert ident(m, "str2double()")[0] == "MATLAB:minrhs"
    assert ident(m, "ones(2, 3)^2")[0] == "MATLAB:square"
    assert ident(m, "switch [1 2], case 1, end")[0] == "MATLAB:badSwitchExpression"
    m.run("function r = noout(), end\ntry, v = noout(); catch e2, end")
    assert m.str("e2.identifier") == "MATLAB:unassignedOutputs"


# -- MException ------------------------------------------------------------------------------------------

def test_mexception_methods(m):
    m.run("""
ME = MException('top:level', 'Top %d', 1);
ME = addCause(ME, MException('low:level', 'Low'));
try, throw(ME); catch err, end
basic = getReport(err, 'basic'); full = getReport(err, 'extended', 'hyperlinks', 'off');
try, try, error('in:ner', 'x'); catch e3, throwAsCaller(e3); end, catch e4, end
last = MException.last; [msg, id] = lasterr; le = lasterror;
try, error(struct('message', 'from struct', 'identifier', 'st:id')); catch e5, end
e6 = MException('a:b', 'msg'); e7 = addCorrection(e6, 'fix');
""")
    assert m.str("err.cause{1}.message") == "Low" and m.str("err.identifier") == "top:level"
    assert "Top 1" in m.str("basic") and "Caused by:" in m.str("full") and "Low" in m.str("full")
    assert m.str("e4.identifier") == "in:ner" and m.str("last.identifier") == "in:ner"
    assert m.str("id") == "in:ner" and m.str("msg") == "x" and m.str("le.identifier") == "in:ner"
    assert m.str("e5.identifier") == "st:id" and m.str("e5.message") == "from struct"
    assert m.num("isempty(e6.stack)").item() and m.num("isempty(e6.cause)").item()
    assert m.str("e7.Correction") == "fix" and m.str("class(e6)") == "MException"
    with pytest.raises(MatlabError):
        m.run("getReport(err, 'nonsense')")
    with pytest.raises(MatlabError, match="hello x"):
        m.run("throw(MException('a:b', 'hello %s', 'x'))")
    m.run("try, error('no id here: %d', 5); catch e8, end\ntry, error('literal\\n'); catch e9, end")
    assert m.str("e8.identifier") == "" and m.str("e8.message") == "no id here: 5"
    assert m.str("e9.message") == "literal\\n"   # a single argument is not a format


# -- arguments blocks -----------------------------------------------------------------------------------

FILES = {
    "sumpairs.m": """function r = sumpairs(a, b)
arguments (Repeating)
    a (1,1) double
    b (1,1) double
end
r = sum([a{:}]) * 10 + sum([b{:}]);
end
""",
    "scaled.m": """function r = scaled(x, opts)
arguments
    x (1,1) double {mustBePositive}
    opts.Scale (1,1) double = 2
    opts.Name (1,:) char = 'n'
end
r = x * opts.Scale;
end
""",
    "defs.m": """function [a, b, n] = defs(a, b)
arguments (Input)
    a (1,1) double
    b (1,1) double {mustBeGreaterThanOrEqual(b, a)} = 2 * a
end
n = nargin;
end
""",
    "outs.m": """function r = outs(x)
arguments (Input)
    x double
end
arguments (Output)
    r (1,1) string
end
r = x;
end
""",
    "vecs.m": """function v = vecs(v, s)
arguments
    v (1,:) double
    s (1,3) double
end
v = v + s;
end
""",
    "nv2.m": """function w = nv2(x, opts)
arguments
    x
    opts.Color (1,1) string {mustBeMember(opts.Color, ["red", "blue"])} = "blue"
    opts.Width (1,1) double {mustBeInteger, mustBePositive} = x
end
w = opts.Color + opts.Width;
end
""",
    "rep2.m": """function s = rep2(name, value, opts)
arguments (Repeating)
    name (1,:) char
    value (1,1) double
end
arguments
    opts.Sep (1,1) string = ","
end
s = '';
for k = 1:numel(name)
    s = [s name{k} num2str(value{k}) char(opts.Sep)];
end
end
""",
    "Pen.m": """classdef Pen
    properties
        Width (1,1) double {mustBePositive} = 1
        Color (1,1) string = "black"
    end
end
""",
    "draw.m": """function s = draw(opts)
arguments
    opts.?Pen
    opts.Fill (1,1) logical = false
end
s = opts;
end
""",
}


@pytest.fixture
def am(m, tmp_path):
    for name, body in FILES.items():
        (tmp_path / name).write_text(body)
    old = os.getcwd()
    os.chdir(tmp_path)
    yield m
    os.chdir(old)


def test_arguments_repeating_and_name_value(am):
    m = am
    assert m.num("sumpairs(1, 2, 3, 4)").item() == 46 and m.num("sumpairs()").item() == 0
    assert m.num("scaled(3)").item() == 6 and m.num("scaled(3, 'Scale', 4)").item() == 12
    assert m.num("scaled(3, Scale=5)").item() == 15 and m.str("class(scaled(int8(3)))") == "double"
    assert m.str("rep2('a', 1, 'b', 2, Sep='-')") == "a1-b2-"
    assert m.str("nv2(1, Color=\"red\", Width=3)") == "red3" and m.str("nv2(1)") == "blue1"
    with pytest.raises(MatlabError, match="groups of 2"):
        m.run("sumpairs(1, 2, 3)")


def test_arguments_errors(am):
    m = am
    assert ident(m, "scaled(-1)") == ("MATLAB:validators:mustBePositive",
                                      "Invalid argument at position 1. Value must be positive.")
    assert ident(m, "scaled([1 2])") == ("MATLAB:validation:IncompatibleSize",
                                         "Invalid argument at position 1. Value must be a scalar.")
    assert ident(m, "scaled()") == ("MATLAB:minrhs", "Not enough input arguments.")
    assert ident(m, "scaled(3, 'Bogus', 4)")[0] == "MATLAB:functionValidation:UnknownNameValue"
    assert ident(m, "scaled({1})")[0] == "MATLAB:validation:UnableToConvert"
    assert ident(m, "nv2(1, Color=\"pink\")") == (
        "MATLAB:validators:mustBeMember",
        "Invalid name-value argument 'Color'. Value must be a member of the allowed set.")


def test_arguments_defaults_sizes_and_outputs(am):
    m = am
    m.run("[a, b, n] = defs(2); [a2, b2, n2] = defs(2, 7);")
    assert (m.num("a").item(), m.num("b").item(), m.num("n").item()) == (2, 4, 1)
    assert (m.num("b2").item(), m.num("n2").item()) == (7, 2)
    assert ident(m, "defs(5, 1)")[0] == "MATLAB:validators:mustBeGreaterThanOrEqual"
    m.run("r = outs(3);")
    assert m.str("class(r)") == "string" and m.str("r") == "3"
    assert ident(m, "outs([1 2])")[0] == "MATLAB:validation:IncompatibleSize"
    # a column vector is accepted for (1,:) and a scalar expands to a fixed size
    np.testing.assert_array_equal(m.num("vecs([1; 2; 3], 5)"), [[6, 7, 8]])
    assert ident(m, "vecs([1 2 3], [1 2])")[0] == "MATLAB:validation:IncompatibleSize"


def test_arguments_from_class_properties(am):
    m = am
    m.run("s = draw(Width=3, Fill=true); t = draw();")
    assert m.num("s.Width").item() == 3 and m.num("s.Fill").item() and m.num("isfield(t, 'Width')").item() == 0
    assert m.num("t.Fill").item() == 0
    assert ident(m, "draw(Width=-1)")[0] == "MATLAB:validators:mustBePositive"
    assert ident(m, "draw(Nope=1)")[0] == "MATLAB:functionValidation:UnknownNameValue"


def test_validators_added(m):
    for good in ("mustBeFloat(1.5)", "mustBeRow([1 2 3])", "mustBeColumn([1; 2])", "mustBeScalarOrEmpty([])",
                 "mustBeValidVariableName('abc_1')", "mustBeNonmissing([1 2])", "mustBeUnderlyingType(1, 'double')",
                 "mustBeBetween(2, 1, 3)", "mustBeMatrix(ones(2, 3))"):
        m.run(good)
    for bad, name in (("mustBeFloat(int8(1))", "mustBeFloat"), ("mustBeRow([1; 2])", "mustBeRow"),
                      ("mustBeColumn([1 2])", "mustBeColumn"), ("mustBeScalarOrEmpty([1 2])", "mustBeScalarOrEmpty"),
                      ("mustBeValidVariableName('1abc')", "mustBeValidVariableName"),
                      ("mustBeNonmissing([1 NaN])", "mustBeNonmissing"),
                      ("mustBeBetween(3, 1, 3, 'open')", "mustBeBetween")):
        assert ident(m, bad)[0] == "MATLAB:validators:" + name


# -- persistent / global -------------------------------------------------------------------------------

def test_clear_function_resets_its_persistent_variables(m, tmp_path):
    (tmp_path / "nextid.m").write_text("function r = nextid()\npersistent n\nif isempty(n), n = 0; end\n"
                                       "n = n + 1; r = n;\nend\n")
    old = os.getcwd()
    os.chdir(tmp_path)
    try:
        m.run("nextid(); nextid(); a = nextid(); clear nextid; b = nextid(); nextid(); clear functions; c = nextid();")
        assert (m.num("a").item(), m.num("b").item(), m.num("c").item()) == (3, 1, 1)
        m.run("nextid(); clear all; d = nextid();")
        assert m.num("d").item() == 1
    finally:
        os.chdir(old)
    m.run("global GG; GG = 5; g1 = isglobal('GG'); clear GG; g2 = exist('GG', 'var'); global GG; g3 = GG;")
    assert m.num("g1").item() and m.num("g2").item() == 0 and m.num("g3").item() == 5


def test_more_octave_statements(m):
    m.run("a = b = c = 5; v = [1 2 3]; v(2)++; s.n = 4; s.n--; z = {1, 2}; z{2} += 5;")
    assert (m.num("a").item(), m.num("b").item(), m.num("c").item()) == (5, 5, 5)
    np.testing.assert_array_equal(m.num("v"), [[1, 3, 3]])
    assert m.num("s.n").item() == 3 and m.num("z{2}").item() == 7
    m.run("M = magic(3); M(2, :) *= 2; q.r.s = 1; q.r.s += 1; k = 0; for i = 1:3, k++; end")
    np.testing.assert_array_equal(m.num("M(2, :)"), [[6, 10, 14]])
    assert m.num("q.r.s").item() == 2 and m.num("k").item() == 3
    assert abs(m.num("e").item() - np.e) < 1e-15
    assert parse("dir *.m\n").body[0].expr.args[0].value == "*.m"


def test_validators_with_options(m):
    m.run("mustBeInRange(1, 1, 10); mustBeInRange(10, 1, 10, 'exclude-lower'); mustBeVector([], 'allow-all-empties')")
    assert ident(m, "mustBeInRange(1, 1, 10, 'exclusive')")[0] == "MATLAB:validators:mustBeInRange"
    assert ident(m, "mustBeInRange(1, 1, 10, 'exclude-lower')")[0] == "MATLAB:validators:mustBeInRange"
    assert ident(m, "mustBeVector(ones(2))")[0] == "MATLAB:validators:mustBeVector"
    with pytest.raises(MatlabError):
        m.run("mustBeInRange(1, 0, 2, 'sideways')")
