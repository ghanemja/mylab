"""Gap fixes found by running public MATLAB code: implicit size and class conversion in property
declarations and arguments blocks (WEC-Sim), and inputParser optional inputs given by name (DGCNN)."""

import os

import numpy as np
import pytest

from mylab.errors import MatlabError


def ident(m, code):
    """Identifier and message of the error that ``code`` raises."""
    m.run("clear e__; try, " + code + "; catch e__, end")
    return m.str("e__.identifier"), m.str("e__.message")


FILES = {
    "SizedProps.m": """classdef SizedProps
    properties
        dt (1,:) {mustBeScalarOrEmpty} = []
        cols (:,3) double = []
        col (:,1) = []
        two (2,:) = 5
        trio (1,3) double = 7
        flag (1,1) double = true
        ints (1,:) double = int8([1 2])
        word (1,:) char = ''
        strs (1,:) string = []
        kids (1,:) Kid = []
        lvl (1,1) Spd = Spd.Low
    end
end
""",
    "Kid.m": """classdef Kid < handle
    properties
        n = 0
    end
    methods
        function obj = Kid(n)
            if nargin < 1
                error('Kid:noDefault', 'Kid needs an input.');
            end
            obj.n = n;
        end
        function bump(obj)
            obj.n = obj.n + 1;
        end
    end
end
""",
    "Spd.m": """classdef Spd
    enumeration
        Low, Medium, High
    end
end
""",
    "BadDefault.m": """classdef BadDefault
    properties
        x (1,1) double = [1 2 3]
    end
end
""",
    "convargs.m": """function r = convargs(f, n, t, s)
arguments
    f (1,1) logical = false
    n (1,:) int8 = []
    t (1,:) char = ''
    s (1,:) string = []
end
r = {f, n, t, s};
end
""",
    "sizedargs.m": """function r = sizedargs(a, b, c, d)
arguments
    a (1,:) double = []
    b (:,1) double = []
    c (2,2) double = 0
    d (1,1) double = false
end
r = {size(a), size(b), c, class(d), class(a)};
end
""",
}


@pytest.fixture
def cm(m, tmp_path):
    for name, body in FILES.items():
        (tmp_path / name).write_text(body)
    old = os.getcwd()
    os.chdir(tmp_path)
    yield m
    os.chdir(old)


# -- property defaults -------------------------------------------------------------------------------

def test_empty_default_takes_the_declared_size(cm):
    cm.run("s = SizedProps();")
    assert cm.num("size(s.dt)").tolist() == [[1, 0]]
    assert cm.num("size(s.cols)").tolist() == [[0, 3]]
    assert cm.num("size(s.col)").tolist() == [[0, 1]]
    assert cm.num("size(s.word)").tolist() == [[1, 0]] and cm.num("ischar(s.word)").item()
    assert cm.num("size(s.strs)").tolist() == [[1, 0]] and cm.num("isstring(s.strs)").item()
    assert cm.num("size(s.kids)").tolist() == [[1, 0]] and cm.str("class(s.kids)") == "Kid"


def test_scalar_default_expands_and_classes_convert(cm):
    cm.run("s = SizedProps();")
    np.testing.assert_array_equal(cm.num("s.two"), [[5], [5]])
    np.testing.assert_array_equal(cm.num("s.trio"), [[7, 7, 7]])
    assert cm.str("class(s.flag)") == "double" and cm.num("s.flag").item() == 1
    assert cm.str("class(s.ints)") == "double"
    np.testing.assert_array_equal(cm.num("s.ints"), [[1, 2]])


def test_bad_default_names_the_size(cm):
    eid, msg = ident(cm, "BadDefault()")
    assert eid == "MATLAB:validation:IncompatibleSize"
    assert msg == "Invalid default value for property 'x' of class 'BadDefault'. Value must be a scalar."


# -- assignments ------------------------------------------------------------------------------------

def test_assignment_conversions(cm):
    cm.run("s = SizedProps(); s.dt = 0.1; s.dt = []; a = size(s.dt); s.cols = []; b = size(s.cols);"
           "s.col = [1 2 3]; s.trio = [4; 5; 6]; s.ints = uint8([3; 4]);")
    assert cm.num("a").tolist() == [[1, 0]] and cm.num("b").tolist() == [[0, 3]]
    np.testing.assert_array_equal(cm.num("s.col"), [[1], [2], [3]])
    np.testing.assert_array_equal(cm.num("s.trio"), [[4, 5, 6]])
    assert cm.str("class(s.ints)") == "double" and cm.num("size(s.ints)").tolist() == [[1, 2]]


def test_assignment_size_errors_name_the_size(cm):
    cm.run("s = SizedProps();")
    eid, msg = ident(cm, "s.dt = ones(2)")
    assert eid == "MATLAB:validation:IncompatibleSize"
    assert msg == ("Error setting property 'dt' of class 'SizedProps'. Value must be a row vector (size 1×:), "
                   "but its size is 2×2.")
    assert "Value must have size :×3" in ident(cm, "s.cols = [1 2]")[1]
    assert "Value must have size 1×3 (or be a scalar" in ident(cm, "s.trio = [1 2]")[1]
    assert ident(cm, "s.flag = []")[1].endswith("Value must be a scalar.")
    # the validator still runs after the conversion
    assert ident(cm, "s.dt = [1 2]")[0] == "MATLAB:validators:mustBeScalarOrEmpty"


def test_enumeration_from_inexact_text(cm):
    cm.run("s = SizedProps(); s.lvl = 'm'; a = s.lvl; s.lvl = \"HIGH\"; b = s.lvl;")
    assert cm.num("a == Spd.Medium").item() and cm.num("b == Spd.High").item()
    assert ident(cm, "s.lvl = 'x'")[0] == "MATLAB:validation:UnableToConvert"


# -- arguments blocks -------------------------------------------------------------------------------

def test_arguments_block_conversions(cm):
    cm.run("r = sizedargs();")
    assert cm.num("r{1}").tolist() == [[1, 0]] and cm.num("r{2}").tolist() == [[0, 1]]
    np.testing.assert_array_equal(cm.num("r{3}"), np.zeros((2, 2)))
    assert cm.str("r{4}") == "double"
    cm.run("r = sizedargs(int16([1; 2]), [3 4 5], 9, true);")
    assert cm.num("r{1}").tolist() == [[1, 2]] and cm.num("r{2}").tolist() == [[3, 1]]
    np.testing.assert_array_equal(cm.num("r{3}"), np.full((2, 2), 9.0))
    assert cm.str("r{4}") == "double" and cm.str("r{5}") == "double"
    cm.run("r = sizedargs(sparse([1; 0; 2]));")  # a sparse column is turned into a row too
    assert cm.num("r{1}").tolist() == [[1, 3]] and cm.str("r{5}") == "double"
    eid, msg = ident(cm, "sizedargs(ones(2, 3))")
    assert eid == "MATLAB:validation:IncompatibleSize"
    assert msg == "Invalid argument at position 1. Value must be a row vector (size 1×:), but its size is 2×3."
    assert ident(cm, "sizedargs([], [], [1 2 3])")[1].startswith("Invalid argument at position 3. Value must "
                                                                  "have size 2×2")


# -- object arrays and methods called as statements --------------------------------------------------

def test_growing_an_object_array_without_gaps_needs_no_default(cm):
    cm.run("k(1) = Kid(1); k(2) = Kid(2); j = Kid(5); j(2) = Kid(6); n = [k.n, j.n];")
    np.testing.assert_array_equal(cm.num("n"), [[1, 2, 5, 6]])
    eid, msg = ident(cm, "k(4) = Kid(4)")  # a gap needs the no-argument constructor, which refuses
    assert "Kid() with no arguments failed" in msg


def test_method_named_as_a_statement_takes_no_outputs(cm):
    cm.run("k = Kid(1); k.bump; k.bump;")
    assert cm.num("k.n").item() == 3


# -- inputParser --------------------------------------------------------------------------------------

def test_optional_inputs_given_by_name(m):
    m.run("p = inputParser; p.addOptional('distance', 'l1', @ischar); p.addOptional('w', 1e-4, @isnumeric);"
          "p.addOptional('trans', @(x) x, @(x) isa(x, 'function_handle')); p.parse('distance', 'tv', 'w', 0.5);"
          "r = p.Results; d = p.UsingDefaults;")
    assert m.str("r.distance") == "tv" and m.num("r.w").item() == 0.5
    assert m.str("func2str(r.trans)") == "@(x)x" and m.str("d{1}") == "trans"
    # several optional inputs by name, in any order
    m.run("p.parse('trans', @sin, 'distance', 'l2'); r = p.Results;")
    assert m.str("r.distance") == "l2" and m.str("func2str(r.trans)") == "sin" and m.num("r.w").item() == 1e-4
    # positional first, then by name
    m.run("p.parse('l2', 'trans', @cos); r = p.Results;")
    assert m.str("r.distance") == "l2" and m.str("func2str(r.trans)") == "cos"
    m.run("p.parse('hellinger', 3); r = p.Results;")
    assert m.str("r.distance") == "hellinger" and m.num("r.w").item() == 3


def test_optional_by_name_with_rejecting_validators(m):
    m.run("p = inputParser; p.addRequired('x');"
          "p.addOptional('distance', 'l1', @(x) ismember(lower(x), {'l1', 'l2', 'tv'}));"
          "p.addOptional('w', 1e-4, @(x) isscalar(x) && x > 0);"
          "p.addOptional('w_attr', 1, @(x) isscalar(x) && x > 0);"
          "p.parse(1, 'w', 2, 'distance', 'tv', 'w_attr', 3); r = p.Results;")
    assert m.str("r.distance") == "tv" and m.num("r.w").item() == 2 and m.num("r.w_attr").item() == 3
    # an exact name wins over a longer one it begins: 'w' is w, not w_attr
    m.run("p.parse(1, 'w', 5); r = p.Results;")
    assert m.num("r.w").item() == 5 and m.num("r.w_attr").item() == 1
    # a unique leading part of an optional input's name is enough (matGeom's fitEllipse3d(p, 'vis', true))
    m.run("q = inputParser; q.addOptional('visualization', false, @islogical); q.parse('vis', true);")
    assert m.num("q.Results.visualization").item()


def test_mixed_optional_and_parameters(m):
    m.run("p = inputParser; p.addOptional('a', 1, @isnumeric); p.addOptional('b', 2, @isnumeric);"
          "p.addOptional('c', 3, @isnumeric); p.addParameter('Tol', 0.1);"
          "p.parse(10, 'c', 30, 'tol', 0.5); r = p.Results;")
    assert [m.num(f"r.{k}").item() for k in ("a", "b", "c", "Tol")] == [10, 2, 30, 0.5]
    assert m.str("strjoin(p.UsingDefaults, ',')") == "b"
    # an optional input given by position cannot be given again by name
    with pytest.raises(MatlabError, match="already given by position"):
        m.run("p.parse(10, 'a', 5)")
    # a name given twice keeps the last value
    m.run("p.parse('Tol', 1, 'Tol', 2); r = p.Results;")
    assert m.num("r.Tol").item() == 2


def test_text_optional_without_validator_is_a_name(m):
    m.run("q = inputParser; q.addOptional('units', 'in'); q.addParameter('Scale', 1);")
    eid, msg = ident(m, "q.parse('cm')")
    assert eid == "MATLAB:InputParser:UnmatchedParameter" and "'cm' is not a recognized parameter" in msg
    m.run("q.parse('units', 'cm'); r = q.Results;")
    assert m.str("r.units") == "cm"
    m.run("q.parse(5, 'Scale', 2); r = q.Results;")  # a non-text value is still positional
    assert m.num("r.units").item() == 5 and m.num("r.Scale").item() == 2


def test_text_accepted_by_the_validator_stays_positional(m):
    m.run("p = inputParser; p.addOptional('mode', 'fast', @ischar); p.addParameter('Tol', 1);"
          "p.parse('slow'); a = p.Results.mode; p.parse('mode'); b = p.Results.mode;"
          "p.parse('slow', 'Tol', 3); c = p.Results.Tol;")
    assert m.str("a") == "slow" and m.str("b") == "mode" and m.num("c").item() == 3


def test_struct_expand_and_partial_matching(m):
    m.run("p = inputParser; p.addOptional('k', 1, @isnumeric); p.addParameter('Alpha', 1);"
          "p.addParameter('Alps', 2, @isnumeric, 'PartialMatchPriority', 2);"
          "s.Alpha = 4; s.k = 9; p.parse(s); r = p.Results; p.parse('Al', 5); t = p.Results;")
    assert m.num("r.Alpha").item() == 4 and m.num("r.k").item() == 9
    assert m.num("t.Alpha").item() == 5 and m.num("t.Alps").item() == 2
    # struct field names are not partially matched
    m.run("p.KeepUnmatched = true; u.Alp = 1; p.parse(u); un = p.Unmatched;")
    assert m.num("un.Alp").item() == 1
    # with KeepUnmatched an ambiguous name goes to Unmatched
    m.run("q = inputParser; q.addParameter('Alpha', 1); q.addParameter('Alps', 2); q.KeepUnmatched = true;"
          "q.parse('Al', 3); un = q.Unmatched; r = q.Results;")
    assert m.num("un.Al").item() == 3 and m.num("r.Alpha").item() == 1
    q_err = ident(m, "q.KeepUnmatched = false; q.parse('Al', 3)")
    assert q_err[0] == "MATLAB:InputParser:AmbiguousParamName"


def test_case_sensitive_names(m):
    m.run("p = inputParser; p.CaseSensitive = true; p.addOptional('w', 1, @isnumeric); p.addParameter('Tol', 1);"
          "p.parse('w', 4); a = p.Results.w;")
    assert m.num("a").item() == 4
    assert ident(m, "p.parse(1, 'TOL', 4)")[0] == "MATLAB:InputParser:UnmatchedParameter"
    assert ident(m, "p.parse(1, 'tol', 4)")[0] == "MATLAB:InputParser:UnmatchedParameter"
    # text in the optional input's place that names nothing and fails its validator reports the validator
    assert ident(m, "p.parse('W', 4)")[0] == "MATLAB:InputParser:ArgumentFailedValidation"


def test_add_method_options_are_checked(m):
    m.run("p = inputParser;")
    with pytest.raises(MatlabError, match="Unknown option 'Bogus'"):
        m.run("p.addParameter('A', 1, @isnumeric, 'Bogus', 2)")
    with pytest.raises(MatlabError, match="no further options"):
        m.run("p.addOptional('B', 1, @isnumeric, 'PartialMatchPriority', 2)")
    with pytest.raises(MatlabError, match="must be a function handle"):
        m.run("p.addOptional('C', 1, 'isnumeric')")
    with pytest.raises(MatlabError, match="positive integer"):
        m.run("p.addParameter('D', 1, 'PartialMatchPriority', 0)")


def test_struct_for_an_optional_input_without_validator(m):
    m.run("p = inputParser; p.addOptional('opts', struct('a', 1)); p.addParameter('Tol', 1);"
          "s.x = 5; p.parse(s); a = p.Results.opts.x; t.Tol = 3; p.parse(t); b = p.Results.Tol;")
    assert m.num("a").item() == 5 and m.num("b").item() == 3  # fields that name inputs are name-value pairs


def test_exist_reports_model_files(cm, tmp_path):
    (tmp_path / "model.slx").write_bytes(b"PK")
    (tmp_path / "data.txt").write_text("x")
    full = str(tmp_path / "model.slx").replace("'", "''")
    assert cm.num(f"exist('{full}', 'file')").item() == 4 and cm.num("exist('model.slx')").item() == 4
    assert cm.num("exist('data.txt', 'file')").item() == 2


def test_arguments_class_conversions(cm):
    cm.run("r = convargs(2, [1.6; 300], \"abc\", {'x', 'y'});")
    assert cm.str("class(r{1})") == "logical" and cm.num("r{1}").item()
    assert cm.str("class(r{2})") == "int8" and cm.num("r{2}").tolist() == [[2, 127]]
    assert cm.str("r{3}") == "abc" and cm.str("class(r{4})") == "string" and cm.num("numel(r{4})").item() == 2
    cm.run("r = convargs();")
    assert cm.num("size(r{2})").tolist() == [[1, 0]] and cm.str("class(r{2})") == "int8"
    assert cm.num("size(r{4})").tolist() == [[1, 0]] and cm.str("class(r{4})") == "string"
    assert ident(cm, "convargs('a')")[0] == "MATLAB:validation:UnableToConvert"
    assert ident(cm, "convargs(NaN)")[0] == "MATLAB:nologicalnan"
