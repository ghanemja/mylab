"""properties, fieldnames and isprop for the classes implemented in Python.

The expected names are the documented public properties of each class; a class without public
properties gives an empty cell.
"""

import pytest

DATETIME = ["Format", "TimeZone", "Year", "Month", "Day", "Hour", "Minute", "Second", "SystemTimeZone"]

CASES = [
    ("datetime(2024, 1, 2)", DATETIME),
    ("datetime(2024, 1, 2) + days(0:2)", DATETIME),
    ("hours(2)", ["Format"]),
    ("calmonths(2)", ["Format"]),
    ("table([1; 2], [3; 4])", ["Var1", "Var2", "Properties", "Row", "Variables"]),
    ("table([1; 2], 'VariableNames', {'speed'}, 'RowNames', {'a', 'b'})", ["speed", "Properties", "Row", "Variables"]),
    ("timetable(hours([1; 2]), [3; 4], 'VariableNames', {'level'})", ["level", "Properties", "Time", "Variables"]),
    ("categorical({'a', 'b'})", []),
    ('["a" "b"]', []),
    ("containers.Map({'a'}, {1})", ["Count", "KeyType", "ValueType"]),
    ("griddedInterpolant(1:3, [4 5 6])", ["GridVectors", "Values", "Method", "ExtrapolationMethod"]),
    ("scatteredInterpolant([0; 1; 0; 1], [0; 0; 1; 1], [1; 2; 3; 4])",
     ["Points", "Values", "Method", "ExtrapolationMethod"]),
    ("triangulation([1 2 3], [0 0; 1 0; 0 1])", ["Points", "ConnectivityList"]),
    ("delaunayTriangulation([0; 1; 0; 1], [0; 0; 1; 1])", ["Points", "ConnectivityList", "Constraints"]),
    ("polyshape([0 1 1 0], [0 0 1 1])", ["Vertices", "NumRegions", "NumHoles"]),
    ("RandStream('mt19937ar')", ["Type", "Seed", "NumStreams", "StreamIndex", "State", "Substream",
                                 "NormalTransform", "Antithetic", "FullPrecision"]),
    ("inputParser", ["FunctionName", "CaseSensitive", "KeepUnmatched", "PartialMatching", "StructExpand",
                     "Parameters", "Results", "Unmatched", "UsingDefaults"]),
    ("MException('a:b', 'text')", ["identifier", "message", "cause", "stack", "Correction"]),
    ("graph([1 2], [2 3])", ["Edges", "Nodes"]),
    ("digraph([1 2], [2 3])", ["Edges", "Nodes"]),
    ("strel('square', 3)", ["Neighborhood", "Dimensionality"]),
    ("imref2d([4 5])", ["XWorldLimits", "YWorldLimits", "ImageSize", "PixelExtentInWorldX", "PixelExtentInWorldY",
                        "ImageExtentInWorldX", "ImageExtentInWorldY", "XIntrinsicLimits", "YIntrinsicLimits"]),
    ("affine2d(eye(3))", ["T", "Dimensionality"]),
    ("memoize(@sin)", ["Function", "Enabled", "CacheSize"]),
    ("optimvar('x')", ["Name", "Type", "IndexNames", "LowerBound", "UpperBound"]),
    ("optimproblem", ["Description", "ObjectiveSense", "Variables", "Objective", "Constraints"]),
    ("missing", []),
]


def names(m, expr):
    return [str(x) for x in m.run(f"disp(strjoin({expr}', '|'))").strip().split("|") if x]


@pytest.mark.parametrize("expr, expected", CASES)
def test_properties_lists_documented_names(m, expr, expected):
    m.run(f"x = {expr}; p = properties(x);")
    assert m.str("class(p)") == "cell"
    if not expected:
        assert m.num("isempty(p)") and m.num("size(p, 2)") <= 1
        return
    assert m.num("iscolumn(p)")
    assert names(m, "p") == expected


@pytest.mark.parametrize("expr, expected", [c for c in CASES if c[0] != '["a" "b"]'])
def test_fieldnames_and_isprop_agree_with_properties(m, expr, expected):
    m.run(f"x = {expr}; p = properties(x); f = fieldnames(x);")
    assert m.num("isequal(p, f)") or (not expected and m.num("isempty(f)"))
    for name in expected:
        assert m.num(f"isprop(x, '{name}')"), name
        m.run(f"v = x.{name};")  # every listed property can be read
    assert not m.num("isprop(x, 'NoSuchPropertyHere')")


def test_datetime_properties_display(m):
    out = m.run("d = datetime(2024, 1, 2); properties(d)")
    lines = [ln.strip() for ln in out.strip().splitlines() if ln.strip()]
    assert lines[0] == "Properties for class datetime:"
    assert lines[1:] == DATETIME


def test_no_properties_display(m):
    out = m.run("properties(categorical({'a'}))")
    assert "No properties for class categorical." in out
    m.run("p = properties(categorical({'a'}));")
    assert m.num("isequal(size(p), [0 1])")


def test_table_lists_renamed_dimensions(m):
    m.run("T = table([1; 2], 'VariableNames', {'w'}); T.Properties.DimensionNames = {'Obs', 'Vars'};")
    assert names(m, "properties(T)") == ["w", "Properties", "Obs", "Vars"]
    assert m.num("isprop(T, 'Obs') && ~isprop(T, 'Row')")


def test_class_names_as_text(m):
    assert names(m, "properties('datetime')") == DATETIME
    assert names(m, "properties('containers.Map')") == ["Count", "KeyType", "ValueType"]
    assert m.num("isempty(properties('double')) && isempty(properties('categorical'))")
    assert "No properties for class double." in m.run("properties('double')")


def test_calendar_duration_format(m):
    assert m.str("getfield(calmonths(3), 'Format')") == "ymdt"
    m.run("c = calmonths(3); c.Format = 'mdt';")
    assert m.str("c.Format") == "mdt"
    with pytest.raises(Exception):
        m.run("c.Format = 'xyz';")


def test_fieldnames_of_non_objects_still_fails(m):
    with pytest.raises(Exception):
        m.run("fieldnames(5)")
