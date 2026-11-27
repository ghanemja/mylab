"""isequal and isequaln for tables, timetables and calendarDuration arrays."""

import pytest

from mylab import Interpreter


def truth(expr: str, setup: str = "") -> bool:
    it = Interpreter()
    it.execute(setup + f"\n__r = {expr};")
    return bool(it.get_var("__r"))


@pytest.mark.parametrize("expr, want", [
    ("isequal(table([1;2],{'a';'b'}), table([1;2],{'a';'b'}))", True),  # two tables built apart
    ("isequal(table([1;2]), table([1;3]))", False),
    ("isequal(table([1;NaN]), table([1;NaN]))", False),
    ("isequaln(table([1;NaN]), table([1;NaN]))", True),
    ("isequal(table([1;2],'VariableNames',{'x'}), table([1;2],'VariableNames',{'y'}))", False),
    ("isequal(table([1;2],'RowNames',{'r1','r2'}), table([1;2],'RowNames',{'r1','r2'}))", True),
    ("isequal(table([1;2],'RowNames',{'r1','r2'}), table([1;2]))", False),
    ("isequal(table([1;2]), table([1;2]), table([1;2]))", True),
    ("isequal(table([1;2]), [1;2])", False),
    ("isequal(timetable(seconds([1;2]), [3;4]), timetable(seconds([1;2]), [3;4]))", True),
    ("isequal(timetable(seconds([1;2]), [3;4]), timetable(seconds([1;5]), [3;4]))", False),
    ("isequal(timetable(seconds([1;2]), [3;4]), table([3;4]))", False),
])
def test_table_equality(expr, want):
    assert truth(expr) is want


def test_table_properties_are_not_compared():
    setup = "A = table([1;2]); B = A; B.Properties.Description = 'notes'; B.Properties.VariableUnits = {'m'};"
    assert truth("isequal(A, B)", setup)


def test_nested_tables_and_cells_compare_by_value():
    setup = "A = table({1; 'x'}, table([1;2])); B = table({1; 'x'}, table([1;2]));"
    assert truth("isequal(A, B)", setup)


def test_calendarduration_isequaln():
    assert truth("isequaln(calmonths(2), calmonths(2))")
    assert not truth("isequaln(calmonths(2), calmonths(3))")
    assert truth("isequal(caldays([1 2]), caldays([1 2]))")


def test_unsupported_calendarduration_function_names_itself():
    from mylab.builtins._util import text
    it = Interpreter()
    it.execute("try, sort(calmonths([2 1])), catch e, msg = e.message; end")
    assert text(it.get_var("msg")) == "Function 'sort' is not defined for calendarDuration arrays."


# Missing values: never equal for isequal, equal to each other for isequaln, also inside containers.
@pytest.mark.parametrize("expr, want", [
    ("isequal(string(missing), string(missing))", False),
    ("isequaln(string(missing), string(missing))", True),
    ('isequal(["a" missing], ["a" missing])', False),
    ('isequaln(["a" missing], ["a" missing])', True),
    ("isequal(NaT, NaT)", False),
    ("isequaln(NaT, NaT)", True),
    ("isequaln({NaT}, {NaT})", True),
    ("isequal({NaT}, {NaT})", False),
    ("isequaln(seconds([1 NaN]), seconds([1 NaN]))", True),
    ("isequaln(categorical({'a', ''}), categorical({'a', ''}))", True),
    ("isequal(categorical({'a', ''}), categorical({'a', ''}))", False),
    ("isequaln(struct('t', NaT), struct('t', NaT))", True),
    ("isequaln(sparse([1 NaN 0]), sparse([1 NaN 0]))", True),
    ("isequal(sparse([1 NaN 0]), sparse([1 NaN 0]))", False),
    ("isequaln(sparse([1 NaN 0]), sparse([1 2 0]))", False),
    ("isequaln(table(NaT), table(NaT))", True),
])
def test_missing_values(expr, want):
    assert truth(expr) is want
